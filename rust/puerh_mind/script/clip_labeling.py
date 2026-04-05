#!/usr/bin/env python3

import math
import json
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from semantic_client import embed_image, embed_text


CLIP_DEFAULT_CLASSES = [
    "portrait",
    "group portrait",
    "interior",
    "street",
    "architecture exterior",
    "landscape",
    "seascape",
    "mountain",
    "night scene",
    "food",
    "item",
    "flower",
    "cat",
    "dog",
    "wildlife",
    "car",
    "motorcycle",
    "airplane",
    "train",
    "sports action",
]

CLIP_PROMPT_TEMPLATES = [
    "a photo of a {}.",
]

CLIP_CLASS_SYNONYMS = {
    "portrait": ["single person portrait", "solo person"],
    "group portrait": ["group portrait", "group photo with multiple people"],
    "interior": ["indoor room interior", "home interior design"],
    "street": ["street scene", "urban candid street scene"],
    "architecture exterior": ["building exterior architecture", "architectural facade exterior"],
    "landscape": ["natural landscape", "scenic nature view"],
    "seascape": ["seascape", "ocean coastline view"],
    "mountain": ["mountain peak scenery", "mountain landscape"],
    "night scene": ["night scene", "night cityscape"],
    "food": ["food dish", "cooked meal"],
    "item": ["item", "commercial item shot"],
    "flower": ["flower bloom", "flower close-up"],
    "cat": ["domestic cat", "cat portrait"],
    "dog": ["domestic dog", "dog portrait"],
    "wildlife": ["wild animal in nature", "wildlife"],
    "car": ["automobile", "sedan", "sports car"],
    "motorcycle": ["motorcycle", "motorbike"],
    "airplane": ["airplane", "aeroplane"],
    "train": ["train", "railway train"],
    "sports action": ["sports action", "athlete in motion"],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MAX_IN_FLIGHT = 128
TEXT_PROTOTYPES_CACHE_FILENAME = "clip_text_prototypes.json"
TEXT_PROTOTYPES_CACHE_VERSION = 1


def dedupe_labels(labels):
    cleaned = [label.strip() for label in labels if label and label.strip()]
    return list(OrderedDict.fromkeys(cleaned))


def list_images(image_dir):
    images = [
        path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise RuntimeError(f"no images found in {image_dir}")
    return images


def cosine_similarity(vec_a, vec_b):
    if len(vec_a) != len(vec_b):
        raise ValueError(f"embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return -1.0
    return dot / (norm_a * norm_b)


def average_embeddings(vectors):
    if not vectors:
        raise ValueError("no embeddings to average")

    dim = len(vectors[0])
    for idx, vec in enumerate(vectors, start=1):
        if len(vec) != dim:
            raise ValueError(
                f"embedding dimension mismatch in average: expected {dim}, got {len(vec)} at item {idx}"
            )

    summed = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec):
            summed[i] += value

    count = float(len(vectors))
    return [value / count for value in summed]


def l2_normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        raise ValueError("embedding norm is zero")
    return [value / norm for value in vector]


def average_normalized_embeddings(vectors):
    normalized_vectors = [l2_normalize(vector) for vector in vectors]
    return l2_normalize(average_embeddings(normalized_vectors))


def response_field(payload, snake_name):
    camel_name = "".join(
        part if index == 0 else part.capitalize()
        for index, part in enumerate(snake_name.split("_"))
    )
    if snake_name in payload:
        return payload[snake_name]
    return payload.get(camel_name)


def score_image_embedding(img_embedding, label_prototypes):
    img_embedding = l2_normalize(img_embedding)

    scores = []
    for label, txt_embedding in label_prototypes.items():
        score = cosine_similarity(img_embedding, txt_embedding)
        scores.append((label, score))

    scores.sort(key=lambda item: item[1], reverse=True)
    return scores


def embed_text_prompt_ensemble(address, label, prompt_templates=None):
    templates = prompt_templates or CLIP_PROMPT_TEMPLATES

    prompt_embeddings = []
    for template in templates:
        prompt = template.format(label)
        txt_result = embed_text(address, prompt)
        txt_embedding = txt_result.get("embedding", [])
        if not txt_embedding:
            raise RuntimeError(f"empty text embedding for prompt: {prompt}")
        prompt_embeddings.append(txt_embedding)

    return average_normalized_embeddings(prompt_embeddings)


def embed_text_label_prototype(address, label, class_synonyms=None, prompt_templates=None):
    synonyms_map = class_synonyms or CLIP_CLASS_SYNONYMS
    synonyms = synonyms_map.get(label.lower(), [label])

    synonym_embeddings = []
    for synonym in synonyms:
        synonym_embeddings.append(
            embed_text_prompt_ensemble(address, synonym, prompt_templates=prompt_templates)
        )

    return average_normalized_embeddings(synonym_embeddings)


def build_label_prototypes(address, labels, class_synonyms=None, prompt_templates=None):
    deduped = dedupe_labels(labels)
    if not deduped:
        raise RuntimeError("no labels provided")

    prototypes = {}
    for label in deduped:
        prototypes[label] = embed_text_label_prototype(
            address,
            label,
            class_synonyms=class_synonyms,
            prompt_templates=prompt_templates,
        )
    return prototypes


def text_prototypes_cache_path(testdata_dir):
    return Path(testdata_dir) / TEXT_PROTOTYPES_CACHE_FILENAME


def _canonical_prototype_config(labels, class_synonyms=None, prompt_templates=None):
    deduped_labels = dedupe_labels(labels)
    templates = list(prompt_templates or CLIP_PROMPT_TEMPLATES)
    synonyms_map = class_synonyms or CLIP_CLASS_SYNONYMS
    per_label_synonyms = {
        label: list(synonyms_map.get(label.lower(), [label])) for label in deduped_labels
    }
    return {
        "labels": deduped_labels,
        "prompt_templates": templates,
        "class_synonyms": per_label_synonyms,
    }


def _validate_or_load_prototypes(payload, expected_labels):
    stored = payload.get("prototypes")
    if not isinstance(stored, dict):
        return None

    prototypes = {}
    expected_dim = payload.get("dimension")
    for label in expected_labels:
        vector = stored.get(label)
        if not isinstance(vector, list) or not vector:
            return None
        try:
            vector = [float(value) for value in vector]
        except (TypeError, ValueError):
            return None

        if expected_dim is not None and len(vector) != expected_dim:
            return None

        prototypes[label] = vector

    return prototypes


def load_text_prototypes_cache(
    cache_path,
    labels,
    class_synonyms=None,
    prompt_templates=None,
):
    cache_path = Path(cache_path)
    expected_config = _canonical_prototype_config(
        labels,
        class_synonyms=class_synonyms,
        prompt_templates=prompt_templates,
    )
    expected_labels = expected_config["labels"]

    if not expected_labels:
        raise RuntimeError("no labels provided")

    if not cache_path.exists():
        return None

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if payload.get("version") != TEXT_PROTOTYPES_CACHE_VERSION:
        return None

    if payload.get("labels") != expected_labels:
        return None

    if payload.get("prompt_templates") != expected_config["prompt_templates"]:
        return None

    if payload.get("class_synonyms") != expected_config["class_synonyms"]:
        return None

    return _validate_or_load_prototypes(payload, expected_labels)


def write_text_prototypes_cache(
    cache_path,
    labels,
    prototypes,
    class_synonyms=None,
    prompt_templates=None,
):
    cache_path = Path(cache_path)
    config = _canonical_prototype_config(
        labels,
        class_synonyms=class_synonyms,
        prompt_templates=prompt_templates,
    )
    deduped_labels = config["labels"]
    if not deduped_labels:
        raise RuntimeError("no labels provided")

    first_label = deduped_labels[0]
    first_vector = prototypes.get(first_label)
    if not isinstance(first_vector, list) or not first_vector:
        raise RuntimeError(f"missing prototype embedding for label: {first_label}")
    dimension = len(first_vector)

    serializable = {}
    for label in deduped_labels:
        vector = prototypes.get(label)
        if not isinstance(vector, list) or len(vector) != dimension:
            raise RuntimeError(f"invalid prototype embedding for label: {label}")
        serializable[label] = [float(value) for value in vector]

    payload = {
        "version": TEXT_PROTOTYPES_CACHE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "labels": deduped_labels,
        "prompt_templates": config["prompt_templates"],
        "class_synonyms": config["class_synonyms"],
        "dimension": dimension,
        "prototypes": serializable,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_or_build_label_prototypes(
    address,
    labels,
    testdata_dir,
    class_synonyms=None,
    prompt_templates=None,
):
    cache_path = text_prototypes_cache_path(testdata_dir)
    cached = load_text_prototypes_cache(
        cache_path,
        labels,
        class_synonyms=class_synonyms,
        prompt_templates=prompt_templates,
    )
    if cached is not None:
        return cached, cache_path, True

    prototypes = build_label_prototypes(
        address,
        labels,
        class_synonyms=class_synonyms,
        prompt_templates=prompt_templates,
    )
    write_text_prototypes_cache(
        cache_path,
        labels,
        prototypes,
        class_synonyms=class_synonyms,
        prompt_templates=prompt_templates,
    )
    return prototypes, cache_path, False


def classify_image_by_prototypes(address, image_path, label_prototypes):
    if not label_prototypes:
        raise RuntimeError("no label prototypes")

    img_result = embed_image(address, image_path)
    img_embedding = response_field(img_result, "embedding") or []
    if not img_embedding:
        raise RuntimeError("empty image embedding")
    return score_image_embedding(img_embedding, label_prototypes)


def classify_images_by_prototypes(
    address,
    image_paths,
    label_prototypes,
    max_in_flight=DEFAULT_MAX_IN_FLIGHT,
    result_callback=None,
):
    if not label_prototypes:
        raise RuntimeError("no label prototypes")

    image_paths = list(image_paths)
    if not image_paths:
        return []

    if max_in_flight < 1:
        raise RuntimeError("max_in_flight must be >= 1")

    max_workers = max(1, min(max_in_flight, len(image_paths)))
    ordered_results = [None] * len(image_paths)
    request_map = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_request = {}
        for index, image_path in enumerate(image_paths):
            request_id = f"img-{uuid.uuid4().hex[:8]}"
            request_map[request_id] = (index, image_path)
            future = executor.submit(embed_image, address, image_path, request_id)
            future_to_request[future] = request_id

        completed = 0
        total = len(image_paths)
        for future in as_completed(future_to_request):
            request_id = future_to_request[future]
            index, image_path = request_map[request_id]
            completed += 1

            try:
                img_result = future.result()
                response_id = response_field(img_result, "request_id")
                if response_id != request_id:
                    raise RuntimeError(
                        f"request_id mismatch for {image_path.name}: expected {request_id}, got {response_id}"
                    )

                img_embedding = response_field(img_result, "embedding") or []
                if not img_embedding:
                    raise RuntimeError(f"empty image embedding for {image_path.name}")

                result = {
                    "request_id": request_id,
                    "image_path": image_path,
                    "scores": score_image_embedding(img_embedding, label_prototypes),
                    "error": None,
                }
            except Exception as exc:
                result = {
                    "request_id": request_id,
                    "image_path": image_path,
                    "scores": [],
                    "error": str(exc),
                }

            ordered_results[index] = result
            if result_callback is not None:
                result_callback(completed, total, image_path, result)

    return ordered_results


def classify_image_by_labels(address, image_path, labels, class_synonyms=None, prompt_templates=None):
    prototypes = build_label_prototypes(
        address,
        labels,
        class_synonyms=class_synonyms,
        prompt_templates=prompt_templates,
    )
    return classify_image_by_prototypes(address, image_path, prototypes)


def decide_prediction(scores, min_score=0.20, min_margin=0.02):
    if not scores:
        raise RuntimeError("no score candidates")

    best_label, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else -1.0
    margin = best_score - second_score

    confident = best_score >= min_score and margin >= min_margin
    return {
        "label": best_label,
        "score": best_score,
        "second_score": second_score,
        "margin": margin,
        "confident": confident,
    }

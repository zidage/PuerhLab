//! Port check of the segmentation engine (SM3.A.2 of the AI Segmentation Mask plan).
//!
//! The engine runs on the pixels and prompts of a reference directory, and its masks are compared
//! with the masks of `scripts/ai_models/sam2/make_segmentation_port_reference.py` (PyTorch
//! preprocessing, ONNX Runtime 1.31 CPU, the lab MVP selection rules). For every prompt, the
//! number of masks, the token of each mask, and the selected index must be equal, and the IoU of
//! each mask (logits > 0 after bilinear upsampling to the photo size) must be at least 0.999. The
//! test also prints the model load, encoder, and decoder times.
//!
//! Image embedding cache (SM3.B): every photo starts with its name as the image key. After all
//! photos, the photos that the cache keeps (the last 4) start again with the key only and decode
//! their first prompt; the masks must match the reference with the same rules. The first photo
//! has left the cache, so its key-only start must fail with `ImageNotCached`.
//!
//! Environment:
//! - `ALCEDO_SEGMENTATION_MODEL_DIR`: the model directory of the platform (SM3.A plan, 2.1);
//! - `ALCEDO_SEGMENTATION_PORT_REFERENCE_DIR`: the output directory of the reference script;
//! - `ALCEDO_SEGMENTATION_PORT_CHECK_DEVICE`: optional device value (`""` is the platform default).
//!
//! The test returns without a check when one of the first two variables is not set.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use alcedo_mind::service::segmentation::engine::{
    IMAGE_EMBEDDING_CACHE_CAPACITY, RgbImageView, SegmentationEngine, SegmentationError,
    StartSessionRequest,
};
use alcedo_mind::service::segmentation::model_set::LOGIT_SIZE;
use alcedo_mind::service::segmentation::platform_backend_factory;
use alcedo_mind::service::segmentation::sam2::{
    PointLabel, PromptBox, PromptPoint, SegmentationPrompt,
};
use serde_json::Value;

const MIN_PHOTO_IOU: f64 = 0.999;
/// Number of decodes of one prompt on one session for the steady-state decode time.
const REPEATED_DECODES: usize = 20;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Taps of a bilinear upsampling with half-pixel centers (PyTorch `align_corners=False`, OpenCV
/// `INTER_LINEAR`): for each output index, the two inputs and the weight of the second.
fn upsample_taps(input: usize, output: usize) -> Vec<(usize, usize, f32)> {
    let scale = input as f32 / output as f32;
    (0..output)
        .map(|index| {
            let source = ((index as f32 + 0.5) * scale - 0.5).max(0.0);
            let low = (source.floor() as usize).min(input - 1);
            let high = (low + 1).min(input - 1);
            (low, high, source - low as f32)
        })
        .collect()
}

/// Inside flags (logit > 0) of 256 x 256 logits upsampled to width x height.
fn mask_at_photo_size(logits: &[f32], width: usize, height: usize) -> Vec<bool> {
    let columns = upsample_taps(LOGIT_SIZE, width);
    let rows = upsample_taps(LOGIT_SIZE, height);
    let mut mask = Vec::with_capacity(width * height);
    let mut row_values = vec![0.0f32; LOGIT_SIZE];
    for &(top, bottom, weight_y) in &rows {
        for (x, value) in row_values.iter_mut().enumerate() {
            let a = logits[top * LOGIT_SIZE + x];
            let b = logits[bottom * LOGIT_SIZE + x];
            *value = a + (b - a) * weight_y;
        }
        for &(left, right, weight_x) in &columns {
            let a = row_values[left];
            let b = row_values[right];
            mask.push(a + (b - a) * weight_x > 0.0);
        }
    }
    mask
}

fn iou(a: &[bool], b: &[bool]) -> f64 {
    let (mut intersection, mut union) = (0u64, 0u64);
    for (x, y) in a.iter().zip(b) {
        intersection += u64::from(*x && *y);
        union += u64::from(*x || *y);
    }
    if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    }
}

fn prompt_from(record: &Value, reference_dir: &Path) -> SegmentationPrompt {
    let points = record["points"]
        .as_array()
        .expect("points")
        .iter()
        .map(|point| PromptPoint {
            x: point[0].as_f64().expect("x") as f32,
            y: point[1].as_f64().expect("y") as f32,
            label: if point[2].as_i64() == Some(0) {
                PointLabel::Subtract
            } else {
                PointLabel::Add
            },
        })
        .collect();
    let prompt_box = record["box"].as_array().map(|corners| PromptBox {
        x0: corners[0].as_f64().expect("x0") as f32,
        y0: corners[1].as_f64().expect("y0") as f32,
        x1: corners[2].as_f64().expect("x1") as f32,
        y1: corners[3].as_f64().expect("y1") as f32,
    });
    let mask_input = record["mask_input"]
        .as_str()
        .map(|file| read_f32(&reference_dir.join(file)))
        .unwrap_or_default();
    SegmentationPrompt {
        points,
        prompt_box,
        mask_input,
    }
}

fn median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}

/// Results of the mask comparisons of the whole check.
struct Comparison {
    failures: Vec<String>,
    minimum_iou: f64,
    compared: usize,
}

/// Decodes @p record on @p session_id and compares the masks with the reference. Returns the
/// time of the engine decode call in microseconds and the decoder point count.
fn decode_and_compare(
    engine: &SegmentationEngine,
    session_id: &str,
    record: &Value,
    reference_dir: &Path,
    (width, height): (usize, usize),
    label: &str,
    comparison: &mut Comparison,
) -> (u64, usize) {
    let prompt = prompt_from(record, reference_dir);
    let point_count = prompt.points.len() + if prompt.prompt_box.is_some() { 2 } else { 0 };
    let started = Instant::now();
    let result = engine
        .decode(session_id, prompt)
        .unwrap_or_else(|err| panic!("{label}: {err}"));
    let decode_us = started.elapsed().as_micros() as u64;
    let reference_masks = record["masks"].as_array().expect("masks");
    if reference_masks.len() != result.masks.masks.len() {
        comparison.failures.push(format!(
            "{label}: {} masks, reference {}",
            result.masks.masks.len(),
            reference_masks.len()
        ));
        return (decode_us, point_count);
    }
    if record["selected_index"].as_u64() != Some(result.masks.selected_index as u64) {
        comparison.failures.push(format!(
            "{label}: selected index {}, reference {}",
            result.masks.selected_index, record["selected_index"]
        ));
    }
    for (level, reference) in result.masks.masks.iter().zip(reference_masks) {
        if reference["token"].as_u64() != Some(u64::from(level.token)) {
            comparison.failures.push(format!(
                "{label}: token {}, reference {}",
                level.token, reference["token"]
            ));
            continue;
        }
        let reference_logits =
            read_f32(&reference_dir.join(reference["logits"].as_str().expect("logits")));
        let max_difference = level
            .logits
            .iter()
            .zip(&reference_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let photo_iou = iou(
            &mask_at_photo_size(&level.logits, width, height),
            &mask_at_photo_size(&reference_logits, width, height),
        );
        comparison.compared += 1;
        comparison.minimum_iou = comparison.minimum_iou.min(photo_iou);
        println!(
            "{label} token {}: photo IoU {photo_iou:.6}, max |logit difference| {max_difference:.5}",
            level.token
        );
        if photo_iou < MIN_PHOTO_IOU {
            comparison.failures.push(format!(
                "{label} token {}: photo IoU {photo_iou:.6}",
                level.token
            ));
        }
    }
    (decode_us, point_count)
}

fn photo_size(photo: &Value) -> (usize, usize) {
    (
        photo["width"].as_u64().expect("width") as usize,
        photo["height"].as_u64().expect("height") as usize,
    )
}

#[test]
fn segmentation_port_check_matches_reference_masks() {
    let (Some(model_dir), Some(reference_dir)) = (
        std::env::var_os("ALCEDO_SEGMENTATION_MODEL_DIR").map(PathBuf::from),
        std::env::var_os("ALCEDO_SEGMENTATION_PORT_REFERENCE_DIR").map(PathBuf::from),
    ) else {
        eprintln!(
            "segmentation port check not run: ALCEDO_SEGMENTATION_MODEL_DIR and ALCEDO_SEGMENTATION_PORT_REFERENCE_DIR are not set"
        );
        return;
    };
    let device = std::env::var("ALCEDO_SEGMENTATION_PORT_CHECK_DEVICE").unwrap_or_default();
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(reference_dir.join("manifest.json")).expect("manifest.json"),
    )
    .expect("manifest JSON");

    let engine = Arc::new(SegmentationEngine::new(platform_backend_factory()));
    let load_started = Instant::now();
    engine.load(&model_dir, &device).expect("load models");
    let load_ms = load_started.elapsed().as_millis();
    let status = engine.status().expect("status");
    println!(
        "backend {} device {} load ms {load_ms} {:?}",
        status.backend,
        status.device,
        status
            .models
            .iter()
            .map(|model| (model.role.as_str(), model.load_ms))
            .collect::<Vec<_>>()
    );

    let photos = manifest["photos"].as_array().expect("photos");
    let mut comparison = Comparison {
        failures: Vec::new(),
        minimum_iou: 1.0,
        compared: 0,
    };
    let mut encoder_ms = Vec::new();
    let mut decode_us: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    let mut first_decode_us = None;
    for photo in photos {
        let name = photo["name"].as_str().expect("name");
        let size = photo_size(photo);
        let pixels = std::fs::read(reference_dir.join(photo["pixels"].as_str().expect("pixels")))
            .expect("pixels");
        let session = engine
            .start_session(StartSessionRequest {
                image: Some(RgbImageView {
                    width: size.0 as u32,
                    height: size.1 as u32,
                    rgb8: &pixels,
                }),
                image_key: name,
            })
            .expect("start session");
        assert!(
            !session.from_cache,
            "{name}: the first start must run the encoder"
        );
        encoder_ms.push(session.encoder_ms);
        for record in photo["prompts"].as_array().expect("prompts") {
            let label = format!("{name} {}", record["name"].as_str().expect("prompt name"));
            let (microseconds, point_count) = decode_and_compare(
                &engine,
                &session.session_id,
                record,
                &reference_dir,
                size,
                &label,
                &mut comparison,
            );
            first_decode_us.get_or_insert(microseconds);
            decode_us.entry(point_count).or_default().push(microseconds);
        }
        engine
            .end_session(&session.session_id)
            .expect("end session");
    }

    // Image embedding cache: the last photos are cached, the first photo has left the cache.
    assert!(photos.len() > IMAGE_EMBEDDING_CACHE_CAPACITY);
    let first_name = photos[0]["name"].as_str().expect("name");
    assert!(matches!(
        engine.start_session(StartSessionRequest {
            image: None,
            image_key: first_name,
        }),
        Err(SegmentationError::ImageNotCached(_))
    ));
    let mut cached_start_us = Vec::new();
    let mut repeated_decode_us = Vec::new();
    for photo in &photos[photos.len() - IMAGE_EMBEDDING_CACHE_CAPACITY..] {
        let name = photo["name"].as_str().expect("name");
        let started = Instant::now();
        let session = engine
            .start_session(StartSessionRequest {
                image: None,
                image_key: name,
            })
            .expect("cached start");
        cached_start_us.push(started.elapsed().as_micros() as u64);
        assert!(
            session.from_cache,
            "{name}: the key-only start must use the cache"
        );
        let record = &photo["prompts"][0];
        let label = format!(
            "{name} {} (cached)",
            record["name"].as_str().expect("prompt name")
        );
        decode_and_compare(
            &engine,
            &session.session_id,
            record,
            &reference_dir,
            photo_size(photo),
            &label,
            &mut comparison,
        );
        // The same prompt on the same features: the decoder session keeps the features bound.
        if repeated_decode_us.is_empty() {
            for _ in 0..REPEATED_DECODES {
                let started = Instant::now();
                engine
                    .decode(&session.session_id, prompt_from(record, &reference_dir))
                    .expect("repeated decode");
                repeated_decode_us.push(started.elapsed().as_micros() as u64);
            }
        }
        engine
            .end_session(&session.session_id)
            .expect("end session");
    }

    println!(
        "compared masks {}, minimum photo IoU {:.6}, encoder ms p50 {} (first {}), first decode us {}",
        comparison.compared,
        comparison.minimum_iou,
        median(&mut encoder_ms.clone()),
        encoder_ms[0],
        first_decode_us.unwrap_or_default()
    );
    for (points, values) in &mut decode_us {
        println!(
            "decode us p50 with {points} decoder points: {} ({} calls)",
            median(values),
            values.len()
        );
    }
    println!(
        "cached start us p50 {} ({} starts); repeated decode us p50 {} ({} calls)",
        median(&mut cached_start_us),
        IMAGE_EMBEDDING_CACHE_CAPACITY,
        median(&mut repeated_decode_us),
        REPEATED_DECODES
    );
    engine.unload().expect("unload");
    assert!(
        comparison.failures.is_empty(),
        "port check failures:\n{}",
        comparison.failures.join("\n")
    );
}

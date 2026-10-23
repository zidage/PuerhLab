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
//! Task masks (SM3.C): when the reference has task masks, every photo also runs Sky, People (all
//! people and each person alone), Subject, and Background. The Sky region count, the People list
//! length, and the list boxes (within 1e-3 cell) must match; the Sky and People unions and each
//! person mask must have an IoU of at least 0.999 at the photo size against the union of the
//! reference masks; the Subject and Background alpha >= 0.5 must have an IoU of at least 0.998 on
//! the S3OD grid. A key-only start of a cached photo must read the stored Subject result.
//!
//! The test returns without a check when one of the first two variables is not set.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use alcedo_mind::service::segmentation::engine::{
    IMAGE_EMBEDDING_CACHE_CAPACITY, RgbImageView, SegmentationEngine, SegmentationError,
    SegmentationTask, StartSessionRequest, TaskMaskValues, TaskRequest, TaskResult,
};
use alcedo_mind::service::segmentation::model_set::{HEAD_GRID, LOGIT_SIZE};
use alcedo_mind::service::segmentation::platform_backend_factory;
use alcedo_mind::service::segmentation::sam2::{
    PointLabel, PromptBox, PromptPoint, SegmentationPrompt,
};
use serde_json::Value;

const MIN_PHOTO_IOU: f64 = 0.999;
/// Smallest IoU of the Subject and Background alpha >= 0.5 on the S3OD grid (the lab macOS
/// acceptance criterion for S3OD).
const MIN_SUBJECT_IOU: f64 = 0.998;
/// Largest difference of a People list box corner, in head grid cells.
const MAX_BOX_DIFFERENCE_CELLS: f32 = 1e-3;
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

/// Times of the task requests of the whole check, in milliseconds.
#[derive(Default)]
struct TaskTimes {
    /// The first Sky request of each photo: the head and the region decodes.
    first_sky_ms: Vec<u64>,
    /// The first People request of each photo: the person decodes (the head output is stored).
    first_people_ms: Vec<u64>,
    /// The first Subject request of each photo: S3OD with its pre- and postprocessing.
    first_subject_ms: Vec<u64>,
    /// Requests that read the stored results only, in microseconds.
    stored_request_us: Vec<u64>,
}

fn run_task(
    engine: &SegmentationEngine,
    session_id: &str,
    task: SegmentationTask,
    person_indices: Vec<u32>,
    label: &str,
) -> TaskResult {
    engine
        .run_task(
            session_id,
            TaskRequest {
                task,
                person_indices,
            },
        )
        .unwrap_or_else(|err| panic!("{label} {task:?}: {err}"))
}

/// Union at the photo size of the reference logits files @p files (the MVP union).
fn reference_union(
    files: &[&str],
    reference_dir: &Path,
    (width, height): (usize, usize),
) -> Vec<bool> {
    let mut union = vec![false; width * height];
    for file in files {
        let mask = mask_at_photo_size(&read_f32(&reference_dir.join(file)), width, height);
        for (target, inside) in union.iter_mut().zip(mask) {
            *target |= inside;
        }
    }
    union
}

/// Compares an R8 union task mask with the union of the reference logits @p files at the photo
/// size.
fn compare_binary_union(
    result: &TaskResult,
    files: &[&str],
    reference_dir: &Path,
    size: (usize, usize),
    label: &str,
    comparison: &mut Comparison,
) {
    let Some(mask) = &result.mask else {
        comparison
            .failures
            .push(format!("{label}: no mask, reference found"));
        return;
    };
    let TaskMaskValues::Binary(values) = &mask.values else {
        panic!("{label}: the mask must be an R8 mask");
    };
    assert_eq!(
        (mask.width as usize, mask.height as usize),
        size,
        "{label}: the union mask has the photo size"
    );
    let inside: Vec<bool> = values.iter().map(|value| *value == 255).collect();
    let photo_iou = iou(&inside, &reference_union(files, reference_dir, size));
    comparison.compared += 1;
    comparison.minimum_iou = comparison.minimum_iou.min(photo_iou);
    println!("{label}: photo IoU {photo_iou:.6}");
    if photo_iou < MIN_PHOTO_IOU {
        comparison
            .failures
            .push(format!("{label}: photo IoU {photo_iou:.6}"));
    }
}

/// Runs the four task masks on @p session_id and compares them with the task reference of
/// @p photo (SM3.C plan, section 8).
fn compare_tasks(
    engine: &SegmentationEngine,
    session_id: &str,
    photo: &Value,
    reference_dir: &Path,
    size: (usize, usize),
    comparison: &mut Comparison,
    times: &mut TaskTimes,
) {
    let name = photo["name"].as_str().expect("name");
    let tasks = &photo["tasks"];

    // Sky.
    let sky = run_task(engine, session_id, SegmentationTask::Sky, vec![], name);
    times.first_sky_ms.push(sky.model_ms);
    let reference_regions: Vec<&str> = tasks["sky"]["regions"]
        .as_array()
        .expect("sky regions")
        .iter()
        .map(|region| region["logits"].as_str().expect("logits"))
        .collect();
    println!(
        "{name} sky: presence {:.4} (reference {:.4}), regions {} (reference {}), {} ms, {}",
        sky.presence,
        tasks["sky_presence"].as_f64().expect("sky presence"),
        sky.region_count,
        reference_regions.len(),
        sky.model_ms,
        sky.message
    );
    if sky.region_count as usize != reference_regions.len() {
        comparison.failures.push(format!(
            "{name} sky: {} regions, reference {}",
            sky.region_count,
            reference_regions.len()
        ));
    } else if reference_regions.is_empty() {
        if sky.mask.is_some() {
            comparison
                .failures
                .push(format!("{name} sky: a mask, reference not found"));
        }
    } else {
        let label = format!("{name} sky union");
        compare_binary_union(
            &sky,
            &reference_regions,
            reference_dir,
            size,
            &label,
            comparison,
        );
    }

    // People.
    let people = run_task(engine, session_id, SegmentationTask::People, vec![], name);
    times.first_people_ms.push(people.model_ms);
    let reference_people = tasks["people"]["people"].as_array().expect("people");
    println!(
        "{name} people: presence {:.4} (reference {:.4}), {} listed (reference {}), {} ms, {}",
        people.presence,
        tasks["person_presence"].as_f64().expect("person presence"),
        people.people.len(),
        reference_people.len(),
        people.model_ms,
        people.message
    );
    if people.people.len() != reference_people.len() {
        comparison.failures.push(format!(
            "{name} people: {} listed, reference {}",
            people.people.len(),
            reference_people.len()
        ));
    } else if reference_people.is_empty() {
        if people.mask.is_some() {
            comparison
                .failures
                .push(format!("{name} people: a mask, reference not found"));
        }
    } else {
        for (index, (entry, reference)) in people.people.iter().zip(reference_people).enumerate() {
            let cells: Vec<f32> = reference["box_cells"]
                .as_array()
                .expect("box cells")
                .iter()
                .map(|value| (value.as_f64().expect("cell") as f32).clamp(0.0, HEAD_GRID as f32))
                .collect();
            let grid = HEAD_GRID as f32;
            let difference = [entry.x0, entry.y0, entry.x1, entry.y1]
                .iter()
                .zip(&cells)
                .map(|(value, cell)| (value * grid - cell).abs())
                .fold(0.0f32, f32::max);
            if difference > MAX_BOX_DIFFERENCE_CELLS {
                comparison.failures.push(format!(
                    "{name} person {index}: box differs by {difference} cells"
                ));
            }
            let started = Instant::now();
            let one = run_task(
                engine,
                session_id,
                SegmentationTask::People,
                vec![index as u32],
                name,
            );
            times
                .stored_request_us
                .push(started.elapsed().as_micros() as u64);
            let label = format!("{name} person {index}");
            compare_binary_union(
                &one,
                &[reference["logits"].as_str().expect("logits")],
                reference_dir,
                size,
                &label,
                comparison,
            );
        }
        let files: Vec<&str> = reference_people
            .iter()
            .map(|person| person["logits"].as_str().expect("logits"))
            .collect();
        let label = format!("{name} people union");
        compare_binary_union(&people, &files, reference_dir, size, &label, comparison);
    }

    // Subject and Background.
    let subject_reference = &tasks["subject"];
    let reference_alpha =
        read_f32(&reference_dir.join(subject_reference["alpha"].as_str().expect("alpha")));
    let reference_size = (
        subject_reference["width"].as_u64().expect("width") as u32,
        subject_reference["height"].as_u64().expect("height") as u32,
    );
    for task in [SegmentationTask::Subject, SegmentationTask::Background] {
        let result = run_task(engine, session_id, task, vec![], name);
        if task == SegmentationTask::Subject {
            times.first_subject_ms.push(result.model_ms);
        }
        let label = format!("{name} {task:?}");
        let reference_inside: Vec<bool> = reference_alpha
            .iter()
            .map(|value| match task {
                SegmentationTask::Background => 1.0 - value >= 0.5,
                _ => *value >= 0.5,
            })
            .collect();
        let Some(mask) = &result.mask else {
            if reference_inside.iter().any(|inside| *inside) {
                comparison
                    .failures
                    .push(format!("{label}: not found, reference found"));
            }
            continue;
        };
        let TaskMaskValues::Alpha(values) = &mask.values else {
            panic!("{label}: the mask must be an alpha mask");
        };
        if (mask.width, mask.height) != reference_size {
            comparison.failures.push(format!(
                "{label}: size {}x{}, reference {}x{}",
                mask.width, mask.height, reference_size.0, reference_size.1
            ));
            continue;
        }
        let inside: Vec<bool> = values.iter().map(|value| *value >= 0.5).collect();
        let grid_iou = iou(&inside, &reference_inside);
        let max_difference = values
            .iter()
            .zip(&reference_alpha)
            .map(|(value, reference)| match task {
                SegmentationTask::Background => (value - (1.0 - reference)).abs(),
                _ => (value - reference).abs(),
            })
            .fold(0.0f32, f32::max);
        println!(
            "{label}: grid IoU {grid_iou:.6}, max |alpha difference| {max_difference:.4}, score {:.4} (reference {:.4}), {} ms",
            result.subject_score,
            subject_reference["score"].as_f64().expect("score"),
            result.model_ms
        );
        if grid_iou < MIN_SUBJECT_IOU {
            comparison
                .failures
                .push(format!("{label}: grid IoU {grid_iou:.6}"));
        }
    }
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
    let mut task_times = TaskTimes::default();
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
        if photo["tasks"].is_object() {
            compare_tasks(
                &engine,
                &session.session_id,
                photo,
                &reference_dir,
                size,
                &mut comparison,
                &mut task_times,
            );
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
        if photo["tasks"].is_object() {
            let started = Instant::now();
            let subject = run_task(
                &engine,
                &session.session_id,
                SegmentationTask::Subject,
                vec![],
                name,
            );
            task_times
                .stored_request_us
                .push(started.elapsed().as_micros() as u64);
            assert_eq!(
                subject.model_ms, 0,
                "{name}: the cached image must keep its Subject result"
            );
        }
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
    if !task_times.first_subject_ms.is_empty() {
        println!(
            "first Sky request ms p50 {} (first photo {}); first People request ms p50 {}; first Subject request ms p50 {} (first photo {}); stored-result request us p50 {} ({} requests)",
            median(&mut task_times.first_sky_ms.clone()),
            task_times.first_sky_ms[0],
            median(&mut task_times.first_people_ms),
            median(&mut task_times.first_subject_ms.clone()),
            task_times.first_subject_ms[0],
            median(&mut task_times.stored_request_us),
            task_times.stored_request_us.len()
        );
    }
    engine.unload().expect("unload");
    assert!(
        comparison.failures.is_empty(),
        "port check failures:\n{}",
        comparison.failures.join("\n")
    );
}

//! Port check of the segmentation engine (SM3.A.2 of the AI Segmentation Mask plan).
//!
//! The engine runs on the pixels and prompts of a reference directory, and its masks are compared
//! with the masks of `scripts/ai_models/sam2/make_segmentation_port_reference.py` (PyTorch
//! preprocessing, ONNX Runtime 1.31 CPU, the lab MVP selection rules). For every prompt, the
//! number of masks, the token of each mask, and the selected index must be equal, and the IoU of
//! each mask (logits > 0 after bilinear upsampling to the photo size) must be at least 0.999. The
//! test also prints the model load, encoder, and decoder times.
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

use alcedo_mind::service::segmentation::engine::{RgbImageView, SegmentationEngine};
use alcedo_mind::service::segmentation::model_set::LOGIT_SIZE;
use alcedo_mind::service::segmentation::platform_backend_factory;
use alcedo_mind::service::segmentation::sam2::{
    PointLabel, PromptBox, PromptPoint, SegmentationPrompt,
};
use serde_json::Value;

const MIN_PHOTO_IOU: f64 = 0.999;

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

    let engine = SegmentationEngine::new(platform_backend_factory());
    let engine = Arc::new(engine);
    engine.load(&model_dir, &device).expect("load models");
    let status = engine.status().expect("status");
    println!(
        "backend {} device {} load ms {:?}",
        status.backend,
        status.device,
        status
            .models
            .iter()
            .map(|model| (model.role.as_str(), model.load_ms))
            .collect::<Vec<_>>()
    );

    let mut failures = Vec::new();
    let mut encoder_ms = Vec::new();
    let mut decoder_ms: BTreeMap<usize, Vec<u64>> = BTreeMap::new();
    let mut minimum_iou = 1.0f64;
    let mut compared = 0usize;
    for photo in manifest["photos"].as_array().expect("photos") {
        let name = photo["name"].as_str().expect("name");
        let width = photo["width"].as_u64().expect("width") as u32;
        let height = photo["height"].as_u64().expect("height") as u32;
        let pixels = std::fs::read(reference_dir.join(photo["pixels"].as_str().expect("pixels")))
            .expect("pixels");
        let session = engine
            .start_session(RgbImageView {
                width,
                height,
                rgb8: &pixels,
            })
            .expect("start session");
        encoder_ms.push(session.encoder_ms);
        for record in photo["prompts"].as_array().expect("prompts") {
            let prompt_name = record["name"].as_str().expect("prompt name");
            let prompt = prompt_from(record, &reference_dir);
            let point_count = prompt.points.len() + if prompt.prompt_box.is_some() { 2 } else { 0 };
            let result = engine
                .decode(&session.session_id, prompt)
                .unwrap_or_else(|err| panic!("{name} {prompt_name}: {err}"));
            decoder_ms
                .entry(point_count)
                .or_default()
                .push(result.decoder_ms);
            let reference_masks = record["masks"].as_array().expect("masks");
            let label = format!("{name} {prompt_name}");
            if reference_masks.len() != result.masks.masks.len() {
                failures.push(format!(
                    "{label}: {} masks, reference {}",
                    result.masks.masks.len(),
                    reference_masks.len()
                ));
                continue;
            }
            if record["selected_index"].as_u64() != Some(result.masks.selected_index as u64) {
                failures.push(format!(
                    "{label}: selected index {}, reference {}",
                    result.masks.selected_index, record["selected_index"]
                ));
            }
            for (level, reference) in result.masks.masks.iter().zip(reference_masks) {
                if reference["token"].as_u64() != Some(u64::from(level.token)) {
                    failures.push(format!(
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
                    &mask_at_photo_size(&level.logits, width as usize, height as usize),
                    &mask_at_photo_size(&reference_logits, width as usize, height as usize),
                );
                compared += 1;
                minimum_iou = minimum_iou.min(photo_iou);
                println!(
                    "{label} token {}: photo IoU {photo_iou:.6}, max |logit difference| {max_difference:.5}",
                    level.token
                );
                if photo_iou < MIN_PHOTO_IOU {
                    failures.push(format!(
                        "{label} token {}: photo IoU {photo_iou:.6}",
                        level.token
                    ));
                }
            }
        }
        engine
            .end_session(&session.session_id)
            .expect("end session");
    }
    println!(
        "compared masks {compared}, minimum photo IoU {minimum_iou:.6}, encoder ms p50 {} (first {})",
        median(&mut encoder_ms.clone()),
        encoder_ms[0]
    );
    for (points, values) in &mut decoder_ms {
        println!(
            "decoder ms p50 with {points} decoder points: {} ({} calls)",
            median(values),
            values.len()
        );
    }
    engine.unload().expect("unload");
    assert!(
        failures.is_empty(),
        "port check failures:\n{}",
        failures.join("\n")
    );
}

//! SM1 measurement program for the EfficientSAM3 ONNX graphs, also used for the SAM 2.1 point and
//! box graphs of the model re-evaluation (#458).
//!
//! `runtime` mode loads the graphs that `scripts/ai_models/efficientsam3/export_onnx.py` or
//! `scripts/ai_models/sam2/export_onnx.py` writes through `ort_runtime::load_session`, the session settings of the production sidecar. It runs
//! the point, box, and text prompts of a prompts JSON file on real photos and writes one JSON
//! report with:
//!
//! - session load time, first inference time, and warm latency (p50, p95, maximum) per graph;
//! - process memory after load and after inference;
//! - the execution provider of each graph node, from an ONNX Runtime profile of a second session
//!   that adds only the profiling option to the production settings;
//! - the masks that this program makes (Rust preprocessing, prompt mapping, mask selection, and
//!   upsampling) compared with the PyTorch reference masks of `compare_onnx_outputs.py`;
//! - the checks `SegmentationGraphsLoadWithProductionSessionSettings`,
//!   `DecoderOutputShapeMatchesExportedSpecification`, and `HostTokenizerMatchesPythonBpeTokens`.
//!
//! `matte` mode loads `subject_matte.onnx` of `scripts/ai_models/birefnet/export_onnx.py` (one
//! graph: pixels to a foreground alpha at the model size) with the same session settings. It
//! measures load, first inference, and warm latency, runs every photo of the prompts file, resizes
//! the alpha to the photo size, compares it with the PyTorch reference alpha of
//! `scripts/ai_models/birefnet/compare_onnx_outputs.py`, and writes soft overlays and the alpha
//! images (`RustPipelineMatteMatchesPytorchReference`, `MatteOutputShapeMatchesExportedSpecification`).
//!
//! `ping` mode measures the `HealthService.Ping` round trip of a running sidecar with one
//! persistent channel and with a new channel per call (`PingLatencyIsReportedForPersistentAndPerCallChannels`).
//!
//! The program needs downloaded model files, so it is not part of `cargo test`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use alcedo_mind::proto::common::PingRequest;
use alcedo_mind::proto::common::health_service_client::HealthServiceClient;
use alcedo_mind::service::ort_runtime::{self, DeviceRequest};
use anyhow::{Context, Result, anyhow, bail};
use image::{DynamicImage, GrayImage, Luma, Rgb, RgbImage, imageops::FilterType};
use ort::session::Session;
use ort::value::Tensor;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

const TEXT_CONTEXT_LENGTH: usize = 16;
const TEXT_SCORE_THRESHOLD: f32 = 0.5;
const STABILITY_DELTA: f32 = 0.05;
const STABILITY_THRESHOLD: f32 = 0.98;
const LOW_RES_LOGIT_LIMIT: f32 = 32.0;
const REQUIRED_REFERENCE_IOU: f64 = 0.99;
const CLIP_START_TOKEN: u32 = 49406;
const CLIP_END_TOKEN: u32 = 49407;

const POINT_ENCODER: &str = "point_encoder";
const POINT_DECODER: &str = "point_decoder";
const TEXT_ENCODER: &str = "text_encoder";
const DETECTOR_GROUNDING: &str = "detector_grounding";
const GRAPH_NAMES: [&str; 4] = [
    POINT_ENCODER,
    POINT_DECODER,
    TEXT_ENCODER,
    DETECTOR_GROUNDING,
];

/// Input size, low-resolution mask size, and pixel normalization of the exported graphs.
/// `export_report.json` states them in `model_input` (SAM 2.1 exports of
/// `scripts/ai_models/sam2/export_onnx.py`). A report without `model_input` is an EfficientSAM3
/// export: 1008 px, 288 px masks, mean and std 0.5.
#[derive(Clone, Copy, Debug)]
struct ModelInput {
    image_size: usize,
    low_res_mask_size: usize,
    mean: [f32; 3],
    std: [f32; 3],
}

impl ModelInput {
    const EFFICIENTSAM3: Self = Self {
        image_size: 1008,
        low_res_mask_size: 288,
        mean: [0.5; 3],
        std: [0.5; 3],
    };

    fn from_export_report(report: &Value) -> Result<Self> {
        let entry = &report["model_input"];
        if entry.is_null() {
            return Ok(Self::EFFICIENTSAM3);
        }
        let size = |key: &str| -> Result<usize> {
            entry[key]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| anyhow!("model_input.{key} must be a positive integer"))
        };
        let triple = |key: &str| -> Result<[f32; 3]> {
            let values: Vec<f32> = entry[key]
                .as_array()
                .ok_or_else(|| anyhow!("model_input.{key} must be a list"))?
                .iter()
                .map(|v| v.as_f64().map(|v| v as f32))
                .collect::<Option<_>>()
                .ok_or_else(|| anyhow!("model_input.{key} must hold numbers"))?;
            values
                .try_into()
                .map_err(|_| anyhow!("model_input.{key} must have three values"))
        };
        Ok(Self {
            image_size: size("image_size")?,
            low_res_mask_size: size("low_res_mask_size")?,
            mean: triple("pixel_mean")?,
            std: triple("pixel_std")?,
        })
    }
}

/// Side length of a square NCHW pixel tensor with three channels.
fn pixel_side(pixels: &[f32]) -> usize {
    ((pixels.len() / 3) as f64).sqrt().round() as usize
}

/// Side length of one square plane in a tensor of `planes` planes.
fn plane_side(values: usize, planes: usize) -> usize {
    ((values / planes) as f64).sqrt().round() as usize
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_default();
    let options = parse_options(args)?;
    match mode.as_str() {
        "runtime" => run_runtime_report(&RuntimeOptions::from_map(&options)?),
        "matte" => run_matte_report(&MatteOptions::from_map(&options)?),
        "ping" => run_ping_report(&PingOptions::from_map(&options)?),
        _ => bail!(
            "usage: segmentation_runtime_report runtime --onnx-dir DIR --device DEVICE --photos DIR \
             --prompts FILE --out FILE [--reference-masks DIR] [--overlay-dir DIR] [--overlay-max-edge N] [--tokenizer FILE] \
             [--encoder-runs N] [--decoder-runs N] [--profile-dir DIR] \n             [--session-settings production|coreml-mlprogram] [--profile-graphs NAME,NAME] [--graph-devices NAME=DEVICE,...]\n\
             segmentation_runtime_report matte --onnx-dir DIR --device DEVICE --photos DIR --prompts FILE --out FILE \
             [--reference-alpha DIR] [--overlay-dir DIR] [--overlay-max-edge N] [--runs N] [--profile-dir DIR]\n\
             segmentation_runtime_report ping --endpoint URL --out FILE [--calls N]"
        ),
    }
}

fn parse_options(args: impl Iterator<Item = String>) -> Result<BTreeMap<String, String>> {
    let mut options = BTreeMap::new();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let key = arg
            .strip_prefix("--")
            .ok_or_else(|| anyhow!("unexpected argument {arg:?}"))?;
        let value = args
            .next()
            .ok_or_else(|| anyhow!("missing value after --{key}"))?;
        options.insert(key.to_string(), value);
    }
    Ok(options)
}

fn required(options: &BTreeMap<String, String>, key: &str) -> Result<String> {
    options
        .get(key)
        .cloned()
        .ok_or_else(|| anyhow!("missing --{key}"))
}

fn parse_count(options: &BTreeMap<String, String>, key: &str, default: usize) -> Result<usize> {
    match options.get(key) {
        Some(value) => value
            .parse()
            .with_context(|| format!("failed to parse --{key} {value:?}")),
        None => Ok(default),
    }
}

/// Session settings of a measurement. Only `Production` is the sidecar setting; every other value
/// is recorded in the report as a non-production measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionSettings {
    /// `ort_runtime::session_builder`: the settings of the `alcedo_mind` sidecar.
    Production,
    /// The production optimization level with the Core ML provider in the MLProgram model format
    /// (the production Core ML provider uses the default NeuralNetwork format), then CPU.
    CoreMlMlProgram,
}

impl SessionSettings {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "production" => Ok(Self::Production),
            "coreml-mlprogram" => Ok(Self::CoreMlMlProgram),
            _ => bail!(
                "unsupported --session-settings {value:?}; use production or coreml-mlprogram"
            ),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Production => "production (ort_runtime::session_builder)",
            Self::CoreMlMlProgram => {
                "NOT production: Level3 optimization, CoreML(ComputeUnits::All, ModelFormat::MLProgram), CPU"
            }
        }
    }

    fn builder(self, device: DeviceRequest) -> Result<ort::session::builder::SessionBuilder> {
        match self {
            Self::Production => ort_runtime::session_builder(device),
            Self::CoreMlMlProgram => {
                if !cfg!(target_os = "macos") {
                    bail!("--session-settings coreml-mlprogram needs macOS");
                }
                let builder = Session::builder()
                    .map_err(|e| anyhow!("failed to create ORT session builder: {e}"))?
                    .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
                    .map_err(|e| anyhow!("failed to set ORT optimization level: {e}"))?;
                builder
                    .with_execution_providers([
                        ort::ep::CoreML::default()
                            .with_compute_units(ort::ep::coreml::ComputeUnits::All)
                            .with_model_format(ort::ep::coreml::ModelFormat::MLProgram)
                            .build(),
                        ort::ep::CPU::default().build(),
                    ])
                    .map_err(|e| anyhow!("failed to configure the Core ML MLProgram session: {e}"))
            }
        }
    }

    fn load(self, path: &Path, device: DeviceRequest) -> Result<Session> {
        match self {
            Self::Production => ort_runtime::load_session(path, device),
            _ => self
                .builder(device)?
                .commit_from_file(path)
                .map_err(|e| anyhow!("failed to load ONNX model {}: {e}", path.display())),
        }
    }
}

struct RuntimeOptions {
    onnx_dir: PathBuf,
    settings: SessionSettings,
    device_text: String,
    device: DeviceRequest,
    photos: PathBuf,
    prompts: PathBuf,
    out: PathBuf,
    reference_masks: Option<PathBuf>,
    overlay_dir: Option<PathBuf>,
    /// Long edge of the overlay images; 0 keeps the photo size.
    overlay_max_edge: u32,
    tokenizer: Option<PathBuf>,
    profile_dir: Option<PathBuf>,
    profile_graphs: Vec<String>,
    /// Device per graph for a diagnostic run (`--graph-devices point_decoder=cpu`). Graphs that
    /// are not listed use `--device`.
    graph_devices: BTreeMap<String, String>,
    encoder_runs: usize,
    decoder_runs: usize,
}

impl RuntimeOptions {
    fn from_map(options: &BTreeMap<String, String>) -> Result<Self> {
        let device_text = required(options, "device")?;
        Ok(Self {
            onnx_dir: required(options, "onnx-dir")?.into(),
            settings: SessionSettings::parse(
                options
                    .get("session-settings")
                    .map_or("production", String::as_str),
            )?,
            device: ort_runtime::parse_device_request(&device_text)?,
            device_text,
            photos: required(options, "photos")?.into(),
            prompts: required(options, "prompts")?.into(),
            out: required(options, "out")?.into(),
            reference_masks: options.get("reference-masks").map(PathBuf::from),
            overlay_dir: options.get("overlay-dir").map(PathBuf::from),
            overlay_max_edge: parse_count(options, "overlay-max-edge", 0)? as u32,
            tokenizer: options.get("tokenizer").map(PathBuf::from),
            profile_dir: options.get("profile-dir").map(PathBuf::from),
            graph_devices: match options.get("graph-devices") {
                Some(list) => list
                    .split(',')
                    .map(|pair| {
                        let (graph, device) = pair
                            .split_once('=')
                            .ok_or_else(|| anyhow!("--graph-devices needs graph=device pairs"))?;
                        ort_runtime::parse_device_request(device)?;
                        Ok((graph.trim().to_string(), device.trim().to_string()))
                    })
                    .collect::<Result<_>>()?,
                None => BTreeMap::new(),
            },
            profile_graphs: options.get("profile-graphs").map_or_else(
                || GRAPH_NAMES.iter().map(|n| n.to_string()).collect(),
                |list| list.split(',').map(|n| n.trim().to_string()).collect(),
            ),
            encoder_runs: parse_count(options, "encoder-runs", 30)?,
            decoder_runs: parse_count(options, "decoder-runs", 100)?,
        })
    }
}

struct PingOptions {
    endpoint: String,
    out: PathBuf,
    calls: usize,
}

impl PingOptions {
    fn from_map(options: &BTreeMap<String, String>) -> Result<Self> {
        Ok(Self {
            endpoint: required(options, "endpoint")?,
            out: required(options, "out")?.into(),
            calls: parse_count(options, "calls", 1000)?,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Statistics, memory, and report helpers

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// p50, p95, and maximum with the nearest-rank method.
fn latency_summary(samples_ms: &[f64]) -> Value {
    if samples_ms.is_empty() {
        return json!({ "status": "unavailable", "reason": "no samples" });
    }
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    let rank =
        |p: f64| sorted[((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len()) - 1];
    json!({
        "runs": sorted.len(),
        "p50_ms": rank(0.50),
        "p95_ms": rank(0.95),
        "max_ms": sorted[sorted.len() - 1],
        "min_ms": sorted[0],
    })
}

fn time_runs(count: usize, mut run: impl FnMut() -> Result<()>) -> Result<Vec<f64>> {
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let started = Instant::now();
        run()?;
        samples.push(milliseconds(started.elapsed()));
    }
    Ok(samples)
}

#[cfg(windows)]
mod process_memory {
    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(
            process: isize,
            counters: *mut ProcessMemoryCounters,
            size: u32,
        ) -> i32;
    }

    pub fn sample() -> serde_json::Value {
        let mut counters = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            ..Default::default()
        };
        // SAFETY: the counters struct has the PROCESS_MEMORY_COUNTERS layout and its size is set.
        let ok =
            unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        if ok == 0 {
            return serde_json::json!({ "status": "unavailable", "reason": "K32GetProcessMemoryInfo failed" });
        }
        serde_json::json!({
            "working_set_mb": counters.working_set_size as f64 / 1048576.0,
            "peak_working_set_mb": counters.peak_working_set_size as f64 / 1048576.0,
            "private_bytes_mb": counters.pagefile_usage as f64 / 1048576.0,
            "gpu_memory": "unavailable: the process counters do not include dedicated GPU memory",
        })
    }
}

#[cfg(target_os = "macos")]
mod process_memory {
    #[repr(C)]
    struct TimeValue {
        seconds: i64,
        microseconds: i32,
    }

    #[repr(C)]
    struct ResourceUsage {
        user_time: TimeValue,
        system_time: TimeValue,
        max_resident_set: i64,
        other: [i64; 13],
    }

    unsafe extern "C" {
        fn getrusage(who: i32, usage: *mut ResourceUsage) -> i32;
    }

    pub fn sample() -> serde_json::Value {
        let mut usage = std::mem::MaybeUninit::<ResourceUsage>::zeroed();
        // SAFETY: RUSAGE_SELF (0) fills the rusage struct, which has the Darwin layout.
        let result = unsafe { getrusage(0, usage.as_mut_ptr()) };
        if result != 0 {
            return serde_json::json!({ "status": "unavailable", "reason": "getrusage failed" });
        }
        // SAFETY: getrusage succeeded and initialized the struct.
        let usage = unsafe { usage.assume_init() };
        serde_json::json!({
            "peak_resident_set_mb": usage.max_resident_set as f64 / 1048576.0,
            "resident_set_mb": "unavailable: getrusage reports only the peak",
            "gpu_memory": "unavailable: the process counters do not include GPU memory",
        })
    }
}

#[cfg(not(any(windows, target_os = "macos")))]
mod process_memory {
    pub fn sample() -> serde_json::Value {
        serde_json::json!({ "status": "unavailable", "reason": "no memory probe for this OS" })
    }
}

// ---------------------------------------------------------------------------------------------
// Image and mask helpers

/// The preprocessing of `Sam3Processor.set_image` and `SAM2Transforms`: stretch to the square
/// model input (no letterbox), then (value / 255 - mean) / std in NCHW order.
fn preprocess(image: &RgbImage, input: &ModelInput) -> Vec<f32> {
    let size = input.image_size;
    let resized = image::imageops::resize(image, size as u32, size as u32, FilterType::Triangle);
    let plane = size * size;
    let mut pixels = vec![0.0f32; 3 * plane];
    for (index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            pixels[channel * plane + index] =
                (pixel[channel] as f32 / 255.0 - input.mean[channel]) / input.std[channel];
        }
    }
    pixels
}

/// Bilinear upsampling with `align_corners=False`, as `torch.nn.functional.interpolate`, then
/// the threshold `logit > 0`.
fn upsample_to_mask(logits: &[f32], size: usize, width: u32, height: u32) -> GrayImage {
    let scale_x = size as f32 / width as f32;
    let scale_y = size as f32 / height as f32;
    let source = |x: u32, scale: f32| -> (usize, usize, f32) {
        let position = ((x as f32 + 0.5) * scale - 0.5).max(0.0);
        let low = (position.floor() as usize).min(size - 1);
        let high = (low + 1).min(size - 1);
        (low, high, position - low as f32)
    };
    let columns: Vec<_> = (0..width).map(|x| source(x, scale_x)).collect();
    let mut mask = GrayImage::new(width, height);
    for y in 0..height {
        let (y0, y1, fy) = source(y, scale_y);
        for (x, &(x0, x1, fx)) in columns.iter().enumerate() {
            let top = logits[y0 * size + x0] * (1.0 - fx) + logits[y0 * size + x1] * fx;
            let bottom = logits[y1 * size + x0] * (1.0 - fx) + logits[y1 * size + x1] * fx;
            let value = top * (1.0 - fy) + bottom * fy;
            mask.put_pixel(x as u32, y, Luma([if value > 0.0 { 255 } else { 0 }]));
        }
    }
    mask
}

fn mask_iou(a: &GrayImage, b: &GrayImage) -> Result<f64> {
    if a.dimensions() != b.dimensions() {
        bail!(
            "mask size {:?} differs from reference size {:?}",
            a.dimensions(),
            b.dimensions()
        );
    }
    let (mut intersection, mut union) = (0u64, 0u64);
    for (pa, pb) in a.pixels().zip(b.pixels()) {
        let (ia, ib) = (pa[0] > 127, pb[0] > 127);
        intersection += (ia && ib) as u64;
        union += (ia || ib) as u64;
    }
    Ok(if union == 0 {
        1.0
    } else {
        intersection as f64 / union as f64
    })
}

fn mask_fraction(mask: &GrayImage) -> f64 {
    let inside = mask.pixels().filter(|p| p[0] > 127).count();
    inside as f64 / (mask.width() as f64 * mask.height() as f64)
}

fn compare_with_reference(
    reference_dir: Option<&Path>,
    file_name: &str,
    mask: &GrayImage,
) -> Value {
    let Some(directory) = reference_dir else {
        return json!({ "status": "unavailable", "reason": "no --reference-masks directory" });
    };
    let path = directory.join(file_name);
    let reference = match image::open(&path) {
        Ok(image) => image.to_luma8(),
        Err(error) => {
            return json!({ "status": "unavailable", "reason": format!("{}: {error}", path.display()) });
        }
    };
    match mask_iou(mask, &reference) {
        Ok(iou) => json!({
            "iou": iou,
            "passed": iou >= REQUIRED_REFERENCE_IOU,
            "reference": file_name,
        }),
        Err(error) => json!({ "status": "failed", "reason": error.to_string() }),
    }
}

const OVERLAY_COLOR: [f32; 3] = [30.0, 144.0, 255.0];

fn write_overlay(
    photo: &RgbImage,
    mask: &GrayImage,
    points: &[(f32, f32, i32)],
    boxes: &[[f32; 4]],
    max_edge: u32,
    path: &Path,
) -> Result<()> {
    let (photo_width, photo_height) = photo.dimensions();
    let long_edge = photo_width.max(photo_height);
    let (mut overlay, mask) = if max_edge > 0 && long_edge > max_edge {
        let scale = max_edge as f64 / long_edge as f64;
        let width = ((photo_width as f64 * scale).round() as u32).max(1);
        let height = ((photo_height as f64 * scale).round() as u32).max(1);
        (
            image::imageops::resize(photo, width, height, FilterType::Triangle),
            image::imageops::resize(mask, width, height, FilterType::Triangle),
        )
    } else {
        (photo.clone(), mask.clone())
    };
    for (pixel, coverage) in overlay.pixels_mut().zip(mask.pixels()) {
        if coverage[0] > 127 {
            for channel in 0..3 {
                pixel[channel] =
                    (pixel[channel] as f32 * 0.45 + OVERLAY_COLOR[channel] * 0.55) as u8;
            }
        }
    }
    let (width, height) = overlay.dimensions();
    let marker = (width.max(height) / 150).max(3) as i64;
    let mut paint = |x: i64, y: i64, color: Rgb<u8>| {
        if x >= 0 && y >= 0 && (x as u32) < width && (y as u32) < height {
            overlay.put_pixel(x as u32, y as u32, color);
        }
    };
    for &(x, y, label) in points {
        let color = if label == 1 {
            Rgb([40, 220, 60])
        } else {
            Rgb([235, 40, 40])
        };
        let (cx, cy) = ((x * width as f32) as i64, (y * height as f32) as i64);
        for dy in -marker..=marker {
            for dx in -marker..=marker {
                paint(cx + dx, cy + dy, color);
            }
        }
    }
    let line = (marker / 2).max(1);
    for b in boxes {
        let (x0, y0) = ((b[0] * width as f32) as i64, (b[1] * height as f32) as i64);
        let (x1, y1) = ((b[2] * width as f32) as i64, (b[3] * height as f32) as i64);
        for t in 0..line {
            for x in x0..=x1 {
                paint(x, y0 + t, Rgb([255, 210, 0]));
                paint(x, y1 - t, Rgb([255, 210, 0]));
            }
            for y in y0..=y1 {
                paint(x0 + t, y, Rgb([255, 210, 0]));
                paint(x1 - t, y, Rgb([255, 210, 0]));
            }
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    DynamicImage::ImageRgb8(overlay)
        .save(path)
        .with_context(|| format!("failed to write {}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// Graph runners

struct PointFeatures {
    image_embed: Vec<f32>,
    high_res_feats_0: Vec<f32>,
    high_res_feats_1: Vec<f32>,
    shapes: [Vec<i64>; 3],
}

fn output_f32(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>)> {
    let value = outputs
        .get(name)
        .ok_or_else(|| anyhow!("missing output {name:?}"))?;
    let (shape, data) = value
        .try_extract_tensor::<f32>()
        .map_err(|e| anyhow!("failed to read output {name:?}: {e}"))?;
    Ok((shape.iter().copied().collect(), data.to_vec()))
}

fn shape_usize(shape: &[i64]) -> Vec<usize> {
    shape.iter().map(|&d| d as usize).collect()
}

fn run_point_encoder(session: &mut Session, pixels: &[f32]) -> Result<PointFeatures> {
    let side = pixel_side(pixels);
    let input = Tensor::from_array(([1usize, 3, side, side], pixels.to_vec()))
        .map_err(|e| anyhow!("failed to build pixel tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs! { "pixel_values" => input })
        .map_err(|e| anyhow!("point encoder run failed: {e}"))?;
    let (s0, image_embed) = output_f32(&outputs, "image_embed")?;
    let (s1, high_res_feats_0) = output_f32(&outputs, "high_res_feats_0")?;
    let (s2, high_res_feats_1) = output_f32(&outputs, "high_res_feats_1")?;
    Ok(PointFeatures {
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        shapes: [s0, s1, s2],
    })
}

struct DecoderOutput {
    mask_logits: Vec<f32>,
    iou_scores: Vec<f32>,
    shapes: BTreeMap<String, Vec<i64>>,
}

/// `coords` are model-input pixels; labels 1 positive, 0 negative, 2 and 3 box corners.
fn run_point_decoder(
    session: &mut Session,
    features: &PointFeatures,
    coords: &[[f32; 2]],
    labels: &[f32],
) -> Result<DecoderOutput> {
    let count = coords.len();
    // The low-resolution mask has the size of the stride-4 features (`high_res_feats_0`).
    let mask_size = features.shapes[1].last().copied().unwrap_or(0) as usize;
    let tensor = |shape: &[i64], data: &[f32]| {
        Tensor::from_array((shape_usize(shape), data.to_vec()))
            .map_err(|e| anyhow!("failed to build decoder input: {e}"))
    };
    let outputs = session
        .run(ort::inputs! {
            "image_embed" => tensor(&features.shapes[0], &features.image_embed)?,
            "high_res_feats_0" => tensor(&features.shapes[1], &features.high_res_feats_0)?,
            "high_res_feats_1" => tensor(&features.shapes[2], &features.high_res_feats_1)?,
            "point_coords" => Tensor::from_array(([1usize, count, 2], coords.concat()))?,
            "point_labels" => Tensor::from_array(([1usize, count], labels.to_vec()))?,
            "mask_input" => Tensor::from_array((
                [1usize, 1, mask_size, mask_size],
                vec![0.0f32; mask_size * mask_size],
            ))?,
            "has_mask_input" => Tensor::from_array(([1usize], vec![0.0f32]))?,
        })
        .map_err(|e| anyhow!("point decoder run failed: {e}"))?;
    let mut shapes = BTreeMap::new();
    let (mask_shape, mask_logits) = output_f32(&outputs, "mask_logits")?;
    let (iou_shape, iou_scores) = output_f32(&outputs, "iou_scores")?;
    let (object_shape, _) = output_f32(&outputs, "object_score_logits")?;
    shapes.insert("mask_logits".to_string(), mask_shape);
    shapes.insert("iou_scores".to_string(), iou_shape);
    shapes.insert("object_score_logits".to_string(), object_shape);
    Ok(DecoderOutput {
        mask_logits,
        iou_scores,
        shapes,
    })
}

/// Mask selection of `MaskDecoder.forward`: tokens 1-3 for multimask output; token 0, or the best
/// multimask token when token 0 is not stable, for single-mask output. Returns (token, score).
fn select_masks(output: &DecoderOutput, multimask: bool) -> Vec<(usize, f32)> {
    let plane = output.mask_logits.len() / 4;
    if multimask {
        return (1..4)
            .map(|token| (token, output.iou_scores[token]))
            .collect();
    }
    let single = &output.mask_logits[..plane];
    let inner = single.iter().filter(|&&v| v > STABILITY_DELTA).count();
    let outer = single.iter().filter(|&&v| v > -STABILITY_DELTA).count();
    let stability = if outer > 0 {
        inner as f32 / outer as f32
    } else {
        1.0
    };
    if stability >= STABILITY_THRESHOLD {
        return vec![(0, output.iou_scores[0])];
    }
    let best = (1..4)
        .max_by(|&a, &b| output.iou_scores[a].total_cmp(&output.iou_scores[b]))
        .unwrap_or(1);
    vec![(best, output.iou_scores[best])]
}

fn token_logits(output: &DecoderOutput, token: usize) -> Vec<f32> {
    let plane = output.mask_logits.len() / 4;
    output.mask_logits[token * plane..(token + 1) * plane]
        .iter()
        .map(|v| v.clamp(-LOW_RES_LOGIT_LIMIT, LOW_RES_LOGIT_LIMIT))
        .collect()
}

struct TextFeatures {
    language_mask: Vec<bool>,
    language_mask_shape: Vec<i64>,
    language_features: Vec<f32>,
    language_features_shape: Vec<i64>,
}

fn run_text_encoder(session: &mut Session, tokens: &[i64]) -> Result<TextFeatures> {
    let outputs = session
        .run(ort::inputs! {
            "tokenized_ids" => Tensor::from_array(([1usize, TEXT_CONTEXT_LENGTH], tokens.to_vec()))?,
        })
        .map_err(|e| anyhow!("text encoder run failed: {e}"))?;
    let mask_value = outputs
        .get("language_mask")
        .ok_or_else(|| anyhow!("missing output language_mask"))?;
    let (mask_shape, mask) = mask_value
        .try_extract_tensor::<bool>()
        .map_err(|e| anyhow!("failed to read language_mask: {e}"))?;
    let (features_shape, features) = output_f32(&outputs, "language_features")?;
    Ok(TextFeatures {
        language_mask: mask.to_vec(),
        language_mask_shape: mask_shape.iter().copied().collect(),
        language_features: features,
        language_features_shape: features_shape,
    })
}

struct Detections {
    scores: Vec<f32>,
    mask_logits: Vec<f32>,
    mask_size: usize,
}

fn run_detector(session: &mut Session, pixels: &[f32], text: &TextFeatures) -> Result<Detections> {
    let outputs = session
        .run(ort::inputs! {
            "pixel_values" => Tensor::from_array(([1usize, 3, pixel_side(pixels), pixel_side(pixels)], pixels.to_vec()))?,
            "language_features" => Tensor::from_array((
                shape_usize(&text.language_features_shape),
                text.language_features.clone(),
            ))?,
            "language_mask" => Tensor::from_array((
                shape_usize(&text.language_mask_shape),
                text.language_mask.clone(),
            ))?,
        })
        .map_err(|e| anyhow!("detector run failed: {e}"))?;
    let (_, scores) = output_f32(&outputs, "scores")?;
    let (mask_shape, mask_logits) = output_f32(&outputs, "mask_logits")?;
    let mask_size = *mask_shape
        .last()
        .ok_or_else(|| anyhow!("empty mask_logits shape"))? as usize;
    Ok(Detections {
        scores,
        mask_logits,
        mask_size,
    })
}

/// CLIP BPE tokens in the layout of the Python `SimpleTokenizer`: start token, the text tokens,
/// end token, zero padding to 16. A longer text is cut and its last token becomes the end token.
fn tokenize(tokenizer: &Tokenizer, text: &str) -> Result<Vec<i64>> {
    let encoding = tokenizer
        .encode(text, false)
        .map_err(|e| anyhow!("failed to tokenize {text:?}: {e}"))?;
    let mut ids = vec![CLIP_START_TOKEN];
    ids.extend(
        encoding
            .get_ids()
            .iter()
            .copied()
            .filter(|&id| id != CLIP_START_TOKEN && id != CLIP_END_TOKEN),
    );
    ids.push(CLIP_END_TOKEN);
    if ids.len() > TEXT_CONTEXT_LENGTH {
        ids.truncate(TEXT_CONTEXT_LENGTH);
        ids[TEXT_CONTEXT_LENGTH - 1] = CLIP_END_TOKEN;
    }
    ids.resize(TEXT_CONTEXT_LENGTH, 0);
    Ok(ids.into_iter().map(i64::from).collect())
}

// ---------------------------------------------------------------------------------------------
// Operator placement

/// Runs `run` on a second session that adds only the profiling option to the production settings,
/// then counts the graph nodes per execution provider from the ONNX Runtime profile.
fn operator_placement(
    graph_path: &Path,
    device: DeviceRequest,
    settings: SessionSettings,
    profile_prefix: &Path,
    run: impl FnOnce(&mut Session) -> Result<()>,
) -> Result<Value> {
    if let Some(parent) = profile_prefix.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut session = settings
        .builder(device)?
        .with_profiling(profile_prefix)
        .map_err(|e| anyhow!("failed to enable profiling: {e}"))?
        .commit_from_file(graph_path)
        .map_err(|e| {
            anyhow!(
                "failed to load {} with profiling: {e}",
                graph_path.display()
            )
        })?;
    run(&mut session)?;
    let profile_path = session
        .end_profiling()
        .map_err(|e| anyhow!("failed to end profiling: {e}"))?;
    let profile: Value = serde_json::from_str(&fs::read_to_string(&profile_path)?)
        .with_context(|| format!("failed to parse profile {profile_path}"))?;
    let events = profile
        .as_array()
        .ok_or_else(|| anyhow!("profile {profile_path} is not a JSON array"))?;

    let mut nodes_per_provider: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let mut time_per_provider_us: BTreeMap<String, f64> = BTreeMap::new();
    let mut seen_nodes = std::collections::BTreeSet::new();
    for event in events {
        if event.get("cat").and_then(Value::as_str) != Some("Node") {
            continue;
        }
        let Some(args) = event.get("args") else {
            continue;
        };
        let (Some(provider), Some(op_name)) = (
            args.get("provider").and_then(Value::as_str),
            args.get("op_name").and_then(Value::as_str),
        ) else {
            continue;
        };
        let name = event
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let node = name.trim_end_matches("_kernel_time").to_string();
        if name.ends_with("_kernel_time") {
            *time_per_provider_us
                .entry(provider.to_string())
                .or_default() += event.get("dur").and_then(Value::as_f64).unwrap_or(0.0);
            if seen_nodes.insert((provider.to_string(), node)) {
                *nodes_per_provider
                    .entry(provider.to_string())
                    .or_default()
                    .entry(op_name.to_string())
                    .or_default() += 1;
            }
        }
    }
    let node_totals: BTreeMap<_, _> = nodes_per_provider
        .iter()
        .map(|(provider, ops)| (provider.clone(), ops.values().sum::<u64>()))
        .collect();
    Ok(json!({
        "settings": format!("{} plus SessionBuilder::with_profiling", settings.label()),
        "profile": profile_path,
        "node_count_per_provider": node_totals,
        "kernel_time_ms_per_provider": time_per_provider_us
            .iter()
            .map(|(p, us)| (p.clone(), us / 1000.0))
            .collect::<BTreeMap<_, _>>(),
        "operator_types_per_provider": nodes_per_provider,
    }))
}

// ---------------------------------------------------------------------------------------------
// Runtime report

struct PromptSet {
    name: String,
    points: Vec<(f32, f32, i32)>,
    boxes: Vec<[f32; 4]>,
}

impl PromptSet {
    /// Model-input coordinates: normalized image coordinates multiplied by the model input size
    /// (1008 or 1024). A box becomes two points with labels 2 and 3 at the start of the list.
    fn model_inputs(&self, input: &ModelInput) -> (Vec<[f32; 2]>, Vec<f32>) {
        let size = input.image_size as f32;
        let mut coords = Vec::new();
        let mut labels = Vec::new();
        for b in &self.boxes {
            coords.push([b[0] * size, b[1] * size]);
            coords.push([b[2] * size, b[3] * size]);
            labels.extend([2.0, 3.0]);
        }
        for &(x, y, label) in &self.points {
            coords.push([x * size, y * size]);
            labels.push(label as f32);
        }
        (coords, labels)
    }
}

fn prompt_sets(entry: &Value) -> Result<Vec<PromptSet>> {
    let mut sets = Vec::new();
    for set in entry["points"].as_array().into_iter().flatten() {
        let points = set["points"]
            .as_array()
            .ok_or_else(|| anyhow!("points entry without a points list"))?
            .iter()
            .map(|p| {
                Ok((
                    p[0].as_f64().ok_or_else(|| anyhow!("bad point x"))? as f32,
                    p[1].as_f64().ok_or_else(|| anyhow!("bad point y"))? as f32,
                    p[2].as_i64().ok_or_else(|| anyhow!("bad point label"))? as i32,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        sets.push(PromptSet {
            name: set["name"].as_str().unwrap_or("points").to_string(),
            points,
            boxes: Vec::new(),
        });
    }
    for set in entry["boxes"].as_array().into_iter().flatten() {
        let values: Vec<f32> = set["box"]
            .as_array()
            .ok_or_else(|| anyhow!("boxes entry without a box"))?
            .iter()
            .map(|v| v.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        if values.len() != 4 || values.iter().any(|v| !v.is_finite()) {
            bail!("box must have four numbers: {set}");
        }
        sets.push(PromptSet {
            name: set["name"].as_str().unwrap_or("box").to_string(),
            points: Vec::new(),
            boxes: vec![[values[0], values[1], values[2], values[3]]],
        });
    }
    Ok(sets)
}

fn run_runtime_report(options: &RuntimeOptions) -> Result<()> {
    ort_runtime::initialize_ort_environment()?;
    let export_report: Value = serde_json::from_str(
        &fs::read_to_string(options.onnx_dir.join("export_report.json"))
            .context("failed to read export_report.json")?,
    )?;
    let model_input = ModelInput::from_export_report(&export_report)?;
    let low_res_mask_size = model_input.low_res_mask_size;
    let prompts: Value = serde_json::from_str(&fs::read_to_string(&options.prompts)?)?;
    let mut report = json!({
        "device_request": options.device_text,
        "graph_device_overrides": options.graph_devices,
        "session_settings": options.settings.label(),
        "model_input": {
            "image_size": model_input.image_size,
            "low_res_mask_size": low_res_mask_size,
            "pixel_mean": model_input.mean,
            "pixel_std": model_input.std,
        },
        "device_description": ort_runtime::describe_device_request(options.device),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "memory_before_load": process_memory::sample(),
        "checks": {},
        "graphs": {},
        "images": [],
    });

    // SegmentationGraphsLoadWithProductionSessionSettings
    let mut sessions: BTreeMap<&str, Session> = BTreeMap::new();
    let mut load_check = json!({});
    for name in GRAPH_NAMES {
        let path = options.onnx_dir.join(format!("{name}.onnx"));
        if !path.exists() {
            load_check[name] = json!({ "status": "unavailable", "reason": "graph file not found" });
            continue;
        }
        let started = Instant::now();
        let loaded = match options.graph_devices.get(name) {
            Some(device_text) => {
                let device = ort_runtime::parse_device_request(device_text)?;
                if matches!(device, DeviceRequest::CoreMl(_)) {
                    options.settings.load(&path, device)
                } else {
                    ort_runtime::load_session(&path, device)
                }
            }
            None => options.settings.load(&path, options.device),
        };
        match loaded {
            Ok(session) => {
                let load_ms = milliseconds(started.elapsed());
                println!("[load] {name}: {load_ms:.0} ms");
                load_check[name] = json!({ "status": "passed" });
                report["graphs"][name] = json!({
                    "load_ms": load_ms,
                    "memory_after_load": process_memory::sample(),
                });
                sessions.insert(name, session);
            }
            Err(error) => {
                println!("[load] {name}: FAILED {error}");
                load_check[name] = json!({ "status": "failed", "error": format!("{error:#}") });
            }
        }
    }
    let load_check_name = if options.settings == SessionSettings::Production {
        "SegmentationGraphsLoadWithProductionSessionSettings"
    } else {
        "SegmentationGraphsLoadWithNonProductionSessionSettings"
    };
    report["checks"][load_check_name] = load_check;
    report["memory_after_all_loads"] = process_memory::sample();

    let tokenizer = match &options.tokenizer {
        Some(path) => {
            Some(Tokenizer::from_file(path).map_err(|e| anyhow!("failed to load tokenizer: {e}"))?)
        }
        None => None,
    };
    let mut tokenizer_check = json!({});
    let mut shape_check: Option<Value> = None;
    let mut reference_failures = 0usize;
    let mut reference_comparisons = 0usize;
    let mut first_image = true;
    let reference_dir = options.reference_masks.as_deref();

    for entry in prompts["images"].as_array().into_iter().flatten() {
        let file = entry["file"]
            .as_str()
            .ok_or_else(|| anyhow!("image entry without file"))?;
        let stem = Path::new(file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(file)
            .to_string();
        let started = Instant::now();
        let photo = image::open(options.photos.join(file))
            .with_context(|| format!("failed to open {file}"))?
            .to_rgb8();
        let decode_ms = milliseconds(started.elapsed());
        let started = Instant::now();
        let pixels = preprocess(&photo, &model_input);
        let preprocess_ms = milliseconds(started.elapsed());
        let (width, height) = photo.dimensions();
        let mut image_report = json!({
            "file": file, "width": width, "height": height,
            "decode_ms": decode_ms, "preprocess_ms": preprocess_ms,
            "prompts": [],
        });
        println!(
            "[image] {file} {width}x{height}: decode {decode_ms:.0} ms, preprocess {preprocess_ms:.0} ms"
        );

        let mut features = None;
        if let Some(encoder) = sessions.get_mut(POINT_ENCODER) {
            let started = Instant::now();
            let encoded = run_point_encoder(encoder, &pixels)?;
            let encode_ms = milliseconds(started.elapsed());
            image_report["point_encoder_ms"] = json!(encode_ms);
            if first_image {
                report["graphs"][POINT_ENCODER]["first_inference_ms"] = json!(encode_ms);
                let samples = time_runs(options.encoder_runs, || {
                    run_point_encoder(encoder, &pixels).map(|_| ())
                })?;
                report["graphs"][POINT_ENCODER]["warm"] = latency_summary(&samples);
                report["graphs"][POINT_ENCODER]["memory_after_inference"] =
                    process_memory::sample();
                println!(
                    "[point_encoder] first {encode_ms:.0} ms, warm {}",
                    report["graphs"][POINT_ENCODER]["warm"]
                );
            }
            features = Some(encoded);
        }

        if let (Some(features), Some(decoder)) = (&features, sessions.get_mut(POINT_DECODER)) {
            for (set_index, set) in prompt_sets(entry)?.iter().enumerate() {
                let (coords, labels) = set.model_inputs(&model_input);
                let started = Instant::now();
                let output = run_point_decoder(decoder, features, &coords, &labels)?;
                let decode_ms = milliseconds(started.elapsed());
                if shape_check.is_none() {
                    let expected = &export_report["graphs"][POINT_DECODER]["outputs"];
                    let observed = &output.shapes;
                    let mask_side = low_res_mask_size as i64;
                    let matches = observed["mask_logits"] == [1, 4, mask_side, mask_side]
                        && observed["iou_scores"] == [1, 4]
                        && observed["object_score_logits"] == [1, 1];
                    shape_check = Some(json!({
                        "status": if matches { "passed" } else { "failed" },
                        "observed": observed,
                        "exported_specification": expected,
                        "required": { "mask_logits": [1, 4, mask_side, mask_side], "iou_scores": [1, 4], "object_score_logits": [1, 1] },
                    }));
                }
                if first_image && set_index == 0 {
                    report["graphs"][POINT_DECODER]["first_inference_ms"] = json!(decode_ms);
                    let samples = time_runs(options.decoder_runs, || {
                        run_point_decoder(decoder, features, &coords, &labels).map(|_| ())
                    })?;
                    report["graphs"][POINT_DECODER]["warm"] = latency_summary(&samples);
                    report["graphs"][POINT_DECODER]["warm_prompt"] = json!(set.name);
                    report["graphs"][POINT_DECODER]["memory_after_inference"] =
                        process_memory::sample();
                    println!(
                        "[point_decoder] first {decode_ms:.1} ms, warm {}",
                        report["graphs"][POINT_DECODER]["warm"]
                    );
                }

                let mut prompt_report = json!({
                    "kind": "point_box", "name": set.name, "decode_ms": decode_ms,
                    "iou_scores": output.iou_scores, "masks": [],
                });
                let mut overlay_mask: Option<(f32, GrayImage)> = None;
                for multimask in [true, false] {
                    for (index, (token, score)) in
                        select_masks(&output, multimask).into_iter().enumerate()
                    {
                        let started = Instant::now();
                        let mask = upsample_to_mask(
                            &token_logits(&output, token),
                            plane_side(output.mask_logits.len(), 4),
                            width,
                            height,
                        );
                        let upsample_ms = milliseconds(started.elapsed());
                        let label =
                            format!("{}{index}", if multimask { "multi" } else { "single" });
                        let comparison = compare_with_reference(
                            reference_dir,
                            &format!("{stem}__{}__{label}.png", set.name),
                            &mask,
                        );
                        if let Some(passed) = comparison.get("passed").and_then(Value::as_bool) {
                            reference_comparisons += 1;
                            reference_failures += (!passed) as usize;
                        }
                        // Overlay: the best multimask candidate for one point, else the single mask.
                        let use_for_overlay = if set.points.len() == 1 && set.boxes.is_empty() {
                            multimask
                        } else {
                            !multimask
                        };
                        if use_for_overlay && overlay_mask.as_ref().is_none_or(|(s, _)| score > *s)
                        {
                            overlay_mask = Some((score, mask.clone()));
                        }
                        prompt_report["masks"].as_array_mut().unwrap().push(json!({
                            "selection": label, "token": token, "score": score,
                            "mask_fraction": mask_fraction(&mask), "upsample_ms": upsample_ms,
                            "reference": comparison,
                        }));
                    }
                }
                if let (Some(dir), Some((_, mask))) = (&options.overlay_dir, &overlay_mask) {
                    write_overlay(
                        &photo,
                        mask,
                        &set.points,
                        &set.boxes,
                        options.overlay_max_edge,
                        &dir.join(format!("{stem}__{}.png", set.name)),
                    )?;
                }
                image_report["prompts"]
                    .as_array_mut()
                    .unwrap()
                    .push(prompt_report);
            }
        }

        let texts: Vec<String> = entry["text"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|t| t.as_str().map(str::to_string))
            .collect();
        if !texts.is_empty()
            && let Some(tokenizer) = &tokenizer
            && sessions.contains_key(TEXT_ENCODER)
            && sessions.contains_key(DETECTOR_GROUNDING)
        {
            for (text_index, text) in texts.iter().enumerate() {
                let tokens = tokenize(tokenizer, text)?;
                let expected = &export_report["text_tokens"][text.as_str()];
                tokenizer_check[text.as_str()] = if expected.is_null() {
                    json!({ "status": "unavailable", "reason": "export_report.json has no tokens for this text", "rust": tokens })
                } else {
                    let matches = expected
                        .as_array()
                        .map(|e| e.iter().map(|v| v.as_i64()).collect::<Vec<_>>())
                        == Some(tokens.iter().map(|&t| Some(t)).collect());
                    json!({ "status": if matches { "passed" } else { "failed" }, "rust": tokens, "python": expected })
                };

                let text_session = sessions.get_mut(TEXT_ENCODER).unwrap();
                let started = Instant::now();
                let text_features = run_text_encoder(text_session, &tokens)?;
                let text_ms = milliseconds(started.elapsed());
                if first_image && text_index == 0 {
                    report["graphs"][TEXT_ENCODER]["first_inference_ms"] = json!(text_ms);
                    let samples = time_runs(options.decoder_runs, || {
                        run_text_encoder(text_session, &tokens).map(|_| ())
                    })?;
                    report["graphs"][TEXT_ENCODER]["warm"] = latency_summary(&samples);
                }

                let detector = sessions.get_mut(DETECTOR_GROUNDING).unwrap();
                let started = Instant::now();
                let detections = run_detector(detector, &pixels, &text_features)?;
                let detect_ms = milliseconds(started.elapsed());
                if first_image && text_index == 0 {
                    report["graphs"][DETECTOR_GROUNDING]["first_inference_ms"] = json!(detect_ms);
                    let samples = time_runs(options.encoder_runs, || {
                        run_detector(detector, &pixels, &text_features).map(|_| ())
                    })?;
                    report["graphs"][DETECTOR_GROUNDING]["warm"] = latency_summary(&samples);
                    report["graphs"][DETECTOR_GROUNDING]["memory_after_inference"] =
                        process_memory::sample();
                    println!(
                        "[detector_grounding] first {detect_ms:.0} ms, warm {}",
                        report["graphs"][DETECTOR_GROUNDING]["warm"]
                    );
                }

                let plane = detections.mask_size * detections.mask_size;
                let mut kept: Vec<(usize, f32)> = detections
                    .scores
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|&(_, s)| s > TEXT_SCORE_THRESHOLD)
                    .collect();
                kept.sort_by(|a, b| b.1.total_cmp(&a.1));
                let mut union_mask = GrayImage::new(width, height);
                let mut masks = Vec::new();
                for (rank, &(query, score)) in kept.iter().enumerate() {
                    let logits = &detections.mask_logits[query * plane..(query + 1) * plane];
                    let mask = upsample_to_mask(logits, detections.mask_size, width, height);
                    for (u, m) in union_mask.pixels_mut().zip(mask.pixels()) {
                        u[0] = u[0].max(m[0]);
                    }
                    let comparison = compare_with_reference(
                        reference_dir,
                        &format!("{stem}__text_{}__{rank}.png", text.replace(' ', "_")),
                        &mask,
                    );
                    if let Some(passed) = comparison.get("passed").and_then(Value::as_bool) {
                        reference_comparisons += 1;
                        reference_failures += (!passed) as usize;
                    }
                    masks.push(json!({ "rank": rank, "score": score, "mask_fraction": mask_fraction(&mask), "reference": comparison }));
                }
                let best_scores: Vec<f32> = {
                    let mut s = detections.scores.clone();
                    s.sort_by(|a, b| b.total_cmp(a));
                    s.into_iter().take(5).collect()
                };
                println!(
                    "[text] {stem} {text:?}: {} detections > {TEXT_SCORE_THRESHOLD}, top scores {best_scores:?}",
                    kept.len()
                );
                if let Some(dir) = &options.overlay_dir {
                    write_overlay(
                        &photo,
                        &union_mask,
                        &[],
                        &[],
                        options.overlay_max_edge,
                        &dir.join(format!("{stem}__text_{}.png", text.replace(' ', "_"))),
                    )?;
                }
                image_report["prompts"].as_array_mut().unwrap().push(json!({
                    "kind": "text", "name": text, "tokens": tokens,
                    "text_encoder_ms": text_ms, "detector_ms": detect_ms,
                    "top_scores": best_scores, "detections": masks,
                }));
            }
        }

        report["images"].as_array_mut().unwrap().push(image_report);
        first_image = false;
    }

    report["checks"]["DecoderOutputShapeMatchesExportedSpecification"] = shape_check
        .unwrap_or_else(|| json!({ "status": "unavailable", "reason": "no point decoder run" }));
    report["checks"]["HostTokenizerMatchesPythonBpeTokens"] = if tokenizer.is_none() {
        json!({ "status": "unavailable", "reason": "no --tokenizer file" })
    } else {
        tokenizer_check
    };
    report["checks"]["RustPipelineMasksMatchPytorchReference"] = json!({
        "required_iou": REQUIRED_REFERENCE_IOU,
        "comparisons": reference_comparisons,
        "below_required_iou": reference_failures,
        "note": "Rust resize (image::imageops Triangle) and PyTorch resize (torchvision bilinear with antialias) are different implementations; see the per-mask IoU values",
    });
    report["memory_after_inference"] = process_memory::sample();

    // Write the measurements before the profiling pass, so that a slow profile run does not
    // hold back the other results.
    report["operator_placement"] = json!({ "status": "unavailable", "reason": "not profiled yet" });
    write_report(&options.out, &report)?;
    if let Some(profile_dir) = &options.profile_dir {
        report["operator_placement"] = profile_graphs(options, &model_input, profile_dir)?;
    } else {
        report["operator_placement"] =
            json!({ "status": "unavailable", "reason": "no --profile-dir" });
    }
    write_report(&options.out, &report)
}

fn write_report(path: &Path, report: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(report)?)?;
    println!("[report] {}", path.display());
    Ok(())
}

fn profile_graphs(
    options: &RuntimeOptions,
    model_input: &ModelInput,
    profile_dir: &Path,
) -> Result<Value> {
    let device_tag = options.device_text.replace(':', "_");
    let mut placement = json!({});
    let pixels = vec![0.0f32; 3 * model_input.image_size * model_input.image_size];
    let path = |name: &str| options.onnx_dir.join(format!("{name}.onnx"));
    let prefix = |name: &str| profile_dir.join(format!("{name}_{device_tag}"));
    let wanted = |name: &str| options.profile_graphs.iter().any(|n| n == name);
    for name in GRAPH_NAMES {
        if !wanted(name) {
            placement[name] =
                json!({ "status": "unavailable", "reason": "not in --profile-graphs" });
        }
    }

    let mut features = None;
    if path(POINT_ENCODER).exists() && !wanted(POINT_ENCODER) && wanted(POINT_DECODER) {
        let mut session = options
            .settings
            .load(&path(POINT_ENCODER), options.device)?;
        features = Some(run_point_encoder(&mut session, &pixels)?);
    }
    if path(POINT_ENCODER).exists() && wanted(POINT_ENCODER) {
        placement[POINT_ENCODER] = operator_placement(
            &path(POINT_ENCODER),
            options.device,
            options.settings,
            &prefix(POINT_ENCODER),
            |s| {
                run_point_encoder(s, &pixels)?;
                features = Some(run_point_encoder(s, &pixels)?);
                Ok(())
            },
        )
        .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") }));
    }
    if let Some(features) = &features
        && path(POINT_DECODER).exists()
        && wanted(POINT_DECODER)
    {
        let coords = [[500.0f32, 500.0]];
        let labels = [1.0f32];
        placement[POINT_DECODER] = operator_placement(
            &path(POINT_DECODER),
            options.device,
            options.settings,
            &prefix(POINT_DECODER),
            |s| {
                run_point_decoder(s, features, &coords, &labels)?;
                run_point_decoder(s, features, &coords, &labels).map(|_| ())
            },
        )
        .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") }));
    }
    let mut tokens = vec![0i64; TEXT_CONTEXT_LENGTH];
    tokens[..3].copy_from_slice(&[CLIP_START_TOKEN as i64, 1929, CLIP_END_TOKEN as i64]);
    let mut text_features = None;
    if path(TEXT_ENCODER).exists() && !wanted(TEXT_ENCODER) && wanted(DETECTOR_GROUNDING) {
        let mut session = options.settings.load(&path(TEXT_ENCODER), options.device)?;
        text_features = Some(run_text_encoder(&mut session, &tokens)?);
    }
    if path(TEXT_ENCODER).exists() && wanted(TEXT_ENCODER) {
        placement[TEXT_ENCODER] = operator_placement(
            &path(TEXT_ENCODER),
            options.device,
            options.settings,
            &prefix(TEXT_ENCODER),
            |s| {
                run_text_encoder(s, &tokens)?;
                text_features = Some(run_text_encoder(s, &tokens)?);
                Ok(())
            },
        )
        .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") }));
    }
    if let Some(text_features) = &text_features
        && path(DETECTOR_GROUNDING).exists()
        && wanted(DETECTOR_GROUNDING)
    {
        placement[DETECTOR_GROUNDING] = operator_placement(
            &path(DETECTOR_GROUNDING),
            options.device,
            options.settings,
            &prefix(DETECTOR_GROUNDING),
            |s| {
                run_detector(s, &pixels, text_features)?;
                run_detector(s, &pixels, text_features).map(|_| ())
            },
        )
        .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") }));
    }
    Ok(placement)
}

// ---------------------------------------------------------------------------------------------
// Matte report (#458, step E2)

const SUBJECT_MATTE: &str = "subject_matte";

struct MatteOptions {
    onnx_dir: PathBuf,
    settings: SessionSettings,
    device_text: String,
    device: DeviceRequest,
    photos: PathBuf,
    prompts: PathBuf,
    out: PathBuf,
    reference_alpha: Option<PathBuf>,
    overlay_dir: Option<PathBuf>,
    /// Long edge of the overlay and alpha images; 0 keeps the photo size.
    overlay_max_edge: u32,
    profile_dir: Option<PathBuf>,
    runs: usize,
}

impl MatteOptions {
    fn from_map(options: &BTreeMap<String, String>) -> Result<Self> {
        let device_text = required(options, "device")?;
        Ok(Self {
            onnx_dir: required(options, "onnx-dir")?.into(),
            settings: SessionSettings::parse(
                options
                    .get("session-settings")
                    .map_or("production", String::as_str),
            )?,
            device: ort_runtime::parse_device_request(&device_text)?,
            device_text,
            photos: required(options, "photos")?.into(),
            prompts: required(options, "prompts")?.into(),
            out: required(options, "out")?.into(),
            reference_alpha: options.get("reference-alpha").map(PathBuf::from),
            overlay_dir: options.get("overlay-dir").map(PathBuf::from),
            overlay_max_edge: parse_count(options, "overlay-max-edge", 0)? as u32,
            profile_dir: options.get("profile-dir").map(PathBuf::from),
            runs: parse_count(options, "runs", 20)?,
        })
    }
}

/// Runs the matte graph; returns the output shape and the alpha values.
fn run_subject_matte(session: &mut Session, pixels: &[f32]) -> Result<(Vec<i64>, Vec<f32>)> {
    let side = pixel_side(pixels);
    let input = Tensor::from_array(([1usize, 3, side, side], pixels.to_vec()))
        .map_err(|e| anyhow!("failed to build pixel tensor: {e}"))?;
    let outputs = session
        .run(ort::inputs!["pixel_values" => input])
        .map_err(|e| anyhow!("subject_matte failed: {e}"))?;
    output_f32(&outputs, "alpha")
}

/// Bilinear resize of a square alpha with `align_corners=False` to the photo size, 8 bit.
fn upsample_alpha(alpha: &[f32], size: usize, width: u32, height: u32) -> GrayImage {
    let scale_x = size as f32 / width as f32;
    let scale_y = size as f32 / height as f32;
    let source = |x: u32, scale: f32| -> (usize, usize, f32) {
        let position = ((x as f32 + 0.5) * scale - 0.5).max(0.0);
        let low = (position.floor() as usize).min(size - 1);
        let high = (low + 1).min(size - 1);
        (low, high, position - low as f32)
    };
    let columns: Vec<_> = (0..width).map(|x| source(x, scale_x)).collect();
    let mut matte = GrayImage::new(width, height);
    for y in 0..height {
        let (y0, y1, fy) = source(y, scale_y);
        for (x, &(x0, x1, fx)) in columns.iter().enumerate() {
            let top = alpha[y0 * size + x0] * (1.0 - fx) + alpha[y0 * size + x1] * fx;
            let bottom = alpha[y1 * size + x0] * (1.0 - fx) + alpha[y1 * size + x1] * fx;
            let value = (top * (1.0 - fy) + bottom * fy).clamp(0.0, 1.0);
            matte.put_pixel(x as u32, y, Luma([(value * 255.0).round() as u8]));
        }
    }
    matte
}

/// IoU of `alpha > 0.5` and the mean absolute alpha difference against the PyTorch alpha.
fn compare_alpha_with_reference(
    reference_dir: Option<&Path>,
    stem: &str,
    matte: &GrayImage,
) -> Value {
    let Some(directory) = reference_dir else {
        return json!({ "status": "unavailable", "reason": "no --reference-alpha directory" });
    };
    let path = directory.join(format!("{stem}.png"));
    let reference = match image::open(&path) {
        Ok(image) => image.to_luma8(),
        Err(error) => {
            return json!({ "status": "unavailable", "reason": format!("{}: {error}", path.display()) });
        }
    };
    if reference.dimensions() != matte.dimensions() {
        return json!({ "status": "failed", "reason": "reference size differs from the photo size" });
    }
    let iou = mask_iou(matte, &reference).unwrap_or(0.0);
    let total: u64 = matte
        .pixels()
        .zip(reference.pixels())
        .map(|(a, b)| a[0].abs_diff(b[0]) as u64)
        .sum();
    json!({
        "iou": iou,
        "mean_abs_alpha_diff": total as f64 / 255.0 / (matte.width() as f64 * matte.height() as f64),
        "passed": iou >= REQUIRED_REFERENCE_IOU,
    })
}

/// The photo with the soft matte blended in the overlay color, and the alpha image, at the same
/// size (the long edge limited to `max_edge`).
fn write_matte_images(
    photo: &RgbImage,
    matte: &GrayImage,
    max_edge: u32,
    overlay_path: &Path,
    alpha_path: &Path,
) -> Result<()> {
    let (photo_width, photo_height) = photo.dimensions();
    let long_edge = photo_width.max(photo_height);
    let (mut overlay, matte) = if max_edge > 0 && long_edge > max_edge {
        let scale = max_edge as f64 / long_edge as f64;
        let width = ((photo_width as f64 * scale).round() as u32).max(1);
        let height = ((photo_height as f64 * scale).round() as u32).max(1);
        (
            image::imageops::resize(photo, width, height, FilterType::Triangle),
            image::imageops::resize(matte, width, height, FilterType::Triangle),
        )
    } else {
        (photo.clone(), matte.clone())
    };
    for (pixel, alpha) in overlay.pixels_mut().zip(matte.pixels()) {
        let weight = 0.55 * alpha[0] as f32 / 255.0;
        for channel in 0..3 {
            pixel[channel] =
                (pixel[channel] as f32 * (1.0 - weight) + OVERLAY_COLOR[channel] * weight) as u8;
        }
    }
    if let Some(parent) = overlay_path.parent() {
        fs::create_dir_all(parent)?;
    }
    DynamicImage::ImageRgb8(overlay)
        .save(overlay_path)
        .with_context(|| format!("failed to write {}", overlay_path.display()))?;
    DynamicImage::ImageLuma8(matte)
        .save(alpha_path)
        .with_context(|| format!("failed to write {}", alpha_path.display()))
}

fn run_matte_report(options: &MatteOptions) -> Result<()> {
    ort_runtime::initialize_ort_environment()?;
    let export_report: Value = serde_json::from_str(
        &fs::read_to_string(options.onnx_dir.join("export_report.json"))
            .context("failed to read export_report.json")?,
    )?;
    let model_input = ModelInput::from_export_report(&export_report)?;
    let size = model_input.image_size;
    let prompts: Value = serde_json::from_str(&fs::read_to_string(&options.prompts)?)?;
    let mut report = json!({
        "device_request": options.device_text,
        "session_settings": options.settings.label(),
        "model_input": {
            "image_size": size,
            "pixel_mean": model_input.mean,
            "pixel_std": model_input.std,
        },
        "variant": export_report["variant"],
        "device_description": ort_runtime::describe_device_request(options.device),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "memory_before_load": process_memory::sample(),
        "checks": {},
        "graphs": {},
        "images": [],
    });

    let path = options.onnx_dir.join(format!("{SUBJECT_MATTE}.onnx"));
    let load_check_name = if options.settings == SessionSettings::Production {
        "SegmentationGraphsLoadWithProductionSessionSettings"
    } else {
        "SegmentationGraphsLoadWithNonProductionSessionSettings"
    };
    let started = Instant::now();
    let mut session = match options.settings.load(&path, options.device) {
        Ok(session) => session,
        Err(error) => {
            println!("[load] {SUBJECT_MATTE}: FAILED {error}");
            report["checks"][load_check_name] =
                json!({ SUBJECT_MATTE: { "status": "failed", "error": format!("{error:#}") } });
            return write_report(&options.out, &report);
        }
    };
    let load_ms = milliseconds(started.elapsed());
    println!("[load] {SUBJECT_MATTE}: {load_ms:.0} ms");
    report["checks"][load_check_name] = json!({ SUBJECT_MATTE: { "status": "passed" } });
    report["graphs"][SUBJECT_MATTE] = json!({
        "load_ms": load_ms,
        "memory_after_load": process_memory::sample(),
    });

    let mut shape_check: Option<Value> = None;
    let mut reference_failures = 0usize;
    let mut reference_comparisons = 0usize;
    let mut first_image = true;
    for entry in prompts["images"].as_array().into_iter().flatten() {
        let file = entry["file"]
            .as_str()
            .ok_or_else(|| anyhow!("image entry without file"))?;
        let stem = Path::new(file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(file)
            .to_string();
        let started = Instant::now();
        let photo = image::open(options.photos.join(file))
            .with_context(|| format!("failed to open {file}"))?
            .to_rgb8();
        let decode_ms = milliseconds(started.elapsed());
        let started = Instant::now();
        let pixels = preprocess(&photo, &model_input);
        let preprocess_ms = milliseconds(started.elapsed());
        let (width, height) = photo.dimensions();

        let started = Instant::now();
        let (shape, alpha) = run_subject_matte(&mut session, &pixels)?;
        let inference_ms = milliseconds(started.elapsed());
        if first_image {
            report["graphs"][SUBJECT_MATTE]["first_inference_ms"] = json!(inference_ms);
            let samples = time_runs(options.runs, || {
                run_subject_matte(&mut session, &pixels).map(|_| ())
            })?;
            report["graphs"][SUBJECT_MATTE]["warm"] = latency_summary(&samples);
            report["graphs"][SUBJECT_MATTE]["memory_after_inference"] = process_memory::sample();
            let expected = [1i64, 1, size as i64, size as i64];
            shape_check = Some(json!({
                "alpha": shape,
                "expected": expected,
                "passed": shape == expected,
            }));
        }
        let started = Instant::now();
        let matte = upsample_alpha(&alpha, size, width, height);
        let upsample_ms = milliseconds(started.elapsed());
        let comparison =
            compare_alpha_with_reference(options.reference_alpha.as_deref(), &stem, &matte);
        if let Some(passed) = comparison["passed"].as_bool() {
            reference_comparisons += 1;
            reference_failures += (!passed) as usize;
        }
        let soft = alpha.iter().filter(|&&a| a > 0.05 && a < 0.95).count();
        let foreground = alpha.iter().filter(|&&a| a > 0.5).count();
        println!(
            "[matte] {file} {width}x{height}: decode {decode_ms:.0} ms, preprocess {preprocess_ms:.0} ms, \
             inference {inference_ms:.1} ms, upsample {upsample_ms:.0} ms, reference {comparison}"
        );
        if let Some(dir) = &options.overlay_dir {
            write_matte_images(
                &photo,
                &matte,
                options.overlay_max_edge,
                &dir.join(format!("{stem}__subject.png")),
                &dir.join(format!("{stem}__alpha.png")),
            )?;
        }
        report["images"].as_array_mut().unwrap().push(json!({
            "file": file, "width": width, "height": height,
            "decode_ms": decode_ms, "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms, "upsample_ms": upsample_ms,
            "foreground_fraction": foreground as f64 / alpha.len() as f64,
            "soft_fraction": soft as f64 / alpha.len() as f64,
            "reference": comparison,
        }));
        first_image = false;
    }
    let per_photo: Vec<f64> = report["images"]
        .as_array()
        .into_iter()
        .flatten()
        .skip(1)
        .filter_map(|i| i["inference_ms"].as_f64())
        .collect();
    report["graphs"][SUBJECT_MATTE]["per_photo_after_first"] = latency_summary(&per_photo);
    report["checks"]["MatteOutputShapeMatchesExportedSpecification"] =
        shape_check.unwrap_or_else(|| json!({ "status": "unavailable", "reason": "no photo" }));
    report["checks"]["RustPipelineMatteMatchesPytorchReference"] = json!({
        "required_iou": REQUIRED_REFERENCE_IOU,
        "comparisons": reference_comparisons,
        "below_required_iou": reference_failures,
        "note": "Rust resize (image::imageops Triangle) and PIL bilinear resize are different implementations",
    });
    report["memory_after_inference"] = process_memory::sample();
    drop(session);

    report["operator_placement"] = json!({ "status": "unavailable", "reason": "not profiled yet" });
    write_report(&options.out, &report)?;
    if let Some(profile_dir) = &options.profile_dir {
        let pixels = vec![0.0f32; 3 * size * size];
        let prefix = profile_dir.join(format!(
            "{SUBJECT_MATTE}_{}",
            options.device_text.replace(':', "_")
        ));
        report["operator_placement"] = json!({
            SUBJECT_MATTE: operator_placement(&path, options.device, options.settings, &prefix, |s| {
                run_subject_matte(s, &pixels)?;
                run_subject_matte(s, &pixels).map(|_| ())
            })
            .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") })),
        });
    } else {
        report["operator_placement"] =
            json!({ "status": "unavailable", "reason": "no --profile-dir" });
    }
    write_report(&options.out, &report)
}

// ---------------------------------------------------------------------------------------------
// Ping report

fn run_ping_report(options: &PingOptions) -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("failed to create tokio runtime")?;
    let report = runtime.block_on(async {
        let endpoint = options.endpoint.clone();
        let mut persistent = HealthServiceClient::connect(endpoint.clone())
            .await
            .with_context(|| format!("failed to connect to {endpoint}"))?;
        for _ in 0..10 {
            persistent.ping(PingRequest {}).await?;
        }
        let mut persistent_samples = Vec::with_capacity(options.calls);
        for _ in 0..options.calls {
            let started = Instant::now();
            persistent.ping(PingRequest {}).await?;
            persistent_samples.push(milliseconds(started.elapsed()));
        }
        let mut per_call_samples = Vec::with_capacity(options.calls);
        for _ in 0..options.calls {
            let started = Instant::now();
            let mut client = HealthServiceClient::connect(endpoint.clone()).await?;
            client.ping(PingRequest {}).await?;
            per_call_samples.push(milliseconds(started.elapsed()));
        }
        anyhow::Ok(json!({
            "check": "PingLatencyIsReportedForPersistentAndPerCallChannels",
            "client": "Rust tonic HealthServiceClient (not the C++ sidecar client)",
            "endpoint": endpoint,
            "persistent_channel": latency_summary(&persistent_samples),
            "new_channel_per_call": latency_summary(&per_call_samples),
        }))
    })?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    if let Some(parent) = options.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&options.out, serde_json::to_string_pretty(&report)?)?;
    Ok(())
}

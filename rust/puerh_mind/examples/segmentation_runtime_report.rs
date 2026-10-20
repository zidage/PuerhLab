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
//! `candidates` mode (#458, step E4) loads `point_encoder.onnx`, `point_decoder.onnx`, and
//! `candidate_decoder.onnx` of `scripts/ai_models/sam2/export_onnx.py`. It computes the click
//! candidates of each photo (the point grid of `SAM2AutomaticMaskGenerator` in batches, the
//! predicted IoU and stability filters, and box non-maximum suppression), measures the time to the
//! candidates, and evaluates each single-point prompt two ways: the live point decoder (one
//! decoder run per pointer position) and the candidate levels under the pointer (the candidates
//! that contain the point, from small to large). It compares each selection with the E1 reference
//! mask of the intended target, and it measures whether the highlight stays the same when the
//! pointer moves a short distance (`CandidateDecoderOutputShapeMatchesExportedSpecification`).
//!
//! `strokes` mode (#458, step E3) loads `point_encoder.onnx` and `point_decoder.onnx`. It places
//! straight strokes inside the E1 reference mask of each single-point prompt (through the prompt
//! point, at a margin from the mask edge), samples points along each stroke as positive points of
//! the point decoder, and compares the selected mask with the reference and with one click. It
//! sweeps the point count, the point spacing, the stroke length, and the stroke direction. It then
//! simulates corrective add and subtract strokes (and corrective clicks for comparison) in the
//! largest error region, with three ways to combine the strokes, and it measures the decoder time
//! per stroke for 1 to 32 points with and without the previous mask as `mask_input`.
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
        "candidates" => run_candidate_report(&CandidateOptions::from_map(&options)?),
        "strokes" => run_stroke_report(&StrokeOptions::from_map(&options)?),
        "ping" => run_ping_report(&PingOptions::from_map(&options)?),
        _ => bail!(
            "usage: segmentation_runtime_report runtime --onnx-dir DIR --device DEVICE --photos DIR \
             --prompts FILE --out FILE [--reference-masks DIR] [--overlay-dir DIR] [--overlay-max-edge N] [--tokenizer FILE] \
             [--encoder-runs N] [--decoder-runs N] [--profile-dir DIR] \n             [--session-settings production|coreml-mlprogram] [--profile-graphs NAME,NAME] [--graph-devices NAME=DEVICE,...]\n\
             segmentation_runtime_report matte --onnx-dir DIR --device DEVICE --photos DIR --prompts FILE --out FILE \
             [--reference-alpha DIR] [--overlay-dir DIR] [--overlay-max-edge N] [--runs N] [--profile-dir DIR]\n\
             segmentation_runtime_report candidates --onnx-dir DIR --device DEVICE --photos DIR --prompts FILE --out FILE \
             [--targets FILE] [--reference-masks DIR] [--reference-scores FILE] [--overlay-dir DIR] [--overlay-points-per-side N] \
             [--mask-dir DIR] [--points-per-side N,N] [--batch N] [--batch-sizes N,N] [--runs N] [--profile-dir DIR] \
             [--score-threshold V] [--stability-threshold V] [--nms-threshold V] [--dedupe box|mask]\n\
             segmentation_runtime_report strokes --onnx-dir DIR --device DEVICE --photos DIR --prompts FILE --out FILE \
             --targets FILE --reference-masks DIR --reference-scores FILE [--mask-dir DIR] [--margin V] \
             [--lengths V,V] [--counts N,N] [--spacings V,V] [--primary-length V] [--primary-count N] \
             [--corrective-steps N] [--stop-iou V] [--timing-counts N,N] [--runs N] [--profile-dir DIR] \
             [--profile-counts N,N] [--pad-points N] [--pad-mode label|repeat] \
             [--decoder-sessions one|per-point-count]\n\
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
    run_point_decoder_with_mask(session, features, coords, labels, None)
}

/// `mask_input` is the low-resolution logits of a previous selection (the `mask_input` of
/// `SAM2ImagePredictor.predict`); `None` sets `has_mask_input` to 0.
fn run_point_decoder_with_mask(
    session: &mut Session,
    features: &PointFeatures,
    coords: &[[f32; 2]],
    labels: &[f32],
    mask_input: Option<&[f32]>,
) -> Result<DecoderOutput> {
    let count = coords.len();
    // The low-resolution mask has the size of the stride-4 features (`high_res_feats_0`).
    let mask_size = features.shapes[1].last().copied().unwrap_or(0) as usize;
    let (mask_values, has_mask) = match mask_input {
        Some(logits) if logits.len() == mask_size * mask_size => (logits.to_vec(), 1.0f32),
        Some(logits) => bail!(
            "mask input has {} values; the decoder needs {mask_size} x {mask_size}",
            logits.len()
        ),
        None => (vec![0.0f32; mask_size * mask_size], 0.0f32),
    };
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
            "mask_input" => Tensor::from_array(([1usize, 1, mask_size, mask_size], mask_values))?,
            "has_mask_input" => Tensor::from_array(([1usize], vec![has_mask]))?,
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

// ---------------------------------------------------------------------------------------------
// Candidate report (#458, step E4)

const CANDIDATE_DECODER: &str = "candidate_decoder";
/// The multimask tokens 1-3 of the mask decoder; `candidate_decoder.onnx` returns only these.
const CANDIDATE_TOKENS: usize = 3;
/// IoU against the reference above which a selection counts as the intended mask.
const TARGET_IOU: f64 = 0.85;
/// IoU against the highlight at the pointer centre above which a nearby highlight is the same.
const SAME_HIGHLIGHT_IOU: f64 = 0.9;

struct CandidateOptions {
    onnx_dir: PathBuf,
    device_text: String,
    device: DeviceRequest,
    photos: PathBuf,
    prompts: PathBuf,
    targets: Option<PathBuf>,
    reference_masks: Option<PathBuf>,
    reference_scores: Option<PathBuf>,
    out: PathBuf,
    overlay_dir: Option<PathBuf>,
    /// The grid whose candidates the overlays show.
    overlay_points_per_side: usize,
    mask_dir: Option<PathBuf>,
    /// Long edge of the evaluation masks and the overlays.
    evaluation_max_edge: u32,
    points_per_side: Vec<usize>,
    batch: usize,
    /// Batch sizes of the timing comparison on the first photo.
    batch_sizes: Vec<usize>,
    runs: usize,
    score_threshold: f32,
    stability_threshold: f32,
    nms_threshold: f32,
    /// Overlap measure of the duplicate removal: `box` is the box non-maximum suppression of the
    /// upstream generator; `mask` compares the masks, so nested masks with similar boxes (a person
    /// and the jacket) both stay.
    dedupe_by_mask: bool,
    level_merge_iou: f64,
    hover_radius: f32,
    hover_points: usize,
    profile_dir: Option<PathBuf>,
}

fn parse_list<T: std::str::FromStr>(
    options: &BTreeMap<String, String>,
    key: &str,
    default: &str,
) -> Result<Vec<T>> {
    options
        .get(key)
        .map_or(default, String::as_str)
        .split(',')
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            v.trim()
                .parse()
                .map_err(|_| anyhow!("failed to parse --{key} value {v:?}"))
        })
        .collect()
}

fn parse_value<T: std::str::FromStr>(
    options: &BTreeMap<String, String>,
    key: &str,
    default: T,
) -> Result<T> {
    match options.get(key) {
        Some(value) => value
            .parse()
            .map_err(|_| anyhow!("failed to parse --{key} {value:?}")),
        None => Ok(default),
    }
}

impl CandidateOptions {
    fn from_map(options: &BTreeMap<String, String>) -> Result<Self> {
        let device_text = required(options, "device")?;
        let points_per_side: Vec<usize> = parse_list(options, "points-per-side", "16,24,32")?;
        Ok(Self {
            onnx_dir: required(options, "onnx-dir")?.into(),
            device: ort_runtime::parse_device_request(&device_text)?,
            device_text,
            photos: required(options, "photos")?.into(),
            prompts: required(options, "prompts")?.into(),
            targets: options.get("targets").map(PathBuf::from),
            reference_masks: options.get("reference-masks").map(PathBuf::from),
            reference_scores: options.get("reference-scores").map(PathBuf::from),
            out: required(options, "out")?.into(),
            overlay_dir: options.get("overlay-dir").map(PathBuf::from),
            overlay_points_per_side: parse_value(
                options,
                "overlay-points-per-side",
                *points_per_side.last().unwrap_or(&32),
            )?,
            mask_dir: options.get("mask-dir").map(PathBuf::from),
            evaluation_max_edge: parse_value(options, "evaluation-max-edge", 1024)?,
            points_per_side,
            batch: parse_value(options, "batch", 64)?,
            batch_sizes: parse_list(options, "batch-sizes", "32,64,128")?,
            runs: parse_count(options, "runs", 5)?,
            // The defaults of `SAM2AutomaticMaskGenerator`.
            score_threshold: parse_value(options, "score-threshold", 0.8)?,
            stability_threshold: parse_value(options, "stability-threshold", 0.95)?,
            nms_threshold: parse_value(options, "nms-threshold", 0.7)?,
            dedupe_by_mask: match options.get("dedupe").map_or("box", String::as_str) {
                "box" => false,
                "mask" => true,
                other => bail!("unsupported --dedupe {other:?}; use box or mask"),
            },
            level_merge_iou: parse_value(options, "level-merge-iou", 0.85)?,
            hover_radius: parse_value(options, "hover-radius", 0.015)?,
            hover_points: parse_value(options, "hover-points", 8)?,
            profile_dir: options.get("profile-dir").map(PathBuf::from),
        })
    }
}

/// A thresholded (logit > 0) mask on the square low-resolution model grid, as a bit set.
#[derive(Clone)]
struct GridMask {
    side: usize,
    bits: Vec<u64>,
    area: u32,
}

impl GridMask {
    fn from_logits(logits: &[f32], side: usize) -> Self {
        let mut bits = vec![0u64; (side * side).div_ceil(64)];
        let mut area = 0;
        for (index, &value) in logits.iter().enumerate() {
            if value > 0.0 {
                bits[index / 64] |= 1 << (index % 64);
                area += 1;
            }
        }
        Self { side, bits, area }
    }

    /// `x` and `y` are normalized image coordinates.
    fn contains(&self, x: f32, y: f32) -> bool {
        let cell = |v: f32| ((v * self.side as f32).max(0.0) as usize).min(self.side - 1);
        let index = cell(y) * self.side + cell(x);
        (self.bits[index / 64] >> (index % 64)) & 1 == 1
    }

    fn fraction(&self) -> f64 {
        self.area as f64 / (self.side * self.side) as f64
    }

    fn iou(&self, other: &Self) -> f64 {
        let (mut intersection, mut union) = (0u32, 0u32);
        for (a, b) in self.bits.iter().zip(&other.bits) {
            intersection += (a & b).count_ones();
            union += (a | b).count_ones();
        }
        if union == 0 {
            1.0
        } else {
            intersection as f64 / union as f64
        }
    }

    /// `batched_mask_to_box`: [x0, y0, x1, y1] with inclusive maximum cells.
    fn bounding_box(&self) -> [f32; 4] {
        let (mut x0, mut y0, mut x1, mut y1) = (self.side, self.side, 0, 0);
        for (word_index, &word) in self.bits.iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let index = word_index * 64 + word.trailing_zeros() as usize;
                let (x, y) = (index % self.side, index / self.side);
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                word &= word - 1;
            }
        }
        [x0 as f32, y0 as f32, x1 as f32, y1 as f32]
    }
}

/// `torchvision.ops.box_iou` on [x0, y0, x1, y1] boxes. The value does not change when the
/// photo is stretched to the square model grid.
fn box_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let area = |r: &[f32; 4]| (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0);
    let width = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let height = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let intersection = width * height;
    let union = area(a) + area(b) - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

struct Candidate {
    mask: GridMask,
    logits: Vec<f32>,
    score: f32,
    stability: f32,
    /// Normalized image coordinates of the grid point that made the candidate.
    point: [f32; 2],
    token: usize,
    bounding_box: [f32; 4],
}

struct CandidateDecoderOutput {
    mask_logits: Vec<f32>,
    iou_scores: Vec<f32>,
    stability_scores: Vec<f32>,
    shapes: BTreeMap<String, Vec<i64>>,
}

/// `coords` are model-input pixels, one positive point per prompt.
fn run_candidate_decoder(
    session: &mut Session,
    features: &PointFeatures,
    coords: &[[f32; 2]],
) -> Result<CandidateDecoderOutput> {
    let tensor = |shape: &[i64], data: &[f32]| {
        Tensor::from_array((shape_usize(shape), data.to_vec()))
            .map_err(|e| anyhow!("failed to build candidate decoder input: {e}"))
    };
    let outputs = session
        .run(ort::inputs! {
            "image_embed" => tensor(&features.shapes[0], &features.image_embed)?,
            "high_res_feats_0" => tensor(&features.shapes[1], &features.high_res_feats_0)?,
            "high_res_feats_1" => tensor(&features.shapes[2], &features.high_res_feats_1)?,
            "point_coords" => Tensor::from_array(([coords.len(), 1usize, 2], coords.concat()))?,
        })
        .map_err(|e| anyhow!("candidate decoder run failed: {e}"))?;
    let mut shapes = BTreeMap::new();
    let (mask_shape, mask_logits) = output_f32(&outputs, "mask_logits")?;
    let (iou_shape, iou_scores) = output_f32(&outputs, "iou_scores")?;
    let (stability_shape, stability_scores) = output_f32(&outputs, "stability_scores")?;
    shapes.insert("mask_logits".to_string(), mask_shape);
    shapes.insert("iou_scores".to_string(), iou_shape);
    shapes.insert("stability_scores".to_string(), stability_shape);
    Ok(CandidateDecoderOutput {
        mask_logits,
        iou_scores,
        stability_scores,
        shapes,
    })
}

/// `build_point_grid`: normalized cell centres, row by row.
fn point_grid(points_per_side: usize) -> Vec<[f32; 2]> {
    let step = 1.0 / points_per_side as f32;
    (0..points_per_side * points_per_side)
        .map(|i| {
            [
                ((i % points_per_side) as f32 + 0.5) * step,
                ((i / points_per_side) as f32 + 0.5) * step,
            ]
        })
        .collect()
}

struct CandidateSet {
    candidates: Vec<Candidate>,
    masks_decoded: usize,
    masks_after_filter: usize,
    decoder_ms: f64,
    filter_ms: f64,
    nms_ms: f64,
    shapes: BTreeMap<String, Vec<i64>>,
}

/// `SAM2AutomaticMaskGenerator._process_crop` without crops: the point grid in batches, the
/// predicted IoU and stability filters, and box non-maximum suppression. The filters use the
/// 256 x 256 logits, not the masks at the photo size.
fn generate_candidates(
    session: &mut Session,
    features: &PointFeatures,
    model_input: &ModelInput,
    points_per_side: usize,
    batch: usize,
    options: &CandidateOptions,
) -> Result<CandidateSet> {
    let side = model_input.low_res_mask_size;
    let plane = side * side;
    let size = model_input.image_size as f32;
    let grid = point_grid(points_per_side);
    let (mut decoder_ms, mut filter_ms) = (0.0, 0.0);
    let mut kept = Vec::new();
    let mut shapes = BTreeMap::new();
    for chunk in grid.chunks(batch) {
        let coords: Vec<[f32; 2]> = chunk.iter().map(|p| [p[0] * size, p[1] * size]).collect();
        let started = Instant::now();
        let output = run_candidate_decoder(session, features, &coords)?;
        decoder_ms += milliseconds(started.elapsed());
        let started = Instant::now();
        for (prompt, point) in chunk.iter().enumerate() {
            for token in 0..CANDIDATE_TOKENS {
                let index = prompt * CANDIDATE_TOKENS + token;
                let (score, stability) = (output.iou_scores[index], output.stability_scores[index]);
                if score <= options.score_threshold || stability < options.stability_threshold {
                    continue;
                }
                let logits = output.mask_logits[index * plane..(index + 1) * plane].to_vec();
                let mask = GridMask::from_logits(&logits, side);
                if mask.area == 0 {
                    continue;
                }
                kept.push(Candidate {
                    bounding_box: mask.bounding_box(),
                    mask,
                    logits,
                    score,
                    stability,
                    point: *point,
                    token: token + 1,
                });
            }
        }
        filter_ms += milliseconds(started.elapsed());
        shapes = output.shapes;
    }
    let masks_after_filter = kept.len();
    let started = Instant::now();
    kept.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut candidates: Vec<Candidate> = Vec::new();
    for candidate in kept {
        let distinct = |c: &Candidate| {
            if options.dedupe_by_mask {
                c.mask.iou(&candidate.mask) <= options.nms_threshold as f64
            } else {
                box_iou(&c.bounding_box, &candidate.bounding_box) <= options.nms_threshold
            }
        };
        if candidates.iter().all(distinct) {
            candidates.push(candidate);
        }
    }
    let nms_ms = milliseconds(started.elapsed());
    Ok(CandidateSet {
        candidates,
        masks_decoded: grid.len() * CANDIDATE_TOKENS,
        masks_after_filter,
        decoder_ms,
        filter_ms,
        nms_ms,
        shapes,
    })
}

/// The selection levels under the pointer: the candidates that contain the point, from the
/// smallest to the largest. A candidate with IoU >= `merge_iou` against the previous level is
/// the same level; the level keeps the candidate with the higher predicted IoU.
fn levels_at(candidates: &[Candidate], x: f32, y: f32, merge_iou: f64) -> Vec<usize> {
    let mut containing: Vec<usize> = (0..candidates.len())
        .filter(|&i| candidates[i].mask.contains(x, y))
        .collect();
    containing.sort_by_key(|&i| candidates[i].mask.area);
    let mut levels: Vec<usize> = Vec::new();
    for index in containing {
        if let Some(last) = levels.last_mut()
            && candidates[index].mask.iou(&candidates[*last].mask) >= merge_iou
        {
            if candidates[index].score > candidates[*last].score {
                *last = index;
            }
            continue;
        }
        levels.push(index);
    }
    levels
}

/// The default level under the pointer, before the user changes the level.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DefaultLevel {
    /// The largest level.
    Largest,
    /// The level with the highest predicted IoU.
    BestScore,
}

impl DefaultLevel {
    const ALL: [Self; 2] = [Self::Largest, Self::BestScore];

    fn name(self) -> &'static str {
        match self {
            Self::Largest => "largest",
            Self::BestScore => "best_score",
        }
    }

    /// Index into `levels`; `levels` is sorted by area.
    fn pick(self, levels: &[usize], scores: impl Fn(usize) -> f32) -> Option<usize> {
        if levels.is_empty() {
            return None;
        }
        Some(match self {
            Self::Largest => levels.len() - 1,
            Self::BestScore => (0..levels.len())
                .max_by(|&a, &b| scores(levels[a]).total_cmp(&scores(levels[b])))
                .unwrap_or(0),
        })
    }
}

struct EvaluationFrame {
    width: u32,
    height: u32,
    photo: RgbImage,
}

impl EvaluationFrame {
    fn new(photo: &RgbImage, max_edge: u32) -> Self {
        let (w, h) = photo.dimensions();
        let scale = (max_edge as f64 / w.max(h) as f64).min(1.0);
        let width = ((w as f64 * scale).round() as u32).max(1);
        let height = ((h as f64 * scale).round() as u32).max(1);
        Self {
            width,
            height,
            photo: image::imageops::resize(photo, width, height, FilterType::Triangle),
        }
    }

    fn mask(&self, logits: &[f32], side: usize) -> GrayImage {
        upsample_to_mask(logits, side, self.width, self.height)
    }

    fn load_reference(&self, path: &Path) -> Result<GrayImage> {
        let reference = image::open(path)
            .with_context(|| format!("failed to open reference {}", path.display()))?
            .to_luma8();
        let mut resized =
            image::imageops::resize(&reference, self.width, self.height, FilterType::Triangle);
        for pixel in resized.pixels_mut() {
            pixel[0] = if pixel[0] > 127 { 255 } else { 0 };
        }
        Ok(resized)
    }
}

/// The E1 reference mask of a target: the box mask, or the best multimask candidate of a point.
fn target_reference_name(
    stem: &str,
    reference: &Value,
    scores: &[Value],
    file: &str,
) -> Result<Option<String>> {
    if reference.is_null() {
        return Ok(None);
    }
    if let Some(box_name) = reference["box"].as_str() {
        return Ok(Some(format!("{stem}__{box_name}__single0.png")));
    }
    let point = reference["point"]
        .as_str()
        .ok_or_else(|| anyhow!("target reference needs box or point: {reference}"))?;
    let best = scores
        .iter()
        .filter(|r| {
            r["image"].as_str() == Some(file)
                && r["prompt"].as_str() == Some(point)
                && r["multimask"].as_bool() == Some(true)
        })
        .max_by(|a, b| {
            let score = |r: &Value| r["score_torch"].as_f64().unwrap_or(f64::MIN);
            score(a).total_cmp(&score(b))
        })
        .ok_or_else(|| anyhow!("no multimask reference scores for {file} {point}"))?;
    Ok(Some(format!(
        "{stem}__{point}__multi{}.png",
        best["mask_index"].as_u64().unwrap_or(0)
    )))
}

/// Points on a circle around (x, y) with `radius` as a fraction of the short photo side.
fn hover_points(
    x: f32,
    y: f32,
    radius: f32,
    count: usize,
    width: u32,
    height: u32,
) -> Vec<(f32, f32)> {
    let short = width.min(height) as f32;
    (0..count)
        .map(|k| {
            let angle = std::f32::consts::TAU * k as f32 / count as f32;
            (
                (x + radius * angle.cos() * short / width as f32).clamp(0.0, 1.0),
                (y + radius * angle.sin() * short / height as f32).clamp(0.0, 1.0),
            )
        })
        .collect()
}

fn mask_contains(mask: &GrayImage, x: f32, y: f32) -> bool {
    let px = ((x * mask.width() as f32) as u32).min(mask.width() - 1);
    let py = ((y * mask.height() as f32) as u32).min(mask.height() - 1);
    mask.get_pixel(px, py)[0] > 127
}

/// Same-highlight share and mean IoU of the highlights on the hover circle against the
/// highlight at the centre. `select` returns the highlighted grid mask at a point.
fn hover_consistency(
    centre: Option<&GridMask>,
    circle: &[(f32, f32)],
    mut select: impl FnMut(f32, f32) -> Result<Option<GridMask>>,
) -> Result<Value> {
    let Some(centre) = centre else {
        return Ok(json!({ "status": "unavailable", "reason": "no highlight at the centre" }));
    };
    if circle.is_empty() {
        return Ok(
            json!({ "status": "unavailable", "reason": "no hover point inside the target" }),
        );
    }
    let mut ious = Vec::new();
    for &(x, y) in circle {
        ious.push(select(x, y)?.map_or(0.0, |m| m.iou(centre)));
    }
    let same = ious.iter().filter(|&&v| v >= SAME_HIGHLIGHT_IOU).count();
    Ok(json!({
        "points": ious.len(),
        "same_highlight": same,
        "same_highlight_share": same as f64 / ious.len() as f64,
        "mean_iou": ious.iter().sum::<f64>() / ious.len() as f64,
        "minimum_iou": ious.iter().copied().fold(f64::INFINITY, f64::min),
    }))
}

/// Distinct colour per candidate (hue steps of 137.5 degrees).
fn candidate_color(index: usize) -> [f32; 3] {
    let hue = (index as f32 * 137.508) % 360.0;
    let (s, v) = (0.75f32, 0.95f32);
    let c = v * s;
    let x = c * (1.0 - ((hue / 60.0) % 2.0 - 1.0).abs());
    let (r, g, b) = match (hue / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [(r + m) * 255.0, (g + m) * 255.0, (b + m) * 255.0]
}

/// All candidates on the photo: the largest first, smaller candidates painted over it, white
/// outlines between labels. Pixels outside every candidate are grey.
fn write_candidate_map(
    frame: &EvaluationFrame,
    candidates: &[Candidate],
    side: usize,
    path: &Path,
) -> Result<()> {
    let (width, height) = (frame.width, frame.height);
    let mut labels = vec![u32::MAX; (width * height) as usize];
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(candidates[i].mask.area));
    for &index in &order {
        let mask = frame.mask(&candidates[index].logits, side);
        for (label, pixel) in labels.iter_mut().zip(mask.pixels()) {
            if pixel[0] > 127 {
                *label = index as u32;
            }
        }
    }
    let mut map = frame.photo.clone();
    for (i, pixel) in map.pixels_mut().enumerate() {
        let label = labels[i];
        if label == u32::MAX {
            let grey =
                (pixel[0] as f32 * 0.3 + pixel[1] as f32 * 0.59 + pixel[2] as f32 * 0.11) * 0.5;
            *pixel = Rgb([grey as u8; 3]);
            continue;
        }
        let color = candidate_color(label as usize);
        for channel in 0..3 {
            pixel[channel] = (pixel[channel] as f32 * 0.45 + color[channel] * 0.55) as u8;
        }
        let (x, y) = (i as u32 % width, i as u32 / width);
        let right = x + 1 < width && labels[i + 1] != label;
        let below = y + 1 < height && labels[i + width as usize] != label;
        if right || below {
            *pixel = Rgb([255, 255, 255]);
        }
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    DynamicImage::ImageRgb8(map)
        .save(path)
        .with_context(|| format!("failed to write {}", path.display()))
}

/// Raw candidate masks for `probe_candidates.py`: N x side x side bytes (1 inside) and a JSON
/// list with the score, stability, token, and grid point of each candidate.
fn write_candidate_masks(directory: &Path, stem: &str, set: &CandidateSet) -> Result<()> {
    fs::create_dir_all(directory)?;
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    for candidate in &set.candidates {
        let side = candidate.mask.side;
        for index in 0..side * side {
            bytes.push(((candidate.mask.bits[index / 64] >> (index % 64)) & 1) as u8);
        }
        entries.push(json!({
            "score": candidate.score, "stability": candidate.stability,
            "token": candidate.token, "point": candidate.point,
            "area_fraction": candidate.mask.fraction(),
        }));
    }
    fs::write(directory.join(format!("{stem}.u8")), bytes)?;
    write_report(
        &directory.join(format!("{stem}.json")),
        &json!({ "side": set.candidates.first().map_or(0, |c| c.mask.side), "candidates": entries }),
    )
}

/// Share of the model grid inside at least one candidate.
fn candidate_coverage(candidates: &[Candidate], side: usize) -> f64 {
    let mut union = vec![0u64; (side * side).div_ceil(64)];
    for candidate in candidates {
        for (u, b) in union.iter_mut().zip(&candidate.mask.bits) {
            *u |= b;
        }
    }
    let covered: u32 = union.iter().map(|w| w.count_ones()).sum();
    covered as f64 / (side * side) as f64
}

fn best_score_token(output: &DecoderOutput) -> usize {
    (1..4)
        .max_by(|&a, &b| output.iou_scores[a].total_cmp(&output.iou_scores[b]))
        .unwrap_or(1)
}

fn run_candidate_report(options: &CandidateOptions) -> Result<()> {
    ort_runtime::initialize_ort_environment()?;
    let export_report: Value = serde_json::from_str(
        &fs::read_to_string(options.onnx_dir.join("export_report.json"))
            .context("failed to read export_report.json")?,
    )?;
    let model_input = ModelInput::from_export_report(&export_report)?;
    let side = model_input.low_res_mask_size;
    let model_size = model_input.image_size as f32;
    let prompts: Value = serde_json::from_str(&fs::read_to_string(&options.prompts)?)?;
    let targets: Value = match &options.targets {
        Some(path) => serde_json::from_str(&fs::read_to_string(path)?)?,
        None => json!({ "targets": {} }),
    };
    let reference_scores: Vec<Value> = match &options.reference_scores {
        Some(path) => {
            let compare: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
            compare["results"].as_array().cloned().unwrap_or_default()
        }
        None => Vec::new(),
    };
    let mut report = json!({
        "device_request": options.device_text,
        "session_settings": SessionSettings::Production.label(),
        "variant": export_report["variant"],
        "device_description": ort_runtime::describe_device_request(options.device),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "generator": {
            "points_per_side": options.points_per_side,
            "batch": options.batch,
            "score_threshold": options.score_threshold,
            "stability_threshold": options.stability_threshold,
            "stability_offset": export_report["candidate_decoder"]["stability_offset"],
            "nms_threshold": options.nms_threshold,
            "dedupe": if options.dedupe_by_mask { "mask IoU" } else { "box IoU (upstream)" },
            "filters": "on the 256 x 256 logits (the upstream generator uses the masks at the photo size)",
        },
        "selection": {
            "level_merge_iou": options.level_merge_iou,
            "target_iou": TARGET_IOU,
            "same_highlight_iou": SAME_HIGHLIGHT_IOU,
            "hover_radius_of_short_side": options.hover_radius,
            "hover_points": options.hover_points,
            "evaluation_max_edge": options.evaluation_max_edge,
        },
        "memory_before_load": process_memory::sample(),
        "checks": {},
        "graphs": {},
        "images": [],
    });

    let mut sessions: BTreeMap<&str, Session> = BTreeMap::new();
    let mut load_check = json!({});
    for name in [POINT_ENCODER, POINT_DECODER, CANDIDATE_DECODER] {
        let path = options.onnx_dir.join(format!("{name}.onnx"));
        let started = Instant::now();
        match ort_runtime::load_session(&path, options.device) {
            Ok(session) => {
                let load_ms = milliseconds(started.elapsed());
                println!("[load] {name}: {load_ms:.0} ms");
                load_check[name] = json!({ "status": "passed" });
                report["graphs"][name] = json!({ "load_ms": load_ms });
                sessions.insert(name, session);
            }
            Err(error) => {
                println!("[load] {name}: FAILED {error}");
                load_check[name] = json!({ "status": "failed", "error": format!("{error:#}") });
            }
        }
    }
    report["checks"]["SegmentationGraphsLoadWithProductionSessionSettings"] = load_check;
    report["memory_after_all_loads"] = process_memory::sample();
    let (Some(mut encoder), Some(mut point_decoder), Some(mut candidate_decoder)) = (
        sessions.remove(POINT_ENCODER),
        sessions.remove(POINT_DECODER),
        sessions.remove(CANDIDATE_DECODER),
    ) else {
        return write_report(&options.out, &report);
    };

    let mut shape_check: Option<Value> = None;
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
        let photo = image::open(options.photos.join(file))
            .with_context(|| format!("failed to open {file}"))?
            .to_rgb8();
        let (width, height) = photo.dimensions();
        let started = Instant::now();
        let pixels = preprocess(&photo, &model_input);
        let preprocess_ms = milliseconds(started.elapsed());
        let frame = EvaluationFrame::new(&photo, options.evaluation_max_edge);
        drop(photo);

        let started = Instant::now();
        let features = run_point_encoder(&mut encoder, &pixels)?;
        let encoder_ms = milliseconds(started.elapsed());
        if first_image {
            let samples = time_runs(options.runs.max(10), || {
                run_point_encoder(&mut encoder, &pixels).map(|_| ())
            })?;
            report["graphs"][POINT_ENCODER]["first_inference_ms"] = json!(encoder_ms);
            report["graphs"][POINT_ENCODER]["warm"] = latency_summary(&samples);
        }
        let mut image_report = json!({
            "file": file, "width": width, "height": height,
            "preprocess_ms": preprocess_ms, "encoder_ms": encoder_ms,
            "grids": {}, "prompts": [],
        });

        let mut sets: BTreeMap<usize, CandidateSet> = BTreeMap::new();
        for &points_per_side in &options.points_per_side {
            let set = generate_candidates(
                &mut candidate_decoder,
                &features,
                &model_input,
                points_per_side,
                options.batch,
                options,
            )?;
            if shape_check.is_none() {
                let observed = &set.shapes;
                let rows = (point_grid(points_per_side).len() as i64).min(options.batch as i64);
                let side = side as i64;
                let matches = observed["mask_logits"] == [rows, 3, side, side]
                    && observed["iou_scores"] == [rows, 3]
                    && observed["stability_scores"] == [rows, 3];
                shape_check = Some(json!({
                    "status": if matches { "passed" } else { "failed" },
                    "observed_last_batch": observed,
                    "exported_specification": export_report["graphs"][CANDIDATE_DECODER]["outputs"],
                }));
            }
            let mut grid_report = json!({
                "masks_decoded": set.masks_decoded,
                "masks_after_filter": set.masks_after_filter,
                "candidates": set.candidates.len(),
                "decoder_ms": set.decoder_ms,
                "filter_ms": set.filter_ms,
                "nms_ms": set.nms_ms,
                "time_to_candidates_ms": encoder_ms + set.decoder_ms + set.filter_ms + set.nms_ms,
                "coverage": candidate_coverage(&set.candidates, side),
            });
            if first_image {
                let samples = time_runs(options.runs, || {
                    generate_candidates(
                        &mut candidate_decoder,
                        &features,
                        &model_input,
                        points_per_side,
                        options.batch,
                        options,
                    )
                    .map(|_| ())
                })?;
                grid_report["warm_candidate_pass"] = latency_summary(&samples);
            }
            if let Some(directory) = &options.mask_dir {
                write_candidate_masks(&directory.join(points_per_side.to_string()), &stem, &set)?;
            }
            if let Some(directory) = &options.overlay_dir
                && points_per_side == options.overlay_points_per_side
            {
                write_candidate_map(
                    &frame,
                    &set.candidates,
                    side,
                    &directory.join(format!("{stem}__candidates.jpg")),
                )?;
            }
            println!(
                "[candidates] {stem} {points_per_side}x{points_per_side}: {} candidates ({} after filter), encoder {encoder_ms:.0} ms + decoder {:.0} ms + host {:.0} ms, coverage {:.3}",
                set.candidates.len(),
                set.masks_after_filter,
                set.decoder_ms,
                set.filter_ms + set.nms_ms,
                grid_report["coverage"].as_f64().unwrap_or(0.0)
            );
            image_report["grids"][points_per_side.to_string()] = grid_report;
            sets.insert(points_per_side, set);
        }

        if first_image {
            // Batch size comparison of the candidate pass on the overlay grid.
            let mut batch_report = json!({});
            for &batch in &options.batch_sizes {
                generate_candidates(
                    &mut candidate_decoder,
                    &features,
                    &model_input,
                    options.overlay_points_per_side,
                    batch,
                    options,
                )?;
                let mut decoder_samples = Vec::new();
                let samples = time_runs(options.runs, || {
                    let set = generate_candidates(
                        &mut candidate_decoder,
                        &features,
                        &model_input,
                        options.overlay_points_per_side,
                        batch,
                        options,
                    )?;
                    decoder_samples.push(set.decoder_ms);
                    Ok(())
                })?;
                batch_report[batch.to_string()] = json!({
                    "candidate_pass": latency_summary(&samples),
                    "decoder": latency_summary(&decoder_samples),
                });
                println!("[batch] {batch}: {}", batch_report[batch.to_string()]);
            }
            report["batch_sizes"] = json!({
                "points_per_side": options.overlay_points_per_side,
                "photo": file,
                "results": batch_report,
            });
            report["memory_after_inference"] = process_memory::sample();
            if let Some(profile_dir) = &options.profile_dir {
                let grid = point_grid(options.overlay_points_per_side);
                let coords: Vec<[f32; 2]> = grid[..options.batch.min(grid.len())]
                    .iter()
                    .map(|p| [p[0] * model_size, p[1] * model_size])
                    .collect();
                let prefix = profile_dir.join(format!(
                    "{CANDIDATE_DECODER}_{}",
                    options.device_text.replace(':', "_")
                ));
                report["operator_placement"] = json!({
                    CANDIDATE_DECODER: operator_placement(
                        &options.onnx_dir.join(format!("{CANDIDATE_DECODER}.onnx")),
                        options.device,
                        SessionSettings::Production,
                        &prefix,
                        |session| {
                            run_candidate_decoder(session, &features, &coords)?;
                            run_candidate_decoder(session, &features, &coords).map(|_| ())
                        },
                    )
                    .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") })),
                    "prompts_per_run": coords.len(),
                });
            }
        }

        // Single-point prompts: the live point decoder and the candidate levels.
        let image_targets = &targets["targets"][file];
        for set_entry in prompt_sets(entry)? {
            if set_entry.points.len() != 1 || !set_entry.boxes.is_empty() {
                continue;
            }
            let (x, y, _) = set_entry.points[0];
            let target = &image_targets[set_entry.name.as_str()];
            let reference_name =
                target_reference_name(&stem, &target["reference"], &reference_scores, file)?;
            let reference = match (&reference_name, &options.reference_masks) {
                (Some(name), Some(directory)) => Some(frame.load_reference(&directory.join(name))?),
                _ => None,
            };
            let iou_with_reference = |logits: &[f32]| -> Result<Option<f64>> {
                match &reference {
                    Some(reference) => Ok(Some(mask_iou(&frame.mask(logits, side), reference)?)),
                    None => Ok(None),
                }
            };

            // Live point decoder: one decoder run per pointer position, tokens 1-3.
            let started = Instant::now();
            let output = run_point_decoder(
                &mut point_decoder,
                &features,
                &[[x * model_size, y * model_size]],
                &[1.0],
            )?;
            let live_ms = milliseconds(started.elapsed());
            let live_best = best_score_token(&output);
            let live_best_mask = GridMask::from_logits(&token_logits(&output, live_best), side);
            let mut live_tokens: Vec<usize> = (1..4).collect();
            let token_area = |t: usize| GridMask::from_logits(&token_logits(&output, t), side).area;
            live_tokens.sort_by_key(|&t| token_area(t));
            let mut live_levels = Vec::new();
            for (level, &token) in live_tokens.iter().enumerate() {
                let logits = token_logits(&output, token);
                if let Some(directory) = &options.overlay_dir {
                    write_overlay(
                        &frame.photo,
                        &frame.mask(&logits, side),
                        &set_entry.points,
                        &[],
                        0,
                        &directory
                            .join(format!("{stem}__{}__live_level{level}.jpg", set_entry.name)),
                    )?;
                }
                live_levels.push(json!({
                    "token": token, "score": output.iou_scores[token],
                    "area_fraction": GridMask::from_logits(&logits, side).fraction(),
                    "reference_iou": iou_with_reference(&logits)?,
                }));
            }
            let live_default_level = live_tokens
                .iter()
                .position(|&t| t == live_best)
                .unwrap_or(0);
            let circle: Vec<(f32, f32)> = hover_points(
                x,
                y,
                options.hover_radius,
                options.hover_points,
                width,
                height,
            )
            .into_iter()
            .filter(|&(px, py)| match &reference {
                Some(reference) => mask_contains(reference, px, py),
                None => live_best_mask.contains(px, py),
            })
            .collect();
            let live_hover = hover_consistency(Some(&live_best_mask), &circle, |px, py| {
                let out = run_point_decoder(
                    &mut point_decoder,
                    &features,
                    &[[px * model_size, py * model_size]],
                    &[1.0],
                )?;
                Ok(Some(GridMask::from_logits(
                    &token_logits(&out, best_score_token(&out)),
                    side,
                )))
            })?;
            // The highlight stays while the pointer is inside it; the decoder runs again only
            // when the pointer leaves the highlighted mask.
            let mut decoder_runs_outside = 0usize;
            let live_hover_kept_inside =
                hover_consistency(Some(&live_best_mask), &circle, |px, py| {
                    if live_best_mask.contains(px, py) {
                        return Ok(Some(live_best_mask.clone()));
                    }
                    decoder_runs_outside += 1;
                    let out = run_point_decoder(
                        &mut point_decoder,
                        &features,
                        &[[px * model_size, py * model_size]],
                        &[1.0],
                    )?;
                    Ok(Some(GridMask::from_logits(
                        &token_logits(&out, best_score_token(&out)),
                        side,
                    )))
                })?;
            if let Some(directory) = &options.overlay_dir {
                write_overlay(
                    &frame.photo,
                    &frame.mask(&token_logits(&output, live_best), side),
                    &set_entry.points,
                    &[],
                    0,
                    &directory.join(format!("{stem}__{}__live.jpg", set_entry.name)),
                )?;
            }
            let live_best_reachable = live_levels
                .iter()
                .filter_map(|l| l["reference_iou"].as_f64())
                .fold(None, |best: Option<f64>, v| {
                    Some(best.map_or(v, |b| b.max(v)))
                });

            let mut grids = json!({});
            for (&points_per_side, set) in &sets {
                let candidates = &set.candidates;
                let started = Instant::now();
                let levels = levels_at(candidates, x, y, options.level_merge_iou);
                let lookup_us = started.elapsed().as_secs_f64() * 1e6;
                let mut level_reports = Vec::new();
                let mut reference_ious = Vec::new();
                for (level, &index) in levels.iter().enumerate() {
                    let candidate = &candidates[index];
                    let reference_iou = iou_with_reference(&candidate.logits)?;
                    reference_ious.push(reference_iou);
                    level_reports.push(json!({
                        "level": level, "candidate": index, "score": candidate.score,
                        "stability": candidate.stability, "token": candidate.token,
                        "grid_point": candidate.point, "area_fraction": candidate.mask.fraction(),
                        "reference_iou": reference_iou,
                    }));
                    if let Some(directory) = &options.overlay_dir
                        && points_per_side == options.overlay_points_per_side
                    {
                        write_overlay(
                            &frame.photo,
                            &frame.mask(&candidate.logits, side),
                            &set_entry.points,
                            &[],
                            0,
                            &directory
                                .join(format!("{stem}__{}__level{level}.jpg", set_entry.name)),
                        )?;
                    }
                }
                let best_reachable = reference_ious
                    .iter()
                    .enumerate()
                    .filter_map(|(level, iou)| iou.map(|v| (level, v)))
                    .max_by(|a, b| a.1.total_cmp(&b.1));
                let mut defaults = json!({});
                for rule in DefaultLevel::ALL {
                    let pick = rule.pick(&levels, |i| candidates[i].score);
                    let centre = pick.map(|level| candidates[levels[level]].mask.clone());
                    let hover = hover_consistency(centre.as_ref(), &circle, |px, py| {
                        let at = levels_at(candidates, px, py, options.level_merge_iou);
                        Ok(rule
                            .pick(&at, |i| candidates[i].score)
                            .map(|level| candidates[at[level]].mask.clone()))
                    })?;
                    defaults[rule.name()] = json!({
                        "level": pick,
                        "reference_iou": pick.and_then(|level| reference_ious[level]),
                        "steps_to_best_reachable": match (pick, best_reachable) {
                            (Some(level), Some((best, _))) => json!(level.abs_diff(best)),
                            _ => Value::Null,
                        },
                        "hover": hover,
                    });
                }
                grids[points_per_side.to_string()] = json!({
                    "levels": level_reports,
                    "lookup_us": lookup_us,
                    "best_reachable": best_reachable
                        .map(|(level, iou)| json!({ "level": level, "reference_iou": iou })),
                    "defaults": defaults,
                });
            }
            image_report["prompts"].as_array_mut().unwrap().push(json!({
                "name": set_entry.name,
                "point": [x, y],
                "intent": target["intent"],
                "reference": reference_name,
                "live_point_decoder": {
                    "decode_ms": live_ms,
                    "levels": live_levels,
                    "default_best_score": {
                        "level": live_default_level,
                        "reference_iou": live_levels[live_default_level]["reference_iou"],
                        "steps_to_best_reachable": Value::Null,
                        "hover": live_hover,
                        "hover_kept_inside_highlight": live_hover_kept_inside,
                        "hover_decoder_runs_outside_highlight": decoder_runs_outside,
                    },
                    "best_reachable_iou": live_best_reachable,
                },
                "candidates": grids,
            }));
        }

        report["images"].as_array_mut().unwrap().push(image_report);
        write_report(&options.out, &report)?;
        first_image = false;
    }
    report["checks"]["CandidateDecoderOutputShapeMatchesExportedSpecification"] = shape_check
        .unwrap_or_else(
            || json!({ "status": "unavailable", "reason": "no candidate decoder run" }),
        );
    report["memory_at_end"] = process_memory::sample();
    write_report(&options.out, &report)
}

// ---------------------------------------------------------------------------------------------
// Stroke selection (#458, step E3)

struct StrokeOptions {
    onnx_dir: PathBuf,
    device_text: String,
    device: DeviceRequest,
    photos: PathBuf,
    prompts: PathBuf,
    targets: PathBuf,
    reference_masks: PathBuf,
    reference_scores: PathBuf,
    out: PathBuf,
    mask_dir: Option<PathBuf>,
    /// Long edge of the evaluation masks.
    evaluation_max_edge: u32,
    /// Minimum distance of each stroke point from the edge of the region, as a fraction of the
    /// short photo side.
    margin: f32,
    /// Stroke lengths as fractions of the inner chord through the prompt point.
    lengths: Vec<f32>,
    /// Point counts along a stroke (2 or more; one point is the click).
    counts: Vec<usize>,
    /// Point spacings as fractions of the short photo side.
    spacings: Vec<f32>,
    max_spacing_points: usize,
    /// Length fraction and point count of the first stroke and of the corrective strokes.
    primary_length: f32,
    primary_count: usize,
    /// Corrective interactions after the first one.
    corrective_steps: usize,
    /// Reference IoU at which a corrective sequence stops.
    stop_iou: f64,
    timing_counts: Vec<usize>,
    runs: usize,
    /// Operator profiles of the point decoder with these point counts.
    profile_dir: Option<PathBuf>,
    profile_counts: Vec<usize>,
    /// Pads every prompt to a fixed point count, so that the decoder always runs with one input
    /// shape.
    pad_points: Option<PointPadding>,
    /// One decoder session per point count, loaded on first use, so that each session runs only
    /// one input shape.
    session_per_point_count: bool,
}

impl StrokeOptions {
    fn from_map(options: &BTreeMap<String, String>) -> Result<Self> {
        let device_text = required(options, "device")?;
        let primary_count = parse_value(options, "primary-count", 5usize)?;
        if primary_count < 2 {
            bail!("--primary-count must be at least 2; one point is a click");
        }
        Ok(Self {
            onnx_dir: required(options, "onnx-dir")?.into(),
            device: ort_runtime::parse_device_request(&device_text)?,
            device_text,
            photos: required(options, "photos")?.into(),
            prompts: required(options, "prompts")?.into(),
            targets: required(options, "targets")?.into(),
            reference_masks: required(options, "reference-masks")?.into(),
            reference_scores: required(options, "reference-scores")?.into(),
            out: required(options, "out")?.into(),
            mask_dir: options.get("mask-dir").map(PathBuf::from),
            evaluation_max_edge: parse_value(options, "evaluation-max-edge", 1024)?,
            margin: parse_value(options, "margin", 0.01)?,
            lengths: parse_list(options, "lengths", "0.25,0.5,0.75,1")?,
            counts: parse_list(options, "counts", "2,3,4,5,8")?,
            spacings: parse_list(options, "spacings", "0.02,0.04,0.08")?,
            max_spacing_points: parse_value(options, "max-spacing-points", 16)?,
            primary_length: parse_value(options, "primary-length", 0.5)?,
            primary_count,
            corrective_steps: parse_value(options, "corrective-steps", 4)?,
            stop_iou: parse_value(options, "stop-iou", 0.9)?,
            timing_counts: parse_list(options, "timing-counts", "1,2,3,5,8,16,32")?,
            runs: parse_count(options, "runs", 20)?,
            profile_dir: options.get("profile-dir").map(PathBuf::from),
            profile_counts: parse_list(options, "profile-counts", "1,5")?,
            pad_points: match options.get("pad-points") {
                None => None,
                Some(value) => Some(PointPadding {
                    total: value
                        .parse()
                        .map_err(|_| anyhow!("failed to parse --pad-points {value:?}"))?,
                    repeat_last_point: match options.get("pad-mode").map_or("label", String::as_str)
                    {
                        "label" => false,
                        "repeat" => true,
                        other => bail!("unsupported --pad-mode {other:?}; use label or repeat"),
                    },
                }),
            },
            session_per_point_count: match options
                .get("decoder-sessions")
                .map_or("one", String::as_str)
            {
                "one" => false,
                "per-point-count" => true,
                other => {
                    bail!("unsupported --decoder-sessions {other:?}; use one or per-point-count")
                }
            },
        })
    }
}

/// Label of a padding point: the prompt encoder uses `not_a_point_embed` and ignores its
/// coordinates.
const PADDING_LABEL: f32 = -1.0;

/// A fixed point count of the decoder prompt.
#[derive(Clone, Copy)]
struct PointPadding {
    total: usize,
    /// `false`: padding points with label -1 (`--pad-mode label`); `true`: copies of the last
    /// point with its label (`--pad-mode repeat`).
    repeat_last_point: bool,
}

impl PointPadding {
    fn name(self) -> &'static str {
        if self.repeat_last_point {
            "repeat the last point"
        } else {
            "padding points with label -1"
        }
    }
}

/// Pads `coords` and `labels` to the point count of `padding`.
fn pad_prompt(
    coords: &mut Vec<[f32; 2]>,
    labels: &mut Vec<f32>,
    padding: Option<PointPadding>,
) -> Result<()> {
    let Some(padding) = padding else {
        return Ok(());
    };
    let total = padding.total;
    if coords.len() > total {
        bail!(
            "{} prompt points do not fit into --pad-points {total}",
            coords.len()
        );
    }
    let (point, label) = if padding.repeat_last_point {
        match (coords.last(), labels.last()) {
            (Some(&point), Some(&label)) => (point, label),
            _ => bail!("--pad-mode repeat needs at least one prompt point"),
        }
    } else {
        ([0.0, 0.0], PADDING_LABEL)
    };
    coords.resize(total, point);
    labels.resize(total, label);
    Ok(())
}

/// Initial value of the inside pixels for the squared distance transform.
const DISTANCE_INSIDE: f64 = 1e12;

/// One line of the squared Euclidean distance transform of Felzenszwalb and Huttenlocher.
/// `v` needs `f.len()` entries and `z` needs `f.len() + 1`.
fn squared_distance_1d(f: &[f64], d: &mut [f64], v: &mut [usize], z: &mut [f64]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    for q in 1..n {
        let qf = q as f64;
        let mut s;
        loop {
            let p = v[k] as f64;
            s = ((f[q] + qf * qf) - (f[v[k]] + p * p)) / (2.0 * (qf - p));
            if s <= z[k] {
                k -= 1;
            } else {
                break;
            }
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for (q, distance) in d.iter_mut().enumerate().take(n) {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let offset = q as f64 - v[k] as f64;
        *distance = offset * offset + f[v[k]];
    }
}

/// Euclidean distance of each inside pixel to the centre of the nearest outside pixel. Pixels
/// outside the photo count as outside; outside pixels have distance 0.
fn distance_to_outside(inside: &[bool], width: usize, height: usize) -> Vec<f32> {
    let (padded_width, padded_height) = (width + 2, height + 2);
    let mut grid = vec![0.0f64; padded_width * padded_height];
    for y in 0..height {
        for x in 0..width {
            if inside[y * width + x] {
                grid[(y + 1) * padded_width + x + 1] = DISTANCE_INSIDE;
            }
        }
    }
    let longest = padded_width.max(padded_height);
    let mut f = vec![0.0f64; longest];
    let mut d = vec![0.0f64; longest];
    let mut v = vec![0usize; longest];
    let mut z = vec![0.0f64; longest + 1];
    let n = padded_height;
    for x in 0..padded_width {
        for y in 0..n {
            f[y] = grid[y * padded_width + x];
        }
        squared_distance_1d(&f[..n], &mut d[..n], &mut v[..n], &mut z[..n + 1]);
        for y in 0..n {
            grid[y * padded_width + x] = d[y];
        }
    }
    let n = padded_width;
    for y in 0..padded_height {
        let row = y * padded_width;
        f[..n].copy_from_slice(&grid[row..row + n]);
        squared_distance_1d(&f[..n], &mut d[..n], &mut v[..n], &mut z[..n + 1]);
        grid[row..row + n].copy_from_slice(&d[..n]);
    }
    let mut distance = vec![0.0f32; width * height];
    for y in 0..height {
        for x in 0..width {
            distance[y * width + x] = grid[(y + 1) * padded_width + x + 1].sqrt() as f32;
        }
    }
    distance
}

/// A binary region on the evaluation frame and the distance of its pixels to its edge.
/// Coordinates are evaluation-frame pixels; pixel (i, j) covers [i, i + 1) x [j, j + 1).
struct Region {
    width: usize,
    height: usize,
    inside: Vec<bool>,
    distance: Vec<f32>,
}

impl Region {
    fn new(inside: Vec<bool>, width: usize, height: usize) -> Self {
        let distance = distance_to_outside(&inside, width, height);
        Self {
            width,
            height,
            inside,
            distance,
        }
    }

    fn from_mask(mask: &GrayImage) -> Self {
        let inside = mask.pixels().map(|p| p[0] > 127).collect();
        Self::new(inside, mask.width() as usize, mask.height() as usize)
    }

    fn index(&self, point: [f32; 2]) -> Option<usize> {
        let (x, y) = (point[0].floor(), point[1].floor());
        if x < 0.0 || y < 0.0 || x >= self.width as f32 || y >= self.height as f32 {
            return None;
        }
        Some(y as usize * self.width + x as usize)
    }

    fn distance_at(&self, point: [f32; 2]) -> f32 {
        self.index(point).map_or(0.0, |i| self.distance[i])
    }

    fn centre(&self, index: usize) -> [f32; 2] {
        [
            (index % self.width) as f32 + 0.5,
            (index / self.width) as f32 + 0.5,
        ]
    }

    fn area_fraction(&self) -> f64 {
        self.inside.iter().filter(|&&v| v).count() as f64 / self.inside.len() as f64
    }

    /// The pixel farthest from the edge and its distance.
    fn deepest(&self) -> Option<(usize, f32)> {
        self.distance
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, d)| d > 0.0)
            .max_by(|a, b| a.1.total_cmp(&b.1))
    }

    /// Principal axis (unit vector) of the 4-connected component that contains `start`.
    fn principal_direction(&self, start: usize) -> [f32; 2] {
        if !self.inside[start] {
            return [1.0, 0.0];
        }
        let mut visited = vec![false; self.inside.len()];
        let mut queue = std::collections::VecDeque::from([start]);
        visited[start] = true;
        let (mut n, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
        let (mut sxx, mut syy, mut sxy) = (0.0f64, 0.0f64, 0.0f64);
        while let Some(index) = queue.pop_front() {
            let (x, y) = (index % self.width, index / self.width);
            let (fx, fy) = (x as f64, y as f64);
            n += 1.0;
            sx += fx;
            sy += fy;
            sxx += fx * fx;
            syy += fy * fy;
            sxy += fx * fy;
            let mut neighbours = [None; 4];
            if x > 0 {
                neighbours[0] = Some(index - 1);
            }
            if x + 1 < self.width {
                neighbours[1] = Some(index + 1);
            }
            if y > 0 {
                neighbours[2] = Some(index - self.width);
            }
            if y + 1 < self.height {
                neighbours[3] = Some(index + self.width);
            }
            for neighbour in neighbours.into_iter().flatten() {
                if self.inside[neighbour] && !visited[neighbour] {
                    visited[neighbour] = true;
                    queue.push_back(neighbour);
                }
            }
        }
        let (mx, my) = (sx / n, sy / n);
        let cxx = sxx / n - mx * mx;
        let cyy = syy / n - my * my;
        let cxy = sxy / n - mx * my;
        let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
        [angle.cos() as f32, angle.sin() as f32]
    }

    /// Distances back and forward from `point` along `direction` to the last positions whose
    /// distance to the edge is at least `margin` (0.5 px steps). (0, 0) if `point` itself is
    /// closer to the edge.
    fn inner_chord(&self, point: [f32; 2], direction: [f32; 2], margin: f32) -> (f32, f32) {
        const STEP: f32 = 0.5;
        if self.distance_at(point) < margin {
            return (0.0, 0.0);
        }
        let reach = |sign: f32| {
            let mut t = 0.0f32;
            loop {
                let next = t + STEP;
                let position = [
                    point[0] + sign * next * direction[0],
                    point[1] + sign * next * direction[1],
                ];
                if self.distance_at(position) < margin {
                    return t;
                }
                t = next;
            }
        };
        (reach(-1.0), reach(1.0))
    }
}

/// A straight stroke in evaluation-frame pixels.
#[derive(Clone, Copy)]
struct Stroke {
    start: [f32; 2],
    end: [f32; 2],
}

impl Stroke {
    /// `fraction` of the inner chord `(back, forward)` through `point`, centred on `point` as far
    /// as the chord allows.
    fn along_chord(
        point: [f32; 2],
        direction: [f32; 2],
        (back, forward): (f32, f32),
        fraction: f32,
    ) -> Option<Self> {
        let chord = back + forward;
        if chord <= 0.0 {
            return None;
        }
        let length = fraction.clamp(0.0, 1.0) * chord;
        let offset = (-length / 2.0).clamp(-back, forward - length);
        let at = |t: f32| [point[0] + t * direction[0], point[1] + t * direction[1]];
        Some(Self {
            start: at(offset),
            end: at(offset + length),
        })
    }

    fn length(&self) -> f32 {
        (self.end[0] - self.start[0]).hypot(self.end[1] - self.start[1])
    }

    /// `count` points at equal spacing from the start to the end; one point is the midpoint.
    fn sample(&self, count: usize) -> Vec<[f32; 2]> {
        if count <= 1 {
            return vec![[
                (self.start[0] + self.end[0]) / 2.0,
                (self.start[1] + self.end[1]) / 2.0,
            ]];
        }
        (0..count)
            .map(|k| {
                let t = k as f32 / (count - 1) as f32;
                [
                    self.start[0] + t * (self.end[0] - self.start[0]),
                    self.start[1] + t * (self.end[1] - self.start[1]),
                ]
            })
            .collect()
    }
}

/// How the interactions of one selection become one mask.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StrokeCombination {
    /// All points of all interactions in one decoder run, with the previous selection as
    /// `mask_input` (the interactive refinement of SAM).
    PointsAndPreviousMask,
    /// All points of all interactions in one decoder run, without `mask_input`.
    Points,
    /// Each stroke decoded alone with positive points; an add stroke takes the union, a subtract
    /// stroke removes its mask (maximum and minimum of the logits).
    UnionOfStrokes,
}

impl StrokeCombination {
    fn name(self) -> &'static str {
        match self {
            Self::PointsAndPreviousMask => "points_and_previous_mask",
            Self::Points => "points",
            Self::UnionOfStrokes => "union_of_strokes",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Interaction {
    /// One click; the multimask output with the best predicted IoU (the E4 default).
    Click,
    /// One click; the largest multimask output (usually the whole object), so that a part
    /// target needs subtract corrections.
    ClickLargestMask,
    /// A stroke of `--primary-length` of the inner chord with `--primary-count` points.
    Stroke,
    /// A stroke along the whole inner chord with `--primary-count` points (first interaction
    /// only).
    FullStroke,
}

impl Interaction {
    fn name(self) -> &'static str {
        match self {
            Self::Click => "click",
            Self::ClickLargestMask => "click_largest_mask",
            Self::Stroke => "stroke",
            Self::FullStroke => "full_stroke",
        }
    }
}

/// A corrective interaction in the largest error region.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Correction {
    Click,
    Stroke,
}

impl Correction {
    fn name(self) -> &'static str {
        match self {
            Self::Click => "click",
            Self::Stroke => "stroke",
        }
    }
}

/// One corrective sequence: the first interaction, then corrections in the largest error region.
struct CorrectionSequence {
    name: &'static str,
    first: Interaction,
    corrective: Correction,
    combination: StrokeCombination,
}

const CORRECTION_SEQUENCES: [CorrectionSequence; 9] = [
    CorrectionSequence {
        name: "click_then_clicks",
        first: Interaction::Click,
        corrective: Correction::Click,
        combination: StrokeCombination::PointsAndPreviousMask,
    },
    CorrectionSequence {
        name: "click_then_clicks_without_mask_input",
        first: Interaction::Click,
        corrective: Correction::Click,
        combination: StrokeCombination::Points,
    },
    CorrectionSequence {
        name: "click_then_strokes",
        first: Interaction::Click,
        corrective: Correction::Stroke,
        combination: StrokeCombination::PointsAndPreviousMask,
    },
    CorrectionSequence {
        name: "stroke_then_strokes",
        first: Interaction::Stroke,
        corrective: Correction::Stroke,
        combination: StrokeCombination::PointsAndPreviousMask,
    },
    CorrectionSequence {
        name: "stroke_then_strokes_without_mask_input",
        first: Interaction::Stroke,
        corrective: Correction::Stroke,
        combination: StrokeCombination::Points,
    },
    CorrectionSequence {
        name: "stroke_then_strokes_union",
        first: Interaction::Stroke,
        corrective: Correction::Stroke,
        combination: StrokeCombination::UnionOfStrokes,
    },
    CorrectionSequence {
        name: "full_stroke_then_strokes",
        first: Interaction::FullStroke,
        corrective: Correction::Stroke,
        combination: StrokeCombination::PointsAndPreviousMask,
    },
    CorrectionSequence {
        name: "largest_mask_click_then_clicks",
        first: Interaction::ClickLargestMask,
        corrective: Correction::Click,
        combination: StrokeCombination::PointsAndPreviousMask,
    },
    CorrectionSequence {
        name: "largest_mask_click_then_strokes",
        first: Interaction::ClickLargestMask,
        corrective: Correction::Stroke,
        combination: StrokeCombination::PointsAndPreviousMask,
    },
];

/// Fixed stroke directions of the direction sweep, in degrees (0 = right, 90 = down).
const STROKE_DIRECTIONS_DEGREES: [f32; 4] = [0.0, 45.0, 90.0, 135.0];

/// The multimask token (1 to 3) with the largest mask on the model grid.
fn largest_area_token(output: &DecoderOutput, side: usize) -> usize {
    (1..4)
        .max_by_key(|&token| GridMask::from_logits(&token_logits(output, token), side).area)
        .unwrap_or(1)
}

/// Per-token result of one decoder run and the token of each selection rule.
fn decode_summary(
    output: &DecoderOutput,
    frame: &EvaluationFrame,
    side: usize,
    reference: &GrayImage,
) -> Result<Value> {
    let single = select_masks(output, false)[0].0;
    let best = best_score_token(output);
    let largest = largest_area_token(output, side);
    let mut tokens = Vec::new();
    for token in 0..4 {
        let logits = token_logits(output, token);
        let grid = GridMask::from_logits(&logits, side);
        tokens.push(json!({
            "token": token,
            "score": output.iou_scores[token],
            "area_fraction": grid.fraction(),
            "reference_iou": mask_iou(&frame.mask(&logits, side), reference)?,
        }));
    }
    Ok(json!({
        "single_mask_token": single,
        "best_score_token": best,
        "largest_token": largest,
        "single_mask_iou": tokens[single]["reference_iou"],
        "best_score_iou": tokens[best]["reference_iou"],
        "tokens": tokens,
    }))
}

/// Normalized image coordinates of evaluation-frame pixels.
fn normalized_points(frame: &EvaluationFrame, points: &[[f32; 2]]) -> Vec<[f32; 2]> {
    points
        .iter()
        .map(|p| [p[0] / frame.width as f32, p[1] / frame.height as f32])
        .collect()
}

fn stroke_json(frame: &EvaluationFrame, stroke: &Stroke) -> Value {
    let ends = normalized_points(frame, &[stroke.start, stroke.end]);
    json!({ "start": ends[0], "end": ends[1], "length_px": stroke.length() })
}

/// The point decoder sessions of the stroke report: `report` for every run, or one session per
/// point count when `--decoder-sessions per-point-count` is set.
struct StrokeDecoders {
    report: Session,
    per_point_count: Option<BTreeMap<usize, Session>>,
    graph_path: PathBuf,
    device: DeviceRequest,
    load_ms: BTreeMap<usize, f64>,
}

impl StrokeDecoders {
    fn session(&mut self, point_count: usize) -> Result<&mut Session> {
        let Some(sessions) = &mut self.per_point_count else {
            return Ok(&mut self.report);
        };
        Ok(match sessions.entry(point_count) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(entry) => {
                let started = Instant::now();
                let session = ort_runtime::load_session(&self.graph_path, self.device)?;
                self.load_ms
                    .insert(point_count, milliseconds(started.elapsed()));
                entry.insert(session)
            }
        })
    }
}

/// The photo, its encoder output, and the reference of one prompt.
struct StrokeContext<'a> {
    frame: &'a EvaluationFrame,
    features: &'a PointFeatures,
    side: usize,
    model_size: f32,
    reference: &'a GrayImage,
    options: &'a StrokeOptions,
}

/// Point decoder run with timing; `points` are evaluation-frame pixels, `labels` 1 positive and
/// 0 negative. The time goes to `decode_ms_by_count` under the point count.
fn timed_decode(
    decoders: &mut StrokeDecoders,
    context: &StrokeContext<'_>,
    points: &[[f32; 2]],
    labels: &[f32],
    mask_input: Option<&[f32]>,
    decode_ms_by_count: &mut BTreeMap<usize, Vec<f64>>,
) -> Result<(DecoderOutput, f64)> {
    let mut coords: Vec<[f32; 2]> = normalized_points(context.frame, points)
        .into_iter()
        .map(|p| [p[0] * context.model_size, p[1] * context.model_size])
        .collect();
    let mut labels = labels.to_vec();
    pad_prompt(&mut coords, &mut labels, context.options.pad_points)?;
    let decoder = decoders.session(coords.len())?;
    let started = Instant::now();
    let output =
        run_point_decoder_with_mask(decoder, context.features, &coords, &labels, mask_input)?;
    let decode_ms = milliseconds(started.elapsed());
    decode_ms_by_count
        .entry(points.len())
        .or_default()
        .push(decode_ms);
    Ok((output, decode_ms))
}

/// The first interactions of one prompt, in evaluation-frame pixels.
struct FirstInteractions {
    click: [f32; 2],
    /// `--primary-length` of the inner chord along the principal axis.
    stroke: Option<Stroke>,
    /// The whole inner chord along the principal axis.
    full_stroke: Option<Stroke>,
}

/// The first interaction, then up to `corrective_steps` corrections. Each correction goes to the
/// error region (missed reference pixels or extra pixels) whose deepest pixel is farthest from
/// its edge, as in the click simulation of the SAM papers: a click at that pixel, or a stroke
/// through it along the principal axis of its component. Missed pixels give an add (positive)
/// correction, extra pixels a subtract (negative) correction.
fn simulate_corrections(
    decoders: &mut StrokeDecoders,
    context: &StrokeContext<'_>,
    sequence: &CorrectionSequence,
    first: &FirstInteractions,
    mask_prefix: Option<&Path>,
    decode_ms_by_count: &mut BTreeMap<usize, Vec<f64>>,
) -> Result<Value> {
    let options = context.options;
    let (frame, side) = (context.frame, context.side);
    let (width, height) = (frame.width as usize, frame.height as usize);
    let first_stroke = match sequence.first {
        Interaction::Stroke => first.stroke.as_ref(),
        Interaction::FullStroke => first.full_stroke.as_ref(),
        Interaction::Click | Interaction::ClickLargestMask => None,
    };
    let (mut points, first_stroke_json) = match (sequence.first, first_stroke) {
        (Interaction::Click | Interaction::ClickLargestMask, _) => (vec![first.click], Value::Null),
        (_, Some(stroke)) => (
            stroke.sample(options.primary_count),
            stroke_json(frame, stroke),
        ),
        (_, None) => {
            return Ok(json!({
                "status": "unavailable",
                "reason": "no first stroke inside the reference at the margin",
            }));
        }
    };
    let mut labels = vec![1.0f32; points.len()];
    let (output, decode_ms) = timed_decode(
        decoders,
        context,
        &points,
        &labels,
        None,
        decode_ms_by_count,
    )?;
    // One click uses the multimask output with the best predicted IoU (the E4 default) or the
    // largest multimask output; several points use the single-mask output.
    let token = match sequence.first {
        Interaction::Click => best_score_token(&output),
        Interaction::ClickLargestMask => largest_area_token(&output, side),
        Interaction::Stroke | Interaction::FullStroke => select_masks(&output, false)[0].0,
    };
    let mut current = token_logits(&output, token);
    let reference_inside: Vec<bool> = context.reference.pixels().map(|p| p[0] > 127).collect();
    let mut steps = vec![json!({
        "step": 0, "interaction": sequence.first.name(), "label": 1,
        "points": normalized_points(frame, &points), "stroke": first_stroke_json,
        "token": token, "decode_ms": decode_ms, "prompt_points": points.len(),
    })];
    let mut stop_reason = "corrective step limit";
    for step in 0..=options.corrective_steps {
        let mask = frame.mask(&current, side);
        let iou = mask_iou(&mask, context.reference)?;
        if let Some(prefix) = mask_prefix {
            let path = PathBuf::from(format!(
                "{}__{}__step{step}.png",
                prefix.display(),
                sequence.name
            ));
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            mask.save(&path)
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        steps[step]["reference_iou"] = json!(iou);
        steps[step]["area_fraction"] = json!(mask_fraction(&mask));
        if iou >= options.stop_iou {
            stop_reason = "reached the stop IoU";
            break;
        }
        if step == options.corrective_steps {
            break;
        }

        let mask_inside: Vec<bool> = mask.pixels().map(|p| p[0] > 127).collect();
        let error_region = |missed: bool| {
            let inside = reference_inside
                .iter()
                .zip(&mask_inside)
                .map(|(&r, &m)| if missed { r && !m } else { m && !r })
                .collect();
            Region::new(inside, width, height)
        };
        let (missed, extra) = (error_region(true), error_region(false));
        let deepest_missed = missed.deepest();
        let deepest_extra = extra.deepest();
        let (region, label, (index, depth)) = match (deepest_missed, deepest_extra) {
            (Some(m), Some(e)) if e.1 > m.1 => (&extra, 0.0f32, e),
            (Some(m), _) => (&missed, 1.0f32, m),
            (None, Some(e)) => (&extra, 0.0f32, e),
            (None, None) => {
                stop_reason = "no error region";
                break;
            }
        };
        if depth < 1.5 {
            stop_reason = "no error region deeper than 1.5 px";
            break;
        }
        let deepest = region.centre(index);
        let (new_points, stroke) = match sequence.corrective {
            Correction::Click => (vec![deepest], None),
            Correction::Stroke => {
                let direction = region.principal_direction(index);
                let chord = region.inner_chord(deepest, direction, depth * 0.5);
                match Stroke::along_chord(deepest, direction, chord, options.primary_length) {
                    Some(stroke) if stroke.length() >= 1.0 => {
                        (stroke.sample(options.primary_count), Some(stroke))
                    }
                    _ => (vec![deepest], None),
                }
            }
        };
        let (output, decode_ms) = match sequence.combination {
            StrokeCombination::PointsAndPreviousMask | StrokeCombination::Points => {
                points.extend_from_slice(&new_points);
                labels.extend(std::iter::repeat_n(label, new_points.len()));
                let mask_input = (sequence.combination == StrokeCombination::PointsAndPreviousMask)
                    .then_some(current.as_slice());
                timed_decode(
                    decoders,
                    context,
                    &points,
                    &labels,
                    mask_input,
                    decode_ms_by_count,
                )?
            }
            StrokeCombination::UnionOfStrokes => {
                let positive = vec![1.0f32; new_points.len()];
                timed_decode(
                    decoders,
                    context,
                    &new_points,
                    &positive,
                    None,
                    decode_ms_by_count,
                )?
            }
        };
        let token = select_masks(&output, false)[0].0;
        let decoded = token_logits(&output, token);
        current = match sequence.combination {
            StrokeCombination::UnionOfStrokes if label > 0.5 => current
                .iter()
                .zip(&decoded)
                .map(|(&a, &b)| a.max(b))
                .collect(),
            StrokeCombination::UnionOfStrokes => current
                .iter()
                .zip(&decoded)
                .map(|(&a, &b)| a.min(-b))
                .collect(),
            _ => decoded,
        };
        let prompt_points = if sequence.combination == StrokeCombination::UnionOfStrokes {
            new_points.len()
        } else {
            points.len()
        };
        steps.push(json!({
            "step": step + 1,
            "interaction": if stroke.is_some() { "stroke" } else { "click" },
            "label": label as i32,
            "error_region": if label > 0.5 { "missed" } else { "extra" },
            "error_region_area_fraction": region.area_fraction(),
            "error_region_depth_px": depth,
            "points": normalized_points(frame, &new_points),
            "stroke": stroke.as_ref().map_or(Value::Null, |s| stroke_json(frame, s)),
            "token": token,
            "decode_ms": decode_ms,
            "prompt_points": prompt_points,
        }));
    }
    let ious: Vec<f64> = steps
        .iter()
        .filter_map(|s| s["reference_iou"].as_f64())
        .collect();
    let interactions_to = |threshold: f64| {
        ious.iter()
            .position(|&v| v >= threshold)
            .map_or(Value::Null, |i| json!(i + 1))
    };
    Ok(json!({
        "first": sequence.first.name(),
        "corrective": sequence.corrective.name(),
        "combination": sequence.combination.name(),
        "interactions_to_iou_0_85": interactions_to(TARGET_IOU),
        "interactions_to_iou_0_90": interactions_to(0.9),
        "final_reference_iou": ious.last(),
        "stop_reason": stop_reason,
        "steps": steps,
    }))
}

/// `count` model-input points on a horizontal line through the photo centre.
fn timing_points(count: usize, model_size: f32) -> Vec<[f32; 2]> {
    (0..count)
        .map(|k| {
            let t = if count > 1 {
                k as f32 / (count - 1) as f32
            } else {
                0.5
            };
            [(0.3 + 0.4 * t) * model_size, 0.5 * model_size]
        })
        .collect()
}

/// Decoder time for `count` points on a horizontal line, with and without the previous mask as
/// `mask_input`, in a new decoder session (whose first input shape is this point count) and in
/// `decoder`, the session of the report, which runs the counts in the order of
/// `--timing-counts`.
fn stroke_decoder_timing(
    decoder: &mut Session,
    options: &StrokeOptions,
    features: &PointFeatures,
    model_size: f32,
    count: usize,
) -> Result<Value> {
    let mut coords = timing_points(count, model_size);
    let mut labels = vec![1.0f32; count];
    pad_prompt(&mut coords, &mut labels, options.pad_points)?;
    let previous = token_logits(&run_point_decoder(decoder, features, &coords, &labels)?, 0);
    let started = Instant::now();
    let mut new_session = ort_runtime::load_session(
        &options.onnx_dir.join(format!("{POINT_DECODER}.onnx")),
        options.device,
    )?;
    let mut entry = json!({
        "decoder_points": coords.len(),
        "new_session_load_ms": milliseconds(started.elapsed()),
    });
    for (session_name, session) in [
        ("new_session", &mut new_session),
        ("report_session", &mut *decoder),
    ] {
        for with_mask in [false, true] {
            let mask_input = with_mask.then_some(previous.as_slice());
            for _ in 0..3 {
                run_point_decoder_with_mask(session, features, &coords, &labels, mask_input)?;
            }
            let samples = time_runs(options.runs, || {
                run_point_decoder_with_mask(session, features, &coords, &labels, mask_input)
                    .map(|_| ())
            })?;
            let key = if with_mask {
                "with_mask_input"
            } else {
                "without_mask_input"
            };
            entry[session_name][key] = latency_summary(&samples);
        }
    }
    Ok(entry)
}

fn run_stroke_report(options: &StrokeOptions) -> Result<()> {
    ort_runtime::initialize_ort_environment()?;
    let export_report: Value = serde_json::from_str(
        &fs::read_to_string(options.onnx_dir.join("export_report.json"))
            .context("failed to read export_report.json")?,
    )?;
    let model_input = ModelInput::from_export_report(&export_report)?;
    let side = model_input.low_res_mask_size;
    let model_size = model_input.image_size as f32;
    let prompts: Value = serde_json::from_str(&fs::read_to_string(&options.prompts)?)?;
    let targets: Value = serde_json::from_str(&fs::read_to_string(&options.targets)?)?;
    let compare: Value = serde_json::from_str(&fs::read_to_string(&options.reference_scores)?)?;
    let reference_scores = compare["results"].as_array().cloned().unwrap_or_default();
    let mut report = json!({
        "device_request": options.device_text,
        "session_settings": SessionSettings::Production.label(),
        "variant": export_report["variant"],
        "device_description": ort_runtime::describe_device_request(options.device),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "strokes": {
            "margin_of_short_side": options.margin,
            "lengths_of_inner_chord": options.lengths,
            "counts": options.counts,
            "spacings_of_short_side": options.spacings,
            "max_spacing_points": options.max_spacing_points,
            "fixed_directions_degrees": STROKE_DIRECTIONS_DEGREES,
            "primary": {
                "direction": "principal axis of the reference component that holds the prompt point",
                "length_of_inner_chord": options.primary_length,
                "count": options.primary_count,
            },
            "corrective_steps": options.corrective_steps,
            "corrective_stroke_margin": "half of the depth of the error region",
            "stop_iou": options.stop_iou,
            "target_iou": TARGET_IOU,
            "evaluation_max_edge": options.evaluation_max_edge,
            "pad_points": options.pad_points.map(|p| json!({ "total": p.total, "mode": p.name() })),
            "decoder_sessions": if options.session_per_point_count { "one per point count" } else { "one" },
            "selection": "one point: multimask output, best predicted IoU (E4 default), or the largest multimask output in the largest_mask sequences; several points: single-mask output (token 0, or the best multimask token when token 0 is not stable)",
        },
        "memory_before_load": process_memory::sample(),
        "checks": {},
        "graphs": {},
        "images": [],
    });

    let mut sessions: BTreeMap<&str, Session> = BTreeMap::new();
    let mut load_check = json!({});
    for name in [POINT_ENCODER, POINT_DECODER] {
        let path = options.onnx_dir.join(format!("{name}.onnx"));
        let started = Instant::now();
        match ort_runtime::load_session(&path, options.device) {
            Ok(session) => {
                let load_ms = milliseconds(started.elapsed());
                println!("[load] {name}: {load_ms:.0} ms");
                load_check[name] = json!({ "status": "passed" });
                report["graphs"][name] = json!({ "load_ms": load_ms });
                sessions.insert(name, session);
            }
            Err(error) => {
                println!("[load] {name}: FAILED {error}");
                load_check[name] = json!({ "status": "failed", "error": format!("{error:#}") });
            }
        }
    }
    report["checks"]["SegmentationGraphsLoadWithProductionSessionSettings"] = load_check;
    report["memory_after_all_loads"] = process_memory::sample();
    let (Some(mut encoder), Some(decoder)) = (
        sessions.remove(POINT_ENCODER),
        sessions.remove(POINT_DECODER),
    ) else {
        return write_report(&options.out, &report);
    };
    let mut decoders = StrokeDecoders {
        report: decoder,
        per_point_count: options.session_per_point_count.then(BTreeMap::new),
        graph_path: options.onnx_dir.join(format!("{POINT_DECODER}.onnx")),
        device: options.device,
        load_ms: BTreeMap::new(),
    };

    let mut decode_ms_by_count: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
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
        let photo = image::open(options.photos.join(file))
            .with_context(|| format!("failed to open {file}"))?
            .to_rgb8();
        let (width, height) = photo.dimensions();
        let pixels = preprocess(&photo, &model_input);
        let frame = EvaluationFrame::new(&photo, options.evaluation_max_edge);
        drop(photo);
        let started = Instant::now();
        let features = run_point_encoder(&mut encoder, &pixels)?;
        let encoder_ms = milliseconds(started.elapsed());
        let short_side = frame.width.min(frame.height) as f32;
        let mut image_report = json!({
            "file": file, "width": width, "height": height,
            "evaluation_width": frame.width, "evaluation_height": frame.height,
            "encoder_ms": encoder_ms, "prompts": [],
        });

        if first_image {
            let mut timing = json!({});
            for &count in &options.timing_counts {
                let entry = stroke_decoder_timing(
                    &mut decoders.report,
                    options,
                    &features,
                    model_size,
                    count,
                )?;
                println!("[timing] {count} points: {entry}");
                timing[count.to_string()] = entry;
            }
            report["decoder_time_per_stroke"] = json!({ "photo": file, "points": timing });
            report["memory_after_inference"] = process_memory::sample();
            if let Some(profile_dir) = &options.profile_dir {
                let mut placement = json!({ "runs_per_profile": options.runs + 1 });
                for &count in &options.profile_counts {
                    let mut coords = timing_points(count, model_size);
                    let mut labels = vec![1.0f32; count];
                    pad_prompt(&mut coords, &mut labels, options.pad_points)?;
                    let prefix = profile_dir.join(format!(
                        "{POINT_DECODER}_{count}pt_{}",
                        options.device_text.replace(':', "_")
                    ));
                    placement[count.to_string()] = operator_placement(
                        &options.onnx_dir.join(format!("{POINT_DECODER}.onnx")),
                        options.device,
                        SessionSettings::Production,
                        &prefix,
                        |session| {
                            for _ in 0..=options.runs {
                                run_point_decoder(session, &features, &coords, &labels)?;
                            }
                            Ok(())
                        },
                    )
                    .unwrap_or_else(|e| json!({ "status": "failed", "error": format!("{e:#}") }));
                }
                report["operator_placement"] = json!({ POINT_DECODER: placement });
            }
        }

        let image_targets = &targets["targets"][file];
        for set_entry in prompt_sets(entry)? {
            if set_entry.points.len() != 1 || !set_entry.boxes.is_empty() {
                continue;
            }
            let (x, y, _) = set_entry.points[0];
            let target = &image_targets[set_entry.name.as_str()];
            let unavailable = |reference: Option<&str>, reason: &str| {
                json!({
                    "name": set_entry.name, "point": [x, y], "intent": target["intent"],
                    "reference": reference, "status": "unavailable", "reason": reason,
                })
            };
            let Some(reference_name) =
                target_reference_name(&stem, &target["reference"], &reference_scores, file)?
            else {
                let skipped = unavailable(None, "no reference mask");
                image_report["prompts"]
                    .as_array_mut()
                    .unwrap()
                    .push(skipped);
                continue;
            };
            let reference = frame.load_reference(&options.reference_masks.join(&reference_name))?;
            let region = Region::from_mask(&reference);
            let click = [x * frame.width as f32, y * frame.height as f32];
            let start_distance = region.distance_at(click);
            if start_distance <= 0.0 {
                let skipped = unavailable(
                    Some(&reference_name),
                    "the prompt point is outside the reference",
                );
                image_report["prompts"]
                    .as_array_mut()
                    .unwrap()
                    .push(skipped);
                continue;
            }
            let context = StrokeContext {
                frame: &frame,
                features: &features,
                side,
                model_size,
                reference: &reference,
                options,
            };
            // The stroke points keep the margin from the edge; a prompt point closer to the edge
            // than the margin uses half of its own distance as the margin.
            let margin = (options.margin * short_side).min(start_distance * 0.5);
            let start_index = region.index(click).unwrap_or(0);
            let principal = region.principal_direction(start_index);
            let mut directions: Vec<(String, [f32; 2])> =
                vec![("principal".to_string(), principal)];
            for degrees in STROKE_DIRECTIONS_DEGREES {
                let radians = degrees.to_radians();
                directions.push((format!("{degrees:.0}"), [radians.cos(), radians.sin()]));
            }

            // One click: the same decoder run gives the multimask and the single-mask selection.
            let (click_output, click_ms) = timed_decode(
                &mut decoders,
                &context,
                &[click],
                &[1.0],
                None,
                &mut decode_ms_by_count,
            )?;
            let click_summary = decode_summary(&click_output, &frame, side, &reference)?;

            let mut sweep = Vec::new();
            for (direction_name, direction) in &directions {
                let chord = region.inner_chord(click, *direction, margin);
                for &fraction in &options.lengths {
                    let Some(stroke) = Stroke::along_chord(click, *direction, chord, fraction)
                    else {
                        continue;
                    };
                    let mut runs: Vec<(usize, Option<f32>)> =
                        options.counts.iter().map(|&c| (c.max(2), None)).collect();
                    for &spacing in &options.spacings {
                        let spacing_px = spacing * short_side;
                        let count = ((stroke.length() / spacing_px).floor() as usize + 1)
                            .clamp(2, options.max_spacing_points);
                        runs.push((count, Some(spacing)));
                    }
                    for (count, spacing) in runs {
                        let points = stroke.sample(count);
                        let labels = vec![1.0f32; count];
                        let (output, decode_ms) = timed_decode(
                            &mut decoders,
                            &context,
                            &points,
                            &labels,
                            None,
                            &mut decode_ms_by_count,
                        )?;
                        sweep.push(json!({
                            "direction": direction_name,
                            "direction_vector": direction,
                            "length_of_inner_chord": fraction,
                            "stroke": stroke_json(&frame, &stroke),
                            "count": count,
                            "spacing_of_short_side": spacing,
                            "spacing_px": stroke.length() / (count - 1) as f32,
                            "decode_ms": decode_ms,
                            "result": decode_summary(&output, &frame, side, &reference)?,
                        }));
                    }
                }
            }

            let principal_chord = region.inner_chord(click, principal, margin);
            let first = FirstInteractions {
                click,
                stroke: Stroke::along_chord(
                    click,
                    principal,
                    principal_chord,
                    options.primary_length,
                ),
                full_stroke: Stroke::along_chord(click, principal, principal_chord, 1.0),
            };
            let mask_prefix = options
                .mask_dir
                .as_ref()
                .map(|d| d.join(format!("{stem}__{}", set_entry.name)));
            let mut sequences = json!({});
            for sequence in &CORRECTION_SEQUENCES {
                sequences[sequence.name] = simulate_corrections(
                    &mut decoders,
                    &context,
                    sequence,
                    &first,
                    mask_prefix.as_deref(),
                    &mut decode_ms_by_count,
                )?;
            }
            let strokes = &sequences["stroke_then_strokes"];
            println!(
                "[strokes] {stem} {}: click {:.3}, stroke {:.3} -> {:.3} after {} interactions, clicks -> {:.3}",
                set_entry.name,
                click_summary["best_score_iou"].as_f64().unwrap_or(0.0),
                strokes["steps"][0]["reference_iou"].as_f64().unwrap_or(0.0),
                strokes["final_reference_iou"].as_f64().unwrap_or(0.0),
                strokes["steps"].as_array().map_or(0, Vec::len),
                sequences["click_then_clicks"]["final_reference_iou"]
                    .as_f64()
                    .unwrap_or(0.0),
            );
            image_report["prompts"].as_array_mut().unwrap().push(json!({
                "name": set_entry.name,
                "point": [x, y],
                "intent": target["intent"],
                "reference": reference_name,
                "reference_area_fraction": mask_fraction(&reference),
                "start_distance_px": start_distance,
                "margin_px": margin,
                "principal_direction": principal,
                "principal_inner_chord_px": [principal_chord.0, principal_chord.1],
                "click": { "decode_ms": click_ms, "result": click_summary },
                "sweep": sweep,
                "sequences": sequences,
            }));
        }

        report["images"].as_array_mut().unwrap().push(image_report);
        write_report(&options.out, &report)?;
        first_image = false;
    }
    let mut by_count = json!({});
    for (count, samples) in &decode_ms_by_count {
        by_count[count.to_string()] = latency_summary(samples);
    }
    report["decode_ms_by_point_count"] = by_count;
    if decoders.per_point_count.is_some() {
        report["decoder_sessions_per_point_count"] = json!({
            "sessions": decoders.load_ms.len(),
            "load_ms": decoders
                .load_ms
                .iter()
                .map(|(count, ms)| (count.to_string(), json!(ms)))
                .collect::<serde_json::Map<_, _>>(),
        });
    }
    report["memory_at_end"] = process_memory::sample();
    write_report(&options.out, &report)
}

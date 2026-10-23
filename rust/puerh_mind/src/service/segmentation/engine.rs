//! Segmentation engine: the loaded model set and the segmentation sessions.
//!
//! The engine owns the models that a load request loads and one image embedding per session. It
//! runs the models only through the backend traits ([`super::backend`]), so it is the same on
//! every platform. Load and unload take the write lock of the model set; start and decode hold its
//! read lock for the whole model run, so an unload waits for running calls and then ends the
//! sessions. Each model has its own mutex: one call per model at a time.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use tracing::info;

use super::backend::{SegmentationBackendFactory, SegmentationModel, SegmentationModelRole};
use super::model_set::{
    HIGH_RES_FEATS_0_SHAPE, HIGH_RES_FEATS_1_SHAPE, IMAGE_EMBED_SHAPE, MASK_INPUT_SHAPE,
    MAX_IMAGE_EDGE, MODEL_SET_ID,
};
use super::sam2::{self, DecodedMasks, SegmentationPrompt};
use super::tensor::TensorF32;

/// Largest number of sessions at one time. The tool keeps one session; a few more cover a host
/// that starts the next image before it ends the previous one.
pub const MAX_SESSIONS: usize = 4;

/// Failure of an engine operation. The text is the real error for the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentationError {
    /// The models are not loaded, or a model file does not load.
    ModelNotReady(String),
    /// A request value fails validation.
    InvalidInput(String),
    /// The named session does not exist.
    SessionNotFound(String),
    /// A model run failed.
    Inference(String),
}

impl fmt::Display for SegmentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotReady(message)
            | Self::InvalidInput(message)
            | Self::SessionNotFound(message)
            | Self::Inference(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for SegmentationError {}

/// An RGB8 image of a start request: row-major, 3 bytes per pixel.
#[derive(Debug, Clone, Copy)]
pub struct RgbImageView<'a> {
    pub width: u32,
    pub height: u32,
    pub rgb8: &'a [u8],
}

/// Residency of one loaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelResidency {
    pub role: SegmentationModelRole,
    pub device: String,
    pub load_ms: u64,
}

/// Engine state for status reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentationStatus {
    pub loaded: bool,
    pub model_set_id: &'static str,
    pub model_directory: String,
    pub device: String,
    pub backend: String,
    pub models: Vec<ModelResidency>,
    pub session_count: usize,
}

/// A started session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartedSession {
    pub session_id: String,
    pub encoder_ms: u64,
}

/// The masks of one decode request.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeResult {
    pub masks: DecodedMasks,
    pub decoder_ms: u64,
}

struct LoadedModel {
    model: Mutex<Box<dyn SegmentationModel>>,
    residency: ModelResidency,
}

impl LoadedModel {
    fn run(&self, inputs: &[&TensorF32]) -> Result<Vec<TensorF32>, SegmentationError> {
        let role = self.residency.role.as_str();
        let mut model = self.model.lock().map_err(|_| {
            SegmentationError::Inference(format!("segmentation {role} lock poisoned"))
        })?;
        model
            .run(inputs)
            .map_err(|err| SegmentationError::Inference(format!("{err:#}")))
    }
}

struct LoadedModelSet {
    directory: PathBuf,
    device: String,
    backend: String,
    encoder: LoadedModel,
    decoder: LoadedModel,
}

/// The encoder outputs of one session image.
struct ImageEmbedding {
    image_embed: TensorF32,
    high_res_feats_0: TensorF32,
    high_res_feats_1: TensorF32,
}

/// See the module documentation.
pub struct SegmentationEngine {
    factory: Arc<dyn SegmentationBackendFactory>,
    models: RwLock<Option<LoadedModelSet>>,
    sessions: Mutex<HashMap<String, Arc<ImageEmbedding>>>,
}

fn poisoned(what: &str) -> SegmentationError {
    SegmentationError::Inference(format!("segmentation {what} lock poisoned"))
}

fn not_loaded() -> SegmentationError {
    SegmentationError::ModelNotReady("segmentation models are not loaded".to_string())
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

impl SegmentationEngine {
    /// An engine without loaded models. @p factory makes the platform backend of each load.
    pub fn new(factory: Arc<dyn SegmentationBackendFactory>) -> Self {
        Self {
            factory,
            models: RwLock::new(None),
            sessions: Mutex::new(HashMap::new()),
        }
    }

    /// Loads the encoder and the decoder from @p directory on @p device.
    ///
    /// The same directory and device again keep the loaded models and the sessions. Another
    /// directory or device ends all sessions and unloads the old models before it loads. A failure
    /// leaves no model loaded and returns the backend error.
    pub fn load(&self, directory: &Path, device: &str) -> Result<(), SegmentationError> {
        if directory.as_os_str().is_empty() || !directory.is_absolute() {
            return Err(SegmentationError::InvalidInput(format!(
                "segmentation model directory {:?} is not an absolute path",
                directory
            )));
        }
        let backend = self
            .factory
            .create(device)
            .map_err(|err| SegmentationError::InvalidInput(err.to_string()))?;
        let device_label = backend.device_label(SegmentationModelRole::Encoder);
        let mut models = self.models.write().map_err(|_| poisoned("model set"))?;
        if let Some(loaded) = models.as_ref()
            && loaded.directory == directory
            && loaded.device == device_label
            && loaded.backend == backend.backend_name()
        {
            return Ok(());
        }
        self.sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .clear();
        *models = None;

        let load = |role: SegmentationModelRole| -> Result<LoadedModel, SegmentationError> {
            let started = Instant::now();
            let model = backend
                .load(directory, role)
                .map_err(|err| SegmentationError::ModelNotReady(format!("{err:#}")))?;
            let residency = ModelResidency {
                role,
                device: backend.device_label(role),
                load_ms: elapsed_ms(started),
            };
            info!(
                "loaded segmentation {} on {} in {} ms",
                role.as_str(),
                residency.device,
                residency.load_ms
            );
            Ok(LoadedModel {
                model: Mutex::new(model),
                residency,
            })
        };
        let encoder = load(SegmentationModelRole::Encoder)?;
        let decoder = load(SegmentationModelRole::Decoder)?;
        *models = Some(LoadedModelSet {
            directory: directory.to_path_buf(),
            device: device_label,
            backend: backend.backend_name().to_string(),
            encoder,
            decoder,
        });
        Ok(())
    }

    /// Ends all sessions and unloads the models. Waits for running model calls. Idempotent.
    pub fn unload(&self) -> Result<(), SegmentationError> {
        let mut models = self.models.write().map_err(|_| poisoned("model set"))?;
        self.sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .clear();
        if models.take().is_some() {
            info!("unloaded segmentation models");
        }
        Ok(())
    }

    pub fn status(&self) -> Result<SegmentationStatus, SegmentationError> {
        let models = self.models.read().map_err(|_| poisoned("model set"))?;
        let session_count = self.sessions.lock().map_err(|_| poisoned("session"))?.len();
        Ok(match models.as_ref() {
            Some(loaded) => SegmentationStatus {
                loaded: true,
                model_set_id: MODEL_SET_ID,
                model_directory: loaded.directory.display().to_string(),
                device: loaded.device.clone(),
                backend: loaded.backend.clone(),
                models: vec![
                    loaded.encoder.residency.clone(),
                    loaded.decoder.residency.clone(),
                ],
                session_count,
            },
            None => SegmentationStatus {
                loaded: false,
                model_set_id: MODEL_SET_ID,
                model_directory: String::new(),
                device: String::new(),
                backend: String::new(),
                models: Vec::new(),
                session_count,
            },
        })
    }

    fn check_session_capacity(&self) -> Result<(), SegmentationError> {
        let count = self.sessions.lock().map_err(|_| poisoned("session"))?.len();
        if count >= MAX_SESSIONS {
            return Err(SegmentationError::InvalidInput(format!(
                "{count} segmentation sessions are open; end a session before starting another (limit {MAX_SESSIONS})"
            )));
        }
        Ok(())
    }

    /// Runs the encoder once on @p image and keeps its outputs as a new session.
    pub fn start_session(
        &self,
        image: RgbImageView<'_>,
    ) -> Result<StartedSession, SegmentationError> {
        let (width, height) = (image.width, image.height);
        if width == 0 || height == 0 || width.max(height) > MAX_IMAGE_EDGE {
            return Err(SegmentationError::InvalidInput(format!(
                "segmentation image {width}x{height} must have a long edge of 1 to {MAX_IMAGE_EDGE} px"
            )));
        }
        let expected = width as usize * height as usize * 3;
        if image.rgb8.len() != expected {
            return Err(SegmentationError::InvalidInput(format!(
                "segmentation image {width}x{height} needs {expected} RGB8 bytes, got {}",
                image.rgb8.len()
            )));
        }
        let models = self.models.read().map_err(|_| poisoned("model set"))?;
        let loaded = models.as_ref().ok_or_else(not_loaded)?;
        self.check_session_capacity()?;

        let started = Instant::now();
        let pixels = sam2::preprocess_image(image.rgb8, width as usize, height as usize)
            .map_err(|err| SegmentationError::InvalidInput(format!("{err:#}")))?;
        let mut outputs = loaded.encoder.run(&[&pixels])?.into_iter();
        let mut next_output =
            |name: &str, shape: &[usize]| -> Result<TensorF32, SegmentationError> {
                let tensor = outputs.next().ok_or_else(|| {
                    SegmentationError::Inference(format!("the encoder returned no {name}"))
                })?;
                tensor
                    .expect_shape(name, shape)
                    .map_err(|err| SegmentationError::Inference(format!("{err:#}")))?;
                Ok(tensor)
            };
        let embedding = ImageEmbedding {
            image_embed: next_output("image_embed", &IMAGE_EMBED_SHAPE)?,
            high_res_feats_0: next_output("high_res_feats_0", &HIGH_RES_FEATS_0_SHAPE)?,
            high_res_feats_1: next_output("high_res_feats_1", &HIGH_RES_FEATS_1_SHAPE)?,
        };
        let encoder_ms = elapsed_ms(started);

        self.check_session_capacity()?;
        let session_id = uuid::Uuid::new_v4().to_string();
        self.sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .insert(session_id.clone(), Arc::new(embedding));
        info!("started segmentation session {session_id} ({width}x{height}) in {encoder_ms} ms");
        Ok(StartedSession {
            session_id,
            encoder_ms,
        })
    }

    /// Runs the decoder once with @p prompt on the embedding of @p session_id and selects the
    /// masks (section 2.4 of the SM3.A plan).
    pub fn decode(
        &self,
        session_id: &str,
        prompt: SegmentationPrompt,
    ) -> Result<DecodeResult, SegmentationError> {
        sam2::validate_prompt(&prompt).map_err(SegmentationError::InvalidInput)?;
        let models = self.models.read().map_err(|_| poisoned("model set"))?;
        let loaded = models.as_ref().ok_or_else(not_loaded)?;
        let embedding = self
            .sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .get(session_id)
            .cloned()
            .ok_or_else(|| {
                SegmentationError::SessionNotFound(format!(
                    "segmentation session {session_id:?} does not exist"
                ))
            })?;

        let started = Instant::now();
        let is_click = prompt.is_click();
        let (coords, labels) = sam2::model_prompt(&prompt);
        let point_count = labels.len();
        let has_mask_input = !prompt.mask_input.is_empty();
        let tensor = |shape: Vec<usize>, data: Vec<f32>| {
            TensorF32::new(shape, data)
                .map_err(|err| SegmentationError::Inference(format!("{err:#}")))
        };
        let point_coords = tensor(vec![1, point_count, 2], coords)?;
        let point_labels = tensor(vec![1, point_count], labels)?;
        let mask_input = if has_mask_input {
            tensor(MASK_INPUT_SHAPE.to_vec(), prompt.mask_input)?
        } else {
            TensorF32::zeros(MASK_INPUT_SHAPE.to_vec())
        };
        let has_mask_input = tensor(vec![1], vec![if has_mask_input { 1.0 } else { 0.0 }])?;
        let outputs = loaded.decoder.run(&[
            &embedding.image_embed,
            &embedding.high_res_feats_0,
            &embedding.high_res_feats_1,
            &point_coords,
            &point_labels,
            &mask_input,
            &has_mask_input,
        ])?;
        let [logits, scores] = <[TensorF32; 2]>::try_from(outputs).map_err(|outputs| {
            SegmentationError::Inference(format!(
                "the decoder returned {} outputs, expected 2",
                outputs.len()
            ))
        })?;
        let masks = sam2::select_masks(&logits, &scores, is_click)
            .map_err(|err| SegmentationError::Inference(format!("{err:#}")))?;
        Ok(DecodeResult {
            masks,
            decoder_ms: elapsed_ms(started),
        })
    }

    /// Ends @p session_id. Returns false when the session does not exist.
    pub fn end_session(&self, session_id: &str) -> Result<bool, SegmentationError> {
        Ok(self
            .sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .remove(session_id)
            .is_some())
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    //! A fake backend for engine and server tests: no model files, counted calls.

    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use anyhow::{Result, bail};

    use super::super::backend::{
        SegmentationBackend, SegmentationBackendFactory, SegmentationModel, SegmentationModelRole,
        UnsupportedDevice,
    };
    use super::super::model_set::{
        HIGH_RES_FEATS_0_SHAPE, HIGH_RES_FEATS_1_SHAPE, IMAGE_EMBED_SHAPE, LOGIT_SIZE,
        MASK_LOGITS_SHAPE, MASK_SCORES_SHAPE,
    };
    use super::super::tensor::TensorF32;

    #[derive(Default)]
    pub struct FakeCalls {
        pub loads: AtomicUsize,
        pub encoder_runs: AtomicUsize,
        /// Inputs of each decoder run.
        pub decoder_inputs: Mutex<Vec<Vec<TensorF32>>>,
    }

    pub struct FakeFactory {
        pub calls: Arc<FakeCalls>,
        pub failing_role: Option<SegmentationModelRole>,
    }

    impl FakeFactory {
        pub fn new() -> Self {
            Self {
                calls: Arc::new(FakeCalls::default()),
                failing_role: None,
            }
        }
    }

    impl SegmentationBackendFactory for FakeFactory {
        fn create(&self, device: &str) -> Result<Box<dyn SegmentationBackend>, UnsupportedDevice> {
            if device != "fake" && device != "fake:other" {
                return Err(UnsupportedDevice(format!(
                    "fake device {device:?} is not supported"
                )));
            }
            Ok(Box::new(FakeBackend {
                device: device.to_string(),
                calls: self.calls.clone(),
                failing_role: self.failing_role,
            }))
        }
    }

    struct FakeBackend {
        device: String,
        calls: Arc<FakeCalls>,
        failing_role: Option<SegmentationModelRole>,
    }

    impl SegmentationBackend for FakeBackend {
        fn backend_name(&self) -> &str {
            "fake"
        }

        fn device_label(&self, _role: SegmentationModelRole) -> String {
            self.device.clone()
        }

        fn load(
            &self,
            _directory: &Path,
            role: SegmentationModelRole,
        ) -> Result<Box<dyn SegmentationModel>> {
            self.calls.loads.fetch_add(1, Ordering::SeqCst);
            if self.failing_role == Some(role) {
                bail!("fake {} file is damaged", role.as_str());
            }
            Ok(Box::new(FakeModel {
                role,
                calls: self.calls.clone(),
            }))
        }
    }

    struct FakeModel {
        role: SegmentationModelRole,
        calls: Arc<FakeCalls>,
    }

    /// Token t of the fake decoder is positive on the first `FAKE_TOKEN_AREAS[t]` logits.
    pub const FAKE_TOKEN_AREAS: [usize; 4] = [3000, 100, 900, 400];
    pub const FAKE_TOKEN_SCORES: [f32; 4] = [0.5, 0.6, 0.9, 0.7];

    impl SegmentationModel for FakeModel {
        fn run(&mut self, inputs: &[&TensorF32]) -> Result<Vec<TensorF32>> {
            match self.role {
                SegmentationModelRole::Encoder => {
                    let run = self.calls.encoder_runs.fetch_add(1, Ordering::SeqCst) + 1;
                    let value = run as f32;
                    Ok(vec![
                        TensorF32::new(
                            IMAGE_EMBED_SHAPE.to_vec(),
                            vec![value; IMAGE_EMBED_SHAPE.iter().product()],
                        )?,
                        TensorF32::new(
                            HIGH_RES_FEATS_0_SHAPE.to_vec(),
                            vec![value; HIGH_RES_FEATS_0_SHAPE.iter().product()],
                        )?,
                        TensorF32::new(
                            HIGH_RES_FEATS_1_SHAPE.to_vec(),
                            vec![value; HIGH_RES_FEATS_1_SHAPE.iter().product()],
                        )?,
                    ])
                }
                SegmentationModelRole::Decoder => {
                    self.calls
                        .decoder_inputs
                        .lock()
                        .expect("decoder inputs")
                        .push(inputs.iter().map(|tensor| (*tensor).clone()).collect());
                    let plane = LOGIT_SIZE * LOGIT_SIZE;
                    let mut logits = vec![-4.0f32; 4 * plane];
                    for (token, area) in FAKE_TOKEN_AREAS.iter().enumerate() {
                        logits[token * plane..token * plane + area].fill(4.0);
                    }
                    Ok(vec![
                        TensorF32::new(MASK_LOGITS_SHAPE.to_vec(), logits)?,
                        TensorF32::new(MASK_SCORES_SHAPE.to_vec(), FAKE_TOKEN_SCORES.to_vec())?,
                    ])
                }
            }
        }
    }

    pub fn model_directory(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("alcedo-segmentation-fake-{name}"))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::super::model_set::LOGIT_SIZE;
    use super::super::sam2::{PointLabel, PromptBox, PromptPoint};
    use super::test_support::{FAKE_TOKEN_AREAS, FakeFactory, model_directory};
    use super::*;

    fn engine_with(
        factory: FakeFactory,
    ) -> (SegmentationEngine, Arc<super::test_support::FakeCalls>) {
        let calls = factory.calls.clone();
        (SegmentationEngine::new(Arc::new(factory)), calls)
    }

    fn gray_image(width: u32, height: u32) -> Vec<u8> {
        vec![100u8; width as usize * height as usize * 3]
    }

    fn start(engine: &SegmentationEngine) -> Result<StartedSession, SegmentationError> {
        let pixels = gray_image(64, 48);
        engine.start_session(RgbImageView {
            width: 64,
            height: 48,
            rgb8: &pixels,
        })
    }

    fn click(x: f32, y: f32) -> SegmentationPrompt {
        SegmentationPrompt {
            points: vec![PromptPoint {
                x,
                y,
                label: PointLabel::Add,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn start_session_before_load_reports_model_not_ready() {
        let (engine, calls) = engine_with(FakeFactory::new());
        assert!(matches!(
            start(&engine),
            Err(SegmentationError::ModelNotReady(_))
        ));
        assert_eq!(engine.status().expect("status").session_count, 0);
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn load_failure_leaves_no_models_and_reports_backend_error() {
        let mut factory = FakeFactory::new();
        factory.failing_role = Some(SegmentationModelRole::Decoder);
        let (engine, _calls) = engine_with(factory);
        let err = engine
            .load(&model_directory("a"), "fake")
            .expect_err("decoder load fails");
        assert!(
            matches!(&err, SegmentationError::ModelNotReady(message) if message.contains("fake decoder file is damaged")),
            "{err}"
        );
        assert!(!engine.status().expect("status").loaded);
    }

    #[test]
    fn load_rejects_unsupported_device_and_relative_directory() {
        let (engine, calls) = engine_with(FakeFactory::new());
        assert!(matches!(
            engine.load(&model_directory("a"), "directml"),
            Err(SegmentationError::InvalidInput(_))
        ));
        assert!(matches!(
            engine.load(Path::new("relative/models"), "fake"),
            Err(SegmentationError::InvalidInput(_))
        ));
        assert_eq!(calls.loads.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn start_session_runs_encoder_once_and_keeps_its_outputs() {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        let session = start(&engine).expect("start");
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 1);
        let status = engine.status().expect("status");
        assert!(status.loaded);
        assert_eq!(status.session_count, 1);
        assert_eq!(status.models.len(), 2);
        assert_eq!(status.model_set_id, MODEL_SET_ID);

        engine
            .decode(&session.session_id, click(0.5, 0.5))
            .expect("decode");
        let inputs = calls.decoder_inputs.lock().expect("inputs");
        assert_eq!(inputs.len(), 1);
        // The fake encoder writes its run number (1) into every output value.
        assert_eq!(inputs[0][0].shape(), &IMAGE_EMBED_SHAPE);
        assert!(inputs[0][0].data().iter().all(|value| *value == 1.0));
        assert!(inputs[0][1].data().iter().all(|value| *value == 1.0));
        assert!(inputs[0][2].data().iter().all(|value| *value == 1.0));
    }

    #[test]
    fn start_session_rejects_invalid_image_without_running_encoder() {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        let short = vec![0u8; 10];
        let long_edge = gray_image(4097, 1);
        let cases = [
            RgbImageView {
                width: 64,
                height: 48,
                rgb8: &short,
            },
            RgbImageView {
                width: 0,
                height: 48,
                rgb8: &short,
            },
            RgbImageView {
                width: 4097,
                height: 1,
                rgb8: &long_edge,
            },
        ];
        for image in cases {
            assert!(matches!(
                engine.start_session(image),
                Err(SegmentationError::InvalidInput(_))
            ));
        }
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 0);
        assert_eq!(engine.status().expect("status").session_count, 0);
    }

    #[test]
    fn session_limit_rejects_fifth_session() {
        let (engine, _calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        for _ in 0..MAX_SESSIONS {
            start(&engine).expect("start within the limit");
        }
        assert!(matches!(
            start(&engine),
            Err(SegmentationError::InvalidInput(_))
        ));
        assert_eq!(engine.status().expect("status").session_count, MAX_SESSIONS);
    }

    #[test]
    fn same_load_request_keeps_models_and_sessions() {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        start(&engine).expect("start");
        engine
            .load(&model_directory("a"), "fake")
            .expect("same load");
        assert_eq!(calls.loads.load(Ordering::SeqCst), 2);
        assert_eq!(engine.status().expect("status").session_count, 1);
    }

    #[test]
    fn load_with_other_directory_ends_sessions() {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        start(&engine).expect("start");
        engine
            .load(&model_directory("b"), "fake")
            .expect("other directory");
        assert_eq!(calls.loads.load(Ordering::SeqCst), 4);
        assert_eq!(engine.status().expect("status").session_count, 0);
        engine
            .load(&model_directory("b"), "fake:other")
            .expect("other device");
        assert_eq!(calls.loads.load(Ordering::SeqCst), 6);
    }

    #[test]
    fn unload_ends_sessions_and_is_idempotent() {
        let (engine, _calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        let session = start(&engine).expect("start");
        engine.unload().expect("unload");
        engine.unload().expect("second unload");
        let status = engine.status().expect("status");
        assert!(!status.loaded);
        assert_eq!(status.session_count, 0);
        assert!(matches!(
            engine.decode(&session.session_id, click(0.5, 0.5)),
            Err(SegmentationError::ModelNotReady(_))
        ));
    }

    #[test]
    fn end_unknown_session_returns_false() {
        let (engine, _calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        let session = start(&engine).expect("start");
        assert!(!engine.end_session("missing").expect("end"));
        assert!(engine.end_session(&session.session_id).expect("end"));
        assert!(!engine.end_session(&session.session_id).expect("end again"));
    }

    #[test]
    fn decode_passes_embedding_and_mask_input_to_decoder() {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        let session = start(&engine).expect("start");
        let mut mask_input = vec![-1.0f32; LOGIT_SIZE * LOGIT_SIZE];
        mask_input[5] = 7.0;
        let prompt = SegmentationPrompt {
            points: vec![
                PromptPoint {
                    x: 0.25,
                    y: 0.5,
                    label: PointLabel::Add,
                },
                PromptPoint {
                    x: 0.75,
                    y: 0.5,
                    label: PointLabel::Subtract,
                },
            ],
            prompt_box: Some(PromptBox {
                x0: 0.1,
                y0: 0.2,
                x1: 0.9,
                y1: 0.8,
            }),
            mask_input: mask_input.clone(),
        };
        let result = engine.decode(&session.session_id, prompt).expect("decode");
        let inputs = calls.decoder_inputs.lock().expect("inputs");
        let inputs = &inputs[0];
        assert_eq!(inputs[3].shape(), &[1, 4, 2]);
        assert_eq!(inputs[4].data(), &[2.0, 3.0, 1.0, 0.0]);
        assert_eq!(inputs[5].shape(), &MASK_INPUT_SHAPE);
        assert_eq!(inputs[5].data(), mask_input.as_slice());
        assert_eq!(inputs[6].data(), &[1.0]);
        // A box selects one mask; the fake token 0 is stable, so it is selected.
        assert_eq!(result.masks.masks.len(), 1);
        assert_eq!(result.masks.masks[0].token, 0);
        assert_eq!(result.masks.masks[0].area as usize, FAKE_TOKEN_AREAS[0]);
    }

    #[test]
    fn decode_click_without_mask_input_returns_three_levels() {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        let session = start(&engine).expect("start");
        let result = engine
            .decode(&session.session_id, click(0.5, 0.5))
            .expect("decode");
        let tokens: Vec<u32> = result.masks.masks.iter().map(|level| level.token).collect();
        assert_eq!(tokens, vec![1, 3, 2]);
        assert_eq!(result.masks.selected_index, 2);
        let inputs = calls.decoder_inputs.lock().expect("inputs");
        assert_eq!(inputs[0][6].data(), &[0.0]);
        assert!(inputs[0][5].data().iter().all(|value| *value == 0.0));
    }

    #[test]
    fn decode_unknown_session_reports_session_not_found() {
        let (engine, _calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        assert!(matches!(
            engine.decode("missing", click(0.5, 0.5)),
            Err(SegmentationError::SessionNotFound(_))
        ));
        assert!(matches!(
            engine.decode("missing", SegmentationPrompt::default()),
            Err(SegmentationError::InvalidInput(_))
        ));
    }
}

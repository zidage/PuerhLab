//! Segmentation engine: the loaded model set, the segmentation sessions, and the image embedding
//! cache.
//!
//! The engine owns the models that a load request loads, one image embedding per session, and the
//! image embedding cache of the SM3.B plan (decisions B1, B2, B6). It runs the models only through
//! the backend traits ([`super::backend`]), so it is the same on every platform. Load and unload
//! take the write lock of the model set; start and decode hold its read lock for the whole model
//! run, so an unload waits for running calls and then ends the sessions and clears the cache. The
//! encoder and the decoder each have their own mutex: one call per model at a time. The engine
//! never holds the session lock and the cache lock at the same time.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

use tracing::info;

use super::backend::{
    DecoderPrompt, ImageFeatures, SegmentationBackendFactory, SegmentationDecoder,
    SegmentationEncoder, SegmentationModelRole,
};
use super::image_embedding_cache::ImageEmbeddingCache;
use super::model_set::{MASK_INPUT_SHAPE, MAX_IMAGE_EDGE, MODEL_SET_ID};
use super::sam2::{self, DecodedMasks, SegmentationPrompt};
use super::tensor::TensorF32;

/// Largest number of sessions at one time. The tool keeps one session; a few more cover a host
/// that starts the next image before it ends the previous one.
pub const MAX_SESSIONS: usize = 4;
/// Largest number of cached image embeddings (SM3.B decision B2; about 16 MB each).
pub const IMAGE_EMBEDDING_CACHE_CAPACITY: usize = 4;
/// Largest length of an image key in bytes.
pub const MAX_IMAGE_KEY_BYTES: usize = 256;

/// Failure of an engine operation. The text is the real error for the host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegmentationError {
    /// The models are not loaded, or a model file does not load.
    ModelNotReady(String),
    /// A request value fails validation.
    InvalidInput(String),
    /// The named session does not exist.
    SessionNotFound(String),
    /// A start request without pixels names an image key that the cache does not hold.
    ImageNotCached(String),
    /// A model run failed.
    Inference(String),
}

impl fmt::Display for SegmentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelNotReady(message)
            | Self::InvalidInput(message)
            | Self::SessionNotFound(message)
            | Self::ImageNotCached(message)
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

/// A start request (SM3.B plan section 2.1). At least one of `image` and `image_key` is set.
#[derive(Debug, Clone, Copy, Default)]
pub struct StartSessionRequest<'a> {
    /// The pixels; `None` starts from the cache only.
    pub image: Option<RgbImageView<'a>>,
    /// The opaque key of the image content; empty: the session is not cached.
    pub image_key: &'a str,
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
    pub cached_image_count: usize,
}

/// A started session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StartedSession {
    pub session_id: String,
    /// Time of the preprocessing and the encoder; 0 for a cache hit.
    pub encoder_ms: u64,
    /// True when the session uses a cached image embedding.
    pub from_cache: bool,
}

/// The masks of one decode request.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeResult {
    pub masks: DecodedMasks,
    pub decoder_ms: u64,
}

struct LoadedModelSet {
    directory: PathBuf,
    device: String,
    backend: String,
    models: Vec<ModelResidency>,
    encoder: Mutex<Box<dyn SegmentationEncoder>>,
    decoder: Mutex<Box<dyn SegmentationDecoder>>,
}

/// The encoder outputs of one image in the native form of the backend. The sessions and the cache
/// entry of one image share one value.
type ImageEmbedding = Arc<dyn ImageFeatures>;

/// See the module documentation.
pub struct SegmentationEngine {
    factory: Arc<dyn SegmentationBackendFactory>,
    models: RwLock<Option<LoadedModelSet>>,
    sessions: Mutex<HashMap<String, ImageEmbedding>>,
    cache: Mutex<ImageEmbeddingCache<ImageEmbedding>>,
}

fn poisoned(what: &str) -> SegmentationError {
    SegmentationError::Inference(format!("segmentation {what} lock poisoned"))
}

fn not_loaded() -> SegmentationError {
    SegmentationError::ModelNotReady("segmentation models are not loaded".to_string())
}

fn inference(err: anyhow::Error) -> SegmentationError {
    SegmentationError::Inference(format!("{err:#}"))
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis() as u64
}

/// Checks the image and the key of @p request before any model work.
fn validate_start_request(request: &StartSessionRequest<'_>) -> Result<(), SegmentationError> {
    if request.image_key.len() > MAX_IMAGE_KEY_BYTES {
        return Err(SegmentationError::InvalidInput(format!(
            "segmentation image key has {} bytes, more than {MAX_IMAGE_KEY_BYTES}",
            request.image_key.len()
        )));
    }
    let Some(image) = request.image else {
        if request.image_key.is_empty() {
            return Err(SegmentationError::InvalidInput(
                "a segmentation start request needs pixels or an image key".to_string(),
            ));
        }
        return Ok(());
    };
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
    Ok(())
}

impl SegmentationEngine {
    /// An engine without loaded models. @p factory makes the platform backend of each load.
    pub fn new(factory: Arc<dyn SegmentationBackendFactory>) -> Self {
        Self {
            factory,
            models: RwLock::new(None),
            sessions: Mutex::new(HashMap::new()),
            cache: Mutex::new(ImageEmbeddingCache::new(IMAGE_EMBEDDING_CACHE_CAPACITY)),
        }
    }

    /// Ends all sessions and clears the image embedding cache. The caller holds the write lock of
    /// the model set.
    fn clear_embeddings(&self) -> Result<(), SegmentationError> {
        self.sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .clear();
        self.cache.lock().map_err(|_| poisoned("cache"))?.clear();
        Ok(())
    }

    /// Loads the encoder and the decoder from @p directory on @p device.
    ///
    /// The same directory and device again keep the loaded models, the sessions, and the cache.
    /// Another directory or device ends all sessions, clears the cache, and unloads the old models
    /// before it loads. A failure leaves no model loaded and returns the backend error.
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
        self.clear_embeddings()?;
        *models = None;

        let not_ready = |err: anyhow::Error| SegmentationError::ModelNotReady(format!("{err:#}"));
        let residency = |role: SegmentationModelRole, started: Instant| {
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
            residency
        };
        let started = Instant::now();
        let encoder = backend.load_encoder(directory).map_err(not_ready)?;
        let encoder_residency = residency(SegmentationModelRole::Encoder, started);
        let started = Instant::now();
        let decoder = backend.load_decoder(directory).map_err(not_ready)?;
        let decoder_residency = residency(SegmentationModelRole::Decoder, started);
        *models = Some(LoadedModelSet {
            directory: directory.to_path_buf(),
            device: device_label,
            backend: backend.backend_name().to_string(),
            models: vec![encoder_residency, decoder_residency],
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
        });
        Ok(())
    }

    /// Ends all sessions, clears the cache, and unloads the models. Waits for running model
    /// calls. Idempotent.
    pub fn unload(&self) -> Result<(), SegmentationError> {
        let mut models = self.models.write().map_err(|_| poisoned("model set"))?;
        self.clear_embeddings()?;
        if models.take().is_some() {
            info!("unloaded segmentation models");
        }
        Ok(())
    }

    pub fn status(&self) -> Result<SegmentationStatus, SegmentationError> {
        let models = self.models.read().map_err(|_| poisoned("model set"))?;
        let session_count = self.sessions.lock().map_err(|_| poisoned("session"))?.len();
        let cached_image_count = self.cache.lock().map_err(|_| poisoned("cache"))?.len();
        Ok(match models.as_ref() {
            Some(loaded) => SegmentationStatus {
                loaded: true,
                model_set_id: MODEL_SET_ID,
                model_directory: loaded.directory.display().to_string(),
                device: loaded.device.clone(),
                backend: loaded.backend.clone(),
                models: loaded.models.clone(),
                session_count,
                cached_image_count,
            },
            None => SegmentationStatus {
                loaded: false,
                model_set_id: MODEL_SET_ID,
                model_directory: String::new(),
                device: String::new(),
                backend: String::new(),
                models: Vec::new(),
                session_count,
                cached_image_count,
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

    fn insert_session(&self, embedding: ImageEmbedding) -> Result<String, SegmentationError> {
        let session_id = uuid::Uuid::new_v4().to_string();
        self.sessions
            .lock()
            .map_err(|_| poisoned("session"))?
            .insert(session_id.clone(), embedding);
        Ok(session_id)
    }

    /// Starts a session (SM3.B plan section 2.1).
    ///
    /// A cached image key starts the session with the cached embedding and does not run the
    /// encoder. Otherwise the encoder runs once on the pixels, and a non-empty key stores the
    /// embedding in the cache. A request without pixels and with an unknown key fails with
    /// `ImageNotCached`. No session is kept after a failure.
    pub fn start_session(
        &self,
        request: StartSessionRequest<'_>,
    ) -> Result<StartedSession, SegmentationError> {
        validate_start_request(&request)?;
        let models = self.models.read().map_err(|_| poisoned("model set"))?;
        let loaded = models.as_ref().ok_or_else(not_loaded)?;
        self.check_session_capacity()?;

        let key = request.image_key;
        if !key.is_empty() {
            let cached = self.cache.lock().map_err(|_| poisoned("cache"))?.get(key);
            if let Some(embedding) = cached {
                let session_id = self.insert_session(embedding)?;
                info!("started segmentation session {session_id} from the cached image {key:?}");
                return Ok(StartedSession {
                    session_id,
                    encoder_ms: 0,
                    from_cache: true,
                });
            }
        }
        let Some(image) = request.image else {
            return Err(SegmentationError::ImageNotCached(format!(
                "the segmentation image {key:?} is not cached; send its pixels"
            )));
        };

        let started = Instant::now();
        let (width, height) = (image.width, image.height);
        let pixels = sam2::preprocess_image(image.rgb8, width as usize, height as usize)
            .map_err(|err| SegmentationError::InvalidInput(format!("{err:#}")))?;
        let embedding = loaded
            .encoder
            .lock()
            .map_err(|_| poisoned("encoder"))?
            .encode(&pixels)
            .map_err(inference)?;
        let encoder_ms = elapsed_ms(started);

        self.check_session_capacity()?;
        if !key.is_empty() {
            self.cache
                .lock()
                .map_err(|_| poisoned("cache"))?
                .insert(key.to_string(), embedding.clone());
        }
        let session_id = self.insert_session(embedding)?;
        info!("started segmentation session {session_id} ({width}x{height}) in {encoder_ms} ms");
        Ok(StartedSession {
            session_id,
            encoder_ms,
            from_cache: false,
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
        let tensor =
            |shape: Vec<usize>, data: Vec<f32>| TensorF32::new(shape, data).map_err(inference);
        let decoder_prompt = DecoderPrompt {
            point_coords: tensor(vec![1, point_count, 2], coords)?,
            point_labels: tensor(vec![1, point_count], labels)?,
            mask_input: if has_mask_input {
                tensor(MASK_INPUT_SHAPE.to_vec(), prompt.mask_input)?
            } else {
                TensorF32::zeros(MASK_INPUT_SHAPE.to_vec())
            },
            has_mask_input: tensor(vec![1], vec![if has_mask_input { 1.0 } else { 0.0 }])?,
        };
        let output = loaded
            .decoder
            .lock()
            .map_err(|_| poisoned("decoder"))?
            .decode(&embedding, &decoder_prompt)
            .map_err(inference)?;
        let masks = sam2::select_masks(&output.mask_logits, &output.iou_scores, is_click)
            .map_err(inference)?;
        Ok(DecodeResult {
            masks,
            decoder_ms: elapsed_ms(started),
        })
    }

    /// Ends @p session_id. Returns false when the session does not exist. The cache entry of the
    /// session image stays.
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
mod tests {
    use std::sync::atomic::Ordering;

    use super::super::model_set::LOGIT_SIZE;
    use super::super::sam2::{PointLabel, PromptBox, PromptPoint};
    use super::super::test_support::{FAKE_TOKEN_AREAS, FakeCalls, FakeFactory, model_directory};
    use super::*;

    fn engine_with(factory: FakeFactory) -> (SegmentationEngine, Arc<FakeCalls>) {
        let calls = factory.calls.clone();
        (SegmentationEngine::new(Arc::new(factory)), calls)
    }

    fn loaded_engine() -> (SegmentationEngine, Arc<FakeCalls>) {
        let (engine, calls) = engine_with(FakeFactory::new());
        engine.load(&model_directory("a"), "fake").expect("load");
        (engine, calls)
    }

    fn gray_image(width: u32, height: u32) -> Vec<u8> {
        vec![100u8; width as usize * height as usize * 3]
    }

    fn start_keyed(
        engine: &SegmentationEngine,
        image_key: &str,
    ) -> Result<StartedSession, SegmentationError> {
        let pixels = gray_image(64, 48);
        engine.start_session(StartSessionRequest {
            image: Some(RgbImageView {
                width: 64,
                height: 48,
                rgb8: &pixels,
            }),
            image_key,
        })
    }

    fn start(engine: &SegmentationEngine) -> Result<StartedSession, SegmentationError> {
        start_keyed(engine, "")
    }

    fn start_key_only(
        engine: &SegmentationEngine,
        image_key: &str,
    ) -> Result<StartedSession, SegmentationError> {
        engine.start_session(StartSessionRequest {
            image: None,
            image_key,
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
        let (engine, calls) = loaded_engine();
        let session = start(&engine).expect("start");
        assert!(!session.from_cache);
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 1);
        let status = engine.status().expect("status");
        assert!(status.loaded);
        assert_eq!(status.session_count, 1);
        assert_eq!(status.cached_image_count, 0);
        assert_eq!(status.models.len(), 2);
        assert_eq!(status.model_set_id, MODEL_SET_ID);

        engine
            .decode(&session.session_id, click(0.5, 0.5))
            .expect("decode");
        let decoder_calls = calls.decoder_calls.lock().expect("decoder calls");
        assert_eq!(decoder_calls.len(), 1);
        assert_eq!(decoder_calls[0].encoder_run(), 1);
    }

    #[test]
    fn start_session_rejects_invalid_image_without_running_encoder() {
        let (engine, calls) = loaded_engine();
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
                engine.start_session(StartSessionRequest {
                    image: Some(image),
                    image_key: "",
                }),
                Err(SegmentationError::InvalidInput(_))
            ));
        }
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 0);
        assert_eq!(engine.status().expect("status").session_count, 0);
    }

    #[test]
    fn start_rejects_invalid_key_and_partial_empty_image() {
        let (engine, calls) = loaded_engine();
        let long_key = "k".repeat(MAX_IMAGE_KEY_BYTES + 1);
        assert!(matches!(
            start_keyed(&engine, &long_key),
            Err(SegmentationError::InvalidInput(_))
        ));
        assert!(matches!(
            start_key_only(&engine, ""),
            Err(SegmentationError::InvalidInput(_))
        ));
        let bytes = vec![0u8; 12];
        assert!(matches!(
            engine.start_session(StartSessionRequest {
                image: Some(RgbImageView {
                    width: 0,
                    height: 0,
                    rgb8: &bytes,
                }),
                image_key: "k",
            }),
            Err(SegmentationError::InvalidInput(_))
        ));
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 0);
        let status = engine.status().expect("status");
        assert_eq!(status.session_count, 0);
        assert_eq!(status.cached_image_count, 0);
    }

    #[test]
    fn start_with_cached_key_skips_encoder_and_shares_embedding() {
        let (engine, calls) = loaded_engine();
        let first = start_keyed(&engine, "photo-a").expect("first start");
        let second = start_keyed(&engine, "photo-a").expect("second start");
        assert!(!first.from_cache);
        assert!(second.from_cache);
        assert_eq!(second.encoder_ms, 0);
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 1);
        let status = engine.status().expect("status");
        assert_eq!(status.session_count, 2);
        assert_eq!(status.cached_image_count, 1);

        engine
            .decode(&first.session_id, click(0.5, 0.5))
            .expect("decode first");
        engine
            .decode(&second.session_id, click(0.5, 0.5))
            .expect("decode second");
        let decoder_calls = calls.decoder_calls.lock().expect("decoder calls");
        assert!(Arc::ptr_eq(
            &decoder_calls[0].features,
            &decoder_calls[1].features
        ));
    }

    #[test]
    fn key_only_start_reports_image_not_cached_without_encoder() {
        let (engine, calls) = loaded_engine();
        assert!(matches!(
            start_key_only(&engine, "photo-a"),
            Err(SegmentationError::ImageNotCached(_))
        ));
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 0);
        assert_eq!(engine.status().expect("status").session_count, 0);
    }

    #[test]
    fn key_only_start_hits_after_session_end() {
        let (engine, calls) = loaded_engine();
        let first = start_keyed(&engine, "photo-a").expect("start");
        assert!(engine.end_session(&first.session_id).expect("end"));
        let cached = start_key_only(&engine, "photo-a").expect("cached start");
        assert!(cached.from_cache);
        assert_eq!(calls.encoder_runs.load(Ordering::SeqCst), 1);
        engine
            .decode(&cached.session_id, click(0.5, 0.5))
            .expect("decode");
        assert_eq!(
            calls.decoder_calls.lock().expect("decoder calls")[0].encoder_run(),
            1
        );
    }

    #[test]
    fn cache_evicts_fifth_key() {
        let (engine, calls) = loaded_engine();
        for index in 1..=IMAGE_EMBEDDING_CACHE_CAPACITY + 1 {
            let session = start_keyed(&engine, &format!("k{index}")).expect("start");
            engine.end_session(&session.session_id).expect("end");
        }
        assert_eq!(
            engine.status().expect("status").cached_image_count,
            IMAGE_EMBEDDING_CACHE_CAPACITY
        );
        assert!(matches!(
            start_key_only(&engine, "k1"),
            Err(SegmentationError::ImageNotCached(_))
        ));
        for index in 2..=IMAGE_EMBEDDING_CACHE_CAPACITY + 1 {
            let session = start_key_only(&engine, &format!("k{index}")).expect("cached start");
            assert!(session.from_cache);
            engine.end_session(&session.session_id).expect("end");
        }
        assert_eq!(
            calls.encoder_runs.load(Ordering::SeqCst),
            IMAGE_EMBEDDING_CACHE_CAPACITY + 1
        );
    }

    #[test]
    fn unload_and_model_change_clear_cached_images() {
        let (engine, _calls) = loaded_engine();
        start_keyed(&engine, "photo-a").expect("start");
        engine.unload().expect("unload");
        assert_eq!(engine.status().expect("status").cached_image_count, 0);
        engine.load(&model_directory("a"), "fake").expect("load");
        assert!(matches!(
            start_key_only(&engine, "photo-a"),
            Err(SegmentationError::ImageNotCached(_))
        ));

        start_keyed(&engine, "photo-b").expect("start");
        engine
            .load(&model_directory("a"), "fake")
            .expect("same load");
        assert_eq!(engine.status().expect("status").cached_image_count, 1);
        engine
            .load(&model_directory("b"), "fake")
            .expect("other directory");
        let status = engine.status().expect("status");
        assert_eq!(status.cached_image_count, 0);
        assert_eq!(status.session_count, 0);
        assert!(matches!(
            start_key_only(&engine, "photo-b"),
            Err(SegmentationError::ImageNotCached(_))
        ));
    }

    #[test]
    fn session_limit_rejects_fifth_session() {
        let (engine, _calls) = loaded_engine();
        for _ in 0..MAX_SESSIONS {
            start_keyed(&engine, "photo-a").expect("start within the limit");
        }
        assert!(matches!(
            start_keyed(&engine, "photo-a"),
            Err(SegmentationError::InvalidInput(_))
        ));
        assert_eq!(engine.status().expect("status").session_count, MAX_SESSIONS);
    }

    #[test]
    fn same_load_request_keeps_models_and_sessions() {
        let (engine, calls) = loaded_engine();
        start(&engine).expect("start");
        engine
            .load(&model_directory("a"), "fake")
            .expect("same load");
        assert_eq!(calls.loads.load(Ordering::SeqCst), 2);
        assert_eq!(engine.status().expect("status").session_count, 1);
    }

    #[test]
    fn load_with_other_directory_ends_sessions() {
        let (engine, calls) = loaded_engine();
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
        let (engine, _calls) = loaded_engine();
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
        let (engine, _calls) = loaded_engine();
        let session = start(&engine).expect("start");
        assert!(!engine.end_session("missing").expect("end"));
        assert!(engine.end_session(&session.session_id).expect("end"));
        assert!(!engine.end_session(&session.session_id).expect("end again"));
    }

    #[test]
    fn decode_passes_embedding_and_mask_input_to_decoder() {
        let (engine, calls) = loaded_engine();
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
        let decoder_calls = calls.decoder_calls.lock().expect("decoder calls");
        let call = &decoder_calls[0];
        assert_eq!(call.encoder_run(), 1);
        assert_eq!(call.prompt.point_coords.shape(), &[1, 4, 2]);
        assert_eq!(call.prompt.point_count(), 4);
        assert_eq!(call.prompt.point_labels.data(), &[2.0, 3.0, 1.0, 0.0]);
        assert_eq!(call.prompt.mask_input.shape(), &MASK_INPUT_SHAPE);
        assert_eq!(call.prompt.mask_input.data(), mask_input.as_slice());
        assert_eq!(call.prompt.has_mask_input.data(), &[1.0]);
        // A box selects one mask; the fake token 0 is stable, so it is selected.
        assert_eq!(result.masks.masks.len(), 1);
        assert_eq!(result.masks.masks[0].token, 0);
        assert_eq!(result.masks.masks[0].area as usize, FAKE_TOKEN_AREAS[0]);
    }

    #[test]
    fn decode_click_without_mask_input_returns_three_levels() {
        let (engine, calls) = loaded_engine();
        let session = start(&engine).expect("start");
        let result = engine
            .decode(&session.session_id, click(0.5, 0.5))
            .expect("decode");
        let tokens: Vec<u32> = result.masks.masks.iter().map(|level| level.token).collect();
        assert_eq!(tokens, vec![1, 3, 2]);
        assert_eq!(result.masks.selected_index, 2);
        let decoder_calls = calls.decoder_calls.lock().expect("decoder calls");
        assert_eq!(decoder_calls[0].prompt.has_mask_input.data(), &[0.0]);
        assert!(
            decoder_calls[0]
                .prompt
                .mask_input
                .data()
                .iter()
                .all(|value| *value == 0.0)
        );
    }

    #[test]
    fn decode_unknown_session_reports_session_not_found() {
        let (engine, _calls) = loaded_engine();
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

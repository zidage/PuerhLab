//! A fake backend for the engine and server tests: no model files, counted calls.

use std::any::Any;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{Result, bail};

use super::backend::{
    DecoderOutput, DecoderPrompt, ImageFeatures, SegmentationBackend, SegmentationBackendFactory,
    SegmentationDecoder, SegmentationEncoder, SegmentationModelRole, UnsupportedDevice,
    features_of,
};
use super::model_set::{LOGIT_SIZE, MASK_LOGITS_SHAPE, MASK_SCORES_SHAPE, PIXEL_VALUES_SHAPE};
use super::tensor::TensorF32;

/// The features of the fake encoder: the number of the encoder run that made them.
pub struct FakeImageFeatures {
    pub encoder_run: usize,
}

impl ImageFeatures for FakeImageFeatures {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// One fake decoder run: the features and the prompt that it received.
pub struct FakeDecoderCall {
    pub features: Arc<dyn ImageFeatures>,
    pub prompt: DecoderPrompt,
}

impl FakeDecoderCall {
    /// The encoder run of the features of this call.
    pub fn encoder_run(&self) -> usize {
        features_of::<FakeImageFeatures>(self.features.as_ref())
            .expect("fake features")
            .encoder_run
    }
}

#[derive(Default)]
pub struct FakeCalls {
    pub loads: AtomicUsize,
    pub encoder_runs: AtomicUsize,
    pub decoder_calls: Mutex<Vec<FakeDecoderCall>>,
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

impl FakeBackend {
    fn count_load(&self, role: SegmentationModelRole) -> Result<()> {
        self.calls.loads.fetch_add(1, Ordering::SeqCst);
        if self.failing_role == Some(role) {
            bail!("fake {} file is damaged", role.as_str());
        }
        Ok(())
    }
}

impl SegmentationBackend for FakeBackend {
    fn backend_name(&self) -> &str {
        "fake"
    }

    fn device_label(&self, _role: SegmentationModelRole) -> String {
        self.device.clone()
    }

    fn load_encoder(&self, _directory: &Path) -> Result<Box<dyn SegmentationEncoder>> {
        self.count_load(SegmentationModelRole::Encoder)?;
        Ok(Box::new(FakeEncoder {
            calls: self.calls.clone(),
        }))
    }

    fn load_decoder(&self, _directory: &Path) -> Result<Box<dyn SegmentationDecoder>> {
        self.count_load(SegmentationModelRole::Decoder)?;
        Ok(Box::new(FakeDecoder {
            calls: self.calls.clone(),
        }))
    }
}

struct FakeEncoder {
    calls: Arc<FakeCalls>,
}

impl SegmentationEncoder for FakeEncoder {
    fn encode(&mut self, pixel_values: &TensorF32) -> Result<Arc<dyn ImageFeatures>> {
        pixel_values.expect_shape("pixel_values", &PIXEL_VALUES_SHAPE)?;
        let encoder_run = self.calls.encoder_runs.fetch_add(1, Ordering::SeqCst) + 1;
        Ok(Arc::new(FakeImageFeatures { encoder_run }))
    }
}

/// Token t of the fake decoder is positive on the first `FAKE_TOKEN_AREAS[t]` logits.
pub const FAKE_TOKEN_AREAS: [usize; 4] = [3000, 100, 900, 400];
pub const FAKE_TOKEN_SCORES: [f32; 4] = [0.5, 0.6, 0.9, 0.7];

struct FakeDecoder {
    calls: Arc<FakeCalls>,
}

impl SegmentationDecoder for FakeDecoder {
    fn decode(
        &mut self,
        features: &Arc<dyn ImageFeatures>,
        prompt: &DecoderPrompt,
    ) -> Result<DecoderOutput> {
        features_of::<FakeImageFeatures>(features.as_ref())?;
        self.calls
            .decoder_calls
            .lock()
            .expect("decoder calls")
            .push(FakeDecoderCall {
                features: features.clone(),
                prompt: prompt.clone(),
            });
        let plane = LOGIT_SIZE * LOGIT_SIZE;
        let mut logits = vec![-4.0f32; 4 * plane];
        for (token, area) in FAKE_TOKEN_AREAS.iter().enumerate() {
            logits[token * plane..token * plane + area].fill(4.0);
        }
        Ok(DecoderOutput {
            mask_logits: TensorF32::new(MASK_LOGITS_SHAPE.to_vec(), logits)?,
            iou_scores: TensorF32::new(MASK_SCORES_SHAPE.to_vec(), FAKE_TOKEN_SCORES.to_vec())?,
        })
    }
}

pub fn model_directory(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("alcedo-segmentation-fake-{name}"))
}

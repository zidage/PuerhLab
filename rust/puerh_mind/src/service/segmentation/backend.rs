//! Platform interface of the segmentation models.
//!
//! The engine runs the models only through these traits. Each platform adapter implements them:
//! [`super::ort_backend`] (ONNX Runtime; DirectML on Windows) and `super::coreml_backend`
//! (native Core ML on macOS). An adapter knows the file and tensor names of the model
//! specification ([`super::model_set`]); it does not know sessions, prompts, or mask rules.
//!
//! The encoder output stays in the native form of the backend ([`ImageFeatures`], SM3.B plan
//! decision B4), so a decoder call does not copy the 16 MB of features.

use std::any::Any;
use std::fmt;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Result, anyhow};

use super::tensor::TensorF32;

/// The model of the set that a backend loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentationModelRole {
    /// SAM 2.1 tiny image encoder.
    Encoder,
    /// SAM 2.1 tiny prompt encoder and mask decoder.
    Decoder,
}

impl SegmentationModelRole {
    pub const ALL: [SegmentationModelRole; 2] = [Self::Encoder, Self::Decoder];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Encoder => "encoder",
            Self::Decoder => "decoder",
        }
    }
}

/// The encoder outputs of one image (`image_embed`, `high_res_feats_0`, `high_res_feats_1`) in the
/// native form of the backend that made them. The values do not change after the encoder run.
/// Sessions and the image embedding cache share one value through `Arc`.
pub trait ImageFeatures: Send + Sync {
    /// The adapter reads its own feature type through this method.
    fn as_any(&self) -> &dyn Any;
}

/// Reads the features of @p features as the type `T` of an adapter. Features of another backend
/// are an error.
pub fn features_of<T: 'static>(features: &dyn ImageFeatures) -> Result<&T> {
    features
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| anyhow!("the image features were made by another segmentation backend"))
}

/// A loaded image encoder.
pub trait SegmentationEncoder: Send {
    /// Runs the encoder once on @p pixel_values (shape [1, 3, 1024, 1024]) and returns its
    /// outputs in native form. The adapter checks the output shapes of the model specification.
    fn encode(&mut self, pixel_values: &TensorF32) -> Result<Arc<dyn ImageFeatures>>;
}

/// The prompt inputs of one decoder run, in the order and shapes of the model specification.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderPrompt {
    /// [1, N, 2], coordinates times 1024.
    pub point_coords: TensorF32,
    /// [1, N].
    pub point_labels: TensorF32,
    /// [1, 1, 256, 256].
    pub mask_input: TensorF32,
    /// [1].
    pub has_mask_input: TensorF32,
}

impl DecoderPrompt {
    /// The decoder point count N (box corners included).
    pub fn point_count(&self) -> usize {
        self.point_labels.data().len()
    }
}

/// The outputs of one decoder run.
#[derive(Debug, Clone, PartialEq)]
pub struct DecoderOutput {
    /// [1, 4, 256, 256].
    pub mask_logits: TensorF32,
    /// [1, 4].
    pub iou_scores: TensorF32,
}

/// A loaded prompt decoder.
pub trait SegmentationDecoder: Send {
    /// Runs the decoder once on @p features (made by the encoder of the same backend) with
    /// @p prompt. The adapter can keep a strong reference to @p features between calls, for
    /// example to keep a device copy bound.
    fn decode(
        &mut self,
        features: &Arc<dyn ImageFeatures>,
        prompt: &DecoderPrompt,
    ) -> Result<DecoderOutput>;
}

/// The models of one platform and one device.
pub trait SegmentationBackend: Send + Sync {
    /// Short name of the runtime and device class, for example `onnxruntime-directml`.
    fn backend_name(&self) -> &str;
    /// Device label of @p role for status reports, for example `coreml-native:cpuonly`.
    fn device_label(&self, role: SegmentationModelRole) -> String;
    /// Loads the encoder from @p directory. The error names the file and carries the runtime
    /// message. The backend never loads a model on another device after an error.
    fn load_encoder(&self, directory: &Path) -> Result<Box<dyn SegmentationEncoder>>;
    /// Loads the decoder from @p directory, with the same error rules as the encoder.
    fn load_decoder(&self, directory: &Path) -> Result<Box<dyn SegmentationDecoder>>;
}

/// A device value that the platform does not support.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsupportedDevice(pub String);

impl fmt::Display for UnsupportedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnsupportedDevice {}

/// Makes the backend of the platform for a device value of the load request.
pub trait SegmentationBackendFactory: Send + Sync {
    /// @p device is `""` for the platform default or one of the platform device values.
    fn create(&self, device: &str) -> Result<Box<dyn SegmentationBackend>, UnsupportedDevice>;
}

//! Platform interface of the segmentation models.
//!
//! The engine runs the models only through these traits. Each platform adapter implements them:
//! [`super::ort_backend`] (ONNX Runtime; DirectML on Windows) and `super::coreml_backend`
//! (native Core ML on macOS). An adapter knows the file and tensor names of the model
//! specification ([`super::model_set`]); it does not know sessions, prompts, or mask rules.

use std::fmt;
use std::path::Path;

use anyhow::Result;

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

/// One loaded model.
pub trait SegmentationModel: Send {
    /// Runs the model once. @p inputs follow the input names of the model specification, in
    /// order. The result holds the outputs of the specification, in order. Every call copies
    /// what it needs; the inputs stay owned by the caller.
    fn run(&mut self, inputs: &[&TensorF32]) -> Result<Vec<TensorF32>>;
}

/// The models of one platform and one device.
pub trait SegmentationBackend: Send + Sync {
    /// Short name of the runtime and device class, for example `onnxruntime-directml`.
    fn backend_name(&self) -> &str;
    /// Device label of @p role for status reports, for example `coreml-native:cpuonly`.
    fn device_label(&self, role: SegmentationModelRole) -> String;
    /// Loads the model of @p role from @p directory. The error names the file and carries the
    /// runtime message. The backend never loads the model on another device after an error.
    fn load(
        &self,
        directory: &Path,
        role: SegmentationModelRole,
    ) -> Result<Box<dyn SegmentationModel>>;
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

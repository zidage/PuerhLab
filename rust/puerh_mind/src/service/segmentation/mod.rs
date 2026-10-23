//! Local AI Mask segmentation (master plan section 9.2.1, SM3.A).
//!
//! The engine ([`engine`]) owns the loaded SAM 2.1 tiny models and one image embedding per
//! session. It runs the models through the platform interface of [`backend`]:
//!
//! - Windows: [`ort_backend`], ONNX Runtime with DirectML (or the CPU for development);
//! - macOS: `coreml_backend`, compiled native Core ML models.
//!
//! [`sam2`] holds the host rules of the models (preprocessing, prompts, mask selection) as pure
//! functions. [`model_set`] names the files and tensors of the model set.

pub mod backend;
#[cfg(target_os = "macos")]
pub mod coreml_backend;
pub mod engine;
pub mod model_set;
pub mod ort_backend;
pub mod sam2;
pub mod tensor;

use std::sync::Arc;

use backend::SegmentationBackendFactory;

/// The backend factory of the platform: native Core ML on macOS, ONNX Runtime elsewhere.
pub fn platform_backend_factory() -> Arc<dyn SegmentationBackendFactory> {
    #[cfg(target_os = "macos")]
    {
        Arc::new(coreml_backend::CoreMlSegmentationBackendFactory)
    }
    #[cfg(not(target_os = "macos"))]
    {
        Arc::new(ort_backend::OrtSegmentationBackendFactory)
    }
}

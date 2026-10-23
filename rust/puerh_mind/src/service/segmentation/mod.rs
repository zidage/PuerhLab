//! Local AI Mask segmentation (master plan sections 9.2.1 and 9.2.2, SM3.A and SM3.B).
//!
//! The engine ([`engine`]) owns the loaded SAM 2.1 tiny models, one image embedding per session,
//! and the image embedding cache ([`image_embedding_cache`]). It runs the models through the
//! platform interface of [`backend`]:
//!
//! - Windows: [`ort_backend`] and [`ort_decoder`], ONNX Runtime with DirectML (or the CPU for
//!   development);
//! - macOS: `coreml_backend`, compiled native Core ML models.
//!
//! [`sam2`] holds the host rules of the models (preprocessing, prompts, mask selection) as pure
//! functions. [`model_set`] names the files and tensors of the model set.

pub mod backend;
#[cfg(target_os = "macos")]
pub mod coreml_backend;
pub mod engine;
pub mod image_embedding_cache;
pub mod model_set;
pub mod ort_backend;
pub mod ort_decoder;
pub mod sam2;
pub mod tensor;
#[cfg(test)]
pub(crate) mod test_support;

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

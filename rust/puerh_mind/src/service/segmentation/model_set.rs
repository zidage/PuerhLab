//! Files, tensor names, and tensor shapes of the AI Mask model set (master plan section 8.17.1).

use super::backend::SegmentationModelRole;

/// Identity of the model set and the directory name that SM9 and SM2 use.
pub const MODEL_SET_ID: &str = "alcedo-ai-mask-v1";
/// Side of the square encoder input. Prompt coordinates are normalized coordinates times this.
pub const IMAGE_SIZE: usize = 1024;
/// Side of the square decoder logits and of the mask input.
pub const LOGIT_SIZE: usize = 256;
/// Number of mask tokens of the decoder output.
pub const MASK_TOKEN_COUNT: usize = 4;
/// Largest decoder point count (box corners included); the range of the Core ML decoder.
pub const MAX_DECODER_POINTS: usize = 64;
/// Largest long edge of a session image.
pub const MAX_IMAGE_EDGE: u32 = 4096;

pub const PIXEL_VALUES_SHAPE: [usize; 4] = [1, 3, IMAGE_SIZE, IMAGE_SIZE];
pub const IMAGE_EMBED_SHAPE: [usize; 4] = [1, 256, 64, 64];
pub const HIGH_RES_FEATS_0_SHAPE: [usize; 4] = [1, 32, 256, 256];
pub const HIGH_RES_FEATS_1_SHAPE: [usize; 4] = [1, 64, 128, 128];
pub const MASK_INPUT_SHAPE: [usize; 4] = [1, 1, LOGIT_SIZE, LOGIT_SIZE];
pub const MASK_LOGITS_SHAPE: [usize; 4] = [1, MASK_TOKEN_COUNT, LOGIT_SIZE, LOGIT_SIZE];
pub const MASK_SCORES_SHAPE: [usize; 2] = [1, MASK_TOKEN_COUNT];

/// File and tensor names of one model. ONNX and Core ML models have the same input names; the
/// decoder output names differ.
#[derive(Debug)]
pub struct ModelSpec {
    pub role: SegmentationModelRole,
    pub onnx_file: &'static str,
    pub coreml_directory: &'static str,
    pub inputs: &'static [&'static str],
    pub onnx_outputs: &'static [&'static str],
    pub coreml_outputs: &'static [&'static str],
}

pub const ENCODER: ModelSpec = ModelSpec {
    role: SegmentationModelRole::Encoder,
    onnx_file: "point_encoder.onnx",
    coreml_directory: "point_encoder.mlmodelc",
    inputs: &["pixel_values"],
    onnx_outputs: &["image_embed", "high_res_feats_0", "high_res_feats_1"],
    coreml_outputs: &["image_embed", "high_res_feats_0", "high_res_feats_1"],
};

pub const DECODER: ModelSpec = ModelSpec {
    role: SegmentationModelRole::Decoder,
    onnx_file: "point_decoder.onnx",
    coreml_directory: "point_decoder.mlmodelc",
    inputs: &[
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1",
        "point_coords",
        "point_labels",
        "mask_input",
        "has_mask_input",
    ],
    onnx_outputs: &["mask_logits", "iou_scores"],
    coreml_outputs: &["masks", "iou_predictions"],
};

pub fn spec(role: SegmentationModelRole) -> &'static ModelSpec {
    match role {
        SegmentationModelRole::Encoder => &ENCODER,
        SegmentationModelRole::Decoder => &DECODER,
    }
}

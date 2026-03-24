#[derive(Debug)]
pub struct TextBatch {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
}

use candle_core::Tensor;

pub struct TextTensors {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
}

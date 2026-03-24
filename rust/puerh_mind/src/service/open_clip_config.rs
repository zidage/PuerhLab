use serde::Deserialize;

use std::fs;
use std::path::Path;

use anyhow::Result;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct OpenClipConfig {
    pub model_cfg: ModelConfig,
    pub preprocess_cfg: PreprocessConfig,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ModelConfig {
    pub embed_dim: usize,
    pub vision_cfg: VisionConfig,
    pub text_cfg: TextConfig,
    #[serde(default)]
    pub custom_text: bool,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct VisionConfig {
    pub timm_model_name: String,
    #[serde(default)]
    pub timm_model_pretrained: bool,
    #[serde(default)]
    pub timm_pool: Option<String>,
    #[serde(default)]
    pub timm_proj: Option<String>,
    #[serde(default)]
    pub timm_drop: f32,
    #[serde(default)]
    pub timm_drop_path: f32,
    pub image_size: usize,
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct TextConfig {
    pub context_length: usize,
    pub vocab_size: usize,
    pub width: usize,
    pub heads: usize,
    pub layers: usize,
    #[serde(default)]
    pub no_causal_mask: bool,
}


#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct PreprocessConfig {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub interpolation: String,
    pub resize_mode: String,
}

impl OpenClipConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        let config = serde_json::from_str(&content)?;
        Ok(config)
    }
}

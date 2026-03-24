use anyhow::Result;

pub trait EmbeddingEngine: Send + Sync {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_image(&self, rgb: &image::RgbImage) -> Result<Vec<f32>>;
    fn default_text_model_name(&self) -> &'static str;
    fn default_image_model_name(&self) -> &'static str;
}

pub struct MockEmbeddingEngine;

impl EmbeddingEngine for MockEmbeddingEngine {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let len = text.len() as f32;

        Ok(vec![
            len,
            len + 1.0,
            len + 2.0,
            len + 3.0,
            len + 4.0,
            len + 5.0,
            len + 6.0,
            len + 7.0,
        ])
    }

    fn embed_image(&self, rgb: &image::RgbImage) -> Result<Vec<f32>> {
        let width = rgb.width() as f32;
        let height = rgb.height() as f32;

        Ok(vec![
            width,
            height,
            width / height.max(1.0),
            width * height,
            1.0,
            2.0,
            3.0,
            4.0,
        ])
    }

    fn default_text_model_name(&self) -> &'static str {
        "mock-text-v1"
    }

    fn default_image_model_name(&self) -> &'static str {
        "mock-image-v1"
    }
}



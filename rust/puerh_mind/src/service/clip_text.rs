use candle_core::Tensor;
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, LayerNormConfig};

pub struct ClipTextModel {
    pub token_embedding: Embedding,
    pub positional_embedding: Tensor,
    pub blocks: Vec<ResidualAttentionBlock>,
    pub ln_final: LayerNorm,
    pub text_projection: Tensor,
}

pub struct ResidualAttentionBlock {
    pub ln_1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln_2: LayerNorm,
    pub mlp: Mlp,
}

pub struct MultiHeadAttention {
    pub in_proj_weight: Tensor,
    pub in_proj_bias: Option<Tensor>,
    pub out_proj: Linear,
    pub num_heads: usize,
}

pub struct Mlp {
    pub c_fc: Linear,
    pub c_proj: Linear,
}

impl ClipTextModel {
    pub fn load(
        vb: candle_nn::VarBuilder,
        config: &crate::service::open_clip_config::TextConfig,
        embed_dim: usize,
    ) -> anyhow::Result<Self> {
        let token_embedding =
            candle_nn::embedding(config.vocab_size, config.width, vb.pp("token_embedding"))?;

        let positional_embedding = vb.get(
            (config.context_length, config.width),
            "positional_embedding",
        )?;

        let ln_final = candle_nn::layer_norm(config.width, candle_nn::LayerNormConfig::default(), vb.pp("ln_final"))?;

        let text_projection = vb.get(
            (config.width, embed_dim),
            "text_projection",
        )?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks: Vec::new(),
            ln_final,
            text_projection,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarBuilder;

    use crate::config::SemanticConfig;
    use crate::service::model_assets::ClipModelPaths;
    use crate::service::open_clip_config::OpenClipConfig;

    #[test]
    fn loads_clip_text_top_level_weights() {
        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        model_paths.validate().expect("model assets should exist");

        let open_clip_config =
            OpenClipConfig::load(&model_paths.config).expect("config should load");

        let device = Device::Cpu;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let text_model = ClipTextModel::load(
            vb.pp("text"),
            &open_clip_config.model_cfg.text_cfg,
            open_clip_config.model_cfg.embed_dim,
        )
        .expect("text model should load");

        assert_eq!(
            text_model.positional_embedding.dims(),
            &[
                open_clip_config.model_cfg.text_cfg.context_length,
                open_clip_config.model_cfg.text_cfg.width,
            ]
        );

        assert_eq!(
            text_model.text_projection.dims(),
            &[
                open_clip_config.model_cfg.text_cfg.width,
                open_clip_config.model_cfg.embed_dim,
            ]
        );

        assert!(text_model.blocks.is_empty());
    }
}

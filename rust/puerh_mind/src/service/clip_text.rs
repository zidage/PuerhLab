use candle_core::{IndexOp, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, ops::softmax};

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

        let ln_final = candle_nn::layer_norm(
            config.width,
            candle_nn::LayerNormConfig::default(),
            vb.pp("ln_final"),
        )?;

        let text_projection = vb.get((config.width, embed_dim), "text_projection")?;

        let mut blocks = Vec::with_capacity(config.layers);
        for layer_idx in 0..config.layers {
            let block = ResidualAttentionBlock::load(
                vb.pp(format!("transformer.resblocks.{layer_idx}")),
                config.width,
                config.heads,
            )?;
            blocks.push(block);
        }

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln_final,
            text_projection,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.token_embedding.forward(input_ids)?;

        let (_, seq_len, _) = x.dims3()?;

        let pos = self.positional_embedding.i((0..seq_len, ..))?;
        let x = x.broadcast_add(&pos)?;

        Ok(x)
    }

    pub fn forward_hidden_states(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = self.embed_tokens(input_ids)?;

        for block in &self.blocks {
            x = block.forward(&x, None)?;
        }

        let x = self.ln_final.forward(&x)?; // all the way to [B, T, C]
        Ok(x)
    }

    fn select_eot_indices(&self, input_ids: &Tensor) -> anyhow::Result<Vec<usize>> {
        let ids = input_ids.to_vec2::<u32>()?;
        if ids.is_empty() {
            anyhow::bail!("input_ids batch is empty");
        }

        ids.into_iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by_key(|(_, token_id)| *token_id)
                    .map(|(index, _)| index)
                    .ok_or_else(|| anyhow::anyhow!("input_ids row is empty"))
            })
            .collect()
    }

    pub fn forward_text_feature(&self, input_ids: &Tensor) -> anyhow::Result<Tensor> {
        let hidden = self.forward_hidden_states(input_ids)?;
        let eot_indices = self.select_eot_indices(input_ids)?;

        let (batch_size, _, width) = hidden.dims3()?;
        if eot_indices.len() != batch_size {
            anyhow::bail!(
                "eot index count {} did not match hidden batch size {batch_size}",
                eot_indices.len()
            );
        }

        let mut token_hidden_rows = Vec::with_capacity(batch_size);
        for (batch_idx, eot_index) in eot_indices.into_iter().enumerate() {
            let token_hidden = hidden.i((batch_idx, eot_index, ..))?;
            token_hidden_rows.push(token_hidden.reshape((1, width))?);
        }

        let token_hidden = Tensor::cat(&token_hidden_rows, 0)?;

        let text_feature = token_hidden.matmul(&self.text_projection)?;
        Ok(text_feature)
    }
}

impl MultiHeadAttention {
    pub fn load(vb: candle_nn::VarBuilder, width: usize, num_heads: usize) -> anyhow::Result<Self> {
        let in_proj_weight = vb.get((3 * width, width), "in_proj_weight")?;
        let in_proj_bias = Some(vb.get(3 * width, "in_proj_bias")?);
        let out_proj = candle_nn::linear(width, width, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj,
            num_heads,
        })
    }

    pub fn split_qkv_weight(&self) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (three_width, _width) = self.in_proj_weight.dims2()?;

        if three_width % 3 != 0 {
            anyhow::bail!("in_proj_weight first dim must be divisible by 3, got {three_width}");
        }

        let part = three_width / 3;

        let q = self.in_proj_weight.i((0..part, ..))?;
        let k = self.in_proj_weight.i((part..2 * part, ..))?;
        let v = self.in_proj_weight.i((2 * part..3 * part, ..))?;

        Ok((q, k, v))
    }

    pub fn split_qkv_bias(&self) -> anyhow::Result<Option<(Tensor, Tensor, Tensor)>> {
        let bias = match &self.in_proj_bias {
            Some(bias) => {
                let len = bias.dims1()?;
                if len % 3 != 0 {
                    anyhow::bail!("in_proj_bias len must be divisible by 3, got {len}");
                }
                let part = len / 3;
                let q = bias.i(0..part)?;
                let k = bias.i(part..2 * part)?;
                let v = bias.i(2 * part..3 * part)?;
                Some((q, k, v))
            }
            None => None,
        };
        Ok(bias)
    }

    fn linear_3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> anyhow::Result<Tensor> {
        let (batch, seq_len, in_features) = x.dims3()?; // [B, T, C]
        let (out_features, weight_in_features) = weight.dims2()?; // [O, C]

        if in_features != weight_in_features {
            anyhow::bail!(
                "input feature dim {in_features} does not match weight input dim {weight_in_features}"
            );
        }

        let x2 = x.reshape((batch * seq_len, in_features))?;
        let y2 = x2.matmul(&weight.t()?)?;

        let y2 = match bias {
            Some(bias) => y2.broadcast_add(bias)?,
            None => y2,
        };

        let y = y2.reshape((batch, seq_len, out_features))?;
        Ok(y)
    }

    pub fn project_qkv(&self, x: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (q_weight, k_weight, v_weight) = self.split_qkv_weight()?;
        let bias = self.split_qkv_bias()?;

        let (q_bias, k_bias, v_bias) = match bias {
            Some((q, k, v)) => (Some(q), Some(k), Some(v)),
            None => (None, None, None),
        };

        let q = Self::linear_3d(x, &q_weight, q_bias.as_ref())?;
        let k = Self::linear_3d(x, &k_weight, k_bias.as_ref())?;
        let v = Self::linear_3d(x, &v_weight, v_bias.as_ref())?;

        Ok((q, k, v))
    }

    pub fn reshape_heads(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (batch, seq_len, width) = x.dims3()?;

        if width % self.num_heads != 0 {
            anyhow::bail!(
                "hidden width {width} is not divisible by num_heads {}",
                self.num_heads
            );
        }
        let head_dim = width / self.num_heads; // D = head_dim

        let x = x.reshape((batch, seq_len, self.num_heads, head_dim))?; // [B,T,H,D]
        let x = x.transpose(1, 2)?; // [B,H,T,D]

        Ok(x)
    }

    pub fn project_qkv_heads(&self, x: &Tensor) -> anyhow::Result<(Tensor, Tensor, Tensor)> {
        let (q, k, v) = self.project_qkv(x)?;
        let q = self.reshape_heads(&q)?;
        let k = self.reshape_heads(&k)?;
        let v = self.reshape_heads(&v)?;
        Ok((q, k, v))
    }

    pub fn forward(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> anyhow::Result<Tensor> {
        let (q, k, v) = self.project_qkv_heads(x)?; // [B, H, T, D]
        let (batch, num_heads, seq_len, head_dim) = q.dims4()?;

        let q = q.reshape((batch * num_heads, seq_len, head_dim))?;
        let k = k.reshape((batch * num_heads, seq_len, head_dim))?;
        let v = v.reshape((batch * num_heads, seq_len, head_dim))?;

        let attn = q.matmul(&k.transpose(1, 2)?)?; // [B*H, T, T]
        let scale = 1f64 / (head_dim as f64).sqrt();
        let mut attn = attn.affine(scale, 0.0)?;

        if let Some(attn_mask) = attn_mask {
            attn = attn.broadcast_add(attn_mask)?;
        }

        let attn = softmax(&attn, candle_core::D::Minus1)?;
        let y = attn.matmul(&v)?; // [B*H, T, D]
        let y = y.reshape((batch, num_heads, seq_len, head_dim))?;
        let y = y.transpose(1, 2)?; // [B, T, H, D]

        let y = y.reshape((batch, seq_len, num_heads * head_dim))?;

        let y = self.out_proj.forward(&y)?;
        Ok(y)
    }
}

impl Mlp {
    pub fn load(vb: candle_nn::VarBuilder, width: usize) -> anyhow::Result<Self> {
        let hidden_width = width * 4;

        let c_fc = candle_nn::linear(width, hidden_width, vb.pp("c_fc"))?;
        let c_proj = candle_nn::linear(hidden_width, width, vb.pp("c_proj"))?;

        Ok(Self { c_fc, c_proj })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let h = self.c_fc.forward(x)?;
        let h = candle_nn::activation::Activation::Gelu.forward(&h)?;
        let h = self.c_proj.forward(&h)?;

        Ok(h)
    }
}

impl ResidualAttentionBlock {
    pub fn load(vb: candle_nn::VarBuilder, width: usize, num_heads: usize) -> anyhow::Result<Self> {
        let ln_1 =
            candle_nn::layer_norm(width, candle_nn::LayerNormConfig::default(), vb.pp("ln_1"))?;

        let attn = MultiHeadAttention::load(vb.pp("attn"), width, num_heads)?;

        let ln_2 =
            candle_nn::layer_norm(width, candle_nn::LayerNormConfig::default(), vb.pp("ln_2"))?;

        let mlp = Mlp::load(vb.pp("mlp"), width)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    pub fn forward_attention(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> anyhow::Result<Tensor> {
        let h = self.ln_1.forward(x)?;
        let h = self.attn.forward(&h, attn_mask)?; // [B, T, C]
        let y = x.broadcast_add(&h)?; // [B, T, C] + [B, T, C]
        Ok(y)
    }

    pub fn forward(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> anyhow::Result<Tensor> {
        let x = self.forward_attention(x, attn_mask)?; // [B, T, C]
        let h = self.ln_2.forward(&x)?; // [B, T, C]
        let h = self.mlp.forward(&h)?; // [B, T, C]
        let y = x.broadcast_add(&h)?; // [B, T, C]

        Ok(y)
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

    fn test_semantic_config() -> SemanticConfig {
        SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        }
    }

    fn load_test_model() -> (OpenClipConfig, Device, ClipTextModel) {
        let semantic_config = test_semantic_config();
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

        (open_clip_config, device, text_model)
    }

    fn make_test_input_ids(device: &Device, seq_len: usize) -> Tensor {
        candle_core::Tensor::from_vec(vec![1u32; seq_len], (1, seq_len), device)
            .expect("input ids should build")
    }

    #[test]
    fn loads_clip_text_top_level_weights() {
        let (open_clip_config, _device, text_model) = load_test_model();

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

        assert_eq!(
            text_model.blocks.len(),
            open_clip_config.model_cfg.text_cfg.layers
        );

        assert_eq!(
            text_model.blocks[0].attn.num_heads,
            open_clip_config.model_cfg.text_cfg.heads
        );

        assert_eq!(
            text_model.blocks[0].attn.in_proj_weight.dims().to_vec(),
            vec![
                3 * open_clip_config.model_cfg.text_cfg.width,
                open_clip_config.model_cfg.text_cfg.width,
            ]
        );

        assert_eq!(
            text_model.blocks[0]
                .attn
                .in_proj_bias
                .as_ref()
                .expect("attention bias should exist")
                .dims()
                .to_vec(),
            vec![3 * open_clip_config.model_cfg.text_cfg.width]
        );
    }

    #[test]
    fn embeds_tokens_with_position_embeddings() {
        let (open_clip_config, device, text_model) = load_test_model();
        let input_ids =
            make_test_input_ids(&device, open_clip_config.model_cfg.text_cfg.context_length);

        let x = text_model
            .embed_tokens(&input_ids)
            .expect("embedding should succeed");

        assert_eq!(
            x.dims().to_vec(),
            vec![
                1,
                open_clip_config.model_cfg.text_cfg.context_length,
                open_clip_config.model_cfg.text_cfg.width,
            ]
        );
    }

    #[test]
    fn projects_qkv_from_embedded_tokens() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let width = open_clip_config.model_cfg.text_cfg.width;
        let input_ids = make_test_input_ids(&device, seq_len);

        let x = text_model
            .embed_tokens(&input_ids)
            .expect("token embedding should succeed");

        let (q, k, v) = text_model.blocks[0]
            .attn
            .project_qkv(&x)
            .expect("qkv projection should succeed");

        assert_eq!(q.dims().to_vec(), vec![1, seq_len, width]);
        assert_eq!(k.dims().to_vec(), vec![1, seq_len, width]);
        assert_eq!(v.dims().to_vec(), vec![1, seq_len, width]);
    }

    #[test]
    fn projects_qkv_into_heads() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let input_ids = make_test_input_ids(&device, seq_len);

        let x = text_model
            .embed_tokens(&input_ids)
            .expect("token embedding should succeed");

        let (q, k, v) = text_model.blocks[0]
            .attn
            .project_qkv_heads(&x)
            .expect("qkv projection should succeed");

        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let width = open_clip_config.model_cfg.text_cfg.width;
        let heads = open_clip_config.model_cfg.text_cfg.heads;
        let head_dim = width / heads;

        assert_eq!(q.dims().to_vec(), vec![1, heads, seq_len, head_dim]);
        assert_eq!(k.dims().to_vec(), vec![1, heads, seq_len, head_dim]);
        assert_eq!(v.dims().to_vec(), vec![1, heads, seq_len, head_dim]);
    }

    #[test]
    fn runs_attention_forward_on_embedded_tokens() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let width = open_clip_config.model_cfg.text_cfg.width;
        let input_ids = make_test_input_ids(&device, seq_len);

        let x = text_model
            .embed_tokens(&input_ids)
            .expect("token embedding should succeed");

        let y = text_model.blocks[0]
            .attn
            .forward(&x, None)
            .expect("attention forward should succeed");

        assert_eq!(y.dims().to_vec(), vec![1, seq_len, width]);
    }

    #[test]
    fn runs_residual_attention_block_attention_path() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let width = open_clip_config.model_cfg.text_cfg.width;
        let input_ids = make_test_input_ids(&device, seq_len);

        let x = text_model
            .embed_tokens(&input_ids)
            .expect("token embedding should succeed");

        let y = text_model.blocks[0]
            .forward_attention(&x, None)
            .expect("residual attention path should succeed");

        assert_eq!(y.dims().to_vec(), vec![1, seq_len, width]);
    }

    #[test]
    fn runs_full_residual_attention_block() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let width = open_clip_config.model_cfg.text_cfg.width;
        let input_ids = make_test_input_ids(&device, seq_len);

        let x = text_model
            .embed_tokens(&input_ids)
            .expect("token embedding should succeed");

        let y = text_model.blocks[0]
            .forward(&x, None)
            .expect("full residual attention block should succeed");

        assert_eq!(y.dims().to_vec(), vec![1, seq_len, width]);
    }

    #[test]
    fn runs_full_text_transformer_hidden_states() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let width = open_clip_config.model_cfg.text_cfg.width;
        let input_ids = make_test_input_ids(&device, seq_len);

        let x = text_model
            .forward_hidden_states(&input_ids)
            .expect("text transformer hidden states should succeed");

        assert_eq!(x.dims().to_vec(), vec![1, seq_len, width]);
    }

    #[test]
    fn produces_text_feature_from_eot_position() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let embed_dim = open_clip_config.model_cfg.embed_dim;

        let mut ids = vec![1u32; seq_len];
        ids[5] = 49407;

        let input_ids =
            Tensor::from_vec(ids, (1, seq_len), &device).expect("input ids should build");

        let feature = text_model
            .forward_text_feature(&input_ids)
            .expect("text feature should succeed");

        assert_eq!(feature.dims().to_vec(), vec![1, embed_dim]);
    }

    #[test]
    fn produces_batched_text_features_from_each_row_eot_position() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let embed_dim = open_clip_config.model_cfg.embed_dim;

        let mut ids = vec![1u32; 2 * seq_len];
        ids[5] = 49407;
        ids[seq_len + 9] = 49407;

        let input_ids =
            Tensor::from_vec(ids, (2, seq_len), &device).expect("input ids should build");

        let feature = text_model
            .forward_text_feature(&input_ids)
            .expect("batched text features should succeed");

        assert_eq!(feature.dims().to_vec(), vec![2, embed_dim]);
    }

    #[test]
    fn photo_prompt_variants_do_not_collapse() {
        let (open_clip_config, device, text_model) = load_test_model();
        let seq_len = open_clip_config.model_cfg.text_cfg.context_length;
        let prompts = vec![
            vec![49406u32, 320, 1125, 539, 320, 1929, 269, 49407],
            vec![49406u32, 320, 1125, 539, 320, 2368, 269, 49407],
            vec![49406u32, 320, 1125, 539, 320, 1615, 269, 49407],
        ];

        let mut flat_ids = Vec::with_capacity(prompts.len() * seq_len);
        for mut prompt in prompts {
            prompt.resize(seq_len, 0);
            flat_ids.extend(prompt);
        }

        let input_ids = Tensor::from_vec(flat_ids, (3, seq_len), &device)
            .expect("input ids should build");

        let features = text_model
            .forward_text_feature(&input_ids)
            .expect("text features should succeed")
            .to_vec2::<f32>()
            .expect("features should materialize");

        fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
            let dot = left
                .iter()
                .zip(right.iter())
                .map(|(lhs, rhs)| lhs * rhs)
                .sum::<f32>();
            let left_norm = left.iter().map(|value| value * value).sum::<f32>().sqrt();
            let right_norm = right.iter().map(|value| value * value).sum::<f32>().sqrt();
            dot / (left_norm * right_norm)
        }

        let dog_cat = cosine_similarity(&features[0], &features[1]);
        let dog_car = cosine_similarity(&features[0], &features[2]);
        let cat_car = cosine_similarity(&features[1], &features[2]);

        assert!(dog_cat < 0.95, "dog vs cat prompt similarity was {dog_cat}");
        assert!(dog_car < 0.95, "dog vs car prompt similarity was {dog_car}");
        assert!(cat_car < 0.95, "cat vs car prompt similarity was {cat_car}");
    }
}

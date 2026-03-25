use candle_core::{D, IndexOp, Tensor};
use candle_nn::{
    Activation, BatchNorm, Conv2d, Conv2dConfig, Linear, Module, ModuleT, ops::softmax,
};

pub struct ConvBn2d {
    pub conv: Conv2d,
    pub bn: BatchNorm,
}

pub struct SqueezeExcite {
    pub fc1: Conv2d,
    pub fc2: Conv2d,
}

impl SqueezeExcite {
    pub fn load(
        vb: candle_nn::VarBuilder,
        in_channels: usize,
        hidden_channels: usize,
    ) -> anyhow::Result<Self> {
        let fc1 = candle_nn::conv2d(
            in_channels,
            hidden_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("fc1"),
        )?;

        let fc2 = candle_nn::conv2d(
            hidden_channels,
            in_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("fc2"),
        )?;

        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let scale = x.mean_keepdim((2, 3))?;
        let scale = self.fc1.forward(&scale)?;
        let scale = Activation::Relu.forward(&scale)?;
        let scale = self.fc2.forward(&scale)?;
        let scale = Activation::Sigmoid.forward(&scale)?;

        let y = x.broadcast_mul(&scale)?;
        Ok(y)
    }
}

impl ConvBn2d {
    pub fn load(
        vb: candle_nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
    ) -> anyhow::Result<Self> {
        let conv = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_size,
            Conv2dConfig {
                stride,
                padding,
                groups,
                ..Default::default()
            },
            vb.pp("conv"),
        )?;

        let bn = candle_nn::batch_norm(out_channels, 1e-5, vb.pp("bn"))?;

        Ok(Self { conv, bn })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.bn.forward_t(&x, false)?;
        Ok(x)
    }
}

pub struct MobileOneBlock {
    pub conv_kxk: ConvBn2d,
    pub conv_scale: Option<ConvBn2d>,
    pub identity: Option<BatchNorm>,
    pub se: Option<SqueezeExcite>,
    pub act: Activation,
    pub use_act: bool,
}

impl MobileOneBlock {
    pub fn load(
        vb: candle_nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        has_conv_scale: bool,
        has_identity: bool,
        se_hidden_channels: Option<usize>,
        use_act: bool,
        act: Activation,
    ) -> anyhow::Result<Self> {
        let conv_kxk = ConvBn2d::load(
            vb.pp("conv_kxk.0"),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
        )?;

        let conv_scale = if has_conv_scale {
            Some(ConvBn2d::load(
                vb.pp("conv_scale"),
                in_channels,
                out_channels,
                1,
                stride,
                0,
                groups,
            )?)
        } else {
            None
        };

        if has_identity && stride != 1 {
            anyhow::bail!("identity branch requires stride=1");
        }

        let identity = if has_identity {
            Some(candle_nn::batch_norm(
                out_channels,
                1e-5,
                vb.pp("identity"),
            )?)
        } else {
            None
        };

        let se = if let Some(hidden_channels) = se_hidden_channels {
            Some(SqueezeExcite::load(
                vb.pp("se"),
                out_channels,
                hidden_channels,
            )?)
        } else {
            None
        };

        Ok(Self {
            conv_kxk,
            conv_scale,
            identity,
            se,
            act,
            use_act,
        })
    }

    pub fn load_final_conv(vb: candle_nn::VarBuilder) -> anyhow::Result<Self> {
        Self::load(
            vb,
            640,
            1280,
            3,
            1,
            1,
            640,
            true,
            false,
            Some(80),
            true,
            Activation::Gelu,
        )
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut y = self.conv_kxk.forward(x)?;

        if let Some(conv_scale) = &self.conv_scale {
            let scale = conv_scale.forward(x)?;
            y = y.broadcast_add(&scale)?;
        }

        if let Some(identity) = &self.identity {
            let identity_out = identity.forward_t(x, false)?;
            y = y.broadcast_add(&identity_out)?;
        }

        if let Some(se) = &self.se {
            y = se.forward(&y)?;
        }

        if self.use_act {
            self.act.forward(&y).map_err(Into::into)
        } else {
            Ok(y)
        }
    }
}

pub struct LayerScale2d {
    pub gamma: Tensor,
}

impl LayerScale2d {
    pub fn load(vb: candle_nn::VarBuilder, channels: usize) -> anyhow::Result<Self> {
        let gamma = vb.get((channels, 1, 1), "gamma")?;
        Ok(Self { gamma })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        Ok(x.broadcast_mul(&self.gamma)?)
    }
}

pub struct ConvMlp {
    pub conv: ConvBn2d,
    pub fc1: Conv2d,
    pub fc2: Conv2d,
    pub act: Activation,
}

impl ConvMlp {
    pub fn load(
        vb: candle_nn::VarBuilder,
        channels: usize,
        hidden_channels: usize,
    ) -> anyhow::Result<Self> {
        let conv = ConvBn2d::load(vb.pp("conv"), channels, channels, 7, 1, 3, channels)?;

        let fc1 = candle_nn::conv2d(
            channels,
            hidden_channels,
            1,
            Conv2dConfig::default(),
            vb.pp("fc1"),
        )?;

        let fc2 = candle_nn::conv2d(
            hidden_channels,
            channels,
            1,
            Conv2dConfig::default(),
            vb.pp("fc2"),
        )?;

        Ok(Self {
            conv,
            fc1,
            fc2,
            act: Activation::Gelu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.conv.forward(x)?;
        let x = self.fc1.forward(&x)?;
        let x = self.act.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}

pub struct RepMixerBlock {
    pub norm: BatchNorm,
    pub token_mixer: MobileOneBlock,
    pub token_mixer_layer_scale: LayerScale2d,
    pub mlp: ConvMlp,
    pub mlp_layer_scale: LayerScale2d,
}

impl RepMixerBlock {
    pub fn load(
        vb: candle_nn::VarBuilder,
        channels: usize,
        mlp_hidden_channels: usize,
    ) -> anyhow::Result<Self> {
        let norm = candle_nn::batch_norm(channels, 1e-5, vb.pp("token_mixer.norm.identity"))?;

        let token_mixer = MobileOneBlock::load(
            vb.pp("token_mixer.mixer"),
            channels,
            channels,
            3,
            1,
            1,
            channels,
            true,
            true,
            None,
            false,
            Activation::Gelu,
        )?;

        let token_mixer_layer_scale =
            LayerScale2d::load(vb.pp("token_mixer.layer_scale"), channels)?;

        let mlp = ConvMlp::load(vb.pp("mlp"), channels, mlp_hidden_channels)?;
        let mlp_layer_scale = LayerScale2d::load(vb.pp("layer_scale"), channels)?;

        Ok(Self {
            norm,
            token_mixer,
            token_mixer_layer_scale,
            mlp,
            mlp_layer_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let normed = self.norm.forward_t(x, false)?;
        let mixed = self.token_mixer.forward(x)?;
        let mixed = mixed.broadcast_sub(&normed)?;
        let mixed = self.token_mixer_layer_scale.forward(&mixed)?;
        let x = x.broadcast_add(&mixed)?;

        let mlp_out = self.mlp.forward(&x)?;
        let mlp_out = self.mlp_layer_scale.forward(&mlp_out)?;
        let x = x.broadcast_add(&mlp_out)?;

        Ok(x)
    }
}

pub struct LargeSmallConv {
    pub large_conv: ConvBn2d,
    pub small_conv: ConvBn2d,
    pub se: Option<SqueezeExcite>,
}

impl LargeSmallConv {
    pub fn load(
        vb: candle_nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        se_hidden_channels: Option<usize>,
    ) -> anyhow::Result<Self> {
        let large_conv = ConvBn2d::load(
            vb.pp("large_conv"),
            in_channels,
            out_channels,
            7,
            stride,
            3,
            in_channels,
        )?;
        let small_conv = ConvBn2d::load(
            vb.pp("small_conv"),
            in_channels,
            out_channels,
            3,
            stride,
            1,
            in_channels,
        )?;
        let se = match se_hidden_channels {
            Some(hidden) => Some(SqueezeExcite::load(vb.pp("se"), out_channels, hidden)?),
            None => None,
        };
        Ok(Self {
            large_conv,
            small_conv,
            se,
        })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let large = self.large_conv.forward(x)?;
        let small = self.small_conv.forward(x)?;
        let mut y = large.broadcast_add(&small)?;

        if let Some(se) = &self.se {
            y = se.forward(&y)?;
        }

        Ok(y)
    }
}

pub struct StageDownsample {
    pub proj0: LargeSmallConv,
    pub proj1: MobileOneBlock,
}

impl StageDownsample {
    fn load(
        vb: candle_nn::VarBuilder,
        in_ch: usize,
        out_ch: usize,
        se_hidden: Option<usize>,
    ) -> anyhow::Result<Self> {
        let proj0 = LargeSmallConv::load(vb.pp("proj.0"), in_ch, out_ch, 2, se_hidden)?;
        let proj1 = MobileOneBlock::load(
            vb.pp("proj.1"),
            out_ch,
            out_ch,
            1,
            1,
            0,
            1,
            false,
            true,
            None,
            true,
            Activation::Gelu,
        )?;
        Ok(Self { proj0, proj1 })
    }

    pub fn load_stage1(vb: candle_nn::VarBuilder) -> anyhow::Result<Self> {
        Self::load(vb, 80, 160, None)
    }
    pub fn load_stage2(vb: candle_nn::VarBuilder) -> anyhow::Result<Self> {
        Self::load(vb, 160, 320, Some(80))
    }
    pub fn load_stage3(vb: candle_nn::VarBuilder) -> anyhow::Result<Self> {
        Self::load(vb, 320, 640, Some(160))
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.proj0.forward(x)?;
        let x = self.proj1.forward(&x)?;
        Ok(x)
    }
}

pub struct RepConditionalPosEnc {
    pub pos_enc: Conv2d,
}

impl RepConditionalPosEnc {
    pub fn load(vb: candle_nn::VarBuilder, channels: usize) -> anyhow::Result<Self> {
        let pos_enc = candle_nn::conv2d(
            channels,
            channels,
            7,
            Conv2dConfig {
                padding: 3,
                groups: channels,
                ..Default::default()
            },
            vb.pp("pos_enc"),
        )?;

        Ok(Self { pos_enc })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let pos = self.pos_enc.forward(x)?;
        let y = x.broadcast_add(&pos)?;
        Ok(y)
    }
}

pub struct Attention2d {
    pub qkv_weight: Tensor,
    pub proj: Linear,
    pub num_heads: usize,
    pub head_dim: usize,
}

impl Attention2d {
    pub fn load(
        vb: candle_nn::VarBuilder,
        channels: usize,
        head_dim: usize,
    ) -> anyhow::Result<Self> {
        if channels % head_dim != 0 {
            anyhow::bail!("channels {channels} must be divisible by head_dim {head_dim}");
        }

        let qkv_weight = vb.get((3 * channels, channels), "qkv.weight")?;
        let proj = candle_nn::linear(channels, channels, vb.pp("proj"))?;

        Ok(Self {
            qkv_weight,
            proj,
            num_heads: channels / head_dim,
            head_dim,
        })
    }

    fn linear_3d_no_bias(x: &Tensor, weight: &Tensor) -> anyhow::Result<Tensor> {
        let (batch, seq_len, in_features) = x.dims3()?;
        let (out_features, weight_in_features) = weight.dims2()?;

        if in_features != weight_in_features {
            anyhow::bail!(
                "input feature dim {in_features} does not match weight input dim {weight_in_features}"
            );
        }

        let x2 = x.reshape((batch * seq_len, in_features))?;
        let y2 = x2.matmul(&weight.t()?)?;
        let y = y2.reshape((batch, seq_len, out_features))?;
        Ok(y)
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let (batch, channels, height, width) = x.dims4()?;
        let num_tokens = height * width;

        let x = x.reshape((batch, channels, num_tokens))?;
        let x = x.transpose(1, 2)?; // [B, N, C]

        let qkv = Self::linear_3d_no_bias(&x, &self.qkv_weight)?;
        let qkv = qkv.reshape((batch, num_tokens, 3, self.num_heads, self.head_dim))?;

        let q = qkv.i((.., .., 0, .., ..))?.transpose(1, 2)?; // [B, H, N, D]
        let k = qkv.i((.., .., 1, .., ..))?.transpose(1, 2)?; // [B, H, N, D]
        let v = qkv.i((.., .., 2, .., ..))?.transpose(1, 2)?; // [B, H, N, D]

        let q = q.reshape((batch * self.num_heads, num_tokens, self.head_dim))?;
        let k = k.reshape((batch * self.num_heads, num_tokens, self.head_dim))?;
        let v = v.reshape((batch * self.num_heads, num_tokens, self.head_dim))?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let q = q.affine(scale, 0.0)?;
        let attn = q.matmul(&k.transpose(1, 2)?)?;
        let attn = softmax(&attn, D::Minus1)?;

        let y = attn.matmul(&v)?; // [B*H, N, D]
        let y = y.reshape((batch, self.num_heads, num_tokens, self.head_dim))?;
        let y = y.transpose(1, 2)?; // [B, N, H, D]
        let y = y.reshape((batch, num_tokens, channels))?;
        let y = self.proj.forward(&y)?;
        let y = y.transpose(1, 2)?;
        let y = y.reshape((batch, channels, height, width))?;

        Ok(y)
    }
}

pub struct AttentionBlock2d {
    pub norm: BatchNorm,
    pub token_mixer: Attention2d,
    pub layer_scale_1: LayerScale2d,
    pub mlp: ConvMlp,
    pub layer_scale_2: LayerScale2d,
}

impl AttentionBlock2d {
    pub fn load(
        vb: candle_nn::VarBuilder,
        channels: usize,
        mlp_hidden_channels: usize,
        head_dim: usize,
    ) -> anyhow::Result<Self> {
        let norm = candle_nn::batch_norm(channels, 1e-5, vb.pp("norm"))?;
        let token_mixer = Attention2d::load(vb.pp("token_mixer"), channels, head_dim)?;
        let layer_scale_1 = LayerScale2d::load(vb.pp("layer_scale_1"), channels)?;
        let mlp = ConvMlp::load(vb.pp("mlp"), channels, mlp_hidden_channels)?;
        let layer_scale_2 = LayerScale2d::load(vb.pp("layer_scale_2"), channels)?;

        Ok(Self {
            norm,
            token_mixer,
            layer_scale_1,
            mlp,
            layer_scale_2,
        })
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let normed = self.norm.forward_t(x, false)?;
        let attn_out = self.token_mixer.forward(&normed)?;
        let attn_out = self.layer_scale_1.forward(&attn_out)?;
        let x = x.broadcast_add(&attn_out)?;

        let mlp_out = self.mlp.forward(&x)?;
        let mlp_out = self.layer_scale_2.forward(&mlp_out)?;
        let x = x.broadcast_add(&mlp_out)?;

        Ok(x)
    }
}

pub struct ClipVisionModel {
    pub stem: Vec<MobileOneBlock>,
    pub stage0: Vec<RepMixerBlock>,
    pub stage1_downsample: StageDownsample,
    pub stage1: Vec<RepMixerBlock>,
    pub stage2_downsample: StageDownsample,
    pub stage2: Vec<RepMixerBlock>,
    pub stage3_downsample: StageDownsample,
    pub stage3_pos_emb: RepConditionalPosEnc,
    pub stage3: Vec<AttentionBlock2d>,
    pub final_conv: MobileOneBlock,
    pub head: Linear,
}

impl ClipVisionModel {
    pub fn load(vb: candle_nn::VarBuilder, embed_dim: usize) -> anyhow::Result<Self> {
        let stem = vec![
            MobileOneBlock::load(
                vb.pp("trunk.stem.0"),
                3,
                80,
                3,
                2,
                1,
                1,
                true,
                false,
                None,
                true,
                Activation::Gelu,
            )?,
            MobileOneBlock::load(
                vb.pp("trunk.stem.1"),
                80,
                80,
                3,
                2,
                1,
                80,
                true,
                false,
                None,
                true,
                Activation::Gelu,
            )?,
            MobileOneBlock::load(
                vb.pp("trunk.stem.2"),
                80,
                80,
                1,
                1,
                0,
                1,
                false,
                true,
                None,
                true,
                Activation::Gelu,
            )?,
        ];

        let stage0 = (0..4)
            .map(|i| RepMixerBlock::load(vb.pp(format!("trunk.stages.0.blocks.{i}")), 80, 240))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let stage1_downsample = StageDownsample::load_stage1(vb.pp("trunk.stages.1.downsample"))?;

        let stage1 = (0..12)
            .map(|i| RepMixerBlock::load(vb.pp(format!("trunk.stages.1.blocks.{i}")), 160, 480))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let stage2_downsample = StageDownsample::load_stage2(vb.pp("trunk.stages.2.downsample"))?;
        let stage2 = (0..24)
            .map(|i| RepMixerBlock::load(vb.pp(format!("trunk.stages.2.blocks.{i}")), 320, 960))
            .collect::<anyhow::Result<Vec<_>>>()?;

        let stage3_downsample = StageDownsample::load_stage3(vb.pp("trunk.stages.3.downsample"))?;
        let stage3_pos_emb = RepConditionalPosEnc::load(vb.pp("trunk.stages.3.pos_emb"), 640)?;

        let stage3 = (0..4)
            .map(|i| {
                AttentionBlock2d::load(vb.pp(format!("trunk.stages.3.blocks.{i}")), 640, 1920, 32)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let final_conv = MobileOneBlock::load_final_conv(vb.pp("trunk.final_conv"))?;
        let head = candle_nn::linear(1280, embed_dim, vb.pp("trunk.head.fc"))?;
        Ok(Self {
            stem,
            stage0,
            stage1_downsample,
            stage1,
            stage2_downsample,
            stage2,
            stage3_downsample,
            stage3_pos_emb,
            stage3,
            final_conv,
            head,
        })
    }

    pub fn forward_stem(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = x.clone();
        for block in &self.stem {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    pub fn forward_stage0(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = x.clone();
        for block in &self.stage0 {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    pub fn forward_stage1(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = self.stage1_downsample.forward(x)?;
        for block in &self.stage1 {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    pub fn forward_stage2(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = self.stage2_downsample.forward(x)?;
        for block in &self.stage2 {
            x = block.forward(&x)?;
        }
        Ok(x)
    }

    pub fn forward_stage3_prelude(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.stage3_downsample.forward(x)?;
        let x = self.stage3_pos_emb.forward(&x)?;
        Ok(x)
    }

    pub fn forward_stage3(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let mut x = self.stage3_downsample.forward(x)?;
        x = self.stage3_pos_emb.forward(&x)?;

        for block in &self.stage3 {
            x = block.forward(&x)?;
        }

        Ok(x)
    }

    pub fn forward_final_conv(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        self.final_conv.forward(x)
    }

    pub fn global_avg_pool_2d(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let pooled = x.mean_keepdim((2, 3))?;
        let (batch, channels, _, _) = pooled.dims4()?;
        let pooled = pooled.reshape((batch, channels))?;
        Ok(pooled)
    }

    pub fn forward_head(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let pooled = self.global_avg_pool_2d(x)?;
        let y = self.head.forward(&pooled)?;

        Ok(y)
    }

    pub fn forward_features(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.forward_stem(x)?;
        let x = self.forward_stage0(&x)?;
        let x = self.forward_stage1(&x)?;
        let x = self.forward_stage2(&x)?;
        let x = self.forward_stage3(&x)?;
        let x = self.forward_final_conv(&x)?;
        Ok(x)
    }

    pub fn forward(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let x = self.forward_features(x)?;
        let x = self.forward_head(&x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;

    use crate::config::SemanticConfig;
    use crate::service::model_assets::ClipModelPaths;
    use crate::service::open_clip_config::OpenClipConfig;

    fn load_test_device() -> Device {
        match std::panic::catch_unwind(|| Device::new_metal(0)) {
            Ok(Ok(device)) => device,
            Ok(Err(err)) => {
                eprintln!("metal unavailable for clip vision tests, falling back to cpu: {err}");
                Device::Cpu
            }
            Err(_) => {
                eprintln!(
                    "metal initialization panicked in clip vision tests, falling back to cpu"
                );
                Device::Cpu
            }
        }
    }

    fn load_test_model() -> (OpenClipConfig, Device, ClipVisionModel) {
        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        model_paths.validate().expect("model assets should exist");

        let open_clip_config =
            OpenClipConfig::load(&model_paths.config).expect("config should load");

        let device = load_test_device();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let vision_model =
            ClipVisionModel::load(vb.pp("visual"), open_clip_config.model_cfg.embed_dim)
                .expect("vision model should load");

        (open_clip_config, device, vision_model)
    }

    #[test]
    fn loads_clip_vision_head_weights() {
        let (open_clip_config, _device, vision_model) = load_test_model();

        assert_eq!(
            vision_model.head.weight().dims(),
            &[open_clip_config.model_cfg.embed_dim, 1280]
        );

        let bias = vision_model
            .head
            .bias()
            .expect("vision head bias should exist");
        assert_eq!(bias.dims(), &[open_clip_config.model_cfg.embed_dim]);
    }

    #[test]
    fn forward_head_projects_1280_channels_to_embed_dim() {
        let (open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 1280, 8, 8), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_head(&x)
            .expect("vision head forward should succeed");

        assert_eq!(y.dims(), &[1, open_clip_config.model_cfg.embed_dim]);
    }

    #[test]
    fn loads_final_conv_kxk_branch_and_runs_forward() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let branch = ConvBn2d::load(
            {
                let semantic_config = SemanticConfig {
                    model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
                    model_dir: "./models/mobileclip2-s2-openclip".to_string(),
                };

                let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
                let vb = unsafe {
                    VarBuilder::from_mmaped_safetensors(
                        &[model_paths.weights.clone()],
                        candle_core::DType::F32,
                        &device,
                    )
                    .expect("weights should load")
                };
                vb.pp("visual.trunk.final_conv.conv_kxk.0")
            },
            640,
            1280,
            3,
            1,
            1,
            640,
        )
        .expect("final conv kxk branch should load");

        let x = Tensor::zeros((1, 640, 8, 8), DType::F32, &device).expect("tensor should build");

        let y = branch.forward(&x).expect("branch forward should succeed");

        assert_eq!(y.dims(), &[1, 1280, 8, 8]);
    }

    #[test]
    fn loads_final_conv_se_and_runs_forward() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let se = SqueezeExcite::load(vb.pp("visual.trunk.final_conv.se"), 1280, 80)
            .expect("final conv se should load");

        let x = Tensor::zeros((1, 1280, 8, 8), DType::F32, &device).expect("tensor should build");

        let y = se.forward(&x).expect("se forward should succeed");

        assert_eq!(y.dims(), &[1, 1280, 8, 8]);
    }

    #[test]
    fn loads_final_conv_block_and_runs_forward() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 640, 8, 8), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_final_conv(&x)
            .expect("final conv forward should succeed");

        assert_eq!(y.dims(), &[1, 1280, 8, 8]);
    }

    #[test]
    fn final_conv_plus_head_produces_embedding_shape() {
        let (open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 640, 8, 8), DType::F32, &device).expect("tensor should build");

        let x = vision_model
            .forward_final_conv(&x)
            .expect("final conv forward should succeed");

        let y = vision_model
            .forward_head(&x)
            .expect("vision head forward should succeed");

        assert_eq!(y.dims(), &[1, open_clip_config.model_cfg.embed_dim]);
    }

    #[test]
    fn mobileone_block_supports_identity_batch_norm_branch() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let conv_kxk = ConvBn2d::load(vb.pp("visual.trunk.stem.2.conv_kxk.0"), 80, 80, 1, 1, 0, 1)
            .expect("stem.2 conv_kxk should load");

        let identity = Some(
            candle_nn::batch_norm(80, 1e-5, vb.pp("visual.trunk.stem.2.identity"))
                .expect("stem.2 identity bn should load"),
        );

        let block = MobileOneBlock {
            conv_kxk,
            conv_scale: None,
            identity,
            se: None,
            act: Activation::Gelu,
            use_act: true,
        };

        let x = Tensor::zeros((1, 80, 32, 32), DType::F32, &device).expect("tensor should build");

        let y = block.forward(&x).expect("block forward should succeed");

        assert_eq!(y.dims(), &[1, 80, 32, 32]);
    }

    #[test]
    fn loads_stem_blocks() {
        let (_open_clip_config, _device, vision_model) = load_test_model();

        assert_eq!(vision_model.stem.len(), 3);
    }

    #[test]
    fn forward_stem_downsamples_to_64x64_with_80_channels() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 3, 256, 256), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_stem(&x)
            .expect("stem forward should succeed");

        assert_eq!(y.dims(), &[1, 80, 64, 64]);
    }

    #[test]
    fn loads_stage0_blocks() {
        let (_open_clip_config, _device, vision_model) = load_test_model();
        assert_eq!(vision_model.stage0.len(), 4);
    }

    #[test]
    fn forward_stage0_preserves_80x64x64_shape() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 80, 64, 64), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_stage0(&x)
            .expect("stage0 forward should succeed");

        assert_eq!(y.dims(), &[1, 80, 64, 64]);
    }

    #[test]
    fn loads_stage1_blocks() {
        let (_open_clip_config, _device, vision_model) = load_test_model();
        assert_eq!(vision_model.stage1.len(), 12);
    }

    #[test]
    fn forward_stage1_downsamples_to_32x32_with_160_channels() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 80, 64, 64), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_stage1(&x)
            .expect("stage1 forward should succeed");

        assert_eq!(y.dims(), &[1, 160, 32, 32]);
    }

    #[test]
    fn large_small_conv_supports_optional_se_branch() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let block = LargeSmallConv::load(
            vb.pp("visual.trunk.stages.2.downsample.proj.0"),
            160,
            320,
            2,
            Some(80),
        )
        .expect("stage2 downsample proj.0 should load");

        let x = Tensor::zeros((1, 160, 32, 32), DType::F32, &device).expect("tensor should build");
        let y = block.forward(&x).expect("forward should succeed");

        assert_eq!(y.dims(), &[1, 320, 16, 16]);
    }

    #[test]
    fn stage1_downsample_still_runs_after_refactor() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let downsample = StageDownsample::load_stage1(vb.pp("visual.trunk.stages.1.downsample"))
            .expect("stage1 downsample should load");

        let x = Tensor::zeros((1, 80, 64, 64), DType::F32, &device).expect("tensor should build");
        let y = downsample.forward(&x).expect("forward should succeed");

        assert_eq!(y.dims(), &[1, 160, 32, 32]);
    }

    #[test]
    fn stage2_downsample_loads_and_runs_forward() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let downsample = StageDownsample::load_stage2(vb.pp("visual.trunk.stages.2.downsample"))
            .expect("stage2 downsample should load");

        let x = Tensor::zeros((1, 160, 32, 32), DType::F32, &device).expect("tensor should build");
        let y = downsample.forward(&x).expect("forward should succeed");

        assert_eq!(y.dims(), &[1, 320, 16, 16]);
    }

    #[test]
    fn stage3_downsample_loads_and_runs_forward() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let downsample = StageDownsample::load_stage3(vb.pp("visual.trunk.stages.3.downsample"))
            .expect("stage3 downsample should load");

        let x = Tensor::zeros((1, 320, 16, 16), DType::F32, &device).expect("tensor should build");
        let y = downsample.forward(&x).expect("forward should succeed");

        assert_eq!(y.dims(), &[1, 640, 8, 8]);
    }

    #[test]
    fn loads_stage2_blocks() {
        let (_open_clip_config, _device, vision_model) = load_test_model();
        assert_eq!(vision_model.stage2.len(), 24);
    }

    #[test]
    fn forward_stage2_downsamples_to_16x16_with_320_channels() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 160, 32, 32), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_stage2(&x)
            .expect("stage2 forward should succeed");

        assert_eq!(y.dims(), &[1, 320, 16, 16]);
    }

    #[test]
    fn loads_stage3_positional_embedding_weights() {
        let (_open_clip_config, _device, vision_model) = load_test_model();

        let weight = vision_model.stage3_pos_emb.pos_enc.weight();

        assert_eq!(weight.dims(), &[640, 1, 7, 7]);

        let bias = vision_model
            .stage3_pos_emb
            .pos_enc
            .bias()
            .expect("stage3 positional conv bias should exist");

        assert_eq!(bias.dims(), &[640]);
    }

    #[test]
    fn forward_stage3_prelude_downsamples_to_8x8_with_640_channels() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 320, 16, 16), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_stage3_prelude(&x)
            .expect("stage3 prelude should succeed");

        assert_eq!(y.dims(), &[1, 640, 8, 8]);
    }

    #[test]
    fn loads_stage3_attention_weights() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let attn = Attention2d::load(vb.pp("visual.trunk.stages.3.blocks.0.token_mixer"), 640, 32)
            .expect("stage3 attention should load");

        assert_eq!(attn.qkv_weight.dims(), &[1920, 640]);
        assert_eq!(attn.proj.weight().dims(), &[640, 640]);

        let bias = attn.proj.bias().expect("attention proj bias should exist");
        assert_eq!(bias.dims(), &[640]);
    }

    #[test]
    fn attention2d_runs_forward_on_stage3_shape() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let attn = Attention2d::load(vb.pp("visual.trunk.stages.3.blocks.0.token_mixer"), 640, 32)
            .expect("stage3 attention should load");

        let x = Tensor::zeros((1, 640, 8, 8), DType::F32, &device).expect("tensor should build");
        let y = attn.forward(&x).expect("attention forward should succeed");

        assert_eq!(y.dims(), &[1, 640, 8, 8]);
    }

    #[test]
    fn attention_block2d_runs_forward_on_stage3_shape() {
        let (_open_clip_config, device, _vision_model) = load_test_model();

        let semantic_config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let model_paths = ClipModelPaths::from_root(&semantic_config.model_dir);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_paths.weights.clone()],
                candle_core::DType::F32,
                &device,
            )
            .expect("weights should load")
        };

        let block = AttentionBlock2d::load(vb.pp("visual.trunk.stages.3.blocks.0"), 640, 1920, 32)
            .expect("stage3 attention block should load");

        let x = Tensor::zeros((1, 640, 8, 8), DType::F32, &device).expect("tensor should build");
        let y = block
            .forward(&x)
            .expect("attention block forward should succeed");

        assert_eq!(y.dims(), &[1, 640, 8, 8]);
    }

    #[test]
    fn loads_stage3_blocks() {
        let (_open_clip_config, _device, vision_model) = load_test_model();
        assert_eq!(vision_model.stage3.len(), 4);
    }

    #[test]
    fn forward_stage3_returns_640_channels_at_8x8() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 320, 16, 16), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_stage3(&x)
            .expect("stage3 forward should succeed");

        assert_eq!(y.dims(), &[1, 640, 8, 8]);
    }

    #[test]
    fn stage3_output_feeds_final_conv() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 320, 16, 16), DType::F32, &device).expect("tensor should build");

        let x = vision_model
            .forward_stage3(&x)
            .expect("stage3 forward should succeed");

        let y = vision_model
            .forward_final_conv(&x)
            .expect("final conv forward should succeed");

        assert_eq!(y.dims(), &[1, 1280, 8, 8]);
    }

    #[test]
    fn forward_features_produces_1280_channels_at_8x8() {
        let (_open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 3, 256, 256), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward_features(&x)
            .expect("vision backbone forward should succeed");

        assert_eq!(y.dims(), &[1, 1280, 8, 8]);
    }

    #[test]
    fn forward_projects_image_to_embed_dim() {
        let (open_clip_config, device, vision_model) = load_test_model();

        let x = Tensor::zeros((1, 3, 256, 256), DType::F32, &device).expect("tensor should build");

        let y = vision_model
            .forward(&x)
            .expect("vision model forward should succeed");

        assert_eq!(y.dims(), &[1, open_clip_config.model_cfg.embed_dim]);
    }
}

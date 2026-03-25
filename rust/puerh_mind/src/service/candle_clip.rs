use anyhow::{Result, bail};

use candle_nn::VarBuilder;
use tokenizers::Tokenizer;

use crate::config::SemanticConfig;

use crate::service::clip_text::ClipTextModel;
use crate::service::clip_vision::ClipVisionModel;

use crate::service::embedding::EmbeddingEngine;
use crate::service::model_assets::ClipModelPaths;
use crate::service::open_clip_config::OpenClipConfig;
use crate::service::text_inputs::{TextBatch, TextTensors};

use candle_core::Device;
use candle_core::Tensor;

pub struct CandleClipEngine {
    model_id: String,
    open_clip_config: OpenClipConfig,
    tokenizer: Tokenizer,
    device: Device,
    text_model: ClipTextModel,
    vision_model: ClipVisionModel,
}

impl CandleClipEngine {
    fn select_device() -> Result<Device> {
        match std::env::var("PUERH_MIND_DEVICE") {
            Ok(value) if value.eq_ignore_ascii_case("cpu") => Ok(Device::Cpu),
            Ok(value) if value.eq_ignore_ascii_case("metal") => Device::new_metal(0)
                .map_err(|e| anyhow::anyhow!("failed to initialize metal device: {e}")),
            Ok(value) => bail!(
                "unsupported PUERH_MIND_DEVICE value {value:?}, expected \"cpu\" or \"metal\""
            ),
            Err(std::env::VarError::NotPresent) => Device::metal_if_available(0)
                .map_err(|e| anyhow::anyhow!("failed to initialize preferred device: {e}")),
            Err(err) => bail!("failed to read PUERH_MIND_DEVICE: {err}"),
        }
    }

    pub fn new(config: &SemanticConfig) -> Result<Self> {
        let model_paths = ClipModelPaths::from_root(&config.model_dir);
        model_paths.validate()?;

        let open_clip_config = OpenClipConfig::load(&model_paths.config)?;
        let tokenizer = Tokenizer::from_file(&model_paths.tokenizer_json)
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let device = Self::select_device()?;

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

        let vision_model =
            ClipVisionModel::load(vb.pp("visual"), open_clip_config.model_cfg.embed_dim)
                .expect("vision model should load");

        Ok(Self {
            model_id: config.model_id.clone(),
            open_clip_config,
            tokenizer,
            device,
            text_model,
            vision_model,
        })
    }

    fn prepare_text_batch(&self, texts: &[&str]) -> Result<TextBatch> {
        if texts.is_empty() {
            bail!("text batch must not be empty");
        }

        let max_len = self.open_clip_config.model_cfg.text_cfg.context_length;
        let mut input_ids = Vec::with_capacity(texts.len());
        let mut attention_mask = Vec::with_capacity(texts.len());

        for (index, text) in texts.iter().enumerate() {
            if text.trim().is_empty() {
                bail!("text at batch index {index} must not be empty");
            }

            let encoding = self
                .tokenizer
                .encode(*text, true)
                .map_err(|e| anyhow::anyhow!("failed to tokenize text: {e}"))?;

            let mut row_input_ids = encoding.get_ids().to_vec();
            let mut row_attention_mask = encoding.get_attention_mask().to_vec();

            row_input_ids.truncate(max_len);
            row_attention_mask.truncate(max_len);

            while row_input_ids.len() < max_len {
                row_input_ids.push(0);
                row_attention_mask.push(0);
            }

            input_ids.push(row_input_ids);
            attention_mask.push(row_attention_mask);
        }

        Ok(TextBatch {
            input_ids,
            attention_mask,
        })
    }

    fn prepare_text_input(&self, text: &str) -> Result<TextBatch> {
        self.prepare_text_batch(&[text])
    }

    fn text_batch_to_tensors(&self, batch: &TextBatch) -> anyhow::Result<TextTensors> {
        if batch.input_ids.is_empty() {
            anyhow::bail!("text batch must not be empty");
        }
        if batch.input_ids.len() != batch.attention_mask.len() {
            anyhow::bail!(
                "input_ids batch size {} did not match attention_mask batch size {}",
                batch.input_ids.len(),
                batch.attention_mask.len()
            );
        }

        let batch_size = batch.input_ids.len();
        let seq_len = batch.input_ids[0].len();
        if seq_len == 0 {
            anyhow::bail!("text batch sequence length must not be zero");
        }

        for (row_idx, row) in batch.input_ids.iter().enumerate() {
            if row.len() != seq_len {
                anyhow::bail!(
                    "input_ids row {row_idx} had length {}, expected {seq_len}",
                    row.len()
                );
            }
        }

        for (row_idx, row) in batch.attention_mask.iter().enumerate() {
            if row.len() != seq_len {
                anyhow::bail!(
                    "attention_mask row {row_idx} had length {}, expected {seq_len}",
                    row.len()
                );
            }
        }

        let input_ids = Tensor::from_vec(
            batch
                .input_ids
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (batch_size, seq_len),
            &self.device,
        )?;

        let attention_mask = Tensor::from_vec(
            batch
                .attention_mask
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>(),
            (batch_size, seq_len),
            &self.device,
        )?;

        Ok(TextTensors {
            input_ids,
            attention_mask,
        })
    }

    fn prepare_text_tensors(&self, text: &str) -> anyhow::Result<TextTensors> {
        let batch = self.prepare_text_input(text)?;
        self.text_batch_to_tensors(&batch)
    }

    fn prepare_text_batch_tensors(&self, texts: &[&str]) -> anyhow::Result<TextTensors> {
        let batch = self.prepare_text_batch(texts)?;
        self.text_batch_to_tensors(&batch)
    }

    fn l2_normalize(mut embedding: Vec<f32>) -> anyhow::Result<Vec<f32>> {
        if embedding.iter().any(|v| !v.is_finite()) {
            anyhow::bail!("embedding contains non-finite values");
        }

        let norm = embedding
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();

        if norm == 0.0 {
            anyhow::bail!("text embedding norm is zero");
        }

        for value in &mut embedding {
            *value = (*value as f64 / norm) as f32;
        }

        Ok(embedding)
    }

    pub fn forward_text_embeddings(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        let tensors = self.prepare_text_batch_tensors(texts)?;

        let feature = self.text_model.forward_text_feature(&tensors.input_ids)?;
        feature
            .to_vec2::<f32>()?
            .into_iter()
            .map(Self::l2_normalize)
            .collect()
    }

    fn forward_text_embedding(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut embeddings = self.forward_text_embeddings(&[text])?;
        if embeddings.len() != 1 {
            anyhow::bail!("expected one text embedding row, got {}", embeddings.len());
        }
        Ok(embeddings.remove(0))
    }

    /* Image Part */
    fn prepare_image_tensor(&self, rgb: &image::RgbImage) -> anyhow::Result<Tensor> {
        let target = self.open_clip_config.model_cfg.vision_cfg.image_size as u32;

        let (src_w, src_h) = rgb.dimensions();
        if src_w == 0 || src_h == 0 {
            anyhow::bail!("image must not be empty");
        }

        let scale = if src_w < src_h {
            target as f32 / src_w as f32
        } else {
            target as f32 / src_h as f32
        };

        let resized_w = ((src_w as f32) * scale).round() as u32;
        let resized_h = ((src_h as f32) * scale).round() as u32;

        let resized = image::imageops::resize(
            rgb,
            resized_w,
            resized_h,
            image::imageops::FilterType::Triangle,
        );

        let crop_x = (resized_w - target) / 2;
        let crop_y = (resized_h - target) / 2;

        let cropped =
            image::imageops::crop_imm(&resized, crop_x, crop_y, target, target).to_image();

        let mut data = Vec::with_capacity((3 * target * target) as usize);

        for channel in 0..3usize {
            for y in 0..target {
                for x in 0..target {
                    let pixel = cropped.get_pixel(x, y);
                    let value = pixel[channel] as f32 / 255.0;
                    data.push(value);
                }
            }
        }

        let tensor =
            Tensor::from_vec(data, (1, 3, target as usize, target as usize), &self.device)?;

        Ok(tensor)
    }

    fn forward_image_embedding(&self, rgb: &image::RgbImage) -> anyhow::Result<Vec<f32>> {
        let image = self.prepare_image_tensor(rgb)?;

        let feature = self.vision_model.forward(&image)?;
        let mut rows = feature.to_vec2::<f32>()?;
        if rows.len() != 1 {
            anyhow::bail!("expected one image feature row, got {}", rows.len());
        }

        Self::l2_normalize(rows.remove(0))
    }
}

impl EmbeddingEngine for CandleClipEngine {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.forward_text_embedding(text)
    }

    fn embed_image(&self, rgb: &image::RgbImage) -> Result<Vec<f32>> {
        self.forward_image_embedding(rgb)
    }

    fn default_text_model_name(&self) -> &str {
        &self.model_id
    }

    fn default_image_model_name(&self) -> &str {
        &self.model_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SemanticConfig;
    use image::ImageReader;
    use std::path::Path;

    fn make_test_engine() -> CandleClipEngine {
        unsafe {
            std::env::set_var("PUERH_MIND_DEVICE", "cpu");
        }
        let config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        CandleClipEngine::new(&config).expect("engine should load")
    }
    #[test]
    fn loads_tokenizer_and_encodes_text() {
        let engine = make_test_engine();

        let encoding = engine
            .tokenizer
            .encode("a red tea cake", true)
            .expect("tokenizer should encode text");

        assert!(!encoding.get_ids().is_empty());
    }

    #[test]
    fn prepare_text_input_returns_fixed_length() {
        let engine = make_test_engine();
        let batch = engine.prepare_text_input("a red tea cake").unwrap();

        assert_eq!(batch.input_ids.len(), 1);
        assert_eq!(batch.attention_mask.len(), 1);
        assert_eq!(batch.input_ids[0].len(), 77);
        assert_eq!(batch.attention_mask[0].len(), 77);
        assert!(batch.attention_mask[0].iter().any(|&v| v == 1));
    }

    #[test]
    fn prepare_text_input_rejects_empty_text() {
        let engine = make_test_engine();
        let err = engine.prepare_text_input("   ").unwrap_err();

        assert!(
            err.to_string()
                .contains("text at batch index 0 must not be empty")
        );
    }

    #[test]
    fn prepare_text_input_truncates_long_text() {
        let engine = make_test_engine();
        let long_text = "tea ".repeat(200);
        let batch = engine.prepare_text_input(&long_text).unwrap();

        assert_eq!(batch.input_ids[0].len(), 77);
        assert_eq!(batch.attention_mask[0].len(), 77);
    }

    #[test]
    fn prepare_text_tensors_returns_expected_shape() {
        let engine = make_test_engine();
        let tensors = engine.prepare_text_tensors("a red tea cake").unwrap();

        assert_eq!(tensors.input_ids.dims(), &[1, 77]);
        assert_eq!(tensors.attention_mask.dims(), &[1, 77]);
        assert_eq!(tensors.input_ids.dtype(), candle_core::DType::U32);
        assert_eq!(tensors.attention_mask.dtype(), candle_core::DType::U32);
    }

    #[test]
    fn prepare_text_batch_tensors_returns_expected_shape() {
        let engine = make_test_engine();
        let tensors = engine
            .prepare_text_batch_tensors(&["dog", "cat", "car"])
            .unwrap();

        assert_eq!(tensors.input_ids.dims(), &[3, 77]);
        assert_eq!(tensors.attention_mask.dims(), &[3, 77]);
        assert_eq!(tensors.input_ids.dtype(), candle_core::DType::U32);
        assert_eq!(tensors.attention_mask.dtype(), candle_core::DType::U32);
    }

    #[test]
    fn model_context_length_check() {
        let engine = make_test_engine();
        assert_eq!(
            engine.open_clip_config.model_cfg.text_cfg.context_length,
            77
        );
        assert_eq!(engine.open_clip_config.model_cfg.text_cfg.vocab_size, 49408);
        assert_eq!(
            engine.open_clip_config.model_cfg.vision_cfg.timm_model_name,
            "fastvit_mci2"
        );
    }

    #[test]
    fn inspect_safetensor_keys() {
        let bytes = std::fs::read("./models/mobileclip2-s2-openclip/open_clip_model.safetensors")
            .expect("should read safetensors");

        let tensors =
            safetensors::SafeTensors::deserialize(&bytes).expect("should parse safetensors");

        let mut names: Vec<_> = tensors.names().into_iter().collect();
        names.sort();

        for name in names.iter().take(80) {
            println!("{name}");
        }

        assert!(!names.is_empty());
    }

    use std::collections::BTreeMap;
    #[test]
    fn inspect_safetensor_prefixes() {
        let bytes = std::fs::read("./models/mobileclip2-s2-openclip/open_clip_model.safetensors")
            .expect("should read safetensors");

        let tensors =
            safetensors::SafeTensors::deserialize(&bytes).expect("should parse safetensors");

        let mut counts = BTreeMap::new();

        for name in tensors.names() {
            let prefix = name.split('.').next().unwrap_or(name);
            *counts.entry(prefix.to_string()).or_insert(0usize) += 1;
        }

        for (prefix, count) in counts {
            println!("{prefix}: {count}");
        }
    }

    #[test]
    fn embeds_text_with_clip_text_model() {
        let config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let engine = CandleClipEngine::new(&config).expect("engine should load");

        let embedding = engine
            .embed_text("a red tea cake")
            .expect("text embedding should succeed");

        assert_eq!(embedding.len(), engine.open_clip_config.model_cfg.embed_dim);
    }

    #[test]
    fn normalizes_text_embedding_to_unit_length() {
        let config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let engine = CandleClipEngine::new(&config).expect("engine should load");

        let embedding = engine
            .embed_text("a red tea cake")
            .expect("text embedding should succeed");

        let norm = embedding
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();

        assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");
    }

    #[test]
    fn prepare_image_tensor_returns_nchw_f32_shape() {
        let engine = make_test_engine();

        let width = 320;
        let height = 200;
        let rgb = image::RgbImage::from_fn(width, height, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });

        let tensor = engine
            .prepare_image_tensor(&rgb)
            .expect("image tensor should be prepared");

        assert_eq!(tensor.dims(), &[1, 3, 256, 256]);
        assert_eq!(tensor.dtype(), candle_core::DType::F32);

        let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(values.iter().all(|v| *v >= 0.0 && *v <= 1.0));
    }

    #[test]
    fn prepare_image_tensor_resizes_shortest_edge_then_center_crops() {
        let engine = make_test_engine();

        let width = 400;
        let height = 200;
        let rgb = image::RgbImage::from_fn(width, height, |x, _y| {
            let red = if (80..320).contains(&x) { 255 } else { 0 };
            image::Rgb([red, 0, 0])
        });

        let tensor = engine
            .prepare_image_tensor(&rgb)
            .expect("image tensor should be prepared");

        let chw = tensor.squeeze(0).unwrap();
        let values = chw.to_vec3::<f32>().unwrap();

        let red = &values[0];
        let left = red[128][0];
        let center = red[128][128];
        let right = red[128][255];

        assert!(left > 0.85, "left={left}");
        assert!(center > 0.95, "center={center}");
        assert!(right > 0.85, "right={right}");
    }

    #[test]
    fn embeds_image_with_clip_vision_model() {
        let engine = make_test_engine();

        let rgb = image::RgbImage::from_pixel(300, 200, image::Rgb([128, 64, 32]));
        let embedding = engine
            .embed_image(&rgb)
            .expect("image embedding should succeed");

        assert_eq!(embedding.len(), engine.open_clip_config.model_cfg.embed_dim);
    }

    #[test]
    fn vision_forward_stages_remain_finite_for_image_input() {
        let engine = make_test_engine();

        let rgb = image::RgbImage::from_fn(300, 200, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });

        fn tensor_is_finite(tensor: &Tensor) -> bool {
            tensor
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .all(|v| v.is_finite())
        }

        let x = engine
            .prepare_image_tensor(&rgb)
            .expect("image tensor should be prepared");
        assert!(tensor_is_finite(&x), "prepared image tensor was not finite");

        let x = engine
            .vision_model
            .forward_stem(&x)
            .expect("stem should succeed");
        assert!(tensor_is_finite(&x), "stem output was not finite");

        let x = engine
            .vision_model
            .forward_stage0(&x)
            .expect("stage0 should succeed");
        assert!(tensor_is_finite(&x), "stage0 output was not finite");

        let x = engine
            .vision_model
            .forward_stage1(&x)
            .expect("stage1 should succeed");
        assert!(tensor_is_finite(&x), "stage1 output was not finite");

        let x = engine
            .vision_model
            .forward_stage2(&x)
            .expect("stage2 should succeed");
        assert!(tensor_is_finite(&x), "stage2 output was not finite");

        let x = engine
            .vision_model
            .forward_stage3(&x)
            .expect("stage3 should succeed");
        assert!(tensor_is_finite(&x), "stage3 output was not finite");

        let x = engine
            .vision_model
            .forward_final_conv(&x)
            .expect("final conv should succeed");
        assert!(tensor_is_finite(&x), "final conv output was not finite");

        let x = engine
            .vision_model
            .forward_head(&x)
            .expect("head should succeed");
        assert!(tensor_is_finite(&x), "head output was not finite");
    }

    #[test]
    fn normalizes_image_embedding_to_unit_length() {
        let engine = make_test_engine();

        let rgb = image::RgbImage::from_fn(300, 200, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });
        let embedding = engine
            .embed_image(&rgb)
            .expect("image embedding should succeed");

        let norm = embedding
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();

        assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");
    }

    #[test]
    fn embeds_text_batch_with_clip_text_model() {
        let engine = make_test_engine();

        let embeddings = engine
            .forward_text_embeddings(&["dog", "cat", "car"])
            .expect("batched text embeddings should succeed");

        assert_eq!(embeddings.len(), 3);
        for embedding in embeddings {
            assert_eq!(embedding.len(), engine.open_clip_config.model_cfg.embed_dim);

            let norm = embedding
                .iter()
                .map(|v| (*v as f64) * (*v as f64))
                .sum::<f64>()
                .sqrt();

            assert!((norm - 1.0).abs() < 1e-3, "norm was {norm}");
        }
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }

    #[test]
    fn dog_image_ranks_batched_text_tags() {
        let engine = make_test_engine();
        let image_path = Path::new("./testdata/dog.png");
        let image = ImageReader::open(image_path)
            .expect("dog test image should open")
            .decode()
            .expect("dog test image should decode")
            .to_rgb8();

        let image_embedding = engine
            .embed_image(&image)
            .expect("image embedding should succeed");

        let labels = ["dog", "cat", "car"];
        let text_embeddings = engine
            .forward_text_embeddings(&labels)
            .expect("batched text embeddings should succeed");

        let mut scored_labels = labels
            .iter()
            .zip(text_embeddings.iter())
            .map(|(label, embedding)| (*label, cosine_similarity(&image_embedding, embedding)))
            .collect::<Vec<_>>();
        scored_labels.sort_by(|(_, left), (_, right)| right.total_cmp(left));

        let (best_label, best_score) = scored_labels[0];
        let (_, worst_score) = scored_labels[scored_labels.len() - 1];

        assert!(labels.contains(&best_label));
        assert!(best_score.is_finite());
        assert!(worst_score.is_finite());
        assert!(best_score > worst_score);
    }
}

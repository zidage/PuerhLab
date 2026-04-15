use std::path::PathBuf;

use anyhow::{Context, bail};
use hf_hub::{Repo, RepoType, api::sync::ApiBuilder};

pub const MOBILECLIP2_ONNX_REPO: &str = "plhery/mobileclip2-onnx";
pub const MOBILECLIP2_ONNX_REVISION: &str = "ba95759a5bdbaca53e9111e2550a76ec09c8fd9e";
pub const MOBILECLIP2_ONNX_VARIANT: &str = "onnx/s2";

pub struct ClipModelPaths {
    pub root: PathBuf,
    pub text_model: PathBuf,
    pub vision_model: PathBuf,
    pub onnx_config: PathBuf,
    pub preprocess_config: PathBuf,
    pub tokenizer_json: PathBuf,
    pub tokenizer_config: PathBuf,
}

impl ClipModelPaths {
    pub fn from_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        let onnx_root = root.join(MOBILECLIP2_ONNX_VARIANT);
        Self {
            text_model: onnx_root.join("text_model.onnx"),
            vision_model: onnx_root.join("vision_model.onnx"),
            onnx_config: onnx_root.join("config.json"),
            preprocess_config: onnx_root.join("preprocessor_config.json"),
            tokenizer_json: root.join("tokenizer.json"),
            tokenizer_config: root.join("tokenizer_config.json"),
            root,
        }
    }

    pub fn ensure_present(&self) -> anyhow::Result<()> {
        std::fs::create_dir_all(&self.root).with_context(|| {
            format!(
                "failed to create model root directory {}",
                self.root.display()
            )
        })?;

        self.download_missing_assets()?;
        self.validate()
    }

    fn download_missing_assets(&self) -> anyhow::Result<()> {
        let assets = [
            ("onnx/s2/text_model.onnx", &self.text_model),
            ("onnx/s2/vision_model.onnx", &self.vision_model),
            ("onnx/s2/config.json", &self.onnx_config),
            ("onnx/s2/preprocessor_config.json", &self.preprocess_config),
            ("tokenizer.json", &self.tokenizer_json),
            ("tokenizer_config.json", &self.tokenizer_config),
        ];

        if assets.iter().all(|(_, local_path)| local_path.exists()) {
            return Ok(());
        }

        let api = ApiBuilder::from_env()
            .with_progress(false)
            .build()
            .context("failed to initialize Hugging Face API client")?;

        let repo = api.repo(Repo::with_revision(
            MOBILECLIP2_ONNX_REPO.to_string(),
            RepoType::Model,
            MOBILECLIP2_ONNX_REVISION.to_string(),
        ));

        for (remote_path, local_path) in assets {
            if local_path.exists() {
                continue;
            }

            if let Some(parent) = local_path.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!(
                        "failed to create model asset directory {}",
                        parent.display()
                    )
                })?;
            }

            let downloaded = repo.get(remote_path).with_context(|| {
                format!(
                    "failed to download {remote_path} from repo {MOBILECLIP2_ONNX_REPO}@{MOBILECLIP2_ONNX_REVISION}"
                )
            })?;

            std::fs::copy(&downloaded, local_path).with_context(|| {
                format!(
                    "failed to copy downloaded asset {} to {}",
                    downloaded.display(),
                    local_path.display()
                )
            })?;
        }

        Ok(())
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.root.exists() {
            bail!("missing model root directory: {}", self.root.display());
        }

        if !self.text_model.exists() {
            bail!("missing text model file: {}", self.text_model.display());
        }

        if !self.vision_model.exists() {
            bail!("missing vision model file: {}", self.vision_model.display());
        }

        if !self.onnx_config.exists() {
            bail!("missing onnx config file: {}", self.onnx_config.display());
        }

        if !self.preprocess_config.exists() {
            bail!(
                "missing preprocess config file: {}",
                self.preprocess_config.display()
            );
        }

        if !self.tokenizer_json.exists() {
            bail!(
                "missing tokenizer_json file: {}",
                self.tokenizer_json.display()
            );
        }

        if !self.tokenizer_config.exists() {
            bail!(
                "missing tokenizer_config file: {}",
                self.tokenizer_config.display()
            );
        }

        Ok(())
    }
}

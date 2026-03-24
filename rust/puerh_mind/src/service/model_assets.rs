use std::path::PathBuf;

use anyhow::bail;

pub struct ClipModelPaths {
    pub root: PathBuf,
    pub weights: PathBuf,
    pub config: PathBuf,
    pub tokenizer_json: PathBuf,
    pub vocab: PathBuf,
    pub merges: PathBuf,
    pub tokenizer_config: PathBuf,
}

impl ClipModelPaths {
    pub fn from_root(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            weights: root.join("open_clip_model.safetensors"),
            config: root.join("open_clip_config.json"),
            tokenizer_json: root.join("tokenizer.json"),
            vocab: root.join("vocab.json"),
            merges: root.join("merges.txt"),
            tokenizer_config: root.join("tokenizer_config.json"),
            root,
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if !self.root.exists() {
            bail!("missing model root directory: {}", self.root.display());
        }

        if !self.weights.exists() {
            bail!("missing weights file: {}", self.weights.display());
        }

        if !self.config.exists() {
            bail!("missing config file: {}", self.config.display());
        }

        if !self.tokenizer_json.exists() {
            bail!("missing tokenizer_json file: {}", self.tokenizer_json.display());
        }

        if !self.vocab.exists() {
            bail!("missing vocab file: {}", self.vocab.display());
        }

        if !self.merges.exists() {
            bail!("missing merges file: {}", self.merges.display());
        }

        if !self.tokenizer_config.exists() {
            bail!("missing tokenizer_config file: {}", self.tokenizer_config.display());
        }

        Ok(())
    }
}

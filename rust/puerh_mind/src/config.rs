pub struct AppConfig {
    pub host: String,
    pub port: u16,
    pub semantic: SemanticConfig,
}

pub struct SemanticConfig {
    pub model_id: String,
    pub model_dir: String,
}

impl AppConfig {
    pub fn load() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 50051,
            semantic: SemanticConfig {
                model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
                model_dir: "./models/mobileclip2-s2-openclip".to_string(),
            },
        }
    }

    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

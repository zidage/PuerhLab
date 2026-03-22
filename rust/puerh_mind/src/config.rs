pub struct AppConfig {
    pub host: String,
    pub port: u16,
}

impl AppConfig {
    pub fn load() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 50051,
        }
    }

    pub fn listen_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

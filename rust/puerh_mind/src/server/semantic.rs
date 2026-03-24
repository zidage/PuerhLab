use std::sync::Arc;
use std::time::Instant;

use tracing::info;

use tonic::{Request, Response, Status};

use crate::proto::semantic::{
    EmbedImageRequest, EmbedTextRequest, EmbeddingResponse, PingRequest, PingResponse,
    semantic_service_server::SemanticService,
};
use crate::service::embedding::EmbeddingEngine;

pub struct SemanticServiceImpl {
    engine: Arc<dyn EmbeddingEngine>,
}

impl SemanticServiceImpl {
    pub fn new(engine: Arc<dyn EmbeddingEngine>) -> Self {
        Self { engine }
    }

    fn validate_text_request(&self, req: &EmbedTextRequest) -> Result<(), Status> {
        if req.text.trim().is_empty() {
            return Err(Status::invalid_argument("text must not be empty"));
        }
        Ok(())
    }

    fn decode_rgb8_image(&self, image_bytes: &[u8]) -> Result<image::RgbImage, Status> {
        if image_bytes.is_empty() {
            return Err(Status::invalid_argument("image_bytes must not be empty"));
        }

        let image = image::load_from_memory(image_bytes)
            .map_err(|e| Status::invalid_argument(format!("failed to decode image: {e}")))?;

        Ok(image.to_rgb8())
    }
}

#[tonic::async_trait]
impl SemanticService for SemanticServiceImpl {
    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        info!("[SemanticService]: received Ping request");

        let start = std::time::Instant::now();

        let inner = request.into_inner();
        let request_id = inner.request_id;

        let response = PingResponse {
            request_id,
            message: "pong".to_string(),
            elapsed_ms: start.elapsed().as_millis() as u64,
        };

        Ok(Response::new(response))
    }

    async fn embed_text(
        &self,
        request: Request<EmbedTextRequest>,
    ) -> Result<Response<EmbeddingResponse>, Status> {
        info!("[SemanticService]: received EmbedText request");
        let start = Instant::now();
        let req = request.into_inner();

        self.validate_text_request(&req)?;

        let embedding = self
            .engine
            .embed_text(&req.text)
            .map_err(|e| Status::internal(format!("failed to embed text: {e}")))?;
        let dimension = embedding.len() as u32;

        let response = EmbeddingResponse {
            request_id: req.request_id,
            embedding,
            dimension,
            model_name: if req.model_name.is_empty() {
                self.engine.default_text_model_name().to_string()
            } else {
                req.model_name
            },
            elapsed_ms: start.elapsed().as_millis() as u64,
        };

        Ok(Response::new(response))
    }

    async fn embed_image(
        &self,
        request: Request<EmbedImageRequest>,
    ) -> Result<Response<EmbeddingResponse>, Status> {
        info!("[SemanticService]: received EmbedImg request");
        let start = Instant::now();
        let req = request.into_inner();

        let rgb = self.decode_rgb8_image(&req.image_bytes)?;

        let embedding = self
            .engine
            .embed_image(&rgb)
            .map_err(|e| Status::internal(format!("failed to embed image: {e}")))?;
        let dimension = embedding.len() as u32;

        let response = EmbeddingResponse {
            request_id: req.request_id,
            embedding,
            dimension,
            model_name: if req.model_name.is_empty() {
                self.engine.default_image_model_name().to_string()
            } else {
                req.model_name
            },
            elapsed_ms: start.elapsed().as_millis() as u64,
        };

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tonic::Request;

    use super::*;
    use crate::config::SemanticConfig;
    use crate::service::candle_clip::CandleClipEngine;

    fn test_semantic_config() -> SemanticConfig {
        SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        }
    }

    #[test]
    fn uses_configured_model_id_as_default_name() {
        let config = SemanticConfig {
            model_id: "timm/MobileCLIP2-S2-OpenCLIP".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let engine = CandleClipEngine::new(&config).expect("engine should load");

        assert_eq!(engine.default_text_model_name(), config.model_id);
        assert_eq!(engine.default_image_model_name(), config.model_id);
    }

    #[tokio::test]
    async fn embeds_text_request_with_candle_engine() {
        let config = test_semantic_config();
        let engine = Arc::new(CandleClipEngine::new(&config).expect("engine should load"));
        let service = SemanticServiceImpl::new(engine);

        let request = Request::new(EmbedTextRequest {
            request_id: "test-1".to_string(),
            text: "a red tea cake".to_string(),
            model_name: String::new(),
        });

        let response = service
            .embed_text(request)
            .await
            .expect("embed text should succeed")
            .into_inner();

        assert_eq!(response.request_id, "test-1");
        assert_eq!(response.dimension as usize, response.embedding.len());
        assert!(!response.embedding.is_empty());
        assert_eq!(response.model_name, "timm/MobileCLIP2-S2-OpenCLIP");
        assert!(response.elapsed_ms <= u64::MAX);
    }
}

use std::time::Instant;
use tonic::{Request, Response, Status};
use tracing::info;

use crate::proto::semantic::{
    EmbedImageRequest, EmbedTextRequest, EmbeddingResponse, PingRequest, PingResponse,
    semantic_service_server::SemanticService,
};

pub struct SemanticServiceImpl;

impl SemanticServiceImpl {
    pub fn new() -> Self {
        Self
    }

    fn mock_text_embedding(&self, text: &str) -> Vec<f32> {
        let len = text.len() as f32;

        vec![
            len,
            len + 1.0,
            len + 2.0,
            len + 3.0,
            len + 4.0,
            len + 5.0,
            len + 6.0,
            len + 7.0,
        ]
    }

    fn mock_image_embedding(&self, image_bytes: &[u8]) -> Vec<f32> {
        let len = image_bytes.len() as f32;

        vec![len, len * 0.5, len * 0.25, len * 0.125, 1.0, 2.0, 3.0, 4.0]
    }

    fn validate_text_request(&self, req: &EmbedTextRequest) -> Result<(), Status> {
        if req.text.trim().is_empty() {
            return Err(Status::invalid_argument("text must not be empty"));
        }
        Ok(())
    }

    fn validate_image_request(&self, req: &EmbedImageRequest) -> Result<(), Status> {
        if req.image_bytes.is_empty() {
            return Err(Status::invalid_argument("image_bytes must not be empty"));
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl SemanticService for SemanticServiceImpl {
    async fn ping(&self, request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
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
        let start = Instant::now();
        let req = request.into_inner();

        self.validate_text_request(&req)?;

        let embedding = self.mock_text_embedding(&req.text);
        let dimension = embedding.len() as u32;

        let response = EmbeddingResponse {
            request_id: req.request_id,
            embedding,
            dimension,
            model_name: if req.model_name.is_empty() {
                "mock-text-v1".to_string()
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
        let start = Instant::now();
        let req = request.into_inner();

        self.validate_image_request(&req)?;

        let embedding = self.mock_image_embedding(&req.image_bytes);
        let dimension = embedding.len() as u32;

        let response = EmbeddingResponse {
            request_id: req.request_id,
            embedding,
            dimension,
            model_name: if req.model_name.is_empty() {
                "mock-image-v1".to_string()
            } else {
                req.model_name
            },
            elapsed_ms: start.elapsed().as_millis() as u64,
        };

        Ok(Response::new(response))
    }
}

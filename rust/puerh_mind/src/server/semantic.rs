use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};
use tokio::time::{Duration, Instant as TokioInstant};
use tracing::info;

use tonic::{Request, Response, Status};

use crate::proto::semantic::{
    EmbedImageRequest, EmbedTextRequest, EmbeddingResponse, PingRequest, PingResponse,
    semantic_service_server::SemanticService,
};
use crate::service::embedding::EmbeddingEngine;

const IMAGE_BATCH_SIZE_CAP: usize = 512;
const IMAGE_BATCH_QUEUE_CAPACITY: usize = 256;
const IMAGE_BATCH_WAIT: Duration = Duration::from_millis(25);

struct PendingImageRequest {
    request_id: String,
    model_name: String,
    rgb: image::RgbImage,
    started_at: Instant,
    response_tx: oneshot::Sender<Result<EmbeddingResponse, Status>>,
}

pub struct SemanticServiceImpl {
    engine: Arc<dyn EmbeddingEngine>,
    image_batch_tx: mpsc::Sender<PendingImageRequest>,
}

impl SemanticServiceImpl {
    pub fn new(engine: Arc<dyn EmbeddingEngine>) -> Self {
        let image_batch_tx = Self::spawn_image_batch_worker(engine.clone());

        Self {
            engine,
            image_batch_tx,
        }
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

    fn spawn_image_batch_worker(
        engine: Arc<dyn EmbeddingEngine>,
    ) -> mpsc::Sender<PendingImageRequest> {
        let (tx, mut rx) = mpsc::channel::<PendingImageRequest>(IMAGE_BATCH_QUEUE_CAPACITY);

        tokio::spawn(async move {
            while let Some(first) = rx.recv().await {
                let mut batch = vec![first];
                let deadline = TokioInstant::now() + IMAGE_BATCH_WAIT;

                while batch.len() < IMAGE_BATCH_SIZE_CAP {
                    let now = TokioInstant::now();
                    if now >= deadline {
                        break;
                    }

                    let remaining = deadline.saturating_duration_since(now);
                    match tokio::time::timeout(remaining, rx.recv()).await {
                        Ok(Some(next)) => batch.push(next),
                        Ok(None) | Err(_) => break,
                    }
                }

                Self::process_image_batch(engine.as_ref(), batch);
            }
        });

        tx
    }

    fn process_image_batch(engine: &dyn EmbeddingEngine, batch: Vec<PendingImageRequest>) {
        let batch_len = batch.len();
        let mut requests = Vec::with_capacity(batch_len);
        let mut images = Vec::with_capacity(batch_len);

        for PendingImageRequest {
            request_id,
            model_name,
            rgb,
            started_at,
            response_tx,
        } in batch
        {
            requests.push((request_id, model_name, started_at, response_tx));
            images.push(rgb);
        }

        info!("[SemanticService]: processing image batch size={batch_len}");

        match engine.embed_images(&images) {
            Ok(embeddings) => {
                if embeddings.len() != batch_len {
                    let status = Status::internal(format!(
                        "batched image embedding count mismatch: expected {batch_len}, got {}",
                        embeddings.len()
                    ));
                    for (_, _, _, response_tx) in requests {
                        let _ = response_tx.send(Err(status.clone()));
                    }
                    return;
                }

                for ((request_id, model_name, started_at, response_tx), embedding) in
                    requests.into_iter().zip(embeddings)
                {
                    let dimension = embedding.len() as u32;
                    let response = EmbeddingResponse {
                        request_id,
                        embedding,
                        dimension,
                        model_name: if model_name.is_empty() {
                            engine.default_image_model_name().to_string()
                        } else {
                            model_name
                        },
                        elapsed_ms: started_at.elapsed().as_millis() as u64,
                    };

                    let _ = response_tx.send(Ok(response));
                }
            }
            Err(err) => {
                let status = Status::internal(format!("failed to embed image batch: {err}"));
                for (_, _, _, response_tx) in requests {
                    let _ = response_tx.send(Err(status.clone()));
                }
            }
        }
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
        let (response_tx, response_rx) = oneshot::channel();
        let pending = PendingImageRequest {
            request_id: req.request_id,
            model_name: req.model_name,
            rgb,
            started_at: start,
            response_tx,
        };

        self.image_batch_tx
            .send(pending)
            .await
            .map_err(|_| Status::unavailable("image batch worker unavailable"))?;

        let response = response_rx
            .await
            .map_err(|_| Status::internal("image batch worker dropped response"))??;

        Ok(Response::new(response))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use anyhow::Result as AnyResult;
    use tonic::Request;

    use super::*;
    use crate::config::SemanticConfig;
    use crate::service::embedding::{EmbeddingEngine, MockEmbeddingEngine};
    use crate::service::ort_clip::OrtClipEngine;

    fn test_semantic_config() -> SemanticConfig {
        SemanticConfig {
            model_id: "plhery/mobileclip2-onnx:s2".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        }
    }

    #[test]
    fn uses_configured_model_id_as_default_name() {
        let config = SemanticConfig {
            model_id: "plhery/mobileclip2-onnx:s2".to_string(),
            model_dir: "./models/mobileclip2-s2-openclip".to_string(),
        };

        let engine = OrtClipEngine::new(&config).expect("engine should load");

        assert_eq!(engine.default_text_model_name(), config.model_id);
        assert_eq!(engine.default_image_model_name(), config.model_id);
    }

    #[tokio::test]
    async fn embeds_text_request_with_ort_engine() {
        let config = test_semantic_config();
        let engine = Arc::new(OrtClipEngine::new(&config).expect("engine should load"));
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
        assert_eq!(response.model_name, "plhery/mobileclip2-onnx:s2");
        assert!(response.elapsed_ms <= u64::MAX);
    }

    #[tokio::test]
    async fn batches_image_requests_up_to_configured_batch_size_and_preserves_request_ids() {
        struct RecordingEngine {
            batches: Mutex<Vec<usize>>,
        }

        impl EmbeddingEngine for RecordingEngine {
            fn embed_text(&self, text: &str) -> AnyResult<Vec<f32>> {
                MockEmbeddingEngine.embed_text(text)
            }

            fn embed_image(&self, rgb: &image::RgbImage) -> AnyResult<Vec<f32>> {
                MockEmbeddingEngine.embed_image(rgb)
            }

            fn embed_images(&self, rgbs: &[image::RgbImage]) -> AnyResult<Vec<Vec<f32>>> {
                self.batches.lock().unwrap().push(rgbs.len());
                MockEmbeddingEngine.embed_images(rgbs)
            }

            fn default_text_model_name(&self) -> &str {
                "mock-text-v1"
            }

            fn default_image_model_name(&self) -> &str {
                "mock-image-v1"
            }
        }

        let engine = Arc::new(RecordingEngine {
            batches: Mutex::new(Vec::new()),
        });
        let service = Arc::new(SemanticServiceImpl::new(engine.clone()));
        let png = {
            let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                16,
                12,
                image::Rgb([64, 128, 192]),
            ));
            let mut cursor = std::io::Cursor::new(Vec::new());
            image
                .write_to(&mut cursor, image::ImageFormat::Png)
                .expect("png encoding should succeed");
            cursor.into_inner()
        };

        let mut tasks = Vec::new();
        const TEST_REQUEST_COUNT: usize = 64;

        for index in 0..TEST_REQUEST_COUNT {
            let service = service.clone();
            let png = png.clone();
            tasks.push(tokio::spawn(async move {
                let request_id = format!("img-{index}");
                let response = service
                    .embed_image(Request::new(EmbedImageRequest {
                        request_id: request_id.clone(),
                        image_bytes: png,
                        image_format_hint: "png".to_string(),
                        model_name: String::new(),
                    }))
                    .await
                    .expect("embed image should succeed")
                    .into_inner();

                (request_id, response)
            }));
        }

        for task in tasks {
            let (request_id, response) = task.await.expect("task should join");
            assert_eq!(response.request_id, request_id);
            assert_eq!(response.dimension as usize, response.embedding.len());
            assert_eq!(response.model_name, "mock-image-v1");
        }

        assert_eq!(
            engine.batches.lock().unwrap().as_slice(),
            &[TEST_REQUEST_COUNT]
        );
    }
}

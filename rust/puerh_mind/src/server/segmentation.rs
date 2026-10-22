//! `SegmentationService` server (master plan section 9.2.1, SM3.A).
//!
//! Converts the proto messages to engine values, runs the engine on the blocking thread pool, and
//! reports each outcome in the `AiResponseHeader`: the gRPC status stays OK, and a failed
//! operation sets the header status, the error code, and the real error message. The service does
//! not keep any state of its own.

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};
use tracing::warn;

use crate::proto::alcedo::ai::{
    AiErrorCode, AiRequestHeader, AiResponseHeader, AiResponseStatus,
    DecodeSegmentationPromptRequest, DecodeSegmentationPromptResponse,
    EndSegmentationSessionRequest, EndSegmentationSessionResponse, GetSegmentationStatusRequest,
    GetSegmentationStatusResponse, LoadSegmentationModelsRequest, LoadSegmentationModelsResponse,
    SegmentationMaskLevel, SegmentationModelResidency, SegmentationPointLabel,
    StartSegmentationSessionRequest, StartSegmentationSessionResponse,
    UnloadSegmentationModelsRequest, UnloadSegmentationModelsResponse,
    segmentation_service_server::SegmentationService,
};
use crate::service::segmentation::engine::{RgbImageView, SegmentationEngine, SegmentationError};
use crate::service::segmentation::model_set::LOGIT_SIZE;
use crate::service::segmentation::sam2::{PointLabel, PromptBox, PromptPoint, SegmentationPrompt};

/// Provider name in the response headers.
const PROVIDER: &str = "local";

pub struct SegmentationServiceImpl {
    engine: Arc<SegmentationEngine>,
}

impl SegmentationServiceImpl {
    pub fn new(engine: Arc<SegmentationEngine>) -> Self {
        Self { engine }
    }

    /// Runs @p operation on the blocking pool with the engine.
    async fn run_blocking<T: Send + 'static>(
        &self,
        operation: impl FnOnce(&SegmentationEngine) -> Result<T, SegmentationError> + Send + 'static,
    ) -> Result<T, SegmentationError> {
        let engine = self.engine.clone();
        tokio::task::spawn_blocking(move || operation(&engine))
            .await
            .unwrap_or_else(|err| {
                Err(SegmentationError::Inference(format!(
                    "segmentation task stopped: {err}"
                )))
            })
    }
}

fn header_of(header: Option<AiRequestHeader>) -> AiRequestHeader {
    header.unwrap_or_default()
}

fn response_header(
    request: &AiRequestHeader,
    task_id: &str,
    started: Instant,
    error: Option<&SegmentationError>,
) -> Option<AiResponseHeader> {
    let (status, error_code, message) = match error {
        None => (
            AiResponseStatus::AiStatusOk,
            AiErrorCode::AiErrorNone,
            String::new(),
        ),
        Some(err) => {
            let (status, code) = match err {
                SegmentationError::ModelNotReady(_) => (
                    AiResponseStatus::AiStatusProviderUnavailable,
                    AiErrorCode::AiErrorModelNotReady,
                ),
                SegmentationError::InvalidInput(_) => (
                    AiResponseStatus::AiStatusInvalidArgument,
                    AiErrorCode::AiErrorInvalidInput,
                ),
                SegmentationError::SessionNotFound(_) => (
                    AiResponseStatus::AiStatusInvalidArgument,
                    AiErrorCode::AiErrorSessionNotFound,
                ),
                SegmentationError::Inference(_) => (
                    AiResponseStatus::AiStatusInternal,
                    AiErrorCode::AiErrorInternal,
                ),
            };
            warn!(
                request_id = %request.request_id,
                task_id,
                error = %err,
                "segmentation request failed"
            );
            (status, code, err.to_string())
        }
    };
    Some(AiResponseHeader {
        request_id: request.request_id.clone(),
        task_id: task_id.to_string(),
        status: status as i32,
        error_code: error_code as i32,
        error_message: message,
        provider: PROVIDER.to_string(),
        model_id: crate::service::segmentation::model_set::MODEL_SET_ID.to_string(),
        elapsed_ms: started.elapsed().as_millis() as i64,
    })
}

/// Converts a decode request to an engine prompt. Rejects an unknown point label.
fn prompt_from_request(
    request: DecodeSegmentationPromptRequest,
) -> Result<(String, SegmentationPrompt), SegmentationError> {
    let points = request
        .points
        .iter()
        .enumerate()
        .map(|(index, point)| {
            let label = match SegmentationPointLabel::try_from(point.label) {
                Ok(SegmentationPointLabel::Add) => PointLabel::Add,
                Ok(SegmentationPointLabel::Subtract) => PointLabel::Subtract,
                Err(_) => {
                    return Err(SegmentationError::InvalidInput(format!(
                        "point {index} has the unknown label {}",
                        point.label
                    )));
                }
            };
            Ok(PromptPoint {
                x: point.x,
                y: point.y,
                label,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let prompt_box = request.r#box.map(|value| PromptBox {
        x0: value.x0,
        y0: value.y0,
        x1: value.x1,
        y1: value.y1,
    });
    Ok((
        request.session_id,
        SegmentationPrompt {
            points,
            prompt_box,
            mask_input: request.mask_input,
        },
    ))
}

#[tonic::async_trait]
impl SegmentationService for SegmentationServiceImpl {
    async fn load_segmentation_models(
        &self,
        request: Request<LoadSegmentationModelsRequest>,
    ) -> Result<Response<LoadSegmentationModelsResponse>, Status> {
        let started = Instant::now();
        let request = request.into_inner();
        let header = header_of(request.header);
        let directory = std::path::PathBuf::from(request.model_directory);
        let device = request.device;
        let result = self
            .run_blocking(move |engine| engine.load(&directory, &device))
            .await;
        Ok(Response::new(LoadSegmentationModelsResponse {
            header: response_header(
                &header,
                "segmentation.load_models",
                started,
                result.as_ref().err(),
            ),
        }))
    }

    async fn unload_segmentation_models(
        &self,
        request: Request<UnloadSegmentationModelsRequest>,
    ) -> Result<Response<UnloadSegmentationModelsResponse>, Status> {
        let started = Instant::now();
        let header = header_of(request.into_inner().header);
        let result = self.run_blocking(|engine| engine.unload()).await;
        Ok(Response::new(UnloadSegmentationModelsResponse {
            header: response_header(
                &header,
                "segmentation.unload_models",
                started,
                result.as_ref().err(),
            ),
        }))
    }

    async fn get_segmentation_status(
        &self,
        request: Request<GetSegmentationStatusRequest>,
    ) -> Result<Response<GetSegmentationStatusResponse>, Status> {
        let started = Instant::now();
        let header = header_of(request.into_inner().header);
        let result = self.engine.status();
        let mut response = GetSegmentationStatusResponse {
            header: response_header(
                &header,
                "segmentation.status",
                started,
                result.as_ref().err(),
            ),
            ..Default::default()
        };
        if let Ok(status) = result {
            response.loaded = status.loaded;
            response.model_set_id = status.model_set_id.to_string();
            response.model_directory = status.model_directory;
            response.device = status.device;
            response.backend = status.backend;
            response.models = status
                .models
                .into_iter()
                .map(|model| SegmentationModelResidency {
                    role: model.role.as_str().to_string(),
                    device: model.device,
                    load_ms: model.load_ms as i64,
                })
                .collect();
            response.session_count = status.session_count as u32;
        }
        Ok(Response::new(response))
    }

    async fn start_segmentation_session(
        &self,
        request: Request<StartSegmentationSessionRequest>,
    ) -> Result<Response<StartSegmentationSessionResponse>, Status> {
        let started = Instant::now();
        let request = request.into_inner();
        let header = header_of(request.header);
        let (width, height, pixels) = (request.width, request.height, request.rgb8_pixels);
        let result = self
            .run_blocking(move |engine| {
                engine.start_session(RgbImageView {
                    width,
                    height,
                    rgb8: &pixels,
                })
            })
            .await;
        let mut response = StartSegmentationSessionResponse {
            header: response_header(
                &header,
                "segmentation.start_session",
                started,
                result.as_ref().err(),
            ),
            ..Default::default()
        };
        if let Ok(session) = result {
            response.session_id = session.session_id;
            response.encoder_ms = session.encoder_ms as i64;
        }
        Ok(Response::new(response))
    }

    async fn decode_segmentation_prompt(
        &self,
        request: Request<DecodeSegmentationPromptRequest>,
    ) -> Result<Response<DecodeSegmentationPromptResponse>, Status> {
        let started = Instant::now();
        let mut request = request.into_inner();
        let header = header_of(request.header.take());
        let result = match prompt_from_request(request) {
            Ok((session_id, prompt)) => {
                self.run_blocking(move |engine| engine.decode(&session_id, prompt))
                    .await
            }
            Err(err) => Err(err),
        };
        let mut response = DecodeSegmentationPromptResponse {
            header: response_header(
                &header,
                "segmentation.decode_prompt",
                started,
                result.as_ref().err(),
            ),
            logit_size: LOGIT_SIZE as u32,
            ..Default::default()
        };
        if let Ok(decoded) = result {
            response.selected_index = decoded.masks.selected_index as u32;
            response.decoder_ms = decoded.decoder_ms as i64;
            response.masks = decoded
                .masks
                .masks
                .into_iter()
                .map(|level| SegmentationMaskLevel {
                    logits: level.logits,
                    score: level.score,
                    token: level.token,
                    area: level.area,
                })
                .collect();
        }
        Ok(Response::new(response))
    }

    async fn end_segmentation_session(
        &self,
        request: Request<EndSegmentationSessionRequest>,
    ) -> Result<Response<EndSegmentationSessionResponse>, Status> {
        let started = Instant::now();
        let request = request.into_inner();
        let header = header_of(request.header);
        let result = self.engine.end_session(&request.session_id);
        Ok(Response::new(EndSegmentationSessionResponse {
            header: response_header(
                &header,
                "segmentation.end_session",
                started,
                result.as_ref().err(),
            ),
            ended: result.unwrap_or(false),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::alcedo::ai::SegmentationPoint;
    use crate::service::segmentation::engine::test_support::{FakeFactory, model_directory};

    fn service() -> SegmentationServiceImpl {
        SegmentationServiceImpl::new(Arc::new(SegmentationEngine::new(Arc::new(
            FakeFactory::new(),
        ))))
    }

    fn header_status(header: &Option<AiResponseHeader>) -> (i32, i32) {
        let header = header.as_ref().expect("header");
        (header.status, header.error_code)
    }

    #[tokio::test]
    async fn server_maps_model_not_ready_to_provider_unavailable() {
        let service = service();
        let response = service
            .start_segmentation_session(Request::new(StartSegmentationSessionRequest {
                header: Some(AiRequestHeader {
                    request_id: "r1".to_string(),
                    ..Default::default()
                }),
                width: 2,
                height: 2,
                rgb8_pixels: vec![0; 12],
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert_eq!(
            header_status(&response.header),
            (
                AiResponseStatus::AiStatusProviderUnavailable as i32,
                AiErrorCode::AiErrorModelNotReady as i32
            )
        );
        assert_eq!(response.header.as_ref().expect("header").request_id, "r1");
        assert!(response.session_id.is_empty());
    }

    async fn loaded_service() -> SegmentationServiceImpl {
        let service = service();
        let response = service
            .load_segmentation_models(Request::new(LoadSegmentationModelsRequest {
                header: None,
                model_directory: model_directory("server").display().to_string(),
                device: "fake".to_string(),
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert_eq!(
            header_status(&response.header).0,
            AiResponseStatus::AiStatusOk as i32
        );
        service
    }

    #[tokio::test]
    async fn server_maps_invalid_image_to_invalid_argument() {
        let service = loaded_service().await;
        let response = service
            .start_segmentation_session(Request::new(StartSegmentationSessionRequest {
                header: None,
                width: 4,
                height: 4,
                rgb8_pixels: vec![0; 10],
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert_eq!(
            header_status(&response.header),
            (
                AiResponseStatus::AiStatusInvalidArgument as i32,
                AiErrorCode::AiErrorInvalidInput as i32
            )
        );
    }

    #[tokio::test]
    async fn server_rejects_unknown_point_label_and_unknown_session() {
        let service = loaded_service().await;
        let response = service
            .decode_segmentation_prompt(Request::new(DecodeSegmentationPromptRequest {
                session_id: "missing".to_string(),
                points: vec![SegmentationPoint {
                    x: 0.5,
                    y: 0.5,
                    label: 2,
                }],
                ..Default::default()
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert_eq!(
            header_status(&response.header),
            (
                AiResponseStatus::AiStatusInvalidArgument as i32,
                AiErrorCode::AiErrorInvalidInput as i32
            )
        );
        let response = service
            .decode_segmentation_prompt(Request::new(DecodeSegmentationPromptRequest {
                session_id: "missing".to_string(),
                points: vec![SegmentationPoint {
                    x: 0.5,
                    y: 0.5,
                    label: SegmentationPointLabel::Add as i32,
                }],
                ..Default::default()
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert_eq!(
            header_status(&response.header),
            (
                AiResponseStatus::AiStatusInvalidArgument as i32,
                AiErrorCode::AiErrorSessionNotFound as i32
            )
        );
    }

    #[tokio::test]
    async fn server_returns_click_levels_with_logit_size() {
        let service = loaded_service().await;
        let started = service
            .start_segmentation_session(Request::new(StartSegmentationSessionRequest {
                header: None,
                width: 8,
                height: 4,
                rgb8_pixels: vec![50; 8 * 4 * 3],
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert!(!started.session_id.is_empty());
        let response = service
            .decode_segmentation_prompt(Request::new(DecodeSegmentationPromptRequest {
                session_id: started.session_id.clone(),
                points: vec![SegmentationPoint {
                    x: 0.5,
                    y: 0.5,
                    label: SegmentationPointLabel::Add as i32,
                }],
                ..Default::default()
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert_eq!(
            header_status(&response.header).0,
            AiResponseStatus::AiStatusOk as i32
        );
        assert_eq!(response.logit_size, LOGIT_SIZE as u32);
        assert_eq!(response.masks.len(), 3);
        assert!(
            response
                .masks
                .iter()
                .all(|level| level.logits.len() == LOGIT_SIZE * LOGIT_SIZE)
        );
        let ended = service
            .end_segmentation_session(Request::new(EndSegmentationSessionRequest {
                header: None,
                session_id: started.session_id,
            }))
            .await
            .expect("transport ok")
            .into_inner();
        assert!(ended.ended);
    }
}

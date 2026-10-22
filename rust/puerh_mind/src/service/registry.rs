use std::collections::HashMap;
use std::sync::Arc;

use tonic::transport::Server;

use crate::config::AppConfig;
use crate::proto::alcedo::ai::AiCapability;
use crate::proto::alcedo::ai::ai_runtime_service_server::AiRuntimeServiceServer;
use crate::proto::alcedo::ai::image_analysis_service_server::ImageAnalysisServiceServer;
use crate::proto::alcedo::ai::segmentation_service_server::SegmentationServiceServer;
use crate::proto::common::health_service_server::HealthServiceServer;
use crate::proto::semantic::model_manager_service_server::ModelManagerServiceServer;
use crate::proto::semantic::semantic_service_server::SemanticServiceServer;
use crate::server::ai_runtime::AiRuntimeServiceImpl;
use crate::server::health::HealthServiceImpl;
use crate::server::image_analysis::ImageAnalysisServiceImpl;
use crate::server::model_manager::ModelManagerServiceImpl;
use crate::server::segmentation::SegmentationServiceImpl;
use crate::server::semantic::SemanticServiceImpl;
use crate::service::cancellation_registry::CancellationRegistry;
use crate::service::credential_vault::CredentialVault;
use crate::service::embedding::EmbeddingEngine;
use crate::service::image_analysis::ImageAnalysisProvider;
use crate::service::segmentation::engine::SegmentationEngine;

const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("semantic_descriptor");
/// Message limit of the segmentation service: a 4096 x 4096 RGB8 image is 50.3 MB.
const SEGMENTATION_MAX_MESSAGE_BYTES: usize = 64 * 1024 * 1024;

pub fn register_services(
    mut builder: Server,
    config: &AppConfig,
    semantic_engine: Arc<dyn EmbeddingEngine>,
    vault: Arc<CredentialVault>,
    cancel_registry: Arc<CancellationRegistry>,
    capabilities: Vec<AiCapability>,
    image_providers: HashMap<String, Arc<dyn ImageAnalysisProvider>>,
    default_image_provider_id: String,
    segmentation_engine: Arc<SegmentationEngine>,
) -> anyhow::Result<tonic::transport::server::Router> {
    let health_service = HealthServiceImpl;
    let semantic_service = SemanticServiceImpl::new(
        semantic_engine,
        config.semantic.batch_cap,
        std::time::Duration::from_millis(config.semantic.batch_wait_ms),
    );
    let model_manager_service =
        ModelManagerServiceImpl::new(&config.semantic.model_root, &config.semantic.hf_endpoint);
    // Phase 3: the AI runtime service owns the credential vault, the
    // cancellation registry, and the advertised capability descriptors.
    let ai_runtime_service =
        AiRuntimeServiceImpl::new(vault.clone(), cancel_registry.clone(), capabilities);
    // Phase 5b: the image analysis service routes DescribeImage / ScoreImage to
    // a registered provider (the mock in 5b; real drivers in 5c), with the vault
    // and cancellation registry shared with the AI runtime service.
    let image_analysis_service = ImageAnalysisServiceImpl::new(
        image_providers,
        default_image_provider_id,
        vault,
        cancel_registry,
    );

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1alpha()
        .expect("failed to build reflection service");

    Ok(builder
        .add_service(reflection_service)
        .add_service(HealthServiceServer::new(health_service))
        .add_service(ModelManagerServiceServer::new(model_manager_service))
        .add_service(
            SemanticServiceServer::new(semantic_service)
                .max_decoding_message_size(config.max_message_bytes)
                .max_encoding_message_size(config.max_message_bytes),
        )
        .add_service(AiRuntimeServiceServer::new(ai_runtime_service))
        .add_service(ImageAnalysisServiceServer::new(image_analysis_service))
        .add_service(
            SegmentationServiceServer::new(SegmentationServiceImpl::new(segmentation_engine))
                .max_decoding_message_size(SEGMENTATION_MAX_MESSAGE_BYTES)
                .max_encoding_message_size(SEGMENTATION_MAX_MESSAGE_BYTES),
        ))
}

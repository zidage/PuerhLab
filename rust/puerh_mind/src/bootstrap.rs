use std::collections::HashMap;
use std::sync::Arc;

use tonic::transport::Server;
use tracing::info;

use crate::config::AppConfig;
use crate::proto::alcedo::ai::AiCapability;
use crate::service::cancellation_registry::CancellationRegistry;
use crate::service::credential_vault::CredentialVault;
use crate::service::embedding::EmbeddingEngine;
use crate::service::image_analysis::ImageAnalysisProvider;
use crate::service::registry::register_services;
use crate::service::segmentation::engine::SegmentationEngine;

#[allow(clippy::too_many_arguments)]
pub async fn start_server(
    config: AppConfig,
    semantic_engine: Arc<dyn EmbeddingEngine>,
    vault: Arc<CredentialVault>,
    cancel_registry: Arc<CancellationRegistry>,
    capabilities: Vec<AiCapability>,
    image_providers: HashMap<String, Arc<dyn ImageAnalysisProvider>>,
    default_image_provider_id: String,
    segmentation_engine: Arc<SegmentationEngine>,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = config.listen_addr().parse()?;

    info!("starting alcedo_mind on {}", addr);

    let router = register_services(
        Server::builder(),
        &config,
        semantic_engine,
        vault,
        cancel_registry,
        capabilities,
        image_providers,
        default_image_provider_id,
        segmentation_engine,
    )?;

    router.serve(addr).await?;

    Ok(())
}

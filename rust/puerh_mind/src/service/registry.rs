use std::sync::Arc;

use tonic::transport::Server;

use crate::config::AppConfig;
use crate::proto::common::health_service_server::HealthServiceServer;
use crate::server::health::HealthServiceImpl;
use crate::proto::semantic::semantic_service_server::SemanticServiceServer;
use crate::server::semantic::SemanticServiceImpl;
use crate::service::candle_clip::CandleClipEngine;

const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("semantic_descriptor");

pub fn register_services(mut builder: Server, config: &AppConfig) -> anyhow::Result<tonic::transport::server::Router> {
    let health_service = HealthServiceImpl;
    let semantic_engine = Arc::new(CandleClipEngine::new(&config.semantic)?);
    let semantic_service = SemanticServiceImpl::new(semantic_engine);

    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
        .build_v1alpha()
        .expect("failed to build reflection service");

    Ok(builder
        .add_service(reflection_service)
        .add_service(HealthServiceServer::new(health_service))
        .add_service(SemanticServiceServer::new(semantic_service)))
}

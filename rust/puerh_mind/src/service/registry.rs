use tonic::transport::Server;

use crate::proto::common::health_service_server::HealthServiceServer;
use crate::server::health::HealthServiceImpl;

pub fn register_services(
    mut builder: Server,
) -> tonic::transport::server::Router {
    let health_service = HealthServiceImpl;

    builder.add_service(HealthServiceServer::new(health_service))
}

use tonic::{Request, Response, Status};
use tracing::info;

use crate::proto::common::{
    GetVersionRequest, GetVersionResponse, PingRequest, PingResponse,
    health_service_server::HealthService,
};

pub struct HealthServiceImpl;

#[tonic::async_trait]
impl HealthService for HealthServiceImpl {
    async fn ping(&self, _request: Request<PingRequest>) -> Result<Response<PingResponse>, Status> {
        info!("received Ping request");
        let response = PingResponse {
            message: "OK".to_string(),
        };
        Ok(Response::new(response))
    }

    async fn get_version(
        &self,
        _request: Request<GetVersionRequest>,
    ) -> Result<Response<GetVersionResponse>, Status> {
        info!("received GetVersion request");

        let response = GetVersionResponse {
            version: "0.1.0".to_string(),
        };
        Ok(Response::new(response))
    }
}

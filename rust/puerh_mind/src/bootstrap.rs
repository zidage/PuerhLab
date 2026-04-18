use tonic::transport::Server;
use tracing::info;

use crate::config::AppConfig;
use crate::service::registry::register_services;

pub async fn start_server() -> Result<(), Box<dyn std::error::Error>> {
    let config = AppConfig::load();
    let addr = config.listen_addr().parse()?;

    info!("staring alcedo_mind on {}", addr);

    let router = register_services(Server::builder(), &config)?;

    router.serve(addr).await?;

    Ok(())
}

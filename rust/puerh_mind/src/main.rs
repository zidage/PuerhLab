mod bootstrap;
mod config;
mod logging;
mod proto;
mod server;
mod service;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    logging::init_logging();
    bootstrap::start_server().await
}

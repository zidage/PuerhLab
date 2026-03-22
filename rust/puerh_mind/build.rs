fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_prost_build::configure()
        .compile_protos(&["proto/common.proto", "proto/semantic.proto"], &["proto"])?;

    println!("cargo:rerun-if-changed=proto/common.proto");
    println!("cargo:rerun-if-changed=proto/semantic.proto");

    Ok(())
}
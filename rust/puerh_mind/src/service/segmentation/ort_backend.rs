//! ONNX Runtime adapter of the segmentation models: DirectML on Windows, CPU for development.
//!
//! Session settings follow the lab MVP (master plan section 8.17.3): graph optimization level 3,
//! 4 intra-op threads, 1 inter-op thread, CPU memory arena off, memory patterns off, sequential
//! execution. A DirectML session registers DirectML with `error_on_failure`, so a registration
//! failure is a load error and never a CPU session. The CPU entry of the list only applies the
//! CPU memory arena setting (`DisableCpuMemArena`) to the CPU execution provider that ONNX Runtime
//! always keeps for node placement.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ort::{
    ep,
    session::{
        Session,
        builder::{GraphOptimizationLevel, SessionBuilder},
    },
    value::TensorRef,
};

use super::backend::{
    SegmentationBackend, SegmentationBackendFactory, SegmentationModel, SegmentationModelRole,
    UnsupportedDevice,
};
use super::model_set::{ModelSpec, spec};
use super::tensor::TensorF32;

const INTRA_OP_THREADS: usize = 4;
const INTER_OP_THREADS: usize = 1;

/// Device of an ONNX Runtime segmentation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrtSegmentationDevice {
    Cpu,
    /// DirectML on the adapter with this index (Windows only).
    DirectMl(i32),
}

impl OrtSegmentationDevice {
    /// Parses a device value of the load request. `""` is the platform default: DirectML on
    /// adapter 0 on Windows. On other platforms only `cpu` is accepted.
    pub fn parse(value: &str) -> Result<Self, UnsupportedDevice> {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized == "cpu" {
            return Ok(Self::Cpu);
        }
        if !cfg!(windows) {
            return Err(UnsupportedDevice(format!(
                "segmentation device {value:?} is not supported by the ONNX Runtime backend on this platform; use \"cpu\""
            )));
        }
        if normalized.is_empty() || normalized == "directml" {
            return Ok(Self::DirectMl(0));
        }
        if let Some(ordinal) = normalized.strip_prefix("directml:") {
            return match ordinal.parse::<i32>() {
                Ok(ordinal) if ordinal >= 0 => Ok(Self::DirectMl(ordinal)),
                _ => Err(UnsupportedDevice(format!(
                    "segmentation device {value:?} has an invalid DirectML adapter index"
                ))),
            };
        }
        Err(UnsupportedDevice(format!(
            "segmentation device {value:?} is not supported; use \"directml\", \"directml:N\", or \"cpu\""
        )))
    }

    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".to_string(),
            Self::DirectMl(ordinal) => format!("directml:{ordinal}"),
        }
    }
}

/// Creates [`OrtSegmentationBackend`] values for the device of each load request.
pub struct OrtSegmentationBackendFactory;

impl SegmentationBackendFactory for OrtSegmentationBackendFactory {
    fn create(&self, device: &str) -> Result<Box<dyn SegmentationBackend>, UnsupportedDevice> {
        Ok(Box::new(OrtSegmentationBackend {
            device: OrtSegmentationDevice::parse(device)?,
        }))
    }
}

/// Loads the ONNX files of the model set on one device.
pub struct OrtSegmentationBackend {
    device: OrtSegmentationDevice,
}

impl OrtSegmentationBackend {
    fn session_builder(&self) -> Result<SessionBuilder> {
        crate::service::ort_runtime::initialize_ort_environment()?;
        let builder = Session::builder()
            .map_err(|e| anyhow!("failed to create the ONNX Runtime session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("failed to set the graph optimization level: {e}"))?
            .with_intra_threads(INTRA_OP_THREADS)
            .map_err(|e| anyhow!("failed to set the intra-op thread count: {e}"))?
            .with_inter_threads(INTER_OP_THREADS)
            .map_err(|e| anyhow!("failed to set the inter-op thread count: {e}"))?
            .with_memory_pattern(false)
            .map_err(|e| anyhow!("failed to switch off memory patterns: {e}"))?
            .with_parallel_execution(false)
            .map_err(|e| anyhow!("failed to select sequential execution: {e}"))?;
        builder
            .with_execution_providers(execution_providers(self.device)?)
            .map_err(|e| {
                anyhow!(
                    "failed to register the {} execution provider: {e}",
                    self.device.label()
                )
            })
    }
}

fn cpu_arena_off() -> ep::ExecutionProviderDispatch {
    ep::CPU::default().with_arena_allocator(false).build()
}

#[cfg(windows)]
fn execution_providers(
    device: OrtSegmentationDevice,
) -> Result<Vec<ep::ExecutionProviderDispatch>> {
    Ok(match device {
        OrtSegmentationDevice::Cpu => vec![cpu_arena_off().error_on_failure()],
        OrtSegmentationDevice::DirectMl(ordinal) => vec![
            ep::DirectML::default()
                .with_device_id(ordinal)
                .build()
                .error_on_failure(),
            cpu_arena_off(),
        ],
    })
}

#[cfg(not(windows))]
fn execution_providers(
    device: OrtSegmentationDevice,
) -> Result<Vec<ep::ExecutionProviderDispatch>> {
    match device {
        OrtSegmentationDevice::Cpu => Ok(vec![cpu_arena_off().error_on_failure()]),
        OrtSegmentationDevice::DirectMl(_) => bail!("DirectML is only available on Windows"),
    }
}

impl SegmentationBackend for OrtSegmentationBackend {
    fn backend_name(&self) -> &str {
        match self.device {
            OrtSegmentationDevice::Cpu => "onnxruntime-cpu",
            OrtSegmentationDevice::DirectMl(_) => "onnxruntime-directml",
        }
    }

    fn device_label(&self, _role: SegmentationModelRole) -> String {
        self.device.label()
    }

    fn load(
        &self,
        directory: &Path,
        role: SegmentationModelRole,
    ) -> Result<Box<dyn SegmentationModel>> {
        let spec = spec(role);
        let path = model_path(directory, spec);
        if !path.is_file() {
            bail!(
                "segmentation {} model {} does not exist",
                role.as_str(),
                path.display()
            );
        }
        let session = self
            .session_builder()?
            .commit_from_file(&path)
            .map_err(|e| {
                anyhow!(
                    "failed to load {} on {}: {e}",
                    path.display(),
                    self.device.label()
                )
            })?;
        validate_session_names(&session, spec, &path)?;
        Ok(Box::new(OrtSegmentationModel { session, spec }))
    }
}

/// Path of the ONNX file of @p spec in @p directory.
pub fn model_path(directory: &Path, spec: &ModelSpec) -> PathBuf {
    directory.join(spec.onnx_file)
}

fn validate_session_names(session: &Session, spec: &ModelSpec, path: &Path) -> Result<()> {
    let inputs: Vec<&str> = session.inputs().iter().map(|input| input.name()).collect();
    for name in spec.inputs {
        if !inputs.contains(name) {
            bail!(
                "{} has no input {name:?}; inputs: {inputs:?}",
                path.display()
            );
        }
    }
    let outputs: Vec<&str> = session
        .outputs()
        .iter()
        .map(|output| output.name())
        .collect();
    for name in spec.onnx_outputs {
        if !outputs.contains(name) {
            bail!(
                "{} has no output {name:?}; outputs: {outputs:?}",
                path.display()
            );
        }
    }
    Ok(())
}

struct OrtSegmentationModel {
    session: Session,
    spec: &'static ModelSpec,
}

impl SegmentationModel for OrtSegmentationModel {
    fn run(&mut self, inputs: &[&TensorF32]) -> Result<Vec<TensorF32>> {
        if inputs.len() != self.spec.inputs.len() {
            bail!(
                "segmentation {} expects {} inputs, got {}",
                self.spec.role.as_str(),
                self.spec.inputs.len(),
                inputs.len()
            );
        }
        let mut values = Vec::with_capacity(inputs.len());
        for (name, tensor) in self.spec.inputs.iter().zip(inputs) {
            let view = TensorRef::from_array_view((tensor.shape(), tensor.data()))
                .map_err(|e| anyhow!("failed to bind input {name}: {e}"))?;
            values.push((*name, view));
        }
        let outputs = self.session.run(values).map_err(|e| {
            anyhow!(
                "failed to run the segmentation {}: {e}",
                self.spec.role.as_str()
            )
        })?;
        self.spec
            .onnx_outputs
            .iter()
            .map(|name| {
                let value = outputs
                    .get(*name)
                    .with_context(|| format!("missing output {name}"))?;
                let (shape, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| anyhow!("output {name} is not a float32 tensor: {e}"))?;
                let shape = shape
                    .iter()
                    .map(|&dim| {
                        usize::try_from(dim)
                            .map_err(|_| anyhow!("output {name} has a negative dimension"))
                    })
                    .collect::<Result<Vec<_>>>()?;
                TensorF32::new(shape, data.to_vec())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ort_device_values_parse_to_one_provider() {
        assert_eq!(
            OrtSegmentationDevice::parse("cpu"),
            Ok(OrtSegmentationDevice::Cpu)
        );
        assert_eq!(
            OrtSegmentationDevice::parse(" CPU "),
            Ok(OrtSegmentationDevice::Cpu)
        );
        if cfg!(windows) {
            assert_eq!(
                OrtSegmentationDevice::parse(""),
                Ok(OrtSegmentationDevice::DirectMl(0))
            );
            assert_eq!(
                OrtSegmentationDevice::parse("directml"),
                Ok(OrtSegmentationDevice::DirectMl(0))
            );
            assert_eq!(
                OrtSegmentationDevice::parse("directml:2"),
                Ok(OrtSegmentationDevice::DirectMl(2))
            );
        } else {
            assert!(OrtSegmentationDevice::parse("").is_err());
            assert!(OrtSegmentationDevice::parse("directml").is_err());
        }
        for rejected in ["coreml", "cuda", "directml:-1", "directml:", "auto"] {
            assert!(
                OrtSegmentationDevice::parse(rejected).is_err(),
                "{rejected} must be rejected"
            );
        }
    }

    #[test]
    fn ort_backend_reports_missing_model_file_path() {
        let directory = std::env::temp_dir().join(format!(
            "alcedo-segmentation-empty-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&directory).expect("create directory");
        let backend = OrtSegmentationBackend {
            device: OrtSegmentationDevice::Cpu,
        };
        let err = backend
            .load(&directory, SegmentationModelRole::Encoder)
            .err()
            .expect("missing file");
        assert!(err.to_string().contains("point_encoder.onnx"), "{err}");
        std::fs::remove_dir_all(&directory).expect("remove directory");
    }
}

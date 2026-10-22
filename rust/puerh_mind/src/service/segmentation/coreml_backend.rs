//! Native Core ML adapter of the segmentation models (macOS).
//!
//! The models are compiled `.mlmodelc` directories (master plan section 8.17.1), loaded with the
//! `coreml` crate as the SigLIP2 encoders are (`coreml_clip.rs`). The ONNX Runtime Core ML
//! execution provider is not used (lab decision U15). Compute units per model (U15): the encoder
//! runs with `ComputeUnits::All`, the decoder with `ComputeUnits::CpuOnly` (about 30 ms per call
//! and no new preparation when the point count changes). The `cpu` device runs every model with
//! `ComputeUnits::CpuOnly`; it exists for development and tests only.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use coreml::{ComputeUnits, DataType, FeatureProvider, Model, ModelConfiguration, MultiArray};

use super::backend::{
    SegmentationBackend, SegmentationBackendFactory, SegmentationModel, SegmentationModelRole,
    UnsupportedDevice,
};
use super::model_set::{ModelSpec, spec};
use super::tensor::TensorF32;

/// Device of the Core ML segmentation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMlSegmentationDevice {
    /// Compute units per model of lab decision U15.
    PerModel,
    /// `ComputeUnits::CpuOnly` for every model.
    CpuOnly,
}

impl CoreMlSegmentationDevice {
    /// Parses a device value of the load request: `""` or `coreml` for the U15 compute units,
    /// `cpu` for CPU only.
    pub fn parse(value: &str) -> Result<Self, UnsupportedDevice> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "coreml" => Ok(Self::PerModel),
            "cpu" => Ok(Self::CpuOnly),
            _ => Err(UnsupportedDevice(format!(
                "segmentation device {value:?} is not supported by the Core ML backend; use \"coreml\" or \"cpu\""
            ))),
        }
    }

    fn compute_units(self, role: SegmentationModelRole) -> ComputeUnits {
        match (self, role) {
            (Self::PerModel, SegmentationModelRole::Encoder) => ComputeUnits::All,
            (Self::PerModel, SegmentationModelRole::Decoder) => ComputeUnits::CpuOnly,
            (Self::CpuOnly, _) => ComputeUnits::CpuOnly,
        }
    }

    fn compute_units_label(self, role: SegmentationModelRole) -> &'static str {
        match self.compute_units(role) {
            ComputeUnits::All => "coreml-native:all",
            ComputeUnits::CpuOnly => "coreml-native:cpuonly",
            ComputeUnits::CpuAndGpu => "coreml-native:cpuandgpu",
            ComputeUnits::CpuAndNeuralEngine => "coreml-native:cpuandneuralengine",
        }
    }
}

/// Creates [`CoreMlSegmentationBackend`] values for the device of each load request.
pub struct CoreMlSegmentationBackendFactory;

impl SegmentationBackendFactory for CoreMlSegmentationBackendFactory {
    fn create(&self, device: &str) -> Result<Box<dyn SegmentationBackend>, UnsupportedDevice> {
        Ok(Box::new(CoreMlSegmentationBackend {
            device: CoreMlSegmentationDevice::parse(device)?,
        }))
    }
}

/// Loads the compiled Core ML models of the model set.
pub struct CoreMlSegmentationBackend {
    device: CoreMlSegmentationDevice,
}

/// Path of the compiled model directory of @p spec in @p directory.
pub fn model_path(directory: &Path, spec: &ModelSpec) -> PathBuf {
    directory.join(spec.coreml_directory)
}

impl SegmentationBackend for CoreMlSegmentationBackend {
    fn backend_name(&self) -> &str {
        "coreml-native"
    }

    fn device_label(&self, role: SegmentationModelRole) -> String {
        self.device.compute_units_label(role).to_string()
    }

    fn load(
        &self,
        directory: &Path,
        role: SegmentationModelRole,
    ) -> Result<Box<dyn SegmentationModel>> {
        let spec = spec(role);
        let path = model_path(directory, spec);
        if !path.is_dir() {
            bail!(
                "segmentation {} model {} does not exist",
                role.as_str(),
                path.display()
            );
        }
        let configuration =
            ModelConfiguration::new().with_compute_units(self.device.compute_units(role));
        let model = Model::load_from_url(&path, &configuration).map_err(|e| {
            anyhow!(
                "failed to load {} with {}: {e}",
                path.display(),
                self.device.compute_units_label(role)
            )
        })?;
        Ok(Box::new(CoreMlSegmentationModel { model, spec }))
    }
}

struct CoreMlSegmentationModel {
    model: Model,
    spec: &'static ModelSpec,
}

// The Rust wrapper of MLModel does not declare Send. The engine calls each model under its own
// mutex, so one thread at a time uses the handle.
unsafe impl Send for CoreMlSegmentationModel {}

impl SegmentationModel for CoreMlSegmentationModel {
    fn run(&mut self, inputs: &[&TensorF32]) -> Result<Vec<TensorF32>> {
        if inputs.len() != self.spec.inputs.len() {
            bail!(
                "segmentation {} expects {} inputs, got {}",
                self.spec.role.as_str(),
                self.spec.inputs.len(),
                inputs.len()
            );
        }
        let mut provider = FeatureProvider::new();
        for (name, tensor) in self.spec.inputs.iter().zip(inputs) {
            let mut array = MultiArray::new_f32(tensor.shape())
                .map_err(|e| anyhow!("failed to allocate Core ML input {name}: {e}"))?;
            array
                .copy_from_f32_slice(tensor.data())
                .map_err(|e| anyhow!("failed to fill Core ML input {name}: {e}"))?;
            provider
                .try_insert_multi_array(name, array)
                .map_err(|e| anyhow!("failed to add Core ML input {name}: {e}"))?;
        }
        let outputs = self.model.predict(&provider).map_err(|e| {
            anyhow!(
                "failed to run the segmentation {}: {e}",
                self.spec.role.as_str()
            )
        })?;
        self.spec
            .coreml_outputs
            .iter()
            .map(|name| {
                let array = outputs.get_multi_array(name).ok_or_else(|| {
                    anyhow!(
                        "missing Core ML output {name:?}; available outputs: {:?}",
                        outputs.keys()
                    )
                })?;
                read_multi_array(name, &array)
            })
            .collect()
    }
}

/// Copies a Float32 or Float16 multi-array into a row-major tensor. Follows the strides that
/// Core ML reports, so a padded output array reads correctly.
fn read_multi_array(name: &str, array: &MultiArray) -> Result<TensorF32> {
    let shape = array.shape();
    let strides = array.strides();
    let values = match array.data_type() {
        DataType::Float32 => {
            let storage = array
                .as_f32_slice()
                .ok_or_else(|| anyhow!("Core ML output {name} has no Float32 storage"))?;
            gather_row_major(&shape, &strides, |offset| storage[offset])
        }
        DataType::Float16 => {
            let storage = array
                .as_f16_slice()
                .ok_or_else(|| anyhow!("Core ML output {name} has no Float16 storage"))?;
            gather_row_major(&shape, &strides, |offset| storage[offset].to_f32())
        }
        other => {
            bail!("Core ML output {name} has data type {other:?}, expected Float32 or Float16")
        }
    };
    TensorF32::new(shape, values)
}

/// Reads the elements of an array with @p shape and @p strides in row-major order through
/// @p read (storage offset to value).
fn gather_row_major(shape: &[usize], strides: &[usize], read: impl Fn(usize) -> f32) -> Vec<f32> {
    let len: usize = shape.iter().product();
    let mut values = Vec::with_capacity(len);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..len {
        let offset = index
            .iter()
            .zip(strides)
            .map(|(position, stride)| position * stride)
            .sum();
        values.push(read(offset));
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coreml_device_values_select_u15_compute_units() {
        let device = CoreMlSegmentationDevice::parse("").expect("default device");
        assert_eq!(
            device.compute_units_label(SegmentationModelRole::Encoder),
            "coreml-native:all"
        );
        assert_eq!(
            device.compute_units_label(SegmentationModelRole::Decoder),
            "coreml-native:cpuonly"
        );
        let cpu = CoreMlSegmentationDevice::parse("cpu").expect("cpu device");
        assert_eq!(
            cpu.compute_units_label(SegmentationModelRole::Encoder),
            "coreml-native:cpuonly"
        );
        assert!(CoreMlSegmentationDevice::parse("directml").is_err());
    }

    #[test]
    fn gather_row_major_follows_padded_strides() {
        // Shape [2, 3] stored with a row stride of 4 (one padding value per row).
        let storage = [0.0, 1.0, 2.0, -1.0, 3.0, 4.0, 5.0, -1.0];
        let values = gather_row_major(&[2, 3], &[4, 1], |offset| storage[offset]);
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }
}

#include "ui/puerhlab_main/editor_dialog/controllers/pipeline_controller.hpp"

#include <json.hpp>

#include "edit/pipeline/default_pipeline_params.hpp"
#include "ui/edit_viewer/edit_viewer.hpp"

namespace puerhlab::ui::controllers {
namespace {

auto MergeMissingDefaults(nlohmann::json& target, const nlohmann::json& defaults) -> bool {
  if (!defaults.is_object()) {
    return false;
  }
  bool changed = false;
  if (!target.is_object()) {
    target  = nlohmann::json::object();
    changed = true;
  }
  for (auto it = defaults.begin(); it != defaults.end(); ++it) {
    if (!target.contains(it.key())) {
      target[it.key()] = it.value();
      changed          = true;
      continue;
    }
    if (it.value().is_object()) {
      if (!target[it.key()].is_object()) {
        target[it.key()] = it.value();
        changed          = true;
        continue;
      }
      for (auto nested_it = it.value().begin(); nested_it != it.value().end(); ++nested_it) {
        if (!target[it.key()].contains(nested_it.key())) {
          target[it.key()][nested_it.key()] = nested_it.value();
          changed                           = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace

void EnsureLoadingOperatorDefaults(const std::shared_ptr<CPUPipelineExecutor>& exec) {
  if (!exec) {
    return;
  }

  auto& global_params = exec->GetGlobalParams();
  auto& loading       = exec->GetStage(PipelineStageName::Image_Loading);

  const auto ensure_defaults =
      [&](OperatorType type, const nlohmann::json& defaults, bool pass_global_params) {
        const auto op = loading.GetOperator(type);
        if (!op.has_value() || !op.value() || !op.value()->op_) {
          if (pass_global_params) {
            loading.SetOperator(type, defaults, global_params);
          } else {
            loading.SetOperator(type, defaults);
          }
          return;
        }

        nlohmann::json params = nlohmann::json::object();
        const auto     exported = op.value()->ExportOperatorParams();
        if (exported.contains("params") && exported["params"].is_object()) {
          params = exported["params"];
        }
        if (!MergeMissingDefaults(params, defaults)) {
          return;
        }
        if (pass_global_params) {
          loading.SetOperator(type, params, global_params);
        } else {
          loading.SetOperator(type, params);
        }
      };

  ensure_defaults(OperatorType::RAW_DECODE,
                  pipeline_defaults::MakeDefaultRawDecodeParams(), false);
  ensure_defaults(OperatorType::LENS_CALIBRATION,
                  pipeline_defaults::MakeDefaultLensCalibParams(), true);
}

void AttachExecutionStages(const std::shared_ptr<CPUPipelineExecutor>& exec,
                           QtEditViewer* viewer) {
  if (!exec) {
    return;
  }
  exec->SetExecutionStages(viewer);
}

}  // namespace puerhlab::ui::controllers

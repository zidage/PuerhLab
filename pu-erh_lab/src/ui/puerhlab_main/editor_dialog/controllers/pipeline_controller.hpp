#pragma once

#include <memory>

#include "edit/pipeline/pipeline_cpu.hpp"

namespace puerhlab {
class QtEditViewer;
}

namespace puerhlab::ui::controllers {

void EnsureLoadingOperatorDefaults(const std::shared_ptr<CPUPipelineExecutor>& exec);
void AttachExecutionStages(const std::shared_ptr<CPUPipelineExecutor>& exec,
                           QtEditViewer* viewer);

}  // namespace puerhlab::ui::controllers

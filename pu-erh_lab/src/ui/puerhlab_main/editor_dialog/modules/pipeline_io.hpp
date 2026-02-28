// pipeline_io.hpp — Maps AdjustmentField / AdjustmentState to pipeline operators.
//
// Free functions extracted from EditorDialog to keep dialog.cpp smaller.
// All functions are stateless: they accept the pipeline executor and state
// structs by reference and return results without touching any UI widgets.
#pragma once

#include <json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <QPointF>

#include "edit/operators/op_base.hpp"
#include "ui/puerhlab_main/editor_dialog/state.hpp"

namespace puerhlab {
class PipelineStage;
class CPUPipelineExecutor;
}  // namespace puerhlab

namespace puerhlab::ui::pipeline_io {

// ---------------------------------------------------------------------------
// Tonal-slider global-param scale factors.
// ---------------------------------------------------------------------------
constexpr float kBlackSliderFromGlobalScale      = 1000.0f;
constexpr float kWhiteSliderFromGlobalScale      = 300.0f;
constexpr float kShadowsSliderFromGlobalScale    = 80.0f;
constexpr float kHighlightsSliderFromGlobalScale = 50.0f;

// ---------------------------------------------------------------------------
// Low-level pipeline-stage readers.
// ---------------------------------------------------------------------------
auto IsOperatorEnabled(const PipelineStage& stage, OperatorType type) -> std::optional<bool>;
auto ReadFloat(const PipelineStage& stage, OperatorType type,
               const char* key) -> std::optional<float>;
auto ReadNestedFloat(const PipelineStage& stage, OperatorType type, const char* key1,
                     const char* key2) -> std::optional<float>;
auto ReadNestedObject(const PipelineStage& stage, OperatorType type,
                      const char* key) -> std::optional<nlohmann::json>;
auto ReadString(const PipelineStage& stage, OperatorType type,
                const char* key) -> std::optional<std::string>;
auto ReadCurvePoints(const PipelineStage& stage,
                     OperatorType type) -> std::optional<std::vector<QPointF>>;

// Read the "params" sub-object from an operator in a named stage.
auto ReadCurrentOperatorParams(CPUPipelineExecutor& exec, PipelineStageName stage_name,
                               OperatorType op_type) -> std::optional<nlohmann::json>;

// ---------------------------------------------------------------------------
// AdjustmentField <-> pipeline mapping.
// ---------------------------------------------------------------------------

// Map a field enum to its (stage, operator) pair.
auto FieldSpec(AdjustmentField field) -> std::pair<PipelineStageName, OperatorType>;

// Build an operator-param JSON blob for a given field from the adjustment state.
// @p exec is used only for RawDecode/LensCalib to merge with existing pipeline
//         params; may be nullptr for other fields.
auto ParamsForField(AdjustmentField field, const AdjustmentState& s,
                    CPUPipelineExecutor* exec) -> nlohmann::json;

// Return true if the field value in @p current differs from @p committed.
auto FieldChanged(AdjustmentField field, const AdjustmentState& current,
                  const AdjustmentState& committed) -> bool;

// ---------------------------------------------------------------------------
// Bulk state I/O.
// ---------------------------------------------------------------------------

// Populate an AdjustmentState from the current pipeline executor state.
// Returns {loaded_state, has_loaded_any}.
auto LoadStateFromPipeline(CPUPipelineExecutor& exec,
                           const AdjustmentState& base_state)
    -> std::pair<AdjustmentState, bool>;

}  // namespace puerhlab::ui::pipeline_io

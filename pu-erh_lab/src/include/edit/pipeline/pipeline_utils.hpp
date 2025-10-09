#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>

#include "edit/operators/op_base.hpp"
#include "edit/operators/operator_factory.hpp"
#include "image/image_buffer.hpp"
#include "tile_scheduler.hpp"

namespace puerhlab {

struct OperatorEntry {
  bool                           _enable = true;
  std::shared_ptr<IOperatorBase> _op;

  bool                           operator<(const OperatorEntry& other) const {
    return _op->GetPriorityLevel() < other._op->GetPriorityLevel();
  }
};

enum class PipelineStageType {
  NonStreamable,
  Streamable,
};

class IPipelineStage {
 protected:
  std::unique_ptr<std::map<OperatorType, OperatorEntry>> _operators;

  IPipelineStage*                                        _prev_stage = nullptr;
  IPipelineStage*                                        _next_stage = nullptr;

  std::shared_ptr<ImageBuffer>                           _input_img;
  std::shared_ptr<ImageBuffer>                           _output_cache;

  bool                                                   _input_cache_valid  = false;
  bool                                                   _output_cache_valid = false;
  bool                                                   _input_set          = false;

  bool                                                   _is_streamable      = false;

  bool                                                   _enable_cache       = true;

 public:
  PipelineStageName _stage;
  PipelineStageType _type;
  explicit IPipelineStage(PipelineStageName stage, bool enable_cache)
      : _stage(stage), _enable_cache(enable_cache) {}
  virtual void SetInputImage(std::shared_ptr<ImageBuffer>)                      = 0;

  virtual void SetOperator(OperatorType, nlohmann::json& param)                 = 0;
  virtual auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*> = 0;
  virtual void EnableOperator(OperatorType, bool enable)                        = 0;

  virtual void SetNeighbors_(IPipelineStage* prev, IPipelineStage* next)        = 0;  // placeholder
  virtual auto GetStageNameString() const -> std::string                        = 0;

  virtual void SetInputCacheValid(bool valid)                                   = 0;
  virtual void SetOutputCacheValid(bool valid)                                  = 0;
  auto         CacheValid() -> bool { return _input_cache_valid && _output_cache_valid; };

  virtual auto HasInput() -> bool                           = 0;

  virtual auto ApplyStage() -> std::shared_ptr<ImageBuffer> = 0;
};

class StreamablePipelineStage : public IPipelineStage {
 private:
  bool                           _is_streamable = true;
  std::unique_ptr<TileScheduler> _tile_scheduler;

  /**
   * @brief streamable operators in this stage, sorted by priority level and stage order
   *
   */
  std::list<OperatorEntry>       _streamable_ops;

 public:
  PipelineStageType _type                                       = PipelineStageType::Streamable;
  StreamablePipelineStage()                                     = delete;
  StreamablePipelineStage(const StreamablePipelineStage& other) = delete;

  StreamablePipelineStage(PipelineStageName stage, bool enable_cache);

  auto MergeStreamable(StreamablePipelineStage& other) -> StreamablePipelineStage;
  auto GetStreamableOps() -> std::list<OperatorEntry>& { return _streamable_ops; }
  void AddStreamableOp(OperatorEntry& op);

  void SetInputImage(std::shared_ptr<ImageBuffer>) override;

  void SetOperator(OperatorType, nlohmann::json& param) override;
  auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*> override;
  void EnableOperator(OperatorType, bool enable) override;

  void SetNeighbors_(IPipelineStage* prev, IPipelineStage* next) override;
  void SetInputCacheValid(bool valid) override;
  void SetOutputCacheValid(bool valid) override;

  auto GetStageNameString() const -> std::string override;

  auto HasInput() -> bool override;

  auto ApplyStage() -> std::shared_ptr<ImageBuffer> override;
};

class PipelineStage : public IPipelineStage {
 private:
  PipelineStage* prev_stage = nullptr;
  PipelineStage* next_stage = nullptr;

  bool           _on_gpu    = false;

 public:
  PipelineStageType _type = PipelineStageType::NonStreamable;
  PipelineStage()         = delete;
  PipelineStage(PipelineStageName stage, bool enable_cache);
  void SetOperator(OperatorType, nlohmann::json& param) override;
  auto GetOperator(OperatorType) const -> std::optional<OperatorEntry*> override;
  void EnableOperator(OperatorType, bool enable) override;
  void SetInputImage(std::shared_ptr<ImageBuffer>) override;

  void SetNeighbors(PipelineStage* prev, PipelineStage* next);
  void SetNeighbors_(IPipelineStage* prev, IPipelineStage* next) override {
    // TODO: dynamic_cast check, migrate to SetNeighbors(IPipelineStage*, IPipelineStage*)
    SetNeighbors(static_cast<PipelineStage*>(prev), static_cast<PipelineStage*>(next));
  }

  void SetInputCacheValid(bool valid) override;
  void SetOutputCacheValid(bool valid) override;

  auto GetStageNameString() const -> std::string override;

  auto HasInput() -> bool override;

  auto ApplyStage() -> std::shared_ptr<ImageBuffer> override;
};
};  // namespace puerhlab
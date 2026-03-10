//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#ifdef HAVE_METAL

#include <condition_variable>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <puerhlab/metal/Metal.hpp>

#include "metal/metal_context.hpp"

namespace puerhlab {
namespace metal {

class ComputePipelineCache {
 private:
  struct LibrarySlot {
    NS::SharedPtr<MTL::Library> library;
    bool                        is_loading = false;
    std::exception_ptr          error;
    std::condition_variable     cv;
  };

  struct PipelineSlot {
    NS::SharedPtr<MTL::ComputePipelineState> pipeline;
    bool                                     is_creating = false;
    std::exception_ptr                       error;
    std::condition_variable                  cv;
  };

  std::mutex                                                    mutex_;
  std::unordered_map<std::string, std::shared_ptr<LibrarySlot>> libraries_;
  std::unordered_map<std::string, std::shared_ptr<PipelineSlot>> pipelines_;

  ComputePipelineCache() = default;

  auto GetLibrarySlot(const std::string& metallib_path) -> std::shared_ptr<LibrarySlot> {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = libraries_[metallib_path];
    if (!slot) {
      slot = std::make_shared<LibrarySlot>();
    }
    return slot;
  }

  auto GetPipelineSlot(const std::string& cache_key) -> std::shared_ptr<PipelineSlot> {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& slot = pipelines_[cache_key];
    if (!slot) {
      slot = std::make_shared<PipelineSlot>();
    }
    return slot;
  }

  auto LoadLibrary(const std::string& metallib_path, const char* debug_label)
      -> NS::SharedPtr<MTL::Library> {
    auto slot = GetLibrarySlot(metallib_path);

    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      if (slot->library) {
        return slot->library;
      }
      if (slot->error) {
        std::rethrow_exception(slot->error);
      }
      if (!slot->is_loading) {
        slot->is_loading = true;
        break;
      }
      slot->cv.wait(lock);
    }
    lock.unlock();

    try {
      auto* device = MetalContext::Instance().Device();
      if (device == nullptr) {
        throw std::runtime_error(std::string(debug_label) + ": Metal device is unavailable.");
      }

      NS::Error* error         = nullptr;
      auto       library_path  = NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
      auto       loaded_library = NS::TransferPtr(device->newLibrary(library_path, &error));
      if (!loaded_library) {
        std::string error_message = std::string(debug_label) + ": failed to load metallib.";
        if (error != nullptr) {
          error_message += " ";
          error_message += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(error_message);
      }

      lock.lock();
      slot->library    = loaded_library;
      slot->is_loading = false;
      slot->cv.notify_all();
      return slot->library;
    } catch (...) {
      lock.lock();
      slot->error      = std::current_exception();
      slot->is_loading = false;
      slot->cv.notify_all();
      throw;
    }
  }

 public:
  ComputePipelineCache(const ComputePipelineCache&)                    = delete;
  auto operator=(const ComputePipelineCache&) -> ComputePipelineCache& = delete;
  ComputePipelineCache(ComputePipelineCache&&)                         = delete;
  auto operator=(ComputePipelineCache&&) -> ComputePipelineCache&      = delete;

  static auto Instance() -> ComputePipelineCache& {
    static ComputePipelineCache cache;
    return cache;
  }

  // Metal pipeline states are immutable, so the same compiled object can be reused safely across
  // concurrent command buffers.
  auto GetPipelineState(const char* metallib_path, const char* function_name,
                        const char* debug_label) -> NS::SharedPtr<MTL::ComputePipelineState> {
    if (metallib_path == nullptr || metallib_path[0] == '\0') {
      throw std::runtime_error(std::string(debug_label) + ": metallib path is not configured.");
    }
    if (function_name == nullptr || function_name[0] == '\0') {
      throw std::runtime_error(std::string(debug_label) + ": compute function name is empty.");
    }

    const std::string library_key = metallib_path;
    const std::string pipeline_key =
        library_key + '\n' + std::string(function_name);
    auto pipeline_slot = GetPipelineSlot(pipeline_key);

    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
      if (pipeline_slot->pipeline) {
        return pipeline_slot->pipeline;
      }
      if (pipeline_slot->error) {
        std::rethrow_exception(pipeline_slot->error);
      }
      if (!pipeline_slot->is_creating) {
        pipeline_slot->is_creating = true;
        break;
      }
      pipeline_slot->cv.wait(lock);
    }
    lock.unlock();

    try {
      auto library = LoadLibrary(library_key, debug_label);

      auto function_name_ns = NS::String::string(function_name, NS::UTF8StringEncoding);
      auto function         = NS::TransferPtr(library->newFunction(function_name_ns));
      if (!function) {
        throw std::runtime_error(std::string(debug_label) +
                                 ": failed to load compute function from metallib.");
      }

      auto* device = MetalContext::Instance().Device();
      if (device == nullptr) {
        throw std::runtime_error(std::string(debug_label) + ": Metal device is unavailable.");
      }

      NS::Error* error          = nullptr;
      auto       created_pipeline =
          NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
      if (!created_pipeline) {
        std::string error_message =
            std::string(debug_label) + ": failed to create compute pipeline.";
        if (error != nullptr) {
          error_message += " ";
          error_message += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error(error_message);
      }

      lock.lock();
      pipeline_slot->pipeline     = created_pipeline;
      pipeline_slot->is_creating  = false;
      pipeline_slot->cv.notify_all();
      return pipeline_slot->pipeline;
    } catch (...) {
      lock.lock();
      pipeline_slot->error        = std::current_exception();
      pipeline_slot->is_creating  = false;
      pipeline_slot->cv.notify_all();
      throw;
    }
  }
};

}  // namespace metal
}  // namespace puerhlab

#endif

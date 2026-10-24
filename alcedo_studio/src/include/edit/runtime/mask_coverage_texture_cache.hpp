//  Copyright 2026 Yurun Zi
//  SPDX-License-Identifier: GPL-3.0-only
//  Additional permission under GPLv3 section 7 applies; see the LICENSE file.

#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "edit/geometry/texture_sampling_plan.hpp"
#include "edit/mask/ai_mask_coverage_texture.hpp"
#include "edit/mask/mask_asset.hpp"
#include "edit/mask/mask_model.hpp"
#include "edit/runtime/mask_asset_reader.hpp"
#include "edit/runtime/texture_format.hpp"

namespace alcedo {

/// Largest number of samples per axis that the raster Mask pass averages for one render pixel.
inline constexpr std::uint32_t kRasterMaskMaximumSamplesPerAxis = 8;

/**
 * @brief Samples per axis of the raster Mask pass (SM6 R7): the texel footprint of one render
 *        pixel, rounded up, in [1, 8].
 *
 * `2^mip_level` is the largest number of texels that one render pixel spans along a render axis.
 * The 1e-3 allowance keeps a footprint of exactly N texels at N samples despite float rounding.
 */
[[nodiscard]] inline auto      RasterMaskSamplesPerAxis(const TextureSamplingPlan& plan)
    -> std::uint32_t {
  const float footprint = std::exp2(plan.mip_level);
  const auto  samples   = static_cast<std::int64_t>(std::ceil(footprint - 1.0e-3f));
  return static_cast<std::uint32_t>(
      std::clamp<std::int64_t>(samples, 1, kRasterMaskMaximumSamplesPerAxis));
}

/// Device bytes that the coverage textures of one render workspace may keep (SM6 R6).
inline constexpr std::size_t kMaskCoverageTextureByteBudget = 256ull << 20;

/**
 * @brief Device R8 coverage textures of AI Masks, one per asset, analysis size, and edge.
 *
 * A miss reads the asset through the workspace reader, makes the coverage texture on the host
 * (@ref MakeAiMaskCoverageTexture), and uploads it once. A frame then samples the texture without
 * a database read or a Feather and Expand pass. The key is the content, so no entry can become
 * stale. A new entry evicts the least recently used entries that neither the current frame nor an
 * in-flight submission uses, until the byte budget holds the new entry; when every entry is in
 * use, the cache keeps them all.
 *
 * Owner: BasicRenderWorkspace. Thread: the render thread of the workspace. Released by
 * BasicRenderWorkspace::ReleaseSessionResources (an image change in the editor, the end of each
 * batch task).
 *
 * @tparam Backend Resource factory with CreateTexture2D, UploadTexture2D, and IsResourceBusy.
 */
template <class Backend>
class MaskCoverageTextureCache {
 public:
  using Texture2D      = typename Backend::Texture2D;
  using CommandContext = typename Backend::CommandContext;

  struct Entry {
    std::string         key;
    Texture2D           texture;
    MaskAssetDescriptor descriptor{};
    std::size_t         bytes                = 0;
    std::uint64_t       last_used_submission = 0;
    std::uint64_t       lru_tick             = 0;
  };

  /**
   * @brief Texture of @p source for the current submission of @p command_context.
   *
   * @throws std::runtime_error from the reader, the coverage texture function, or the upload.
   *         The cache then keeps no entry for the source.
   */
  auto Acquire(const AiSegmentationMaskSource& source, const MaskAssetReader& reader,
               Backend& backend, CommandContext& command_context) -> const Entry& {
    const auto key        = MakeKey(source);
    const auto submission = command_context.SubmissionId();
    ++lru_clock_;
    for (auto& entry : entries_) {
      if (entry->key == key) {
        entry->last_used_submission = submission;
        entry->lru_tick             = lru_clock_;
        return *entry;
      }
    }

    const auto asset    = reader.Read(source.asset_key);
    auto       coverage = MakeAiMaskCoverageTexture(asset, source.analysis_extent, source.edge);
    EvictFor(coverage.pixels.size(), backend, submission);

    auto entry        = std::make_unique<Entry>();
    entry->key        = key;
    entry->descriptor = coverage.descriptor;
    entry->bytes      = coverage.pixels.size();
    entry->texture    = backend.CreateTexture2D(coverage.descriptor.extent.width,
                                                coverage.descriptor.extent.height, TextureFormat::R8);
    backend.UploadTexture2D(
        entry->texture,
        std::span<const std::byte>(reinterpret_cast<const std::byte*>(coverage.pixels.data()),
                                   coverage.pixels.size()),
        command_context);
    entry->last_used_submission = submission;
    entry->lru_tick             = lru_clock_;
    used_bytes_ += entry->bytes;
    ++upload_count_;
    entries_.push_back(std::move(entry));
    return *entries_.back();
  }

  /// Drop every texture. @pre The GPU no longer uses them (the caller waited for idle).
  void Clear() {
    entries_.clear();
    used_bytes_ = 0;
  }

  [[nodiscard]] auto EntryCount() const -> std::size_t { return entries_.size(); }
  [[nodiscard]] auto UsedBytes() const -> std::size_t { return used_bytes_; }
  /// Number of uploaded textures since construction; a hit does not count.
  [[nodiscard]] auto UploadCount() const -> std::uint64_t { return upload_count_; }

 private:
  /// Asset key, analysis size, and the exact bits of the edge values.
  static auto MakeKey(const AiSegmentationMaskSource& source) -> std::string {
    return std::string{source.asset_key.Value()} + "/" +
           std::to_string(source.analysis_extent.width) + "x" +
           std::to_string(source.analysis_extent.height) + "/" +
           std::to_string(std::bit_cast<std::uint32_t>(source.edge.feather_px)) + "/" +
           std::to_string(source.edge.expand_px);
  }

  void EvictFor(std::size_t new_bytes, const Backend& backend, std::uint64_t submission) {
    while (!entries_.empty() && used_bytes_ + new_bytes > kMaskCoverageTextureByteBudget) {
      auto victim = entries_.end();
      for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        const auto& entry = **it;
        if (entry.last_used_submission == submission ||
            backend.IsResourceBusy(entry.last_used_submission)) {
          continue;
        }
        if (victim == entries_.end() || entry.lru_tick < (*victim)->lru_tick) {
          victim = it;
        }
      }
      if (victim == entries_.end()) {
        return;
      }
      used_bytes_ -= (*victim)->bytes;
      entries_.erase(victim);
    }
  }

  std::vector<std::unique_ptr<Entry>> entries_;
  std::size_t                         used_bytes_   = 0;
  std::uint64_t                       lru_clock_    = 0;
  std::uint64_t                       upload_count_ = 0;
};

}  // namespace alcedo

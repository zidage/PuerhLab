# Roadmap

## Upcoming Features and Improvements

### December 2025

#### SleeveFS

- Refactor `ImageLoader` module for `SleeveFS` integration.
- Integrate `SleeveView` (rolling thubmbnail cache) with `Pipeline`  for thumbnail generation.

#### Pipeline and Image Processing

- [x] Add preliminary implementation for CUDA pipelines.
- [ ] Modify preview mode in `PipelineScheduler` to support full-resolution image silent processing.
- [x] Implement RCD algorithm as the default method for raw demosaicing.
- [x] Add experimental panorama stitching module (complete algorithm, needs integration).
- [x] Fix magic numbers in highlight reconstruction algorithm (sort of, thanks to Claude Opus 4.5).
- [ ] Explore multi-threading / SIMD optimizations for image processing tasks (on-going).
- [ ] Add support for JSON serialization of pipeline configurations.
- [x] Add support for custom WB adjustments in raw processing pipeline.

### January 2026

#### Pipeline and Image Processing

- [ ] Add global parameter sychronization across different devices.
- [ ] Modify LUT texture object lifecycle management
- [ ] Improve CUDA pipeline stability and error handling.

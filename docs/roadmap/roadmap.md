# Roadmap

## Upcoming Features and Improvements

### December 2025

#### SleeveFS

- Refactor `ImageLoader` module for `SleeveFS` integration.
- Integrate `SleeveView` (rolling thubmbnail cache) with `Pipeline`  for thumbnail generation.

#### Pipeline and Image Processing

- Add preliminary implementation for CUDA pipelines.
- Modify preview mode in `PipelineScheduler` to support full-resolution image silent processing.
- Implement RCD algorithm as the default method for raw demosaicing.
- Add experimental panorama stitching module.
- Fix magic numbers in highlight reconstruction algorithm.
- Explore multi-threading / SIMD optimizations for image processing tasks.
- Add support for JSON serialization of pipeline configurations.

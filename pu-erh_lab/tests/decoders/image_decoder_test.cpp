#include "decoders/image_decoder.hpp"
#include "type/type.hpp"

#include <gtest/gtest.h>

TEST(SingleImageDecoder, BasicAssertions) {
  // Test decode only one image
  puerhlab::ImageDecoder image_decoder(1, 2);
}
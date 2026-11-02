/*
 Copyright 2016-2026 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <cstddef>

namespace ccl::sycl_policy {

// Scaleout pipeline chunk size used when no tuning rule picks one.
constexpr std::size_t default_pipeline_chunk_size = 2 * 1024 * 1024;

// Smallest pipeline chunk alignment, in bytes.
constexpr std::size_t min_pipeline_alignment = 4;

// Pipeline chunk alignment; a chunk must be at least this large.
constexpr std::size_t pipeline_alignment(std::size_t dsize) noexcept {
    return dsize > min_pipeline_alignment ? dsize : min_pipeline_alignment;
}

}  // namespace ccl::sycl_policy

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

#include "host_buffer.h"

namespace oneccl_test {

HostBuffer::HostBuffer(size_t bytes) : storage_(bytes) {}

HostBuffer::HostBuffer(std::vector<std::byte> host_values)
    : storage_(std::move(host_values)) {}

std::byte *HostBuffer::data() { return storage_.data(); }

const std::byte *HostBuffer::data() const { return storage_.data(); }

size_t HostBuffer::size() const { return storage_.size(); }

} // namespace oneccl_test

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

#include <optional>

#include "coll/algorithms/utils/sycl_policy.hpp"
#include "coll/algorithms/utils/sycl_selection.hpp"
#include "coll/selection/policy/selection_engine.hpp"

namespace ccl::sycl_policy_resolver::json {

// JSON policy adapter: applies the decision's env-bound knobs and returns it as
// a tune attr, or nullopt if the policy has no usable decision.
std::optional<sycl_policy::sycl_scaleup_tune_attr>
    scaleup(const sycl_policy::policy_query& q);

std::optional<::sycl_scaleout_tune_attr>
    scaleout(const sycl_policy::policy_query& q);
std::optional<sycl_policy::sycl_scaleup_tune_attr>
    scaleup(const sycl_policy::policy_query& q,
             const selection::SelectionEngine& engine);

std::optional<::sycl_scaleout_tune_attr>
    scaleout(const sycl_policy::policy_query& q,
              const selection::SelectionEngine& engine);

}  // namespace ccl::sycl_policy_resolver::json

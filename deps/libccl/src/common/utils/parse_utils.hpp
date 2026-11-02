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

// Number syntax of the selection policy (CCL_SELECTION_POLICY_PATH): filter
// values, leaf knobs and global_config values, including global_config CCL_*
// entries. Environment variables themselves keep their operator>> parsing.
//
//   <number>          plain number
//   <number>K|M|G|T   decimal multiplier: x1000, x1000^2, x1000^3, x1000^4
//   <number>Kb|Mb|... binary multiplier:  x1024, x1024^2, x1024^3, x1024^4
//
// Suffixes are case-insensitive: 12M = 12m = 12000000, 12Mb = 12mb = 12MB =
// 12 * 1024 * 1024. Rejected: empty input, spaces, a bare `b`, `iB`, hex,
// exponents and overflow. Only parse_int and parse_double accept a sign, and
// only parse_double accepts a fraction.
//
// Hand-rolled (no <charconv>), as in the selection engine: some icpx
// toolchains don't ship a matching libstdc++ for it.

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace ccl {
namespace utils {

namespace parse_detail {

struct split_number {
    std::string_view digits; // the number without its suffix
    std::uint64_t multiplier;
};

inline char to_lower(char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

// 0 if `c` is not a unit letter.
inline int unit_power(char c) {
    switch (to_lower(c)) {
        case 'k': return 1;
        case 'm': return 2;
        case 'g': return 3;
        case 't': return 4;
        default: return 0;
    }
}

inline std::uint64_t power(std::uint64_t base, int exp) {
    std::uint64_t r = 1;
    for (int i = 0; i < exp; ++i) {
        r *= base;
    }
    return r;
}

// Splits "<number>[<unit>[b]]"; nullopt for a malformed suffix or no number.
inline std::optional<split_number> split_suffix(std::string_view s) {
    if (s.empty()) {
        return std::nullopt;
    }
    std::size_t n = s.size();
    std::uint64_t mult = 1;
    if (to_lower(s[n - 1]) == 'b') {
        // binary: needs a unit letter right before the `b`
        if (n < 2 || unit_power(s[n - 2]) == 0) {
            return std::nullopt;
        }
        mult = power(1024, unit_power(s[n - 2]));
        n -= 2;
    }
    else if (unit_power(s[n - 1]) != 0) {
        mult = power(1000, unit_power(s[n - 1]));
        n -= 1;
    }
    if (n == 0) {
        return std::nullopt;
    }
    return split_number{ s.substr(0, n), mult };
}

inline std::optional<std::uint64_t> parse_digits(std::string_view s) {
    if (s.empty()) {
        return std::nullopt;
    }
    constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9') {
            return std::nullopt;
        }
        const auto d = static_cast<std::uint64_t>(c - '0');
        if (v > (max - d) / 10) {
            return std::nullopt;
        }
        v = v * 10 + d;
    }
    return v;
}

} // namespace parse_detail

// Unsigned size or count, e.g. "4096", "8M", "8Mb".
inline std::optional<std::uint64_t> parse_size(std::string_view s) {
    const auto split = parse_detail::split_suffix(s);
    if (!split) {
        return std::nullopt;
    }
    const auto v = parse_detail::parse_digits(split->digits);
    if (!v || *v > std::numeric_limits<std::uint64_t>::max() / split->multiplier) {
        return std::nullopt;
    }
    return *v * split->multiplier;
}

// Signed integer: parse_size with an optional leading '-', e.g. "-1", "-2K".
inline std::optional<std::int64_t> parse_int(std::string_view s) {
    const bool negative = !s.empty() && s[0] == '-';
    const auto mag = parse_size(negative ? s.substr(1) : s);
    if (!mag) {
        return std::nullopt;
    }
    constexpr auto max = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (negative) {
        if (*mag > max + 1) {
            return std::nullopt;
        }
        return *mag == max + 1 ? std::numeric_limits<std::int64_t>::min()
                               : -static_cast<std::int64_t>(*mag);
    }
    if (*mag > max) {
        return std::nullopt;
    }
    return static_cast<std::int64_t>(*mag);
}

// Real number: [-]<digits>[.<digits>] with an optional suffix, e.g. "0.5", "1.5M".
inline std::optional<double> parse_double(std::string_view s) {
    const bool negative = !s.empty() && s[0] == '-';
    const auto split = parse_detail::split_suffix(negative ? s.substr(1) : s);
    if (!split) {
        return std::nullopt;
    }
    // validate the shape ourselves so strtod can't accept "inf", "0x1p3", "1e5", ...
    bool seen_digit = false;
    bool seen_dot = false;
    for (char c : split->digits) {
        if (c >= '0' && c <= '9') {
            seen_digit = true;
        }
        else if (c == '.' && !seen_dot) {
            seen_dot = true;
        }
        else {
            return std::nullopt;
        }
    }
    if (!seen_digit) {
        return std::nullopt;
    }
    const std::string text(split->digits);
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size()) {
        return std::nullopt;
    }
    const double r = v * static_cast<double>(split->multiplier);
    return negative ? -r : r;
}

// "yes" / "true" / "1" and "no" / "false" / "0".
inline std::optional<bool> parse_bool(std::string_view s) {
    if (s == "yes" || s == "true" || s == "1") {
        return true;
    }
    if (s == "no" || s == "false" || s == "0") {
        return false;
    }
    return std::nullopt;
}

// Reads any arithmetic type with the syntax above; nullopt if malformed or out
// of range for T.
template <class T>
std::optional<T> parse_number(std::string_view s) {
    static_assert(std::is_arithmetic<T>::value, "parse_number needs an arithmetic type");
    if constexpr (std::is_same<T, bool>::value) {
        return parse_bool(s);
    }
    else if constexpr (std::is_floating_point<T>::value) {
        const auto v = parse_double(s);
        if (!v) {
            return std::nullopt;
        }
        return static_cast<T>(*v);
    }
    else if constexpr (std::is_unsigned<T>::value) {
        const auto v = parse_size(s);
        if (!v || *v > static_cast<std::uint64_t>(std::numeric_limits<T>::max())) {
            return std::nullopt;
        }
        return static_cast<T>(*v);
    }
    else {
        const auto v = parse_int(s);
        if (!v || *v < static_cast<std::int64_t>(std::numeric_limits<T>::min()) ||
            *v > static_cast<std::int64_t>(std::numeric_limits<T>::max())) {
            return std::nullopt;
        }
        return static_cast<T>(*v);
    }
}

} // namespace utils
} // namespace ccl

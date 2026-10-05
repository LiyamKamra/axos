// SPDX-License-Identifier: BSD-3-Clause
//
// Debug printing of tensors (docs/TENSOR_SPEC.md §8): nested brackets, one
// level per dimension, innermost elements separated by two spaces, every
// sub-array on its own line indented by two spaces per level:
//
//   [
//     [1  2  3]
//     [4  5  6]
//   ]
//
// Device tensors are copied to the host first. Not a serialization format.
#pragma once

#include "tensorET.h"
#include <array>
#include <ostream>
#include <string>
#include <type_traits>

namespace AXOS {
namespace io_detail {

template <class T>
void
put(std::ostream &os, const T &v)
{
    if constexpr (is_half_v<T>)
        os << static_cast<float>(v);
    else if constexpr (std::is_same_v<T, signed char> ||
                       std::is_same_v<T, unsigned char>)
        os << static_cast<int>(v);
    else
        os << v;
}

template <int DIM, class T>
void
print_level(std::ostream &os, const tensorET<DIM, T> &t, int d,
    std::array<size_t, DIM> &idx)
{
    const std::string pad(size_t(2 * d), ' ');
    if (d == DIM - 1) {
        os << pad << '[';
        for (size_t i = 0; i < t.size(d); ++i) {
            idx[size_t(d)] = i;
            size_t off = 0;
            for (int k = 0; k < DIM; ++k)
                off += idx[size_t(k)] * t.stride(k);
            if (i) os << "  ";
            put(os, t.data[off]);
        }
        os << ']';
        return;
    }
    os << pad << "[\n";
    for (size_t i = 0; i < t.size(d); ++i) {
        idx[size_t(d)] = i;
        print_level<DIM, T>(os, t, d + 1, idx);
        os << '\n';
    }
    os << pad << ']';
}

} // namespace io_detail

template <int DIM, class T, class S>
std::ostream &
operator<<(std::ostream &os, const tensorET<DIM, T, S> &t)
{
    if constexpr (!is_host_storage_v<S>) {
        return os << tensorET<DIM, T>(t);
    } else {
        std::array<size_t, DIM> idx{};
        io_detail::print_level<DIM, T>(
            os, static_cast<const tensorET<DIM, T> &>(t), 0, idx);
        return os;
    }
}

} // namespace AXOS

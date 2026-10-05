// SPDX-License-Identifier: BSD-3-Clause
//
// QPS reader: MPS plus a quadratic objective section (QUADOBJ, QMATRIX or
// QSECTION), as used by the Maros-Meszaros test set. Returns a QpProblem
// with Q stored as a full symmetric CSR matrix (duplicates summed). A
// maximization problem is negated (c, offset and Q), matching the LP reader.
#pragma once

#include "solver/io/mps.h"
#include "solver/qp/qp_model.h"
#include <fstream>
#include <istream>
#include <string>
#include <vector>

namespace AXOS {
namespace Solver {

inline QpProblem
read_qps(std::istream &in, bool fixed_format = false)
{
    std::vector<QuadEntry> quad;
    QpProblem p;
    p.lp = read_mps(in, &quad, fixed_format);
    const size_t n = p.lp.cols();
    Sparse::CooBuilder<double> b(n, n);
    b.reserve(quad.size());
    const double sgn = p.lp.maximize ? -1.0 : 1.0;
    for (const auto &e : quad)
        if (e.v != 0) b.add(static_cast<size_t>(e.i), static_cast<size_t>(e.j), sgn * e.v);
    p.Q = b.build(true);
    return p;
}

inline QpProblem
read_qps_file(const std::string &path)
{
    std::vector<char> buf(1 << 20); // large files: a bigger stream buffer
    std::ifstream f;
    f.rdbuf()->pubsetbuf(buf.data(), static_cast<std::streamsize>(buf.size()));
    f.open(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open QPS file: " + path);
    QpProblem p;
    try {
        p = read_qps(f);
    } catch (const std::runtime_error &) {
        // names with spaces: retry reading by column positions
        f.clear();
        f.seekg(0);
        p = read_qps(f, true);
    }
    if (p.lp.name.empty()) {
        const size_t s = path.find_last_of("/\\");
        p.lp.name = path.substr(s == std::string::npos ? 0 : s + 1);
    }
    return p;
}

} // namespace Solver
} // namespace AXOS

// SPDX-License-Identifier: BSD-3-Clause
//
// MPS reader and writer (free-format tokenization, which also reads normal
// fixed-format files as long as names contain no spaces).
//
// Supported: NAME, OBJSENSE / OBJSENSE MAX|MIN, OBJNAME, ROWS (N/L/G/E, extra
// N rows are dropped), COLUMNS (with MARKER INTORG/INTEND), RHS (a value on
// the objective row is minus the objective constant), RANGES, BOUNDS
// (UP LO FX FR MI PL BV LI UI), ENDATA. Values with magnitude >= 1e20 are
// treated as infinite. SC (semi-continuous) bounds are rejected.
// Quadratic objective sections (QPS: QUADOBJ with one triangle, QMATRIX with
// both, QSECTION for the objective row) are returned through `quad` as full
// symmetric entries of Q (objective 1/2 x^T Q x); without `quad` they are an
// error. See solver/io/qps.h.
//
// A maximization problem is returned negated with `maximize = true`, matching
// the LpProblem convention.
#pragma once

#include "solver/model.h"
#include "sparse/sparse_cpu.h" // Csr::transpose() in write_mps
#include <cstdlib>
#include <fstream>
#include <istream>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace AXOS {
namespace Solver {

namespace mps_detail {

inline constexpr double kMpsInf = 1e20;

inline double
clamp_inf(double v)
{
    if (v >= kMpsInf) return kInf;
    if (v <= -kMpsInf) return -kInf;
    return v;
}

inline std::vector<std::string>
tokens(const std::string &line)
{
    std::vector<std::string> t;
    std::istringstream is(line);
    std::string w;
    while (is >> w)
        t.push_back(w);
    return t;
}

// Fixed-format MPS data line: the non-empty fields at columns 2-3, 5-12,
// 15-22, 25-36, 40-47 and 50-61, trimmed (names may contain spaces).
inline std::vector<std::string>
fixed_tokens(const std::string &line)
{
    static const size_t pos[6][2] = {{1, 3}, {4, 12}, {14, 22}, {24, 36}, {39, 47}, {49, 61}};
    std::vector<std::string> t;
    for (const auto &f : pos) {
        if (f[0] >= line.size()) break;
        std::string s = line.substr(f[0], f[1] - f[0]);
        const size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
        if (a != std::string::npos) t.push_back(s.substr(a, b - a + 1));
    }
    return t;
}

inline double
to_double(const std::string &s, size_t line)
{
    char *end = nullptr;
    double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || *end != '\0')
        throw std::runtime_error("MPS line " + std::to_string(line) +
                                 ": bad number '" + s + "'");
    return v;
}

inline std::string
upper(std::string s)
{
    for (auto &c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

} // namespace mps_detail

struct QuadEntry {
    int i, j;
    double v;
};

// fixed_format: split data lines by column position (for files whose names
// contain spaces); header lines are always split on whitespace.
inline LpProblem
read_mps(std::istream &in, std::vector<QuadEntry> *quad = nullptr,
    bool fixed_format = false)
{
    using namespace mps_detail;
    LpProblem p;
    enum class Sec { None, Name, ObjSense, ObjName, Rows, Columns, Rhs, Ranges, Bounds, Quad, End };
    bool quad_full = false; // QMATRIX lists both triangles, QUADOBJ one
    Sec sec = Sec::None;

    std::string obj_row, obj_name_wanted;
    std::vector<char> row_type;                     // per constraint row
    std::vector<double> rhs, range;                 // per constraint row
    std::vector<std::string> row_names;
    std::unordered_map<std::string, int> row_index; // constraint rows only
    std::unordered_set<std::string> dropped_rows;   // extra N rows
    std::unordered_map<std::string, int> col_index;
    std::vector<double> c, lb, ub;
    std::vector<uint8_t> integer;
    std::vector<std::string> col_names;
    std::vector<int> ci_i, ci_j; // triplets
    std::vector<double> ci_v;
    bool maximize = false, in_int = false, have_rhs_obj = false;
    double obj_rhs = 0;

    auto col_of = [&](const std::string &name) {
        auto it = col_index.find(name);
        if (it != col_index.end()) return it->second;
        int id = static_cast<int>(col_names.size());
        col_index[name] = id;
        col_names.push_back(name);
        c.push_back(0);
        lb.push_back(0);
        ub.push_back(kInf);
        integer.push_back(in_int ? 1 : 0);
        return id;
    };

    std::string line;
    size_t ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '*') continue;
        const bool data_line = std::isspace(static_cast<unsigned char>(line[0]));
        auto t = (fixed_format && data_line) ? fixed_tokens(line) : tokens(line);
        if (t.empty()) continue;

        if (!std::isspace(static_cast<unsigned char>(line[0]))) { // header
            std::string h = upper(t[0]);
            if (h == "NAME") {
                sec = Sec::Name;
                p.name = t.size() > 1 ? t[1] : "";
            } else if (h == "OBJSENSE") {
                sec = Sec::ObjSense;
                if (t.size() > 1) maximize = upper(t[1]).rfind("MAX", 0) == 0;
            } else if (h == "OBJNAME") {
                sec = Sec::ObjName;
            } else if (h == "ROWS") sec = Sec::Rows;
            else if (h == "COLUMNS") sec = Sec::Columns;
            else if (h == "RHS") sec = Sec::Rhs;
            else if (h == "RANGES") sec = Sec::Ranges;
            else if (h == "BOUNDS") sec = Sec::Bounds;
            else if (h == "QUADOBJ" || h == "QMATRIX" || h == "QSECTION") {
                if (!quad)
                    throw std::runtime_error("MPS line " + std::to_string(ln) +
                                             ": quadratic objective; read it with read_qps");
                sec = Sec::Quad;
                quad_full = (h == "QMATRIX");
                if (h == "QSECTION" && t.size() > 1 && t[1] != obj_row)
                    throw std::runtime_error("MPS line " + std::to_string(ln) +
                                             ": quadratic constraints are not supported");
            }
            else if (h == "ENDATA") { sec = Sec::End; break; }
            else
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": unknown section '" + t[0] + "'");
            continue;
        }

        switch (sec) {
        case Sec::ObjSense:
            maximize = upper(t[0]).rfind("MAX", 0) == 0;
            break;
        case Sec::ObjName:
            obj_name_wanted = t[0];
            break;
        case Sec::Rows: {
            if (t.size() < 2)
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": bad ROWS entry");
            char ty = static_cast<char>(std::toupper(t[0][0]));
            if (ty == 'N') {
                if (obj_row.empty() &&
                    (obj_name_wanted.empty() || obj_name_wanted == t[1]))
                    obj_row = t[1];
                else
                    dropped_rows.insert(t[1]); // extra N rows are ignored
            } else if (ty == 'L' || ty == 'G' || ty == 'E') {
                row_index[t[1]] = static_cast<int>(row_type.size());
                row_type.push_back(ty);
                rhs.push_back(0);
                range.push_back(std::nan(""));
                row_names.push_back(t[1]);
            } else {
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": bad row type '" + t[0] + "'");
            }
            break;
        }
        case Sec::Columns: {
            if (t.size() >= 3 && t[1] == "'MARKER'") {
                in_int = (t[2] == "'INTORG'");
                break;
            }
            if (t.size() < 3 || t.size() % 2 == 0)
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": bad COLUMNS entry");
            int j = col_of(t[0]);
            for (size_t k = 1; k + 1 < t.size(); k += 2) {
                double v = to_double(t[k + 1], ln);
                if (t[k] == obj_row) {
                    c[j] += v;
                } else {
                    auto it = row_index.find(t[k]);
                    if (it == row_index.end()) {
                        if (dropped_rows.count(t[k])) continue;
                        throw std::runtime_error("MPS line " + std::to_string(ln) +
                                                 ": unknown row '" + t[k] + "'");
                    }
                    ci_i.push_back(it->second);
                    ci_j.push_back(j);
                    ci_v.push_back(v);
                }
            }
            break;
        }
        case Sec::Rhs:
        case Sec::Ranges: {
            size_t start = (t.size() % 2 == 1) ? 1 : 0; // optional set name
            for (size_t k = start; k + 1 < t.size(); k += 2) {
                double v = to_double(t[k + 1], ln);
                if (sec == Sec::Rhs && t[k] == obj_row) {
                    obj_rhs = v;
                    have_rhs_obj = true;
                    continue;
                }
                auto it = row_index.find(t[k]);
                if (it == row_index.end()) {
                    if (dropped_rows.count(t[k])) continue;
                    throw std::runtime_error("MPS line " + std::to_string(ln) +
                                             ": unknown row '" + t[k] + "'");
                }
                if (sec == Sec::Rhs) rhs[it->second] = v;
                else range[it->second] = v;
            }
            break;
        }
        case Sec::Bounds: {
            std::string ty = upper(t[0]);
            bool valued = !(ty == "FR" || ty == "MI" || ty == "PL" || ty == "BV");
            size_t need = valued ? 3 : 2; // without a bound-set name
            if (t.size() < need)
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": bad BOUNDS entry");
            size_t name_at = (t.size() == need) ? 1 : 2;
            int j = col_of(t[name_at]);
            double v = valued ? clamp_inf(to_double(t[name_at + 1], ln)) : 0;
            if (ty == "UP") {
                ub[j] = v;
                if (v < 0 && lb[j] == 0) lb[j] = -kInf;
            } else if (ty == "LO") lb[j] = v;
            else if (ty == "FX") lb[j] = ub[j] = v;
            else if (ty == "FR") { lb[j] = -kInf; ub[j] = kInf; }
            else if (ty == "MI") lb[j] = -kInf;
            else if (ty == "PL") ub[j] = kInf;
            else if (ty == "BV") { lb[j] = 0; ub[j] = 1; integer[j] = 1; }
            else if (ty == "LI") { lb[j] = v; integer[j] = 1; }
            else if (ty == "UI") { ub[j] = v; integer[j] = 1; }
            else
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": unsupported bound type '" + t[0] + "'");
            break;
        }
        case Sec::Quad: {
            if (t.size() < 3)
                throw std::runtime_error("MPS line " + std::to_string(ln) +
                                         ": bad quadratic entry");
            const int a = col_of(t[0]), b = col_of(t[1]);
            const double v = to_double(t[2], ln);
            quad->push_back({a, b, v});
            if (!quad_full && a != b) quad->push_back({b, a, v});
            break;
        }
        default:
            throw std::runtime_error("MPS line " + std::to_string(ln) +
                                     ": data outside a section");
        }
    }
    if (obj_row.empty() && row_type.empty() && col_names.empty())
        throw std::runtime_error("MPS: empty or malformed file");

    const size_t m = row_type.size(), n = col_names.size();
    p.row_lb.assign(m, -kInf);
    p.row_ub.assign(m, kInf);
    for (size_t i = 0; i < m; ++i) {
        const double r = clamp_inf(rhs[i]);
        const bool has_range = !std::isnan(range[i]);
        const double R = has_range ? std::abs(range[i]) : 0;
        switch (row_type[i]) {
        case 'L':
            p.row_ub[i] = r;
            if (has_range) p.row_lb[i] = r - R;
            break;
        case 'G':
            p.row_lb[i] = r;
            if (has_range) p.row_ub[i] = r + R;
            break;
        default: // E
            if (!has_range) { p.row_lb[i] = p.row_ub[i] = r; }
            else if (range[i] >= 0) { p.row_lb[i] = r; p.row_ub[i] = r + R; }
            else { p.row_lb[i] = r - R; p.row_ub[i] = r; }
        }
    }
    Sparse::CooBuilder<double> b(m, n);
    b.reserve(ci_v.size());
    for (size_t k = 0; k < ci_v.size(); ++k)
        b.add(ci_i[k], ci_j[k], ci_v[k]);
    p.A = b.build();
    p.c = c;
    p.col_lb = lb;
    p.col_ub = ub;
    p.offset = have_rhs_obj ? -obj_rhs : 0.0;
    p.row_names = row_names;
    p.col_names = col_names;
    bool any_int = false;
    for (auto v : integer) any_int |= (v != 0);
    if (any_int) p.is_integer = integer;
    if (maximize) {
        p.maximize = true;
        for (auto &v : p.c) v = -v;
        p.offset = -p.offset;
    }
    return p;
}

inline LpProblem
read_mps_file(const std::string &path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open MPS file: " + path);
    return read_mps(f);
}

// Writes free-format MPS that read_mps() (and other readers) accept.
inline void
write_mps(std::ostream &out, const LpProblem &p)
{
    const size_t m = p.rows(), n = p.cols();
    auto rname = [&](size_t i) {
        return i < p.row_names.size() ? p.row_names[i] : "R" + std::to_string(i + 1);
    };
    auto cname = [&](size_t j) {
        return j < p.col_names.size() ? p.col_names[j] : "C" + std::to_string(j + 1);
    };
    out.precision(17);
    const double sgn = p.maximize ? -1.0 : 1.0;
    out << "NAME " << (p.name.empty() ? "AXOS" : p.name) << "\n";
    if (p.maximize) out << "OBJSENSE\n    MAX\n";
    out << "ROWS\n N OBJ\n";
    // A free row (both bounds infinite) constrains nothing; it is written as
    // an extra N row, which readers drop.
    for (size_t i = 0; i < m; ++i) {
        const bool lo = std::isfinite(p.row_lb[i]), hi = std::isfinite(p.row_ub[i]);
        char ty = (lo && hi && p.row_lb[i] == p.row_ub[i]) ? 'E'
                  : hi ? 'L' : lo ? 'G' : 'N';
        out << ' ' << ty << ' ' << rname(i) << "\n";
    }
    // Column-wise entries via the transpose.
    HostMatrix At = p.A.transpose();
    out << "COLUMNS\n";
    bool in_int = false;
    int marker = 0;
    for (size_t j = 0; j < n; ++j) {
        const bool is_int = !p.is_integer.empty() && p.is_integer[j];
        if (is_int != in_int) {
            out << "    MARKER" << marker++ << " 'MARKER' '"
                << (is_int ? "INTORG" : "INTEND") << "'\n";
            in_int = is_int;
        }
        if (p.c[j] != 0) out << "    " << cname(j) << " OBJ " << sgn * p.c[j] << "\n";
        for (int k = At.row_ptr()[j]; k < At.row_ptr()[j + 1]; ++k)
            out << "    " << cname(j) << ' ' << rname(At.col_ind()[k]) << ' '
                << At.values()[k] << "\n";
        if (p.c[j] == 0 && At.row_ptr()[j] == At.row_ptr()[j + 1])
            out << "    " << cname(j) << " OBJ 0\n"; // keep empty columns
    }
    if (in_int) out << "    MARKER" << marker++ << " 'MARKER' 'INTEND'\n";
    out << "RHS\n";
    if (p.offset != 0) out << "    RHS OBJ " << -sgn * p.offset << "\n";
    std::vector<int> ranged;
    for (size_t i = 0; i < m; ++i) {
        const bool lo = std::isfinite(p.row_lb[i]), hi = std::isfinite(p.row_ub[i]);
        double r = (lo && hi && p.row_lb[i] == p.row_ub[i]) ? p.row_lb[i]
                   : hi ? p.row_ub[i] : lo ? p.row_lb[i] : 0.0;
        if (r != 0) out << "    RHS " << rname(i) << ' ' << r << "\n";
        if (lo && hi && p.row_lb[i] != p.row_ub[i]) ranged.push_back(static_cast<int>(i));
    }
    if (!ranged.empty()) {
        out << "RANGES\n";
        for (int i : ranged)
            out << "    RNG " << rname(i) << ' ' << (p.row_ub[i] - p.row_lb[i]) << "\n";
    }
    out << "BOUNDS\n";
    for (size_t j = 0; j < n; ++j) {
        const double l = p.col_lb[j], u = p.col_ub[j];
        const bool is_int = !p.is_integer.empty() && p.is_integer[j];
        if (l == u) { out << " FX BND " << cname(j) << ' ' << l << "\n"; continue; }
        if (std::isinf(l) && std::isinf(u)) { out << " FR BND " << cname(j) << "\n"; continue; }
        if (std::isinf(l)) out << " MI BND " << cname(j) << "\n";
        else if (l != 0) out << " LO BND " << cname(j) << ' ' << l << "\n";
        if (std::isfinite(u)) out << " UP BND " << cname(j) << ' ' << u << "\n";
        else if (is_int) out << " PL BND " << cname(j) << "\n";
    }
    out << "ENDATA\n";
}

} // namespace Solver
} // namespace AXOS

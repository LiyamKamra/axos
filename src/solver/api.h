// SPDX-License-Identifier: BSD-3-Clause
//
// The AXOS solver API: one entry point for LP, QP and MILP.
//
//   #include "solver/api.h"
//   using namespace AXOS::Solver;
//
//   Model m = read_model("problem.mps");      // MPS, or QPS (QUADOBJ / QMATRIX)
//   Options o;
//   o.time_limit = 60;                        // seconds
//   Result r = solve(m, o);                   // LP, QP or MILP, from the model
//   if (r.has_solution()) use(r.x, r.objective);
//
// The problem type follows from the model: MILP when a column is integer,
// QP when the objective has a quadratic part, else LP (Options::type forces
// one, e.g. the LP relaxation of a MILP). Methods (Options::method):
//
//   LP    auto     dual simplex; HPR on the GPU (or CPU) if it fails or the
//                  problem is very large
//         simplex  bounded dual simplex (exact vertex solution)
//         ipm      Mehrotra interior point
//         pdlp     restarted PDHG (first order, CPU)
//         hpr      Halpern Peaceman-Rachford (first order, GPU when available)
//   QP    auto     interior point when its factorization is cheap, else HPR-QP
//         hprqp    HPR-QP (GPU or CPU)
//         pdhcg    PDHCG (GPU or CPU)
//         ipm      interior point (CPU)
//   MILP  auto     branch and cut (presolve, probing, cuts, heuristics; with
//                  Device::Gpu / Auto the GPU probes, runs feasibility-jump
//                  walkers and batched strong branching)
//
// Objective values are in the model's own sense (a MAX model reports its
// maximum). Every returned solution is checked against the model:
// Result::max_violation is its largest bound / row violation (relative to
// 1 + |side|), integrality included for a MILP.
//
// MPI (build with -DAXOS_ENABLE_MPI): with Options::comm over several ranks,
// every rank calls solve() with the same model and options; LPs and QPs are
// solved by HPR (the "auto", "hpr" and "hprqp" methods) with the constraint
// rows split over the ranks, one GPU per rank (qp_dist.h), and every rank
// gets the whole solution. MILPs run on one process.
#pragma once

#include "solver/io/mps.h"
#include "solver/io/qps.h"
#include "solver/lp/simplex.h"
#include "solver/lp/solve_lp.h"
#include "solver/milp/solve_milp.h"
#include "solver/qp/solve_qp.h"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace AXOS {
namespace Solver {

inline constexpr const char *kAxosVersion = "1.0";

enum class ProblemType { Auto, LP, QP, MILP };
enum class Device { Auto, Cpu, Gpu };

inline const char *
to_string(ProblemType t)
{
    switch (t) {
    case ProblemType::LP: return "LP";
    case ProblemType::QP: return "QP";
    case ProblemType::MILP: return "MILP";
    default: return "auto";
    }
}

inline const char *
to_string(Device d)
{
    return d == Device::Cpu ? "cpu" : d == Device::Gpu ? "gpu" : "auto";
}

// An optimization model: the linear part (constraints, bounds, costs,
// integrality) in qp.lp, the quadratic objective part (if any) in qp.Q.
struct Model {
    QpProblem qp;

    ProblemType
    type() const
    {
        if (qp.lp.has_integers()) return ProblemType::MILP;
        if (qp.has_quadratic()) return ProblemType::QP;
        return ProblemType::LP;
    }
    const std::string &name() const { return qp.lp.name; }
    size_t rows() const { return qp.rows(); }
    size_t cols() const { return qp.cols(); }
    size_t nnz() const { return qp.lp.A.nnz(); }
    size_t integers() const
    {
        size_t k = 0;
        for (auto v : qp.lp.is_integer) k += v ? 1 : 0;
        return k;
    }
    bool maximize() const { return qp.lp.maximize; }
};

// Reads an MPS or QPS file (free or fixed format).
inline Model
read_model(const std::string &path)
{
    Model m;
    m.qp = read_qps_file(path);
    const std::string bad = m.qp.validate();
    if (!bad.empty()) throw std::runtime_error("invalid model " + path + ": " + bad);
    return m;
}

struct Options {
    ProblemType type = ProblemType::Auto; // force LP / QP / MILP
    std::string method = "auto";
    Device device = Device::Auto;
    double time_limit = 3600; // seconds
    double tol = 1e-6;        // LP / QP: relative KKT tolerance
    double gap = 1e-4;        // MILP: relative gap
    int verbose = 0;
    bool presolve = true;     // MILP switches
    bool cuts = true;
    bool heuristics = true;
    const QpComm *comm = nullptr; // MPI ranks of a distributed LP / QP solve (qp_dist.h)
};

struct Result {
    Status status = Status::NotSolved;
    ProblemType type = ProblemType::Auto;
    std::string method, device, message;
    double objective = std::numeric_limits<double>::quiet_NaN(); // of x, in the model's sense
    double bound = std::numeric_limits<double>::quiet_NaN();     // dual bound (MILP) / dual objective (LP, QP)
    std::vector<double> x;     // primal solution (columns), empty if none
    std::vector<double> y;     // row duals (LP, QP; minimization form)
    double seconds = 0;
    long iterations = 0;       // simplex / IPM / first-order iterations (MILP: LP iterations)
    long nodes = 0;            // MILP
    double max_violation = std::numeric_limits<double>::quiet_NaN();

    bool has_solution() const { return !x.empty(); }
    // relative gap between objective and bound (MILP convention)
    double
    gap() const
    {
        if (!has_solution() || std::isnan(bound)) return std::numeric_limits<double>::infinity();
        const double d = std::abs(objective - bound);
        if (d <= 1e-9) return 0;
        return d / std::max(std::abs(objective), 1e-9);
    }
};

namespace api_detail {

inline double
max_violation(const LpProblem &p, const std::vector<double> &x, bool integrality)
{
    double v = 0;
    for (size_t j = 0; j < p.cols(); ++j) {
        if (x[j] < p.col_lb[j]) v = std::max(v, (p.col_lb[j] - x[j]) / (1 + std::abs(p.col_lb[j])));
        if (x[j] > p.col_ub[j]) v = std::max(v, (x[j] - p.col_ub[j]) / (1 + std::abs(p.col_ub[j])));
        if (integrality && !p.is_integer.empty() && p.is_integer[j]) v = std::max(v, std::abs(x[j] - std::round(x[j])));
    }
    const int32_t *rp = p.A.row_ptr(), *ci = p.A.col_ind();
    const double *va = p.A.values();
    for (size_t i = 0; i < p.rows(); ++i) {
        double a = 0;
        for (int32_t k = rp[i]; k < rp[i + 1]; ++k) a += va[k] * x[ci[k]];
        if (a < p.row_lb[i]) v = std::max(v, (p.row_lb[i] - a) / (1 + std::abs(p.row_lb[i])));
        if (a > p.row_ub[i]) v = std::max(v, (a - p.row_ub[i]) / (1 + std::abs(p.row_ub[i])));
    }
    return v;
}

inline bool
gpu_wanted(Device d)
{
    return d == Device::Gpu || (d == Device::Auto && qp_gpu_available());
}

inline void
from_qp(const QpSolution &s, double sense, Result &r)
{
    r.status = s.status;
    if (!s.x.empty() && (s.status == Status::Optimal || s.status == Status::TimeLimit ||
                            s.status == Status::IterationLimit)) {
        r.x = s.x;
        r.y = s.y;
        r.objective = sense * s.primal_objective;
        r.bound = sense * s.dual_objective;
    }
    r.iterations = s.iterations;
    r.method = s.method;
    r.device = s.device;
}

inline Result
solve_lp_model(const Model &m, const Options &o)
{
    Result r;
    const LpProblem &p = m.qp.lp;
    const double sense = p.maximize ? -1.0 : 1.0;
    const std::string meth = o.method;
    auto hpr = [&](double tl) {
        QpProblem q = m.qp; // the reader's Q: n x n, empty
        q.lp.is_integer.clear();
        QpOptions qo;
        qo.method = QpMethod::HprQp;
        qo.use_gpu = gpu_wanted(o.device);
        qo.tol = o.tol;
        qo.time_limit = tl;
        qo.verbose = o.verbose;
        qo.comm = o.comm;
        from_qp(solve_qp(q, qo), sense, r);
    };
    if (o.comm && o.comm->distributed()) { // over MPI ranks: HPR only
        if (meth != "auto" && meth != "hpr")
            throw std::invalid_argument("over MPI ranks an LP is solved by HPR (method auto or hpr)");
        hpr(o.time_limit);
        return r;
    }
    LpProblem q = p;
    q.is_integer.clear();
    SolverOptions so;
    so.time_limit = o.time_limit;
    so.verbose = o.verbose > 1;
    so.set_tolerance(o.tol);
    if (meth == "hpr") {
        hpr(o.time_limit);
        return r;
    }
    if (meth == "auto" && q.A.nnz() > 5000000 && gpu_wanted(o.device)) {
        hpr(o.time_limit);
        return r;
    }
    so.method = meth == "ipm" ? LpMethod::Ipm : meth == "pdlp" ? LpMethod::Pdlp : LpMethod::Simplex;
    if (meth != "auto" && meth != "simplex" && meth != "ipm" && meth != "pdlp")
        throw std::invalid_argument("unknown LP method '" + meth + "' (auto, simplex, ipm, pdlp, hpr)");
    const auto t0 = std::chrono::steady_clock::now();
    LpSolution s = solve_lp(q, so);
    r.status = s.status;
    r.iterations = s.iterations;
    r.method = meth == "auto" ? "simplex" : meth;
    r.device = "cpu";
    if (!s.x.empty() && s.status == Status::Optimal) {
        r.x = s.x;
        r.y = s.y;
        r.objective = sense * s.primal_objective;
        r.bound = sense * s.dual_objective;
    }
    if (meth == "auto" && s.status != Status::Optimal && s.status != Status::Infeasible &&
        s.status != Status::Unbounded) {
        const double used = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (o.time_limit - used > 1) {
            Result keep = r;
            hpr(o.time_limit - used);
            if (!r.has_solution()) r = keep;
            else r.method = "simplex+hpr";
        }
    }
    return r;
}

inline Result
solve_qp_model(const Model &m, const Options &o)
{
    Result r;
    const double sense = m.qp.lp.maximize ? -1.0 : 1.0;
    QpProblem q = m.qp;
    q.lp.is_integer.clear();
    QpOptions qo;
    const std::string &meth = o.method;
    if (meth == "auto") qo.method = QpMethod::Auto;
    else if (meth == "hprqp" || meth == "hpr") qo.method = QpMethod::HprQp;
    else if (meth == "pdhcg") qo.method = QpMethod::Pdhcg;
    else if (meth == "ipm") qo.method = QpMethod::Ipm;
    else throw std::invalid_argument("unknown QP method '" + meth + "' (auto, hprqp, pdhcg, ipm)");
    qo.auto_device = o.device == Device::Auto;
    qo.use_gpu = o.device == Device::Gpu || (qo.auto_device && qp_gpu_available());
    qo.tol = o.tol;
    qo.time_limit = o.time_limit;
    qo.verbose = o.verbose;
    qo.comm = o.comm;
    if (o.comm && o.comm->distributed() && qo.method != QpMethod::Auto && qo.method != QpMethod::HprQp)
        throw std::invalid_argument("over MPI ranks a QP is solved by HPR-QP (method auto or hprqp)");
    from_qp(solve_qp(q, qo), sense, r);
    return r;
}

inline Result
solve_milp_model(const Model &m, const Options &o)
{
    Result r;
    if (o.method != "auto" && o.method != "bnb")
        throw std::invalid_argument("unknown MILP method '" + o.method + "' (auto, bnb)");
    const LpProblem &p = m.qp.lp;
    const double sense = p.maximize ? -1.0 : 1.0;
    MilpOptions mo;
    mo.time_limit = o.time_limit;
    mo.gap_rel = o.gap;
    mo.verbose = o.verbose;
    mo.presolve = o.presolve;
    mo.cuts = o.cuts;
    mo.heuristics = o.heuristics;
#if defined(AXOS_ENABLE_CUDA)
    mo.gpu = o.device == Device::Gpu || (o.device == Device::Auto && qp_gpu_available());
#endif
    const MilpSolution s = solve_milp(p, mo);
    r.status = s.status;
    r.iterations = s.lp_iterations;
    r.nodes = s.nodes;
    r.method = "branch-and-cut";
    r.device = mo.gpu ? "cpu+gpu" : "cpu";
    r.message = s.notes;
    if (s.has_solution()) {
        r.x = s.x;
        r.objective = sense * s.objective;
    }
    if (std::isfinite(s.bound)) r.bound = sense * s.bound;
    return r;
}

} // namespace api_detail

// Solves the model; never throws for solver failures (status NumericalError
// and Result::message instead), only for invalid options.
inline Result
solve(const Model &m, const Options &o = Options())
{
    const auto t0 = std::chrono::steady_clock::now();
    ProblemType t = o.type == ProblemType::Auto ? m.type() : o.type;
    if (t == ProblemType::QP && !m.qp.has_quadratic()) t = ProblemType::LP;
    if (t == ProblemType::LP && m.qp.has_quadratic()) t = ProblemType::QP; // an LP of a QP is not defined
    if (t == ProblemType::MILP && m.qp.has_quadratic())
        throw std::invalid_argument("mixed-integer QP is not supported");
    const bool dist = o.comm && o.comm->distributed();
    if (t == ProblemType::MILP && dist)
        throw std::invalid_argument("a MILP runs on one process (start it without mpiexec)");
    Result r;
    try {
        if (t == ProblemType::MILP) r = api_detail::solve_milp_model(m, o);
        else if (t == ProblemType::QP) r = api_detail::solve_qp_model(m, o);
        else r = api_detail::solve_lp_model(m, o);
    } catch (const std::invalid_argument &) {
        throw;
    } catch (const std::exception &e) {
        if (dist) throw; // the other ranks may be waiting in a collective: the caller aborts
        r = Result();
        r.status = Status::NumericalError;
        r.message = e.what();
    }
    r.type = t;
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (r.has_solution() && r.x.size() == m.cols())
        r.max_violation = api_detail::max_violation(m.qp.lp, r.x, t == ProblemType::MILP);
    return r;
}

// The solution as "name value" lines (zero values omitted), after a header.
inline void
write_solution(std::ostream &out, const Model &m, const Result &r)
{
    out << "# AXOS " << kAxosVersion << " solution of " << m.name() << "\n";
    out << "# status " << to_string(r.status) << "\n";
    out << std::setprecision(17);
    if (!r.has_solution()) return;
    out << "# objective " << r.objective << "\n";
    const auto &names = m.qp.lp.col_names;
    for (size_t j = 0; j < r.x.size(); ++j)
        if (r.x[j] != 0) out << (j < names.size() ? names[j] : "x" + std::to_string(j)) << " " << r.x[j] << "\n";
}

namespace api_detail {
inline std::string
json_num(double v)
{
    if (!std::isfinite(v)) return "null";
    std::ostringstream s;
    s << std::setprecision(17) << v;
    return s.str();
}
inline std::string
json_str(const std::string &v)
{
    std::string o = "\"";
    for (char c : v) {
        if (c == '"' || c == '\\') o += '\\';
        if (c == '\n') { o += "\\n"; continue; }
        o += c;
    }
    return o + "\"";
}
} // namespace api_detail

// The result as one JSON object (with the solution when with_x).
inline std::string
result_json(const Model &m, const Result &r, bool with_x = true)
{
    using api_detail::json_num;
    using api_detail::json_str;
    std::ostringstream s;
    s << "{\"solver\":\"AXOS " << kAxosVersion << "\",\"model\":" << json_str(m.name())
      << ",\"type\":" << json_str(to_string(r.type)) << ",\"rows\":" << m.rows() << ",\"cols\":" << m.cols()
      << ",\"nnz\":" << m.nnz() << ",\"integers\":" << m.integers() << ",\"status\":"
      << json_str(to_string(r.status)) << ",\"objective\":" << json_num(r.objective) << ",\"bound\":"
      << json_num(r.bound) << ",\"gap\":" << json_num(r.gap()) << ",\"seconds\":" << json_num(r.seconds)
      << ",\"iterations\":" << r.iterations << ",\"nodes\":" << r.nodes << ",\"method\":" << json_str(r.method)
      << ",\"device\":" << json_str(r.device) << ",\"max_violation\":" << json_num(r.max_violation)
      << ",\"message\":" << json_str(r.message);
    if (with_x && r.has_solution()) {
        const auto &names = m.qp.lp.col_names;
        s << ",\"x\":{";
        bool first = true;
        for (size_t j = 0; j < r.x.size(); ++j) {
            if (r.x[j] == 0) continue;
            s << (first ? "" : ",") << json_str(j < names.size() ? names[j] : "x" + std::to_string(j)) << ":"
              << json_num(r.x[j]);
            first = false;
        }
        s << "}";
    }
    s << "}";
    return s.str();
}

} // namespace Solver
} // namespace AXOS

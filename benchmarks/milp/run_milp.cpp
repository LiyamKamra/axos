// SPDX-License-Identifier: BSD-3-Clause
//
// Runs the AXOS MILP solver on MPS files and prints one CSV line per solve:
//
//   run_milp <file.mps | directory> [options]
//     --time-limit 60       seconds per problem
//     --gap 1e-4            relative gap to stop at
//     --no-cuts --no-heuristics --no-presolve --no-propagation --no-probing
//     --gpu                 GPU probing, feasibility-jump walkers and batched
//                           PDHG strong branching (build with AXOS_ENABLE_CUDA)
//     --gpu-min-nnz N       batched strong branching from N nonzeros (10000)
//     --verbose 0|1|2
//     --out results.csv     also append the lines to this file
//     --only NAME[,NAME...] solve only these problems of a directory
//     --sol DIR             write each final solution to DIR/<name>.sol
//                           (one "name value" line per nonzero column)
//
// Objective and bound are in the problem's own sense. `feasible` is the
// check of the reported solution on the original problem (check_milp).
#include "solver/io/mps.h"
#include "solver/milp/solve_milp.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace AXOS::Solver;
namespace fs = std::filesystem;

static std::string
status_name(Status s)
{
    std::string t = to_string(s);
    std::replace(t.begin(), t.end(), ' ', '_');
    return t;
}

int
main(int argc, char **argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <file.mps|dir> [--time-limit s] [--gap g] [--no-cuts] "
                             "[--no-heuristics] [--no-presolve] [--no-propagation] [--verbose v] "
                             "[--out file.csv] [--only a,b] [--sol dir]\n", argv[0]);
        return 2;
    }
    MilpOptions opt;
    opt.time_limit = 60;
    std::string out_path, sol_dir;
    std::set<std::string> only;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + a);
            return argv[++i];
        };
        if (a == "--time-limit") opt.time_limit = std::stod(next());
        else if (a == "--gap") opt.gap_rel = std::stod(next());
        else if (a == "--no-cuts") opt.cuts = false;
        else if (a == "--no-heuristics") opt.heuristics = false;
        else if (a == "--no-presolve") opt.presolve = false;
        else if (a == "--no-propagation") opt.propagation = false;
        else if (a == "--no-probing") opt.probing = false;
        else if (a == "--gpu") opt.gpu = true;
        else if (a == "--gpu-min-nnz") opt.gpu_min_nnz = std::stol(next());
        else if (a == "--verbose") opt.verbose = std::stoi(next());
        else if (a == "--out") out_path = next();
        else if (a == "--sol") sol_dir = next();
        else if (a == "--only") {
            std::stringstream ss(next());
            std::string t;
            while (std::getline(ss, t, ',')) only.insert(t);
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    std::vector<fs::path> files;
    const fs::path target(argv[1]);
    if (fs::is_directory(target)) {
        for (auto &e : fs::directory_iterator(target)) {
            const std::string ext = e.path().extension().string();
            if (ext == ".mps" || ext == ".MPS")
                if (only.empty() || only.count(e.path().stem().string())) files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(target);
    }
    const std::string header =
        "problem,n,m,nnz,nint,solver,status,seconds,objective,bound,gap,nodes,lp_iterations,"
        "first_solution_seconds,root_bound,cuts,incumbent_source,feasible,read_seconds,notes";
    std::printf("%s\n", header.c_str());
    std::ofstream out;
    if (!out_path.empty()) {
        const bool fresh = !fs::exists(out_path);
        out.open(out_path, std::ios::app);
        if (fresh) out << header << "\n";
    }
    for (const auto &f : files) {
        const std::string name = f.stem().string();
        char line[2048];
        try {
            const auto t0 = std::chrono::steady_clock::now();
            const LpProblem p = read_mps_file(f.string());
            const double read_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            size_t nint = 0;
            for (auto v : p.is_integer) nint += v ? 1 : 0;
            const MilpSolution s = solve_milp(p, opt);
            const double sense = p.maximize ? -1.0 : 1.0;
            bool feas = false;
            if (s.has_solution()) feas = check_milp(p, s.x).ok(1e-5, 1e-6);
            std::string src = s.incumbent_source, notes = s.notes;
            std::replace(src.begin(), src.end(), ',', ';');
            std::replace(notes.begin(), notes.end(), ',', ';');
            std::snprintf(line, sizeof line,
                "%s,%zu,%zu,%zu,%zu,%s,%s,%.4f,%.12e,%.12e,%.3e,%ld,%ld,%.4f,%.12e,%d,%s,%d,%.3f,%s",
                name.c_str(), p.cols(), p.rows(), p.A.nnz(), nint, opt.gpu ? "axos-milp-gpu" : "axos-milp",
                status_name(s.status).c_str(),
                s.seconds, s.has_solution() ? sense * s.objective : std::nan(""),
                sense * s.bound, s.gap(), s.nodes, s.lp_iterations, s.first_solution_seconds,
                sense * s.root_bound, s.cuts, src.c_str(), feas ? 1 : 0, read_s, notes.c_str());
            if (!sol_dir.empty() && s.has_solution()) {
                fs::create_directories(sol_dir);
                std::ofstream so((fs::path(sol_dir) / (name + ".sol")).string());
                so.precision(17);
                for (size_t j = 0; j < s.x.size(); ++j)
                    if (s.x[j] != 0)
                        so << (j < p.col_names.size() ? p.col_names[j] : "x" + std::to_string(j)) << " "
                           << s.x[j] << "\n";
            }
        } catch (const std::exception &e) {
            std::string msg = e.what();
            std::replace(msg.begin(), msg.end(), ',', ';');
            std::replace(msg.begin(), msg.end(), '\n', ' ');
            std::snprintf(line, sizeof line, "%s,,,,,%s,error,,,,,,,,,,%s,0,,", name.c_str(),
                opt.gpu ? "axos-milp-gpu" : "axos-milp", msg.substr(0, 300).c_str());
        }
        std::printf("%s\n", line);
        std::fflush(stdout);
        if (out) {
            out << line << "\n";
            out.flush();
        }
    }
    return 0;
}

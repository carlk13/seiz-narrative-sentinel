#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace py = pybind11;

namespace seiz {

using State = std::array<double, 4>;
using Params = std::array<double, 6>;

inline void validate_state_and_params(
    const State& state,
    const Params& params,
    double dt,
    int max_steps,
    double t0
) {
    if (max_steps < 0) {
        throw std::invalid_argument("max_steps must be >= 0.");
    }
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("dt must be finite and >= 0.0.");
    }
    if (!std::isfinite(t0)) {
        throw std::invalid_argument("t0 must be finite.");
    }

    double total_n = 0.0;
    for (double val : state) {
        if (!std::isfinite(val)) {
            throw std::invalid_argument("All initial state compartments must be finite.");
        }
        total_n += val;
    }
    if (!std::isfinite(total_n) || total_n <= 0.0) {
        throw std::invalid_argument("Total population N = S + E + I + Z must be finite and > 0.");
    }

    for (double val : params) {
        if (!std::isfinite(val)) {
            throw std::invalid_argument("All SEIZ parameters must be finite.");
        }
    }
    const double beta = params[0];
    const double p = params[1];
    const double l = params[2];
    const double b = params[3];
    const double rho = params[4];
    const double epsilon = params[5];

    if (beta < 0.0 || b < 0.0 || rho < 0.0 || epsilon < 0.0) {
        throw std::invalid_argument("Rate parameters (beta, b, rho, epsilon) must be >= 0.0.");
    }
    if (p < 0.0 || p > 1.0 || l < 0.0 || l > 1.0) {
        throw std::invalid_argument("Probability parameters (p, l) must lie within [0.0, 1.0].");
    }
}

inline State compute_seiz_derivatives(const State& y, const Params& params) {
    const double S = y[0];
    const double E = y[1];
    const double I = y[2];
    const double Z = y[3];

    if (!std::isfinite(S) || !std::isfinite(E) || !std::isfinite(I) || !std::isfinite(Z)) {
        throw std::invalid_argument("State compartments must be finite.");
    }

    const double N = S + E + I + Z;
    if (!std::isfinite(N) || N <= 0.0) {
        throw std::invalid_argument("Total population N must be finite and strictly positive.");
    }

    const double beta = params[0];
    const double p = params[1];
    const double l = params[2];
    const double b = params[3];
    const double rho = params[4];
    const double epsilon = params[5];

    const double si_flow = beta * S * I / N;
    const double sz_flow = b * S * Z / N;
    const double ei_flow = rho * E * I / N;
    const double incubation_flow = epsilon * E;

    return State{
        -si_flow - sz_flow,
        (1.0 - p) * si_flow + (1.0 - l) * sz_flow - ei_flow - incubation_flow,
        p * si_flow + ei_flow + incubation_flow,
        l * sz_flow
    };
}

inline void rk4_integrate_into_buffers(
    const State& initial_state,
    const Params& params,
    double dt,
    int max_steps,
    double t0,
    double* t_ptr,
    double* s_ptr,
    double* e_ptr,
    double* i_ptr,
    double* z_ptr,
    double* n_ptr,
    double* y_flat_ptr
) {
    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    State y = initial_state;

    const double max_rate = std::max({params[0], params[3], params[4], params[5]});
    int sub_steps = 1;
    if (dt > 0.0 && max_rate > 0.0) {
        const double ratio = std::ceil((max_rate * dt) / 0.2);
        const double max_int = static_cast<double>(std::numeric_limits<int>::max());
        sub_steps = std::max(1, static_cast<int>(std::min(ratio, max_int)));
    }
    const double h = dt / static_cast<double>(sub_steps);
    const double half_h = 0.5 * h;
    const double sixth_h = h / 6.0;

    auto write_step = [&](std::size_t step_idx, double t_val, const State& state_val) {
        if (t_ptr != nullptr) {
            t_ptr[step_idx] = t_val;
        }
        if (s_ptr != nullptr) {
            s_ptr[step_idx] = state_val[0];
        }
        if (e_ptr != nullptr) {
            e_ptr[step_idx] = state_val[1];
        }
        if (i_ptr != nullptr) {
            i_ptr[step_idx] = state_val[2];
        }
        if (z_ptr != nullptr) {
            z_ptr[step_idx] = state_val[3];
        }
        if (n_ptr != nullptr) {
            n_ptr[step_idx] = state_val[0] + state_val[1] + state_val[2] + state_val[3];
        }
        if (y_flat_ptr != nullptr) {
            y_flat_ptr[0 * num_points + step_idx] = state_val[0];
            y_flat_ptr[1 * num_points + step_idx] = state_val[1];
            y_flat_ptr[2 * num_points + step_idx] = state_val[2];
            y_flat_ptr[3 * num_points + step_idx] = state_val[3];
        }
    };

    write_step(0, t0, y);

    for (int step = 1; step <= max_steps; ++step) {
        if (dt > 0.0) {
            for (int sub = 0; sub < sub_steps; ++sub) {
                const State k1 = compute_seiz_derivatives(y, params);

                const State y2{
                    y[0] + half_h * k1[0],
                    y[1] + half_h * k1[1],
                    y[2] + half_h * k1[2],
                    y[3] + half_h * k1[3]
                };
                const State k2 = compute_seiz_derivatives(y2, params);

                const State y3{
                    y[0] + half_h * k2[0],
                    y[1] + half_h * k2[1],
                    y[2] + half_h * k2[2],
                    y[3] + half_h * k2[3]
                };
                const State k3 = compute_seiz_derivatives(y3, params);

                const State y4{
                    y[0] + h * k3[0],
                    y[1] + h * k3[1],
                    y[2] + h * k3[2],
                    y[3] + h * k3[3]
                };
                const State k4 = compute_seiz_derivatives(y4, params);

                y[0] += sixth_h * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
                y[1] += sixth_h * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
                y[2] += sixth_h * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
                y[3] += sixth_h * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]);
            }
        }
        const double t_curr = t0 + static_cast<double>(step) * dt;
        write_step(static_cast<std::size_t>(step), t_curr, y);
    }
}

py::dict solve_rk4(
    const State& initial_state,
    const Params& params,
    double dt,
    int max_steps,
    double t0 = 0.0
) {
    validate_state_and_params(initial_state, params, dt, max_steps, t0);

    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    py::array_t<double> t_arr(num_points);
    py::array_t<double> s_arr(num_points);
    py::array_t<double> e_arr(num_points);
    py::array_t<double> i_arr(num_points);
    py::array_t<double> z_arr(num_points);
    py::array_t<double> n_arr(num_points);
    py::array_t<double> y_flat_arr(4 * num_points);

    double* t_ptr = t_arr.mutable_data();
    double* s_ptr = s_arr.mutable_data();
    double* e_ptr = e_arr.mutable_data();
    double* i_ptr = i_arr.mutable_data();
    double* z_ptr = z_arr.mutable_data();
    double* n_ptr = n_arr.mutable_data();
    double* y_flat_ptr = y_flat_arr.mutable_data();

    {
        py::gil_scoped_release release;
        rk4_integrate_into_buffers(
            initial_state,
            params,
            dt,
            max_steps,
            t0,
            t_ptr,
            s_ptr,
            e_ptr,
            i_ptr,
            z_ptr,
            n_ptr,
            y_flat_ptr
        );
    }

    py::dict result;
    result["t"] = std::move(t_arr);
    result["S"] = std::move(s_arr);
    result["E"] = std::move(e_arr);
    result["I"] = std::move(i_arr);
    result["Z"] = std::move(z_arr);
    result["N"] = std::move(n_arr);
    result["y_flat"] = std::move(y_flat_arr);
    return result;
}

py::dict solve_rk4_scalars(
    double s0,
    double e0,
    double i0,
    double z0,
    double beta,
    double p,
    double l,
    double b,
    double rho,
    double epsilon,
    double dt,
    int max_steps,
    double t0 = 0.0
) {
    return solve_rk4(
        State{s0, e0, i0, z0},
        Params{beta, p, l, b, rho, epsilon},
        dt,
        max_steps,
        t0
    );
}

py::array_t<double> solve_rk4_flat(
    const State& initial_state,
    const Params& params,
    double dt,
    int max_steps
) {
    validate_state_and_params(initial_state, params, dt, max_steps, 0.0);

    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    py::array_t<double> y_flat_arr(4 * num_points);
    double* y_flat_ptr = y_flat_arr.mutable_data();

    {
        py::gil_scoped_release release;
        rk4_integrate_into_buffers(
            initial_state,
            params,
            dt,
            max_steps,
            0.0,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            y_flat_ptr
        );
    }

    return y_flat_arr;
}

inline double linear_percentile(const std::vector<double>& sorted_vals, double q) {
    const std::size_t n = sorted_vals.size();
    if (n == 1) {
        return sorted_vals[0];
    }
    const double idx = q * static_cast<double>(n - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = std::min(lo + 1, n - 1);
    const double frac = idx - static_cast<double>(lo);
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

inline void gillespie_single_trajectory(
    long long s0,
    long long e0,
    long long i0,
    long long z0,
    const Params& params,
    double dt,
    int max_steps,
    double t0,
    std::uint64_t rng_seed,
    double* out_traj
) {
    long long S = s0;
    long long E = e0;
    long long I = i0;
    long long Z = z0;
    const double N = static_cast<double>(S + E + I + Z);

    const double beta = params[0];
    const double p = params[1];
    const double l = params[2];
    const double b = params[3];
    const double rho = params[4];
    const double epsilon = params[5];

    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    const double t_end = t0 + static_cast<double>(max_steps) * dt;

    std::mt19937_64 rng(rng_seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    auto record_at_step = [&](int step_idx) {
        const std::size_t idx = static_cast<std::size_t>(step_idx);
        out_traj[0 * num_points + idx] = static_cast<double>(S);
        out_traj[1 * num_points + idx] = static_cast<double>(E);
        out_traj[2 * num_points + idx] = static_cast<double>(I);
        out_traj[3 * num_points + idx] = static_cast<double>(Z);
    };

    record_at_step(0);
    int next_step = 1;
    double t = t0;

    while (next_step <= max_steps) {
        const double sd = static_cast<double>(S);
        const double ed = static_cast<double>(E);
        const double id = static_cast<double>(I);
        const double zd = static_cast<double>(Z);

        const double si_rate = beta * sd * id / N;
        const double sz_rate = b * sd * zd / N;
        const double ei_rate = rho * ed * id / N;
        const double inc_rate = epsilon * ed;

        const double a0 = p * si_rate;          // S -> I
        const double a1 = (1.0 - p) * si_rate;  // S -> E via I
        const double a2 = l * sz_rate;          // S -> Z
        const double a3 = (1.0 - l) * sz_rate;  // S -> E via Z
        const double a4 = ei_rate;              // E -> I via I
        const double a5 = inc_rate;             // E -> I spontaneous

        const double a_tot = a0 + a1 + a2 + a3 + a4 + a5;
        if (a_tot <= 0.0 || dt <= 0.0) {
            while (next_step <= max_steps) {
                record_at_step(next_step);
                ++next_step;
            }
            break;
        }

        double u1 = 1.0 - dist(rng);
        if (u1 <= 0.0) {
            u1 = std::numeric_limits<double>::min();
        }
        const double tau = -std::log(u1) / a_tot;
        const double t_next = t + tau;

        while (next_step <= max_steps) {
            const double t_grid = t0 + static_cast<double>(next_step) * dt;
            if (t_grid <= t_next) {
                record_at_step(next_step);
                ++next_step;
            } else {
                break;
            }
        }

        if (t_next > t_end || next_step > max_steps) {
            break;
        }

        t = t_next;
        const double r = dist(rng) * a_tot;
        double cum = a0;
        if (a0 > 0.0 && r < cum && S > 0) {
            --S;
            ++I;
        } else {
            cum += a1;
            if (a1 > 0.0 && r < cum && S > 0) {
                --S;
                ++E;
            } else {
                cum += a2;
                if (a2 > 0.0 && r < cum && S > 0) {
                    --S;
                    ++Z;
                } else {
                    cum += a3;
                    if (a3 > 0.0 && r < cum && S > 0) {
                        --S;
                        ++E;
                    } else {
                        cum += a4;
                        if (a4 > 0.0 && r < cum && E > 0) {
                            --E;
                            ++I;
                        } else if (E > 0) {
                            --E;
                            ++I;
                        }
                    }
                }
            }
        }
    }
}

inline void gillespie_ensemble_into_buffers(
    const State& initial_state,
    const Params& params,
    double dt,
    int max_steps,
    double t0,
    int num_simulations,
    std::uint64_t seed,
    double* t_ptr,
    double* n_ptr,
    double* p5_ptr,
    double* p50_ptr,
    double* p95_ptr
) {
    const long long s0 = std::llround(initial_state[0]);
    const long long e0 = std::llround(initial_state[1]);
    const long long i0 = std::llround(initial_state[2]);
    const long long z0 = std::llround(initial_state[3]);
    const double total_n = static_cast<double>(s0 + e0 + i0 + z0);

    const std::size_t n_sims = static_cast<std::size_t>(num_simulations);
    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    const std::size_t traj_stride = 4 * num_points;

    std::vector<double> all_trajectories(n_sims * traj_stride);

    const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t n_threads = std::min<std::size_t>(n_sims, static_cast<std::size_t>(hw_threads));

    if (n_threads <= 1) {
        for (std::size_t sim = 0; sim < n_sims; ++sim) {
            const std::uint64_t sim_seed = seed + static_cast<std::uint64_t>(sim) * 0x9E3779B97F4A7C15ULL;
            gillespie_single_trajectory(
                s0, e0, i0, z0, params, dt, max_steps, t0, sim_seed,
                all_trajectories.data() + sim * traj_stride
            );
        }
    } else {
        std::vector<std::thread> workers;
        workers.reserve(n_threads);
        const std::size_t chunk = (n_sims + n_threads - 1) / n_threads;

        for (std::size_t tid = 0; tid < n_threads; ++tid) {
            const std::size_t start_sim = tid * chunk;
            const std::size_t end_sim = std::min(n_sims, start_sim + chunk);
            if (start_sim >= end_sim) {
                break;
            }
            workers.emplace_back([&, start_sim, end_sim]() {
                for (std::size_t sim = start_sim; sim < end_sim; ++sim) {
                    const std::uint64_t sim_seed =
                        seed + static_cast<std::uint64_t>(sim) * 0x9E3779B97F4A7C15ULL;
                    gillespie_single_trajectory(
                        s0, e0, i0, z0, params, dt, max_steps, t0, sim_seed,
                        all_trajectories.data() + sim * traj_stride
                    );
                }
            });
        }
        for (auto& w : workers) {
            w.join();
        }
    }

    auto aggregate_steps = [&](std::size_t step_start, std::size_t step_end) {
        std::vector<double> vals(n_sims);
        for (std::size_t step = step_start; step < step_end; ++step) {
            if (t_ptr != nullptr) {
                t_ptr[step] = t0 + static_cast<double>(step) * dt;
            }
            if (n_ptr != nullptr) {
                n_ptr[step] = total_n;
            }
            for (std::size_t comp = 0; comp < 4; ++comp) {
                const std::size_t offset = comp * num_points + step;
                for (std::size_t sim = 0; sim < n_sims; ++sim) {
                    vals[sim] = all_trajectories[sim * traj_stride + offset];
                }
                std::sort(vals.begin(), vals.end());
                p5_ptr[offset] = linear_percentile(vals, 0.05);
                p50_ptr[offset] = linear_percentile(vals, 0.50);
                p95_ptr[offset] = linear_percentile(vals, 0.95);
            }
        }
    };

    const std::size_t agg_threads = std::min<std::size_t>(num_points, n_threads);
    if (agg_threads <= 1) {
        aggregate_steps(0, num_points);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(agg_threads);
        const std::size_t step_chunk = (num_points + agg_threads - 1) / agg_threads;

        for (std::size_t tid = 0; tid < agg_threads; ++tid) {
            const std::size_t step_start = tid * step_chunk;
            const std::size_t step_end = std::min(num_points, step_start + step_chunk);
            if (step_start >= step_end) {
                break;
            }
            workers.emplace_back([&, step_start, step_end]() {
                aggregate_steps(step_start, step_end);
            });
        }
        for (auto& w : workers) {
            w.join();
        }
    }
}

py::dict solve_gillespie(
    const State& initial_state,
    const Params& params,
    double dt,
    int max_steps,
    double t0 = 0.0,
    int num_simulations = 1000,
    std::uint64_t seed = 42
) {
    validate_state_and_params(initial_state, params, dt, max_steps, t0);
    for (double val : initial_state) {
        if (val < 0.0) {
            throw std::invalid_argument("Initial state compartments must be non-negative.");
        }
    }
    if (num_simulations < 1) {
        throw std::invalid_argument("num_simulations must be >= 1.");
    }

    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    const std::size_t flat_size = 4 * num_points;

    py::array_t<double> t_arr(num_points);
    py::array_t<double> n_arr(num_points);
    py::array_t<double> p5_arr(flat_size);
    py::array_t<double> p50_arr(flat_size);
    py::array_t<double> p95_arr(flat_size);

    double* t_ptr = t_arr.mutable_data();
    double* n_ptr = n_arr.mutable_data();
    double* p5_ptr = p5_arr.mutable_data();
    double* p50_ptr = p50_arr.mutable_data();
    double* p95_ptr = p95_arr.mutable_data();

    {
        py::gil_scoped_release release;
        gillespie_ensemble_into_buffers(
            initial_state,
            params,
            dt,
            max_steps,
            t0,
            num_simulations,
            seed,
            t_ptr,
            n_ptr,
            p5_ptr,
            p50_ptr,
            p95_ptr
        );
    }

    auto slice_comp = [num_points](const double* src, std::size_t comp) -> py::array_t<double> {
        py::array_t<double> out(num_points);
        std::copy_n(src + comp * num_points, num_points, out.mutable_data());
        return out;
    };

    py::dict result;
    result["t"] = std::move(t_arr);
    result["N"] = std::move(n_arr);

    result["S"] = slice_comp(p50_ptr, 0);
    result["E"] = slice_comp(p50_ptr, 1);
    result["I"] = slice_comp(p50_ptr, 2);
    result["Z"] = slice_comp(p50_ptr, 3);

    result["S_p5"] = slice_comp(p5_ptr, 0);
    result["S_p50"] = slice_comp(p50_ptr, 0);
    result["S_p95"] = slice_comp(p95_ptr, 0);

    result["E_p5"] = slice_comp(p5_ptr, 1);
    result["E_p50"] = slice_comp(p50_ptr, 1);
    result["E_p95"] = slice_comp(p95_ptr, 1);

    result["I_p5"] = slice_comp(p5_ptr, 2);
    result["I_p50"] = slice_comp(p50_ptr, 2);
    result["I_p95"] = slice_comp(p95_ptr, 2);

    result["Z_p5"] = slice_comp(p5_ptr, 3);
    result["Z_p50"] = slice_comp(p50_ptr, 3);
    result["Z_p95"] = slice_comp(p95_ptr, 3);

    result["p5"] = std::move(p5_arr);
    result["p50"] = p50_arr;
    result["y_flat"] = std::move(p50_arr);
    result["p95"] = std::move(p95_arr);
    return result;
}

py::tuple solve_gillespie_percentiles(
    const State& initial_state,
    const Params& params,
    double dt,
    int max_steps,
    double t0 = 0.0,
    int num_simulations = 1000,
    std::uint64_t seed = 42
) {
    validate_state_and_params(initial_state, params, dt, max_steps, t0);
    for (double val : initial_state) {
        if (val < 0.0) {
            throw std::invalid_argument("Initial state compartments must be non-negative.");
        }
    }
    if (num_simulations < 1) {
        throw std::invalid_argument("num_simulations must be >= 1.");
    }

    const std::size_t num_points = static_cast<std::size_t>(max_steps) + 1;
    const std::size_t flat_size = 4 * num_points;

    py::array_t<double> p5_arr(flat_size);
    py::array_t<double> p50_arr(flat_size);
    py::array_t<double> p95_arr(flat_size);

    double* p5_ptr = p5_arr.mutable_data();
    double* p50_ptr = p50_arr.mutable_data();
    double* p95_ptr = p95_arr.mutable_data();

    {
        py::gil_scoped_release release;
        gillespie_ensemble_into_buffers(
            initial_state,
            params,
            dt,
            max_steps,
            t0,
            num_simulations,
            seed,
            nullptr,
            nullptr,
            p5_ptr,
            p50_ptr,
            p95_ptr
        );
    }

    return py::make_tuple(std::move(p5_arr), std::move(p50_arr), std::move(p95_arr));
}

using BatchStateArray = Eigen::Array<double, Eigen::Dynamic, 4, Eigen::ColMajor>;
using BatchParamArray = Eigen::Array<double, Eigen::Dynamic, 6, Eigen::ColMajor>;

inline void compute_seiz_derivatives_batch(
    const Eigen::Ref<const BatchStateArray>& Y,
    const Eigen::Ref<const BatchParamArray>& P,
    Eigen::Ref<BatchStateArray> dY
) {
    const auto S = Y.col(0);
    const auto E = Y.col(1);
    const auto I = Y.col(2);
    const auto Z = Y.col(3);

    const auto inv_N = (S + E + I + Z).inverse();

    const auto beta = P.col(0);
    const auto p = P.col(1);
    const auto l = P.col(2);
    const auto b = P.col(3);
    const auto rho = P.col(4);
    const auto epsilon = P.col(5);

    const auto si_flow = beta * S * I * inv_N;
    const auto sz_flow = b * S * Z * inv_N;
    const auto ei_flow = rho * E * I * inv_N;
    const auto inc_flow = epsilon * E;

    dY.col(0) = -si_flow - sz_flow;
    dY.col(1) = (1.0 - p) * si_flow + (1.0 - l) * sz_flow - ei_flow - inc_flow;
    dY.col(2) = p * si_flow + ei_flow + inc_flow;
    dY.col(3) = l * sz_flow;
}

py::array_t<double> seiz_derivatives_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& y_batch,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& params_batch
) {
    if (y_batch.ndim() != 2 || y_batch.shape(1) != 4) {
        throw std::invalid_argument("y_batch must be a 2D array with shape (N, 4).");
    }
    if (params_batch.ndim() != 2 || params_batch.shape(1) != 6) {
        throw std::invalid_argument("params_batch must be a 2D array with shape (N, 6).");
    }
    const py::ssize_t n_rows = y_batch.shape(0);
    if (params_batch.shape(0) != n_rows) {
        throw std::invalid_argument("y_batch and params_batch must have matching batch size N.");
    }
    if (n_rows < 1) {
        throw std::invalid_argument("Batch size N must be >= 1.");
    }

    const double* y_ptr = y_batch.data();
    const double* p_ptr = params_batch.data();

    BatchStateArray Y(n_rows, 4);
    BatchParamArray P(n_rows, 6);
    for (py::ssize_t i = 0; i < n_rows; ++i) {
        double total_n = 0.0;
        for (int c = 0; c < 4; ++c) {
            const double val = y_ptr[i * 4 + c];
            if (!std::isfinite(val)) {
                throw std::invalid_argument("All entries in y_batch must be finite.");
            }
            Y(i, c) = val;
            total_n += val;
        }
        if (!std::isfinite(total_n) || total_n <= 0.0) {
            throw std::invalid_argument("Every row in y_batch must have total population N > 0.");
        }
        for (int c = 0; c < 6; ++c) {
            const double pval = p_ptr[i * 6 + c];
            if (!std::isfinite(pval)) {
                throw std::invalid_argument("All entries in params_batch must be finite.");
            }
            P(i, c) = pval;
        }
    }

    BatchStateArray dY(n_rows, 4);
    {
        py::gil_scoped_release release;
        compute_seiz_derivatives_batch(Y, P, dY);
    }

    py::array_t<double> out_arr({n_rows, static_cast<py::ssize_t>(4)});
    double* out_ptr = out_arr.mutable_data();
    for (py::ssize_t i = 0; i < n_rows; ++i) {
        for (int c = 0; c < 4; ++c) {
            out_ptr[i * 4 + c] = dY(i, c);
        }
    }
    return out_arr;
}

py::array_t<double> solve_rk4_batch(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& initial_state,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& params_batch,
    double dt,
    int max_steps,
    double t0 = 0.0
) {
    if (max_steps < 0) {
        throw std::invalid_argument("max_steps must be >= 0.");
    }
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("dt must be finite and >= 0.0.");
    }
    if (!std::isfinite(t0)) {
        throw std::invalid_argument("t0 must be finite.");
    }
    if (params_batch.ndim() != 2 || params_batch.shape(1) != 6) {
        throw std::invalid_argument("params_batch must be a 2D array with shape (N, 6).");
    }
    const py::ssize_t batch_size = params_batch.shape(0);
    if (batch_size < 1) {
        throw std::invalid_argument("Batch size N in params_batch must be >= 1.");
    }

    const bool broadcast_state = (initial_state.ndim() == 1 && initial_state.shape(0) == 4);
    const bool batched_state = (initial_state.ndim() == 2 && initial_state.shape(0) == batch_size && initial_state.shape(1) == 4);
    if (!broadcast_state && !batched_state) {
        throw std::invalid_argument(
            "initial_state must have shape (4,) or (N, 4) matching params_batch."
        );
    }

    const double* init_ptr = initial_state.data();
    const double* param_ptr = params_batch.data();

    if (broadcast_state) {
        double total_n = 0.0;
        for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(init_ptr[c])) {
                throw std::invalid_argument("Initial state values must be finite.");
            }
            total_n += init_ptr[c];
        }
        if (!std::isfinite(total_n) || total_n <= 0.0) {
            throw std::invalid_argument("Total population N must be finite and > 0.");
        }
    } else {
        for (py::ssize_t i = 0; i < batch_size; ++i) {
            double total_n = 0.0;
            for (int c = 0; c < 4; ++c) {
                if (!std::isfinite(init_ptr[i * 4 + c])) {
                    throw std::invalid_argument("Initial state values must be finite.");
                }
                total_n += init_ptr[i * 4 + c];
            }
            if (!std::isfinite(total_n) || total_n <= 0.0) {
                throw std::invalid_argument("Every initial state row must have N > 0.");
            }
        }
    }

    for (py::ssize_t i = 0; i < batch_size; ++i) {
        const double beta = param_ptr[i * 6 + 0];
        const double p = param_ptr[i * 6 + 1];
        const double l = param_ptr[i * 6 + 2];
        const double b = param_ptr[i * 6 + 3];
        const double rho = param_ptr[i * 6 + 4];
        const double epsilon = param_ptr[i * 6 + 5];
        if (!std::isfinite(beta) || !std::isfinite(p) || !std::isfinite(l) ||
            !std::isfinite(b) || !std::isfinite(rho) || !std::isfinite(epsilon)) {
            throw std::invalid_argument("All parameter entries in params_batch must be finite.");
        }
        if (beta < 0.0 || b < 0.0 || rho < 0.0 || epsilon < 0.0) {
            throw std::invalid_argument("Rate parameters (beta, b, rho, epsilon) must be >= 0.0.");
        }
        if (p < 0.0 || p > 1.0 || l < 0.0 || l > 1.0) {
            throw std::invalid_argument("Probability parameters (p, l) must lie within [0.0, 1.0].");
        }
    }

    const py::ssize_t num_points = static_cast<py::ssize_t>(max_steps) + 1;
    py::array_t<double> out_arr({batch_size, num_points, static_cast<py::ssize_t>(4)});
    double* out_ptr = out_arr.mutable_data();

    {
        py::gil_scoped_release release;

        auto run_row_range = [&](py::ssize_t row_start, py::ssize_t row_end) {
            constexpr py::ssize_t kTileSize = 256;
            BatchStateArray Y(kTileSize, 4);
            BatchStateArray k1(kTileSize, 4);
            BatchStateArray k2(kTileSize, 4);
            BatchStateArray k3(kTileSize, 4);
            BatchStateArray k4(kTileSize, 4);
            BatchStateArray y_tmp(kTileSize, 4);
            BatchParamArray P_tile(kTileSize, 6);

            for (py::ssize_t tile_start = row_start; tile_start < row_end; tile_start += kTileSize) {
                const py::ssize_t tile_len = std::min(kTileSize, row_end - tile_start);
                auto Y_view = Y.topRows(tile_len);
                auto P_view = P_tile.topRows(tile_len);
                auto k1_view = k1.topRows(tile_len);
                auto k2_view = k2.topRows(tile_len);
                auto k3_view = k3.topRows(tile_len);
                auto k4_view = k4.topRows(tile_len);
                auto tmp_view = y_tmp.topRows(tile_len);

                double tile_max_rate = 0.0;
                for (py::ssize_t r = 0; r < tile_len; ++r) {
                    const py::ssize_t g_row = tile_start + r;
                    const double* src_y = broadcast_state ? init_ptr : (init_ptr + g_row * 4);
                    const double* src_p = param_ptr + g_row * 6;
                    for (int c = 0; c < 4; ++c) {
                        Y_view(r, c) = src_y[c];
                        out_ptr[(g_row * num_points) * 4 + c] = src_y[c];
                    }
                    for (int c = 0; c < 6; ++c) {
                        P_view(r, c) = src_p[c];
                    }
                    tile_max_rate = std::max({tile_max_rate, src_p[0], src_p[3], src_p[4], src_p[5]});
                }

                int sub_steps = 1;
                if (dt > 0.0 && tile_max_rate > 0.0) {
                    const double ratio = std::ceil((tile_max_rate * dt) / 0.2);
                    const double max_int = static_cast<double>(std::numeric_limits<int>::max());
                    sub_steps = std::max(1, static_cast<int>(std::min(ratio, max_int)));
                }
                const double h = dt / static_cast<double>(sub_steps);
                const double half_h = 0.5 * h;
                const double sixth_h = h / 6.0;

                for (int step = 1; step <= max_steps; ++step) {
                    if (dt > 0.0) {
                        for (int sub = 0; sub < sub_steps; ++sub) {
                            compute_seiz_derivatives_batch(Y_view, P_view, k1_view);

                            tmp_view = Y_view + half_h * k1_view;
                            compute_seiz_derivatives_batch(tmp_view, P_view, k2_view);

                            tmp_view = Y_view + half_h * k2_view;
                            compute_seiz_derivatives_batch(tmp_view, P_view, k3_view);

                            tmp_view = Y_view + h * k3_view;
                            compute_seiz_derivatives_batch(tmp_view, P_view, k4_view);

                            Y_view += sixth_h * (k1_view + 2.0 * k2_view + 2.0 * k3_view + k4_view);
                        }
                    }

                    const py::ssize_t step_offset = static_cast<py::ssize_t>(step);
                    for (py::ssize_t r = 0; r < tile_len; ++r) {
                        const py::ssize_t g_row = tile_start + r;
                        double* dst = out_ptr + (g_row * num_points + step_offset) * 4;
                        dst[0] = Y_view(r, 0);
                        dst[1] = Y_view(r, 1);
                        dst[2] = Y_view(r, 2);
                        dst[3] = Y_view(r, 3);
                    }
                }
            }
        };

        const unsigned int hw_threads = std::max(1u, std::thread::hardware_concurrency());
        const py::ssize_t max_threads = (batch_size < 32)
            ? static_cast<py::ssize_t>(1)
            : std::min<py::ssize_t>(batch_size, static_cast<py::ssize_t>(hw_threads));

        if (max_threads <= 1) {
            run_row_range(0, batch_size);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(max_threads));
            const py::ssize_t chunk = (batch_size + max_threads - 1) / max_threads;

            for (py::ssize_t tid = 0; tid < max_threads; ++tid) {
                const py::ssize_t r_start = tid * chunk;
                const py::ssize_t r_end = std::min(batch_size, r_start + chunk);
                if (r_start >= r_end) {
                    break;
                }
                workers.emplace_back([&, r_start, r_end]() {
                    run_row_range(r_start, r_end);
                });
            }
            for (auto& w : workers) {
                w.join();
            }
        }
    }

    return out_arr;
}

}  // namespace seiz

PYBIND11_MODULE(seiz_cpp, m) {
    m.doc() = "High-performance C++17 RK4 solver for the SEIZ disinformation ODE model.";

    m.def(
        "seiz_derivatives",
        &seiz::compute_seiz_derivatives,
        py::arg("y"),
        py::arg("params"),
        "Compute normalized SEIZ derivatives [dS/dt, dE/dt, dI/dt, dZ/dt]."
    );

    m.def(
        "solve_rk4",
        &seiz::solve_rk4,
        py::arg("initial_state"),
        py::arg("params"),
        py::arg("dt"),
        py::arg("max_steps"),
        py::arg("t0") = 0.0,
        "Integrate the SEIZ ODE system using native 4th-order Runge-Kutta (RK4)."
    );

    m.def(
        "solve_rk4",
        &seiz::solve_rk4_scalars,
        py::arg("s0"),
        py::arg("e0"),
        py::arg("i0"),
        py::arg("z0"),
        py::arg("beta"),
        py::arg("p"),
        py::arg("l"),
        py::arg("b"),
        py::arg("rho"),
        py::arg("epsilon"),
        py::arg("dt"),
        py::arg("max_steps"),
        py::arg("t0") = 0.0,
        "Integrate the SEIZ ODE system from scalar initial state and parameter arguments."
    );

    m.def(
        "solve_rk4_flat",
        &seiz::solve_rk4_flat,
        py::arg("initial_state"),
        py::arg("params"),
        py::arg("dt"),
        py::arg("max_steps"),
        "Integrate the SEIZ ODE system and return a single flattened 1D array of shape (4 * (max_steps + 1),)."
    );

    m.def(
        "solve_gillespie",
        &seiz::solve_gillespie,
        py::arg("initial_state"),
        py::arg("params"),
        py::arg("dt"),
        py::arg("max_steps"),
        py::arg("t0") = 0.0,
        py::arg("num_simulations") = 1000,
        py::arg("seed") = 42,
        "Run parallel stochastic Gillespie SEIZ simulations and return 5th, 50th, and 95th percentile arrays."
    );

    m.def(
        "solve_gillespie_percentiles",
        &seiz::solve_gillespie_percentiles,
        py::arg("initial_state"),
        py::arg("params"),
        py::arg("dt"),
        py::arg("max_steps"),
        py::arg("t0") = 0.0,
        py::arg("num_simulations") = 1000,
        py::arg("seed") = 42,
        "Run parallel stochastic Gillespie SEIZ simulations and return (p5, p50, p95) flattened arrays."
    );

    m.def(
        "seiz_derivatives_batch",
        &seiz::seiz_derivatives_batch,
        py::arg("y_batch"),
        py::arg("params_batch"),
        "Compute vectorized Eigen SEIZ derivatives for a batch of [N, 4] states and [N, 6] parameters."
    );

    m.def(
        "solve_rk4_batch",
        &seiz::solve_rk4_batch,
        py::arg("initial_state"),
        py::arg("params_batch"),
        py::arg("dt"),
        py::arg("max_steps"),
        py::arg("t0") = 0.0,
        "Integrate a batch of [N, 6] parameter sets using Eigen-vectorized RK4, returning [N, time_steps, 4]."
    );

    m.def(
        "solve_rk4",
        &seiz::solve_rk4_batch,
        py::arg("initial_state"),
        py::arg("params_batch"),
        py::arg("dt"),
        py::arg("max_steps"),
        py::arg("t0") = 0.0,
        "Batched Eigen-vectorized RK4 solver accepting a 2D [N, 6] parameter array."
    );
}

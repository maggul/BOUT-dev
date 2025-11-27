/**************************************************************************
 * SUNDIALS Temporal Filtering 
 * NOTE: This is currently in beta testing so use with cautious optimism
 *
 **************************************************************************
 * Copyright 2010-2025 BOUT++ contributors
 *
 * Contact: Ben Dudson, dudson2@llnl.gov
 *
 * This file is part of BOUT++.
 *
 * BOUT++ is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * BOUT++ is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with BOUT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 **************************************************************************/

#ifndef BOUT_TEMPORAL_FILTERING_H
#define BOUT_TEMPORAL_FILTERING_H

#include "bout/build_defines.hxx"
#include "bout/sundials_backports.hxx"
#include "bout/bout_enum_class.hxx"
#include "bout/bout_types.hxx"
#include "bout/region.hxx"
#include <nvector/nvector_parallel.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_config.h>


// Type of averaging to perform 
// EMA = Exponential Moving Average
// SRA = Simple Running Average
BOUT_ENUM_CLASS(FilteringType, None, EMA, SRA);

class TemporalFiltering {
public:
  TemporalFiltering() = default;

  void initialize(N_Vector uvec) {
    ASSERT1(uvec != nullptr);
    if (!uvecmean) {
      uvecmean = N_VClone(uvec);
    }

    // Initial mean = initial state
    N_VScale(1.0, uvec, uvecmean);
    reset_state();
  }

  void configure(FilteringType type,
                 BoutReal tau_mean_in,
                 BoutReal start_time_in) {
    average_type    = type;
    tau_mean        = tau_mean_in;
    mean_start_time = start_time_in;

    if (average_type != FilteringType::None && tau_mean <= 0.0) {
      throw BoutException("TemporalFiltering: tau_mean must be > 0 when filtering is enabled\n");
    }
  }

  void reset_state() {
    prev_time    = 0.0;
    tau_mean_cur = 0.0;
    started      = false;
  }

  /// Update the internal mean based on current time and solution
  void update(BoutReal time, N_Vector uvec) {
    if (average_type == FilteringType::None || uvecmean == nullptr) {
      return;
    }

    // Not yet time to start averaging
    if (!started && (time + 1.0e-14 < mean_start_time)) {
      return;
    }

    // First step of this averaging
    if (!started) {
      N_VScale(1.0, uvec, uvecmean);
      prev_time    = time;
      tau_mean_cur = 0.0;
      started      = true;
      return;
    }

    BoutReal dt = time - prev_time;
    if (dt <= 1.0e-14) {
      // just ignore if this ever happens
      return;
    }

    tau_mean_cur += dt;

    if (average_type == FilteringType::EMA) {
      // Exponential moving average with timescale tau_mean
      // alpha = dt / (tau_mean + dt)    (stable for dt << tau_mean)
      // alpha = 1 - exp(-dt / tau_mean) (exact integration) -- not yet implemented
      BoutReal alpha;
      if(100.0*dt > tau_mean)
        alpha = -expm1(-dt / tau_mean);
      else
        alpha = dt / (tau_mean + dt);

      N_VLinearSum(1.0 - alpha, uvecmean, alpha, uvec, uvecmean);
    } else if (average_type == FilteringType::SRA) {
      // Simple running average over this window:
      // u_mean_new = (T_old * u_mean_old + dt * u) / (T_old + dt)
      BoutReal Told = tau_mean_cur - dt;
      if (Told <= 0.0) {
        // First effective dt in this window: mean = current state
        N_VScale(1.0, uvec, uvecmean);
      } else {
        N_VLinearSum(Told / tau_mean_cur, uvecmean,
                     dt   / tau_mean_cur, uvec,     uvecmean);
      }
    }

    prev_time = time;
  }

  /// Do we have a usable mean?
  bool has_mean() const {
    return started && (uvecmean != nullptr);
  }

  /// For SRA: has this window reached at least tau_mean in duration?
  bool window_full() const {
    return (average_type == FilteringType::SRA) &&
           (started && tau_mean_cur >= tau_mean);
  }

  /// Restart the averaging window (used only for SRA finite-memory behavior)
  void restart_window(BoutReal current_time) {
    started       = false;
    tau_mean_cur  = 0.0;
    mean_start_time = current_time; // next window starts from "now"
  }

  N_Vector get_mean_vector() const { return uvecmean; }

  ~TemporalFiltering() {
    if (uvecmean) {
      N_VDestroy(uvecmean);
    }
  }

private:
  FilteringType average_type{FilteringType::None};
  N_Vector uvecmean{nullptr};

  BoutReal tau_mean{0.0};
  BoutReal mean_start_time{0.0};
  BoutReal prev_time{0.0};

  /// Time accumulated in the current averaging window
  BoutReal tau_mean_cur{0.0};

  /// Has this averaging window been started at all?
  bool started{false};
};

#endif // BOUT_TEMPORAL_FILTERING_H
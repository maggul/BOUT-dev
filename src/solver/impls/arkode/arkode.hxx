/**************************************************************************
 * Interface to ARKODE solver
 * NOTE: ARKode is currently in beta testing so use with cautious optimism
 *
 * NOTE: Only one solver can currently be compiled in
 *
 **************************************************************************
 * Copyright 2010-2024 BOUT++ contributors
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

#ifndef BOUT_ARKODE_SOLVER_H
#define BOUT_ARKODE_SOLVER_H

#include "bout/build_defines.hxx"
#include "bout/solver.hxx"

#if not BOUT_HAS_ARKODE

namespace {
RegisterUnavailableSolver
    registerunavailablearkode("arkode", "BOUT++ was not configured with ARKODE/SUNDIALS");
}

#else

#include "bout/bout_enum_class.hxx"
#include "bout/bout_types.hxx"
#include "bout/region.hxx"
#include "bout/sundials_backports.hxx"
#include "temporal_filtering.hxx"

#include <nvector/nvector_parallel.h>
#include <sundials/sundials_config.h>

#define ARKODE_CONTROLLER_SUPPORT SUNDIALS_VERSION_AT_LEAST(6, 7, 0)
#define ARKODE_TABLE_BY_NAME_SUPPORT SUNDIALS_VERSION_AT_LEAST(6, 4, 0)
// ARKStepSetOptimalParams is deprecated since SUNDIALS 6.1.0
#define ARKODE_OPTIMAL_PARAMS_SUPPORT SUNDIALS_VERSION_LESS_THAN(7, 1, 0)

#if ARKODE_CONTROLLER_SUPPORT
#include <sundials/sundials_adaptcontroller.h> // IWYU pragma: export
#endif

#include <string>
#include <vector>

class ArkodeSolver;
class Options;

namespace {
RegisterSolver<ArkodeSolver> registersolverarkode("arkode");
}

// enum describing treatment of equations
// Note: Capitalized because `explicit` is a C++ reserved keyword
BOUT_ENUM_CLASS(Treatment, ImEx, Implicit, Explicit);

// Adaptivity method
BOUT_ENUM_CLASS(AdapMethod, PID, PI, I, Explicit_Gustafsson, Implicit_Gustafsson,
                ImEx_Gustafsson);

// Shim for the ARKstep -> ARKode prefix change in SUNDIALS 7.1.0
#if SUNDIALS_VERSION_LESS_THAN(7, 1, 0)
#include <arkode/arkode_arkstep.h>
static constexpr auto ARKodeFree = ARKStepFree;
static constexpr auto ARKodeSetUserData = ARKStepSetUserData;
static constexpr auto ARKodeSetLinear = ARKStepSetLinear;
static constexpr auto ARKodeSetFixedStep = ARKStepSetFixedStep;
static constexpr auto ARKodeSetOrder = ARKStepSetOrder;
static constexpr auto ARKodeSetCFLFraction = ARKStepSetCFLFraction;
#if ARKODE_CONTROLLER_SUPPORT
static constexpr auto ARKodeSetAdaptController = ARKStepSetAdaptController;
static constexpr auto ARKodeSetAdaptivityAdjustment = ARKStepSetAdaptivityAdjustment;
#endif
static constexpr auto ARKodeSVtolerances = ARKStepSVtolerances;
static constexpr auto ARKodeSStolerances = ARKStepSStolerances;
static constexpr auto ARKodeSetMaxNumSteps = ARKStepSetMaxNumSteps;
static constexpr auto ARKodeSetMaxStep = ARKStepSetMaxStep;
static constexpr auto ARKodeSetMinStep = ARKStepSetMinStep;
static constexpr auto ARKodeSetInitStep = ARKStepSetInitStep;
static constexpr auto ARKodeSetNonlinearSolver = ARKStepSetNonlinearSolver;
static constexpr auto ARKodeSetLinearSolver = ARKStepSetLinearSolver;
static constexpr auto ARKodeSetPreconditioner = ARKStepSetPreconditioner;
static constexpr auto ARKodeSetJacTimes = ARKStepSetJacTimes;
static constexpr auto ARKodeGetNumSteps = ARKStepGetNumSteps;
static constexpr auto ARKodeGetNumNonlinSolvIters = ARKStepGetNumNonlinSolvIters;
static constexpr auto ARKodeGetNumPrecEvals = ARKStepGetNumPrecEvals;
static constexpr auto ARKodeGetNumLinIters = ARKStepGetNumLinIters;
static constexpr auto ARKodeEvolve = ARKStepEvolve;
static constexpr auto ARKodeGetCurrentTime = ARKStepGetCurrentTime;
static constexpr auto ARKodeGetDky = ARKStepGetDky;
static constexpr auto ARKodeGetLastStep = ARKStepGetLastStep;
#endif

class ArkodeSolver : public Solver {
public:
  explicit ArkodeSolver(Options* opts = nullptr);
  ~ArkodeSolver() override;

  BoutReal getCurrentTimestep() override { return hcur; }

  int init() override;

  int run() override;
  BoutReal run(BoutReal tout);

  // These functions used internally (but need to be public)
  void rhs_e(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void rhs_i(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void rhs(BoutReal t, BoutReal* udata, BoutReal* dudata);
  void pre(BoutReal t, BoutReal gamma, BoutReal delta, BoutReal* udata, BoutReal* rvec,
           BoutReal* zvec);
  void jac(BoutReal t, BoutReal* ydata, BoutReal* vdata, BoutReal* Jvdata);

private:
  BoutReal hcur; //< Current internal timestep

  bool diagnose{false}; //< Output additional diagnostics

  N_Vector uvec{nullptr};    //< Values
  void* arkode_mem{nullptr}; //< ARKODE internal memory block

  BoutReal pre_Wtime{0.0}; //< Time in preconditioner
  int pre_ncalls{0};       //< Number of calls to preconditioner

  /// Maximum number of steps to take between outputs
  int mxsteps;
  /// Integrator treatment enum: IMEX, Implicit or Explicit
  Treatment treatment;
  /// Use linear implicit solver (only evaluates jacobian inversion once)
  bool set_linear;
  /// Solve explicit portion in fixed timestep mode. NOTE: This is not recommended except
  /// for code comparison
  bool fixed_step;
  /// Order of internal step
  int order;
  /// Name of the implicit Butcher table
  std::string implicit_table;
  /// Name of the explicit Butcher table
  std::string explicit_table;
  /// Fraction of the estimated explicitly stable step to use
  BoutReal cfl_frac;
  /// Timestep adaptivity function
  AdapMethod adap_method;
  /// Threshold for “multiple” successive error failures
  int small_nef;
  /// Maximum step size growth factor upon multiple successive failures
  BoutReal etamxf;
  /// Minimum value of step size growth factor upon a failure
  BoutReal eta_min;
  /// Maximum allowed growth factor in step size between consecutive steps
  BoutReal mx_growth;
  /// Bias factor to slightly exaggerate the temporal error
  BoutReal error_bias;
  /// Absolute tolerance
  BoutReal abstol;
  /// Relative tolerance
  BoutReal reltol;
  /// Use separate absolute tolerance for each field
  bool use_vector_abstol;
  /// Maximum timestep (only used if greater than zero)
  BoutReal max_timestep;
  /// Minimum timestep (only used if greater than zero)
  BoutReal min_timestep;
  /// Initial timestep (only used if greater than zero)
  BoutReal start_timestep;
  /// Use accelerated fixed point solver instead of Newton iterative
  bool fixed_point;
  /// Use user-supplied preconditioner function
  bool use_precon;
  /// Number of Krylov basis vectors to use
  int maxl;
  /// Use right preconditioning instead of left preconditioning
  bool rightprec;
  /// Use user-supplied Jacobian function
  bool use_jacobian;
  /// ARKodePrintAllStats: print all available integrator statistics at the end of the run
  bool print_allstats;
  /// Temporal filtering of solution
  /// - Whether to use temporal filtering
  bool use_temporal_filtering{false};
  /// Temporal filtering object
  TemporalFiltering temp_filtering;
  /// Type of temporal filtering to use
  FilteringType filtering_type{FilteringType::None};
  /// Interval over which means are calculated
  /// - EMA: time scale of the exponential memory
  /// - SRA: window length of each running-average block
  BoutReal tau_mean{0.0};
  /// Time at which averaging starts
  BoutReal mean_start_time{0.0};
  /// Relaxation/nudging parameter for temporal filtering
  BoutReal lambda{0.0};

#if ARKODE_OPTIMAL_PARAMS_SUPPORT
  /// Use ARKode optimal parameters
  bool optimize;
#endif

  // Diagnostics from ARKODE
  int nsteps{0};
  int nfe_evals{0};
  int nfi_evals{0};
  int nniters{0};
  int npevals{0};
  int nliters{0};

  void set_abstol_values(BoutReal* abstolvec_data, std::vector<BoutReal>& f2dtols,
                         std::vector<BoutReal>& f3dtols);
  void loop_abstol_values_op(Ind2D i2d, BoutReal* abstolvec_data, int& p,
                             std::vector<BoutReal>& f2dtols,
                             std::vector<BoutReal>& f3dtols, bool bndry);
  void apply_temporal_filtering(BoutReal internal_time, N_Vector uvec);
  /// SPGMR solver structure
  SUNLinearSolver sun_solver{nullptr};
  /// Solver for implicit stages
  SUNNonlinearSolver nonlinear_solver{nullptr};
#if ARKODE_CONTROLLER_SUPPORT
  /// Timestep controller
  SUNAdaptController controller{nullptr};
#endif
  /// Context for SUNDIALS memory allocations
  sundials::Context suncontext;
};

#endif // BOUT_HAS_ARKODE
#endif // BOUT_ARKODE_SOLVER_H

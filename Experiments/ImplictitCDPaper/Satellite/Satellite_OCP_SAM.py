import logging
from timeit import default_timer as timer
from typing import List, Tuple

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from casadi.tools import struct_symSX, entry, struct_SX

from Core.averagedynamicsapproximations import CentralDifferences, ForwardDifferences, LIPS2
from Core.configuration import IPOPT_OPTIONS_LINEAR_SOLVER, PLT_GRID_ALPHA
from Core.controlparameterization import FourierControlParameterization, PiecewiseConstantControlParametrization, \
    ZeroControlParameterization
from Core.CycleImplementations.FixedPeriodShooting import CycleSingleShootingFixedPeriod, \
    CycleMultipleShootingFixedPeriod, CycleCollocationFixedPeriod
from Core.integrators import RK4, OthorgonalCollocation, LegendreCollocation, RadauCollocation, Lobatto3A_Order2, \
    Lobatto3A_Order4
from Core.polynomials import LagrangePoly
from Core.macrodiscretizationschemes import MacroDiscretizationScheme
from Core.parameters import Parameters, SymbolicParameter, FreeSymbolicParameter, FixedSymbolicParameter
from Core.tools import NumpyStruct, smoothStep_tanh_f
from Core.figuretools import latexify
from Models.satellite2d import Satellite2D

latexify()

# %% Model and initial position

# epsilon and horizon
epsilon_bar = 0.005
L = 1 / 5  # final time


params = Parameters()
params.Nmicro = 30  # number of integration intervals per cycle
params.dmacro = 5  # number of collocation points for the macro discretization
params.dmicro = 4  # number of collocation points for the micro discretization
params.dControl = 3  # degree of the control polynomial
params.N_har = 1

# symbolic parameters
params.declareSyms([FreeSymbolicParameter('U_cycles', [ca.DM.zeros((Satellite2D.nu, params.N_har + 2)) for i in
                                                   range(params.dControl)],
                                      shape=(Satellite2D.nu, params.N_har + 2), repeat=params.dControl),
                    FixedSymbolicParameter('epsilon', 0)])

params.tauf = ca.floor(L / ca.norm_1(params.epsilon))

# model
model = Satellite2D(params)

ADA_TYPE = 'CD2'

# initial position
x0bar = model.x_struct(0)
x0bar["p"] = [1, 0]
x0bar["v"] = [0, 1 + 0.05]
x0bar = x0bar.cat

# target endpoint (on a circular orbit
xendbar = model.x_struct(0)
xendbar["p"] = [1.5, 0]
xendbar["v"] = [0, np.sqrt(1 / xendbar["p", 0])]
xendbar = xendbar.cat

# point used for initialization of the cycles
xinit = model.x_struct(0)
px_init = 1.15
# px_init = {'CD2': 1.1, 'CD3': 1.3, 'FD': 1.25}[ADA_TYPE]  # THIS IS STUPID, but otherwise IPOPT doesnt converge
xinit["p"] = [px_init, 0]
xinit["v"] = [0, np.sqrt(1 / xinit["p", 0])]

# cost function: minimize the control effort
minimize_controls_f = ca.Function('minimize_controls_f', [model.x, model.tau, model.u] + model.symbolicParams,
                                  [model.u_struct['thrust'].T @ model.u_struct['thrust']])

# terminal condition to be on the target constant orbit
radius_target = ca.sqrt(xendbar[0:2].T @ xendbar[0:2])
E_f = ca.Function('E_f', [model.x], [ca.vertcat(model.x_struct['p'].T @ model.x_struct['p'] - radius_target ** 2,
                                                # model.x_struct['p'].T @ model.x_struct['v'],
                                                model.ecc_vec(model.x)
                                                )])

# %% define the average ocp

# collocation scheme for controls, with points at chebyshev nodes
controlPoly = LagrangePoly(params.dControl, 'legendre')
controlParameterization = FourierControlParameterization(model.nu, controlPoly, parameters=params, N_har=params.N_har)

# initialization Function
_r_init = ca.norm_2(model.x_struct['p'])
x_init = ca.vertcat(_r_init * ca.cos(2 * ca.pi * model.tau),
                    _r_init * ca.sin(2 * ca.pi * model.tau),
                    -ca.sqrt(1 / _r_init) * ca.sin(2 * ca.pi * model.tau),
                    ca.sqrt(1 / _r_init) * ca.cos(2 * ca.pi * model.tau),
                    )
xinit_f = ca.Function('xinit_f', [model.x, model.tau], [x_init])

microIntegrator = RadauCollocation(params.dmicro)
cycleDiscretizationClass = CycleCollocationFixedPeriod(model=model,
                                                       fixedPeriod=1,
                                                       Nint=params.Nmicro,
                                                       stageCostFunction=minimize_controls_f,
                                                       controlParameterization=controlParameterization,
                                                       initializationFunction=xinit_f,
                                                       parameters=params
                                                       )

# discretize the OCP
macroCollPoints = np.array(ca.collocation_points(params.dmacro, 'legendre'))
macroIntegrator = OthorgonalCollocation(collPoints=macroCollPoints)
macroScheme = MacroDiscretizationScheme(tauf=params.tauf, Nintervals=1, integrationScheme=macroIntegrator)
H = params.tauf / macroScheme.Nintervals  # todo: move this to the macroScheme class

if ADA_TYPE == 'FD':
    averageDynamicsApprox = ForwardDifferences(cycleSim=cycleDiscretizationClass, parameters=params)
elif ADA_TYPE == 'CD2':
    averageDynamicsApprox = CentralDifferences(cycleSim=cycleDiscretizationClass, K=2, parameters=params)
elif ADA_TYPE == 'CD3':
    averageDynamicsApprox = CentralDifferences(cycleSim=cycleDiscretizationClass, K=3, parameters=params)
else:
    raise ValueError(f"ADA_TYPE {ADA_TYPE} not recognized")

# %% construct the average OCP

# variables
w = struct_symSX([
    (
        # entry("controls", shape=(cycleDiscretizationClass.nu, controlParameterization.Nctrl),
        #       repeat=params.dControl),
        entry("Xcoll", shape=cycleDiscretizationClass.model.x.shape,
              repeat=[macroScheme.Nintervals, macroScheme.integrationScheme.d]),
        # collocation state nodes
        entry("Vcoll", shape=cycleDiscretizationClass.model.x.shape,
              repeat=[macroScheme.Nintervals, macroScheme.integrationScheme.d]),
        # collocation state derivative nodes
        entry("Zcoll", struct=averageDynamicsApprox.Z,
              repeat=[macroScheme.Nintervals, macroScheme.integrationScheme.d])
        # collocation algebraic state nodes
    ),
    entry("Xe", shape=cycleDiscretizationClass.model.x.shape, repeat=macroScheme.Nintervals + 1),  # ms state nodes
    entry("p_free", struct=params.syms_free_struct)  # parameters to optimize
])
params.replaceFreeSyms(w['p_free'])

# initialization
w0 = w(0)
w0['Xe'] = xinit
w0['Xcoll'] = xinit
w0['Vcoll'] = 0
w0['Zcoll'] = averageDynamicsApprox.getZ0(xinit)
w0['p_free', 'U_cycles'] = 0
# w0['controls', 0, :, 0] = 1

# constraints
constraints_struct = struct_symSX([
    entry('inital condition', shape=cycleDiscretizationClass.model.x.shape),
    entry('final condition', shape=E_f(x0bar).shape),
    # RK equations for every integration interval
    entry('IRK_dynamics', shape=model.x.shape, repeat=[macroScheme.Nintervals,
                                                       macroScheme.integrationScheme.d]),
    entry('IRK_algebraic', struct=averageDynamicsApprox.constraints_struct, repeat=[macroScheme.Nintervals,
                                                                                    macroScheme.integrationScheme.d]),
    entry('IRK_equation', shape=model.x.shape, repeat=[macroScheme.Nintervals, macroScheme.integrationScheme.d]),
    entry('IRK_terminal', shape=model.x.shape, repeat=[macroScheme.Nintervals]),
])
constraints = struct_SX(constraints_struct)

# unpack RK scheme coefficients
c, A, b = macroScheme.integrationScheme.unpack()

# collect stage cost
Jstage = 0

# initial condition
constraints['inital condition'] = w['Xe', 0] - x0bar

# U_list_w = w['controls']

# iterate stages (hopefully only one)
for k in range(macroScheme.Nintervals):

    Xs = w["Xcoll", k]
    Vs = w["Vcoll", k]
    Zs = w["Zcoll", k]

    # get start point of the interval
    X0 = w["Xe", k]

    # connect integration variables in RK scheme
    Xc_end = X0
    for i in range(macroScheme.integrationScheme.d):

        # get tau for this stage
        tau_i = H * c[i] + k * H

        # unpack variables
        x = Xs[i]
        z = averageDynamicsApprox.Z(Zs[i])
        v = Vs[i]

        # IRK dynamics equations - average dynamics
        constraints['IRK_dynamics', k, i] = H * averageDynamicsApprox.F(x, z, tau_i, *params.syms_list) - v

        # IRK algebraic equations - implicit approximation
        constraints['IRK_algebraic', k, i] = averageDynamicsApprox.G(x, z, tau_i, *params.syms_list)

        # IRK stage values
        xj_bar = X0
        for j in range(macroScheme.integrationScheme.d):
            xj_bar = xj_bar + A[i, j] * Vs[j]
        constraints['IRK_equation', k, i] = x - xj_bar

        # IRK endpoint
        Xc_end = Xc_end + b[i] * Vs[i]

        # IRK stage cost integration
        Jstage += b[i] * H * averageDynamicsApprox.f_averageCycleCost(x, z, tau_i, *params.syms_list)

    # connect IRK endpoint to next interval
    constraints['IRK_terminal', k] = Xc_end - w["Xe", k + 1]

Xend = w["Xe", macroScheme.Nintervals]

constraints['final condition'] = E_f(Xend)

# regularize the first derivative of the average dynamics
# for i in range(macroScheme.integrationScheme.d):
#     Jstage += 1E-4 * w['Vcoll', 0, i].T @ w['Vcoll', 0, i] * macroIntegrator.b[i] * params.tauf


# construct and solve the NLP
problem = {'f': Jstage,
           'x': w,
           'g': constraints,
           'p': params.syms_fix_struct
           }
options = {'ipopt.linear_solver': 'MA27',
           'ipopt.max_iter': 500,
           'ipopt.max_soc': 0,
           }
# options.update(additionalSolverOptions)

solver = ca.nlpsol('solver', 'ipopt', problem, options)

constraints_f = ca.Function('constraint_f', [w, params.syms_fix_struct], [constraints])
constraints_eval = constraints_struct(constraints_f(w0, params.syms_fix_struct(epsilon_bar)))
lbg = constraints_struct(0)
ubg = constraints_struct(0)

# %% solve the ocp
nlpsolution = solver(x0=w0, lbx=-ca.inf, ubx=ca.inf, lbg=lbg, ubg=ubg, p=params.syms_fix_struct(epsilon_bar))
wopt = w(nlpsolution['x'])
params.addSymsOptimalValues(wopt['p_free'], params.syms_fix_struct(epsilon_bar))

# %%  perform the simulation
controlParameterization_opt = FourierControlParameterization(model.nu, controlPoly, parameters=params,
                                                             N_har=params.N_har)
model_opt = Satellite2D(params=params)
minimize_controls_f_opt = ca.Function('minimize_controls_f', [model.x, model.tau, model.u] + model_opt.symbolicParams,
                                      [minimize_controls_f(model.x, model.tau, model.u, epsilon_bar)])
cycleClass_Simulation = CycleSingleShootingFixedPeriod(model=model_opt,
                                                       Nint=params.Nmicro * 30,
                                                       stageCostFunction=minimize_controls_f_opt,
                                                       controlParameterization=controlParameterization_opt,
                                                       parameters=params
                                                       )
simulation = cycleClass_Simulation.simulateNCycles(x0bar, Ncycles=int(params.tauf))

# %%  postprocess and plot

# postprocessing for plotting
tau_plot = np.linspace(0, params.tauf, int(params.tauf) * 100)
Xa_plot = ca.horzcat(*wopt['Xcoll', 0]).full()
# Xcoll_with_start = [wopt['Xe', 0]] + wopt['Xcoll', 0]
Xcoll_with_end = wopt['Xcoll', 0] + [wopt['Xe', 1]]
macro_poly_X  = LagrangePoly(macroIntegrator.d + 1, np.concatenate([macroIntegrator.c,[1]]))
Xcoll_poly_f = macro_poly_X.getEvalFunction(shape=model.x.shape, fixedCoeffs=Xcoll_with_end)
Xcoll_poly = Xcoll_poly_f.map(tau_plot.size)(tau_plot / params.tauf).full()

wopt_plotting = NumpyStruct(wopt)

# _sigma_plot = np.linspace(0,1,100)
_sigma_plot = np.linspace(-1, 1, 2000)  # will be filled with nans outside the range of the approximation
cycles_output: List[Tuple[np.ndarray, np.ndarray]] = []

averageCycleCosts = []

for i in range(macroScheme.integrationScheme.d):
    # individual cycles
    x = wopt['Xcoll', 0, i]
    z = wopt['Zcoll', 0, i]
    tau = params.tauf * macroScheme.integrationScheme.c[i]

    _x_cycle = averageDynamicsApprox.f_MicroIntOutputMap.map(_sigma_plot.size)(_sigma_plot, x, z, tau,
                                                                               *params.syms_opt_all_list).full()
    _tau_cyle = tau + _sigma_plot

    cycles_output.append((_tau_cyle, _x_cycle))

    averageCycleCosts.append(averageDynamicsApprox.f_averageCycleCost(x, z, tau, *params.syms_opt_all_list).full())

averageCycleCosts = np.vstack(averageCycleCosts)
# plot the average cycle costs
plt.figure()
plt.plot(params.tauf * c, averageCycleCosts, 'o', label='average')
plt.plot(np.arange(params.tauf), np.vstack([simulation.cycles[k].cycleCost for k in range(int(params.tauf))]), '.-',
         label='simulation')
plt.ylim([0, np.max(averageCycleCosts) * 1.1])
plt.grid(alpha=0.5)
plt.legend()
plt.show()

# plot the results

plt.figure(figsize=(15, 10))
for k in range(model.x.shape[0] - 1):
    plt.subplot(2, 2, k + 1)

    for i in range(macroScheme.integrationScheme.d):
        plt.plot(cycles_output[i][0], cycles_output[i][1][k, :], 'C2')
        _collocation_Nodes = NumpyStruct(wopt)['Zcoll', 0, i, 'Z_cycles', 0, 'X_Nodes']
        # plt.plot(np.linspace(0,1,cycleDiscretizationClass.Nint+1) + H*c[i] - 0.5, _collocation_Nodes[:, k], 'r.-', alpha=0.5)

        _x_plus = NumpyStruct(wopt)['Zcoll', 0, i, 'Z_cycles', 0, 'x_plus']
        _x_minus = NumpyStruct(wopt)['Zcoll', 0, i, 'Z_cycles', 0, 'x_minus']

        plt.plot(params.tauf * c[i] - 0.5, _x_minus[k], 'C3o')
        plt.plot(params.tauf * c[i] + 0.5, _x_plus[k], 'C3o')

        # _collocation_x_irk = NumpyStruct(wopt)['Zcoll', 0, i, 'Z_cycles', 0, 'z_irk',:,'x_irk']
    plt.plot(params.tauf * c, Xa_plot[k], 'C0o', markersize=3)
    plt.plot(tau_plot, Xcoll_poly[k, :], 'C0-')
    plt.plot(np.arange(2) * params.tauf, wopt_plotting['Xe', :, k], 'C0o')
    plt.plot([], [], 'C0o-', label=model.stateLabels[k])
    plt.plot(simulation.tau, simulation.X[k, :], 'C1', alpha=0.5, label=model.stateLabels[k])
    # plt.plot(tauSim_MPC, Xsim_MPC[k, :], 'C4.-', alpha=0.5, markersize=3,label="MPC SIM")
    plt.legend()
    plt.grid(alpha=PLT_GRID_ALPHA)

plt.tight_layout()
plt.show()

# %%  plot the optimal control trajectory

u_opt = controlParameterization_opt.getPlottingFunction().map(tau_plot.size)(tau_plot).full().squeeze()
plt.figure()
plt.plot(tau_plot, u_opt[0], label='thrust_p1')
plt.plot(tau_plot, u_opt[1], label='thrust_p2')
plt.legend()
plt.show()

# %% simulate final orbit
cycleClass_Simulation_periodic = CycleSingleShootingFixedPeriod(model=Satellite2D(Parameters(epsilon=0)),
                                                                Nint=params.Nmicro * 10,
                                                                parameters=params,
                                                                controlParameterization=ZeroControlParameterization()
                                                                )

finalOrbit = cycleClass_Simulation_periodic.simulateNCycles(xendbar, Ncycles=1)
startOrbit = cycleClass_Simulation_periodic.simulateNCycles(x0bar, Ncycles=1)

# %% Create nice plot for paper

fig, axes = plt.subplot_mosaic("AAA;AAA;AAA;BBB;CCC", figsize=(8.27, 11.69))
plt.sca(axes['A'])
plt.plot(simulation.X[0, :], simulation.X[1, :], 'C1', alpha=0.3)
plt.plot(startOrbit.X[0, :], startOrbit.X[1, :], 'C3-.', alpha=1, label='Initial Orbit')
plt.plot(finalOrbit.X[0, :], finalOrbit.X[1, :], 'C3--', alpha=1, label='Target Orbit')

for i in range(macroScheme.integrationScheme.d):
    plt.plot(cycles_output[i][1][0, :], cycles_output[i][1][1, :], 'C2', alpha=0.5)

plt.plot(Xa_plot[0, :], Xa_plot[1, :], 'C0o', markersize=3)
plt.plot(Xcoll_poly[0, :], Xcoll_poly[1, :], 'C0-')
plt.plot(wopt_plotting['Xe', :, 0], wopt_plotting['Xe', :, 1], 'C0o')
plt.plot([], [], 'C0o-', label='Average Solution')
plt.plot([], [], 'C2-', label='Micro Integrations', alpha=0.5)
plt.plot([], [], 'C1-', label='Simulation', alpha=0.3)
# add a circle with radius 0.3 and fill it with a grey texture
plt.gca().add_artist(plt.Circle((0, 0), 0.3, color='lightblue', alpha=1))

plt.xlabel('$p_1$')
plt.ylabel('$p_2$')

plt.grid(alpha=PLT_GRID_ALPHA)
plt.legend(loc='lower left')

# make the axis equal
plt.gca().set_aspect('equal', 'datalim')

for axkey, stateindex in zip(['B', 'C'], [0, 1]):
    plt.sca(axes[axkey])

    plt.plot(simulation.tau, simulation.X[stateindex, :], 'C1', alpha=0.3)
    # plt.plot(simulation.X[-1, :], simulation.X[stateindex, :], 'C1', alpha=0.3)
    for i in range(macroScheme.integrationScheme.d):
        plt.plot(cycles_output[i][0], cycles_output[i][1][stateindex, :], 'C2')
        # plt.plot(cycles_output[i][1][-1, :], cycles_output[i][1][stateindex, :], 'C2')

    # numerical time
    plt.plot(params.tauf * c, Xa_plot[stateindex], 'C0o', markersize=3)
    plt.plot(tau_plot, Xcoll_poly[stateindex, :], 'C0-')
    plt.plot(np.arange(2) * params.tauf, wopt_plotting['Xe', :, stateindex], 'C0o')

    plt.plot([], [], 'C0o-', label='Average Solution')
    plt.plot([], [], 'C2-', label='Micro Integrations')
    plt.plot([], [], 'C1-', label='Simulation', alpha=0.3)

    plt.ylabel(f'$p_{stateindex + 1}$')
    plt.xlabel('$\\tau$')
    # plt.xlabel('$t$')
    plt.legend(loc='lower left')
    plt.grid(alpha=PLT_GRID_ALPHA)

plt.tight_layout()
plt.savefig('../../../_Export/ErrorCD/Satellite_OCP.pdf')
plt.show()

# %% Log some results

error = np.linalg.norm(simulation.X[:, -1] - wopt['Xe',-1].full().flatten())
error_rel = error / np.linalg.norm(simulation.X[:, -1])

error_J = np.linalg.norm(simulation.Jint - nlpsolution['f'])
error_J_rel = error_J / np.linalg.norm(simulation.Jint)

logging.info(f'epsilon: {epsilon_bar:0.3e}')
logging.info(f'tauf: {params.tauf}')
logging.info(f'CtrlParam: {controlParameterization.__class__.__name__}')
logging.info(f'\t - N_Har: {controlParameterization.N_har}')

# SAM settings
logging.info('')
logging.info('SAM settings:')
logging.info(params)
logging.info(f'\tADA: {averageDynamicsApprox.name}')

# log nlp size
logging.info('')
logging.info('NLP Size:')
logging.info(f'\tw size: {w.size}')
logging.info(f'\tg size: {constraints.size}')

# log nlp solution stats
logging.info('')
logging.info('Solution stats:')
logging.info(f'\tJopt: {nlpsolution["f"]}')
logging.info(f'\tError_Jopt: {error_J_rel * 100:0.9f} %')
logging.info(f'\tError_End: {error:0.3e}')
logging.info(f'\tError_End_rel: {error_rel * 100:0.6f} %')
_iter_ms = (solver.stats()["t_wall_total"] / (solver.stats()["iter_count"])) * 1000
logging.info(f'\tt_iter: {_iter_ms:0.3f} ms')
from Core.configuration import logging
from timeit import default_timer as timer
from typing import List, Tuple

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
from casadi.tools import struct_symSX, entry, struct_SX
from Core.configuration import IPOPT_OPTIONS_LINEAR_SOLVER, PLT_GRID_ALPHA
from Core.controlparameterization import FourierControlParameterization, ZeroControlParameterization
from Core.CycleImplementations.FixedPeriodShooting import CycleSingleShootingFixedPeriod, \
    CycleMultipleShootingFixedPeriod, CycleCollocationFixedPeriod
from Core.integrators import RK4, OthorgonalCollocation, LegendreCollocation, RadauCollocation
from Core.macrodiscretizationschemes import MacroDiscretizationScheme
from Core.parameters import Parameters, FreeSymbolicParameter
from Core.polynomials import LagrangePoly
from Core.tools import NumpyStruct, smoothStep_tanh_f
from Core.figuretools import latexify, smoothPlotLimits
from Models.satellite2d import Satellite2D

latexify()

# %% Model and initial position

# epsilon and horizon
# epsilon = 0
# epsilon = 2 ** (-8)
# epsilon = 2 ** (-9)
# L = 1 / 16  # final time
# tauf = int(L / np.abs(epsilon))  # should be 256

# SAM settings
# N_perCycle = 50  # number of integration intervals per cycle

epsilon = 0.005
L = 1/5  # final time


# epsilon = 2 ** (-12)
# tauf = 30

# model
params = Parameters()
params.Nmicro = 30  # number of integration intervals per cycle
params.dmicro = 4 # number of collocation points for the micro discretization
params.dControl = 3  # degree of the control polynomial
params.N_har = 1

# symbolic parameters
params.declareSyms([FreeSymbolicParameter('U_cycles', [ca.DM.zeros((Satellite2D.nu, params.N_har + 2)) for i in
                                                   range(params.dControl)],
                                      shape=(Satellite2D.nu, params.N_har + 2), repeat=params.dControl)])

params.epsilon = epsilon
params.tauf = int(ca.floor(L / ca.norm_1(params.epsilon)))

# model
model = Satellite2D(params)

# initial position
x0bar = model.x_struct(0)
x0bar["p"] = [1, 0]
x0bar["v"] = [0, (np.sqrt(1 / x0bar["p",0]) + 0.05)]
x0bar = x0bar.cat

# target endpoint (on a circular orbit
xendbar = model.x_struct(0)
xendbar["p"] = [1.5, 0]
xendbar["v"] = [0, np.sqrt(1 / xendbar["p", 0])]
xendbar = xendbar.cat

# point used for initialization of the cycles
xinit = model.x_struct(0)
xinit["p"] = [1.2, 0]
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
                    # model.tau * model.T_exact(model.x)
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
# macroScheme = MacroDiscretizationScheme(tauf=tauf, Nintervals=1, integrationScheme=macroIntegrator)

logging.info('Constructing the full OCP ...')

# %% construct the average OCP
Ncycles = params.tauf
# variables
w = struct_symSX([
    entry("p_free", struct=params.syms_free_struct),  # parameters to optimize
    entry("Z_cycles", struct=cycleDiscretizationClass.z,
              repeat=[Ncycles])
])
params.replaceFreeSyms(w['p_free'])

# initialization
w0 = w(0)
w0['Z_cycles'] = cycleDiscretizationClass.z0(xinit)
w0['p_free','U_cycles'] = 0


# constraints
constraints_struct = struct_symSX([
    entry('inital condition', shape=cycleDiscretizationClass.model.x.shape),
    entry('final condition', shape=E_f(x0bar).shape),
    # RK equations for every integration interval
    entry('cycle_conditions', struct=cycleDiscretizationClass.cycleConditions_struct,repeat=[Ncycles]),
    entry('connect_cycles', shape = model.x.shape, repeat=[Ncycles-1])
])
constraints = struct_SX(constraints_struct)

# collect stage cost
Jstage = 0

# initial condition
constraints['inital condition'] = w['Z_cycles', 0,'x_minus'] - x0bar

# U_list_SX = w['controls']

for n in range(Ncycles):
    tau_cycle = n # start of the cycle

    # cycle conditions for this cycle
    constraints['cycle_conditions',n] = cycleDiscretizationClass.f_cycleConditions(w['Z_cycles', n], tau_cycle, *params.syms_list)

    # connect to next cycle
    if n < Ncycles - 1:
        constraints['connect_cycles', n] = w['Z_cycles', n + 1, 'x_minus'] - w['Z_cycles', n, 'x_plus']

    # cost function
    Jstage += cycleDiscretizationClass.f_cycleCost(w['Z_cycles', n], tau_cycle, *params.syms_list)

# final condition
Xend = w['Z_cycles', -1, 'x_plus']
constraints['final condition'] = E_f(Xend)

# construct and solve the NLP
supressOutput = False
problem = {'f': Jstage,
           'x': w,
           'g': constraints,
           }
options = {'ipopt.linear_solver': IPOPT_OPTIONS_LINEAR_SOLVER,
           'ipopt.max_iter': 200,
           'ipopt.print_level': 0 if supressOutput else 5,
           'print_time': False if supressOutput else True,
           'ipopt.max_soc': 0,
           # 'ipopt.mu_init': 1e-5,
           }
# options.update(additionalSolverOptions)

solver = ca.nlpsol('solver', 'ipopt', problem, options)


# %% solve the ocp

nlpsolution = solver(x0=w0, lbx=-ca.inf, ubx=ca.inf, lbg=0, ubg=0)


# %% extract the optimal solution
wopt = w(nlpsolution['x'])
params.addSymsOptimalValues(wopt['p_free'])


# %%  perform the simulation
controlParameterization_opt = FourierControlParameterization(model.nu, controlPoly, parameters=params,
                                                             N_har=params.N_har)
model_opt = Satellite2D(params=params)
minimize_controls_f_opt = ca.Function('minimize_controls_f', [model.x, model.tau, model.u] + model_opt.symbolicParams,
                                      [minimize_controls_f(model.x, model.tau, model.u)])
cycleClass_Simulation = CycleSingleShootingFixedPeriod(model=model_opt,
                                                       Nint=params.Nmicro * 50,
                                                       stageCostFunction=minimize_controls_f_opt,
                                                       controlParameterization=controlParameterization_opt,
                                                       parameters=params
                                                       )
simulation = cycleClass_Simulation.simulateNCycles(x0bar, Ncycles=int(params.tauf))

# %%  postprocess and plot

# postprocessing for plotting
tau_plot = np.linspace(0, params.tauf + 0.1, params.tauf * 100,endpoint=True)
# tau_plot = np.linspace(params.tauf-1, params.tauf-1E8, 2000,endpoint=True)
tau_plot_strobo = np.arange(0,params.tauf+1,dtype=float)
tau_plot_strobo[-1] = params.tauf -1E-12

from Core.tools import constructPiecewiseCasadiExpression
cyclesEval_fs = []
edges = []
cycleCosts = []
for n in range(Ncycles):
    _sigma = model.tau - ca.floor(model.tau)
    _zopt_cycle = wopt['Z_cycles', n]
    cyclesEval_fs.append(cycleDiscretizationClass.f_outputMap(_sigma, _zopt_cycle, model.tau, *params.syms_opt_all_list))
    edges.append(n)

    cycleCosts.append(cycleDiscretizationClass.f_cycleCost(_zopt_cycle, n, *params.syms_opt_all_list))

edges.append(Ncycles)
cyclesEval = constructPiecewiseCasadiExpression(model.tau,edges, cyclesEval_fs)
cyclesEval_f = ca.Function('cyclesEval_f', [model.tau], [cyclesEval])

Xopt = cyclesEval_f.map(tau_plot.size)(tau_plot).full().squeeze()
Xopt_strobo = cyclesEval_f.map(tau_plot_strobo.size)(tau_plot_strobo).full().squeeze()

# plot the results
plt.figure(figsize=(15, 10))
for k in range(model.x.shape[0]):
    plt.subplot(2, 2, k + 1)


    plt.plot(tau_plot, Xopt[k, :], 'C0-')
    plt.plot(tau_plot_strobo, Xopt_strobo[k, :], 'C0.--')
    # plt.plot(np.arange(2) * tauf, wopt_plotting['Xe', :, k], 'C0o')
    # plt.plot([], [], 'C0o-', label=model.stateLabels[k])
    plt.plot(simulation.tau, simulation.X[k, :], 'C1-', alpha=0.5, label=model.stateLabels[k])
    # plt.plot(simulation.tau, simulation.X[k, :], 'C1-', alpha=0.5, label="SIM")
    # plt.plot([cycle.tau[-1] for cycle in simulation.cycles],
    #          [cycle.X[k,-1] for cycle in simulation.cycles], 'C1.', alpha=0.5, label='SIM')
    # plt.plot(tauSim_MPC, Xsim_MPC[k, :], 'C4.-', alpha=0.5, markersize=3,label="MPC SIM")
    plt.legend()
    plt.grid(alpha=PLT_GRID_ALPHA)

plt.tight_layout()
plt.show()


# %% debug: plot the last cycle


tau_plot_last = np.linspace(39, 40, 2000)
tau_Nodes_last = np.linspace(39, 40, params.Nmicro+1)
_sigma = model.tau - ca.floor(model.tau)
lastcyclesEval = cycleDiscretizationClass.f_outputMap(_sigma, wopt['Z_cycles', 39], model.tau, *params.syms_opt_all_list)
lastcyclesEval_f = ca.Function('lastcyclesEval_f', [model.tau], [lastcyclesEval])

Xopt_last = lastcyclesEval_f.map(tau_plot_last.size)(tau_plot_last).full().squeeze()
X_Nodes = ca.horzcat(*wopt['Z_cycles', 39,'X_Nodes',:]).full().squeeze()

plt.figure()
for k in range(model.x.shape[0]):
    plt.subplot(2, 2, k + 1)
    plt.plot(tau_plot_last, Xopt_last[k, :], 'C0-',label=model.stateLabels[k])
    plt.plot([39,40], ca.vertcat(wopt['Z_cycles', 39,'x_minus'][k],wopt['Z_cycles', 39,'x_plus'][k]).full().flatten(), 'C1.')
    plt.plot(tau_Nodes_last, X_Nodes[k,:], 'C1.')
    plt.legend()
    plt.grid(alpha=PLT_GRID_ALPHA)

    # zoom into the last integration interval
    # plt.xlim(smoothPlotLimits(40 - 1/params.Nmicro, 40))
    # plt.ylim(smoothPlotLimits(*X_Nodes[k,[-2,-1]]))

plt.show()



# %%  plot the optimal control trajectory

u_opt = controlParameterization_opt.getPlottingFunction().map(tau_plot.size)(tau_plot).full().squeeze()
plt.figure()
plt.plot(tau_plot, u_opt[0], label='thrust_p1')
plt.plot(tau_plot, u_opt[1], label='thrust_p2')
plt.show()

# %% simulate final orbit


cycleClass_Simulation_periodic = CycleSingleShootingFixedPeriod(model=Satellite2D(params=Parameters(epsilon=0)),
                                                      Nint=params.Nmicro*10,
                                                      parameters=params,
                                                      controlParameterization=ZeroControlParameterization()
                                                      )

finalOrbit = cycleClass_Simulation_periodic.simulateNCycles(xendbar, Ncycles=1)
startOrbit = cycleClass_Simulation_periodic.simulateNCycles(x0bar, Ncycles=1)


# plot the average cycle costs

# %% plot the average cycle costs
plt.figure()
plt.plot(np.arange(params.tauf), ca.vertcat(*cycleCosts).full(),'o-', label='cyclesCosts')
plt.plot(np.arange(params.tauf), np.vstack([simulation.cycles[k].cycleCost for k in range(params.tauf)]), '.-',
         label='simulation')
plt.ylim([0, np.max(cycleCosts) * 1.1])
plt.grid(alpha=0.5)
plt.legend()
plt.show()

# %% Create nice plot for paper

fig, axes = plt.subplot_mosaic("AAA;AAA;AAA;BBB;CCC", figsize=(8.27, 11.69))
plt.sca(axes['A'])
plt.plot(simulation.X[0, :], simulation.X[1, :], 'C1', alpha=0.3)
plt.plot(startOrbit.X[0, :], startOrbit.X[1, :], 'C3-.', alpha=1, label='Initial Orbit')
plt.plot(finalOrbit.X[0, :], finalOrbit.X[1, :], 'C3--', alpha=1, label='Target Orbit')

plt.plot(Xopt[0, :], Xopt[1, :], 'C0-',alpha=0.4)
plt.plot(Xopt_strobo[0, :], Xopt_strobo[1, :], 'C0.-')
plt.plot([], [], 'C0.-', label='Full Solution')
# plt.plot([], [], 'C2-', label='Micro Integrations', alpha=0.5)
# plt.plot([], [], 'C1-', label='Simulation', alpha=1)
# add a circle with radius 0.3 and fill it with a grey texture
plt.gca().add_artist(plt.Circle((0, 0), 0.3, color='lightblue', alpha=1))

plt.xlabel('$p_1$')
plt.ylabel('$p_2$')

plt.grid(alpha=PLT_GRID_ALPHA)
plt.gca().set_aspect('equal', 'datalim')
plt.legend(loc='lower left')
# make the axis equal

for axkey, stateindex in zip(['B', 'C'], [0, 1]):
    plt.sca(axes[axkey])

    # plt.plot(simulation.tau, simulation.X[stateindex, :], 'C1-', alpha=0.3)
    plt.plot(simulation.tau, simulation.X[stateindex, :], 'C1', alpha=0.3)
    # numerical time
    plt.plot(tau_plot, Xopt[stateindex], 'C0-',alpha=0.5)
    plt.plot(tau_plot_strobo, Xopt_strobo[stateindex, :], 'C0.')
    # plt.plot(np.arange(2) * tauf, wopt_plotting['Xe', :, stateindex], 'C0o')

    # physical time
    # plt.plot(Xa_plot[-1], Xa_plot[stateindex], 'C0o', markersize=3)
    # plt.plot(Xcoll_poly[-1], Xcoll_poly[stateindex, :], 'C0-')
    # plt.plot(wopt_plotting['Xe', :, -1], wopt_plotting['Xe', :, stateindex], 'C0o')

    plt.plot([], [], 'C0o-', label='Full Solution')
    # plt.plot([], [], 'C2-', label='Micro Integrations')
    # plt.plot([], [], 'C1-', label='Simulation', alpha=0.3)

    plt.ylabel(f'$p_{stateindex + 1}$')
    plt.xlabel('$\\tau$')
    # plt.xlabel('$t$')
    plt.legend(loc='lower left')
    plt.grid(alpha=PLT_GRID_ALPHA)

plt.tight_layout()
plt.savefig('../../../_Export/ErrorCD/Satellite_OCP_full.pdf')
plt.show()

# %% Log some results

error = np.linalg.norm(simulation.X[:,-1] - wopt['Z_cycles',-1,'x_plus'])
error_rel = error / np.linalg.norm(simulation.X[:,-1])

error_J = np.linalg.norm(simulation.Jint - nlpsolution['f'])
error_J_rel = error_J / np.linalg.norm(simulation.Jint)

logging.info(f'epsilon: {epsilon:0.3e}')
logging.info(f'tauf: {params.tauf}')
logging.info(f'CtrlParam: {controlParameterization.__class__.__name__}')
logging.info(f'\t - NHar: {controlParameterization.N_har}')

# log nlp size
logging.info('')
logging.info('NLP Size:')
logging.info(f'\tw size: {w.size}')
logging.info(f'\tg size: {constraints.size}')

# log nlp solution stats
logging.info('')
logging.info('Solution stats:')
logging.info(f'\tJopt: {nlpsolution["f"]}')
logging.info(f'\tError_Jopt: {error_J_rel*100:0.3f} %')
logging.info(f'\tError_End: {error:0.3e}')
logging.info(f'\tError_End_rel: {error_rel*100:0.3f} %')
_iter_ms = (solver.stats()["t_wall_total"]/(solver.stats()["iter_count"]))*1000
logging.info(f'\tt_iter: {_iter_ms:0.3f} ms')







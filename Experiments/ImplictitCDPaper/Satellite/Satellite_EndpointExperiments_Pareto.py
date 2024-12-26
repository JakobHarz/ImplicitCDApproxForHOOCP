import uuid
from typing import List

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from casadi.tools import struct_symSX, entry

from Core.controlparameterization import ZeroControlParameterization
from Core.parameters import Parameters
from Core.tools import floatToExpStr
from Core.figuretools import latexify
from Core.cycle import Cycle
from Core.CycleImplementations.FixedPeriodShooting import CycleSingleShootingFixedPeriod
from Models.satellite2d import Satellite2D
from Core.configuration import PLT_GRID_ALPHA, IPOPT_OPTIONS_LINEAR_SOLVER
from Core.averagedynamicsapproximations import AverageDynamicsApproximation, ForwardDifferences, \
    CentralDifferences, LIPS2
from Core.integrators import RungeKuttaIntegrator, RK4, LegendreCollocation, RadauCollocation, Lobatto3A_Order2

latexify()

# values to iterate over
INTERVAL_SCALED = True
# epsilons = [5E-4,1E-3,1E-2]
# epsilons = [1E-3, 0.5E-3, 1E-4]
# epsilons = [2**(-12)]
# epsilons = [0.5E-2]

# nMicroInts = np.logspace(1.5, 2.5, 15).astype(int)
# nMicroInts = [100]
L = 1 / 16  # final time
# L = 1 / 16  # final time
# nMacroInts = [2, 8]

if INTERVAL_SCALED:
    tau_f = lambda epsilon: int(L / np.abs(epsilon))
else:
    tau_f = lambda epsilon: int(L * 10000)

# central difference approximations
Kmax = 2
# Ks = np.arange(2, Kmax + 1)

model = Satellite2D(Parameters(epsilon=0))
x0bar = model.x_struct(0)
x0bar["p"] = [1, 0]
x0bar["v"] = [0, 1]
x0bar = x0bar.cat

additionalSolverOptions = {'ipopt.tol': 1e-8,
                           'ipopt.acceptable_tol': 1e-8}  # low tolerances for the solver


def constructImplicitSimulationProblem(cycleSim: Cycle,
                                       averApprox: AverageDynamicsApproximation,
                                       N: int,
                                       X0bar: ca.DM,
                                       Nmacro: int,
                                       IntMacro: RungeKuttaIntegrator) -> dict:
    """
    Constructs the equality constraints for the simulation using SAM.
    :param cycleSim: CycleSimulation object, e.g. CycleSingleShootingFixedPeriod
    :param averApprox: AverageDynamicsApproximation object, e.g. ForwardDifferences
    :param N: total number of cycles to simulate
    :param X0bar: initial point
    :param Nmacro: number of macro integration steps
    :param IntMacro: macro integration scheme
    :return: a dictionary to be used with ca.NLPsol
    """

    print('Constructing SAM Problem with the following parameters:')
    print(f'\t N = {N}')
    print(f'\t Nmacro = {Nmacro}')
    print(f'\t IntMacro = {IntMacro}')
    print(f'\t NMicro = {cycleSim.Nint}')
    print(f'\t AverApprox = {averApprox.name}')

    # get the model from the cycle sim
    model = cycleSim.model

    H = N / Nmacro  # macro integration step size

    # build the variables
    w = struct_symSX([
        (
            entry("Xcoll", shape=model.x.shape,
                  repeat=[Nmacro, IntMacro.d]),
            # collocation state nodes
            entry("Vcoll", shape=model.x.shape,
                  repeat=[Nmacro, IntMacro.d]),
            # collocation state derivative nodes
            entry("Zcoll", struct=averApprox.Z,
                  repeat=[Nmacro, IntMacro.d])
            # collocation algebraic state nodes
        ),
        entry("X", shape=model.x.shape, repeat=Nmacro + 1)  # ms state nodes
    ])

    # intial guess
    w0 = w(0)
    w0['Zcoll', :, :] = averApprox.getZ0(x0bar)
    w0['X', :] = X0bar
    w0['Xcoll', :, :] = X0bar

    g = []  # list of constraints, only equality!

    g.append(w['X', 0] - X0bar)  # initial condition

    X0 = w['X', 0]

    for k in range(Nmacro):

        # Connect Integration Variables
        Xc_end = X0

        for i in range(IntMacro.d):

            tau_coll_k = H * IntMacro.c[i] + H*k
            Xcoll_k = w['Xcoll', k, i]
            Vcoll_k = w['Vcoll', k, i]
            Zcoll_k = w['Zcoll', k, i]

            # Append dynamic equations of DAE
            g.append(H * (averApprox.F(Xcoll_k, Zcoll_k,tau_coll_k)) - Vcoll_k)

            # Append algebraic equation of DAE
            g.append(averApprox.G(Xcoll_k, Zcoll_k,tau_coll_k))

            # equation for rk
            xj_bar = X0

            for j in range(IntMacro.d):
                xj_bar = xj_bar + IntMacro.A[i, j] * w['Vcoll', k, j]
            g.append(xj_bar - Xcoll_k)

            # Add contribution to the end state
            Xc_end = Xc_end + IntMacro.b[i] * Vcoll_k

        # Add equality constraint
        Xend = w['X', k + 1]
        g.append(Xc_end - Xend)

        X0 = Xend

    return {'x': w, 'x0': w0, 'g': ca.vertcat(*g), 'f': 0}


# %%%

def experiment(epsilon_list: List,
               microN_list: List,
               approximations_list: List[AverageDynamicsApproximation],
               macroN_list: List[int],
               macroInt_list: List[RungeKuttaIntegrator]) -> \
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Performs the experiment for the given list of parameters
    :param epsilon_list:
    :param microN_list:
    :param approximations_list:
    :param macroN_list:
    :param macroInt_list:
    :return: efforts, errors, efforts_full, errors_full
    """

    print('Performing Experiment with the following parameters:')
    print(f'epsilons = {epsilon_list}')
    print(f'nMicroInts = {microN_list}')
    print(f'Approximations = {[approx.name for approx in approximations_list]}')
    print(f'nMacroInts = {macroN_list}')
    print(f'MacroIntegrators = {[str(integrator) for integrator in macroInt_list]}')
    totalNLPS = len(epsilon_list) * len(microN_list) * len(approximations_list) * len(macroN_list) * len(macroInt_list)

    errors = np.zeros((len(epsilon_list), len(microN_list), len(approximations_list), len(macroN_list), len(macroInt_list)))
    efforts = np.zeros((len(epsilon_list), len(microN_list), len(approximations_list), len(macroN_list), len(macroInt_list)))

    errors_full = np.zeros((len(epsilon_list), len(microN_list)))
    efforts_full = np.zeros((len(epsilon_list), len(microN_list)))

    # nlp counter
    nlp_counter = 0

    # iterate over epsilons
    for index_eps, epsilon in enumerate(epsilon_list):
        print(f'epsilon = {epsilon}')
        model = Satellite2D(Parameters(epsilon=epsilon))

        x0bar = model.x_struct(0)
        x0bar["p"] = [1, 0]
        x0bar["v"] = [0, 1]
        x0bar = x0bar.cat

        # perform 'true' simulation
        cycleSim_true = CycleSingleShootingFixedPeriod(model, Nint=3000, controlParameterization=ZeroControlParameterization())
        ZoptTrue = cycleSim_true.solveNCycles(x0bar, tau_f(epsilon), supressOutput=True,
                                              additionalSolverOptions=additionalSolverOptions)
        xend_true = ZoptTrue[-1]['x_plus'].full()

        # iterate over number of micro-integrations
        for index_nMu, nMicroInt in enumerate(microN_list):

            print(f'nMicroInt = {nMicroInt}')
            # cycleSimulationClass for this configuration
            cycleSim_forward = CycleSingleShootingFixedPeriod(model, Nint=nMicroInt, controlParameterization=ZeroControlParameterization())
            cycleSim_forward.id = uuid.uuid4()

            # compute the endpoint full integration
            Zoptfull = cycleSim_forward.solveNCycles(x0bar, tau_f(epsilon), supressOutput=True,
                                                     additionalSolverOptions=additionalSolverOptions)
            xend_full = Zoptfull[-1]['x_plus'].full()
            errors_full[index_eps, index_nMu] = np.linalg.norm(np.diag([1, 1, 1, 1] @ (xend_true - xend_full)))
            efforts_full[index_eps, index_nMu] = nMicroInt * tau_f(epsilon)

            # iterate over K
            for index_approximations, approximation in enumerate(approximations_list):
                # iterate over tau_f
                for index_nMacro, nMacro in enumerate(macroN_list):

                    # true endpoint
                    for index_int, integrator in enumerate(macroInt_list):
                        nlp_counter += 1
                        print(
                            f'epsilon = ({index_eps + 1}|{len(epsilon_list)}),'
                            f' nMicroInt = ({index_nMu + 1}|{len(microN_list)}),'
                            f'Approx = ({index_approximations + 1}|{len(approximations_list)}), '
                            f'nMacro = ({index_nMacro + 1}|{len(macroN_list)}),'
                            f' Integrators = ({index_int + 1}|{len(macroInt_list)})',
                            f'\t Total: ({nlp_counter}|{totalNLPS})',
                            # end='\r',
                            flush=True)

                        # solve the SAM problem
                        N = tau_f(epsilon)
                        _approximation = approximation.copy(cycleSim_forward)
                        problemDict = constructImplicitSimulationProblem(cycleSim_forward,
                                                                         _approximation,
                                                                         N,
                                                                         x0bar,
                                                                         nMacro, integrator)
                        supressOutput = True
                        problem = {'f': 0, 'x': problemDict['x'], 'g': problemDict['g']}
                        options = {'ipopt.linear_solver': IPOPT_OPTIONS_LINEAR_SOLVER,
                                   'ipopt.max_iter': 50,
                                   'ipopt.print_level': 0 if supressOutput else 5,
                                   'print_time': False if supressOutput else True,
                                   }
                        options.update(additionalSolverOptions)

                        solver = ca.nlpsol('solver', 'ipopt', problem, options)
                        nlpsolution = solver(x0=problemDict['x0'], lbx=-ca.inf, ubx=ca.inf, lbg=0, ubg=0)
                        wopt = problemDict['x'](nlpsolution['x'])
                        xend = wopt['X', -1].full()

                        # check if solver failed
                        solver_failed = (solver.stats()['return_status'] != 'Solve_Succeeded')
                        if solver_failed:
                            print(f'WARNING: Solver failed! Return status: {solver.stats()["return_status"]}')

                        # collect the errors and efforts
                        error = np.linalg.norm(np.diag([1, 1, 1, 1]) @ (xend - xend_true))
                        if solver_failed: error = np.nan

                        # store the error
                        errors[index_eps, index_nMu, index_approximations, index_nMacro, index_int] = error
                        effort = approximation.Ncycles * nMicroInt * integrator.d * nMacro  # 4 due to rk4
                        efforts[index_eps, index_nMu, index_approximations, index_nMacro, index_int] = effort

                        # build a settings dict with list of strings of arguments
                        settings = {'epsilon_list': epsilon_list,
                                    'microN_list': microN_list,
                                    'macroN_list': macroN_list}
                        settings['approximations_list'] = [approx.name for approx in approximations_list]
                        settings['macroInt_list'] = [str(integrator) for integrator in macroInt_list]


    print('Done!')
    return efforts, errors, efforts_full, errors_full, settings


# print(asdf)

# # %% Experiment 1: Varying Nmicro, Variying Nmacro, Varying K
#
# epsilons_1 = [2 ** (-12)]
# nMicroInts_1 = np.logspace(1.5, 2.5, 5).astype(int)
# nMacroInts_1 = [2, 8]
#
# # central difference approximations
# Kmax = 2
# Ks_1 = np.arange(2, Kmax + 1)
# integrators_1 = [RK4()]
# approximations_1 = [CentralDifferences(CycleSingleShootingFixedPeriod(Satellite2D(), fixedPeriod=1, Nctrl=1, NintPerCtrl=10), x0bar, K) for K in Ks_1]
#
#
# efforts_1, errors_1, efforts_full_1, errors_full_1 = experiment(epsilons_1, nMicroInts_1, approximations_1, nMacroInts_1,
#                                                                 integrators_1)
#
# # %% Experiment 1 - Plotting
# plt.figure(figsize=(len(nMacroInts_1) * 4.5, 4 * len(epsilons_1)))
#
# for index_eps, epsilon in enumerate(epsilons_1):
#     for index_nMacro, nMacro in enumerate(nMacroInts_1):
#         plt.subplot(len(epsilons_1), len(nMacroInts_1), index_eps * len(nMacroInts_1) + index_nMacro + 1)
#
#         for index_K, K in enumerate(Ks_1):
#             alpha = 0.3 if K % 2 == 1 else 1
#             plt.loglog(efforts_1[index_eps, :, index_K, index_nMacro], errors_1[index_eps, :, index_K, index_nMacro], ".-",
#                        label=f'K = {K}', alpha=alpha)
#
#         # plot the full
#         plt.loglog(efforts_full_1[index_eps, :], errors_full_1[index_eps, :], ".-", color='red', label=f'Full', alpha=1)
#
#         # plt.title(
#         #     f"$\\epsilon = ${floatTo10ExpStr(epsilon)} $\\rightarrow \\tau_f = {tau_f(epsilon)}$, $H = {tau_f(epsilon) / nMacro}$")
#
#         plt.title(
#             f"$\\epsilon = ${floatToExpStr(epsilon, base=2)}, $N_\\mathrm{{macro}} = {nMacro}$")
#         plt.xlabel("Effort in total number of micro-int steps")
#         plt.ylabel("Error")
#         plt.grid(alpha=PLT_GRID_ALPHA)
#         plt.legend(loc='lower left')
#         plt.xlim([2E2, 2E5])
#         plt.ylim([1E-6, 5E-1])
# # plt.gcf().suptitle(
# #     f"Endpoint error, integration interval " + ("$[0,L/\\epsilon]$" if INTERVAL_SCALED else "$[0,L]$") + "\n",
# #     fontsize=15)
# plt.tight_layout()
# plt.savefig('../../../_Export/ErrorCD/Satellite_EndpointError_onlyTwo.pdf')
# plt.show()

# %% Experiment 2: Vary approximation schemes, Nmicro, epsilon and Nmacro

# epsilons_2 = [2 ** (-11), 2 ** (-12), 2 ** (-13)]
# nMicroInts_2 = np.logspace(1.5, 2.5, 5).astype(int)
# nMacroInts_2 = [2,4]
#
# # central difference approximations
# integrators_2 = [RK4()]
# _baseCycleSim = CycleSingleShootingFixedPeriod(Satellite2D(), fixedPeriod=1, Nctrl=1, NintPerCtrl=10)
# approximations_2 = [CentralDifferences(_baseCycleSim, x0bar, 2),
#                     CentralDifferences(_baseCycleSim, x0bar, 3),
#                     ForwardDifferences(_baseCycleSim, x0bar)]
#
#
# efforts_2, errors_2, efforts_full_2, errors_full_2 = experiment(epsilons_2, nMicroInts_2, approximations_2, nMacroInts_2,
#                                                                 integrators_2)
#
#
# # %% Experiment 2 - Plotting
# plt.figure(figsize=(len(nMacroInts_2) * 4.5, 4 * len(epsilons_2)))
#
# for index_eps, epsilon in enumerate(epsilons_2):
#     for index_nMacro, nMacro in enumerate(nMacroInts_2):
#         plt.subplot(len(epsilons_2), len(nMacroInts_2), index_eps * len(nMacroInts_2) + index_nMacro + 1)
#
#         for approximation_index, approximation in enumerate(approximations_2):
#             # alpha = 0.3 if K % 2 == 1 else 1
#             alpha = 1
#             plt.loglog(efforts_2[index_eps, :, approximation_index, index_nMacro], errors_2[index_eps, :, approximation_index, index_nMacro], ".-",
#                        label=f'{approximation.name}', alpha=alpha)
#
#         # plot the full
#         plt.loglog(efforts_full_2[index_eps, :], errors_full_2[index_eps, :], ".-", color='red', label=f'Full', alpha=1)
#
#         # plt.title(
#         #     f"$\\epsilon = ${floatTo10ExpStr(epsilon)} $\\rightarrow \\tau_f = {tau_f(epsilon)}$, $H = {tau_f(epsilon) / nMacro}$")
#
#         plt.title(
#             f"$\\epsilon = ${floatToExpStr(epsilon, base=2)}, $N_\\mathrm{{macro}} = {nMacro}$")
#         plt.xlabel("Effort in total number of micro-int steps")
#         plt.ylabel("Error")
#         plt.grid(alpha=PLT_GRID_ALPHA)
#         plt.legend(loc='lower left')
#         plt.xlim([2E2, 2E5])
#         plt.ylim([1E-6, 5E-1])
# # plt.gcf().suptitle(
# #     f"Endpoint error, integration interval " + ("$[0,L/\\epsilon]$" if INTERVAL_SCALED else "$[0,L]$") + "\n",
# #     fontsize=15)
# plt.tight_layout()
# plt.savefig('../../../_Export/ErrorCD/Satellite_EndpointError_onlyTwo.pdf')
# plt.show()

# %% Experiment 3: Vary approximation schemes, MacroScheme, Nmicro, Nmacro

epsilons_3 = [2 ** (-12)]
nMicroInts_3 = np.logspace(1.5, 3, 15).astype(int).tolist()
nMacroInts_3 = [1, 2]

integrators_3 = [
    LegendreCollocation(2),
    LegendreCollocation(3),
    LegendreCollocation(4),
    # LegendreCollocation(1),
    RK4(),
    # RadauCollocation(1),
    RadauCollocation(2),
    # Lobatto3A(),
    RadauCollocation(3),
    RadauCollocation(4)
]
# _baseCycleSim = CycleSingleShootingFixedPeriod(Satellite2D(), fixedPeriod=1, Nctrl=1, NintPerCtrl=10)
_baseCycleSim = CycleSingleShootingFixedPeriod(Satellite2D(Parameters(epsilon=0)), Nint=50, controlParameterization=ZeroControlParameterization())
approximations_3 = [
                    # LIPS2(_baseCycleSim),
    CentralDifferences(_baseCycleSim, K=2),
    CentralDifferences(_baseCycleSim, K=3),
    ForwardDifferences(_baseCycleSim)
                    ]

efforts_3, errors_3, efforts_full_3, errors_full_3, settings_3 = experiment(epsilons_3, nMicroInts_3, approximations_3,
                                                                nMacroInts_3,
                                                                integrators_3)

# %% Experiment 3 - Plotting
plt.figure(figsize=(len(integrators_3) * 4.5, 4 * len(epsilons_3)))

# errors = (epsilons, nMicroInts, Approximations, nMacroInts, MacroIntegrators)

for index_eps, epsilon in enumerate(epsilons_3):
    for index_int, integrator in enumerate(integrators_3):
        plt.subplot(len(epsilons_3), len(integrators_3), index_eps * len(integrators_3) + index_int + 1)

        for approximation_index, approximation in enumerate(approximations_3):
            # alpha = 0.3 if K % 2 == 1 else 1
            alpha = 1
            plt.loglog(efforts_3[index_eps, :, approximation_index, 0, index_int],
                       errors_3[index_eps, :, approximation_index, 0, index_int], ".-",
                       label=f'{approximation.name}', alpha=alpha)

        # plot the full
        plt.loglog(efforts_full_3[index_eps, :], errors_full_3[index_eps, :], ".-", color='red', label=f'Full', alpha=1)

        # plt.title(
        #     f"$\\epsilon = ${floatTo10ExpStr(epsilon)} $\\rightarrow \\tau_f = {tau_f(epsilon)}$, $H = {tau_f(epsilon) / nMacro}$")

        plt.title(
            f"$\\epsilon = ${floatToExpStr(epsilon, base=2)}, Integrator: {integrator}")
        plt.xlabel("Effort in total number of micro-int steps")
        plt.ylabel("Endpoint Error")
        plt.grid(alpha=PLT_GRID_ALPHA)
        plt.legend(loc='lower left')
        plt.xlim([2E2, 2E5])
        plt.ylim([1E-6, 5E-1])
# plt.gcf().suptitle(
#     f"Endpoint error, integration interval " + ("$[0,L/\\epsilon]$" if INTERVAL_SCALED else "$[0,L]$") + "\n",
#     fontsize=15)
plt.tight_layout()
# plt.savefig('../../../_Export/ErrorCD/Satellite_EndpointError_onlyTwo.pdf')
plt.show()




# %% Experiment 3, plot all effort-error point
from Core.tools import linearInterpolation
plt.figure(figsize=(7, 5))

# errors = (epsilons, nMicroInts, Approximations, nMacroInts, MacroIntegrators)

for index_eps, epsilon in enumerate(epsilons_3):

    for index_int, integrator in enumerate(integrators_3):
        # plt.subplot(len(epsilons_3), len(integrators_3), index_eps * len(integrators_3) + index_int + 1)

        for index_nmacro, nmacro in enumerate(nMacroInts_3):
            for approximation_index, approximation in enumerate(approximations_3):
                # alpha = 0.3 if K % 2 == 1 else 1
                alpha = 0.1
                plt.loglog(efforts_3[index_eps, :, approximation_index, index_nmacro, index_int],
                           errors_3[index_eps, :, approximation_index, index_nmacro, index_int], "C0.-",
                           alpha=alpha, markersize=1.3)

            # plot the full

            # plt.title(
            #     f"$\\epsilon = ${floatTo10ExpStr(epsilon)} $\\rightarrow \\tau_f = {tau_f(epsilon)}$, $H = {tau_f(epsilon) / nMacro}$")

            # plt.title(
            #     f"$\\epsilon = ${floatToExpStr(epsilon, base=2)}, Integrator: {integrator}")

            # plt.xlim([2E2, 2E5])
            # plt.ylim([1E-6, 5E-1])
# plt.gcf().suptitle(
#     f"Endpoint error, integration interval " + ("$[0,L/\\epsilon]$" if INTERVAL_SCALED else "$[0,L]$") + "\n",
#     fontsize=15)
plt.loglog(efforts_full_3[index_eps, :], errors_full_3[index_eps, :], ".-", color='red', label=f'Full Simulation',
           alpha=1)

# index slices that are plotted bold
pareto_functions = []
labels = []
index_switch = 6
for index_approx, index_nmacro, index_int in [[0,0,1],[0,0,0]]:

    if index_int == 0:
        slice_pf = slice(0, index_switch)
    else:
        slice_pf = slice(index_switch, -3)

    x_vals = efforts_3[0, slice_pf, index_approx, index_nmacro, index_int]
    y_vals = errors_3[0, slice_pf, index_approx, index_nmacro, index_int]
    plt.loglog(x_vals,
               y_vals, f"C{index_int}.", alpha=1)

    # construct a linear interpolation
    pareto_functions.append(linearInterpolation(np.log10(efforts_3[0, :, index_approx, index_nmacro, index_int]),
                                                np.log10(errors_3[0, :, index_approx, index_nmacro, index_int])))
    labels.append(f'{approximations_3[index_approx].name} - {integrators_3[index_int]} - $N_\\mathrm{{macro}}$={nMacroInts_3[index_nmacro]}')

    # append constant function, that go to the right infinity for each data point
    for x_val, y_val in zip(x_vals,y_vals):
        x_inter = np.array([x_val,np.inf])
        y_inter = np.array([y_val,y_val])
        pareto_functions.append(linearInterpolation(np.log10(x_inter),np.log10(y_inter)))

# build the pareto front
x = ca.SX.sym('x')
y_pareto = ca.Function('y_pareto', [x], [-ca.norm_inf(ca.vertcat(*[-f(x) for f in pareto_functions]))])

x_pareto = np.linspace(1.795, 3.5, 1000)
# y_pareto = y_pareto(x_pareto).full().flatten()

index_switch = 6
switch_pareto = 2.56
x_pareto_1 = x_pareto[x_pareto<=switch_pareto]
x_pareto_2 = x_pareto[x_pareto>switch_pareto]

plt.loglog( np.power(10,x_pareto_1), np.power(10,y_pareto(x_pareto_1).full().flatten()), "C0-", label=labels[1], alpha=1)
plt.loglog( np.power(10,x_pareto_2), np.power(10,y_pareto(x_pareto_2).full().flatten()), "C1-", label=labels[0], alpha=1)


plt.xlabel("Effort in total number of micro-int steps")
plt.ylabel("Error")
plt.grid(alpha=PLT_GRID_ALPHA)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig('../../../_Export/ErrorCD/Satellite_EndpointError_Pareto.pdf')
plt.show()

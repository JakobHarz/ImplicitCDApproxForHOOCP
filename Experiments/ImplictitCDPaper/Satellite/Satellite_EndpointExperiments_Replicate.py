import logging
import uuid
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from casadi.tools import struct_symSX, entry

from Core.controlparameterization import ZeroControlParameterization
from Core.parameters import Parameters
from Core.tools import floatToExpStr, getCDCoefficients
from Core.figuretools import latexify, smoothPlotLimits
from Core.CycleImplementations.FixedPeriodShooting import CycleSingleShootingFixedPeriod
from Models.satellite2d import Satellite2D
from Core.configuration import PLT_GRID_ALPHA, IPOPT_OPTIONS_LINEAR_SOLVER

latexify()

# values to iterate over
INTERVAL_SCALED = True
# epsilons = [5E-4,1E-3,1E-2]
# epsilons = [1E-3, 0.5E-3, 1E-4]
epsilons = [2**(-12)]
# epsilons = [0.5E-2]

nMicroInts = np.logspace(1.5, 3, 15).astype(int)
# nMicroInts = [100]
L = 1/16  # final time
nMacroInts = [2, 8]

if INTERVAL_SCALED:
    tau_f = lambda epsilon: int(L / np.abs(epsilon))
else:
    tau_f = lambda epsilon: int(L * 100)

# central difference approximations
Kmax = 4
Ks = np.arange(2, Kmax + 1)

additionalSolverOptions = {'ipopt.tol': 1e-14,
                           'ipopt.acceptable_tol': 1e-14}  # low tolerances for the solver


def rk4Step(xk, cycleSim, K, H) -> np.ndarray:
    h = H
    # print(f'RK4 step from {xk} with h = {h}')
    # rk4 integration step
    k1 = computeDynamicsApprox(xk, cycleSim, K)
    k2 = computeDynamicsApprox(xk + h / 2 * k1, cycleSim, K)
    k3 = computeDynamicsApprox(xk + h / 2 * k2, cycleSim, K)
    k4 = computeDynamicsApprox(xk + h * k3, cycleSim, K)

    xend = xk + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return xend


def computeEndpointCD(x0bar, cycleSim, K, nMacro, tau_f) -> np.ndarray:
    xstart = x0bar
    H = tau_f / nMacro
    # print(f'H = {H}, nMacro = {nMacro}')
    for i in range(nMacro):
        xend = rk4Step(xstart, cycleSim, K, H)
        xstart = xend

    return xend


solvers = {}


def computeDynamicsApprox(xlocal, cycleSim, K) -> np.ndarray:
    # try to get the solver from the dict
    solver_and_ouputF = solvers.get(cycleSim.id, {}).get(K, None)

    if solver_and_ouputF is None:
        # print(f'Creating solver for K = {K}, cyclSim id = {cycleSim.id}')

        Ncycles = K - 1
        Z = struct_symSX([entry("z", struct=cycleSim.z, repeat=Ncycles)])
        Z0 = Z(0)
        Z0['z'] = cycleSim.z0(xlocal)
        lbZ = Z(-ca.inf)
        lbZ['z'] = cycleSim.lbz
        ubZ = Z(ca.inf)
        ubZ['z'] = cycleSim.ubz

        # initial values and bounds
        # get the coefficients
        taus, b, c = getCDCoefficients(K)
        b = ca.DM(b)
        c = ca.DM(c)

        # create the variables

        x0 = model.x_struct  # parameter for the solver

        # array for the conditions
        conditions = []

        # connect the cycles
        for k in range(Ncycles - 1):
            conditions.append(Z['z', k, 'x_plus'] - Z['z', k + 1, 'x_minus'])

        # simulate the cycles
        for k in range(Ncycles):
            conditions.append(cycleSim.f_cycleConditions(Z['z', k], []))

        Xks = [Z['z', k, 'x_minus'] for k in range(Ncycles)] + [Z['z', Ncycles - 1, 'x_plus']]
        Xks = ca.horzcat(*Xks)

        ## poly constraint for x
        conditions.append(x0 - Xks @ b)  # last row is for constant term c_0, that should be equal to x0

        f_polyDer = ca.Function('f_polyDer', [Z], [Xks @ c])
        supressOutput = True
        # Create an NLP solver
        problem = {'f': 0, 'x': Z, 'g': ca.vertcat(*conditions), 'p': x0}
        options = {'ipopt.linear_solver': IPOPT_OPTIONS_LINEAR_SOLVER,
                   'ipopt.max_iter': 2000,
                   'ipopt.print_level': 0 if supressOutput else 5,
                   'print_time': False if supressOutput else True,
                   }

        # options.update(additionalSolverOptions)
        solver = ca.nlpsol('solver', 'ipopt', problem, options)

        # save the solver
        if solvers.get(cycleSim.id, None) is None:
            solvers[cycleSim.id] = {}

        solvers[cycleSim.id][K] = (solver, f_polyDer, Z0)

    else:
        # print(f' --- Retreiving solver for K = {K}, cyclSim hash = {cycleSim.id}')
        pass

    solver, f_polyDer, Z0 = solvers[cycleSim.id][K]

    # Solve the NLP
    nlpsolution = solver(x0=Z0, lbx=-ca.inf, ubx=ca.inf, lbg=0, ubg=0, p=xlocal)

    if solver.stats()['return_status'] not in ['Solve_Succeeded', 'Search_Direction_Becomes_Too_Small']:
        print(f'Solver failed with status {solver.stats()["return_status"]}')

    return f_polyDer(nlpsolution['x']).full()


# %%%
errors = np.zeros((len(epsilons), len(nMicroInts), len(Ks), len(nMacroInts)))
efforts = np.zeros((len(epsilons), len(nMicroInts), len(Ks), len(nMacroInts)))

errors_full = np.zeros((len(epsilons), len(nMicroInts)))
efforts_full = np.zeros((len(epsilons), len(nMicroInts)))

# iterate over epsilons
for index_eps, epsilon in enumerate(epsilons):
    print(f'epsilon = {epsilon}')
    model = Satellite2D(Parameters(epsilon=epsilon))

    x0bar = model.x_struct(0)
    x0bar["p"] = [1, 0]
    x0bar["v"] = [0, 1]
    x0bar = x0bar.cat

    # perform 'true' simulation
    logging.info(f'Solving true solution for epsilon = {epsilon} ... ')
    cycleSim_true = CycleSingleShootingFixedPeriod(model, Nint=1000, controlParameterization=ZeroControlParameterization())
    ZoptTrue = cycleSim_true.solveNCycles(x0bar, N=tau_f(epsilon),
                                          supressOutput=True, additionalSolverOptions=additionalSolverOptions)
    xend_true = ZoptTrue[-1]['x_plus'].full()
    logging.info(f'... done!')

    # iterate over number of micro-integrations
    for index_nMu, nMicroInt in enumerate(nMicroInts):

        print(f'nMicroInt = {nMicroInt}')
        # cycleSimulationClass for this configuration
        # startConditions = LinearStartConditions(q=ca.DM([0, 0, 1]), b=ca.DM(0))
        cycleSim_forward = CycleSingleShootingFixedPeriod(model, Nint=nMicroInt, controlParameterization=ZeroControlParameterization())
        cycleSim_forward.id = uuid.uuid4()

        # compute the endpoint full integration
        Zoptfull = cycleSim_forward.solveNCycles(x0bar, N=tau_f(epsilon),
                                                 supressOutput=True,
                                                 additionalSolverOptions=additionalSolverOptions)
        xend_full = Zoptfull[-1]['x_plus'].full()
        errors_full[index_eps, index_nMu] = np.linalg.norm(np.diag([1, 1, 1, 1] @ (xend_true - xend_full)))
        efforts_full[index_eps, index_nMu] = nMicroInt * tau_f(epsilon)

        # iterate over tau_f
        for index_nMacro, nMacro in enumerate(nMacroInts):

            # iterate over K
            for index_K, K in enumerate(Ks):
                # true endpoint

                print(
                    f'epsilon = ({index_eps + 1}|{len(epsilons)}), nMicroInt = ({index_nMu + 1}|{len(nMicroInts)}), K = ({index_K + 1}|{len(Ks)}), nMacro = ({index_nMacro + 1}|{len(nMacroInts)})',
                    end='\r', flush=True)

                xend = computeEndpointCD(x0bar, cycleSim_forward, K, nMacro, tau_f(epsilon))

                error = np.linalg.norm(np.diag([1, 1, 1, 1]) @ (xend - xend_true))
                errors[index_eps, index_nMu, index_K, index_nMacro] = error
                effort = (K - 1) * nMicroInt * 4 * nMacro  # 4 due to rk4
                efforts[index_eps, index_nMu, index_K, index_nMacro] = effort

# %%
plt.figure(figsize=(len(nMacroInts) * 4.5,4 * len(epsilons)))

for index_eps, epsilon in enumerate(epsilons):
    for index_nMacro, nMacro in enumerate(nMacroInts):
        plt.subplot( len(epsilons),len(nMacroInts),  index_eps * len(nMacroInts) + index_nMacro + 1)

        for index_K, K in enumerate(Ks):
            alpha = 0.3 if K % 2 == 1 else 1
            plt.loglog(efforts[index_eps, :, index_K, index_nMacro], errors[index_eps, :, index_K, index_nMacro], ".-",
                       label=f'K = {K}', alpha=alpha)

        # plot the full
        plt.loglog(efforts_full[index_eps, :], errors_full[index_eps, :], ".-", color='red', label=f'Full', alpha=1)

        # plt.title(
        #     f"$\\epsilon = ${floatTo10ExpStr(epsilon)} $\\rightarrow \\tau_f = {tau_f(epsilon)}$, $H = {tau_f(epsilon) / nMacro}$")

        plt.title(
            f"$\\epsilon =\,${floatToExpStr(epsilon, base=2)}, $N_\\mathrm{{macro}} = {nMacro}$")
        plt.xlabel("Effort in total number of micro-int steps")
        plt.ylabel("Error")
        plt.grid(alpha=PLT_GRID_ALPHA)
        plt.legend(loc='lower left')
        plt.xlim([2E2, 2E5])
        plt.ylim([1E-6, 5E-1])
# plt.gcf().suptitle(
#     f"Endpoint error, integration interval " + ("$[0,L/\\epsilon]$" if INTERVAL_SCALED else "$[0,L]$") + "\n",
#     fontsize=15)
plt.tight_layout()
plt.savefig('../../../_Export/ErrorCD/Satellite_EndpointError_onlyTwo.pdf')
plt.show()

# %% Visualize true solution

tauCycle = np.linspace(0, 1, 50)
XsimPlot = []
tauPlot = []
XStrobo = [x0bar]
for index_k, zopt_k in enumerate(ZoptTrue):
    print(f'Reconstructing cycle {index_k + 1}/{len(ZoptTrue)}', end='\r', flush=True)
    XsimPlot.append(
        cycleSim_true.f_outputMap.map(tauCycle.size)(tauCycle, zopt_k, ca.DM.zeros(cycleSim_true.U.shape)).full())
    XStrobo.append(zopt_k['x_plus'].full())
    tauPlot.append(tauCycle + index_k)

    # if index_k == len(ZoptTrue)-1:
    # XsimPlot.append(zopt_k['x_plus'].full())
    # tauPlot.append(np.array([index_k+1]))
XsimPlot_cct = np.concatenate(XsimPlot, axis=1)
XStrobo = np.concatenate(XStrobo, axis=1)
tauPlot = np.concatenate(tauPlot, axis=0)

# %% plot
plt.figure()
for stateIndex in range(2):
    plt.subplot(2, 1, stateIndex + 1)
    plt.plot(tauPlot, XsimPlot_cct[stateIndex, :])
    plt.plot(np.arange(0, tau_f(epsilon) + 1), XStrobo[stateIndex, :], '.-')
    plt.grid(alpha=PLT_GRID_ALPHA)
plt.show()

plt.figure()
for index_k, xsim_k in enumerate(XsimPlot):
    plt.plot(xsim_k[0, :], xsim_k[1, :], '-', color='C0', alpha=(index_k + 1) / len(XsimPlot))
plt.plot(XStrobo[0, :], XStrobo[1, :], 'C1.-')
plt.plot(0, 0, 'ro')
plt.axis('equal')
plt.show()

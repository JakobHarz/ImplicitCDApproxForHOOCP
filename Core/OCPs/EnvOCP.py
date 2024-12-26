import logging

module_logger = logging.getLogger('')

from typing import List

import casadi as ca
import numpy as np
from casadi.tools import struct_symSX, entry

from Core.OCPs.discreteOCP import NCycleOCP, OCPSolution, DiscreteOCP
from Core.cycle import Cycle
from Core.macrodiscretizationschemes import MacroDiscretizationScheme
from Core.hotrajectory import HighlyOscillatoryTrajectory
from Core.integrators import Collocation, LegendreCollocation


class DiscreteEnvelopeOCP(DiscreteOCP):
    #######################
    ##### Constructor #####
    #######################

    def __init__(self,
                 ocp: NCycleOCP,
                 cycleSimClass: type(Cycle),
                 macroDisScheme: MacroDiscretizationScheme,
                 supressOutput: bool = False,
                 regEpsilon=None,
                 regWeightMatrix=None,
                 dControl=1
                 ):
        """
        Class for a discretized Envelope-OCP that can be solved.
        The given continuos time N-Cycle ocp is discretized in the implemented method construct()
        and solved using the NLP solver IPOPT in the method solve().

        :param ocp: The continuous time N-Cycle OCP
        :param cycleSimClass: The class of the cycle simulation that is used to simulate the cycle
        :param macroDisScheme: The macro-discretization scheme that is used to discretize the DAE
        :param supressOutput: If true, suppress the output of IPOPT
        """

        wEntries = [
            (
                entry("controls", shape=(cycleSimClass.nu, cycleSimClass.Nctrl),
                      repeat=[macroDisScheme.Nintervals,
                              dControl]),
                entry("Xcoll", shape=cycleSimClass.model.x.shape,
                      repeat=[macroDisScheme.Nintervals, macroDisScheme.integrationScheme.d]),
                # collocation state nodes
                entry("Vcoll", shape=cycleSimClass.model.x.shape,
                      repeat=[macroDisScheme.Nintervals, macroDisScheme.integrationScheme.d]),
                # collocation state derivative nodes
                entry("Zcoll", struct=cycleSimClass.z,
                      repeat=[macroDisScheme.Nintervals, macroDisScheme.integrationScheme.d])
                # collocation algebraic state nodes
            ),
            entry("Xe", shape=cycleSimClass.model.x.shape, repeat=macroDisScheme.Nintervals + 1)  # ms state nodes
        ]

        super().__init__(ocp, cycleSimClass, wEntries)

        self.macroDisScheme: MacroDiscretizationScheme = macroDisScheme
        """ The settings for the discretization scheme of the macro-integration, has to be specified before construct() """

        self.supressOutput = supressOutput
        """ If true, suppress the output of IPOPT """

        self.model = ocp.model

        self._solution: EnvOCPSolution = None  # Not Solved Yet

        # control parametrization?
        self.dControl: int = dControl
        """number of controls for one macro-integration interval"""

        # regularization?
        self._usesRegularization = regEpsilon is not None
        self.regEpsilon = regEpsilon
        self.regWeightMatrix = regWeightMatrix
        if self.regWeightMatrix is None:
            self.regWeightMatrix = np.eye(self.model.nx)

        # sanity checks for ocp:
        self.ocp.runSanityChecks()

    #######################
    ##### Methods #########
    #######################

    def constructNLP(self, parameters=ca.vertcat([])):

        self.runSanityChecks()

        logging.info("Constructing NLP ...")

        ##########################
        ### Preparation ####
        ##########################

        # N = self.ocp.tf # total number of cycles
        dMacro = self.macroDisScheme.integrationScheme.d

        # initial value and bounds for cycle controls
        U0 = ca.DM.zeros((self.cycleSimClass.nu, self.cycleSimClass.Nctrl))
        lbU = np.tile(self.model.lbu.cat, self.cycleSimClass.Nctrl)
        ubU = np.tile(self.model.ubu.cat, self.cycleSimClass.Nctrl)

        x0bar = self.ocp.x0bar

        # Collocation integration here: # TODO: support other schemes
        RKInt = self.macroDisScheme.integrationScheme
        c, A, b = RKInt.unpackButcher()
        self._c = c

        # collocation scheme for controls, with points at chebyshev nodes
        # self.controlRKInt = ChebyshevCollocation(self.dControl)
        self.controlRKInt = LegendreCollocation(self.dControl)

        #########################
        ### BUild NLP ###########
        #########################

        g = []
        lbg = []
        ubg = []
        self.X_ret = []
        self.Z_ret = []
        self.U_ret = []
        self.T_ret = []
        tau_macro_ret = []  # array of all taus of the integration points of the macro integration
        tau_micro_ret = []  # list of all micro-integration taus

        self.J_int = 0  # cumulative cost function
        self.J_final = 0  # final cost
        self.J_reg = 0  # regularization cost

        # "Lift" initial conditions
        X0 = self.w["Xe", 0]
        self.X_ret.append(X0)
        tau_macro_ret.append(0)
        self.lbw["Xe", 0] = self.model.lbx
        self.ubw["Xe", 0] = self.model.ubx
        self.w0["Xe", 0] = x0bar

        # initial conditions
        g.append(X0 - x0bar)
        lbg.append(np.zeros(self.model.x.shape))
        ubg.append(np.zeros(self.model.x.shape))

        # iterate stages
        for k in range(self.macroDisScheme.Nintervals):

            # duration of this stage (in tau)
            tau_stage_duration = self.macroDisScheme.tauf_stages

            # construct polynomial function for controls
            U_stage_reformatted = ca.horzcat(
                *[ca.reshape(self.w["controls", k, m], (self.cycleSimClass.U.numel(), 1)) for m in
                  range(self.dControl)])
            U_poly_stage = self.controlRKInt.getPolyEvalCAExpression(self.cycleSimClass.U.numel())

            # Set bounds and initilization for controls
            for m in range(self.dControl):
                self.w0["controls", k, m] = U0
                self.lbw["controls", k, m] = lbU
                self.ubw["controls", k, m] = ubU
                self.U_ret.append(self.w["controls", k, m])

            # preparation
            _slice = slice(0, dMacro)  # slice for the variables of this stage

            # State, Derivative and Algebraic Variables at collocation points
            Xs = self.w["Xcoll", k, _slice]
            Vs = self.w["Vcoll", k, _slice]
            Zs = self.w["Zcoll", k, _slice]

            # add to return function
            self.X_ret.extend(Xs)
            self.Z_ret.extend(Zs)
            tau_macro_ret.extend((tau_stage_duration * (k + c)))

            # bounds
            self.lbw["Xcoll", k, _slice] = self.model.lbx
            self.ubw["Xcoll", k, _slice] = self.model.ubx

            self.lbw["Vcoll", k, _slice] = self.model.lbv  # TODO: this looks wrong
            self.ubw["Vcoll", k, _slice] = self.model.ubv  # TODO: this looks wrong

            self.lbw["Zcoll", k, _slice] = self.cycleSimClass.lbz
            self.ubw["Zcoll", k, _slice] = self.cycleSimClass.ubz



            # linear interpolation to find start and endpoint
            def linInter(n):
                # return (self.ocp.x0bar.cat * (self.macroDisScheme.Nintervals + 1 - n) + self.ocp.xendguess.cat * n) / (
                #         self.macroDisScheme.Nintervals + 1)
                x0 = self.ocp.x0bar.cat
                xend = self.ocp.xendguess.cat
                return x0 + (xend - x0) * n / (self.macroDisScheme.Nintervals)

            x0stage = linInter(k)
            xendstage = linInter(k + 1)

            # create Integration Variables
            for i in range(dMacro):
                # interpolate to find initial guess for the state at the collocation point
                xguess_macro = x0stage + (xendstage - x0stage) * c[i]  # TODO: WRONG!

                # project the guess to satisfy the phase conditions
                xguess_micro_minus = self.cycleSimClass.projectInitialGuess(xguess_macro)
                xguess_micro_plus = self.cycleSimClass.projectInitialGuess(xguess_macro, target='plus')

                # log the values
                logging.info(f"Guess for x at collocation point {i} of stage {k}: {xguess_macro}")

                # differential variables
                self.w0["Xcoll", k, i] = ca.DM(xguess_macro)
                self.w0["Vcoll", k, i] = xguess_micro_plus - xguess_micro_minus  # TODO: Think about this
                self.w0["Zcoll", k, i] = self.cycleSimClass.z0(xguess_micro_minus)

            # Connect Integration Variables
            Xc_end = X0
            for i in range(dMacro):
                x = Xs[i]
                z = self.cycleSimClass.z(Zs[i])
                v = Vs[i]

                # get the control, by evaluating the polynomial and then reshaping it to the right shape
                U = ca.reshape(U_poly_stage(c[i], U_stage_reformatted), self.cycleSimClass.U.shape)

                # we have to define this somewhere, since we don't have a DAE class anymore
                self.dae_shift = -0.5

                # Append dynamic equations of DAE
                g.append(tau_stage_duration * (z['x_plus'] - z['x_minus']) - v)
                lbg.append(np.zeros(self.model.x.shape))
                ubg.append(np.zeros(self.model.x.shape))

                ## Append Algebraic Equations of DAE
                g.append(self.cycleSimClass.f_cycleConditions(z, U))
                g.append(z['x_plus'] + z['x_minus'] - 2 * x)
                lbg.append(np.zeros(z.shape))
                ubg.append(np.zeros(z.shape))

                # equation for rk
                xj_bar = X0
                for j in range(dMacro):
                    xj_bar = xj_bar + A[i, j] * Vs[j]
                g.append(xj_bar - x)
                lbg.append(np.zeros(self.model.x.shape))
                ubg.append(np.zeros(self.model.x.shape))

                # Integrate stage cost
                self.J_int += tau_stage_duration * b[i] * self.cycleSimClass.f_cycleCost(z, U)

                # Add contribution to the end state
                Xc_end = Xc_end + b[i] * Vs[i]

            # New NLP variable for state at end of interval
            Xend = self.w["Xe", k + 1]
            self.X_ret.append(Xend)
            tau_macro_ret.append((k + 1) * tau_stage_duration)
            self.lbw["Xe", k + 1] = self.model.lbx
            self.ubw["Xe", k + 1] = self.model.ubx
            self.w0["Xe", k + 1] = xendstage

            # Add equality constraint
            g.append(Xc_end - Xend)
            lbg.append(np.zeros(self.model.x.shape))
            ubg.append(np.zeros(self.model.x.shape))

            X0 = Xend

            # add regularization of this interval
            if self._usesRegularization:
                for i in range(dMacro):
                    # slope derivative
                    polydervals = np.array([l.deriv(2)(c[i]) for l in RKInt.polynomials])
                    Vi_dot = ca.horzcat(*Vs) @ polydervals
                    Vi_dot_squared = Vi_dot.T @ self.regWeightMatrix @ Vi_dot
                    self.J_reg = self.J_reg + self.regEpsilon * b[i] * Vi_dot_squared

                # tau_SX = ca.SX.sym("tau_SX")
                # Vcoll_SX = ca.SX.sym("Vcoll_SX", (self.model.nx, self.macroDisScheme.integrationScheme.d))
                # _envelopeDerivativePoly = self.macroDisScheme.integrationScheme.getPolyEvalCAExpression(self.model.nx)(
                #     tau_SX, Vcoll_SX)
                # _envelopeSecondDerivativePoly = ca.Function("envelopeSecondDerivativePoly", [tau_SX, Vcoll_SX],
                #                                             [ca.jacobian(_envelopeDerivativePoly, tau_SX)])
                # # endpoint of second derivative is equal to the integral of the third derivative
                # endpointSecondDerivative = _envelopeSecondDerivativePoly(self.ocp.N,
                #                                                          ca.horzcat(*self.w["Vcoll",k, _slice]))
                # self.J_reg += self.regEpsilon * endpointSecondDerivative.T @ self.regWeightMatrix @ endpointSecondDerivative

        ##############################
        # Final Cost and Constraints #
        ##############################

        # final cost
        self.J_final = self.ocp.finalCostFunction(Xend)

        # final constraint
        g_term, lbg_term, ubg_term = self.ocp.terminalConstraints(Xend)
        g.append(g_term)
        lbg.append(lbg_term)
        ubg.append(ubg_term)

        # store in attributes (nasty)
        self.p = parameters
        self.g = g
        self.lbg = lbg
        self.ubg = ubg

        self.Xend = Xend
        self.tau_macro_ret = tau_macro_ret
        self.tau_micro_ret = tau_micro_ret

    def finalizeNLPConstruction(self):
        # Concatenate vectors

        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)

        # Function to retrieve the solution
        self.f_getXopt = ca.Function("getXopt", [self.w, self.p], [ca.horzcat(*self.X_ret)])
        # self.f_getXopt_oneCycle = ca.Function("getXopt", [self.dae.z], [ca.horzcat(*self.dae.z_x)])
        self.f_getUopt = ca.Function("getUopt", [self.w, self.p], [ca.vertcat(*self.U_ret)])
        self.f_getZopt = ca.Function("getZopt", [self.w, self.p], [ca.horzcat(*self.Z_ret)])
        self.f_getTopt = ca.Function("getTopt", [self.w, self.p], [ca.vertcat(*self.T_ret)])
        self.f_getJintopt = ca.Function("getJintopt", [self.w, self.p], [self.J_int])
        self.f_getJfinopt = ca.Function("getJfinopt", [self.w, self.p], [self.J_final])
        self.f_getJregopt = ca.Function("getJfinopt", [self.w, self.p], [self.J_reg])
        self.f_getJtotalopt = ca.Function("getJtotalopt", [self.w, self.p], [self.J_final + self.J_int])

        self.f_tau_macro_ret: ca.Function = ca.Function("getTauMacro", [self.w, self.p],
                                                        [ca.vertcat(*self.tau_macro_ret)])

        # cost function
        self.f = self.J_int + self.J_final + self.J_reg

    def formateOCPResult(self, nlpsolution: dict, params, solverStats: DiscreteOCP.SolverStats, solverStatsDict) -> 'EnvOCPSolution':

        logging.debug("Formating OCP result...")

        wopt = nlpsolution['x']
        solution = EnvOCPSolution()  # empty solution class

        # reference optimal control problem
        solution.ocp = self.ocp

        # store solution
        solution.Xopt = Xopt = self.f_getXopt(wopt, params).full()
        solution.wopt = self.w(wopt)

        # split return matrix of size (nu*?,Nctrl) into a list of ? matrices of shape (nu,Nctrl)
        # _Uopt = self.f_getUopt(wopt, params).full()
        solution.Uopt = solution.wopt["controls"]
        logging.debug(f"solution.Uopt: len:{len(solution.Uopt)}")

        solution.Zopt = struct_symSX(self.cycleSimClass.z_entries).repeated(self.f_getZopt(wopt, params).full())
        # solution.Topt = ca.vertcat(*solution.Zopt[:, 'T']).full()
        solution.Jopt = self.f_getJtotalopt(wopt, params).full()
        solution.Jintopt = self.f_getJintopt(wopt, params).full()
        solution.Jfinopt = self.f_getJfinopt(wopt, params).full()
        solution.Jregopt = self.f_getJregopt(wopt, params).full()

        solution.solverStats = solverStats
        solution.solverStats_dict = solverStatsDict


        # store additional helpful values
        solution.tau_macro = self.f_tau_macro_ret(wopt, params).full()

        solution.plotting_cycleTrajs = []
        solution.plotting_cycleTaus = []

        _tauSingleStage = np.linspace(0, self.macroDisScheme.tauf_stages, 100)
        _tauStages = [self.macroDisScheme.tauf_stages * k + _tauSingleStage for k in range(self.macroDisScheme.Nintervals)]

        _envelopePolys = []

        # reconstruct micro integration time grid
        solution.tau_micro = []
        for k in range(self.macroDisScheme.Nintervals):

            # duration in tau
            tau_stage_duration = self.macroDisScheme.tauf / self.macroDisScheme.Nintervals

            # construct polynomial function for controls
            _sigma = ca.SX.sym("tau")
            U_stage_reformatted = ca.horzcat(
                *[ca.reshape(solution.wopt["controls", k, m], (self.cycleSimClass.U.numel(), 1)) for m in
                  range(self.dControl)])
            U_poly_stage = ca.Function("U_poly_stage", [_sigma], [
                self.controlRKInt.getPolyEvalCAExpression(self.cycleSimClass.U.numel())(_sigma, U_stage_reformatted)])

            for i in range(self.macroDisScheme.integrationScheme.d):
                _ind = i + k * self.macroDisScheme.integrationScheme.d
                _control = ca.reshape(U_poly_stage(self.macroDisScheme.integrationScheme.c[i]),
                                      self.cycleSimClass.U.shape)
                solution.tau_micro.append(
                    self.cycleSimClass.f_getLocalTau(solution.Zopt[_ind], _control).full()
                    + (k + self.macroDisScheme.integrationScheme.c[i]) * tau_stage_duration
                    + self.dae_shift)

            for i in range(self.macroDisScheme.integrationScheme.d):  # iterate cycles

                # convert Zopt to state trajectory
                _ZoptCycle = solution.Zopt[i + k * self.macroDisScheme.integrationScheme.d]
                _control = ca.reshape(U_poly_stage(self.macroDisScheme.integrationScheme.c[i]),
                                      self.cycleSimClass.U.shape)
                _cycleSim = self.cycleSimClass.f_getX(_ZoptCycle, _control).full()
                solution.plotting_cycleTrajs.append(_cycleSim)

            _envelopePolys.append(Collocation.evaluatePolynomial(
                Xopt[:, k * (self.macroDisScheme.integrationScheme.d + 1):(k + 1) * (
                        self.macroDisScheme.integrationScheme.d + 1)], _tauSingleStage,
                np.concatenate([[0], self.macroDisScheme.integrationScheme.c]) * tau_stage_duration))

        solution.plotting_tauCont = np.hstack(_tauStages)
        solution.plotting_envCont = np.hstack(_envelopePolys)
        # solution.plotting_Xopt_Tau = np.concatenate([_N * n + np.concatenate([[0], self._collTimes]) * _N  for n in range(self.macroDisScheme.Nctrl)] + [np.array([self.macroDisScheme.Nctrl*_N])])

        # add optimal nlp solution to problem
        solution.nlpsolution = nlpsolution

        # store hessian of lagrangian
        # logging.info("Calculating Hessian of Lagrangian...")
        # lagrangian_hess_fun = self.solver.get_function('nlp_hess_l')
        #
        # hess_opt = lagrangian_hess_fun(nlpsolution['x'],
        #                                params,
        #                                1,
        #                                nlpsolution['lam_g']).full()
        #
        # solution.lagrangian_hessian = hess_opt
        # logging.info("Done.")

        solution.macroDisScheme = self.macroDisScheme

        # log some information
        logging.info(f"Optimal Objective: {solution.Jopt}")
        logging.info(f"- StageCost: {solution.Jintopt}")
        logging.info(f"- TerminalCost: {solution.Jfinopt}")
        logging.info(f"- Regularization Cost: {solution.Jregopt}")

        return solution

    def runSimulation(self, solution: 'EnvOCPSolution', NintPerCtrl=50) -> HighlyOscillatoryTrajectory:
        """
        Simulate all N Cycles with the given solution

        :param solution:  solution of the optimal control problem
        :param NintPerCtrl: number of integration steps per control
        :param NoutPerCtrl: number of output states per control
        :return:
        """

        # assert self._solution is not None
        # check that N is an integer
        assert self.ocp.tf == int(self.ocp.tf), "N must be an integer!"

        sol = solution

        x0stage = self.ocp.x0bar

        # empty trajectory to store result
        FullTrajectory = HighlyOscillatoryTrajectory()

        # iterate stages
        for k in range(self.macroDisScheme.Nintervals):

            # multiple controls per stage? prepare controls
            U_stage = solution.wopt["controls", k, :]
            _sigma = ca.SX.sym("_sigma")
            U_poly_stage = ca.Function("U_poly_stage", [_sigma], [
                self.controlRKInt.getPolyEvalFunction(self.cycleSimClass.U.shape, fixedValues=U_stage)(_sigma)])

            # evaluate the polynomial for each cycle
            _Uopt_stage = []

            # check if N is an integer
            tau_stages = self.macroDisScheme.tauf_stages

            for n in range(tau_stages):
                _Uopt_stage.append(U_poly_stage((n + 0.5) / tau_stages))

            # create cycle sim class with high accuracy for simulation
            cycleSimClass = self.cycleSimClass.getHigherAccuracyCopy()

            # simulate the cycles for this stage
            HOTrajectory = cycleSimClass.simulateNCycles(x0stage, tau_stages)
            x0stage = HOTrajectory.X[:, [-1]]
            # x0stage = x0stage - phaseconditions.q @ (phaseconditions.bplus - phaseconditions.bminus)  # from x+ to x-
            x0stage = cycleSimClass.projectInitialGuess(x0stage)

            FullTrajectory += HOTrajectory

        return FullTrajectory

    def plotSolution(self, timeScale='physical'):
        raise NotImplementedError()

    def runSanityChecks(self):
        """ Sanity checks to see if the OCP is initialized properly."""

        logging.info("Performing sanity checks for discrete envelope OCP ... ")

        assert self.macroDisScheme is not None, "Macro Scheme has to be defined!"
        assert self.ocp.stageCostFunction is not None
        assert self.ocp.finalCostFunction is not None
        assert self.ocp.x0bar is not None
        assert self.ocp.xendguess is not None
        # assert self.ocp.DEPRIVATED_phaseConditions is not None

        assert self.ocp.tf > 0

        # only collocation schemes for macro-integration
        assert isinstance(self.macroDisScheme, MacroDiscretizationScheme)
        assert isinstance(self.cycleSimClass, Cycle)

        logging.info("... Done!")


class EnvOCPSolution(OCPSolution):
    """
    Class to store and post process the solution of an Envelope OCP
    """

    def __init__(self):
        super().__init__()

        self.Xopt: np.ndarray = None
        """ The macro-integration state trajectory"""
        self._Zopt: np.ndarray = None

        self.Uopt: List[np.ndarray] = None
        """ An list of U matrices, each of shape (nu,Nctrl)"""

        # self.Topt: np.ndarray = None
        # """ WHAT IS THIS?"""

        self.macroDisScheme: MacroDiscretizationScheme = None
        """ The macro-integration scheme"""

        self.plotting_cycleTrajs: List[np.ndarray] = None
        """ A list of all micro-integration trajectories"""

        self.plotting_cycleTaus: List[np.ndarray] = None
        """ A list of all micro-integration tau grids"""

        self.plotting_tauCont: np.ndarray = None
        """ For the plotting of the continous Envelope Poly, a grid in numerical time"""

        self.plotting_envCont: np.ndarray = None
        """ The Continuous Envelope Poly, a grid of Values"""

        self.tau_macro: np.ndarray = None
        """ The numerical times of the macro-integration collocation points"""
        self.tau_micro: List[np.ndarray] = None
        """ The numerical times of each of the micro-integrations """

    def plotState(self, timeScale='physical', simulation: HighlyOscillatoryTrajectory = None):
        """
        plots the found solution for each state.
        Assumes the last state to be the timestate
        """
        self.checkInitialized()

        assert timeScale in ['physical', 'numerical']

        logging.info("Plotting Solution")
        import matplotlib.pyplot as plt

        # TODO: remove unpacking
        model = self.ocp.model

        nx = self.ocp.model.nx

        plt.figure("Solution by State", figsize=(10, 2 * (nx + 1) // 2))

        # iterate states
        for j in range(self.ocp.model.nx):
            plt.subplot(2, (nx + 1) // 2, j + 1)

            # time_grid = sol.Xopt[-1,:] if timeScale == 'physical'
            if timeScale == 'physical':
                # plot state envelope in blue
                plt.plot(self.Xopt[-1, :], self.Xopt[j], 'C0o')
                plt.plot(self.plotting_envCont[-1], self.plotting_envCont[j], 'C0-')

                # plot micro-integrations in green
                for cycleTraj in self.plotting_cycleTrajs:
                    plt.plot(cycleTraj[-1], cycleTraj[j], 'C2-')

                # plot simulation in orange
                if simulation is not None:
                    plt.plot(simulation.X[-1], simulation.X[j], 'C1--', alpha=0.5, label='Simulation')

                plt.xlabel(r"Physical Time $t$")

            if timeScale == 'numerical':
                # plot micro-integrations in green
                for cyleTau, cycleTraj in zip(self.tau_micro, self.plotting_cycleTrajs):
                    plt.plot(cyleTau, cycleTraj[j], 'C2-')

                # plot state envelope in blue
                plt.plot(self.tau_macro, self.Xopt[j], 'C0o')
                plt.plot(self.plotting_tauCont, self.plotting_envCont[j], 'C0-')

                # plot simulation in orange
                if simulation is not None:
                    plt.plot(simulation.tau, simulation.X[j], 'C1--', alpha=0.5, label='Simulation')

                plt.xlabel(r"Numerical Time $\tau$")

            # legend entries
            plt.plot([], [], 'C0o-', label='Envelope')
            plt.legend()

            # other formatting
            plt.ylabel(model.stateLabels[j])
            plt.grid()

        plt.tight_layout()
        plt.show()

    def plotControls(self, timeScale='physical'):
        """
        plots the found solution for each state.
        Assumes the last state to be the timestate
        """
        assert timeScale in ['physical', 'numerical']
        timeScale = 'numerical'

        logging.info("Plotting Solution")
        import matplotlib.pyplot as plt

        # TODO: remove unpacking
        model = self.ocp.model

        nu = self.ocp.model.nu
        plt.figure("Solution by Control", figsize=(10, 2 * (nu)))

        # iterate controls
        for j in range(nu):
            # iterate stages
            for i in range(self.macroDisScheme.Nintervals):
                # new subplot
                plt.subplot(nu, self.macroDisScheme.Nintervals, self.macroDisScheme.Nintervals * j + 1 + i)
                plt.gca().set_title(f"Stage {i + 1}")

                # time_grid = sol.Xopt[-1,:] if timeScale == 'physical'
                if timeScale == 'physical':
                    raise NotImplementedError("Physical time not implemented for controls")

                if timeScale == 'numerical':
                    # plot controls
                    # raise NotImplementedError("Physical time not implemented for controls")
                    assert len(self.Uopt[0]) == 1, "Only a constant control per stage supported for plotting"
                    plt.stairs(self.Uopt[i][0][j, :].full()[0], np.linspace(0, 1, self.Uopt[i][0].shape[1] + 1),
                               label=model.controlLabels[j])

                    # plot simulation in orange
                    plt.xlabel(r"Numerical Time $\tau$")

                # legend entries
                plt.plot([], [], 'C0o-', label='Envelope')
                plt.legend()

                # other formatting
                plt.ylabel(model.controlLabels[j])
                plt.grid()

        plt.tight_layout()
        plt.show()

    @property
    def Zopt(self):
        """The macro-integration algebraic variables"""
        return self._Zopt

    @Zopt.setter
    def Zopt(self, value):
        self._Zopt = value

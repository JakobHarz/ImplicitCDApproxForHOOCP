import numpy as np
import casadi as ca
import logging
from Core.OCPs.discreteOCP import DiscreteOCP
from Core.integrators import LegendreCollocation
from casadi.tools import entry

from Core.OCPs.discreteOCP import NCycleOCP, OCPSolution
from Core.tools import numericalToPhysicalTime
from Core.hotrajectory import HighlyOscillatoryTrajectory
from Core.cycle import Cycle


class NCcyleOCPSolution(OCPSolution):
    """
    Class to store and post process the solution of an N-Cycle-OCP
    """

    def __init__(self):
        super().__init__()

        self.higOscTraj: HighlyOscillatoryTrajectory = None

        self.cycleSimClass = None

    def plotState(self, timeScale='physical'):
        """
        plots the found solution for each state.
        Assumes the last state to be the timestate
        """
        assert timeScale in ['physical', 'numerical']

        logging.info("Plotting Solution")
        import matplotlib.pyplot as plt

        model = self.ocp.model

        nx = self.ocp.model.nx

        plt.figure(f"Solution by State, Problem: {self.__class__.__name__}", figsize=(10, 2 * (nx + 1) // 2))

        # iterate states
        for j in range(model.nx):
            stateName = model.stateLabels[j]
            plt.subplot(2, (nx + 1) // 2, j + 1)

            # time_grid = sol.Xopt[-1,:] if timeScale == 'physical'
            if timeScale == 'physical':
                plt.plot(self.higOscTraj.X[-1, :],
                         self.higOscTraj.X[j,:], '-')

                # if self._simulation is not None:
                #     plt.plot(self._simulation[-1], self._simulation[j], 'C1--', alpha=0.5)

                plt.xlabel(r"Physical Time $t$")

            if timeScale == 'numerical':
                plt.plot(self.higOscTraj.tau, self.higOscTraj.X[model.stateNameToSlice(stateName)].T, 'C0-')
                plt.xlabel(r"Numerical Time $\tau$")

            # plt.plot([], [], 'C0-', label='x(t)')
            plt.ylabel(stateName)
            # plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    @property
    def Zopt(self):
        return self._Zopt

    @Zopt.setter
    def Zopt(self, value):
        self._Zopt = value

    def plotControls(self, timeScale='physical'):
        """
        Plots the solution for every control.
        :param timeScale: 'physical' or 'numerical'
        """
        assert timeScale in ['physical', 'numerical']

        model = self.ocp.model
        nu = self.ocp.model.nu

        import matplotlib.pyplot as plt

        plt.figure("Solution by Control", figsize=(10, 2 * (nu + 1) // 2))

        # Option 1: One control for each stage
        # plot periodic control for each stage
        if self.ocp.DEPRIVATED_constantStageControls:
            # iterate controls
            for j in range(nu):
                # iterate stages
                for i in range(self.Nintervals):
                    # new subplot
                    plt.subplot(nu, self.Nintervals, j * self.Nintervals + i + 1)
                    plt.gca().set_title(f"Stage {i + 1}")

                    # time_grid = sol.Xopt[-1,:] if timeScale == 'physical'
                    if timeScale == 'physical':
                        raise NotImplementedError("Physical time not implemented for controls")

                    if timeScale == 'numerical':
                        # plot controls
                        Uopt = self.higOscTraj.cycles[i * self.ocp.tf].U
                        plt.stairs(Uopt[j, :], np.linspace(0, 1, Uopt.shape[1] + 1),
                                   label=model.controlLabels[j])

                        # plot simulation in orange
                        plt.xlabel(r"Numerical Time $\tau$")

                    # legend entries
                    plt.legend()

                    # other formatting
                    plt.ylabel(model.controlLabels[j])
                    plt.grid()

        # Option 2: Individual control for each cycle
        if not self.ocp.DEPRIVATED_constantStageControls:

            # build edge values for stair plot
            edges_single_cycle = np.linspace(0, 1, self.cycleSimClass.Nctrl, endpoint=False)
            tau_controls = np.concatenate([edges_single_cycle + n for n in range(self.ocp.tf * self.Nintervals)] +
                                          [np.array([self.ocp.tf * self.Nintervals])])

            # iterate controls
            for j in range(nu):
                # build control vector
                U_all = np.concatenate(
                    [self.higOscTraj.cycles[n].U[j, :].flatten() for n in range(self.ocp.tf * self.Nintervals)])

                plt.subplot(nu, 1, (j + 1))

                # time_grid = sol.Xopt[-1,:] if timeScale == 'physical'
                if timeScale == 'physical':
                    t_controls = numericalToPhysicalTime(tau_controls, self.higOscTraj.Ts)
                    plt.stairs(U_all, t_controls, label=f'U_{j}', hatch='//')
                    plt.xlabel(r"Physical Time $t$")

                if timeScale == 'numerical':
                    plt.stairs(U_all, tau_controls, label=f'U_{j}', hatch='//')
                    plt.xlabel(r"Numerical Time $\tau$")

                # plt.plot([], [], 'C0-', label='x(t)')
                # plt.ylabel(model.stateLabels[j])
                plt.legend()
                plt.grid()

        plt.tight_layout()
        plt.show()


class DiscreteNCycleOCP(DiscreteOCP):

    #######################
    ##### Constructor #####
    #######################

    def __init__(self,
                 ocp: NCycleOCP,
                 cycleSimClass: type(Cycle),
                 Nintervals: int,
                 dControl=None
                 ):
        """
        Class for a discretized N-Cycle-OCP that can be solved.
        The given continuous time ocp is discretized in the implemented method construct()
        and solved using the NLP solver IPOPT in the method solve().

        :param ocp: An instance of a continous time N-Cycle OCP
        :param discretizationScheme: The discretization scheme.
        :param Nctrl: The number of controls in each cycle
        :param NintPerCtrl: The number of integration steps per control.
        """
        self.cycleSimClass: Cycle = cycleSimClass
        self.ocp = ocp
        self.Nintervals = Nintervals

        # no number of controls given? have a control every cycle
        self.dControl = dControl
        if dControl is None:
            self.dControl = ocp.tf
        else:
            # self.controlRKInt = ChebyshevCollocation(self.dControl)
            self.controlRKInt = LegendreCollocation(self.dControl)

        wEntries = []  # N cycles
        wEntries.append(entry("cycles", struct=cycleSimClass.z, repeat=[self.Nintervals, ocp.tf]))
        wEntries.append(entry("controls", shape=(cycleSimClass.model.nu, cycleSimClass.Nctrl),
                              repeat=[self.Nintervals, self.dControl]))  # controls

        super().__init__(ocp, cycleSimClass, wEntries)

    #######################
    ##### Methods #########
    #######################

    def constructNLP(self, parameters=ca.vertcat([])):

        assert self.ocp.x0bar is not None
        assert self.ocp.xendguess is not None
        assert self.ocp.tf > 0

        # assert type(self.discretizationScheme) in (DiscretizationSchemes.SingleShooting,
        #                                            DiscretizationSchemes.MultipleShooting,
        #                                            DiscretizationSchemes.Collocation)

        logging.info("Constructing NLP ...")

        ##########################
        ### For easier typing ####
        ##########################

        model = self.ocp.model
        Nctrl = self.cycleSimClass.Nctrl
        N = self.ocp.tf
        Ntotal = self.ocp.tf * self.Nintervals

        if self.ocp.Uguess is None:
            Uinitial = np.zeros((model.nu, Nctrl))
        else:
            Uinitial = self.ocp.Uguess

        Umin = np.tile(model.lbu.cat, Nctrl)
        Umax = np.tile(model.ubu.cat, Nctrl)

        x0bar = self.ocp.x0bar
        xendguess = self.ocp.xendguess

        #########################
        ### BUild NLP ###########
        #########################

        self.g = []
        self.lbg = []
        self.ubg = []
        self.X_ret = []
        self.Z_ret = []
        self.U_ret = []
        self.T_ret = []
        self.tau_states_ret = []
        self.parameters = parameters

        self.J_int = 0  # cumulative cost function
        self.J_final = 0  # final cost

        x_plus_kmin1 = x0bar

        # iterate stages
        for k in range(self.Nintervals):
            logging.debug(f"Building {N} Cycles in Stage {k + 1}/{self.Nintervals}")

            # if dControl is specified, create a control polynomial
            if not (self.dControl == self.ocp.tf):
                U_stage = self.w["controls", k, :]
                U_poly_stage = self.controlRKInt.getPolyEvalFunction(self.cycleSimClass.U.shape)

            # Set bounds and initilization for controls
            for m in range(self.dControl):
                self.w0["controls", k, m] = Uinitial
                self.lbw["controls", k, m] = Umin
                self.ubw["controls", k, m] = Umax
                # self.self.U_ret.append(self.w["controls", k, m])

            # iterate cycles
            for n in range(self.ocp.tf):
                logging.debug(f"Building Cycle {n + 1}/{N} in Stage {k + 1}/{self.Nintervals}")

                index = n + k * self.ocp.tf

                # if dControl is not specified, have a fixed control for each cycle
                if not (self.dControl == self.ocp.tf):
                    U_n = U_poly_stage(n / self.ocp.tf, *U_stage)
                else:
                    U_n = self.w["controls", k, n]

                #######################
                ## Cycle Conditions ###
                #######################

                # new cycle conditions for this cycle
                x_minus_guess = ((Ntotal - index) * x0bar.cat + index * xendguess.cat) / Ntotal
                x_plus_guess = ((Ntotal - (index + 1)) * x0bar.cat + (
                        index + 1) * xendguess.cat) / Ntotal  # TODO: approx could be better

                # project guesses to phase conditions
                x_minus_guess = self.cycleSimClass.projectInitialGuess(x_minus_guess)
                x_plus_guess = self.cycleSimClass.projectInitialGuess(x_plus_guess)

                z_n = self.cycleSimClass.z(self.w["cycles", k, n])  # cast to casadi struct

                # append variables for solver
                self.w0["cycles", k, n] = self.cycleSimClass.z0(x_minus_guess)
                self.lbw["cycles", k, n] = self.cycleSimClass.lbz
                self.ubw["cycles", k, n] = self.cycleSimClass.ubz

                cycleConditions = self.cycleSimClass.f_cycleConditions(z_n, U_n)
                self.g.append(cycleConditions)
                self.lbg.append(np.zeros(cycleConditions.shape))
                self.ubg.append(np.zeros(cycleConditions.shape))

                # append variables for output functions
                self.X_ret.append(self.cycleSimClass.f_getX(z_n, U_n))
                self.tau_states_ret.append(
                    self.cycleSimClass.f_getLocalTau(z_n,
                                                     U_n) + n)  # shift numerical time of cyle ([0,1]) to this cycle time
                self.Z_ret.append(z_n)

                # add cycle cost to total integral cost
                self.J_int += self.cycleSimClass.f_cycleCost(z_n, U_n)

                # attach xplus_last and xmin
                self.g.append(x_plus_kmin1 - z_n["x_minus"])
                self.lbg.append(np.zeros((model.nx, 1)))
                self.ubg.append(np.zeros((model.nx, 1)))

                # for next cycle: update x_plus_(k-1)
                x_plus_kmin1 = z_n["x_plus"]

        #############################
        #   Terminal Constraints    #
        #############################

        g_term, lbg_term, ubg_term = self.ocp.terminalConstraints(x_plus_kmin1)
        self.g.append(g_term)
        self.lbg.append(lbg_term)
        self.ubg.append(ubg_term)

        #############################
        #   Terminal Cost Function  #
        #############################

        # add final cost to objective
        self.J_final += self.ocp.finalCostFunction(x_plus_kmin1)

    def finalizeNLPConstruction(self):
        ##############################
        ##### Finalize NLP Constr ####
        ##############################

        # Concatenate vectors
        self.p = self.parameters  # emtpy variable if no parameters
        self.g = ca.vertcat(*self.g)
        self.lbg = ca.vertcat(*self.lbg)
        self.ubg = ca.vertcat(*self.ubg)

        # Function to retrieve the solution
        self.f_getXopt = ca.Function("getXopt", [self.w, self.p], [ca.horzcat(*self.X_ret)])
        self.f_getUopt = ca.Function("getUopt", [self.w, self.p], [ca.vertcat(*self.U_ret)])
        self.f_getZopt = ca.Function("getZopt", [self.w, self.p], [ca.horzcat(*self.Z_ret)])
        self.f_getTopt = ca.Function("getTopt", [self.w, self.p], [ca.vertcat(*self.T_ret)])
        self.f_getTauX = ca.Function("getTauX", [self.w, self.p], [ca.vertcat(*self.tau_states_ret)])
        self.f_getJintopt = ca.Function("getJintopt", [self.w, self.p], [self.J_int])
        self.f_getJfinopt = ca.Function("getJfinopt", [self.w, self.p], [self.J_final])
        self.f_getJtotalopt = ca.Function("getJtotalopt", [self.w, self.p], [self.J_final + self.J_int])

        logging.info("... finished building NLP!")

    def formateOCPResult(self, nlpsolution: dict, params, solverStats: DiscreteOCP.SolverStats, solverStatsDict) -> NCcyleOCPSolution:
        """ Formats the output of the NLP solver into a NCcyleOCPSolution object"""

        logging.info("Formatting NLP solution ...")

        wopt = nlpsolution['x']
        _wopt = self.w(wopt)
        Xopt = self.f_getXopt(wopt, params).full()
        # solution.Uopt = self.f_getUopt(wopt).full()

        # construct HO trajectory
        logging.debug("Constructing Highly Oscillatory Trajectory ...")

        higOscTraj = HighlyOscillatoryTrajectory()
        Zopt_List = []

        for k in range(self.Nintervals):
            Zopt_List_stage = []
            if self.dControl == self.ocp.tf:
                Uopt_stage = _wopt["controls", k]
            else:
                Uopt_stage = self.controlRKInt.getPolyEvalFunction(self.cycleSimClass.U.shape,
                                                                   fixedValues=_wopt["controls", k])

            for n in range(self.ocp.tf):  # iterate cycles
                index = k * self.ocp.tf + n

                Zopt = self.cycleSimClass.z(_wopt['cycles', k, n])
                Zopt_List_stage.append(Zopt)

                Uopt_Cycle = Uopt_stage[n] if self.dControl == self.ocp.tf else Uopt_stage(n / self.ocp.tf)
                Uopt_Cycle = Uopt_Cycle.full()  # convert to numpy array
                Topt = float(self.cycleSimClass.f_getT(Zopt, Uopt_Cycle).full())
                Xopt_Cycle = self.cycleSimClass.f_getX(Zopt, Uopt_Cycle).full()
                Tau_Cycle = self.cycleSimClass.f_getLocalTau(Zopt, Uopt_Cycle).full() + index

                higOscTraj.addCycle(Xopt_Cycle, Topt, Uopt_Cycle, Tau_Cycle)

            Zopt_List.append(Zopt_List_stage)

        logging.debug("Fill solution instance ...")
        # create and fill solution instance
        solution = NCcyleOCPSolution()  # empty solution class

        solution.wopt = self.w(nlpsolution['x'])
        solution.ocp = self.ocp
        solution.Uopt = _wopt["controls"]
        solution.Jopt = self.f_getJtotalopt(wopt, params).full()
        solution.Jintopt = self.f_getJintopt(wopt, params).full()
        solution.Jfinopt = self.f_getJfinopt(wopt, params).full()
        solution.cycleSimClass = self.cycleSimClass
        solution.higOscTraj = higOscTraj

        solution.nlpsolution = nlpsolution
        solution.Zopt = Zopt_List

        solution.parameterValues = params.full()
        solution.Nintervals = self.Nintervals

        solution.solverStats = solverStats
        solution.solverStats_dict = solverStatsDict

        # store hessian of lagrangian
        # logging.debug("Calculating Hessian of Lagrangian ...")
        # lagrangian_hess_fun = self.solver.get_function('nlp_hess_l')
        #
        # hess_opt = lagrangian_hess_fun(nlpsolution['x'],
        #                                [],
        #                                1,
        #                                nlpsolution['lam_g']).full()
        #
        # solution.lagrangian_hessian = hess_opt

        # log some information
        logging.info(f"Optimal Objective: {solution.Jopt}")
        logging.info(f"- StageCost: {solution.Jintopt}")
        logging.info(f"- TerminalCost: {solution.Jfinopt}")

        logging.info("... finished!")

        return solution

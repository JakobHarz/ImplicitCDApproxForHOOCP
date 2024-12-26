import copy
import logging
import time
from typing import List, Dict, Any, Tuple
import casadi as ca
import numpy as np
from Core.controlparameterization import ControlParameterization, ZeroControlParameterization
from casadi.tools import struct_symSX
from Core import configuration
from Core.hotrajectory import HighlyOscillatoryTrajectory
from Core.model import Model
from Core.parameters import Parameters


class Cycle:
    """
    A class that simulates a single cycle of a model with the given controls U.

    After construction, the instance has the following attributes:

    Casadi SX:
       - z: variables
       - g: equality constraints

    Casadi DM:
       - lbz: lower bounds
       - ubz: upper bounds
       - z0: initial values

    The class has the attribute 'z' that stores the algebraic variables
    of the cycle simulation in a casadi structure. By default, z contains:

    - z["x_minus"] size (nx,1)
    - z["x_plus"] size (nx,1)
    - z["T"] size (1,1)

    Inheriting classes have to implement the functions

    - generateVariables() to add additional variables to the casadi structure z and
    - construct microintegration () to implement the micro-integrator
    """

    def __init__(self, model: Model, stageCostFunction: ca.Function, controlParameterization: ControlParameterization, parameters,
                 nu: int = None):
        self.model = model
        # self.Nctrl = Nctrl
        self.stageCostFunction = stageCostFunction

        self.controlParameterization = controlParameterization
        self.parameters: Parameters = parameters

        # custom number of control inputs?
        self.nu = nu if nu is not None else self.model.nu

        # generate the control matrix
        # self.U = ca.SX.sym("U", (self.nu, self.Nctrl))
        # """ Symbolic variable for the control matrix U """

        self.f_cycleConditions: ca.Function = None
        """ Casadi function that evaluates the cycle conditions C(z,U) """

        self.f_cycleCost: ca.Function = None
        self.f_getX: ca.Function = None
        self.f_getLocalTau: ca.Function = None
        self.f_getT: ca.Function = None

        self.f_outputMap = None

        self.z = None
        self.g = None

        self._lbz = None
        self._ubz = None

        self._cycleConditions_struct: struct_symSX = None

    @property
    def lbz(self) -> ca.tools.struct:
        """ Get the lower bound of the algebraic variable z """
        if self._lbz is None:
            lbz = self.z(-np.inf)
            lbz["x_plus"] = self.model.lbx
            lbz["x_minus"] = self.model.lbx
            self._lbz = lbz
        return self._lbz

    @property
    def ubz(self) -> ca.tools.struct:
        """ Get the lower bound of the algebraic variable z """
        if self._ubz is None:
            ubz = self.z(np.inf)
            ubz["x_plus"] = self.model.ubx
            ubz["x_minus"] = self.model.ubx
            self._ubz = ubz
        return self._ubz

    def z0(self, xminusguess: ca.DM) -> ca.tools.struct:
        """ Get an initialization guess for the cycle simulation.
        Returns a struct with the same structure as z.
        Can be potentially overwritten by inheriting classes.
        """
        z0 = self.z(0)
        z0["x_plus"] = xminusguess
        z0["x_minus"] = xminusguess
        return z0

    @property
    def cycleConditions_struct(self) -> ca.tools.struct_symSX:
        """ the structure of the equations inside the cycle conditions.
        Useful to cast the expressions into a larger substructure, for example:

        ``conditions['cycle'] = cycleConditions_struct(C(z,U))``
        """
        assert self._cycleConditions_struct is not None, "cycleConditions_struct is not set yet"
        assert self._cycleConditions_struct.size == self.z.size - self.model.nx, \
            f"cycleConditions_struct has the wrong size, z:{self.z.size} - nx:{self.model.nx} = {self.z.size - self.model.nx}, but cond are {self._cycleConditions_struct.size}"
        return self._cycleConditions_struct

    @cycleConditions_struct.setter
    def cycleConditions_struct(self, value: ca.tools.struct_symSX):
        self._cycleConditions_struct = value

    def logInfo(self):
        logging.debug("-------- Cycle Simulation ---------")
        logging.debug("Class: " + self.__class__.__name__)
        logging.debug(f" model = ")
        # logging.info(f"Nctrl = {self.Nctrl}")
        logging.debug(f"stageCostFunction = {self.stageCostFunction}")
        logging.debug(f"z = {self.z}")

    @property
    def nz(self) -> int:
        """ Get the number of algebraic variables """
        return self.z.cat.size1()

    def solveCycle(self, x0: ca.DM,
                   U: List[ca.DM] = None,
                   tau_start: float = 0,
                   supressOutput: bool = True,
                   additionalSolverOptions: Dict[str, Any] = {}) -> ca.tools.struct:
        """
        Solve the cycle simulation from a given initial guess x0 and control parameters [U] that parameterize the controls.
        The cycle simulation starts at time tau_start, which also is used to evaluate the controls.
        :param x0: initial points of the cycle simulation
        :param U: list of control parameters that parameterize the controls
        :param tau_start: start time of the cycle simulation
        :param supressOutput:
        :param additionalSolverOptions:
        :return:
        """

        assert x0.shape == self.model.x.shape, "x0 has to be of the same shape as the model state"
        assert type(U) in [list, type(None)], "U has to be a list of d control matrices of shape (nu, Nctrl)"

        # if no control parameters are given, get them from the control parameterization
        if U is None:
            # assert isinstance(self.controlParameterization, ZeroControlParameterization)
            if self.controlParameterization is not None:
                U = [ca.DM.zeros((self.controlParameterization.Nctrl, self.controlParameterization.nu))
                     for _ in range(self.controlParameterization.slowPolynomial.d)]
            else:
                U = []

        # suppress output if desired
        _loggingLevelBefore = logging.getLogger().getEffectiveLevel()
        if supressOutput:
            logging.getLogger().setLevel(logging.ERROR)

        logging.info("Transferring NLP to solver ...")

        conditions = ca.vertcat(x0 - self.z["x_minus"],
                                self.f_cycleConditions(self.z, tau_start, *U))

        # Create an NLP solver
        problem = {'f': 0, 'x': self.z, 'g': conditions}
        options = {'ipopt.linear_solver': configuration.IPOPT_OPTIONS_LINEAR_SOLVER,
                   'ipopt.max_iter': 2000,
                   'ipopt.print_level': 0 if supressOutput else 5,
                   'print_time': False if supressOutput else True,
                   }
        options.update(additionalSolverOptions)

        self.solver = ca.nlpsol('solver', 'ipopt', problem, options)

        logging.info(f"Solving NLP ...")

        # Solve the NLP
        nlpsolution = self.solver(x0=self.z0(x0), lbx=self.lbz, ubx=self.ubz, lbg=0, ubg=0)

        if not supressOutput:
            logging.info(f"... finshed solving NLP,"
                         f" wall time: {self.solver.stats()['t_wall_total'] * 1000:0.2f} ms,"
                         f" CPU time: {self.solver.stats()['t_proc_total'] * 1000:0.2f} ms")

        # get the solution into the right format
        zopt = self.z(nlpsolution["x"])

        # reset logging level
        if supressOutput:
            logging.getLogger().setLevel(_loggingLevelBefore)

        return zopt

    def solveNCycles(self, x0: ca.DM, N: int, supressOutput: bool = False, backwards=False,
                     additionalSolverOptions: Dict[str, Any] = {}, tau_start: float = 0) -> List[ca.tools.struct]:
        """
        Efficiently solve the cycle simulation over N Cycles with the given controls [U_{0},..U_{N-1}] and initial guess x0

        returns a list of optimal algebraic variables z for each cycle

        :param x0: start point for the first cycle
        :param U_list: list of control matrices U that parameterize the control over the cycles
        :param supressOutput: supress the output of the solver?
        :param backwards: solve the cycles backwards in time?
        :param additionalSolverOptions: additional options for the solver
        :param tau_start: the start time in numerical time

        """
        # assert type(U_list) == list, "U has to be a list of d control matrices of shape (nu, Nctrl)"
        assert x0.shape == self.model.x.shape, "x0 has to be of the same shape as the model state"

        # suppress output if desired
        _loggingLevelBefore = logging.getLogger().getEffectiveLevel()
        if supressOutput:
            logging.getLogger().setLevel(logging.ERROR)

        # define problem to be solved for each cycle
        x0bar = ca.SX.sym("x0bar", self.model.nx)
        tau_n = ca.SX.sym("tau_n")  # symbolic time variable for the current cycle number
        conditions = ca.vertcat(x0bar - self.z["x_minus"],
                                self.f_cycleConditions(self.z, tau_n))

        # Create an NLP solver
        logging.info(f"Setting up cylce solver ...")
        problem = {'f': 0, 'x': self.z, 'g': conditions,
                   'p': ca.vertcat(tau_n, x0bar)}
        options = {'ipopt.linear_solver': configuration.IPOPT_OPTIONS_LINEAR_SOLVER,
                   'ipopt.max_iter': 2000,
                   'ipopt.print_level': 0 if supressOutput else 5,
                   'print_time': False if supressOutput else True,
                   }
        options.update(additionalSolverOptions)

        self.solver = ca.nlpsol('solver', 'ipopt', problem, options)

        logging.info(f"Solving the cycle simulation for {N} cycles  ...")

        # fordwards or backwards in time?
        w0 = self.z0(x0)
        lbz = copy.deepcopy(self.lbz)
        ubz = copy.deepcopy(self.ubz)

        if backwards:
            w0["T"] = -w0["T"]
            lbz["T"], ubz["T"] = -self.ubz["T"], -self.lbz["T"]

        returnList = []
        _startTime = time.time()

        # Solve the NLP for each cycle
        for n in range(N):
            print(f"Cycle {n + 1}/{N}", end='\r', flush=True)

            nlpSolution = self.solver(x0=w0, lbx=lbz, ubx=ubz, lbg=0, ubg=0,
                                      p=ca.vertcat(n, x0))

            # get the solution into the right format
            zopt = self.z(nlpSolution["x"])
            returnList.append(zopt)

            # update the initial guess for the next cycle
            w0 = nlpSolution["x"]

            # project xplus onto the subspace of xminus
            x0 = self.projectInitialGuess(zopt["x_plus"], 'minus')

            # check result of optimization and projection
            # if not self.phaseConditions.phiminus(
            #         x0) == 0.0: logging.warning(
            #     f"Phasecondition at start of cycle {n}, xmin={x0} is: {self.phaseConditions.phiminus(x0)}")
            if not self.solver.stats()['success']:
                logging.warning(f'WARNING: {self.solver.stats()["return_status"]}')

        # print a new line to make the output look nicer
        print("")
        logging.info(f"... Done! Total time: {(time.time() - _startTime) * 1000:.2f} ms")

        # reset logging level
        if supressOutput:
            logging.getLogger().setLevel(_loggingLevelBefore)

        return returnList

    def simulateNCycles(self, x0: ca.DM, Ncycles: int, supressOutput=True) -> HighlyOscillatoryTrajectory:
        '''
        Solves the cycle simulation for Ncycles cycles with the given control matrices UCycles and initial guess x0.
        returns a highly oscillatory trajectory.

        :param x0: start point
        :param Ncycles: number of cycles
        :param p: periodicity condition
        :param UCycles: (1) None (2) Single Control Matrix (nu,Nctrl) (3) Series of Control Matrices for each cycle of size (nu,Nctrl, Ncycles)
        :param NintPerCtrl: number of integration steps per control, default 20
        :param NoutPerCtrl: number of output points per control, default 1
        :param supressOutput: if True, no output is printed

        :return: a highly oscillatory trajectory
        '''

        # suppress output if desired
        if supressOutput:
            _levelBefore = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.ERROR)

        logging.info("")
        logging.info(f'Performing a forward simulation:')

        assert type(Ncycles) == int, "Ncycles has to be an integer"

        # def formatControl(U) -> List[ca.DM]:
        #     # check for correct type
        #     assert type(U) in [type(None), ca.DM, list], f"Wrong type, U is of type: {type(U)}"
        #
        #     # option 1: no U given -> set U to 0 for all cycles, assume 10 controls per cycle
        #     if U is None:
        #         logging.info(f' - with zero controls U = 0')
        #         return [ca.DM.zeros((self.model.nu, 10)) for _ in range(Ncycles)]
        #
        #     # option 2: single U (nu,Nctrl) given -> set this U for all cycles
        #     if type(U) == ca.DM:
        #         logging.info(f' - with a given constant control U_const of shape {U.shape},')
        #         assert U.shape[0] == self.model.nu, "Given control has the wrong number of inputs"
        #         return [U for _ in range(Ncycles)]
        #
        #     # option 3: U (nu, Nctrl, Ncycles) given
        #     if type(U) == list:
        #         assert all([type(u) == ca.DM for u in U]), "Given controls have to be casadi.DM matrices"
        #         assert len(U) == Ncycles, "Given control has the wrong number of cycles"
        #         logging.info(f' - with given controls U_0, U_1, .. for each cycle, of shape {U[0].shape},')
        #         return U
        #
        #     raise Exception("Format detection failed!")

        # fix control format for different input types
        # UCycles = formatControl(UCycles)
        # Nctrl = UCycles[0].shape[1]

        # logging.info(f' - with {self.NintPerCtrl} integration steps per control')
        logging.info(f' - over {Ncycles} cycles')

        # 2. Call the classes solveNcycles method
        zcycles = self.solveNCycles(x0, Ncycles, supressOutput=supressOutput)

        # 3. reformate the results
        returnTrajectory = HighlyOscillatoryTrajectory()

        for n in range(Ncycles):
            tau_n = n
            z = zcycles[n]
            # U = UCycles[n]

            # retrieve the results from the cycle class
            Xopt = self.f_getX(z, tau_n).full()
            tauopt = self.f_getLocalTau(z, tau_n).full()
            cycleCost = self.f_cycleCost(z, tau_n).full()
            Topt = float(self.f_getT(z, tau_n).full())

            returnTrajectory.addCycle(Xopt, Topt, np.zeros(self.model.u.shape), tauopt + n,
                                      cycleCost)  # TODO: get rid of trajectory class

        # reset output level
        if supressOutput:
            logging.getLogger().setLevel(_levelBefore)

        return returnTrajectory

    def projectInitialGuess(self, x_guess_macro: ca.DM, target='minus'):
        """
        Project the guess for the macro variables onto the subspace of the micro variables if necessary.
        The default implementation returns the value without any projection.
        :param x_guess_macro:
        :param target: 'plus' or 'minus', the target subspace
        :return: the projected guess
        """
        return x_guess_macro

    def getHigherAccuracyCopy(self) -> "Cycle":
        """
        Returns a copy of the cycle simulation with higher accuracy
        """
        raise NotImplementedError("getHigherAccuracyCopy() not implemented yet")

    def revertTimeConstraintsandInit(self, z0, lbz, ubz) -> Tuple[ca.DM, ca.DM, ca.DM]:
        """ Modifies the given initial guess z0 and bounds lbz, ubz to make sure that
        a cycle is computed backwards in time """
        raise NotImplementedError("revertTimeConstraintsandInit() not implemented yet")


############################
# FIXED PERIOD TYPE CYCLES #
############################


###############################
# PHASE CONDITION TYPE CYCLES #
###############################


# class CycleCollocationCondensed(Cycle):
#
#     def __init__(self,
#                  model: Model,
#                  phaseConditions: LinearPhaseConditions,
#                  Nctrl: int,
#                  stageCostFunction: Callable = (lambda x, u: 0),
#                  NintPerCtrl: int = 5,
#                  d_coll=3,
#                  type="legendre"):
#
#         assert type in ["legendre", "radau"], "type must be legendre or radau"
#
#         self.NintPerCtrl = NintPerCtrl
#         self.d_coll = d_coll
#
#         super().__init__(model, phaseConditions, Nctrl, stageCostFunction)
#
#         #
#         self.z_entries = [
#             entry(f"x_plus", struct=self.model.x_struct),
#             entry(f"x_minus", struct=self.model.x_struct),
#             entry(f"T"),
#             entry(f"V_Coll", shape=(self.model.nx, self.d_coll * self.Nctrl * self.NintPerCtrl))
#         ]
#         """ The entries of the casadi struct z. Can be extended by inheriting classes. """
#
#         # generate the z-structure
#         self.z = struct_symSX(self.z_entries)
#
#         # generate the control matrix
#         self.U = ca.SX.sym("U", (self.model.nu, self.Nctrl))
#
#         ####### BUILD THE CYCLE SIMULATION #######
#
#         # log the settings for the cycle simulation
#         self.logInfo()
#
#         #########################
#         ### Preparation #########
#         #########################
#
#         # collect equalities and return states in here
#         self.g = []
#         self.x_ret = []
#         self.tau_cycle_ret = []  # the respective numerical times for each value in x_ret, \in [0,1]
#
#         #####################
#         #### x minus k ######
#         #####################
#
#         # new xminus
#         x_minus_k = self.z["x_minus"]
#         self.x_ret.append(x_minus_k)
#         self.tau_cycle_ret.append(0)
#
#         # phase condition on xminus
#         self.g.append(self.phaseConditions.phiminus(x_minus_k))
#
#         #####################
#         # Micro Integration #
#         #####################
#
#         self.cycle_cost = 0  # collect the cycle cost
#
#         # Construct the micro-integration
#         logging.debug("Constructing SingleShooting Micro-Integration ...")
#         # start integration at x_minus
#         x_micro_m = self.z["x_minus"]
#         T = self.z["T"]
#
#         # Set up integrator
#         h = 1 / (self.NintPerCtrl * self.Nctrl)  # integrator step
#         self.integrator = RK4.explicitFunction(self.model.f, h, self.model.x, self.model.u, self.model.T,
#                                                quad=self.stageCostFunction)
#
#         # collect piecewise output functions and edges
#         _piecewise_funcs = []
#         _piecewise_edges = []
#         _sigma = ca.SX.sym("sigma")  # local numerical time, used for output function
#
#         logging.debug("Constructing Collocation Micro-Integration ...")
#
#         # start integration at x_minus
#         x_micro_m = self.z["x_minus"]
#         T = self.z["T"]
#
#         # set up integrator
#         h = 1 / (self.NintPerCtrl * self.Nctrl)  # integrator step
#         self.collTimes = np.array(ca.collocation_points(self.d_coll, type))
#         RKInt = OthorgonalCollocation(self.collTimes)
#         polyFunc = RKInt.getPolyEvalCAExpression(model.nx, includeZero=True)
#         c, A, b = RKInt.getButcher()
#
#         # Micro-Integrate Cycle
#         for m in range(self.Nctrl):
#
#             # logging.info(f"\t Control {m+1}/{Nctrl}")
#
#             for n in range(self.NintPerCtrl):
#                 # logging.info(f"Control: {m}/{Nctrl}, Int step: {n}/{NintPerCtrl}")
#
#                 # State at collocation points
#                 Xs = []
#                 Vs = []
#
#                 # create RK variables and bound them
#                 for i in range(self.d_coll):
#                     # index for the current variable
#                     current_index_COLL = i + n * self.d_coll + m * self.NintPerCtrl * self.d_coll  # index in array X_Coll
#                     # logging.info(f"X_COLL Index:  {current_index_COLL}")
#
#                     # differential variables
#                     # Xj = self.z["X_Coll", :, current_index_COLL]
#                     # Xs.append(Xj)
#                     # self.x_ret.append(Xj)
#                     self.tau_cycle_ret.append((m * self.NintPerCtrl + n) * h + h * c[i])
#
#                     # self.lbz["X_Coll", :, current_index_COLL] = self.model.lbx
#                     # self.ubz["X_Coll", :, current_index_COLL] = self.model.ubx
#
#                     # slope variables
#                     Vj = self.z["V_Coll", :, current_index_COLL]
#                     Vs.append(Vj)
#
#                     # bounds of the slope variable leave them at -inf, +inf
#                     self.lbz["V_Coll", :, current_index_COLL] = self.model.lbv
#                     self.ubz["V_Coll", :, current_index_COLL] = self.model.ubv
#
#                 # RK Integrator equations
#                 q_end = 0
#                 Xk_end = x_micro_m
#                 for i in range(self.d_coll):
#                     v = Vs[i]
#                     u = self.U[:, m]
#
#                     # equation for rk
#                     xj_bar = x_micro_m
#                     for j in range(self.d_coll):
#                         xj_bar = xj_bar + A[i, j] * Vs[j]
#                     self.x_ret.append(xj_bar)
#                     Xs.append(xj_bar)
#                     # self.g.append(xj_bar - x)
#
#                     # Append collocation equations
#                     self.g.append(h * self.model.f(xj_bar, u, T) - v)
#
#                     # Add contribution to the end state
#                     Xk_end = Xk_end + b[i] * Vs[i]
#                     q_end = q_end + b[i] * self.stageCostFunction(xj_bar, u)
#
#                 self.cycle_cost += q_end
#
#                 # create multiple shooting node at end of collocation interval
#                 current_index_MS = m * self.NintPerCtrl + n
#
#                 # add piecewise values
#                 _sigma_local = (m * self.NintPerCtrl + n) * h
#                 _intermediateValue = polyFunc((_sigma - _sigma_local) / h, ca.horzcat(*([x_micro_m] + Xs)))
#                 _piecewise_funcs.append(_intermediateValue)
#                 _piecewise_edges.append(_sigma_local)
#
#                 # dont create new node in last interval
#                 if current_index_MS < self.NintPerCtrl * self.Nctrl - 1:
#
#                     x_micro_m = Xk_end
#                     # self.lbz["X_Nodes", :, current_index_MS] = self.model.lbx
#                     # self.ubz["X_Nodes", :, current_index_MS] = self.model.ubx
#
#                     self.x_ret.append(x_micro_m)
#                     self.tau_cycle_ret.append((m * self.NintPerCtrl + n + 1) * h)
#
#                     # constrain end of intervall to this variable
#                     # self.g.append(Xk_end - Xk)
#
#                     # x_micro_m = Xk
#                 else:
#                     # will be constrained to x_plus in a moment
#                     x_micro_m = Xk_end
#
#         _piecewise_edges.append(1 + 1E-12)  # add the last edge, a bit shifted to allow sigma to be 1 in output function
#
#         integrationEndpoint = x_micro_m
#
#         #####################
#         #### x plus k #######
#         #####################
#
#         # New x_plus
#         x_plus_k = self.z["x_plus"]
#
#         self.x_ret.append(x_plus_k)
#         self.tau_cycle_ret.append(1)
#
#         # phase condition on xplus
#         self.g.append(self.phaseConditions.phiplus(x_plus_k))
#
#         # attach endpoint of micro-integration and xplus
#         self.g.append(x_plus_k - integrationEndpoint)
#
#         # output format
#         self.x_ret: List[ca.SX] = self.x_ret
#
#         # output functions
#         self.f_cycleConditions: ca.Function = ca.Function("CycleConditions", [self.z, self.U], [ca.vertcat(*self.g)])
#         self.f_cycleCost: ca.Function = ca.Function("CycleCost", [self.z, self.U], [self.cycle_cost])
#         self.f_getX = ca.Function("GetCycleTrajectorie", [self.z, self.U], [ca.horzcat(*self.x_ret)])
#         self.f_getLocalTau = ca.Function("GetCycleTaus", [self.z, self.U], [ca.vertcat(*self.tau_cycle_ret)])
#         self.f_getT = ca.Function("GetCyclePeriod", [self.z, self.U], [self.z["T"]])
#
#         self.f_outputMap = ca.Function("OutputMap", [_sigma, self.z, self.U],
#                                        [constructPiecewiseCasadiExpression(_sigma, _piecewise_edges, _piecewise_funcs)],
#                                        ["sigma", "z", "U"], ["Phi"])
#
#     def z0(self, x_minus_guess: ca.DM) -> ca.tools.struct:
#         """ Get an initialization guess for the cycle simulation.
#         Returns a struct with the same structure as z.
#         Can be potentially overwritten by inheriting classes.
#         """
#         z0 = self.z(0)
#         z0["x_minus"] = x_minus_guess
#         z0["T"] = self.model.getApproxPeriod(x_minus_guess)
#
#         # perform a forward simulation to find the inital points w0
#         Xinit = self.model.forwardSimSimple(x_minus_guess, self.model.getApproxPeriod((x_minus_guess)),
#                                             (self.d_coll + 1) * self.NintPerCtrl * self.Nctrl,
#                                             np.zeros(self.model.u.shape))
#
#         # set X_Nodes initial values
#         for m in range(self.Nctrl):
#             for n in range(self.NintPerCtrl):
#                 current_index_MS = m * self.NintPerCtrl + n
#
#                 for i in range(self.d_coll):
#                     # index for the current variable
#                     current_index_COLL = i + n * self.d_coll + m * self.NintPerCtrl * self.d_coll  # index in array X_Coll
#                     current_index_INIT = i + (n + m * self.NintPerCtrl) * (self.d_coll + 1)  # index in array Xinit
#
#                     # z0["X_Coll", :, current_index_COLL] = Xinit[:, [current_index_INIT]]
#                     z0["V_Coll", :, current_index_COLL] = np.zeros(self.model.x.shape)
#         z0["x_plus"] = ca.DM(Xinit[:, -1])
#
#         return z0



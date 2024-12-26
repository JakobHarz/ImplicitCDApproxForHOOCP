import logging
from typing import List, Callable

import casadi as ca
import numpy as np
from casadi.tools import entry, struct_symSX

from Core.controlparameterization import ControlParameterization, PiecewiseConstantControlParametrization
from Core.cycle import Cycle
from Core.integrators import RungeKuttaIntegrator, RK4, OthorgonalCollocation
from Core.model import Model
from Core.parameters import Parameters
from Core.phaseconditions import LinearBoundaryConditions
from Core.tools import constructPiecewiseCasadiExpression


class CycleShootingPhaseConditions(Cycle):
    """ Base Class for Cycle Simulation with Phase Conditions """

    def __init__(self,
                 model: Model,
                 phaseConditions: LinearBoundaryConditions,
                 controlParameterization: ControlParameterization,
                 stageCostFunction: ca.Function = None,
                 Nint: int = 20,
                 shooting_type='single',
                 parameters: Parameters = Parameters(),
                 initializationFunction: ca.Function = None):

        # check if the model implements timescaling
        assert model.requiresTimescaling, "The model has to implement timescaling for this type of cycle simulation"

        # fill in the stage cost function if not given
        if stageCostFunction is None:
            _T = ca.SX.sym("T")
            if controlParameterization is None:
                stageCostFunction = ca.Function('stageCostFunction', [model.x, model.tau, _T], [0], ['x', 'tau', 'T'],['0'])
            else:
                stageCostFunction = ca.Function('stageCostFunction', [model.x, model.tau, _T, model.u], [0], ['x', 'tau', 'T','u'],['0'])

        super().__init__(model, stageCostFunction, controlParameterization, parameters)

        if initializationFunction is None:
            initializationFunction = ca.Function('initializationFunction', [self.model.tau],
                                                 [ca.DM.zeros(self.model.x.shape)])
        assert initializationFunction.sx_out()[
                   0].shape == self.model.x.shape, "initializationFunction has to return a vector of size nx"
        assert initializationFunction.sx_in()[0].shape == (
            1, 1), "initializationFunction has to take a single argument (tau) in [0,1]"
        self.initializationFunction = initializationFunction

        # single or multiple shooting?
        self.shooting_type: str = shooting_type
        assert self.shooting_type in ['single', 'multiple'], "shooting_type has to be 'single' or 'multiple'"

        # phaseconditions
        self.phaseConditions = phaseConditions

        # number of integration intervals
        self.Nint = Nint
        self.N_Nodes = {'single': 2, 'multiple': self.Nint + 1}[
            self.shooting_type]  # number of integration nodes per cycle

        # generate the z-structure
        self.z_entries = [
            entry(f"x_plus", struct=self.model.x_struct),
            entry(f"x_minus", struct=self.model.x_struct),
            entry(f"X_Nodes", struct=self.model.x_struct, repeat=self.N_Nodes),
            entry(f"T"),
        ]
        self.z = struct_symSX(self.z_entries)

        # generate structure for the constraints of the cycle conditions
        self.cycleConditions_struct = struct_symSX([
            entry(f"CC_connect_start", shape=(self.model.nx - 1, 1)),
            entry(f"CC_integration", struct=self.model.x_struct, repeat=self.N_Nodes - 1),
            entry(f"CC_connect_end", shape=(self.model.nx - 1, 1)),
            entry(f"CC_phase_start", shape=(1, 1)),
            entry(f"CC_phase_end", shape=(1, 1)),
            entry(f"CC_phase_connect", shape=(1, 1)),

        ])
        cycleConditions = ca.tools.struct_SX(self.cycleConditions_struct)

        # check if the control parameterization is piecewise constant
        # if so, check if the number of integration intervals is divisible by the number of control intervals
        if isinstance(controlParameterization, PiecewiseConstantControlParametrization):
            if Nint % controlParameterization.Nctrl != 0:
                raise ValueError("Nint must be divisible by NintPerCtrl for PiecewiseConstantControlParametrization")
            if Nint / controlParameterization.Nctrl < 1:
                raise ValueError("There have to be at least one integration interval per control for a "
                                 "PiecewiseConstantControlParametrization")

        # populate the upper and lower bound for the algebrac variables
        self.lbz['X_Nodes'] = self.model.lbx
        self.ubz['X_Nodes'] = self.model.ubx
        self.lbz['T'] = model.lbT
        self.ubz['T'] = model.ubT

        ####### BUILD THE CYCLE SIMULATION #######

        # log the settings for the cycle simulation
        self.logInfo()

        #########################
        ### Preparation #########
        #########################

        # collect equalities and return states in here
        self.x_ret = []
        self.tau_cycle_ret = []  # the respective numerical times for each value in x_ret, \in [0,1]

        #####################
        #### x minus k ######
        #####################

        # new xminus
        x_minus_k = self.z["x_minus"]

        # start point of the integration
        x_start = self.z["X_Nodes", 0]
        self.tau_cycle_ret.append(0)
        self.x_ret.append(x_start)

        # connect all states but the constrained ones
        cycleConditions['CC_connect_start'] = phaseConditions.Q.T @ (x_minus_k - x_start)

        # phase condition on the integration start point
        cycleConditions['CC_phase_start'] = self.phaseConditions.phiminus(x_start)

        #####################
        # Micro Integration #
        #####################

        self.cycle_cost = 0  # collect the cycle cost

        # Construct the micro-integration
        logging.debug(f"Constructing {self.shooting_type}-Shooting Micro-Integration ...")

        # start integration at x_minus
        x_micro_m = x_start
        T = self.z["T"]

        # Set up integrator
        h_tau = 1 / self.Nint  # integrator step

        # collect piecewise output functions and edges
        _piecewise_funcs = []
        _piecewise_edges = []
        _sigma_SX = ca.SX.sym("sigma")  # local numerical time, used for output function

        # the numerical time at which the cycle STARTS
        tau0 = ca.SX.sym("tau0")

        # control expression for the cycle
        if controlParameterization is not None:
            u = controlParameterization.u_f(self.model.tau, self.model.tau - tau0, *parameters.syms_list)
            _integrator_expressions = [T, u] + model.symbolicParams
        else:
            _integrator_expressions = [T] + model.symbolicParams

        # integrator for this interval, with different control expression
        self.integrator = RungeKuttaIntegrator(RK4(), self.model.f, expressions= _integrator_expressions,
                                               constant_syms=[tau0, T] + parameters.syms_list,
                                               quad=self.stageCostFunction).explicitFunction()

        # Micro-Integrate Cycle
        for n in range(self.Nint):
            # start of this control interval in cycle time ∈ [0,1]
            _sigma_local = n * h_tau

            # intermediate values for continuous plotting
            _intermediateValue = self.integrator(x_micro_m,
                                                 tau0 + _sigma_local,
                                                 (_sigma_SX - _sigma_local),
                                                 *([tau0, T] + parameters.syms_list))['x_f']
            _piecewise_funcs.append(_intermediateValue)
            _piecewise_edges.append(_sigma_local)

            # compute the integration step
            integrator_result = self.integrator(x_micro_m,
                                                tau0 + _sigma_local,
                                                h_tau,
                                                *([tau0, T] + parameters.syms_list))  # Integration Result of this step
            x_micro_m, quadrature = integrator_result['x_f'], integrator_result['q_f']

            # cycle cost is scaled with the period duration
            self.cycle_cost += quadrature * T

            # collect the variables that are used to plot the cycle
            self.x_ret.append(x_micro_m)
            self.tau_cycle_ret.append((n + 1) * h_tau)

            # multiple shooting? Then also connect the intermediate variables
            if self.shooting_type == 'multiple' and n < self.Nint - 1:
                # connect the node at the of the interval to the integration result
                x_MS_m = self.z["X_Nodes", n + 1]  # get next multiple shooting node
                cycleConditions['CC_integration', n] = x_MS_m - x_micro_m  # connect the node to the integration result
                x_micro_m = x_MS_m

        _piecewise_edges.append(1 + 1E-12)  # add the last edge, a bit shifted to allow sigma to be 1 in output function

        #####################
        #### x plus k #######
        #####################

        # New x_plus
        x_plus_k = self.z["x_plus"]
        xend = self.z["X_Nodes", -1]
        self.x_ret.append(x_plus_k)
        self.tau_cycle_ret.append(1)

        # attach endpoint of micro-integration and integration end point
        cycleConditions['CC_integration', -1] = xend - x_micro_m

        # phase condition on integration end point
        cycleConditions['CC_phase_end'] = self.phaseConditions.phiplus(xend)

        # attach xplus and integration endpoint in the unconstrained states
        cycleConditions['CC_connect_end'] = phaseConditions.Q.T @ (x_plus_k - xend)

        # constrain the relation between xplus and xminus in the constrained states, since we know it
        cycleConditions['CC_phase_connect'] = (phaseConditions.q.T @ (x_plus_k - x_minus_k) -
                                               (phaseConditions.bplus - phaseConditions.bminus))

        # output format
        self.x_ret: List[ca.SX] = self.x_ret

        # # control parameters list
        # U_list_SX = self.controlParameterization.U_list_SYM

        # output functions
        self.f_cycleConditions: ca.Function = ca.Function("CycleConditions", [self.z, tau0] + parameters.syms_list,
                                                          [cycleConditions],
                                                          ["z", "tau_0"] + parameters.syms_list_names,
                                                          ['cycle Conditions'])
        self.f_cycleCost: ca.Function = ca.Function("CycleCost", [self.z, tau0] + parameters.syms_list,
                                                    [self.cycle_cost], ["z", "tau_0"] + parameters.syms_list_names,
                                                    ['cycle Cost'])
        self.f_outputMap = ca.Function("OutputMap", [_sigma_SX, self.z, tau0] + parameters.syms_list,
                                       [constructPiecewiseCasadiExpression(_sigma_SX, _piecewise_edges,
                                                                           _piecewise_funcs)],
                                       ["sigma", "z", "tau_0"] + parameters.syms_list_names, ['output map'])

        self.f_getX = ca.Function("GetCycleTrajectorie", [self.z, tau0] + parameters.syms_list,
                                  [ca.horzcat(*self.x_ret)])
        self.f_getT = ca.Function("GetCyclePeriod", [self.z, tau0] + parameters.syms_list, [1])

        self.f_getLocalTau = ca.Function("GetCycleTaus", [self.z, tau0] + parameters.syms_list,
                                         [ca.vertcat(*self.tau_cycle_ret)])

    def z0(self, xminusguess: ca.DM) -> ca.tools.struct:
        """ Get an initialization guess for the cycle simulation.
        Returns a struct with the same structure as z.
        Can be potentially overwritten by inheriting classes.
        """
        z0 = self.z(0)
        z0["x_plus"] = xminusguess
        z0["x_minus"] = xminusguess

        # initialize the multiple shooting nodes with the initialization function
        tau_Nodes = ca.linspace(0, 1, self.N_Nodes)
        z0["X_Nodes"] = ca.horzsplit(self.initializationFunction.map(self.N_Nodes)(tau_Nodes))

        z0["T"] = self.model.getApproxPeriod(xminusguess)
        return z0

    def getHigherAccuracyCopy(self, Nint_multiplier: int = 5):
        return CycleShootingPhaseConditions(self.model,
                                            self.phaseConditions,
                                            self.controlParameterization,
                                            self.stageCostFunction,
                                            self.Nint * Nint_multiplier,  # 5 times more integration points
                                            self.shooting_type)

    def projectInitialGuess(self, x_guess_macro: ca.DM, target='minus'):
        return self.phaseConditions.project(x_guess_macro, target=target)


class CycleSingleShootingPhaseCond(CycleShootingPhaseConditions):
    """ Single Shooting Cycle Simulation """

    def __init__(self,
                 model: Model,
                 phaseConditions: LinearBoundaryConditions,
                 controlParameterization: ControlParameterization = None,
                 stageCostFunction: ca.Function = None, Nint: int = 20,
                 parameters: Parameters = Parameters(),
                 initializationFunction=None):
        super().__init__(model, phaseConditions, controlParameterization, stageCostFunction, Nint,
                         shooting_type='single', parameters=parameters, initializationFunction=initializationFunction)


class CycleMultipleShootingPhaseCond(CycleShootingPhaseConditions):
    """ Multiple Shooting Cycle Simulation """

    def __init__(self,
                 model: Model,
                 phaseConditions: LinearBoundaryConditions,
                 controlParameterization: ControlParameterization = None,
                 stageCostFunction: ca.Function = None, Nint: int = 20, parameters: Parameters = Parameters(),
                 initializationFunction=None):
        super().__init__(model, phaseConditions, controlParameterization, stageCostFunction, Nint,
                         shooting_type='multiple', parameters=parameters, initializationFunction=initializationFunction)


class CycleCollocation(CycleShootingPhaseConditions):

    def __init__(self,
                 model: Model,
                 phaseConditions: LinearBoundaryConditions,
                 Nctrl: int,
                 stageCostFunction: Callable = None,
                 NintPerCtrl: int = 5,
                 d_coll: int = 3,
                 type: str = "legendre"):

        assert type in ["legendre", "radau"], "type must be legendre or radau"

        self.NintPerCtrl: int = NintPerCtrl
        self.d_coll: int = d_coll
        self.type: str = type

        if stageCostFunction is None:
            stageCostFunction = ca.Function('stageCostFunction', [model.x, model.u], [0])

        super().__init__(model, phaseConditions, Nctrl, stageCostFunction)

        self.phaseConditions = phaseConditions
        #
        self.z_entries = [
            entry(f"x_plus", struct=self.model.x_struct),
            entry(f"x_minus", struct=self.model.x_struct),
            entry(f"x_start", struct=self.model.x_struct),
            entry(f"x_end", struct=self.model.x_struct),
            entry(f"T"),

            # TODO: replace those with repeats, remove the x_start and x_end
            entry(f"X_Nodes", shape=(self.model.nx, self.Nctrl * self.NintPerCtrl - 1)),
            entry(f"X_Coll", shape=(self.model.nx, self.d_coll * self.Nctrl * self.NintPerCtrl)),
            entry(f"V_Coll", shape=(self.model.nx, self.d_coll * self.Nctrl * self.NintPerCtrl))
        ]
        """ The entries of the casadi struct z. Can be extended by inheriting classes. """

        # generate the z-structure
        self.z = struct_symSX(self.z_entries)

        # generate the control matrix
        self.U = ca.SX.sym("U", (self.model.nu, self.Nctrl))

        ####### BUILD THE CYCLE SIMULATION #######

        # log the settings for the cycle simulation
        self.logInfo()

        #########################
        ### Preparation #########
        #########################

        # collect equalities and return states in here
        self.g = []
        self.x_ret = []
        self.tau_cycle_ret = []  # the respective numerical times for each value in x_ret, \in [0,1]

        #####################
        #### x minus k ######
        #####################

        # new xminus
        x_minus_k = self.z["x_minus"]
        self.x_ret.append(x_minus_k)
        self.tau_cycle_ret.append(0)

        x_start = self.z["x_start"]

        # connect all states but the constrained ones
        self.g.append(phaseConditions.Q.T @ (x_minus_k - x_start))

        # phase condition on the integration start point
        self.g.append(self.phaseConditions.phiminus(x_start))

        #####################
        # Micro Integration #
        #####################

        self.cycle_cost = 0  # collect the cycle cost

        # Construct the micro-integration
        logging.debug("Constructing SingleShooting Micro-Integration ...")

        # Set up integrator
        self.integrator = RK4.explicitFunction(self.model.f, self.model.x, self.model.u, quad=self.stageCostFunction)

        # collect piecewise output functions and edges
        _piecewise_funcs = []
        _piecewise_edges = []
        _sigma = ca.SX.sym("sigma")  # local numerical time, used for output function

        logging.debug("Constructing Collocation Micro-Integration ...")

        # start integration at x_minus
        x_micro_m = self.z["x_start"]
        T = self.z["T"]

        # set up integrator
        h_tau = 1 / (self.NintPerCtrl * self.Nctrl)  # integrator step
        self.collTimes = np.array(ca.collocation_points(self.d_coll, type))
        RKInt = OthorgonalCollocation(self.collTimes)
        polyFunc = RKInt.getPolyEvalCAExpression(model.nx, includeZero=True)
        c, A, b = RKInt.getButcher()

        # Micro-Integrate Cycle
        for m in range(self.Nctrl):

            # logging.info(f"\t Control {m+1}/{Nctrl}")

            for n in range(self.NintPerCtrl):
                # logging.info(f"Control: {m}/{Nctrl}, Int step: {n}/{NintPerCtrl}")

                # State at collocation points
                Xs = []
                Vs = []

                # create RK variables and bound them
                for i in range(self.d_coll):
                    # index for the current variable
                    current_index_COLL = i + n * self.d_coll + m * self.NintPerCtrl * self.d_coll  # index in array X_Coll
                    # logging.info(f"X_COLL Index:  {current_index_COLL}")

                    # differential variables
                    Xj = self.z["X_Coll", :, current_index_COLL]
                    Xs.append(Xj)
                    self.x_ret.append(Xj)
                    self.tau_cycle_ret.append((m * self.NintPerCtrl + n) * h_tau + h_tau * c[i])

                    self.lbz["X_Coll", :, current_index_COLL] = self.model.lbx
                    self.ubz["X_Coll", :, current_index_COLL] = self.model.ubx

                    # slope variables
                    Vj = self.z["V_Coll", :, current_index_COLL]
                    Vs.append(Vj)

                    # bounds of the slope variable leave them at -inf, +inf
                    self.lbz["V_Coll", :, current_index_COLL] = self.model.lbv
                    self.ubz["V_Coll", :, current_index_COLL] = self.model.ubv

                # RK Integrator equations
                q_end = 0
                Xk_end = x_micro_m
                Xjs = []
                for i in range(self.d_coll):
                    x = Xs[i]
                    v = Vs[i]
                    u = self.U[:, m]

                    # Append collocation equations
                    self.g.append(T * h_tau * self.model.f(x, u) - v)

                    # equation for rk
                    xj_bar = x_micro_m
                    for j in range(self.d_coll):
                        xj_bar = xj_bar + A[i, j] * Vs[j]
                    Xjs.append(xj_bar)
                    self.g.append(xj_bar - x)

                    # Add contribution to the end state
                    Xk_end = Xk_end + b[i] * Vs[i]
                    q_end = q_end + b[i] * self.stageCostFunction(x, u)

                self.cycle_cost += q_end

                # create multiple shooting node at end of collocation interval
                current_index_MS = m * self.NintPerCtrl + n

                # add piecewise values
                _sigma_local = (m * self.NintPerCtrl + n) * h_tau
                _intermediateValue = polyFunc((_sigma - _sigma_local) * h_tau,
                                              ca.horzcat(*([x_micro_m] + Xjs)))  # might be broken
                _piecewise_funcs.append(_intermediateValue)
                _piecewise_edges.append(_sigma_local)

                # dont create new node in last interval
                if current_index_MS < self.NintPerCtrl * self.Nctrl - 1:

                    Xk = self.z["X_Nodes", :, current_index_MS]
                    self.lbz["X_Nodes", :, current_index_MS] = self.model.lbx
                    self.ubz["X_Nodes", :, current_index_MS] = self.model.ubx

                    self.x_ret.append(Xk)
                    self.tau_cycle_ret.append((m * self.NintPerCtrl + n + 1) * h_tau)

                    # constrain end of intervall to this variable
                    self.g.append(Xk_end - Xk)

                    x_micro_m = Xk
                else:
                    # will be constrained to x_plus in a moment
                    x_micro_m = Xk_end

        _piecewise_edges.append(1 + 1E-12)  # add the last edge, a bit shifted to allow sigma to be 1 in output function

        integrationEndpoint = x_micro_m

        #####################
        #### x plus k #######
        #####################

        # New x_plus
        x_plus_k = self.z["x_plus"]
        xend = self.z["x_end"]
        self.x_ret.append(x_plus_k)
        self.tau_cycle_ret.append(1)

        # phase condition on integration end point
        self.g.append(self.phaseConditions.phiplus(xend))

        # attach endpoint of micro-integration and integration end point
        self.g.append(xend - integrationEndpoint)

        # attach xplus and integration endpoint in the unconstrained states
        self.g.append(phaseConditions.Q.T @ (x_plus_k - xend))

        # constrain the relation between xplus and xminus in the constrained states, since we know it
        self.g.append(phaseConditions.q.T @ (x_plus_k - x_minus_k) - (phaseConditions.bplus - phaseConditions.bminus))

        # output format
        self.x_ret: List[ca.SX] = self.x_ret

        # output functions
        self.f_cycleConditions: ca.Function = ca.Function("CycleConditions", [self.z, self.U], [ca.vertcat(*self.g)])
        self.f_cycleCost: ca.Function = ca.Function("CycleCost", [self.z, self.U], [self.cycle_cost])
        self.f_getX = ca.Function("GetCycleTrajectorie", [self.z, self.U], [ca.horzcat(*self.x_ret)])
        self.f_getLocalTau = ca.Function("GetCycleTaus", [self.z, self.U], [ca.vertcat(*self.tau_cycle_ret)])
        self.f_getT = ca.Function("GetCyclePeriod", [self.z, self.U], [self.z["T"]])
        self.f_outputMap = ca.Function("OutputMap", [_sigma, self.z, self.U],
                                       [constructPiecewiseCasadiExpression(_sigma, _piecewise_edges, _piecewise_funcs)],
                                       ["sigma", "z", "U"], ["Phi"])

    def z0(self, x_minus_guess: ca.DM) -> ca.tools.struct:
        """ Get an initialization guess for the cycle simulation.
        Returns a struct with the same structure as z.
        Can be potentially overwritten by inheriting classes.
        """
        z0 = self.z(0)
        z0["x_minus"] = x_minus_guess
        z0["x_start"] = x_minus_guess
        z0["x_end"] = x_minus_guess
        z0["T"] = self.model.getApproxPeriod(x_minus_guess)

        # perform a forward simulation to find the inital points w0
        Xinit = self.model.forwardSimSimple(x_minus_guess, self.model.getApproxPeriod((x_minus_guess)),
                                            (self.d_coll + 1) * self.NintPerCtrl * self.Nctrl,
                                            np.zeros(self.model.u.shape))

        # set X_Nodes initial values
        for m in range(self.Nctrl):
            for n in range(self.NintPerCtrl):
                current_index_MS = m * self.NintPerCtrl + n

                for i in range(self.d_coll):
                    # index for the current variable
                    current_index_COLL = i + n * self.d_coll + m * self.NintPerCtrl * self.d_coll  # index in array X_Coll
                    current_index_INIT = i + (n + m * self.NintPerCtrl) * (self.d_coll + 1)  # index in array Xinit

                    z0["X_Coll", :, current_index_COLL] = Xinit[:, [current_index_INIT]]
                    z0["V_Coll", :, current_index_COLL] = np.zeros(self.model.x.shape)

                # dont create new node in last interval
                if current_index_MS < self.NintPerCtrl * self.Nctrl - 1:
                    z0["X_Nodes", :, current_index_MS] = ca.DM(Xinit[:, [current_index_INIT + 1]])

        z0["x_plus"] = ca.DM(Xinit[:, -1])

        return z0

    def getHigherAccuracyCopy(self) -> "CycleCollocation":
        return CycleCollocation(self.model,
                                self.phaseConditions,
                                self.Nctrl,
                                self.stageCostFunction,
                                self.NintPerCtrl * 5,
                                self.d_coll,
                                self.type)

import logging
from typing import List, Tuple

import casadi as ca
from casadi.tools import entry, struct_symSX

from Core.controlparameterization import ControlParameterization, \
    PiecewiseConstantControlParametrization
from Core.cycle import Cycle
from Core.integrators import RK4, ButcherTableau, RungeKuttaIntegrator
from Core.model import Model
from Core.parameters import Parameters
from Core.tools import constructPiecewiseCasadiExpression


class CycleShootingFixedPeriod(Cycle):
    """ Single Shooting Cycle Simulation """

    def __init__(self,
                 model: Model,
                 fixedPeriod: float,
                 controlParameterization: ControlParameterization = None,
                 stageCostFunction: ca.Function = None,
                 Nint: int = 20,
                 shooting_type='single',
                 initializationFunction=None, parameters: Parameters = Parameters()):

        # fill in the stage cost function if not given
        if stageCostFunction is None:
            stageCostFunction = ca.Function('stageCostFunction', [model.x, model.tau] + ([] if controlParameterization is None else [model.u]), [0])

        super().__init__(model, stageCostFunction, controlParameterization,parameters)

        # single or multiple shooting?
        self.shooting_type = shooting_type
        assert self.shooting_type in ['single', 'multiple'], "shooting_type has to be 'single' or 'multiple'"

        if initializationFunction is None:
            initializationFunction = ca.Function('initializationFunction', [self.model.x, self.model.tau],
                                                 [self.model.x])
        assert initializationFunction.sx_out()[
                   0].shape == self.model.x.shape, "initializationFunction has to return a vector of size nx"
        assert initializationFunction.sx_in()[1].shape == (
            1, 1), "initializationFunction has to take a single argument (tau) in [0,1]"
        self.initializationFunction = initializationFunction

        # duration of the cycle (usually 1)
        self.fixedPeriod = fixedPeriod

        # number of integration intervals
        self.Nint = Nint
        N_Nodes = {'single': 2, 'multiple': self.Nint + 1}[self.shooting_type]  # number of integration nodes per cycle

        # generate the z-structure
        z_entries = [
            entry(f"x_plus", struct=self.model.x_struct),
            entry(f"x_minus", struct=self.model.x_struct),
            entry(f"X_Nodes", struct=self.model.x_struct, repeat=N_Nodes),
        ]
        self.z = struct_symSX(z_entries)

        # generate structure for the constraints of the cycle conditions
        self.cycleConditions_struct = struct_symSX([
            entry(f"CC_connect_start", struct=self.model.x_struct),
            entry(f"CC_integration", struct=self.model.x_struct, repeat=N_Nodes - 1),
            entry(f"CC_connect_end", struct=self.model.x_struct)
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

        ####### BUILD THE CYCLE SIMULATION #######

        # log the settings for the cycle simulation
        self.logInfo()

        #########################
        ### Preparation #########
        #########################

        # return states in here
        self.x_ret = []
        self.tau_cycle_ret = []  # the respective numerical times for each value in x_ret, \in [0,1]

        #####################
        #### x minus k ######
        #####################
        # x minus
        x_minus_k = self.z["x_minus"]

        # start point of the integration
        x_start = self.z["X_Nodes", 0]
        self.tau_cycle_ret.append(0)
        self.x_ret.append(x_start)

        cycleConditions['CC_connect_start'] = x_minus_k - x_start

        #####################
        # Micro Integration #
        #####################

        self.cycle_cost = 0  # collect the cycle cost

        # Construct the micro-integration
        logging.debug(f"Constructing {self.shooting_type}-Shooting Micro-Integration ...")

        # start integration at x_minus
        x_micro_m = x_start

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
            _integrator_expressions = [u] + model.symbolicParams
        else:
            _integrator_expressions = model.symbolicParams

        # integrator for this interval, with different control expression
        self.integrator = RungeKuttaIntegrator(RK4(), self.model.f, expressions= _integrator_expressions ,
                                               constant_syms=[tau0] + parameters.syms_list,
                                               quad=self.stageCostFunction).explicitFunction()

        # Micro-Integrate Cycle
        for n in range(self.Nint):
            _sigma_local = n * h_tau  # start of this control interval in cycle time

            # intermediate values for continuous plotting
            _intermediateValue = self.integrator(x_micro_m,
                                                 tau0 + _sigma_local,
                                                 fixedPeriod * (_sigma_SX - _sigma_local),
                                                 *([tau0] + parameters.syms_list))['x_f']
            _piecewise_funcs.append(_intermediateValue)
            _piecewise_edges.append(_sigma_local)

            integrator_result = self.integrator(x_micro_m, tau0 + _sigma_local,
                                                    fixedPeriod * h_tau,
                                                    *([tau0] + parameters.syms_list))  # Integration Result of this step
            x_micro_m, quadrature = integrator_result['x_f'], integrator_result['q_f']

            # cycle cost is scaled with the period duration
            self.cycle_cost += quadrature

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

        # attach xplus and integration endpoint
        cycleConditions['CC_integration', -1] = xend - x_micro_m
        cycleConditions['CC_connect_end'] = x_plus_k - xend

        # output format
        self.x_ret: List[ca.SX] = self.x_ret

        # control parameters list
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

        # initialize the multiple shooting nodes
        _tau_grid_Nodes = ca.linspace(0, self.fixedPeriod, self.Nint + 1)
        _h = 1 / self.Nint
        x_init_Nodes = self.initializationFunction.map(self.Nint + 1)(xminusguess, _tau_grid_Nodes)
        z0["X_Nodes"] = ca.horzsplit(x_init_Nodes)

        return z0

    def revertTimeConstraintsandInit(self, z0, lbz, ubz) -> Tuple[ca.DM, ca.DM, ca.DM]:
        z0["T"] = -z0["T"]
        lbz["T"], ubz["T"] = -self.ubz["T"], -self.lbz["T"]
        return z0, lbz, ubz

    def getHigherAccuracyCopy(self, Nint_multiplier: int = 5):
        return CycleShootingFixedPeriod(model=self.model,
                                        fixedPeriod=self.fixedPeriod,
                                        controlParameterization=self.controlParameterization,
                                        stageCostFunction=self.stageCostFunction,
                                        Nint=self.Nint * Nint_multiplier,  # 5 times more integration points
                                        shooting_type=self.shooting_type)


class CycleSingleShootingFixedPeriod(CycleShootingFixedPeriod):
    """ Single Shooting Cycle Simulation """

    def __init__(self,
                 model: Model,
                 controlParameterization: ControlParameterization = None,
                 stageCostFunction: ca.Function = None, Nint: int = 20, parameters: Parameters = Parameters(),
                 initializationFunction=None):
        super().__init__(model,
                         fixedPeriod=1,
                         controlParameterization=controlParameterization,
                         stageCostFunction=stageCostFunction,
                         Nint=Nint,
                         shooting_type='single',
                         parameters=parameters,
                         initializationFunction=initializationFunction)


class CycleMultipleShootingFixedPeriod(CycleShootingFixedPeriod):
    """ Multiple Shooting Cycle Simulation """

    def __init__(self,
                 model: Model,
                 controlParameterization: ControlParameterization = None,
                 stageCostFunction: ca.Function = None, Nint: int = 20, initializationFunction=None,
                 parameters: Parameters = Parameters()):
        super().__init__(model,
                         fixedPeriod=1,
                         controlParameterization=controlParameterization,
                         stageCostFunction=stageCostFunction,
                         Nint=Nint,
                         shooting_type='multiple',
                         initializationFunction=initializationFunction,
                         parameters=parameters)


class CycleCollocationFixedPeriod(Cycle):
    """ Collocation Cycle Simulation """

    def __init__(self,
                 model: Model,
                 fixedPeriod: float,
                 controlParameterization: ControlParameterization = None,
                 butcherTableau=RK4(),
                 stageCostFunction: ca.Function = None, Nint: int = 20,
                 initializationFunction=None, parameters: Parameters = Parameters()):

        # fill in the stage cost function if not given
        if stageCostFunction is None:
            stageCostFunction = ca.Function('stageCostFunction', [model.x, model.tau, model.u], [0])

        super().__init__(model, stageCostFunction, controlParameterization,parameters)

        if initializationFunction is None:
            initializationFunction = ca.Function('initializationFunction', [self.model.x, self.model.tau],
                                                 [self.model.x])
        assert initializationFunction.sx_out()[
                   0].shape == self.model.x.shape, "initializationFunction has to return a vector of size nx"
        assert initializationFunction.sx_in()[1].shape == (
            1, 1), "initializationFunction has to take a single argument (tau) in [0,1]"
        self.initializationFunction = initializationFunction

        # duration of the cycle (usually 1)
        self.fixedPeriod = fixedPeriod

        # number of integration intervals
        self.Nint = Nint
        N_Nodes = self.Nint + 1  # number of integration nodes per cycle

        self.butcherTableau: ButcherTableau = butcherTableau

        # the numerical time at which the cycle STARTS
        tau0 = ca.SX.sym("tau0")

        # control expression for the cycle
        u = controlParameterization.u_f(self.model.tau, model.tau - tau0, *parameters.syms_list)

        # get the structure of the algebraic variables of the integrator
        integrator = RungeKuttaIntegrator(self.butcherTableau, model.f,
                                          expressions=[u] + model.symbolicParams,
                                          constant_syms=[tau0] + parameters.syms_list,
                                          quad=self.stageCostFunction)
        G_irk, Y_irk, z_irk, G_irk_struct = integrator.implicitFunction()

        # intermediate value integrator
        _RK4_explicit = RungeKuttaIntegrator(RK4(), self.model.f,
                                             expressions=[u] + model.symbolicParams,
                                             constant_syms=[tau0] + parameters.syms_list,
                                             quad=self.stageCostFunction).explicitFunction()

        # generate the z-structure
        z_entries = [
            entry(f"x_plus", struct=self.model.x_struct),
            entry(f"x_minus", struct=self.model.x_struct),
            entry(f"X_Nodes", struct=self.model.x_struct, repeat=N_Nodes),
            entry(f'z_irk', struct=z_irk, repeat=Nint),
        ]
        self.z = struct_symSX(z_entries)

        # generate structure for the constraints of the cycle conditions
        self.cycleConditions_struct = struct_symSX([
            entry(f"CC_connect_start", struct=self.model.x_struct),
            entry(f"CC_integration_irk", struct=G_irk_struct, repeat=Nint),
            entry(f"CC_connect_end", struct=self.model.x_struct)
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
        # x minus
        x_minus_k = self.z["x_minus"]

        # start point of the integration
        x_start = self.z["X_Nodes", 0]
        self.tau_cycle_ret.append(0)
        self.x_ret.append(x_start)

        cycleConditions['CC_connect_start'] = x_minus_k - x_start

        #####################
        # Micro Integration #
        #####################

        self.cycle_cost = 0  # collect the cycle cost

        # Construct the micro-integration
        logging.debug(f"Constructing Collocation Micro-Integration ...")

        # Set up integrator
        h_tau = 1 / self.Nint  # integrator step

        # collect piecewise output functions and edges
        _piecewise_funcs = []
        _piecewise_edges = []
        _sigma_SX = ca.SX.sym("sigma")  # local numerical time, used for output function

        # Micro-Integrate Cycle
        for n in range(self.Nint):
            _sigma_local = n * h_tau  # start of this control interval in cycle time

            # implicit integration
            startNode = self.z["X_Nodes", n]  # MS node at the start of the interval
            endNode = self.z["X_Nodes", n + 1]  # MS node at the end of the interval
            z_irk_n = self.z["z_irk", n]  # get the irk variables for this interval

            # intermediate values for continuous plotting
            _intermediateValue = _RK4_explicit(startNode,
                                               tau0 + _sigma_local,
                                               fixedPeriod * (_sigma_SX - _sigma_local),
                                               *([tau0] + parameters.syms_list))['x_f']

            _piecewise_funcs.append(_intermediateValue)
            _piecewise_edges.append(_sigma_local)
            # TODO: use the implicit integrator for this (or evaluate collocation poly)

            cycleConditions['CC_integration_irk', n] = G_irk(startNode,
                                                             endNode,
                                                             tau0 + _sigma_local,
                                                             fixedPeriod * h_tau,
                                                             z_irk_n,
                                                             *([tau0] + parameters.syms_list))

            quadrature = Y_irk(startNode, tau0 + _sigma_local, h_tau, z_irk_n, *([tau0] + parameters.syms_list))['q_f']

            self.cycle_cost += quadrature

            # collect the variables that are used to plot the cycle
            self.x_ret.append(startNode)
            self.tau_cycle_ret.append((n + 1) * h_tau)

        _piecewise_edges.append(1 + 1E-12)  # add the last edge, a bit shifted to allow sigma to be 1 in output function

        #####################
        #### x plus k #######
        #####################

        # New x_plus
        x_plus_k = self.z["x_plus"]
        xend = self.z["X_Nodes", -1]
        self.x_ret.append(x_plus_k)
        self.tau_cycle_ret.append(1)

        # attach xplus and integration endpoint
        # cycleConditions['CC_integration', -1] = xend - x_micro_m
        cycleConditions['CC_connect_end'] = x_plus_k - xend

        # output format
        self.x_ret: List[ca.SX] = self.x_ret

        # control parameters list
        # U_list_SX = self.controlParameterization.U_list_SYM

        # output functions
        # self.f_cycleConditions: ca.Function = ca.Function("CycleConditions", [self.z, self.model.tau, *U_list_SX],
        #                                                   [cycleConditions])
        # self.f_cycleCost: ca.Function = ca.Function("CycleCost", [self.z, self.model.tau, *U_list_SX],
        #                                             [self.cycle_cost])
        # self.f_getX = ca.Function("GetCycleTrajectorie", [self.z, self.model.tau, *U_list_SX],
        #                           [ca.horzcat(*self.x_ret)])
        # self.f_getLocalTau = ca.Function("GetCycleTaus", [self.z, self.model.tau, *U_list_SX],
        #                                  [ca.vertcat(*self.tau_cycle_ret)])
        # self.f_getT = ca.Function("GetCyclePeriod", [self.z, self.model.tau, *U_list_SX], [fixedPeriod])
        # self.f_outputMap = ca.Function("OutputMap", [_sigma_SX, self.z, self.model.tau, *U_list_SX],
        #                                [constructPiecewiseCasadiExpression(_sigma_SX, _piecewise_edges,
        #                                                                    _piecewise_funcs)])

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
                                       ["sigma", "z", "tau_0"] + parameters.syms_list_names, ['output function'])

        self.f_getX = ca.Function("GetCycleTrajectorie", [self.z, tau0] + parameters.syms_list,
                                  [ca.horzcat(*self.x_ret)], )
        self.f_getT = ca.Function("GetCyclePeriod", [self.z, tau0] + parameters.syms_list, [1])

        self.f_getLocalTau = ca.Function("GetCycleTaus", [self.z, tau0] + parameters.syms_list,
                                         [ca.vertcat(*self.tau_cycle_ret)])

    def z0(self, xminusguess: ca.DM) -> ca.tools.struct:
        """ Get an initialization guess for the cycle simulation.
        Returns a struct with the same structure as z.
        Can be potentially overwritten by inheriting classes.
        """
        z0 = self.z(0)

        _tau_grid_Nodes = ca.linspace(0, self.fixedPeriod, self.Nint + 1)
        _h = 1 / self.Nint
        _tau_interval_irk = self.butcherTableau.c * _h
        # _tau_grid_irk = ca.vertcat(*[_tau_node + _tau_interval_irk for _tau_node in ca.vertsplit(_tau_grid_Nodes[:-1])])

        x_init_Nodes = self.initializationFunction.map(self.Nint + 1)(xminusguess, _tau_grid_Nodes)
        # x_init_irk = self.initializationFunction.map(_tau_grid_irk.shape[0])(xminusguess, _tau_grid_irk).full()

        z0["x_plus"] = xminusguess
        z0["x_minus"] = xminusguess
        z0["X_Nodes"] = ca.horzsplit(x_init_Nodes)

        # initialize the irk variables
        for intervalIndex, tau0 in enumerate(ca.vertsplit(_tau_grid_Nodes[:-1])):
            x_irk_init = self.initializationFunction.map(self.butcherTableau.d)(xminusguess, tau0 + _tau_interval_irk)
            v_irk_init = self.model.f.map(self.butcherTableau.d)(x_irk_init, tau0 + _tau_interval_irk, [],
                                                                 *self.model.symbolicParamsDefaultValues)
            z0['z_irk', intervalIndex, 'x_irk', :] = ca.horzsplit(x_irk_init.full())
            z0['z_irk', intervalIndex, 'v_irk', :] = ca.horzsplit(v_irk_init.full())
        # z0['z_irk', :, 'v_irk'] = 0

        return z0

    def revertTimeConstraintsandInit(self, z0, lbz, ubz) -> Tuple[ca.DM, ca.DM, ca.DM]:
        z0["T"] = -z0["T"]
        lbz["T"], ubz["T"] = -self.ubz["T"], -self.lbz["T"]
        return z0, lbz, ubz

    def getHigherAccuracyCopy(self, Nint_multiplier: int = 5):
        return CycleCollocationFixedPeriod(model=self.model,
                                           fixedPeriod=self.fixedPeriod,
                                           controlParameterization=self.controlParameterization,
                                           stageCostFunction=self.stageCostFunction,
                                           Nint=self.Nint * Nint_multiplier,  # 5 times more integration points
                                           )

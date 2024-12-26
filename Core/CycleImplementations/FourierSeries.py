import logging
from typing import List

import casadi as ca
import numpy as np
from casadi.tools import entry, struct_symSX

from Core.controlparameterization import ControlParameterization, ZeroControlParameterization
from Core.cycle import Cycle
from Core.model import Model
from Core.parameters import Parameters
from Core.tools import computeTrajectoryFourierCoefficients


class CycleFourierSeries(Cycle):
    """ Fourier Cycle Simulation for periodic solutions. """

    def __init__(self, model: Model, Nharmonics: int,
                 controlParameterization: ControlParameterization = ZeroControlParameterization(),
                 stageCostFunction: ca.Function = None,
                 parameters: Parameters = Parameters()):

        # fill in the stage cost function if not given
        if stageCostFunction is None:
            stageCostFunction = ca.Function('stageCostFunction', [model.x, model.tau, model.u], [0])

        self.Nharmonics = Nharmonics  # number of harmonics

        super().__init__(model, stageCostFunction, controlParameterization,parameters)

        self.z_entries = [entry(f"x_plus", struct=self.model.x_struct),
                          entry(f"x_minus", struct=self.model.x_struct),
                          entry(f"a", shape=(self.model.nx, self.Nharmonics + 1)),  # a[:,0] is the constant term
                          entry(f"b", shape=(self.model.nx, self.Nharmonics + 1)),  # b[:,0] is the linear term
                          ]

        # generate the z-structure
        self.z = struct_symSX(self.z_entries)

        # generate structure for the constraints of the cycle conditions
        self.M = 2 * self.Nharmonics  # number of collocation points
        self.cycleConditions_struct = struct_symSX([
            entry(f"CC_connect_start", struct=self.model.x_struct),
            entry(f"CC_connect_end", struct=self.model.x_struct),
            entry(f"CC_collocation", struct=self.model.x_struct, repeat=self.M),
            entry(f"CC_b0iszero", struct=self.model.x_struct, repeat=1),
        ])
        cycleConditions = ca.tools.struct_SX(self.cycleConditions_struct)

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

        # new xminus
        x_minus_k = self.z["x_minus"]
        self.tau_cycle_ret.append(0)
        self.x_ret.append(x_minus_k)

        #####################
        # Micro Integration #
        #####################

        logging.debug("Constructing FourierSeries Micro-Integration ...")
        # start integration at x_minus

        # define the collocation points
        # self.sigmas = ca.DM(np.arange(0, self.M) / self.M)
        self.sigmas = ca.DM(np.random.rand(self.M))
        self.omegas = ca.DM(np.arange(0, self.Nharmonics + 1) * 2 * np.pi)

        # intermediate variables for easier typing
        a = self.z["a"]
        b = self.z["b"]

        sigmaSX = ca.SX.sym("sigma")
        _poly_states = a @ ca.cos(self.omegas * sigmaSX) \
                       + b @ ca.sin(self.omegas * sigmaSX)
        self.poly_states = ca.Function("poly_states", [self.z, sigmaSX], [_poly_states])
        self.poly_states_der = ca.Function("poly_states_der", [self.z, sigmaSX],
                                           [ca.jacobian(self.poly_states(self.z, sigmaSX), sigmaSX)])

        # constraint to ensure that b0 is zero
        cycleConditions['CC_b0iszero'] = self.z['b', :, 0]

        # connect start point of cycle
        cycleConditions['CC_connect_start'] = self.poly_states(self.z, 0) - self.z[
            "x_minus"]  # the sum of all a_k is x_minus

        # iterate over all collocation points
        for m in range(self.M):
            _sigma_m = self.sigmas[m]
            _poly_states_local = self.poly_states(self.z, _sigma_m)
            _poly_states_der_local = self.poly_states_der(self.z, _sigma_m)
            cycleConditions['CC_collocation', m] = _poly_states_der_local - self.model.f(_poly_states_local,
                                                                                         self.model.tau, ca.DM.zeros(
                    self.model.u.shape))

        #####################
        #### x plus k #######
        #####################

        self.x_ret.append(self.z["x_plus"])
        self.tau_cycle_ret.append(1)

        # attach endpoint of micro-integration and xplus
        cycleConditions['CC_connect_end'] = self.z["x_plus"] - self.poly_states(self.z, 1)

        # output format
        self.x_ret: List[ca.SX] = self.x_ret

        # output functions
        self.f_cycleConditions: ca.Function = ca.Function("CycleConditions",
                                                          [self.z, self.model.tau] + parameters.syms_list,
                                                          [cycleConditions], ["z", "tau"] + parameters.syms_list_names,
                                                          ["CC"])
        # self.f_getX = ca.Function("GetCycleTrajectorie", [self.z, self.U], [ca.horzcat(*self.x_ret)])
        # self.f_getLocalTau = ca.Function("GetCycleTaus", [self.z, self.U], [ca.vertcat(*self.tau_cycle_ret)])

        _sigma = ca.SX.sym("sigma")  # local numerical time, used for output function
        self.f_outputMap = ca.Function("OutputMap", [_sigma, self.z, self.model.tau] + parameters.syms_list,
                                       [self.poly_states(self.z, _sigma)],
                                       ["sigma", "z", "tau"] + parameters.syms_list_names, ["Phi"])

    def z0(self, xminusguess: ca.DM) -> ca.tools.struct:
        """ Get an initialization guess for the cycle simulation.
        Returns a struct with the same structure as z.
        Can be potentially overwritten by inheriting classes.
        """
        z0 = self.z(0)

        # perform a forward simulation over one cycle
        sim = self.model.forwardSimSimple(xminusguess, 1, self.M)[:, :self.M]
        a, b, _ = computeTrajectoryFourierCoefficients(sim)

        z0["a"] = ca.DM(a)
        z0["b"] = ca.DM(b)
        z0["x_minus"] = xminusguess
        z0["x_plus"] = xminusguess

        return z0

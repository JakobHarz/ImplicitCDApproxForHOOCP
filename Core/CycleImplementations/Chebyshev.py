import logging

import casadi as ca
import numpy as np
from casadi.tools import entry, struct_symSX

from Core.controlparameterization import ControlParameterization, ZeroControlParameterization
from Core.cycle import Cycle
from Core.model import Model
from Core.parameters import Parameters
from Core.polynomials import ChebyshevPoly


class CycleChebyshev(Cycle):
    """ Spectral Chebyshev Periodic Cycle Simulation """

    def __init__(self, model: Model,
                 d: int,
                 periodic: bool = False,
                 controlParameterization: ControlParameterization = ZeroControlParameterization(),
                 parameters: Parameters = Parameters(),
                 stageCostFunction: ca.Function = None):

        # fill in the stage cost function if not given
        if stageCostFunction is None:
            stageCostFunction = ca.Function('stageCostFunction', [model.x, model.tau, model.u], [0])
        super().__init__(model, stageCostFunction, controlParameterization, parameters)
        self.d = d  # number of collacation points
        self.z_entries = [
            entry(f"x_plus", struct=self.model.x_struct),
            entry(f"x_minus", struct=self.model.x_struct),
            entry(f"coeffs", shape=[self.model.nx, self.d]),  # coefficients of the Chebyshev polynomial
        ]

        # generate the z-structure
        self.z = struct_symSX(self.z_entries)

        # generate structure for the constraints of the cycle conditions
        entries_CC = [entry(f"CC_connect_start", struct=self.model.x_struct),
                      entry(f"CC_connect_end", struct=self.model.x_struct)]
        if periodic:
            entries_CC += [entry(f"CC_collocation", struct=self.model.x_struct, repeat=[self.d - 2]),
                           entry(f"CC_periodicity", struct=self.model.x_struct)]
        else:
            entries_CC += [entry(f"CC_collocation", struct=self.model.x_struct, repeat=[self.d - 1])]
        self.cycleConditions_struct = struct_symSX(entries_CC)
        cycleConditions = ca.tools.struct_SX(self.cycleConditions_struct)

        ####### BUILD THE CYCLE SIMULATION #######

        # log the settings for the cycle simulation
        self.logInfo()

        #####################
        # Micro Integration #
        #####################
        logging.debug("Constructing Chebyshev Micro-Integration ...")

        # create a chebyshev polynomial
        poly = ChebyshevPoly(d)

        # define the collocation points
        # (one less because of additional periodicity constraint)
        self.collPoints = ChebyshevPoly.collTimes(d - 2 if periodic else d - 1)
        self.poly_states = ca.Function("poly_states", [self.z, poly.tau], [self.z['coeffs'] @ poly.basis(poly.tau)])
        self.poly_states_der = ca.Function("poly_states_der", [self.z, poly.tau],
                                           [self.z['coeffs'] @ poly.basis_der(poly.tau)])

        # connect start point of cycle
        cycleConditions['CC_connect_start'] = self.poly_states(self.z, 0) - self.z["x_minus"]

        # iterate over all collocation points
        # (enforce not the last point, as it is already covered by the periodicity, otherwise maybe LICQ is violated)
        for index, collPoint in enumerate(self.collPoints.full()):
            _poly_states_local = self.poly_states(self.z, collPoint)
            _poly_states_der_local = self.poly_states_der(self.z, collPoint)
            cycleConditions['CC_collocation', index] = _poly_states_der_local - self.model.f(_poly_states_local,
                                                                                             self.model.tau,
                                                                                             ca.DM.zeros(
                                                                                                 self.model.u.shape))

        # periodicity constraint
        if periodic: cycleConditions['CC_periodicity'] = self.poly_states(self.z, 1) - self.poly_states(self.z, 0)

        #####################
        #### x plus k #######
        #####################

        # attach endpoint of micro-integration and xplus
        cycleConditions['CC_connect_end'] = self.z["x_plus"] - self.poly_states(self.z, 1)

        # control parameters list
        U_list_SX = self.controlParameterization.U_list_SYM

        # output functions
        self.f_cycleConditions: ca.Function = ca.Function("CycleConditions", [self.z, self.model.tau, *U_list_SX],
                                                          [cycleConditions])
        _sigma = ca.SX.sym("sigma")  # local numerical time, used for output function
        self.f_outputMap = ca.Function("OutputMap", [_sigma, self.z, self.model.tau, *U_list_SX],
                                       [self.poly_states(self.z, _sigma)])

    def z0(self, xminusguess: ca.DM) -> ca.tools.struct:
        """ Get an initialization guess for the cycle simulation.
        Returns a struct with the same structure as z.
        Can be potentially overwritten by inheriting classes.
        """
        z0 = self.z(0)

        # perform a forward simulation over one cycle

        z0['x_minus'] = xminusguess
        z0['x_plus'] = xminusguess

        # estimate the coefficients from a forward simulation
        sim = self.model.forwardSimSimple(xminusguess, 1, self.d - 2)  # then sim contains d-1 points
        tau_sim = np.linspace(0, 1, self.d - 1)
        coeffs_opt = np.polynomial.chebyshev.chebfit(tau_sim * 2 - 1, sim.T,
                                                     self.d - 1).T  # fit, shift the time to [-1,1]

        z0["coeffs"] = coeffs_opt

        return z0

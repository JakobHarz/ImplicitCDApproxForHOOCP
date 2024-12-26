from Core.cycle import Cycle
import casadi as ca
from casadi.tools import struct_symSX, entry, struct_SX

from Core.tools import getCDCoefficients
from Core.parameters import Parameters
from Core.tools import constructPiecewiseCasadiExpression


class AverageDynamicsApproximation:

    def __init__(self, cycleSim: Cycle, Ncycles, parameters):
        """

        :param parameters:
        :param cycleSim:
        """

        self.cycleSim: Cycle = cycleSim
        self.Ncycles = Ncycles
        self.Z: struct_symSX = None
        self.G: ca.Function = None
        self.F: ca.Function = None
        self.parameters = parameters

    @property
    def F(self) -> ca.Function:
        """ A function (X, Z, tau, *param_list) -> (dot_X) that approximates the average dynamics of the system at some
        macro-integration point X at some integration time tau, using the algebraic variables Z."""
        return self._F

    @F.setter
    def F(self, newFunction: ca.Function):
        self._F = newFunction

    @property
    def G(self) -> ca.Function:
        """ A function (X, Z, tau, *param_list) -> 0 that gives the alebraic equations for the approximation of the
        average dynamics at some point X at some integration time tau, using the algebraic variables Z."""
        # experimental feature: return a casadi struct instead of SX

        return self._G

    @G.setter
    def G(self, newFunction: ca.Function):
        self._G = newFunction

    def copy(self, cycleSim: Cycle):
        """
        Creates a copy of the approximation with a different cycle simulation.
        This is needed to efficiently go through different parameters in experiments.
        :param cycleSim:
        :return:
        """
        raise NotImplementedError('Implement this in the child class')

    @property
    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return self.name

    def getZ0(self, x0bar: ca.DM):
        """
        Returns the initial value for the algebraic variables at some macro-integration point
        :param x0bar:
        :return:
        """
        Z0 = self.Z(0)
        Z0['Z_cycles', :] = self.cycleSim.z0(x0bar)
        return Z0


class LIPS2(AverageDynamicsApproximation):

    def __init__(self, cycleSim: Cycle):
        super().__init__(cycleSim, 2, parameters)

        self.Z = struct_symSX([entry('Z_cycles', struct=cycleSim.z, repeat=self.Ncycles)])  # algebraic variables
        self._x = self.cycleSim.model.x_struct

        # unpack control parameterization
        U_list_SX = cycleSim.controlParameterization.U_list_SYM

        # symbolic variable for the time, needed in eval of the control parameterization
        tau = ca.SX.sym('tau')

        # retreive the X'ks
        Q0 = self.Z['Z_cycles', 0, 'x_minus']
        Q1 = self.Z['Z_cycles', 0, 'x_plus']
        Q2 = self.Z['Z_cycles', 1, 'x_plus']

        # build function to approximate the average dynamics
        self.F = ca.Function('F', [self._x, self.Z, tau, *U_list_SX], [-3 / 2 * Q0 + 1 / 2 * Q1 + Q2])

        # build constraints struct
        self.constraints_struct = struct_symSX([entry('ADA_connect', shape=cycleSim.model.x.shape),
                                                entry('ADA_CC',
                                                      struct=cycleSim.cycleConditions_struct,
                                                      repeat=self.Ncycles),
                                                entry('ADA_connect_between_cycles', shape= cycleSim.model.x.shape, repeat=self.Ncycles-1)
                                                ])
        constraints = struct_SX(self.constraints_struct)

        # build constraints
        for n in range(self.Ncycles):
            # cycle conditions
            constraints['ADA_CC', n]= self.cycleSim.f_cycleConditions(self.Z['Z_cycles', n], tau, *U_list_SX)

        constraints['ADA_connect_between_cycles', 0] = self.Z['Z_cycles', 1, 'x_minus'] - (3 / 2 * Q0 - 1 / 2 * Q1)  # Q2 = P(3/2*Q0 - 1/2*Q1)

        # connect to integration point
        constraints['ADA_connect'] = self._x - Q0

        self.G = ca.Function('G', [self._x, self.Z, tau, *U_list_SX], [constraints])

    def copy(self, cycleSim: Cycle):
        return LIPS2(cycleSim)


class ForwardDifferences(AverageDynamicsApproximation):

    def __init__(self, cycleSim: Cycle, parameters=Parameters()):
        super().__init__(cycleSim, 1, parameters)

        self.Z = struct_symSX([entry('Z_cycles', struct=cycleSim.z, repeat=self.Ncycles)])  # algebraic variables
        self._x = self.cycleSim.model.x_struct

        # symbolic variable for the time, needed in eval of the control parameterization
        tau = ca.SX.sym('tau')

        # build constraints struct
        self.constraints_struct = struct_symSX([entry('ADA_connect', shape=cycleSim.model.x.shape),
                                                entry('ADA_CC',
                                                      struct=cycleSim.cycleConditions_struct,
                                                      repeat=self.Ncycles)
                                                ])
        constraints = struct_SX(self.constraints_struct)

        # fill constraints expressions (cast to correct struct if neccesary)
        constraints['ADA_connect'] = self._x - self.Z['Z_cycles', 0, 'x_minus']
        constraints['ADA_CC', 0] = self.cycleSim.cycleConditions_struct(
            self.cycleSim.f_cycleConditions(self.Z, tau, *parameters.syms_list))

        # build function to approximate the average dynamics
        self.F = ca.Function('F', [self._x, self.Z, tau, *parameters.syms_list],
                             [self.Z['Z_cycles', 0, 'x_plus'] - self.Z['Z_cycles', 0, 'x_minus']])

        # build constraints function
        self.G = ca.Function('G', [self._x, self.Z, tau, *parameters.syms_list], [constraints])

        # output function for plotting
        _sigma, _tau = ca.SX.sym('sigma'), ca.SX.sym('tau')  # temporary symbols
        _ouputExpression = self.cycleSim.f_outputMap(_sigma, self.Z, _tau, *parameters.syms_list)
        self.f_MicroIntOutputMap = ca.Function('outputFunction', [_sigma, self._x, self.Z, _tau, *parameters.syms_list],
                                               [_ouputExpression], ['sigma', 'X', 'Z', 'tau'] + parameters.syms_list_names, ['output'])

        # stagecost function
        _cycleCostAverage = self.cycleSim.f_cycleCost(self.Z['Z_cycles', 0], _tau, *parameters.syms_list) / 1
        self.f_averageCycleCost = ca.Function('stageCost', [self._x, self.Z, _tau, *parameters.syms_list],
                                              [_cycleCostAverage], ['X', 'Z', 'tau'] + parameters.syms_list_names, ['stageCost'])

    def copy(self, cycleSim: Cycle):
        return ForwardDifferences(cycleSim, parameters= self.parameters)


class CentralDifferences(AverageDynamicsApproximation):

    def __init__(self, cycleSim: Cycle, K: int = 2, parameters: Parameters = Parameters()):
        super().__init__(cycleSim, K - 1, parameters)

        assert K >= 2, 'K has to be at least 2'
        self.deltaTau, self.b, self.c = getCDCoefficients(K)
        self.K = K

        self.Z = struct_symSX([entry('Z_cycles', struct=cycleSim.z, repeat=self.Ncycles)])  # algebraic variables
        self._x = self.cycleSim.model.x_struct

        # symbolic variable for the time, needed in eval of the control parameterization
        tau = ca.SX.sym('tau')

        # build constraints struct
        self.constraints_struct = struct_symSX([entry('ADA_connect', shape=cycleSim.model.x.shape),
                                                entry('ADA_CC',
                                                      struct=cycleSim.cycleConditions_struct,
                                                      repeat=self.Ncycles),
                                                entry('ADA_connect_between_cycles', shape= cycleSim.model.x.shape, repeat=self.Ncycles-1)
                                                ])
        constraints = struct_SX(self.constraints_struct)

        # fill constraints expressions (cast to correct struct if neccesary)
        for n in range(self.Ncycles):
            # cycle conditions
            constraints['ADA_CC', n] = self.cycleSim.f_cycleConditions(self.Z['Z_cycles', n],
                                                                       tau + self.deltaTau[n],
                                                                       *parameters.syms_list)

        for n in range(self.Ncycles - 1):
            # connect between cycles
            constraints['ADA_connect_between_cycles', n] = self.Z['Z_cycles', n, 'x_plus'] - self.Z['Z_cycles', n + 1, 'x_minus']

        # retreive the X'ks
        Xks = [self.Z['Z_cycles', 0, 'x_minus']]
        for n in range(self.Ncycles):
            Xks.append(self.Z['Z_cycles', n, 'x_plus'])

        # connect to integration point
        constraints['ADA_connect'] = self._x - ca.horzcat(*Xks) @ self.b

        # build function to approximate the average dynamics
        self.F = ca.Function('F', [self._x, self.Z, tau] + parameters.syms_list,
                             [ca.horzcat(*Xks) @ self.c],
                             ['X', 'Z', 'tau'] + parameters.syms_list_names,
                             ['dot_X'])

        # build constraints function
        self.G = ca.Function('G', [self._x, self.Z, tau] + parameters.syms_list,
                             [constraints],
                             ['X', 'Z', 'tau'] + parameters.syms_list_names,
                             ['constraints']
                             )

        # output function for plotting
        _sigma, _tau = ca.SX.sym('sigma'), ca.SX.sym('tau')  # temporary symbols
        _edges = self.deltaTau.tolist()
        _values_f = []
        for n in range(self.Ncycles):
            _sigma_cycle = _sigma - self.deltaTau[
                n]  # shift the sigma of the cycle that is between dtau[n] and dtau[n+1] to the interval [0,1]
            _values_f.append(
                self.cycleSim.f_outputMap(_sigma_cycle, self.Z['Z_cycles', n], _tau + self.deltaTau[n], *parameters.syms_list))
        _outputExpression = constructPiecewiseCasadiExpression(_sigma, _edges, _values_f)
        self.f_MicroIntOutputMap = ca.Function('outputFunction', [_sigma, self._x, self.Z, _tau, *parameters.syms_list],
                                               [_outputExpression])

        # stagecost function
        _cycleCostAverage = 0
        for n in range(self.Ncycles):
            # the term 'sum(self.c[n+1:])' follows from a reformulation of a quadrature state to an avaerage cycle cost
            _cycleCostAverage += sum(self.c[n+1:]) * self.cycleSim.f_cycleCost(self.Z['Z_cycles', n],
                                                                              _tau + self.deltaTau[n], *parameters.syms_list) / 1
        self.f_averageCycleCost = ca.Function('stageCost', [self._x, self.Z, _tau] + parameters.syms_list, [_cycleCostAverage])

    def copy(self, cycleSim: Cycle):
        return CentralDifferences(cycleSim, K = self.K, parameters= self.parameters)

    @property
    def name(self):
        return f'CD{self.K}'

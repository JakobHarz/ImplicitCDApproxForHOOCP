from typing import List, Union
from Core.integrators import RungeKuttaIntegrator, OthorgonalCollocation
from Core.polynomials import Polynomial

import casadi as ca

from Core.parameters import Parameters


class ControlParameterization:

    def __init__(self, nu: int, Nctrl: int, slowPolynomial: Polynomial):
        """

        :param nu: the number of controls of the model
        :param Nctrl: the number of coefficients that parameterize the control
        :param slowPolynomial: the integrator used for the slowly changing parameters of the control
        :param tauf: the horizon of the control
        """

        assert isinstance(slowPolynomial, Polynomial)

        self.nu = nu
        self.Nctrl = Nctrl
        self.slowPolynomial = slowPolynomial

        # self.tauf_SYM : Union[float, ca.SX] = ca.SX.sym('tauf', 1)
        """ The horizon of the control and thus the polynomials of the slowly chaning parameters"""

        # generate a list of symbolic variables of shape (nu,Nctrl)
        # self.U_list_SYM = [ca.SX.sym(f'U_{k}', (self.nu, self.Nctrl)) for k in range(self.controlRKInt.d)]

        self._u_f: ca.Function = None

    def getPlottingFunction(self) -> ca.Function:
        """
        Returns a casadi function
            u(tau) = u(tau; U_1,...,U_Nctrl)
        that can be used for plotting, most likely not for optimization

        :param U_list: list of DMs of shape (Nctrl,nu)
        :param tauf: the horizon of the control polynomials
        :return:
        """

        tau = ca.SX.sym('tau')
        sigma = tau - ca.floor(tau)
        # assert len(U_list) == self.controlRKInt.d
        # for U in U_list:
        #     assert U.shape == (self.nu, self.Nctrl), \
        #         f"The shape of the control matrix is supposed to be (nu,Nctrl)=({self.nu, self.Nctrl}) but is {U.shape}"
        #     assert type(U) == ca.DM

        u_SX = self.u_f(tau, sigma)
        return ca.Function('u', [tau], [u_SX])

    def getControlExpression(self,
                             tau: Union[float,ca.SX],
                             sigma: Union[float,ca.SX],
                             U_list: List[ca.SX]) -> ca.SX:

        raise NotImplementedError('Implement this in the subclass')


    @property
    def u_f(self) -> ca.Function:
        """
            A casadi fucntion for the control evaluated at some time tau and local time sigma.
                u(tau; sigma, U_1,...,U_Nctrl, tau_f)
            the expression is of shape (nu,1) and depends on the control parameters U_list.
            :param tau: global time at which the control is evaluated
            :param sigma: local time at which the control is evaluated
            :param U_list: list of SXs of shape (Nctrl,nu)
            :return:
        """
        return self._u_f

    @u_f.setter
    def u_f(self, u_f: ca.Function):
        assert type(u_f) == ca.Function
        self._u_f = u_f




# class NoControlParameterization(ControlParameterization):
#
#         def __init__(self):
#             """
#             Empty Control Parametrization, i.e. no control
#             :param nu: the number of controls of the model
#             :param controlRKInt: the integrator used for the slowly changing parameters of the control
#             :param tauf: the horizon of the control
#             """
#             super().__init__(0, 0, NoneRK(), 0)
#
#             self.U_list_SX = []
#
#
#         def getControlExpression(self, tau: Union[float,ca.SX], sigma: Union[float,ca.SX], U_list: List[ca.SX]) -> ca.SX:
#             assert len(U_list) == 0
#             for U in U_list:
#                 assert U.shape == (self.nu, self.Nctrl), \
#                     f"The shape of the control matrix is supposed to be (nu,Nctrl)=({self.nu, self.Nctrl}) but is {U.shape}"
#                 assert type(U) == ca.SX
#
#             return ca.DM.zeros(0, 0)

class ZeroControlParameterization(ControlParameterization):

        def __init__(self):
            """
            Zero Control Parametrization, i.e. u = 0
            :param nu: the number of controls of the model
            :param controlRKInt: the integrator used for the slowly changing parameters of the control
            :param tauf: the horizon of the control
            """
            super().__init__(0, 0, Polynomial(0))

            self.u_f = ca.Function('u', [ca.SX.sym('tau'), ca.SX.sym('sigma')],
                                   [ca.DM.zeros(self.nu,1)],
                                   ['tau', 'sigma'],
                                   ['u'])

        # def getControlExpression(self, tau: Union[float,ca.SX], sigma: Union[float,ca.SX], U_list: List[ca.SX]) -> ca.SX:
        #     assert len(U_list) == 0
        #     return 0


class FourierControlParameterization(ControlParameterization):

    def __init__(self, nu: int, slowPolynomial: Polynomial, parameters: Parameters = Parameters(), N_har: int = 1):
        """

        implements a control expression of the form

        u(tau) = u_0(tau) + \sum_{i=1}^{N_Har} u_i(tau) * sin(2*pi*i*\tau) + v_i(tau) * cos(2*pi*i*tau)

        where $N_Har = (Nctrl - 1)/2$ is the number of harmonics.

        :param nu: the number of controls of the model
        :param N_har: the number of harmonics
        :param slowPolynomial: the integrator used for the slowly changing parameters of the control
        """

        parameters.require('U_cycles')
        parameters.require('tauf')

        Nctrl = 2*N_har + 1
        assert Nctrl % 2 == 1, "The number of coefficients has to be odd"

        super().__init__(nu, Nctrl, slowPolynomial)

        self.N_har : int = N_har
        """ The number of harmonics, not including the offset term"""

        sigma_SYM: ca.SX = ca.SX.sym('sigma')
        basisFunctions = [1]  # offset term
        for i in range(1, self.N_har + 1):
            basisFunctions.append(ca.sin(2 * ca.pi * i * sigma_SYM))
            basisFunctions.append(ca.cos(2 * ca.pi * i * sigma_SYM))

        self.periodicBasis_f = ca.Function('periodicBasis', [sigma_SYM], [ca.vertcat(*basisFunctions)])

        # constant symbolics from outside
        # assert type(parameters.tauf) in [ca.SX, float, int]
        assert type(parameters.U_cycles) is list
        # assert type(parameters.U_cycles[0]) is ca.SX


        # construct the function for u(tau)
        tau_SYM = ca.SX.sym('tau')
        coeffPoly_f = self.slowPolynomial.getEvalFunction(shape=(self.nu, self.Nctrl))
        coefficients = coeffPoly_f(tau_SYM / parameters.tauf, *parameters.U_cycles) # slow coeffs of shape (nu,Nctrl)
        periodicBasis = self.periodicBasis_f(sigma_SYM)

        self.u_f = ca.Function('u', [tau_SYM, sigma_SYM] + parameters.syms_list,
                               [coefficients @ periodicBasis],
                               ['tau', 'sigma'] + parameters.syms_list_names,
                               ['u'])

    def getControlExpression(self, tau: Union[float,ca.SX], sigma: Union[float,ca.SX], U_list: List[ca.SX]) -> ca.SX:

        raise NotImplementedError

        # assert len(U_list) == self.controlRKInt.d
        # for U in U_list:
        #     assert U.shape == (self.nu, self.Nctrl), \
        #         f"The shape of the control matrix is supposed to be (nu,Nctrl)=({self.nu, self.Nctrl}) but is {U.shape}"
        #     assert type(U) == ca.SX
        # return


class PiecewiseConstantControlParametrization(ControlParameterization):

    def __init__(self, nu: int, slowPolynomial: Polynomial, parameters: Parameters(), Nctrl: int):
        """

        :param nu: the number of controls of the model
        :param Nctrl: the number of constant control intervals over one cycle
        :paran tauf: the horizon of the control
        :param slowPolynomial: the integrator used for the slowly changing parameters of the control
        """
        super().__init__(nu, Nctrl, slowPolynomial)

        # dont define basis functions for now
        sigma = ca.SX.sym('sigma')
        basisFunctions = []

        for i in range(Nctrl):
            # step function between i/Nctrl and (i+1)/Nctrl
            basisFunctions.append(ca.if_else((i/Nctrl <= sigma)*(sigma < (i+1)/Nctrl), 1, 0))

        self.periodicBasis_f = ca.Function('periodicBasis', [sigma], [ca.vertcat(*basisFunctions)])


        tau_SYM = ca.SX.sym('tau')
        sigma_SYM = ca.SX.sym('sigma')
        coeffPoly_f = self.slowPolynomial.getEvalFunction(shape=(self.nu, self.Nctrl))
        # coefficients = coeffPoly_f(ca.floor(tau_SYM) / parameters.tauf, *parameters.U_cycles)
        coefficients = coeffPoly_f(tau_SYM / parameters.tauf, *parameters.U_cycles)

        self.u_f = ca.Function('u', [tau_SYM, sigma_SYM] + parameters.syms_list,
                               [coefficients @ self.periodicBasis_f(sigma_SYM)],
                               ['tau', 'sigma'] + parameters.syms_list_names,
                               ['u'])








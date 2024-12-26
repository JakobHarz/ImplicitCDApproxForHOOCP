import logging

import casadi as ca
import numpy as np
from casadi.tools import entry
from Core.model import Model

# class LinearOscillator(Model):
#     def __init__(self, A: ca.DM = ca.DM([[0, -1], [1, 0]]), B: ca.DM = ca.DM([[0], [1]])):
#         self.A = A
#         self.B = B
#
#         stateEntries = [entry('x', shape=A.shape[0]), entry('t', shape=1)]
#         controlEntries = [entry('u', shape=1)]
#
#         super().__init__(stateEntries, controlEntries)
#
#         ode =  self.T * ca.vertcat(self.A @self.x[:-1] + self.B @ self.u,1)
#         self.f = ca.Function('Dynamics', [self.x, self.u, self.T], [ode], ['x', 'u', 'T'], ['xdot'])
#
#     def getApproxPeriod(self, x):
#         evals, evecs = np.linalg.eig(self.A.full())
#         return 2*np.pi/np.abs(evals[0])
#
#
# class LinearOscillatorWithoutTime(Model):
#     def __init__(self, A: ca.DM = ca.DM([[0, -1], [1, 0]]), B: ca.DM = ca.DM([[0], [1]])):
#         self.A = A
#         self.B = B
#
#         stateEntries = [entry('x', shape=A.shape[0])]
#         controlEntries = []
#
#         super().__init__(stateEntries, controlEntries)
#
#         ode =  self.T * self.A @self.x
#         self.f = ca.Function('Dynamics', [self.x, self.u, self.T], [ode], ['x', 'u', 'T'], ['xdot'])
#
#     def getApproxPeriod(self, x):
#         evals, evecs = np.linalg.eig(self.A.full())
#         return 2*np.pi/np.abs(evals[0])

class LinearOscillatorScaled(Model):
    """ Linear oscillator with scaled dynamics such that the period is 1"""

    def __init__(self, epsilon: float = 0.01, parametrize_T: bool = False):
        super().__init__(stateEntries=[entry('x', shape=2), entry('t')], controlEntries=[])

        # assert epsilon ** 2 < 1, "epsilon**2 must be smaller than 1"
        self.epsilon = epsilon
        self.A = ca.DM([[0, -2 * ca.pi], [2 * ca.pi, 0]]) - epsilon * ca.DM.eye(2)


        if parametrize_T:
            T = ca.SX.sym('T')
            ode = T * ca.vertcat(self.A @ self.x_struct["x"], epsilon)
            self.f = ca.Function('Dynamics', [self.x, self.tau, T], [ode], ['x', 'tau', 'T'], ['xdot'])
            self.requiresTimescaling = True
        else:
            ode = ca.vertcat(self.A @ self.x_struct["x"], epsilon)
            self.f = ca.Function('Dynamics', [self.x, self.tau], [ode], ['x', 'tau'], ['xdot'])

        # analytical solution
        x0bar = ca.SX.sym('x0bar', 3)
        c1 = x0bar[0]
        c2 = x0bar[1]

        tau = ca.SX.sym('tau', 1)

        _rotMattop = ca.horzcat(ca.cos(2* ca.pi * tau), -ca.sin(2 * ca.pi * tau))
        _rotMatbot = ca.horzcat(ca.sin(2 * ca.pi * tau), ca.cos(2* ca.pi * tau))
        _rotMat = ca.vertcat(_rotMattop,_rotMatbot)
        analyticalSolution_x = ca.exp(-epsilon*tau)@_rotMat@x0bar[:2]

        self.solutionMap = ca.Function('analyticalSolution', [x0bar, tau],
                                       [ca.vertcat(analyticalSolution_x, x0bar[2] + epsilon*tau)])
        """ Analytical solution of the model, of the form x(tau0 + tau) = Phi(x0,tau) where x0 is x(tau0)"""

        self.solutionMapAverageDyn = ca.Function('analyticalSolutionAverage', [x0bar, tau], [ca.vertcat(ca.exp(-epsilon * tau) * x0bar[0:2],x0bar[2] +  epsilon*tau)])

    def getApproxPeriod(self, x):
        return 1

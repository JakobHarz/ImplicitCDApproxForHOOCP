from typing import List

import casadi as ca
import matplotlib.pyplot as plt
from casadi.tools import entry
from Core.model import Model
from Core.parameters import Parameters


# new class for the model
class Satellite2D_Hamiltonian(Model):

    def __init__(self, epsilon: float = 1):
        stateEntries = [entry('q', shape=2),  # position vector
                        entry('p', shape=2),  # velocity vector
                        entry('t', shape=1)  # time
                        ]

        controlEntries = [entry('thrust'),
                          entry('angle')]  # thrust

        super().__init__(stateEntries, controlEntries)

        self.epsilon = epsilon

        self.lbu = self.u_struct((0, -ca.inf))
        self.ubu = self.u_struct((1, ca.inf))

        pos = self.x_struct['q']
        vel = self.x_struct['p']

        r = ca.sqrt(pos[0] ** 2 + pos[1] ** 2)
        # energy = 0.5 * (vel[0]**2 + vel[1]**2) - 1/r
        energy = 0.5 * (vel[0] ** 2 + vel[1] ** 2) - ca.sqrt(1 / (r ** 2 + 1E-5 ** 2))
        V = - 1 / (2 * r ** 3) + 3 * pos[0] ** 2 / (2 * r ** 5)
        G = - ca.gradient(V, pos)
        L = pos[0] * vel[1] - pos[1] * vel[0]
        self.T_exact = ca.Function('T_exact', [self.x], [2 * ca.pi * (-2 * energy) ** (-1.5)], ['x'], ['T_exact'])
        # self.T_exact = ca.Function('T_exact', [self.x], [2 * ca.pi], ['x'], ['T_exact'])

        # acceleration = ca.vertcat(-pos[0]/r**3, -pos[1]/r**3) \
        #                + epsilon/10 *  self.u_struct['thrust'] * ca.vertcat(ca.cos(self.u_struct['angle']), ca.sin(self.u_struct['angle'])) \
        #                + epsilon * 1 * G

        thrust = epsilon * self.u_struct['thrust'] * ca.vertcat(ca.cos(self.u_struct['angle']),
                                                                ca.sin(self.u_struct['angle']))
        J = ca.DM.zeros(4, 4)
        J[0:2, 2:4] = ca.DM.eye(2)
        J[2:4, 0:2] = -ca.DM.eye(2)

        xdot = ca.vertcat(ca.inv(J) @ ca.gradient(energy, self.x)[0:4], 1) + ca.vertcat(0, 0, thrust, 0)
        ode = self.T * self.T_exact(self.x) * xdot
        # ode = self.T*self.T_unperturbed(self.x)*ca.vertcat(self.x[2], self.x[3], acceleration,1)

        self.r = ca.Function('Radius', [self.x], [r], ['x'], ['r'])
        self.G = ca.Function('Gradient', [self.x], [G], ['x'], ['G'])
        self.f = ca.Function('Dynamics', [self.x, self.u, self.T], [ode], ['x', 'u', 'T'], ['xdot'])
        self.E = ca.Function('Energy', [self.x], [energy], ['x'], ['E'])
        self.G = ca.Function('Gradient', [self.x], [G], ['x'], ['G'])
        self.L = ca.Function('AngularMomentum', [self.x], [L], ['x'], ['L'])

    def getApproxPeriod(self, x):
        return 1


class Satellite2D(Model):
    nx = 4  # number of states
    nu = 2  # number of controls

    def __init__(self, params: Parameters, parametrize_T: bool = False):
        """
        Model of a satellite in 2D
        :param params: Parameters, have to contain at least the parameter epsilon
        :param parametrize_T: if True, the period is a parameter, otherwise it is a function of the state
        """

        # make sure that the parameters instance contain epsilon
        params.require('epsilon')

        # symbolic model parameters? # todo: fix this such that it returns a list of [SymbolicParameter]
        symbolicParams, symbolicParamsNames, defaultValues = params.getSymsListByNames(['epsilon'])

        stateEntries = [entry('p', shape=2),  # position vector
                        entry('v', shape=2),  # velocity vector
                        ]

        controlEntries = [entry('thrust', shape=(2, 1))]

        super().__init__(stateEntries, controlEntries, symbolicParams=symbolicParams,
                         symbolicParamsNames=symbolicParamsNames, symbolicParamsDefaultValues=defaultValues)

        self.epsilon = params.epsilon

        pos = self.x_struct['p']
        vel = self.x_struct['v']

        r = ca.sqrt(pos[0] ** 2 + pos[1] ** 2)
        energy = 0.5 * (vel[0] ** 2 + vel[1] ** 2) - 1 / r
        V = - 1 / (2 * r ** 3) + 3 * pos[0] ** 2 / (2 * r ** 5)
        G = - ca.gradient(V, pos)

        # period of the unperturbed system
        self.T_exact = ca.Function('T_exact', [self.x], [2 * ca.pi * (-2 * energy) ** (-1.5)], ['x'], ['T_exact'])

        u = self.u_struct['thrust']

        xdot = ca.vertcat(vel,
                          -(1 / (r ** 3 + 1E-10)) * pos + self.epsilon * G +  self.epsilon *u,
                          )

        if parametrize_T:
            self.T = ca.SX.sym('T')
            ode = self.T * xdot
        else:
            ode = self.T_exact(self.x) * xdot

        self.r = ca.Function('Radius', [self.x], [r], ['x'], ['r'])
        self.G = ca.Function('Gradient', [self.x], [G], ['x'], ['G'])
        if parametrize_T:
            self.f = ca.Function('Dynamics', [self.x, self.tau, self.u, self.T], [ode], ['x', 'tau', 'u', 'T'],
                                 ['xdot'])
        else:
            self.f = ca.Function('Dynamics', [self.x, self.tau, self.u, *self.symbolicParams], [ode],
                                 ['x', 'tau', 'u'] + self.symbolicParamsNames, ['xdot'])
        self.E = ca.Function('Energy', [self.x], [energy], ['x'], ['E'])
        self.G = ca.Function('Gradient', [self.x], [G], ['x'], ['G'])
        # self.L = ca.Function('AngularMomentum', [self.x], [L], ['x'], ['L'])

        self.ecc_vec = ca.Function('Eccentricity', [self.x], [(vel.T @ vel - 1 / r) * pos - (pos.T @ vel) * vel], ['x'],
                                   ['eccentricity'])

    def getApproxPeriod(self, x):
        return self.T_exact(x)

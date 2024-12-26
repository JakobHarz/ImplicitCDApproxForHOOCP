from typing import Union

import numpy as np
from scipy import linalg
import casadi as ca



class LinearStartConditions:

    def __init__(self, q: ca.DM, b: ca.DM, bplus: ca.DM = None):
        """
        A class to store the linear start conditions of the method, characterized by the vector q and value b.
        """

        assert q.shape[1] == 1
        assert type(q) == ca.DM, 'q has to be a casadi.DM object'
        assert type(b) == ca.DM, 'bminus has to be a casadi.DM object'
        assert b.shape == (1, 1)

        self.q: ca.DM = q
        self.b: ca.DM = b
        self.Q: ca.DM = linalg.null_space(q.T)


class LinearBoundaryConditions:

    def __init__(self, q: ca.DM, bminus: ca.DM, bplus: ca.DM = None):
        """
        A class to store the linear boundary conditions of the method, characterized by the vector a, bminus and bplus.
        """

        assert q.shape[1] == 1
        assert type(q) == ca.DM, 'q has to be a casadi.DM object'
        assert type(bminus) == ca.DM, 'bminus has to be a casadi.DM object'
        assert bminus.shape == (1,1)
        if bplus is not None: assert bplus.shape == (1,1)

        self.q: ca.DM = q
        self.bminus: ca.DM  = bminus
        if bplus is None:
            self.bplus: ca.DM= bminus
        else:
            self.bplus: ca.DM = bplus

        self.Q: ca.DM = linalg.null_space(q.T)

    def phiminus(self, x: Union[np.ndarray, ca.DM]):
        """
        Condition for the startpoint of a cycle

        :param x: state where the condition is evaluated
        :return: phiminus(x) in R
        """
        return self.q.T @ x - self.bminus

    def phiplus(self, x: Union[np.ndarray, ca.DM]):
        """
        Condition for the endpoint of a cycle

        :param x: state where the condition is evaluated
        :return: phiplus(x) in R
        """
        return self.q.T @ x - self.bplus

    def __str__(self):
        return f'LinearPhaseConditions with q = {self.q.full().flatten()}, bminus = {self.bminus.full()}, bplus = {self.bplus.full()}'

    def project(self, x, target='minus'):
        """ Projects the state x onto the affine subspace defined by the either the minus or plus phase condition.
        :param x: state to be projected
        :param target: either 'minus' or 'plus', to select the target subspace
        """

        assert target in ['minus', 'plus']
        b = self.bminus if target == 'minus' else self.bplus

        return x - self.q @ (self.q.T @ x - b)

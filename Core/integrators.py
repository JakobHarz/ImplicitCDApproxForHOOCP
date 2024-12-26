import logging
from typing import Tuple, Union, List, Type
import numpy as np
import casadi as ca
from casadi.tools import struct_symSX, struct_SX, entry

from Core.tools import StructFunction


class ButcherTableau:
    """ Base Class for all the different Butcher Tableaus of the Integration Methods."""
    c = None
    A = None
    b = None
    bhat = None
    d = 0

    _continuousb = None
    __name__ = None

    @property
    def hasDenseOutput(self) -> bool:
        """ returns True if the Butcher Tableau has a dense output function"""
        return self._continuousb is not None

    def continuousb(self) -> ca.Function:
        """ return a casadi function b: [0,1] -> R^d to compute the coefficients b(theta) ∈ R^d of the
        continuous output x(theta) = x0 +  Σ b_i(theta) v_i, compare Hairer 1, Chp.2.6, p.188"""
        raise NotImplementedError('This method is not available for this Butcher Tableau!')

    def unpack(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Returns the Tuple (c,A,b) of the Butcher Tableau"""
        return self.c, self.A, self.b

    @property
    def isExplicit(self):
        return np.allclose(self.A, np.tril(self.A))

    @property
    def isEmbedded(self):
        return self.bhat is not None

    @property
    def isCollocationMethod(self):
        # check if there are double entries in the c vector
        return len(self.c) == len(np.unique(self.c))

    @property
    def isSymplectic(self):
        """Check wether the method is symplectic (if used on a hamiltonian system)"""
        for i in range(self.d):
            for j in range(self.d):
                # the condition has to be satisfied for all i,j
                cond = (self.b[i] * self.A[i, j] + self.b[j] * self.A[j, i] != self.b[i] * self.b[j])
                if cond: return False
        return True

    def printButcherTableau(self):
        for i in range(self.d):
            print(f'{self.c[i]:.2f} | {" ".join([f"{self.A[i, j]:.2f}" for j in range(self.d)])}')
        # vertical line
        print('---' + ' '.join(['-' for _ in range(5 * self.d)]))
        print(f'   | {" ".join([f"{self.b[i]:.2f}" for i in range(self.d)])}')

    @property
    def name(self):
        if self.__name__ is None:
            return self.__class__.__name__

        return self.__name__

    @name.setter
    def name(self, name: str):
        self.__name__ = name


class Integrator:
    @property
    def isExplicit(self) -> bool:
        raise NotImplementedError

    def explicitFunction(self) -> StructFunction:
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__


class RungeKuttaIntegrator(Integrator):
    """
    Runge Kutta Integration Method Implementation
    """

    def __init__(self, butcherTableau: ButcherTableau, ode: ca.Function, expressions: List[ca.SX] = [],
                 constant_syms: List[ca.SX] = [], quad: ca.Function = None):
        """
        Creates an integrator object for the given ode function and its arguments.

        The user can provide a list of expressions that are evaluate, in addition to x and tau at each call of the ode as
        f(x,tau,*expressions), for example a parametrized control u(tau;p_1,...,p_n) or a free model parameter \theta_1.
        They can depend either on a symbolic variable with the name 'x' or 'tau', or on other given symbolic variables
        that are constant over the interval.

        :param ode: a function xdot = f(x,tau,*expressions)
        :param expressions: a list of casadi expressions
        :param constant_syms: list of symbolic variables that are constant over the interval, such as controls parameters p_1,p2,... , timescalings T, parameters p, etc.
        :param quad: a function that is supposed to be integrated of the form q = quad(x,tau,*expressions)

        """

        self.butcherTableau: butcherTableau = butcherTableau
        self.c, self.A, self.b, self.d = self.butcherTableau.c, self.butcherTableau.A, self.butcherTableau.b, self.butcherTableau.d

        # check inputs
        if quad is not None:
            assert quad.n_in() == len(
                expressions) + 2, f"The function quad() has the wrong number of inputs (is {quad.n_in()} but should be {len(expressions) + 2})!, quad: {str(quad)} "
            assert quad.sx_out()[0].shape == (
                1, 1), f"The function quad() has the wrong output shape: {quad.sx_out()[0].shape}!"

        # no names given? create them
        # if len(constant_syms_names) == 0:
        #     constant_syms_names = [f'sym_{i}' for i in range(len(constant_syms))]
        # assert len(constant_syms) == len(
        #     constant_syms_names), "The number of constant symbols and their names must be equal!"

        assert type(ode) == ca.Function, "ode has to be a casadi function!"
        assert ode.n_in() == len(expressions) + 2, (
            f"The function ode() has the wrong number of inputs! ode: {str(ode)}"
            f", len(expressions): {len(expressions)}")

        assert ode.sx_in()[0].shape == ode.sx_out()[0].shape, "The function ode() has the wrong input/output shape!"

        # assert ode(x, tau, *expressions).shape == x.shape, "The function ode() has the wrong output shape!"

        logging.debug(f'Creating Explicit Function for Integrator {self} with ode={ode.name()}')
        logging.debug(f"Quadrature: {quad is not None}")

        # copy arguments
        self.ode = ode
        self.nx = ode.sx_in()[0].shape[0]
        self.quad = quad
        self.expressions = expressions
        self.constant_syms = constant_syms
        self.constant_syms_names = [sym.str() for sym in constant_syms]

        self.outputStruct = struct_symSX([
            entry('x_irk', shape=(self.nx, 1), repeat=self.d),
            entry('v_irk', shape=(self.nx, 1), repeat=self.d),
            entry('x_0', shape=(self.nx, 1)),
            entry('x_f', shape=(self.nx, 1)),
            entry('q_f', shape=(self.quad.sx_out()[0].shape) if quad is not None else (1, 1))
        ])
        """ The structure of the results of a single step of the RK integrator 
        
        Contains: 
            - x_irk: the state at the collocation points
            - v_irk: the derivative at the collocation points
            - x_0: the state at the beginning of the interval
            - x_end: the state at the end of the interval
            - q_end: the quadrature at the end of the interval
        
        """

        # find all the (x,tau) symbolic variables that the expressions depend on, to replace them later
        self._expressions_x_tau = {}
        """ A dictionary that contains the symbolic variables x and tau for each expression (u(x,tau), T(tau) ... )
         in the list of expressions."""

        for ex in self.expressions:
            self._expressions_x_tau[ex.str()] = {'x': None, 'tau': None}
            for var in ca.symvar(ex):  # iterate over all symbolic variables that ex depends on
                if var.name() in ['x', 'tau']:
                    # sanity check
                    target_shape = (self.nx, 1) if var.name() == 'x' else (1, 1)
                    assert var.shape == target_shape, f"The shape of the state variable {var.name()} is wrong, should be {target_shape}!"

                    # replace in dictionary
                    self._expressions_x_tau[ex.str()][var.name()] = var


        # misc stuff for implicit solution
        self.__solver = None
        self.__w = None

    def unpackButcher(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Returns the Butcher Tableau (c,A,b) of the Integrator"""
        return self.butcherTableau.c, self.butcherTableau.A, self.butcherTableau.b

    def continuousOutput(self) -> ca.Function:
        """ If available, returns a continuous function that approximates the solution over the integration interval"""
        sigma = ca.SX.sym('sigma')
        if self.butcherTableau.hasDenseOutput:
            f_b = self.butcherTableau.continuousb()
        else:
            logging.warning('No continuous output available for this integrator, using the start point of the interval as a constant!')
            f_b = ca.Function('f_b', [sigma], [ca.DM.zeros((self.d, 1))])

        b_sigma = f_b(sigma)

        v_i_matrix = ca.horzcat(*self.outputStruct['v_irk'])

        x_sigma = self.outputStruct['x_0'] + v_i_matrix @ b_sigma

        return ca.Function('continuousOutput', [sigma, self.outputStruct], [x_sigma], ['sigma', 'outputStruct'],
                           ['x(sigma)'])

    def explicitFunction(self, useEmbeddedFormula=False):
        """
        Returns a casadi-function for the computation of one integrator interval. Only available if the integrator is explicit.

        :return: a function [x0, tau0, h, *constant_syms_values] -> [output struct with 'x_irk','v_irk','x_0', 'x_f','q_f']
        """

        if not self.isExplicit:
            raise NotImplementedError('Integrator is not explicit!')

        if useEmbeddedFormula and not self.butcherTableau.isEmbedded:
            raise NotImplementedError('The Integrator is not embedded!')

        b = (self.butcherTableau.bhat if useEmbeddedFormula else self.butcherTableau.b)

        tau0 = ca.SX.sym('tau_0')
        x0 = ca.SX.sym('x_0', (self.nx, 1))
        xend = x0
        qend = 0

        h = ca.SX.sym('h')
        output = struct_SX(self.outputStruct)

        v_s = [ca.DM.inf() for _ in range(self.d)] # we get inf if bad implementation

        # actual runge kutta step
        for i in range(self.d):

            # compute intermediate stage state
            s_i = x0
            for j in range(i):
                # compute the intermediate state
                s_i += self.A[i, j] * v_s[j]

            # time at this stage
            tau_i = tau0 + self.c[i] * h

            # substitue the expressions for this stage
            expressions_sub = self._getSubstitutedExpressions(s_i, tau_i)

            # evaluate dynamics for this stage
            v_i = h * self.ode(s_i, tau_i, *expressions_sub)
            v_s[i] = v_i

            xend += b[i] * v_i

            # do quadrature if needed
            if self.quad is not None:
                qend += h * b[i] * self.quad(s_i, tau_i, *expressions_sub)

            # store in output
            output['x_irk', i] = s_i
            output['v_irk', i] = v_i

        output['x_0'] = x0
        output['x_f'] = xend
        output['q_f'] = qend
        return StructFunction('explicitIntegrator', [x0, tau0, h, *self.constant_syms], [output],
                              ['x_0', 'tau_0', 'h', *self.constant_syms_names],
                              [f"OutputStruct with {[key for key in self.outputStruct.keys()]}"],
                              struct=self.outputStruct)

    def implicitFunction(self) -> Tuple[ca.Function, ca.Function, struct_symSX, struct_SX]:

        """
        Returns the tools for the implicit computation of one integrator interval.

        - [a] A function G_irk(x0,xend,tau0,h,z_irk,*constant_syms) -> [0]
        - [b] A function Y_irk(x0, tau0, h, z_irk, *constant_syms) -> [ output struct with 'x_irk','v_irk','x_0', 'x_f','q_f']
        - [c] A structure of symbolic variables z_irk that contain the collocation variables
        - [d] A structure of the constraints

        z_irk containts:
         - x_irk = [x_1, ..., x_d]
         - v_irk = [v_1, ..., v_d]

        """

        # structure for the irk variables
        z_irk = struct_symSX([entry('x_irk', shape=(self.nx, 1), repeat=self.d),
                              entry('v_irk', shape=(self.nx, 1), repeat=self.d)])

        # constraints for the irk constraints
        G_irk_struct = struct_symSX([
            # RK equations for every integration interval
            entry('IRK_dynamics', shape=(self.nx, 1), repeat=self.d),
            entry('IRK_equation', shape=(self.nx, 1), repeat=self.d),
            entry('IRK_terminal', shape=(self.nx, 1)),
        ])
        constraints = struct_SX(G_irk_struct)

        # symbolic variable for the step size (in numerical time)
        h = ca.SX.sym('h')
        tau0 = ca.SX.sym('tau_0')
        x0 = ca.SX.sym('x0', (self.nx, 1))
        xend = ca.SX.sym('xend', (self.nx, 1))


        # construct constraints
        Xc_end = x0
        qend = 0
        for i in range(self.d):

            # get tau for this stage
            tau_i = tau0 + h * self.c[i]

            # substitute the expressions for this stage
            expressions_sub = self._getSubstitutedExpressions(z_irk['x_irk', i], tau_i)

            # dynamics
            constraints['IRK_dynamics', i] = h * self.ode(z_irk['x_irk', i], tau_i, *expressions_sub) - z_irk[
                'v_irk', i]

            # IRK stage values
            xi_bar = x0
            for j in range(self.d):
                xi_bar = xi_bar + self.A[i, j] * z_irk['v_irk', j]
            constraints['IRK_equation', i] = z_irk['x_irk', i] - xi_bar

            # IRK endpoint
            Xc_end = Xc_end + self.b[i] * z_irk['v_irk', i]

            # quadrature
            if self.quad is not None:
                qend += h * self.b[i] * self.quad(xi_bar, tau_i, *expressions_sub)

        # terminal constraint
        constraints['IRK_terminal'] = xend - Xc_end

        # create casadi functions
        G_irk = ca.Function('G_irk', [x0, xend, tau0, h, z_irk.cat, *self.constant_syms], [constraints],
                            ['x0', 'xend', 'tau_0', 'h', 'z_irk', *[s.str() for s in self.constant_syms]], ['G_irk'])

        # create the output struct & function
        output = struct_SX(self.outputStruct)
        output['x_irk'] = z_irk['x_irk']
        output['v_irk'] = z_irk['v_irk']
        output['x_0'] = x0
        output['q_f'] = qend
        output['x_f'] = Xc_end

        Y_irk = StructFunction('Y_irk', [x0, tau0, h, z_irk.cat, *self.constant_syms], [output],
                               ['x_0', 'tau', 'h', 'z_irk', *[s.str() for s in self.constant_syms]],
                               [f"OutputStruct with {[key for key in self.outputStruct.keys()]}"],
                               struct=self.outputStruct)

        return G_irk, Y_irk, z_irk, G_irk_struct

    def implicitSolution(self, x0, tau0, h, constant_syms_values: List):
        """ Solves an ivp with the implicit integrator by solving the nonlinear system using IPOPT.
        """

        G_irk, Y_irk, z_irk, _ = self.implicitFunction()
        # Todo: make these properties?

        # create a solver
        if self.__solver is None:
            x_0_sym = ca.SX.sym('x_0', (self.nx, 1))
            x_end_sym = ca.SX.sym('x_end', (self.nx, 1))

            w = struct_symSX([entry('z_irk', struct=z_irk),
                              entry('x_end', shape=(self.nx, 1))])
            self.__w = w

            # construct and solve the NLP
            problem = {'x': ca.vertcat(w['z_irk'], x_end_sym),
                       'g': G_irk(x_0_sym, x_end_sym, tau0, h, w['z_irk'], *self.constant_syms),
                       'p': ca.vertcat(x_0_sym, *self.constant_syms)
                       }
            options = {'ipopt.linear_solver': 'MA27',
                       'ipopt.max_iter': 1000,
                       'ipopt.print_level': 0,  # suppress IPOPT output
                       'print_time': False,
                       # accuracy
                       'ipopt.tol': 1e-12,
                       'ipopt.acceptable_tol': 1e-12,
                       }
            # options.update(additionalSolverOptions)

            self.__solver = ca.nlpsol('solver', 'ipopt', problem, options)

        solver = self.__solver
        w = self.__w
        # create the initial guess
        w0 = w(0)
        w0['z_irk', 'x_irk', :] = x0

        #  solve the ocp
        nlpsolution = solver(x0=w0, lbg=0, ubg=0, p=ca.vertcat(x0, *constant_syms_values))
        wopt = w(nlpsolution['x'])

        return Y_irk(x0, tau0, h, wopt['z_irk'], *constant_syms_values)

    def getPolyEvalCAExpression(self, nx: int, includeZero: bool = False) -> ca.Function:
        """
        Generates a casadi function that can be used to evaluate the polynomial at a given point, of the form F(t, p)
        where p is a (nx, d) (or (nx, d+1)) matrix of values and t is a scalar in [0,1].

        :param nx: the number of states
        :param includeZero: if true, the collocation point at time 0 is included

        TODO: Technically we should move this in the RK class

        :return: a casadi function of the form [0,1]x(nx,d(+1)) -> nx

        """
        assert self.isCollocationMethod is True, "Can only reconstruct polynomial for collocation methods!"

        # append zero if needed
        if includeZero:
            collPoints = ca.DM(np.concatenate([[0], self.c]))
            d = self.d + 1
        else:
            collPoints = ca.DM(self.c)
            d = self.d

        t = ca.SX.sym('t')
        p_vals = ca.SX.sym('p', (nx, d))

        # create list of polynomials
        _ls = []
        for j in range(d):
            l = 1
            for r in range(d):
                if r != j:
                    l *= (t - collPoints[r]) / (collPoints[j] - collPoints[r])
            _ls.append(l)

        # evaluate polynomials
        sum = ca.DM.zeros((nx, 1))
        for i in range(d):
            sum += p_vals[:, i] * _ls[i]

        return ca.Function('polyEval', [t, p_vals], [sum])

    def getPolyEvalFunction(self, shape: Tuple[int, int], includeZero: bool = False, includeOne: bool = False,
                            fixedValues: List[ca.DM] = None) -> ca.Function:
        """
        Generates a casadi function that evaluates the polynomial at a given point t of the form

        x(t) = F(t, [x0], x1, ..., xd)

        where t is a scalar in [0,1] and x0, ..., xd are the collocation points of the provided shape.

        If fixed values for the nodes x0, ..., xd are provided, the function will be of the form

        x(t) = F(t)

        :param shape: the shape of the collocation nodes, can be matrices or vectors
        :param includeZero: if true, the collocation point at time 0 is included
        :param fixedValues: a list of fixed values for the nodes, if provided, the function will be of the form x(t) = F(t)
        """
        assert self.isCollocationMethod is True, "Can only reconstruct polynomial for collocation methods!"

        assert not (includeOne and includeZero), 'either includeOne or includeZero can be true, not both!'

        # append zero if needed
        if includeZero:
            collPoints = ca.DM(np.concatenate([[0], self.c]))
            d = self.d + 1
        elif includeOne:
            collPoints = ca.DM(np.concatenate([self.c, [1]]))
            d = self.d + 1
        else:
            collPoints = ca.DM(self.c)
            d = self.d

        nx = shape[0] * shape[1]
        t = ca.SX.sym('t')

        if fixedValues is None:
            # create symbolic variables for the nodes
            Xs = []
            for i in np.arange((0 if includeZero else 1), self.d + 1):
                Xs.append(ca.SX.sym(f'x{i}', shape))

        else:
            assert len(
                fixedValues) == d, f"The number of fixed values ({len(fixedValues)}) must be equal to the number of collocation points ({d})!"
            assert all([v.shape == shape for v in
                        fixedValues]), "The shape of the fixed values must be equal to the shape of the collocation points!"
            assert all([type(v) == ca.DM for v in fixedValues]), "The fixed values must be of type casadi.DM!"
            Xs = fixedValues



        # create list of polynomials
        _ls = []
        for j in range(d):
            l = 1
            for r in range(d):
                if r != j:
                    l *= (t - collPoints[r]) / (collPoints[j] - collPoints[r])
            _ls.append(l)

        # evaluate polynomials
        sum = ca.DM.zeros((nx, 1))
        for i in range(d):
            sum += p_vals[:, i] * _ls[i]

        # reshape the result into the original shape
        result = ca.reshape(sum, shape)

        if fixedValues is None:
            return ca.Function('polyEval', [t] + Xs, [result], ['t'] + [f'x{i}' for i in range(d)], ['p(t)'])
        else:
            return ca.Function('polyEval', [t], [result], ['t'], ['p(t)'])


    @property
    def isExplicit(self):
        return self.butcherTableau.isExplicit

    @property
    def isCollocationMethod(self):
        # check if there are double entries in the c vector
        return self.butcherTableau.isCollocationMethod

    @property
    def isSymplectic(self):
        """Check wether the method is symplectic (if used on a hamiltonian system)"""
        return self.butcherTableau.isSymplectic

    def _getSubstitutedExpressions(self, x_i, tau_i) -> List[ca.SX]:
        """
        Returns a list of the provided expressions, where the symbolic variables x and tau are substituted by the values and time of the integrator stage.
        This makes use of ca.substitute() to replace the symbolic variables in the expressions.

        For example, the control expression u(x,tau) = c*tau + x[0] will be substituted by u(x_irk, tau_i) = c*tau_i + x_i.

        :param x_i: the symbolic variable for the state at the integrator stage
        :param tau_i: the symbolic variable for the time at the integrator stages
        :return: a list of substituted expressions
        """

        expression_subs = []
        for ex in self.expressions:
            substituted_expression = ex

            # substitute 'x' for the integrator stage value if needed
            if self._expressions_x_tau[ex.str()]['x'] is not None:
                substituted_expression = ca.substitute(substituted_expression, self._expressions_x_tau[ex.str()]['x'],
                                                       x_i)

            # substitute 'tau' for the integrator stage time if needed
            if self._expressions_x_tau[ex.str()]['tau'] is not None:
                substituted_expression = ca.substitute(substituted_expression, self._expressions_x_tau[ex.str()]['tau'],
                                                       tau_i)

            expression_subs.append(substituted_expression)

        return expression_subs

class Lobatto3A_Order2(ButcherTableau):
    # (trapazoidal rule)
    c = np.array([0, 1])
    A = np.array([[0, 0], [0.5, 0.5]])
    b = np.array([0.5, 0.5])
    d = c.shape[0]


class Lobatto3A_Order4(ButcherTableau):
    # (trapazoidal rule)
    c = np.array([0, 0.5, 1])
    A = np.array([[0, 0, 0], [5 / 24, 1 / 3, -1 / 24], [1 / 6, 2 / 3, 1 / 6]])
    b = np.array([1 / 6, 2 / 3, 1 / 6])
    d = c.shape[0]


class GaussLeg2(ButcherTableau):
    c = np.array([0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6])
    A = np.array([[0.25, 0.25 - np.sqrt(3) / 6], [0.25 + np.sqrt(3) / 6, 0.25]])
    b = np.array([0.5, 0.5])
    d = c.shape[0]


class ForwardEuler(ButcherTableau):
    c = np.array([0])
    A = np.array([[0]])
    b = np.array([1])
    d = c.shape[0]


class BackwardEuler(ButcherTableau):
    c = np.array([1])
    A = np.array([[1]])
    b = np.array([1])
    d = c.shape[0]


class RK4(ButcherTableau):
    c = np.array([0, 0.5, 0.5, 1])
    A = np.array([[0, 0, 0, 0],
                  [0.5, 0, 0, 0],
                  [0, 0.5, 0, 0],
                  [0, 0, 1, 0]])
    b = np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6])
    d = c.shape[0]


class RK45(ButcherTableau):
    # Fehlberg method order 5 with embedded error estimate order 4., (NOT DORMAND-PRINCE which is used in ode45)!

    c = np.array([0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2])
    A = np.array([[0, 0, 0, 0, 0, 0],
                  [0.25, 0, 0, 0, 0, 0],
                  [3 / 32, 9 / 32, 0, 0, 0, 0],
                  [1932 / 2197, -7200 / 2197, 7296 / 2197, 0, 0, 0],
                  [439 / 216, -8, 3680 / 513, -845 / 4104, 0, 0],
                  [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40, 0]])
    # 5(4)
    b = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55])
    bhat = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0])

    # 4(5)
    # b = np.array([25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]) # 4th order
    # bhat = np.array([16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]) # 5th order

    d = c.shape[0]


class HeunEuler(ButcherTableau):
    """ The simplest adaptive Runge–Kutta method involves combining Heun's method, which is order 2,
     with the Euler method, which is order 1 """

    c = np.array([0, 1])
    A = np.array([[0, 0], [1, 0]])
    b = np.array([0.5, 0.5])
    bhat = np.array([1, 0])
    d = c.shape[0]

    def __init__(self):
        super(HeunEuler, self).__init__()
        self.__name__ = 'Heun-Euler'

        # create list of polynomials
        _ls = []
        for j in range(self.d):
            l = np.poly1d([1])
            for r in range(self.d):
                if r != j:
                    l *= np.poly1d([1, -self.c[r]]) / (self.c[j] - self.c[r])
            _ls.append(l)

        # continuous output function
        sigma = ca.SX.sym('sigma')
        b_cont = [np.polyint(l)(sigma) for l in _ls]

        self._continuousb = ca.Function('continuousb', [sigma], [ca.vertcat(*b_cont)])

    def continuousb(self) -> ca.Function:
        return self._continuousb


class ExpMid(ButcherTableau):
    c = np.array([0, 0.5])
    A = np.array([[0, 0], [0.5, 0]])
    b = np.array([0, 1])
    d = c.shape[0]


class Heun(ButcherTableau):
    c = np.array([0, 1])
    A = np.array([[0, 0], [1, 0]])
    b = np.array([0.5, 0.5])
    d = c.shape[0]


class RK3(ButcherTableau):
    c = np.array([0, 0.5, 1])
    A = np.array([[0, 0, 0],
                  [0.5, 0, 0],
                  [-1, 2, 0]])
    b = np.array([1 / 6, 2 / 3, 1 / 6])
    d = c.shape[0]


class OthorgonalCollocation(ButcherTableau):
    def __init__(self, collPoints: np.array, name='Oth. Coll.', addEmbeddedFormula=False):
        assert collPoints.ndim == 1
        assert np.all(np.unique(collPoints, return_counts=True)[1] <= 1), 'CollPoints have to be distinct!'
        # assert np.all(collPoints <= 1) and np.all(0 <= collPoints), 'CollPoints must be between 0 and 1'

        super(OthorgonalCollocation, self).__init__()
        self.collPoints = collPoints
        self.d = collPoints.shape[-1]
        self.name = name

        # create list of polynomials
        self._ls = []
        for j in range(self.d):
            l = np.poly1d([1])
            for r in range(self.d):
                if r != j:
                    l *= np.poly1d([1, -collPoints[r]]) / (collPoints[j] - collPoints[r])
            self._ls.append(l)

        self.c = collPoints
        self.b = np.array([np.polyint(l)(1) for l in self._ls])
        self.A = np.array([[np.polyint(l)(ci) for l in self._ls] for ci in self.c])

        # continuous output function
        sigma = ca.SX.sym('sigma')
        b_cont = [np.polyint(l)(sigma) for l in self._ls]

        # do we also want an embedded formula?
        if addEmbeddedFormula:
            assert 0 not in collPoints, 'Cannot add embedded formula if 0 is a collocation point!'
            # todo: check if all eigenvalues of A are nonzero
            c_new = np.concatenate([[0], self.c, ])
            A_new = np.vstack([np.hstack(([[0]], np.zeros((1, self.d)))),
                               np.hstack((np.zeros((self.d, 1)), self.A,))
                               ])
            b_new = np.concatenate([[0], self.b])

            # build embedded integrator
            gamma0 = 1 / 8
            b = self.b
            C_matrix = np.vander(self.c, increasing=True).T
            rhs = np.array([1 - gamma0] + (1 / np.arange(2, self.d + 1)).tolist())
            b0hat = gamma0
            bhat = np.linalg.solve(C_matrix, rhs)
            bhat = np.concatenate([[b0hat], bhat])

            # use lower order formula

            # overwrite the values
            self.c = c_new
            self.A = A_new
            self.b = b_new
            self.bhat = bhat
            # self.b = bhat
            # self.bhat = b_new
            self.d = self.d + 1
            b_cont = [0] + b_cont

        self._continuousb = ca.Function('continuousb', [sigma], [ca.vertcat(*b_cont)])

    def continuousb(self) -> ca.Function:
        return self._continuousb

    @property
    def polynomials(self) -> List[np.poly1d]:
        """A list of the numpy polynomials that correspond to the lagrange polynomials"""
        return self._ls

    def __str__(self):
        return f"{self.name} - {self.d} stages"


class LegendreCollocation(OthorgonalCollocation):
    def __init__(self, d, addEmbeddedFormula=False):
        assert d > 0, 'd has to be positive'
        from Core.tools import gaussQuadraturePoints
        coll_points = gaussQuadraturePoints(d)
        super(LegendreCollocation, self).__init__(collPoints=coll_points, addEmbeddedFormula=addEmbeddedFormula,
                                                  name=f'L{d}')

    def __str__(self):
        return f"L{self.d}"

    @property
    def isSymplectic(self):
        # according to hairer CHP6, THR 4.2
        return True


class RadauCollocation(OthorgonalCollocation):
    def __init__(self, d: int, addEmbeddedFormula=False):
        assert d > 0, 'd has to be positive'
        super(RadauCollocation, self).__init__(np.array(ca.collocation_points(d, 'radau')),
                                               addEmbeddedFormula=addEmbeddedFormula, name=f'R{d}')

    def __str__(self):
        return f"R{self.d}"


# class ChebyshevCollocation(OthorgonalCollocation):
#     def __init__(self, d: int):
#         assert d > 0, 'd has to be positive'
#
#         # compute the chebyshev points
#         chebyshevPoints = np.array([np.cos((2*i-1)/(2*d)*np.pi) for i in range(1,d+1)])
#
#         # shift the points to the interval [0,1]
#         chebyshevPoints = (chebyshevPoints + 1)/2
#
#         super(ChebyshevCollocation,self).__init__(chebyshevPoints)
#
#     def __str__(self):
#         return f"RadauCollocation with {self.d} stages"

class StoermerVerlet(Integrator):

    @classmethod
    def explicitFunction(cls, qdot: ca.Function, pdot: ca.Function, h: float, x: struct_symSX, u: ca.SX,
                         T: ca.SX = None,
                         quad: ca.Function = None):
        """
         Returns a casadi function that executes one step of the stoermer verlet method for the symplectic integration of  hamiltonian system.

         State x = [q,p,t]

        :param qdot: a casadi function that computes the derivative of the position qdot(x)
        :param pdot: a casadi function that computes the derivative of the velocity pdot(x,u)
        :param h: the step size
        :param x: the state x = [q,p]
        :param u: a control input that is applied to the force
        :param T: a timescaling factor that is applied to the step size
        :param quad: a function that is supposed to be integrated. TODO: implement
        :return: a casadi function of the form xnext = F(x,u,T)
        """

        assert type(x) == struct_symSX, 'x has to be a struct_symSX'
        assert 'q' in x.keys(), 'x has to have a q entry for the position'
        assert 'p' in x.keys(), 'x has to have a p entry for the momentum'
        assert x['q'].shape[0] == x['p'].shape[0], 'q and p have to have the same dimension'

        if T is None:
            T = ca.DM(1)
        x_0 = x
        p_0 = x['p']
        q_0 = x['q']
        t_0 = x['t']

        p_05 = p_0 + h / 2 * T * pdot(x_0, u)
        q_1 = q_0 + h * T * qdot(x)
        p_1 = p_05 + h / 2 * T * pdot(ca.vertcat(q_1, p_05, t_0), u)
        t_1 = t_0 + h * T  # euler step for the time
        return ca.Function('F', [x, u, T], [ca.vertcat(q_1, p_1, t_1)], ['x', 'u', 'T'], ['xnext'])


# %%

class DEPRIVATED_LGPoly():
    """ A callable object that represents a Lagrange-Polynomial
            :math:`x(t) = sum(x_i,l_i(t))`
            of order d that is constructed from the collocation values v_1,...,v_d
            at the collocation points t_1,...,t_d.
    """

    def __init__(self, collPoints: np.ndarray, collValues: Union[np.ndarray, ca.SX], scaling: float = 1.0):

        # the collocation points have to lie between 0 and 1
        assert np.all(collPoints >= 0) and np.all(collPoints <= 1)
        assert collPoints.ndim == 1

        self.c = collPoints
        self.d = collPoints.shape[0]

        # the values are given as a (n_x,d) matrix.
        assert len(collValues.shape) == 2
        assert collValues.shape[1] == self.d
        self.nx = collValues.shape[0]

        self.collValues = collValues

        # value that scales the input such that tau/N is in the range [0,1]
        self.scaling = scaling

        # create list of polynomials
        self.ls = []
        for j in range(self.d):
            l = np.poly1d([1])
            for r in range(self.d):
                if r != j:
                    l *= np.poly1d([1, -collPoints[r]]) / (collPoints[j] - collPoints[r])
            self.ls.append(l)

    @property
    def shape(self):
        return (self.nx,)

    def getBasisPolyVals(self, tau) -> np.ndarray:
        """
        Returns the values of the single lagrange-polynomials in the basis, evaluated at tau

        :param tau: evaluation time
        :return: vector of size d
        """
        return np.array([l(tau) for l in self.ls])

    def __call__(self, tau) -> Union[np.ndarray, ca.SX]:
        return self.collValues @ self.getBasisPolyVals(tau / self.scaling)


class Collocation():

    @staticmethod
    def evaluatePolynomial(v: np.ndarray, t: np.ndarray, tau_roots: np.ndarray, derivative=0):
        """
        Evaluate the lagrange polynomial of order d at the times t.

        :param v: (d) construction points, shape (nx,d)
        :param t: vector of times
        :param scheme: 'legendre' or 'radau'
        :param tau_roots: times of the roots of the lagrange polynomial between 0..1
        :return: matrix of size (nx,len(t))
        """
        d = v.shape[-1]
        assert d == tau_roots.size

        # create list of polynomials
        ls = []
        for j in range(d):
            l = np.poly1d([1])
            for r in range(d):
                if r != j:
                    l *= np.poly1d([1, -tau_roots[r]]) / (tau_roots[j] - tau_roots[r])
            ls.append(l)

        # evaluate polynomials
        sum = np.zeros((v.shape[0], t.shape[0]))
        for i in range(d):
            sum += np.outer(v[:, i], np.polyder(ls[i], derivative)(t))

        return sum

    @staticmethod
    def getCollocationCoefficients(d: int, scheme='legendre', customCollPoints: np.ndarray = None, includeZero=True) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the coefficients of a legendre polynomical.

        Based of the casadi example pack.

        :param d: Degree of interpolating polynomial
        :return: Tuple, consisting of:
                    - C: Coefficients of the collocation equation (d+1)x(d+1)
                    - D: Coefficients of the continuity equation  (d+1)
                    - B: Coefficients of the quadrature function  (d+1)
        """

        if customCollPoints is None:
            assert scheme in ['legendre', 'radau']
            # Get collocation points
            tau_root = ca.collocation_points(d, scheme)
        else:
            assert 0 not in customCollPoints
            assert customCollPoints.ndim == 1
            assert d == customCollPoints.shape[0]
            tau_root = customCollPoints

        if includeZero:
            tau_root = np.append(0, tau_root)
            N = d + 1
        else:
            N = d

        # Coefficients of the collocation equation
        C = np.zeros((N, N))

        # Coefficients of the continuity equation
        D = np.zeros(N)

        # Coefficients of the quadrature function
        B = np.zeros(N)

        # Construct polynomial basis
        for j in range(N):
            # Construct Lagrange polynomials to get the polynomial basis at the collocation point
            p = np.poly1d([1])
            for r in range(N):
                if r != j:
                    p *= np.poly1d([1, -tau_root[r]]) / (tau_root[j] - tau_root[r])

            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            D[j] = p(1.0)

            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            pder = np.polyder(p)
            for r in range(N):
                C[j, r] = pder(tau_root[r])

            # Evaluate the integral of the polynomial to get the coefficients of the quadrature function
            pint = np.polyint(p)
            B[j] = pint(1.0)

        return C, D, B

    def __init__(self, model: 'Model', d: int, Ncoll: int, Tend, x0bar: np.ndarray, uconst=np.zeros, bounds=None):
        """

        :param model: instance of model class
        :param d: collocation order
        :param Ncoll: number of collocation steps
        :param Tend: final time
        :param x0bar: start position
        :param uconst: constant input that is applied at every step

        :ivar collVariables: List of all Collocation Variables
        :type collVariables: List[ca.MX]

        """

        self.x0bar = x0bar
        self.Tend = Tend
        self.Ncoll = Ncoll
        self.d = d
        self.model = model

        self.h = 1 / Ncoll

        coeff_coll, coeff_cont, coeff_quad = self.getCollocationCoefficients(d)

        self._collEquations = []
        self._collVariables = []

        # Prepare
        _lbw = []
        _ubw = []
        _lbg = []
        _ubg = []
        _w0 = []
        _lbg = []
        _ubg = []

        _xplot = []

        # check if bounds variable is given
        if bounds is None:
            bounds = {}
        if bounds.get('lbx', None) is None:
            bounds['lbx'] = model.lbx
        if bounds.get('ubx', None) is None:
            bounds['ubx'] = model.ubx
        lbx = bounds['lbx']
        ubx = bounds['ubx']

        # collocation variable and equation at start
        X0 = ca.MX.sym('X0', model.x.shape)
        self._collVariables.append(X0)
        _lbw.append(lbx)
        _ubw.append(ubx)
        _w0.append(x0bar)

        self._collEquations.append(X0 - x0bar)
        _lbg.append(np.zeros(model.x.shape))
        _ubg.append(np.zeros(model.x.shape))

        Xk = X0
        _xplot.append(Xk)

        for k in range(Ncoll):
            # State at collocation points
            Xc = []
            for j in range(d):
                Xkj = ca.MX.sym('X_' + str(k) + '_' + str(j), model.x.shape)
                Xc.append(Xkj)
                self._collVariables.append(Xkj)
                _lbw.append(lbx)
                _ubw.append(ubx)
                _w0.append(x0bar)

            # Loop over collocation points
            Xk_end = coeff_cont[0] * Xk
            for j in range(1, d + 1):
                # Expression for the state derivative at the collocation point
                pdot = coeff_coll[0, j] * Xk

                for r in range(d):
                    pdot = pdot + coeff_coll[r + 1, j] * Xc[r]

                # Append collocation equations
                xdot = model.f(Xc[j - 1], uconst, Tend)
                self._collEquations.append(self.h * xdot - pdot)
                _lbg.append(np.zeros(model.x.shape))
                _ubg.append(np.zeros(model.x.shape))

                # Add contribution to the end state
                Xk_end = Xk_end + coeff_cont[j] * Xc[j - 1]

            # New NLP variable for state at end of interval
            Xk = ca.MX.sym('X_' + str(k + 1), model.x.shape)
            _xplot.append(Xk)
            self._collVariables.append(Xk)
            _lbw.append(lbx)
            _ubw.append(ubx)
            _w0.append(x0bar)

            # Add equality constraint
            self._collEquations.append(Xk_end - Xk)
            _lbg.append(np.zeros(model.x.shape))
            _ubg.append(np.zeros(model.x.shape))

        # finish up
        self.lbw = ca.vertcat(*_lbw)
        self.ubw = ca.vertcat(*_ubw)
        self.lbg = ca.vertcat(*_lbg)
        self.ubg = ca.vertcat(*_ubg)
        self.w0 = ca.vertcat(*_w0)

        self.collEquations = ca.vertcat(*self._collEquations)
        self.collVariables = self._collVariables  # type: List[ca.MX]

        # Function to get x and u trajectories from w
        self._trajectories = ca.Function('trajectories', [ca.vertcat(*self.collVariables)], [ca.horzcat(*_xplot)],
                                         ['w'], ['x'])

    def solToTrajectorie(self, collVariablesSolution) -> np.ndarray:
        return self._trajectories(collVariablesSolution).full()

from typing import List, Tuple, Union

import casadi as ca
import numpy as np
import logging

from casadi.tools import struct_symSX
from casadi.tools.structure3 import DMStruct, SXStruct

from Core.configuration import IPOPT_OPTIONS_LINEAR_SOLVER

# %%
x = ca.SX.sym('x')
smoothStep_tanh_f = ca.Function('shifted_tanh', [x], [0.5 * (ca.tanh(x) + 1)])
""" A casadi function that implements the shifted tanh function `0.5*(ca.tanh(x) + 1)`
, useful for smooth step functions. """


# a casadi function that implement a smooth step function between two values and a smoothing width
y_start = ca.SX.sym('y_start')
y_end = ca.SX.sym('y_end')
width = ca.SX.sym('width')
smoothStep = ca.Function('smoothStep', [x, y_start, y_end, width], [y_start + (y_end - y_start) * smoothStep_tanh_f(2 * ((x - 0.5) / width))])
"""A casadi function step(x,start,end,width), x in [0,1] that implements a smooth step function between two values and a 
smoothing width."""

# %%

def solveRootfindingProblem(g: ca.Function, w0: ca.DM, supressOutput: bool = True) -> ca.DM:
    """
    Solves the rootfinding problem g(w)=0 for w using IPOPT.
    :param g: casadi function g(w)
    :param w0: initial guess
    :return: w such that g(w)=0
    """

    w = ca.SX.sym('w', w0.shape[0])
    problem = {'x': w, 'g': g(w)}
    options = {'ipopt.linear_solver': IPOPT_OPTIONS_LINEAR_SOLVER,
               'ipopt.max_iter': 50,
               'ipopt.print_level': 0 if supressOutput else 5,
               'print_time': False if supressOutput else True,
               'ipopt.tol': 1e-14
               }

    # create solver
    solver = ca.nlpsol('solver', 'ipopt', problem, options)

    # solve
    sol = solver(x0=w0, lbx=-np.inf, ubx=np.inf, lbg=0, ubg=0)
    # extract solution
    w = sol['x']
    return w


def physicalToNumericalTime(t: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Converts the values of physical time [0,t_end] to their respective values in numerical time [0,N]
    :param t: array of physical times of shape (*,)
    :param T: array of cycle durations [T_0,T_1,...,T_N-1] of shape (N,)
    :return: array of numerical times of shape (*,)
    """
    N = T.shape[0]
    assert N > 0

    # "The output is the same shape and type as x",
    # https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html
    assert t.dtype == np.float64, "ndtype of array has to be float, piecewise interpolation wont work otherwise!"

    tks = np.concatenate([np.array([0]), np.cumsum(T)])

    # check that ts are not out of bound
    assert np.max(t) <= np.max(tks)
    assert np.min(t) >= np.min(tks)

    # conditions = [np.logical_and(t > tks[k], t <= tks[k + 1]) for k in range(N)]
    conditions = [np.logical_and(t > tks[k], t <= tks[k + 1]) for k in range(N)]

    # k=k since https://stackoverflow.com/questions/45925683/list-comprehension-and-lambdas-in-python
    functions = [lambda x, k=k: k + (x - tks[k]) / T[k] for k in range(N)]

    return np.piecewise(t, conditions, functions)


def numericalToPhysicalTime(tau: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Converts the values of numerical time [0,t_end] to their respective values in physical time [0,N]
    :param tau: array of numerical times of shape (*,)
    :param T: array of cycle durations [T_0,T_1,...,T_N-1] of shape (N,)
    :return: array of physical times of shape (*,)
    """
    N = T.shape[0]
    assert N > 0

    # check bounds on tau
    assert np.min(tau) >= 0
    assert np.max(tau) <= N

    # "The output is the same shape and type as x",
    # https://numpy.org/doc/stable/reference/generated/numpy.piecewise.html
    assert tau.dtype == np.float64, "ndtype of array has to be float, piecewise interpolation wont work otherwise!"

    tks = np.concatenate([np.array([0]), np.cumsum(T)])

    # conditions = [np.logical_and(t > tks[k], t <= tks[k + 1]) for k in range(N)]
    conditions = [np.logical_and(tau > k, tau <= k + 1) for k in range(N)]

    # k=k since https://stackoverflow.com/questions/45925683/list-comprehension-and-lambdas-in-python
    functions = [lambda x, k=k: tks[k] + (x - k) * T[k] for k in range(N)]

    return np.piecewise(tau, conditions, functionls)


def constructPiecewiseCasadiExpression(decisionVariable: ca.SX, edges: List, expressions: List[ca.SX]) -> ca.SX:
    """
    Construct a piecewise casadi expression from a list of edges and functions for a given decision variable.
    For example, if the decision variable is x and the edges are [0,1,2] and the expressions are [f1(x),f2(x)] then the
    resulting expression is f1(x) if x is in [0,1) and f2(x) if x is in [1,2). (DAAMN COPILOT).
    For values outside, the function will return nan.

    DO NOT USE THIS FUNCTION FOR OPTIMIZATION, it is not differentiable at the edges.

    :param decisionVariable: the variable that is evaluated
    :param edges: a list of edge values, in ascending order
    :param expressions: a list of casadi.SX expressions.
    :return:
    """
    assert type(decisionVariable) == ca.SX, "The decision variable has to be a casadi.SX!"
    assert decisionVariable.shape == (1, 1), "The decision variable has to be a scalar!"
    assert type(edges) == list, "The edges have to be a list!"
    assert type(expressions) == list, "The functions have to be a list!"
    assert len(expressions) > 0, "There has to be at least one function!"
    assert len(edges) == len(expressions) + 1, "The number of edges has to be one more than the number of functions!"

    # check that edges are in ascending order
    assert np.all(np.diff(edges) > 0), "The edges have to be in ascending order!"

    outputExpression = ca.DM(0)

    # add nan for values outside the edges
    outputExpression += ca.if_else(decisionVariable < edges[0], ca.DM.nan(), 0)
    outputExpression += ca.if_else(decisionVariable >= edges[-1], ca.DM.nan(), 0)

    # iterate edges
    for edge_index in range(len(edges) - 1):
        # condition that we are in the interval
        _condition = (decisionVariable >= edges[edge_index]) * (decisionVariable < edges[edge_index + 1])

        # add the function to the output expression
        outputExpression += ca.if_else(_condition, expressions[edge_index], 0)

    return outputExpression


def linearInterpolation(xVals: np.ndarray, yVals: np.ndarray) -> ca.Function:
    """
    Constructs a casadi function that performs linear interpolation between the given points.
    :param xVals: (N,) np array
    :param yVals: (N,) np array
    :return: casadi function y_lin(x) for the linear interpolation
    """

    N = xVals.shape[0]
    assert N == yVals.shape[0]
    assert xVals.ndim == 1
    assert yVals.ndim == 1

    # sort the x values
    sort_indices = np.argsort(xVals)
    xVals = xVals[sort_indices]
    yVals = yVals[sort_indices]

    # symbolic variable for functions
    x = ca.SX.sym('x')

    linearInterpolations = []

    for n in range(N - 1):
        xcurrent, xnext = xVals[n], xVals[n + 1]
        ycurrent, ynext = yVals[n], yVals[n + 1]

        # linear interpolations between the 2 points
        y_lin_local = ycurrent + (ynext - ycurrent) / (xnext - xcurrent) * (x - xcurrent)

        linearInterpolations.append(y_lin_local)

    # combine to piecewiese linear interpolation
    y_lin = constructPiecewiseCasadiExpression(x, xVals.tolist(), linearInterpolations)

    return ca.Function('y_lin', [x], [y_lin])


def logSection(title: str):
    _numberDashes = 100
    logging.info("=" * _numberDashes)
    logging.info("=" * int((_numberDashes - len(title)) / 2 - 1) + " " + title + " " + "=" * int(
        (_numberDashes - len(title)) / 2 - (len(title) - 1) % 2))
    logging.info("=" * _numberDashes)


def gaussQuadraturePoints(N: int) -> np.ndarray:
    """
    Computes the Gauss quadrature points for the interval [0,1]
    :param N: number of points
    :return: (points
    """

    # compute the points and weights
    points, weights = np.polynomial.legendre.leggauss(N)

    # shift the points to the interval [0,1]
    points = (points + 1) / 2

    return points


class NumpyStruct:
    """
    Helper class to convert the values of a casadi DM struct to numpy arrays, especially for plotting.
    Indexing the instance will index the DM struct and return a numpy array of the result, with proper shapes.

    Example:

    >>> Xsim = model.x_struct.repeated(Xsim)
    >>> Xsim_plotting = NumpyStruct(Xsim)
    >>> plt.plot(Xsim_plotting[:, 't'], Xsim_plotting[:, 'x', 0])

    """

    def __init__(self, DMStruct):
        self.DMStruct = DMStruct

    def __getitem__(self, slice) -> np.ndarray:
        slice_result = self.DMStruct[slice]
        # convert to numpy array and squeeze
        return np.array(slice_result).squeeze()


def computeOrder(xvalues: np.ndarray, yvalues: np.ndarray) -> np.ndarray:
    """ Compute the order of convergence for given x and y values.
    Implements the formula order_i = (log(y_i+1)-log(y_i))/(log(x_i+1)-log(x_i))
    Artificially adds a final point to have the same length of the output and x (useful for plotting).
    """
    assert type(xvalues) == type(yvalues) == np.ndarray, 'xvalues and yvalues must be a numpy array'
    assert xvalues.size == yvalues.size, 'x and y must have the same size'

    difflogx = np.diff(np.log(xvalues))
    difflogy = np.diff(np.log(yvalues))

    order = difflogy / difflogx
    order = np.hstack([order, order[-1]])  # add last point to have the same length
    return order


def floatToExpStr(x: float, base=10) -> str:
    """
    Formats a float x to a string in the format $a \cdot 10^{b}$
    :param x: Input float value
    :param base: The base for the scientific notation, default is 10
    :return: A formatted LaTeX string
    """
    if x == 0:
        return '$0$'

    sign = '-' if x < 0 else ''
    x = np.abs(x)
    exponent = int(np.floor(np.emath.logn(base, x)))
    mantissa = x / base ** exponent

    if np.isclose(mantissa, 1, atol=1e-10):
        return f'${sign}{base}^{{{exponent}}}$'
    else:
        return f'${sign}{mantissa:g}\\cdot {base}^{{{exponent}}}$'

def getCDCoefficients(K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the coefficients the of the central difference approximation pattern,
     i.e. evaluate the value and derivative of the lg polys as b_k and c_k.

    For the implicit scheme use K even, for the explicit scheme use K odd.

    :param K: number of points that are used to calculate the central difference
    :return: taus, b, c
    """

    taus = np.arange(- (K - 1) / 2, (K - 1) / 2 + 0.01, 1)

    # build lagrange polynomials
    b = []
    c = []

    for k in range(K):
        tau = ca.SX.sym('tau')

        # build lg poly
        lgpoly = ca.DM(1)
        for j in range(K):
            if k != j:
                lgpoly *= (tau - taus[j]) / (taus[k] - taus[j])

        coeffs_f = ca.Function('coeffs_f', [tau], [lgpoly, ca.jacobian(lgpoly, tau)])
        bk, ck = coeffs_f(0)
        b.append(bk.full().squeeze())
        c.append(ck.full().squeeze())
    return taus, np.stack(b), np.stack(c)


def computeTrajectoryFourierCoefficients(sim: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Fourier coefficients a_0...a_D and b_0=0,b_1,...,b_D of a trajectory that spans the interval t ∈ [0,1]
    The fourier coefficients then relate to the fourier series approximation of the trajectory as

    x(t) = a_0 + sum_{d=1}^{D} a_d cos(2*pi*d*t) + b_d sin(2*pi*d*t)

    with D = N//2 + 1.

    :param sim: (numpy array) of shape (nx,N) where nx is the number of states and N is the number of time points
    :return: the fourier coefficients a and b of the trajectory
    """
    assert sim.ndim == 2, "The input has to be a 2D array"
    nx, N = sim.shape

    # fourier transform the trajectory
    sim_fft = np.fft.rfft(sim, norm='forward')  # the normalization is here, such that we don't have to divide the
    # coeffs by N later
    freq = np.fft.rfftfreq(N, d=1. / N)

    # compute sine and cosine coefficients from the fourier coefficients
    a = np.zeros((nx, freq.size))
    b = np.zeros((nx, freq.size))

    # offset term
    a[:, 0] = sim_fft[:, 0].real

    # since we have a real signal, the fourier coefficients are symmetric, thus easier formulas
    a[:, 1:] = 2 * np.real(sim_fft[:, 1:freq.size])
    b[:, 1:] = -2 * np.imag(sim_fft[:, 1:freq.size])
    return a, b, freq


class StructFunction(ca.Function):
    def __init__(self, *args, struct: struct_symSX = None):
        """ A wrapper around casadi.Function that enables that the output of the function is a structure.

        :param struct: The structure of the output of the function, the output values are casted into this structure
        """

        super().__init__(*args)
        self.struct: struct_symSX = struct

        # check the size of the struct and of the output of the function
        assert struct is not None, 'A struct must be provided!'
        assert self.n_out() == 1, 'Currently only functions with one output are supported'
        if type(self.struct) is struct_symSX:
            assert self.struct.shape == self.sx_out()[0].shape, f'The shape of the struct and the only output of the function must be the same but are {self.struct.shape} and {self.sx_out()[0].shape} respectively.'

    def __call__(self, *args, **kwargs) -> Union[DMStruct,SXStruct]:
        # cast the output of the function into the struct
        return self.struct(super().__call__(*args, **kwargs))

    def map(self, N) -> 'StructFunction':
        """ Returns a new StructFunction that casts the mapped output of the function into a REPEATED struct with N elements."""

        # somehow ca.Function(ca.Function(..)) works
        return StructFunction(super().map(N), struct = self.struct.repeated)

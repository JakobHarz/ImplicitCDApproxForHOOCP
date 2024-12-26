from typing import Union, List, Tuple

import casadi as ca
import numpy as np
from numpy.polynomial.polynomial import Polynomial as Poly_np
from scipy.special import comb


class Polynomial:

    def __init__(self, d: int):
        self.d = d # number of coefficients
        self.degree = d - 1
        self.tau = ca.SX.sym('tau', 1)

    @property
    def basis(self) -> ca.Function:
        """ Construct a function R -> R^d that evaluates the basis of the polynomial at a given tau"""
        raise NotImplementedError('This method should be implemented in the child class')

    def getEvalFunction(self, shape: Tuple[int, int] = (1, 1), fixedCoeffs: List[ca.DM] = None) -> ca.Function:
        """
        Generates a casadi function that evaluates the polynomial at a given point t of the form

        `F(t, [x0], x1, ..., xd)`

        where t is a scalar in [0,1] and x0, ..., xd are the collocation points of the provided shape.

        If fixed values for the nodes x0, ..., xd are provided, the function will be of the form

        `F(t)`

        :param shape: the shape of the collocation nodes, can be matrices or vectors
        :param fixedCoeffs: a list of fixed values for the nodes, if provided, the function will be of the form x(t) = F(t)
        """

        assert len(shape) == 2, 'shape should be a tuple of length 2'

        # total number of coefficients
        nx = shape[0] * shape[1]

        # did the user provide fixed coefficients?
        if fixedCoeffs is None:
            # create symbolic variables for the nodes
            Xs = []
            for i in np.arange(1, self.d + 1):
                Xs.append(ca.SX.sym(f'x{i}', shape))
        else:
            assert type(fixedCoeffs) == list, 'fixedCoeffs should be a list of DMs'
            assert len(fixedCoeffs) > 0, 'fixedCoeffs should contain at least one element'
            assert type(fixedCoeffs[0]) == ca.DM, 'fixedCoeffs should be a list of DMs'
            assert fixedCoeffs[0].shape == shape, 'The shape of the fixed coefficients should be equal to the shape of the collocation nodes'
            Xs = fixedCoeffs

        # reshape input variables into a matrix of shape (nx, d)
        p_vals = ca.horzcat(*[X.reshape((nx, 1)) for X in Xs])

        poly_val = p_vals @ self.basis(self.tau)

        # reshape the result into the original shape
        poly_val = ca.reshape(poly_val, shape)

        if fixedCoeffs is None:
            return ca.Function('PolyEval', [self.tau] + Xs, [poly_val], ['tau']+[f'coeff_{i} of shape {shape}' for i in range(1,self.d+1)], ['poly'])
        else:
            # assert type(fixedCoeffs) == ca.DM
            # assert fixedCoeffs.shape == (self.d, shape)
            return ca.Function('PolyEval', [self.tau], [poly_val], ['tau'], ['poly'])



class ChebyshevPoly(Polynomial):

    def __init__(self, d: int):
        super().__init__(d)
        self.times: ca.DM = self.collTimes(d)
    @property
    def basis(self) -> ca.Function:
        """ Construct a function [0,1] -> R^d that evaluates the Chebyshev basis at a given tau"""
        chebBasis = ca.vertcat(*[ca.cos(i * ca.acos(2 * self.tau - 1)) for i in range(self.d)])
        return ca.Function('ChebBasis', [self.tau], [chebBasis], ['tau'], ['basis'])

    @property
    def basis_der(self) -> ca.Function:
        """ Construct a function [0,1] -> R^d that evaluates the derivative of the Chebyshev basis at a given tau
        Covers the edge cases tau = 0, tau = 1.
        """

        tau = ca.SX.sym('tau')
        basis_der = ca.Function('ChebBasis', [tau], [ca.jacobian(self.basis(tau),tau)], ['tau'], ['basis'])

        # add if_else for the edges (0,1) to avoid numerical issues
        return_expression = ca.DM.zeros(self.d,1)

        return_expression += ca.if_else(tau == 0, 2*(-1)**(np.arange(0,self.d)+1)*(np.arange(0,self.d)**2), 0)
        return_expression += ca.if_else(tau == 1, 2*np.arange(0,self.d)**2, 0)
        return_expression += ca.if_else((tau > 0)*(tau < 1), basis_der(tau), 0)

        return ca.Function('ChebBasisDer', [tau], [return_expression], ['tau'], ['basis_der'])

    @classmethod
    def collTimes(cls,d):
        """ Compute the collocation points for a Chebyshev polynomial of degree d"""
        return ca.DM((np.cos(np.pi * np.arange(0,d)/(d-1)) +1)/2)

class MonomialPoly(Polynomial):

    def __init__(self, d: int):
        super().__init__(d)

    @property
    def basis(self) -> ca.Function:
        """ Construct a function R -> R^d  that evaluates the monomial basis [1,t,t^2,...t^(d-1)] at a given tau"""
        monoBasis = ca.vertcat(*[self.tau ** i for i in range(self.d)])
        return ca.Function('MonoBasis', [self.tau], [monoBasis], ['tau'], ['basis'])

    def getConversionMatrixToBernstein(self):
        """ Construct the conversion matrix from the monomial to the bernstein basis,
         such that the coefficients can be converted as c_bernstein = G @ c_monomial """
        G = np.zeros((self.d, self.d))
        for j in range(self.d):
            for k in range(0, j + 1):
                G[j, k] = comb(j, k) / comb(self.degree, k)
        return G


class LagrangePoly(Polynomial):

    def __init__(self, d: int, collTimes: Union[str,list] = 'legendre'):
        super().__init__(d)

        if type(collTimes) == str:
            assert collTimes in ['legendre', 'radau']
            self.times = ca.collocation_points(self.d, collTimes)
        else:
            assert len(collTimes) == self.d
            self.times = collTimes

    @property
    def basis(self) -> ca.Function:
        """ Construct a function R -> R^d that evaluates the Lagrange basis [l1,l2,...,ld] at a given tau

        :param collPoints: list of collocation points in [0,1] or string 'legendre' or 'radau'
        """



        # construct the lagrange basis
        _ls = []
        for j in range(self.d):
            l = 1
            for r in range(self.d):
                if r != j:
                    l *= (self.tau - self.times[r]) / (self.times[j] - self.times[r])
            _ls.append(l)
        lagrBasis = ca.vertcat(*_ls)
        return ca.Function('LagrBasis', [self.tau], [lagrBasis], ['tau'], ['basis'])

    @property
    def basis_np(self) -> List[Poly_np]:
        """ construct a list of basis NUMPY polynomials"""
        lg_basis = []
        times = self.times

        for i in range(self.d):
            l = Poly_np([1])

            for j in range(self.d):
                if j != i:
                    l *= Poly_np([-times[j], 1]) / (times[i] - times[j])
            lg_basis.append(l)
        return lg_basis
    def getConversionMatrixToMonomial(self) -> np.ndarray:
        """ Construct the conversion matrix from the lagrange to the monomial basis,
         such that the coefficients can be converted as c_monomial = M @ c_lagrange """
        V = np.vander(self.times, increasing=True)
        return np.linalg.inv(V)

    def getConversionMatrixToBernstein(self) -> np.ndarray:
        """ Construct the conversion matrix from the lagrange to the bernstein basis,
         such that the coefficients can be converted as c_bernstein = G @ c_lagrange """
        M = self.getConversionMatrixToMonomial()
        monoPoly = MonomialPoly(self.d)
        G = monoPoly.getConversionMatrixToBernstein()
        return G @ M

class BernsteinPoly(Polynomial):

    def __init__(self,d: int):
        super().__init__(d)

    @property
    def basis(self) -> ca.Function:
        """ Construct a function R -> R^d that evaluates the Bernstein basis at a given tau"""
        bernBasis = ca.vertcat(*[comb(self.degree,i) * self.tau**i * (1-self.tau)**(self.degree-i) for i in range(self.d)])
        return ca.Function('BernBasis', [self.tau], [bernBasis], ['tau'], ['basis'])
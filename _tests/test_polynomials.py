import unittest
import casadi as ca
import numpy as np
from Core.polynomials import MonomialPoly,LagrangePoly


class TestPolynomials(unittest.TestCase):

    def test_mono_basis(self):
        poly = MonomialPoly(4)
        monoBasis = poly.basis
        assert np.allclose(monoBasis(0),ca.DM([1,0,0,0]))
        assert np.allclose(monoBasis(1),ca.DM([1,1,1,1]))
        assert monoBasis(0).shape == (4,1)

    def test_mono_eval(self):
        poly = MonomialPoly(4)
        polyEvalF = poly.getEvalFunction()
        assert np.allclose(polyEvalF(0, *[1,1,1,1]),1)


    def test_lagr_basis(self):
        poly = LagrangePoly(3, collTimes=[0,0.5,1])
        lagrBasis = poly.basis
        assert np.allclose(lagrBasis(0),ca.DM([1,0, 0]))
        assert np.allclose(lagrBasis(1),ca.DM([0,0,1]))
        assert lagrBasis(0).shape == (3,1)

    def test_lagr_eval(self):
        poly = LagrangePoly(3, collTimes=[0,0.5,1])
        polyEvalF = poly.getEvalFunction()
        assert np.allclose(polyEvalF(0, *[1,0, 1]),1)
        assert np.allclose(polyEvalF(1, *[1,0, 1]),1)




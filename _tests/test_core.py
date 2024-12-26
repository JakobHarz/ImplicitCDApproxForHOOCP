import unittest
import casadi as ca
import numpy as np

if __name__ == '__main__':
    unittest.main()

class TestNCycleOCP(unittest.TestCase):
    def test_easy(self):
        #suppress output of matplotlib
        import matplotlib
        matplotlib.use('Agg')


class TestEnvelopeOCP(unittest.TestCase):
    def test_easy(self):
        #suppress output of matplotlib
        import matplotlib
        matplotlib.use('Agg')


class TestHelperFunction(unittest.TestCase):

    def test_piecewieExpression(self):
        from Core.tools import constructPiecewiseCasadiExpression

        x = ca.SX.sym("x")

        edges = [-1,0,1]
        values = [0,1]

        y = constructPiecewiseCasadiExpression(x,edges,values)
        f = ca.Function("f", [x], [y])

        assert f(-0.5) == 0
        assert f(0.5) == 1

import unittest
import numpy as np
from Core.integrators import RK4, ForwardEuler,ExpMid,LegendreCollocation, RadauCollocation, RungeKuttaIntegrator
import casadi as ca

from Core.tools import solveRootfindingProblem



class TestIntegrators(unittest.TestCase):

    def test_implit_explicit(self):

        butcherTableaus = [RK4(), ForwardEuler(), ExpMid(), LegendreCollocation(3), RadauCollocation(3)]

        from Models.linearoscillator import LinearOscillatorScaled

        # create model
        model = LinearOscillatorScaled(epsilon=0.001)

        x0bar = model.x_struct(0).cat
        x0bar[0] = 1

        # one step of size
        h = 0.01

        for butcherTableau in butcherTableaus:

            # RK4_explicit = integrator.explicitFunction(model.f, model.x, model.tau,expressions=[model.u], constant_syms=[model.u])

            xend_analytic = model.solutionMap(x0bar, h)
            # xend_explicit = RK4_explicit(x0bar,0, h, 0)

            # implicit solution
            integrator = RungeKuttaIntegrator(butcherTableau,model.f)
            G_irk, Y_irk, z_irk, G_irk_struct = integrator.implicitFunction()

            xend_SX = ca.SX.sym('xend', model.x.shape)

            w = ca.vertcat(xend_SX, z_irk)

            G_root = ca.Function('G_root', [w], [G_irk(x0bar, xend_SX, 0, h, z_irk)])
            wopt = solveRootfindingProblem(G_root, w0=ca.DM.zeros(w.shape))

            # Y_irk_sol = Y_irk(x0bar, 0, h, wopt)

            xend_implicit = wopt[:model.nx]
            assert np.allclose(xend_analytic.full(),xend_implicit.full(),atol=1e-2)

            # explicit solution (if available)
            if butcherTableau.isExplicit:
                F_expl = integrator.explicitFunction()
                xend_explicit = F_expl(x0bar, 0, h)['x_f']
                assert np.allclose(xend_analytic.full(),xend_explicit.full(),atol=1e-2)


    def test_RK4(self):
        x_sym = ca.SX.sym('x', 1)
        f = ca.Function('f', [x_sym, ca.SX.sym('t')], [-x_sym])

        butcherTableau = RK4()
        integrator = RungeKuttaIntegrator(butcherTableau, f)

        expl_res = integrator.explicitFunction()(1, 0, 0.1)
        x_end_irk = expl_res['x_f']

        # basci RK4 implementation

        def RK4step(f, x0, t0, h):
            k1 = f(x0, t0)
            k2 = f(x0 + h / 2 * k1, t0 + h / 2)
            k3 = f(x0 + h / 2 * k2, t0 + h / 2)
            k4 = f(x0 + h * k3, t0 + h)
            return x0 + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        diff_impl = RK4step(f, 1, 0, 0.1) - integrator.implicitSolution(1, 0, 0.1, [])["x_f"]
        diff_expl = RK4step(f, 1, 0, 0.1) - x_end_irk

        assert np.allclose(diff_impl.full(), 0)
        assert np.allclose(diff_expl.full(), 0)


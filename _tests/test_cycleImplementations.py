import unittest
from Models.linearoscillator import LinearOscillatorScaled
import casadi as ca

class TestCycleImpelmentation(unittest.TestCase):

    def test_SingleShootingFixedPeriod(self):
        from Core.CycleImplementations.FixedPeriodShooting import CycleSingleShootingFixedPeriod

        # create model and initial position
        model = LinearOscillatorScaled(epsilon=0)
        x0bar = model.x_struct(0)
        x0bar['x',0] = 1

        # create cycle simulation object and solve one cycle
        cycleSim = CycleSingleShootingFixedPeriod(model, Nint=100)
        zopt = cycleSim.solveCycle(x0bar)

        # check if the solution is correct (periodic solution)
        res = zopt['x_plus'] - x0bar
        res_norm = res[0:2].T@res[0:2] # only check first two states, third is time
        assert res_norm < 1E-3

    def test_MultipleShootingFixedPeriod(self):
        from Core.CycleImplementations.FixedPeriodShooting import CycleMultipleShootingFixedPeriod

        # create model and initial position
        model = LinearOscillatorScaled(epsilon=0)
        x0bar = model.x_struct(0)
        x0bar['x',0] = 1

        # create cycle simulation object and solve one cycle
        cycleSim = CycleMultipleShootingFixedPeriod(model, Nint=100)
        zopt = cycleSim.solveCycle(x0bar)

        # check if the solution is correct (periodic solution)
        res = zopt['x_plus'] - x0bar
        res_norm = res[0:2].T@res[0:2]
        assert res_norm < 1E-3

    def test_SingleShootingPhaseCond(self):
        from Core.CycleImplementations.PhaseConditionShooting import CycleSingleShootingPhaseCond
        from Core.phaseconditions import LinearBoundaryConditions

        # create model and initial position
        model = LinearOscillatorScaled(epsilon=0, parametrize_T=True)
        x0bar = model.x_struct(0)
        x0bar['x', 0] = 1

        # create cycle simulation object and solve one cycle
        phaseConditions = LinearBoundaryConditions(q = ca.DM([0,1,0]),bminus=ca.DM(0))
        cycleSim = CycleSingleShootingPhaseCond(model, phaseConditions, Nint=100)
        zopt = cycleSim.solveCycle(x0bar)

        # check if the solution is correct (periodic solution)
        res = zopt['x_plus'] - x0bar
        res_norm = res[0:2].T @ res[0:2]  # only check first two states, third is time
        assert res_norm < 1E-3

    def test_MultipleShootingPhaseCond(self):
        from Core.CycleImplementations.PhaseConditionShooting import CycleMultipleShootingPhaseCond
        from Core.phaseconditions import LinearBoundaryConditions

        # create model and initial position
        model = LinearOscillatorScaled(epsilon=0, parametrize_T=True)
        x0bar = model.x_struct(0)
        x0bar['x', 0] = 1

        # create cycle simulation object and solve one cycle
        phaseConditions = LinearBoundaryConditions(q = ca.DM([0,1,0]),bminus=ca.DM(0))
        cycleSim = CycleMultipleShootingPhaseCond(model, phaseConditions, Nint=100)
        zopt = cycleSim.solveCycle(x0bar)

        # check if the solution is correct (periodic solution)
        res = zopt['x_plus'] - x0bar
        res_norm = res[0:2].T @ res[0:2]  # only check first two states, third is time
        assert res_norm < 1E-3
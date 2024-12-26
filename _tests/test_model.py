import unittest

from casadi.tools import entry

from Core.model import Model

class GeneralModelTest(unittest.TestCase):
    def test_create_model(self):
        """
        Create a model with states a, b, c and controls u, v, w and check the dimensions
        """
        myModel = Model([entry('a', shape=1), entry('b', shape=1), entry('c', shape=1)],
                        [entry('u', shape=1), entry('v', shape=1), entry('w', shape=1)])
        self.assertEqual(myModel.x.shape, (3,1))
        self.assertEqual(myModel.u.shape, (3,1))
        self.assertEqual(myModel.stateLabels, ['a', 'b', 'c'])
        self.assertEqual(myModel.controlLabels, ['u', 'v', 'w'])
        self.assertEqual(myModel.nx, 3)
        self.assertEqual(myModel.nu, 3)

class PredatorPreyTest(unittest.TestCase):
    def test_create_predator_prey_model(self):
        """
        Create the predator prey model and check the dimensions
        """
        from Models.predatorprey import PredatorPrey
        from Core.parameters import Parameters
        params = Parameters()
        params.epsilon = 0.05

        myModel = PredatorPrey(params=params)
        self.assertEqual(myModel.x.shape, (3,1))
        self.assertEqual(myModel.u.shape, (1,1))
        self.assertEqual(myModel.T.shape, (1,1))
        self.assertEqual(myModel.stateLabels, ['r', 's', 't'])
        self.assertEqual(myModel.controlLabels, ['u'])
        self.assertEqual(myModel.nx, 3)
        self.assertEqual(myModel.nu, 1)
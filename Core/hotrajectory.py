from dataclasses import dataclass
from typing import List
import numpy as np


class HighlyOscillatoryTrajectory:
    """ Class to store a highly oscillatory trajectory by cycles

    Contains the following properties:
        - cycles: List[CycleTrajectory], a list of all cycles
        - N: int, the number of cycles
        - X: np.array (nx,..) the full state trajectory
        - xn: np.array (nx,N) the endpoints of all cycles
        - tau: np.array(..,) the full numerical times
        - Ts: np.array(N) the cycle durations
    Instances can be added to combine their cycles.

    TODO: add size checks
    """

    @dataclass
    class CycleTrajectory:
        """
        Class to store the trajectory of a single cycle
        """
        X: np.ndarray
        T: float
        U: np.ndarray
        tau: np.ndarray
        cycleCost: float = 0

    def __init__(self):
        self._cycles: List['HighlyOscillatoryTrajectory.CycleTrajectory'] = []

    def addCycle(self, X: np.ndarray, T: float, U: np.ndarray, tau:np.ndarray, cycleCost: float = 0):
        """ Adds a single cycle to the trajectory """
        assert type(X) == np.ndarray
        assert type(T) == float
        assert type(U) == np.ndarray, f'U has type {type(U)}'
        assert type(tau) == np.ndarray

        assert X.shape[1] == tau.shape[0], f'X.shape[1] = {X.shape[1]} != tau.shape[0] = {tau.shape[0]}'

        self._cycles.append(self.CycleTrajectory(X, T, U, tau, cycleCost))

    @property
    def N(self):
        """ The number of cycles"""
        return len(self._cycles)

    @property
    def X(self) -> np.ndarray:
        """ The concatenated state trajectory of all cycles"""
        return np.hstack([cycle.X for cycle in self._cycles])

    @property
    def U(self) -> np.ndarray:
        """ The concatenated state trajectory of all cycles"""
        return np.hstack([cycle.U for cycle in self._cycles])

    @property
    def Jint(self) -> float:
        """ The sum of all cycle costs"""
        return sum([cycle.cycleCost for cycle in self._cycles])

    def xn(self) -> np.ndarray:
        """ The startpoints of all cycles"""
        return np.hstack([cycle.X[:, 0] for cycle in self._cycles])

    @property
    def Ts(self) -> np.ndarray:
        """ The cycle durations of all cycles"""
        return np.array([float(cycle.T) for cycle in self._cycles])

    @property
    def tau(self) -> np.ndarray:
        """ The concatenated numerical time trajectory for all cycles"""
        return np.concatenate([cycle.tau for cycle in self._cycles])

    @property
    def cycles(self) -> List['HighlyOscillatoryTrajectory.CycleTrajectory']:
        """ A list of all cycles"""
        return self._cycles

    def __add__(self, other: 'HighlyOscillatoryTrajectory') -> 'HighlyOscillatoryTrajectory':
        # create an empty instance
        newTrajc = HighlyOscillatoryTrajectory()
        # add own cycles to instance
        newTrajc._cycles.extend(self._cycles)

        # shift tau by N
        for cycle in other.cycles:
            cycle.tau += self.N
        # add others cycles to instance
        newTrajc._cycles.extend(other._cycles)
        return newTrajc

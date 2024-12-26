import casadi as ca
from Core import integrators
from casadi.tools import struct_symSX, entry
import numpy as np
from typing import List



class Model:
    def __init__(self, stateEntries: List[entry], controlEntries: List[entry] = [], requiresTimescaling: bool = False,
                 parameters=None, symbolicParams=[], symbolicParamsNames=[], symbolicParamsDefaultValues=[]):

        # no parameters? create an empty one

        # symbolic model paramters?
        self.symbolicParams: List[ca.SX] = symbolicParams
        self.symbolicParamsNames: List[str] = symbolicParamsNames
        self.symbolicParamsDefaultValues: List[str] = symbolicParamsDefaultValues

        self._stateEntries = stateEntries
        self._controlEntries = controlEntries

        self.x_struct = struct_symSX(self._stateEntries)
        self.u_struct = struct_symSX(self._controlEntries)

        self.x = self.x_struct.cat
        self.u = self.u_struct.cat

        # self.T: ca.SX = ca.SX.sym('T', 1)
        self.tau = ca.SX.sym('tau')
        self.nx = self.x.shape[0]
        self.nu = self.u.shape[0]

        self._f = None  # The property has to be set!

        self.requiresTimescaling = requiresTimescaling
        """ If true, the model requires timescaling such that the unperturbed solution is 1-periodic. """

        # default value for state bounds
        self.lbx: struct_symSX = self.x_struct(-ca.inf)
        self.ubx: struct_symSX = self.x_struct(ca.inf)

        # default value for control bounds
        self.lbu: struct_symSX = self.u_struct(-ca.inf)
        self.ubu: struct_symSX = self.u_struct(ca.inf)

        # default value for timescaling bounds
        self.lbT = -ca.inf
        self.ubT = ca.inf

        self._stateStructLabels = self.x_struct.keys()
        self._controlStructLabels = self.u_struct.keys()

        self.lbv: struct_symSX = self.x_struct(-ca.inf)
        self.ubv: struct_symSX = self.x_struct(ca.inf)

        self._f = None  # Function handle for the casadi function

        # construct a dictionary from statenames to stateindeces
        self._stateIndexDict = {}
        _idx = 0
        for stateLabel in self._stateStructLabels:
            _state = self.x_struct[stateLabel]
            _len = _state.shape[0]
            self._stateIndexDict[stateLabel] = slice(_idx, _idx + _len)
            _idx += _len

        # construct a dictionary from controlnames to controlindeces‚
        self._controlIndexDict = {}
        _idx = 0
        for controlLabel in self._controlStructLabels:
            _control = self.u_struct[controlLabel]
            _len = _control.shape[0]
            self._controlIndexDict[controlLabel] = slice(_idx, _idx + _len)
            _idx += _len

        # construct a list of state names
        self.stateLabels: List[str] = []
        for stateName in self._stateStructLabels:
            if self.x_struct[stateName].shape[0] == 1:
                self.stateLabels.append(stateName)
            else:
                for i in range(self.x_struct[stateName].shape[0]):
                    self.stateLabels.append(stateName + "_" + str(i))

        self.controlLabels: List[str] = []
        for controlName in self._controlStructLabels:
            if self.u_struct[controlName].shape[0] == 1:
                self.controlLabels.append(controlName)
            else:
                for i in range(self.u_struct[controlName].shape[0]):
                    self.controlLabels.append(controlName + str(i))

        _solutionMap = None

    @property
    def solutionMap(self):
        if self._solutionMap is None:
            raise NotImplementedError('No analytical solution for this model!')
        return self._solutionMap

    @solutionMap.setter
    def solutionMap(self, newSolutionMap):
        self._solutionMap = newSolutionMap


    def stateNameToSlice(self, name: str):
        return self._stateIndexDict[name]

    def controlNameToSlice(self, name: str):
        return self._controlIndexDict[name]

    @property
    def f(self) -> ca.Function:
        """
        A casadi function handle for the dynamics of the model of the form

        ``xdot = f(x,tau,*)``

        where x is the state, tau is the time
        """
        if self._f is None:
            raise NotImplementedError('The dynamics function xdot = f(x,tau,*) has to be defined in the child class!')
        return self._f

    @f.setter
    def f(self, newFunction):
        # check if the function has the correct signature
        assert isinstance(newFunction, ca.Function)
        assert newFunction.n_in() >= 1
        assert newFunction.size_in(0) == (self.nx, 1) # first input is x
        assert newFunction.size_in(1) == (1, 1) # second input is tau s
        assert newFunction.name_in()[0] == 'x'
        assert newFunction.name_in()[1] == 'tau'
        if self.requiresTimescaling:
            # if the model requires timescaling, the third input has to be the period
            assert newFunction.n_in() >= 2
            assert newFunction.size_in(2) == (1, 1)
            assert newFunction.name_in()[2] == 'T'

        assert newFunction.n_out() == 1
        assert newFunction.size_out(0) == (self.nx, 1), f"The function has to return a vector of size ({self.nx},1), but has size {newFunction.size_out(0)}"
        self._f = newFunction

    def getApproxPeriod(self, x):
        """Approximates the period at a given state x"""
        raise NotImplementedError

    def forwardSimSimple(self, x0: np.ndarray, Tend, N: int, Uconst: np.ndarray = None, params=None,
                         NintPerN=5) -> np.ndarray:
        """
        Performs a simple simulation of the model with initial state x0 over the interval [0,Tend]. The number of RK4 integration steps is N.

        :param x0: the initial state
        :param Tend: duration
        :param N: number of integration points
        :param Uconst:
        :return: simulation result (nx,N+1)
        """

        if Uconst is None:
            Uconst = np.zeros(self.u.shape)

        # safety checks
        assert Uconst.shape == self.u.shape
        assert x0.shape == self.x.shape

        # build integrator
        h = Tend / (N * NintPerN)
        integrator = integrators.RK4()
        F_single = integrator.explicitFunction(self.f, self.x, self.tau, [self.u],[self.u])

        # build chained RK4 steps
        xnext = self.x
        for _m in range(NintPerN):
            t_m = _m * h
            xnext = F_single(xnext,t_m, h,Uconst)
        F_step = ca.Function('F_step', [self.x], [xnext], ['xk'], ['xk+1'])
        Fsim = F_step.mapaccum(N)

        # fix model parameters

        # perform actual simulation
        Xsim = Fsim(x0)

        # add initial state
        Xsim = ca.horzcat(x0, Xsim)

        return Xsim.full()
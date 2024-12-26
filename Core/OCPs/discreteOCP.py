from __future__ import annotations
import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Type

from casadi.tools import struct_symSX, entry

import Core
import casadi as ca
import numpy as np
from matplotlib import pyplot as plt

from Core import configuration
from Core.cycle import Cycle
from Core.macrodiscretizationschemes import MacroDiscretizationScheme
from Core.phaseconditions import LinearBoundaryConditions
from Core.model import Model



class DiscreteOCP:
    """ Base class for all discrete OCPs. """

    @dataclass
    class SolverStats:
        """Container Class for the Solver Statistics"""
        tProc: float = 0 # Processing time
        tWall: float = 0 # Wall time
        tProcperIter: float = 0 # CPU time per iteration
        Niter: int = 0 # Number of iterations
        Nsolve: int = 0 # Number of times the NLP was solved, for averaging

    def __init__(self, ocp: NCycleOCP, cycleSimClass: Cycle, wEntries: List[type(entry())]):

        self.ocp = ocp
        """The continuous time ocp """

        self.cycleSimClass = cycleSimClass
        """ The class that implements the micro-integration"""

        logging.info("Creating discrete OCP from the following continuous OCP:")
        self.ocp.printInfo()

        logging.info(f"With the following cycle simulation of type: {self.cycleSimClass} ")

        self.w = struct_symSX(wEntries)
        """ The optimization variables """


        self.ubw: ca.SX = self.w(-np.inf)
        """ The upper bound on the optimization variables, values are specified in the buildNLP() method """

        self.lbw: ca.SX = self.w(np.inf) # this is as nice trick to enforce that the user specifies the bounds
        """ The lower bound on the optimization variables """

        self.w0: ca.SX = self.w(0)
        """ The initial guess for the optimization variables """

        self.g: ca.SX = None
        self.lbg: ca.SX = None
        self.ubg: ca.SX = None

        self.p: ca.SX = None

        self.J_int: ca.SX = 0
        self.J_final: ca.SX = 0
        self.J_reg: ca.SX = 0

    def solveNLP(self, parameters: ca.DM = ca.vertcat([]),
                 max_iter=configuration.IPOPT_OPTIONS_MAX_ITER,
                 solver: str='ipopt',
                 additionalSolverOptions: dict = {},
                 supressOutput = False,
                 averageTimes_N= 1) -> 'Core.OCPs.NCycleOCP.NCcyleOCPSolution':
        """
        Solves the NLP associated with the Discrete Optimal Control Problem
        """
        # check that the NLP has been constructed
        _message = "NLP not built yet! Call buildNLP() first!"
        assert self.J_int is not None, _message
        assert self.J_final is not None, _message
        assert self.w is not None, _message
        assert self.g is not None, _message
        assert self.p is not None, _message
        assert solver in ['ipopt','sqpmethod'], f"Solver {solver} not implemented!"

        # check that the variables are of the correct shape
        assert self.w.shape == self.w0.shape == self.lbw.shape == self.ubw.shape, "w, w0, lbw, ubw have different shapes!"

        # check that the parameters are of the correct size
        assert parameters.shape == self.p.shape, f"Given parameters have wrong shape: {parameters.shape} instead of {self.p.shape}"

        logging.info("Transfering NLP to solver ...")

        # Create an NLP solver
        problem = {'f': self.J_int + self.J_final + self.J_reg, 'x': self.w, 'g': self.g, 'p': self.p}
        if solver == 'ipopt':
            options = {'ipopt.linear_solver': configuration.IPOPT_OPTIONS_LINEAR_SOLVER,
                       'ipopt.max_iter': max_iter,
                       'ipopt.print_level': 0 if supressOutput else 5,
                       # 'print_time': False if supressOutput else True,
                       }
        elif solver == 'sqpmethod':
            options = {'print_time': False,
                       'print_iteration': False,
                       'print_header': False,
                       'print_status': False,
                       'qpsol_options': {'printLevel': 'none'}}
            options = {}
        else:
            raise NotImplementedError(f"Solver {solver} not implemented!")


        # add additional options
        options.update(additionalSolverOptions)

        self.solver = ca.nlpsol('solver', solver, problem, options)

        logging.info("... finished!")

        logging.info(f"Solving NLP with parameters {parameters} ...")

        # Solve the NLP
        average_tProc = 0
        average_tWall = 0
        average_tProcPerIter = 0
        average_Niter = 0

        for i in range(averageTimes_N):
            nlpsolution = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg, p=parameters)

            tWall_ms = self.solver.stats()['t_wall_total'] * 1000
            tProc_ms = self.solver.stats()['t_proc_total'] * 1000
            Niter = self.solver.stats()['iter_count']

            logging.info(f"... finshed solving NLP,"
                         f" wall time: {tWall_ms :0.2f} ms,"
                         f" CPU time: {tProc_ms :0.2f} ms")

            average_tProc += tProc_ms/averageTimes_N
            average_tWall += tWall_ms/averageTimes_N
            average_tProcPerIter += (tProc_ms/(Niter if Niter>0 else 1))/averageTimes_N
            average_Niter += Niter/averageTimes_N

        solverStats = DiscreteOCP.SolverStats(tProc=average_tProc, tWall=average_tWall,tProcperIter=average_tProcPerIter, Niter=average_Niter, Nsolve=averageTimes_N)

        return self.formateOCPResult(nlpsolution, parameters,solverStats, self.solver.stats())

    def formateOCPResult(self, nlpSolution: dict, parameters: ca.DM, solverStats: DiscreteOCP.SolverStats, solverStatsDict: dict) -> 'Core.OCPs.NCycleOCP.NCcyleOCPSolution':
        raise NotImplementedError("This method has to be implemented by the child class!")

    def finalizeNLPConstruction(self):
        """
        Finalize the NLP constrution, by concatenating the constraints into one vectors and creating return functions.
        Has to be implemented by the child class.
        """
        raise NotImplementedError("This method has to be implemented by the child class!")

    def solveNLPHomotopy(self,
                         parameter: ca.SX,
                         steps: int,
                         max_iter: int = configuration.IPOPT_OPTIONS_MAX_ITER,
                         linear_solver: str = configuration.IPOPT_OPTIONS_LINEAR_SOLVER,
                         mu_target_homotopy: float = configuration.IPOPT_OPTIONS_MU_TARGET_HOMOTOPY):
        """
        Solve the NLP using a homotopy scheme
        :param parameter: homotopy parameter that is changed during the homotopy from 1 to 0
        :param steps: number of homotopy steps
        :param max_iter: maximum number of iterations for IPOPT
        :param linear_solver: linear solver for IPOPT
        :param mu_target_homotopy: mu target for the homotopy
        :return:
        """

        _message = "NLP not built yet! Call buildNLP() first!"
        assert self.J_int is not None, _message
        assert self.J_final is not None, _message
        assert self.w is not None, _message
        assert self.g is not None, _message
        assert self.p is not None, _message

        assert type(self.w) is not list, "w is a list, but should be a casadi.SX, did you call finalizeNLP()?"

        # check that the parameters are of the correct size
        assert parameter.shape == (1, 1), "Currently only homotopy for one parameter is supported!"
        assert parameter.shape == self.p.shape, f"Given parameter have wrong shape: {parameter.shape} instead of {self.p.shape}"

        logging.info(f"Solving the NLP with a homotopy scheme with {steps} steps ...")

        # create problem
        problem = {'f': self.J_int + self.J_final, 'x': self.w, 'g': self.g, 'p': self.p}

        # options for the initial solver
        options_initial = {'ipopt.linear_solver': linear_solver,
                           'ipopt.max_iter': max_iter,
                           'ipopt.print_level': 0 if self.supressOutput else 5,
                           'print_time': False if self.supressOutput else True,
                           'ipopt.mu_target': mu_target_homotopy,
                           'ipopt.tol': 1e-4,
                           }

        # options for the intermediate homotopy solver
        options_homotopy = {'ipopt.linear_solver': linear_solver,
                            'ipopt.max_iter': max_iter,
                            'ipopt.print_level': 0 if self.supressOutput else 5,
                            'print_time': False if self.supressOutput else True,
                            'ipopt.mu_init': mu_target_homotopy,
                            'ipopt.mu_target': mu_target_homotopy,
                            'ipopt.tol': 1e-4,
                            }

        # options for the final solver
        options_final = {'ipopt.linear_solver': linear_solver,
                         'ipopt.max_iter': max_iter,
                         'ipopt.print_level': 0 if self.supressOutput else 5,
                         'print_time': False if self.supressOutput else True,
                         'ipopt.mu_init': mu_target_homotopy,
                         }

        logging.info("Transfering NLP to solvers...")
        self.solver_start = ca.nlpsol('solver', 'ipopt', problem, options_initial)
        self.solver_homotopy = ca.nlpsol('solver', 'ipopt', problem, options_homotopy)
        self.solver_final = ca.nlpsol('solver', 'ipopt', problem, options_final)
        logging.info(".. done!")

        # start timer
        _starttime  = time.time()

        # solve the initial NLP
        logging.info("Solving initial Problem ...")
        nlpsolution_initial = self.solver_start(x0=self.w0,
                                                lbx=self.lbw,
                                                ubx=self.ubw,
                                                lbg=self.lbg,
                                                ubg=self.ubg,
                                                p=ca.DM(1))

        logging.info("... done!")

        # run the homotopy steps
        w0 = nlpsolution_initial['x']
        stepValues = np.linspace(1, 0, steps)
        for i in range(steps):
            logging.info(f"Running homotopy step {i + 1} of {steps} ...")
            nlpsolution_homotopy = self.solver_homotopy(x0=w0,
                                                        lbx=self.lbw,
                                                        ubx=self.ubw,
                                                        lbg=self.lbg,
                                                        ubg=self.ubg,
                                                        p=ca.DM(stepValues[i])
                                                        )
            w0 = nlpsolution_homotopy['x']

        logging.info("... done!")

        # solve the final NLP
        logging.info("Solving final Problem ...")
        nlpsolution_final = self.solver_final(x0=w0,
                                              lbx=self.lbw,
                                              ubx=self.ubw,
                                              lbg=self.lbg,
                                              ubg=self.ubg,
                                              p=ca.DM(0))
        logging.info("... done!")

        # stop timer
        _duration = time.time() - _starttime
        logging.info(f"... finshed solving NLP, wall time: {_duration * 1000:0.2f} ms ")

        return self.formateOCPResult(nlpsolution_final, ca.DM(0))

    def constructNLP(self):
        raise NotImplementedError("This method has to be implemented in the child class!")




class NCycleOCP():


    def __init__(self,
                 model: Model,
                 tf: int,
                 x0bar: ca.DM,
                 stageCostFunction: Callable = None,
                 finalCostFunction: Callable = None,
                 terminalConstraints: Callable = None,
                 xendguess: ca.DM = None,
                 Uguess: ca.DM = None):

        """Continuous N-Cycle OCP
        Class to store functions and parameter that define an N-Cycle-OCP.

        :param model: used model
        :param N: number of cycles
        :param x0bar: start position
        :param phaseConditions: phase conditions that define one cycle
        :param stageCostFunction: callable function x,u -> L(x,u)
        :param finalCostFunction: callable function x -> E(x)
        :param xendguess: guess for the final point, useful for initialization
        """

        self.model: Model = model
        """ used model """

        self.tf: int = tf
        """horizon, should be an integer for most applications"""

        self.x0bar: ca.DM = x0bar
        """ start position """

        self._defaultStageCostFunction: Callable = (lambda x, u: 0)
        self.stageCostFunction: Callable = (stageCostFunction if stageCostFunction is not None else self._defaultStageCostFunction)
        """ callable function x,u -> L(x,u), default is zero cost"""

        self._defaultFinalCostFunction: Callable = (lambda x: 0)
        self.finalCostFunction: Callable = (finalCostFunction if finalCostFunction is not None else self._defaultFinalCostFunction)
        """ finalCostFunction: callable function x -> E(x), default is zero cost """

        self._defaultTerminalConstraints: ca.Function = (lambda x: ([], [], []))
        self.terminalConstraints: Callable = (terminalConstraints if terminalConstraints is not None else self._defaultTerminalConstraints)
        """ terminalConstraints: callable function x -> (g,lbg,ubg), default is empty constraints"""

        self.xendguess: ca.DM = xendguess
        """ guess for the final point, useful for initialization """

        self.Uguess: ca.DM = Uguess
        """ guess for the control input matrix over one cycle """
    def printInfo(self):
        """ Logs information about the OCP. """
        logging.info("-------- OCP info ------------")
        logging.info("Model: " + self.model.__class__.__name__)

        logging.info(f"N: {self.tf}")
        logging.info(f"x0bar: {self.x0bar.cat.full().flatten()}")
        # logging.info(f"phaseConditions: {self.DEPRIVATED_phaseConditions}")
        logging.info(
            f"stageCostFunction: {'ZeroCost' if self.stageCostFunction == self._defaultStageCostFunction else 'Custom StageCost Function'} : {self.stageCostFunction}")
        logging.info(
            f"finalCostFunction: {'ZeroCost' if self.finalCostFunction == self._defaultFinalCostFunction else 'Custom FinalCost Function'} : {self.finalCostFunction}")
        logging.info(
            f"terminalConstraints: {'No Constraints' if self.terminalConstraints == self._defaultTerminalConstraints else 'Custom Terminal Constraints'}: {self.terminalConstraints}")
        logging.info("-------------------------------")

    def runSanityChecks(self):
        logging.info("Performing sanity checks for cont. OCP ... ")

        # check for if all needed values are set
        assert self.x0bar is not None
        # assert self.DEPRIVATED_phaseConditions is not None

        # check if parameters are not set
        if self.stageCostFunction is None:
            def _stageCost(x, u):
                return 0

            self.stageCostFunction = _stageCost  # TODO: check if this is correct

        if self.finalCostFunction is None:
            def _finalCost(x):
                return 0

            self.finalCostFunction = _finalCost  # TODO: check if this is correct

        if self.terminalConstraints is None:
            def _terminalConstraints(x):
                return 0, 0, 0  # TODO: values

            self.terminalConstraints = _terminalConstraints

        if self.xendguess is None:
            self.xendguess = self.x0bar

        assert callable(self.stageCostFunction)
        assert callable(self.finalCostFunction)
        assert self.x0bar.shape == self.model.x.shape
        # assert type(self.phaseConditions) == LinearPhaseConditions

        logging.info("... Done!")


class OCPSolution:
    """ Abstract Class for the solution of an OCP"""

    def __init__(self):
        self.ocp: NCycleOCP = None
        """The N-Cycle OCP that this instance describes the solution of"""

        self.Jopt: np.ndarray = None
        """The optimal objective"""

        self.Jintopt: np.ndarray = None
        """The optimal integral objective"""

        self.Jfinopt: np.ndarray = None
        """The optimal terminal objective"""

        self.nlpsolution: dict = None
        """ The NLP solution, dict as returned by CasADi"""

        self.lagrangian_hessian: np.ndarray = 0
        """ The Hessian of the Lagrangian at the optimal solution"""

        self.solverStats: DiscreteOCP.SolverStats = None

        self.solverStats_dict: dict = None

    def logComparison(self: OCPSolution, other: OCPSolution):
        logging.info(f'\t\t\t | {self.__class__.__name__}] \t | {other.__class__.__name__}')
        logging.info('-----------------------------------------')
        logging.info('Status \t | %.12s  | %.12s', str(self.solverStats_dict['return_status']), other.solverStats_dict['return_status'])
        logging.info('-----------------------------------------')
        logging.info('Full Cost \t | %f \t | %f', self.Jopt, other.Jopt)
        logging.info('Stage Cost\t | %f \t | %f', self.Jintopt, other.Jintopt)
        logging.info('Final Cost\t | %f \t | %f', self.Jfinopt, other.Jfinopt)
        logging.info("-----------------------------------------")
        logging.info('tProc \t \t | %0.2f s \t | %0.2f s', self.solverStats.tProc/1000, other.solverStats.tProc/1000)
        logging.info('tProc/Iter\t | %0.2f ms \t | %0.2f ms ', self.solverStats.tProcperIter, other.solverStats.tProcperIter)

    def plotState(self):
        raise NotImplementedError()

    def plotControls(self):
        raise NotImplementedError()

    def checkInitialized(self):
        """Check if all the required attributes have been set"""
        for key, val in vars(self).items():
            assert val is not None, f"Attribute {key} not set!"

    def plotHessian(self):
        """Plot the sparsity pattern of the Hessian"""
        plt.figure('Sparsity of Hessian')
        plt.spy(self.lagrangian_hessian)
        plt.plot()

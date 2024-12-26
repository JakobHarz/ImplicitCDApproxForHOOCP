from Core.integrators import ButcherTableau, LegendreCollocation, RadauCollocation

class MacroDiscretizationScheme:

    def __init__(self, tauf: int, Nintervals: int, integrationScheme: ButcherTableau):
        """
        Class used to store configuration for the discretization scheme for the macro-integrator

        :param tauf: Length of the macro-integration horizon (usually integer)
        :param Nintervals:  Number of intervals used for the macro-integrator
        :param integrationScheme: Integration scheme (e.g. RK4,Collocation) used for the macro-integrator
        """

        assert isinstance(integrationScheme, ButcherTableau)
        assert isinstance(Nintervals, int)
        assert Nintervals >= 1

        self.tauf = tauf
        """ Length of the macro-integration horizon (usually integer)"""

        self.tauf_stages = tauf/Nintervals
        """ Length of a single stage (usually integer)"""

        self.integrationScheme: ButcherTableau = integrationScheme
        """ Integration scheme (e.g. RK4,Collocation) used for the macro-integrator"""

        self.Nintervals: int = Nintervals
        """ Number of intervals used for the macro-integrator"""

    def __str__(self):
        return f"Nintervals = {self.Nintervals}, integrationScheme={self.integrationScheme}"

class LegendreCollocationMacroScheme(MacroDiscretizationScheme):

    def __init__(self, tauf: int, Nintervals: int, dColl: int):
        """
        Class used to store configuration for the discretization scheme for the macro-integrator

        :param tauf: Length of the macro-integration horizon (usually integer)
        :param Nintervals:  Number of intervals used for the macro-integrator
        :param dColl: Degree of the collocation polynomials
        """

        assert isinstance(dColl, int)
        assert dColl >= 1

        self.d_coll: int = dColl
        """ Number of stages of the collocation method"""

        super().__init__(tauf,Nintervals, LegendreCollocation(dColl))

class RadauCollocationMacroScheme(MacroDiscretizationScheme):

    def __init__(self, tauf: int, Nintervals: int, dColl: int):
        """
        Class used to store configuration for the discretization scheme for the macro-integrator

        :param tauf: Length of the macro-integration horizon (usually integer)
        :param Nintervals:  Number of intervals used for the macro-integrator
        :param d_coll: Degree of the collocation polynomials
        """
        assert isinstance(dColl, int)
        assert dColl >= 1

        self.d_coll: int = dColl
        """ Number of stages of the collocation method"""

        super().__init__(tauf,Nintervals, RadauCollocation(dColl))


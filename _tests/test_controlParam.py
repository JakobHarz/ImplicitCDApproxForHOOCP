from Core.controlparameterization import FourierControlParameterization


def test_controlParam():
    import casadi as ca
    from Core.parameters import Parameters
    from Core.polynomials import LagrangePoly
    import numpy as np
    from Core.controlparameterization import FourierControlParameterization

    tauf = 30
    nu = 2
    Nfourier = 1
    controlPoly = LagrangePoly(2)

    U1 = ca.DM([[1, 0, 1],
                [1, 0, 0]])
    U2 = ca.DM([[1, 0, 0],
                [1, 0, 1]])

    params = Parameters()
    params.U_cycles = [U1, U2]
    params.tauf = tauf

    controlParam = FourierControlParameterization(nu, controlPoly, N_har=Nfourier, parameters=params)

    tau_plot = np.linspace(0, tauf, 100)
    uplot = controlParam.getPlottingFunction().map(tau_plot.size)(tau_plot).full().squeeze()
    express = controlParam.u_f(0, 0)

    assert uplot.shape == (nu, tau_plot.size)
    assert express.shape == (nu, 1)

    # plt.plot(tau_plot, uplot[0])
    # plt.plot(tau_plot, uplot[1])
    # plt.show()
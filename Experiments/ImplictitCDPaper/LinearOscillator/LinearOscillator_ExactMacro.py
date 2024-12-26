import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from Core.tools import computeOrder, getCDCoefficients
from Core.figuretools import latexify, smoothPlotLimits
from Models.linearoscillator import LinearOscillatorScaled
from Core.configuration import PLT_GRID_ALPHA

latexify()  # change to latex fonts

model = LinearOscillatorScaled(epsilon=0)

x0bar = model.x_struct(0)
x0bar["x", 0] = 1
x0bar["x", 1] = 0
x0bar_x = x0bar['x'].full()

N = 1
Kmax = 7
epsilons = -np.logspace(-0.5, -4.5, 30)

errors_CD = np.zeros((epsilons.size, Kmax))

for index_eps, epsilon in enumerate(epsilons):

    # tau_f = 50
    tau_f = -0.05/epsilon

    Amodel = np.array([[0, -2 * np.pi], [2 * np.pi, 0]]) + epsilon * np.eye(2)
    Aaverage = epsilon * np.eye(2)

    expmA = sc.linalg.expm(np.matrix(Amodel))
    xendpoint_true = sc.linalg.expm(np.matrix(Aaverage) * tau_f) @ x0bar_x

    # tau_plotting = np.linspace(0,10,1000)
    # Xanalytic = model.analyticalSolution_tau.map(tau_plotting.size)(x0bar,tau_plotting).full()
    # Xaverage = model.analyticalAverageSolution_tau.map(tau_plotting.size)(x0bar,tau_plotting).full()

    # compute the matrix A_F of the implicit approximation
    for index_K, K in enumerate(np.arange(2, Kmax + 0.01, 1, dtype=int)):
        nx = 2  # two states
        D = np.zeros((K * nx, K * nx))
        # fill diagonal blocks
        for i in range(K - 1):
            D[i * nx:(i + 1) * nx, i * nx:(i + 1) * nx] = expmA
            D[i * nx:(i + 1) * nx, (i + 1) * nx:(i + 2) * nx] = -1 * np.eye(nx)

        # compute coefficients of the interpolation polynomial
        taus, b, c = getCDCoefficients(K)

        # fill last row
        D[K * nx - 2:K * nx, :] = np.hstack([np.eye(nx) * b_k for b_k in b])

        matrix_picklastentry = np.zeros((K * nx, 2))
        matrix_picklastentry[-nx:, :] = np.eye(nx)

        c_matrix = np.hstack([np.eye(nx) * c_k for c_k in c])
        A_FiCD = c_matrix @ np.linalg.inv(D) @ matrix_picklastentry

        xendpoint_CD = sc.linalg.expm(A_FiCD * tau_f) @ x0bar_x

        errors_CD[index_eps, index_K] = np.linalg.norm(xendpoint_true - xendpoint_CD)

# %% plot the error


plt.figure(figsize=(7, 5))
# plt.subplot(2,1,1)
# plt.title("Error of the endpoint with exact macro-integration")

colors = np.repeat(['C0','C1','C2','C3'],2)
linestyles = np.tile(['-','--'],Kmax//2)
alpha = np.tile([1,0.5],Kmax//2)
orders = np.repeat(np.arange(2,Kmax+0.01,2,dtype=int)+1,2)

for index_K, K in enumerate(np.arange(2, Kmax + 0.01, 1, dtype=int)):
    errors = np.abs(errors_CD[:, index_K]).T
    errors[errors*np.abs(epsilons) < 1E-16] = np.nan
    plt.loglog(np.abs(epsilons), errors,
               color = colors[index_K],
               linestyle = linestyles[index_K],
               alpha = alpha[index_K],
               # label=f'$F_{{\\mathrm{{CD,{K}}}}} - \mathcal{{O}}\\!\\left(\\epsilon^{{-1}}\\epsilon^{orders[index_K]}\\right)$')
               label=f'$F_{{\\mathrm{{CD,{K}}}}}$')

plt.gca().invert_xaxis()
plt.grid(alpha=PLT_GRID_ALPHA)
plt.legend()
plt.xlabel(r'$\epsilon$')
plt.ylabel('Error')

# plt.subplot(2,1,2)
# for index_K,K in enumerate(np.arange(2,Kmax+0.01,2)):
#     errors = errors_iCD[:, index_K]
#     order = computeOrder(epsilons,errors )
#     order[errors<1E-14] = np.nan
#     plt.semilogx(epsilons,order , label=f'$F_{{\\mathrm{{iCD,{K}}}}}$')
# plt.ylim(smoothPlotLimits(2,10))
# plt.yticks(np.arange(2,10.1,1))
# plt.grid(alpha=PLT_GRID_ALPHA)
# plt.gca().invert_xaxis()
# plt.legend()
# plt.xlabel(f'$\epsilon$')
# plt.ylabel('Approximation Order')
plt.tight_layout()
plt.savefig('../../../_Export/ErrorCD/LinearOscillator_ExactMacroMicro.pdf')
plt.show()

 # %% compute the solution via expm
# X_iF_expm = []
# for tau in tau_plotting:
#     X_iF_expm.append(sc.linalg.expm(A_F*tau)@x0bar['x'].full())
# X_iF_expm = np.hstack(X_iF_expm)

# %% Plotting
# plt.figure(figsize=(10,6))
# plt.plot(tau_plotting, Xanalytic[0], alpha=0.5, label='analytic')
# plt.plot(tau_plotting, Xaverage[0], alpha=0.5, label='average')
# plt.plot(tau_plotting, X_iF_expm[0], label='exactMacro')
# plt.legend()
# plt.show()

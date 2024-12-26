# Implicit Central Difference Approximations of Averaged Dynamics of Highly Oscillatory Systems with Applications to Direct Optimal Control

Code repository for the replication of the experimental part of the Paper _'Implicit Central Difference Approximations of Averaged Dynamics of Highly Oscillatory Systems with Applications to Direct Optimal Control'_
by Jakob Harzer, Jochem De Schutter, Per Rutquit, Moritz Diehl

Link to full Paper: TODO

| Fig. | Description                                                                                                                                                                                                        | File |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---|
| 7    | Damped Linear Oscillator - Order of the Average Dynamnics Approximation                                                                                                                                            | [LinearOscillator_ExactMacro.py](Experiments%2FImplictitCDPaper%2FLinearOscillator%2FLinearOscillator_ExactMacro.py) |
| 8    | Perturbed Kepler Problem - Integration Endpoint Error                                                                                                                                                              | [Satellite_EndpointExperiments_Replicate.py](Experiments%2FImplictitCDPaper%2FSatellite%2FSatellite_EndpointExperiments_Replicate.py) |
| 9    | Perturbed Kepler Problem - Pareto Optimal Configuration                                                                                                                                                            |  [Satellite_EndpointExperiments_Pareto.py](Experiments%2FImplictitCDPaper%2FSatellite%2FSatellite_EndpointExperiments_Pareto.py) |
| 10   | Perturbed Kepler Problem - Optimal Control Example - SAM Problem, in the code modify the line `ADA_TYPE = 'CD2'` to either `'FD'`,`'CD3'`, or `'CD2'` to select a different average dynamics approximation method. | [Satellite_OCP_SAM.py](Experiments%2FImplictitCDPaper%2FSatellite%2FSatellite_OCP_SAM.py) |
| 11   | Perturbed Kepler Problem - Optimal Control Example - Full Problem                                                                                                                                                  | [Satellite_OCP_Full.py](Experiments%2FImplictitCDPaper%2FSatellite%2FSatellite_OCP_Full.py) |

## Requirements

- Python 3.8 with:
- CasADi
- Linear Solver `MA27` from [HSL](https://licences.stfc.ac.uk/product/coin-hsl)
- Numpy
- Matplotlib
- Scipy
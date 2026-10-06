# NumPy Portfolio Risk Simulator

A compact quantitative-finance project built around **NumPy**. It uses Monte Carlo simulation to estimate a portfolio's:

- Expected one-period return
- 5% Value at Risk (VaR)
- 5% Conditional Value at Risk (CVaR / Expected Shortfall)
- Probability of losing more than a chosen threshold
- Simulated terminal portfolio-value distribution

The project is intentionally NumPy-centric so that the numerical work—vectorized simulation, covariance calculations, random sampling, and risk statistics—is visible rather than hidden behind a finance library.

## Project structure

```text
numpy_portfolio_risk_simulator/
├── README.md
├── requirements.txt
├── config.json
├── portfolio_risk/
│   ├── __init__.py
│   ├── cli.py
│   └── simulator.py
└── tests/
    └── test_simulator.py
```

## Requirements

- Python 3.10+
- NumPy

Install:

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m portfolio_risk.cli
```

Or supply your own configuration:

```bash
python -m portfolio_risk.cli --config config.json
```

Example output:

```text
Portfolio Risk Simulation
-------------------------
Initial portfolio value : $100,000.00
Expected terminal value : $101,234.56
Expected return         : 1.23%
VaR (5%)                : $-7,845.12
CVaR (5%)               : $-10,432.71
P(loss > 10%)           : 18.42%
Simulations              : 100,000
```

Exact values vary with the random seed and configuration.

## Model

For each asset \(i\), the one-period simple return is approximated as:

\[
R_i \sim N(\mu_i,\Sigma)
\]

where:

- \(\mu_i\) is the expected return
- \(\Sigma\) is the covariance matrix

Correlated returns are generated with a Cholesky factorization:

\[
R = \mu + ZL^\top
\]

where \(Z\) contains independent standard-normal draws and \(LL^\top=\Sigma\).

Portfolio returns are:

\[
R_p = Rw
\]

and terminal portfolio values are:

\[
V_1 = V_0(1+R_p)
\]

### Risk measures

At confidence level \(c\), the loss distribution is:

\[
L = V_0 - V_1
\]

VaR is the \(c\)-quantile of losses:

\[
VaR_c = Q_c(L)
\]

CVaR is the mean loss among observations at or beyond that quantile:

\[
CVaR_c = E[L\mid L\ge VaR_c]
\]

## Example portfolio

The included `config.json` models a fictional portfolio containing:

- BTC: 35%
- ETH: 20%
- Gold: 20%
- U.S. equities: 25%

The parameters are illustrative, not investment advice.

## Why this is a useful NumPy project

This project demonstrates several core NumPy skills that transfer directly to data science and quantitative research:

1. Array broadcasting and vectorization
2. Matrix multiplication
3. Covariance matrices
4. Cholesky decomposition
5. Random-number generation
6. Quantiles and conditional statistics
7. Avoiding Python loops over individual simulations

## Extend it

Good next steps:

- Add historical-return estimation from CSV
- Compare Monte Carlo VaR with historical VaR
- Add stress scenarios
- Add portfolio optimization
- Add matplotlib visualizations
- Replace the normal-return assumption with Student's t or bootstrap sampling

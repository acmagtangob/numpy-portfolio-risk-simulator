from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PortfolioConfig:
    initial_value: float
    confidence: float
    loss_threshold: float
    simulations: int
    seed: int
    names: tuple[str, ...]
    weights: np.ndarray
    means: np.ndarray
    volatilities: np.ndarray
    correlation: np.ndarray


@dataclass(frozen=True)
class SimulationResult:
    terminal_values: np.ndarray
    portfolio_returns: np.ndarray
    losses: np.ndarray
    expected_terminal_value: float
    expected_return: float
    var: float
    cvar: float
    probability_of_large_loss: float


def load_config(path: str | Path) -> PortfolioConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    assets = raw["assets"]
    names = tuple(asset["name"] for asset in assets)
    weights = np.array([asset["weight"] for asset in assets], dtype=float)
    means = np.array([asset["mean"] for asset in assets], dtype=float)
    volatilities = np.array([asset["volatility"] for asset in assets], dtype=float)
    correlation = np.array(raw["correlation"], dtype=float)

    _validate(
        initial_value=float(raw["initial_value"]),
        confidence=float(raw["confidence"]),
        loss_threshold=float(raw["loss_threshold"]),
        simulations=int(raw["simulations"]),
        names=names,
        weights=weights,
        means=means,
        volatilities=volatilities,
        correlation=correlation,
    )

    return PortfolioConfig(
        initial_value=float(raw["initial_value"]),
        confidence=float(raw["confidence"]),
        loss_threshold=float(raw["loss_threshold"]),
        simulations=int(raw["simulations"]),
        seed=int(raw["seed"]),
        names=names,
        weights=weights,
        means=means,
        volatilities=volatilities,
        correlation=correlation,
    )


def _validate(
    *,
    initial_value: float,
    confidence: float,
    loss_threshold: float,
    simulations: int,
    names: tuple[str, ...],
    weights: np.ndarray,
    means: np.ndarray,
    volatilities: np.ndarray,
    correlation: np.ndarray,
) -> None:
    n = len(names)

    if initial_value <= 0:
        raise ValueError("initial_value must be positive.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")
    if loss_threshold < 0:
        raise ValueError("loss_threshold must be non-negative.")
    if simulations < 1:
        raise ValueError("simulations must be at least 1.")
    if len({len(weights), len(means), len(volatilities), n}) != 1:
        raise ValueError("Asset vectors must all have the same length.")
    if correlation.shape != (n, n):
        raise ValueError("Correlation matrix must be square and match asset count.")
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    if np.any(weights < 0):
        raise ValueError("This example expects long-only weights.")
    if np.any(volatilities <= 0):
        raise ValueError("Volatilities must be positive.")
    if not np.allclose(correlation, correlation.T, atol=1e-10):
        raise ValueError("Correlation matrix must be symmetric.")
    if not np.allclose(np.diag(correlation), 1.0, atol=1e-10):
        raise ValueError("Correlation matrix diagonal must be 1.")
    eigenvalues = np.linalg.eigvalsh(correlation)
    if eigenvalues.min() < -1e-10:
        raise ValueError("Correlation matrix must be positive semidefinite.")


def simulate_portfolio(config: PortfolioConfig) -> SimulationResult:
    _validate(
        initial_value=config.initial_value,
        confidence=config.confidence,
        loss_threshold=config.loss_threshold,
        simulations=config.simulations,
        names=config.names,
        weights=config.weights,
        means=config.means,
        volatilities=config.volatilities,
        correlation=config.correlation,
    )

    rng = np.random.default_rng(config.seed)

    covariance = (
        config.volatilities[:, None]
        * config.correlation
        * config.volatilities[None, :]
    )

    # Add tiny diagonal regularization only for numerical stability.
    covariance = covariance + np.eye(len(config.names)) * 1e-12
    cholesky = np.linalg.cholesky(covariance)

    # Vectorized Monte Carlo simulation:
    # Z is (simulations x assets), and Z @ L.T produces correlated shocks.
    z = rng.standard_normal((config.simulations, len(config.names)))
    asset_returns = config.means + z @ cholesky.T

    portfolio_returns = asset_returns @ config.weights
    terminal_values = config.initial_value * (1.0 + portfolio_returns)
    losses = config.initial_value - terminal_values

    alpha = 1.0 - config.confidence
    var = float(np.quantile(losses, config.confidence))

    tail_losses = losses[losses >= var]
    cvar = float(tail_losses.mean())

    threshold_loss = config.initial_value * config.loss_threshold
    probability_of_large_loss = float(np.mean(losses >= threshold_loss))

    return SimulationResult(
        terminal_values=terminal_values,
        portfolio_returns=portfolio_returns,
        losses=losses,
        expected_terminal_value=float(terminal_values.mean()),
        expected_return=float(portfolio_returns.mean()),
        var=var,
        cvar=cvar,
        probability_of_large_loss=probability_of_large_loss,
    )

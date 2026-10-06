import numpy as np
import pytest

from portfolio_risk.simulator import PortfolioConfig, simulate_portfolio


def make_config() -> PortfolioConfig:
    return PortfolioConfig(
        initial_value=10_000.0,
        confidence=0.95,
        loss_threshold=0.10,
        simulations=20_000,
        seed=123,
        names=("A", "B", "C"),
        weights=np.array([0.5, 0.3, 0.2]),
        means=np.array([0.01, 0.005, 0.002]),
        volatilities=np.array([0.03, 0.02, 0.01]),
        correlation=np.array(
            [
                [1.0, 0.4, 0.1],
                [0.4, 1.0, 0.2],
                [0.1, 0.2, 1.0],
            ]
        ),
    )


def test_simulation_shape_and_finiteness():
    config = make_config()
    result = simulate_portfolio(config)

    assert result.terminal_values.shape == (config.simulations,)
    assert result.portfolio_returns.shape == (config.simulations,)
    assert result.losses.shape == (config.simulations,)
    assert np.isfinite(result.terminal_values).all()
    assert np.isfinite(result.losses).all()


def test_reproducibility():
    config = make_config()
    first = simulate_portfolio(config)
    second = simulate_portfolio(config)

    np.testing.assert_array_equal(first.terminal_values, second.terminal_values)


def test_expected_return_is_reasonable():
    config = make_config()
    result = simulate_portfolio(config)

    # The theoretical weighted mean is 0.0071.
    assert abs(result.expected_return - 0.0071) < 0.001


def test_cvar_is_at_least_var():
    config = make_config()
    result = simulate_portfolio(config)

    assert result.cvar >= result.var


def test_invalid_confidence():
    with pytest.raises(ValueError):
        config = make_config()
        object.__setattr__(config, "confidence", 1.0)
        simulate_portfolio(config)

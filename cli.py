from __future__ import annotations

import argparse
from pathlib import Path

from .simulator import load_config, simulate_portfolio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="NumPy Monte Carlo portfolio risk simulator."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.json"),
        help="Path to the JSON configuration file.",
    )
    return parser


def money(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    result = simulate_portfolio(config)

    print("Portfolio Risk Simulation")
    print("-------------------------")
    print(f"Initial portfolio value : {money(config.initial_value)}")
    print(f"Expected terminal value : {money(result.expected_terminal_value)}")
    print(f"Expected return         : {result.expected_return:.2%}")
    print(f"VaR ({config.confidence:.0%})               : {money(result.var)}")
    print(f"CVaR ({config.confidence:.0%})              : {money(result.cvar)}")
    print(
        f"P(loss > {config.loss_threshold:.0%})       : "
        f"{result.probability_of_large_loss:.2%}"
    )
    print(f"Simulations              : {config.simulations:,}")


if __name__ == "__main__":
    main()

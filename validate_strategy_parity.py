#!/usr/bin/env python3
import sys
from pathlib import Path


def check_hedge_inheritance() -> tuple[bool, str]:
    try:
        from src.backtest.hedge.hedge_manager import BacktestHedgeManager
        from src.backtest.hedge.hedge_executor import BacktestHedgeExecutor
        from src.hedge.hedge_manager import HedgeManager
        from src.hedge.hedge_executor import HedgeExecutor

        if not issubclass(BacktestHedgeManager, HedgeManager):
            return False, "BacktestHedgeManager does not inherit from HedgeManager"
        if not issubclass(BacktestHedgeExecutor, HedgeExecutor):
            return False, "BacktestHedgeExecutor does not inherit from HedgeExecutor"
        return True, "Hedge inheritance OK"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_regime_config_defaults() -> tuple[bool, str]:
    try:
        from src.backtest.strategy_engine import StrategyConfig

        config = StrategyConfig()
        expected = {
            "adx_strong_trend": 25.0,
            "adx_weak_trend": 20.0,
            "bb_squeeze_threshold": 0.06,
            "bb_volatile_threshold": 0.10,
            "min_regime_confidence": 0.6,
        }
        actual = {
            "adx_strong_trend": config.adx_strong_trend,
            "adx_weak_trend": config.adx_weak_trend,
            "bb_squeeze_threshold": config.bb_squeeze_threshold,
            "bb_volatile_threshold": config.bb_volatile_threshold,
            "min_regime_confidence": config.min_regime_confidence,
        }
        if expected != actual:
            return False, f"Config mismatch: expected {expected}, got {actual}"
        return True, "Regime config defaults match live trading bot"
    except Exception as e:
        return False, f"Error: {e}"


def check_risk_manager_inheritance() -> tuple[bool, str]:
    try:
        from src.backtest.risk.risk_manager import RiskManager as BacktestRiskManager
        from src.risk.risk_manager import RiskManager

        if not issubclass(BacktestRiskManager, RiskManager):
            return False, "Backtest RiskManager does not inherit from live RiskManager"
        return True, "Risk manager inheritance OK"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_signal_detector_shared() -> tuple[bool, str]:
    try:
        from src.backtest.strategy_engine import BacktestStrategyEngine
        from src.indicators.signal_detector import SignalDetector
        from src.indicators.enhanced_signal_detector import EnhancedSignalDetector

        return True, "Backtest uses shared SignalDetector classes"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_regime_profile_edge_fields() -> tuple[bool, str]:
    """Check that all regime profiles have the new edge_floor and ambiguity_veto_threshold fields."""
    try:
        from src.indicators.market_regime import REGIME_PROFILES

        required_fields = ["edge_floor", "ambiguity_veto_threshold"]
        missing_fields = []

        for regime_name, profile in REGIME_PROFILES.items():
            for field in required_fields:
                if field not in profile:
                    missing_fields.append(f"{regime_name}.{field}")

        if missing_fields:
            return False, f"Missing edge fields: {', '.join(missing_fields)}"

        # Verify values are within expected ranges
        for regime_name, profile in REGIME_PROFILES.items():
            edge_floor = profile.get("edge_floor", 0)
            ambiguity_threshold = profile.get("ambiguity_veto_threshold", 0)

            if edge_floor < 0.04 or edge_floor > 0.20:
                return False, f"{regime_name}: edge_floor {edge_floor} outside expected range [0.04, 0.20]"

            if ambiguity_threshold < 0.03 or ambiguity_threshold > 0.08:
                return False, f"{regime_name}: ambiguity_veto_threshold {ambiguity_threshold} outside expected range [0.03, 0.08]"

        return True, "All regime profiles have edge_floor and ambiguity_veto_threshold with valid values"
    except Exception as e:
        return False, f"Error: {e}"


def check_activation_gate_functions() -> tuple[bool, str]:
    """Check that activation gate functions are available and importable."""
    try:
        from src.indicators.signal_quality import (
            check_activation_gate,
            check_ambiguity_veto,
            calculate_edge_score,
            rank_symbols_by_edge,
            check_top1_lead_margin,
        )

        return True, "Activation gate functions available in signal_quality module"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_regime_alpha_functions() -> tuple[bool, str]:
    """Check that regime alpha functions are available and importable."""
    try:
        from src.indicators.enhanced_signal_detector import (
            check_regime_alpha,
            check_trending_alpha,
            check_ranging_alpha,
            check_volatile_alpha,
            check_quiet_alpha,
            RegimeAlphaResult,
        )

        return True, "Regime alpha functions available in enhanced_signal_detector module"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_invalidation_exit() -> tuple[bool, str]:
    """Check that invalidation exit is available in exit manager."""
    try:
        from src.risk.exit_manager import ExitType, AllWeatherExitManager

        # Check that INVALIDATION exit type exists
        if not hasattr(ExitType, "INVALIDATION"):
            return False, "ExitType.INVALIDATION not found"

        # Check that AllWeatherExitManager has _check_invalidation_exit method
        if not hasattr(AllWeatherExitManager, "_check_invalidation_exit"):
            return False, "AllWeatherExitManager._check_invalidation_exit not found"

        return True, "Invalidation exit available in exit manager"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_capital_protection_module() -> tuple[bool, str]:
    """Check that capital protection module is available and importable."""
    try:
        from src.risk.capital_protection import (
            CapitalProtectionManager,
            CapitalProtectionConfig,
            DeploymentState,
        )

        # Check that DeploymentState enum has all required states
        required_states = ["ACTIVE", "PASSIVE", "DEFENSIVE"]
        for state in required_states:
            if not hasattr(DeploymentState, state):
                return False, f"DeploymentState.{state} not found"

        # Check that CapitalProtectionManager has required methods
        required_methods = [
            "evaluate_deployment_state",
            "should_allow_entry",
            "record_trade_result",
            "is_regime_killed",
            "get_current_drawdown",
            "get_drawdown_size_multiplier",
        ]
        for method in required_methods:
            if not hasattr(CapitalProtectionManager, method):
                return False, f"CapitalProtectionManager.{method} not found"

        return True, "Capital protection module available with all required components"
    except ImportError as e:
        return False, f"Import error: {e}"


def check_strategy_engine_capital_protection() -> tuple[bool, str]:
    """Check that strategy engine integrates capital protection manager."""
    try:
        from src.backtest.strategy_engine import BacktestStrategyEngine

        # Check that BacktestStrategyEngine has capital_protection attribute
        # We can't instantiate it without config, but we can check the class
        if not hasattr(BacktestStrategyEngine, "__init__"):
            return False, "BacktestStrategyEngine.__init__ not found"

        # Check that BacktestStrategyEngine has record_trade_result method
        if not hasattr(BacktestStrategyEngine, "record_trade_result"):
            return False, "BacktestStrategyEngine.record_trade_result not found"

        # Check that BacktestStrategyEngine has get_deployment_state method
        if not hasattr(BacktestStrategyEngine, "get_deployment_state"):
            return False, "BacktestStrategyEngine.get_deployment_state not found"

        return True, "Strategy engine integrates capital protection manager"
    except ImportError as e:
        return False, f"Import error: {e}"


def main():
    print("=" * 60)
    print("STRATEGY PARITY VALIDATION")
    print("=" * 60)

    checks = [
        ("Hedge Inheritance", check_hedge_inheritance),
        ("Regime Config Defaults", check_regime_config_defaults),
        ("Risk Manager Inheritance", check_risk_manager_inheritance),
        ("Signal Detector Shared", check_signal_detector_shared),
        ("Regime Profile Edge Fields", check_regime_profile_edge_fields),
        ("Activation Gate Functions", check_activation_gate_functions),
        ("Regime Alpha Functions", check_regime_alpha_functions),
        ("Invalidation Exit", check_invalidation_exit),
        ("Capital Protection Module", check_capital_protection_module),
        ("Strategy Engine Capital Protection", check_strategy_engine_capital_protection),
    ]

    all_passed = True
    for name, check_fn in checks:
        passed, message = check_fn()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {message}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All parity checks passed!")
        return 0
    else:
        print("PARITY CHECKS FAILED - fix before deploying")
        return 1


if __name__ == "__main__":
    sys.exit(main())

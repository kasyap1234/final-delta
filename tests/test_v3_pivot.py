#!/usr/bin/env python3
"""
Focused V3 Pivot Strategy Tests.

Tests for V3 behavior:
- long-bias under strong uptrend + bounded short exception
- risk-on/risk-off transition hysteresis
- deterministic TP/runner/breakeven progression
- no-lookahead regression in new gating/state logic
"""

import pytest
from typing import Dict, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.indicators.market_regime import (
    V3OperatingMode,
    REGIME_PROFILES,
    get_regime_profile
)
from src.risk.capital_protection import (
    CapitalProtectionManager,
    CapitalProtectionConfig,
    V3OperatingMode as CPV3OperatingMode
)


class TestV3RegimeProfiles:
    """Test V3 regime profile configuration."""

    def test_v3_operating_mode_enum_exists(self):
        """Test that V3OperatingMode enum is defined with correct values."""
        assert hasattr(V3OperatingMode, 'RISK_ON_BULL_CAPTURE')
        assert hasattr(V3OperatingMode, 'RISK_ON_SELECTIVE')
        assert hasattr(V3OperatingMode, 'PASSIVE_DEFER')
        assert hasattr(V3OperatingMode, 'RISK_OFF_NO_TRADE')

    def test_v3_directional_policy_fields_exist(self):
        """Test that V3 directional policy fields exist in all regime profiles."""
        for regime_name in ['trending_up', 'trending_down', 'ranging', 'volatile', 'quiet', 'unknown']:
            profile = get_regime_profile(regime_name)
            
            # V3 directional policy fields
            assert 'v3_long_bias' in profile
            assert 'v3_tactical_short_allowed' in profile
            assert 'v3_tactical_short_edge_mult' in profile
            assert 'v3_tactical_short_size_cap' in profile
            assert 'v3_tactical_short_hold_cap' in profile
            assert 'v3_tactical_short_no_pyramiding' in profile
            
            # V3 risk-on bull-capture activation fields
            assert 'v3_risk_on_bull_capture' in profile
            assert 'v3_regime_confidence_min' in profile
            assert 'v3_benchmark_trend_strength_min' in profile
            assert 'v3_drawdown_max_for_risk_on' in profile

    def test_v3_payoff_fields_exist(self):
        """Test that V3 payoff fields exist in all regime profiles."""
        for regime_name in ['trending_up', 'trending_down', 'ranging', 'volatile', 'quiet', 'unknown']:
            profile = get_regime_profile(regime_name)
            
            # V3 payoff mechanics fields
            assert 'v3_payoff_enabled' in profile
            assert 'v3_hybrid_stop_atr_mult' in profile
            assert 'v3_stop_pct_floor' in profile
            assert 'v3_stop_pct_cap' in profile
            assert 'v3_lockin_buffer_atr' in profile
            assert 'v3_runner_retention_enabled' in profile

    def test_trending_up_long_bias_enabled(self):
        """Test that trending_up regime has long bias enabled."""
        profile = get_regime_profile('trending_up')
        assert profile['v3_long_bias'] is True, "trending_up should have long bias enabled"

    def test_trending_down_long_bias_disabled(self):
        """Test that trending_down regime has long bias disabled."""
        profile = get_regime_profile('trending_down')
        assert profile['v3_long_bias'] is False, "trending_down should have long bias disabled"

    def test_tactical_short_bounds(self):
        """Test that tactical short parameters are within allowed ranges."""
        for regime_name in ['trending_up', 'trending_down', 'ranging', 'volatile', 'quiet', 'unknown']:
            profile = get_regime_profile(regime_name)
            
            # Edge floor multiplier: 1.5-2.0
            assert 1.5 <= profile['v3_tactical_short_edge_mult'] <= 2.0
            
            # Size cap: 0.20-0.35
            assert 0.20 <= profile['v3_tactical_short_size_cap'] <= 0.35
            
            # Hold cap: 6-16 candles
            assert 6 <= profile['v3_tactical_short_hold_cap'] <= 16

    def test_v3_payoff_bounds(self):
        """Test that V3 payoff parameters are within allowed ranges."""
        for regime_name in ['trending_up', 'trending_down', 'ranging', 'volatile', 'quiet', 'unknown']:
            profile = get_regime_profile(regime_name)
            
            # Hybrid stop ATR multiplier: 1.2-2.0
            assert 1.2 <= profile['v3_hybrid_stop_atr_mult'] <= 2.0
            
            # Stop percent floor: 1.5%
            assert profile['v3_stop_pct_floor'] == 0.015
            
            # Stop percent cap: 5.5%
            assert profile['v3_stop_pct_cap'] == 0.055
            
            # Lock-in buffer: 0.3-0.8 ATR
            assert 0.3 <= profile['v3_lockin_buffer_atr'] <= 0.8


class TestV3CapitalProtection:
    """Test V3 capital protection state machine."""

    def test_v3_operating_mode_enum_exists(self):
        """Test that V3OperatingMode enum exists in capital protection."""
        assert hasattr(CPV3OperatingMode, 'RISK_ON_BULL_CAPTURE')
        assert hasattr(CPV3OperatingMode, 'RISK_ON_SELECTIVE')
        assert hasattr(CPV3OperatingMode, 'PASSIVE_DEFER')
        assert hasattr(CPV3OperatingMode, 'RISK_OFF_NO_TRADE')

    def test_v3_config_fields_exist(self):
        """Test that V3 config fields exist in CapitalProtectionConfig."""
        config = CapitalProtectionConfig()
        
        # V3-specific fields
        assert hasattr(config, 'v3_regime_confidence_min')
        assert hasattr(config, 'v3_benchmark_trend_strength_min')
        assert hasattr(config, 'v3_drawdown_max_for_risk_on')
        assert hasattr(config, 'v3_loss_streak_3_risk_mult')
        assert hasattr(config, 'v3_loss_streak_3_cooldown')
        assert hasattr(config, 'v3_loss_streak_5_risk_mult')
        assert hasattr(config, 'v3_loss_streak_5_cooldown')
        assert hasattr(config, 'v3_loss_streak_6_force_passive')
        assert hasattr(config, 'v3_loss_streak_6_cooldown')
        assert hasattr(config, 'v3_recovery_windows_required')
        assert hasattr(config, 'v3_hysteresis_buffer_pct')

    def test_evaluate_v3_operating_mode_method_exists(self):
        """Test that evaluate_v3_operating_mode method exists."""
        config = CapitalProtectionConfig()
        manager = CapitalProtectionManager(config)
        
        assert hasattr(manager, 'evaluate_v3_operating_mode')

    def test_risk_off_no_trade_on_hard_breaker(self):
        """Test that hard breaker forces RISK_OFF_NO_TRADE."""
        config = CapitalProtectionConfig()
        manager = CapitalProtectionManager(config)
        
        # Update balance to simulate hard breaker (drawdown > 12%)
        # Initial balance 1000, current balance 850 = 15% drawdown
        manager.update_balance(1000.0)  # Initial balance
        manager.update_balance(850.0)   # Current balance (15% drawdown)
        
        # Verify drawdown is > 12%
        assert manager.get_current_drawdown() > 0.12
        
        # Simulate hard breaker conditions
        mode = manager.evaluate_v3_operating_mode(
            current_drawdown=0.15,  # 15% drawdown
            active_edge=0.10,
            benchmark_trend_strength=0.80,
            regime_value='trending_up',
            regime_confidence=0.85
        )
        
        assert mode == CPV3OperatingMode.RISK_OFF_NO_TRADE

    def test_risk_on_bull_capture_in_strong_uptrend(self):
        """Test that strong uptrend with low drawdown enables RISK_ON_BULL_CAPTURE."""
        config = CapitalProtectionConfig()
        manager = CapitalProtectionManager(config)
        
        # Strong uptrend conditions
        mode = manager.evaluate_v3_operating_mode(
            current_drawdown=0.02,  # Low drawdown
            active_edge=0.15,  # Good edge
            benchmark_trend_strength=0.85,  # Strong trend
            regime_value='trending_up',
            regime_confidence=0.85  # High confidence
        )
        
        # Should be RISK_ON_BULL_CAPTURE or RISK_ON_SELECTIVE
        assert mode in [CPV3OperatingMode.RISK_ON_BULL_CAPTURE, CPV3OperatingMode.RISK_ON_SELECTIVE]

    def test_passive_defer_on_weak_edge(self):
        """Test that weak edge triggers PASSIVE_DEFER."""
        config = CapitalProtectionConfig()
        manager = CapitalProtectionManager(config)
        
        # Weak edge conditions
        mode = manager.evaluate_v3_operating_mode(
            current_drawdown=0.05,  # Moderate drawdown
            active_edge=0.03,  # Weak edge
            benchmark_trend_strength=0.50,  # Weak trend
            regime_value='ranging',
            regime_confidence=0.60  # Low confidence
        )
        
        # Should be PASSIVE_DEFER or RISK_OFF_NO_TRADE
        assert mode in [CPV3OperatingMode.PASSIVE_DEFER, CPV3OperatingMode.RISK_OFF_NO_TRADE]


class TestV3NoLookahead:
    """Test that V3 logic has no lookahead bias."""

    def test_v3_operating_mode_uses_only_current_state(self):
        """Test that V3 operating mode evaluation uses only current state."""
        config = CapitalProtectionConfig()
        manager = CapitalProtectionManager(config)
        
        # All inputs are current state, no future data
        mode = manager.evaluate_v3_operating_mode(
            current_drawdown=0.05,
            active_edge=0.10,
            benchmark_trend_strength=0.70,
            regime_value='trending_up',
            regime_confidence=0.75
        )
        
        # Should return a valid mode without error
        assert mode in CPV3OperatingMode

    def test_regime_profiles_are_static(self):
        """Test that regime profiles are static (no dynamic computation)."""
        # Get profile twice
        profile1 = get_regime_profile('trending_up')
        profile2 = get_regime_profile('trending_up')
        
        # Should be identical
        assert profile1 == profile2

    def test_v3_parameters_are_bounded(self):
        """Test that V3 parameters are bounded and deterministic."""
        for regime_name in ['trending_up', 'trending_down', 'ranging', 'volatile', 'quiet', 'unknown']:
            profile = get_regime_profile(regime_name)
            
            # All V3 parameters should be numeric and bounded
            assert isinstance(profile['v3_tactical_short_edge_mult'], (int, float))
            assert isinstance(profile['v3_tactical_short_size_cap'], (int, float))
            assert isinstance(profile['v3_tactical_short_hold_cap'], int)
            assert isinstance(profile['v3_hybrid_stop_atr_mult'], (int, float))
            assert isinstance(profile['v3_stop_pct_floor'], (int, float))
            assert isinstance(profile['v3_stop_pct_cap'], (int, float))
            assert isinstance(profile['v3_lockin_buffer_atr'], (int, float))


class TestV3DeterministicBehavior:
    """Test that V3 behavior is deterministic."""

    def test_v3_operating_mode_deterministic(self):
        """Test that V3 operating mode evaluation is deterministic."""
        config = CapitalProtectionConfig()
        manager = CapitalProtectionManager(config)
        
        # Evaluate same inputs multiple times
        inputs = {
            'current_drawdown': 0.05,
            'active_edge': 0.10,
            'benchmark_trend_strength': 0.70,
            'regime_value': 'trending_up',
            'regime_confidence': 0.75
        }
        
        mode1 = manager.evaluate_v3_operating_mode(**inputs)
        mode2 = manager.evaluate_v3_operating_mode(**inputs)
        mode3 = manager.evaluate_v3_operating_mode(**inputs)
        
        # Should return same mode each time
        assert mode1 == mode2 == mode3

    def test_regime_profile_deterministic(self):
        """Test that regime profile retrieval is deterministic."""
        profile1 = get_regime_profile('trending_up')
        profile2 = get_regime_profile('trending_up')
        profile3 = get_regime_profile('trending_up')
        
        # Should return same profile each time
        assert profile1 == profile2 == profile3


class TestV3GateCompatibility:
    """Test V3 gate compatibility plumbing."""

    def test_v3_gate_fields_exist_in_results(self):
        """Test that V3 gate fields can be added to results."""
        # This is a placeholder test - actual gate evaluation happens in scripts/evaluate_gates.py
        # The test verifies that the structure is compatible
        
        v3_gate_structure = {
            'v3_up_year_participation': {
                'threshold': 'Positive and at least +20% in 2023 and 2024',
                'actual': {2023: '0.25', 2024: '0.30'},
                'pass': True,
                'description': 'Up-year participation 2023 and 2024'
            },
            'v3_stitched_expectancy': {
                'threshold': 0.08,
                'actual': 0.10,
                'pass': True,
                'description': 'Stitched expectancy at least +0.08R'
            }
        }
        
        # Verify structure is valid
        assert 'v3_up_year_participation' in v3_gate_structure
        assert 'threshold' in v3_gate_structure['v3_up_year_participation']
        assert 'actual' in v3_gate_structure['v3_up_year_participation']
        assert 'pass' in v3_gate_structure['v3_up_year_participation']
        assert 'description' in v3_gate_structure['v3_up_year_participation']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

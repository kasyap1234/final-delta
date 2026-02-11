"""
Indicator Manager Module

Manages all technical indicators for multiple symbols, including
calculation, caching, and updates from OHLCV data.

Supports incremental O(1) updates when a single new candle is added,
falling back to full O(n) recalculation when data is bulk-replaced.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from datetime import datetime
import logging

from .technical_indicators import (
    calculate_ema,
    calculate_rsi,
    calculate_atr,
    calculate_pivot_points_from_ohlcv,
    calculate_all_emas,
    CrossoverType,
    calculate_ema_crossover,
    calculate_adx
)

logger = logging.getLogger(__name__)


@dataclass
class IndicatorValues:
    """Container for all indicator values for a symbol."""
    
    # EMAs
    ema_9: Optional[float] = None
    ema_21: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    
    # RSI
    rsi: Optional[float] = None
    rsi_period: int = 14
    
    # ATR
    atr: Optional[float] = None
    atr_period: int = 14
    
    # ADX
    adx: Optional[float] = None
    adx_period: int = 14
    
    # Pivot Points
    pivot: Optional[float] = None
    r1: Optional[float] = None
    s1: Optional[float] = None
    r2: Optional[float] = None
    s2: Optional[float] = None
    
    # Trend
    trend: str = 'neutral'  # 'uptrend', 'downtrend', 'neutral'
    
    # Crossover
    last_crossover: CrossoverType = CrossoverType.NONE
    crossover_index: Optional[int] = None
    
    # Metadata
    timestamp: Optional[datetime] = None
    symbol: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert indicator values to dictionary."""
        return {
            'ema_9': self.ema_9,
            'ema_21': self.ema_21,
            'ema_50': self.ema_50,
            'ema_200': self.ema_200,
            'rsi': self.rsi,
            'rsi_period': self.rsi_period,
            'atr': self.atr,
            'atr_period': self.atr_period,
            'adx': self.adx,
            'adx_period': self.adx_period,
            'pivot': self.pivot,
            'r1': self.r1,
            's1': self.s1,
            'r2': self.r2,
            's2': self.s2,
            'trend': self.trend,
            'last_crossover': self.last_crossover.value,
            'crossover_index': self.crossover_index,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'symbol': self.symbol
        }


class IndicatorManager:
    """
    Manages technical indicators for multiple trading symbols.
    
    Handles calculation, caching, and updates of all indicators
    used in the trading strategy.
    """
    
    # EMA multipliers (precomputed)
    _EMA_MULTIPLIERS = {
        9: 2.0 / 10.0,
        21: 2.0 / 22.0,
        50: 2.0 / 51.0,
        200: 2.0 / 201.0,
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.rsi_period = self.config.get('rsi_period', 14)
        self.atr_period = self.config.get('atr_period', 14)
        self.pivot_lookback = self.config.get('pivot_lookback', 10)
        self.ema_periods = self.config.get('ema_periods', [9, 21, 50, 200])
        
        self._ohlcv_cache: Dict[str, List[List[float]]] = defaultdict(list)
        self._indicator_cache: Dict[str, IndicatorValues] = {}
        self._ema_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._max_cache_size = self.config.get('max_cache_size', 500)
        
        # Incremental state per symbol
        self._inc_state: Dict[str, Dict[str, Any]] = {}
        # Track data length to detect incremental vs bulk updates
        self._prev_len: Dict[str, int] = {}
        # Cached price arrays (invalidated on data change)
        self._price_arrays_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._price_arrays_len: Dict[str, int] = {}
        
        logger.info(f"IndicatorManager initialized with RSI({self.rsi_period}), "
                   f"ATR({self.atr_period}), Pivot({self.pivot_lookback})")
    
    def update_ohlcv(self, symbol: str, ohlcv: List[List[float]]) -> None:
        if not ohlcv:
            return
        
        new_len = len(ohlcv)
        state = self._inc_state.get(symbol)
        
        if state is not None:
            prev_len = state.get('data_len', 0)
            if new_len == prev_len + 1:
                # One new candle appended — incremental is safe
                pass
            elif new_len == prev_len:
                # Same length: either trim+append (net zero) or unchanged
                last_close = ohlcv[-1][4]
                if last_close != state.get('prev_close'):
                    # New candle at end after trim — still incrementally updateable
                    # The previous candle's data (used for prev_close/high/low)
                    # is now at index -2
                    if new_len >= 2:
                        prev_candle = ohlcv[-2]
                        state['prev_close'] = prev_candle[4]
                        state['prev_high'] = prev_candle[2]
                        state['prev_low'] = prev_candle[3]
                        state['data_len'] = new_len - 1  # so calculate_all sees +1
                    else:
                        self._inc_state.pop(symbol, None)
            else:
                self._inc_state.pop(symbol, None)
        
        self._ohlcv_cache[symbol] = ohlcv
        
        if new_len > self._max_cache_size:
            self._ohlcv_cache[symbol] = ohlcv[-self._max_cache_size:]
            self._inc_state.pop(symbol, None)
        
        self._price_arrays_cache.pop(symbol, None)
        
        logger.debug(f"Updated OHLCV for {symbol}: {len(ohlcv)} candles")
    
    def add_candle(self, symbol: str, candle: List[float]) -> None:
        self._ohlcv_cache[symbol].append(candle)
        
        if len(self._ohlcv_cache[symbol]) > self._max_cache_size:
            self._ohlcv_cache[symbol].pop(0)
            self._inc_state.pop(symbol, None)
        
        self._prev_len[symbol] = len(self._ohlcv_cache[symbol])
        self._price_arrays_cache.pop(symbol, None)
    
    def calculate_all(self, symbol: str, ohlcv: Optional[List[List[float]]] = None) -> IndicatorValues:
        if ohlcv is not None:
            self.update_ohlcv(symbol, ohlcv)
        
        data = self._ohlcv_cache.get(symbol, [])
        
        if not data or len(data) < 10:
            logger.warning(f"Insufficient data for {symbol} to calculate indicators")
            return IndicatorValues(symbol=symbol)
        
        state = self._inc_state.get(symbol)
        if state is not None and len(data) > state.get('min_len', 999999):
            return self._incremental_calculate(symbol, data, state)
        else:
            return self._full_calculate(symbol, data)
    
    def _full_calculate(self, symbol: str, data: List[List[float]]) -> IndicatorValues:
        """Full recalculation — saves running state for future incremental updates."""
        closes = np.array([c[4] for c in data])
        highs = np.array([c[2] for c in data])
        lows = np.array([c[3] for c in data])
        
        indicators = IndicatorValues(symbol=symbol)
        indicators.timestamp = datetime.utcnow()
        
        n = len(data)
        period = self.atr_period  # 14
        
        # --- EMAs (full calc, save last values) ---
        emas = calculate_all_emas(closes)
        self._ema_cache[symbol] = emas
        
        ema_vals = {}
        for name, arr in emas.items():
            v = arr[-1] if len(arr) > 0 and not np.isnan(arr[-1]) else None
            ema_vals[name] = v
        
        indicators.ema_9 = ema_vals['ema_9']
        indicators.ema_21 = ema_vals['ema_21']
        indicators.ema_50 = ema_vals['ema_50']
        indicators.ema_200 = ema_vals['ema_200']
        
        # --- RSI (full calc, save avg_gain/avg_loss) ---
        rsi_values = calculate_rsi(closes, period)
        indicators.rsi = rsi_values[-1] if len(rsi_values) > 0 and not np.isnan(rsi_values[-1]) else None
        indicators.rsi_period = self.rsi_period
        
        # Reconstruct avg_gain/avg_loss from the RSI calculation
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        # --- ATR (full calc, save last ATR) ---
        atr_values = calculate_atr(highs, lows, closes, period)
        indicators.atr = atr_values[-1] if len(atr_values) > 0 and not np.isnan(atr_values[-1]) else None
        indicators.atr_period = self.atr_period
        
        # --- ADX (full calc, save running state) ---
        adx_values = calculate_adx(highs, lows, closes, period)
        indicators.adx = adx_values[-1] if len(adx_values) > 0 and not np.isnan(adx_values[-1]) else None
        indicators.adx_period = self.atr_period
        
        # Reconstruct ADX running state
        adx_atr = None
        adx_plus_di = None
        adx_minus_di = None
        adx_val = None
        if n >= period * 2 + 1:
            # Recalculate TR, +DM, -DM
            tr = np.zeros(n)
            tr[0] = highs[0] - lows[0]
            plus_dm = np.zeros(n)
            minus_dm = np.zeros(n)
            for i in range(1, n):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i - 1])
                tr3 = abs(lows[i] - closes[i - 1])
                tr[i] = max(tr1, tr2, tr3)
                up = highs[i] - highs[i - 1]
                down = lows[i - 1] - lows[i]
                plus_dm[i] = up if (up > down and up > 0) else 0.0
                minus_dm[i] = down if (down > up and down > 0) else 0.0
            
            s_tr = np.mean(tr[:period])
            s_pdi = np.mean(plus_dm[:period])
            s_mdi = np.mean(minus_dm[:period])
            dx_arr = []
            if s_pdi + s_mdi > 0:
                dx_arr.append(100 * abs(s_pdi - s_mdi) / (s_pdi + s_mdi))
            else:
                dx_arr.append(0.0)
            
            for i in range(period, n):
                s_tr = (s_tr * (period - 1) + tr[i]) / period
                s_pdi = (s_pdi * (period - 1) + plus_dm[i]) / period
                s_mdi = (s_mdi * (period - 1) + minus_dm[i]) / period
                dx = 100 * abs(s_pdi - s_mdi) / (s_pdi + s_mdi) if (s_pdi + s_mdi) > 0 else 0.0
                dx_arr.append(dx)
            
            adx_atr = s_tr
            adx_plus_di = s_pdi
            adx_minus_di = s_mdi
            
            if len(dx_arr) >= period:
                adx_val = np.mean(dx_arr[:period])
                for i in range(period, len(dx_arr)):
                    adx_val = (adx_val * (period - 1) + dx_arr[i]) / period
        
        # --- Pivot Points ---
        pivot_data = calculate_pivot_points_from_ohlcv(highs, lows, closes, self.pivot_lookback)
        indicators.pivot = pivot_data['pivot'][-1] if len(pivot_data['pivot']) > 0 and not np.isnan(pivot_data['pivot'][-1]) else None
        indicators.r1 = pivot_data['r1'][-1] if len(pivot_data['r1']) > 0 and not np.isnan(pivot_data['r1'][-1]) else None
        indicators.s1 = pivot_data['s1'][-1] if len(pivot_data['s1']) > 0 and not np.isnan(pivot_data['s1'][-1]) else None
        indicators.r2 = pivot_data['r2'][-1] if len(pivot_data['r2']) > 0 and not np.isnan(pivot_data['r2'][-1]) else None
        indicators.s2 = pivot_data['s2'][-1] if len(pivot_data['s2']) > 0 and not np.isnan(pivot_data['s2'][-1]) else None
        
        # --- Trend ---
        if indicators.ema_50 is not None and indicators.ema_200 is not None:
            if indicators.ema_50 > indicators.ema_200:
                indicators.trend = 'uptrend'
            elif indicators.ema_50 < indicators.ema_200:
                indicators.trend = 'downtrend'
            else:
                indicators.trend = 'neutral'
        
        # --- Crossover ---
        if len(emas['ema_9']) >= 2 and len(emas['ema_21']) >= 2:
            crossover_type, crossover_idx = calculate_ema_crossover(emas['ema_9'], emas['ema_21'])
            indicators.last_crossover = crossover_type
            indicators.crossover_index = crossover_idx
        
        self._indicator_cache[symbol] = indicators
        
        # Save incremental state
        last_candle = data[-1]
        self._inc_state[symbol] = {
            'ema_9': indicators.ema_9,
            'ema_21': indicators.ema_21,
            'ema_50': indicators.ema_50,
            'ema_200': indicators.ema_200,
            'prev_ema_9': emas['ema_9'][-2] if len(emas['ema_9']) >= 2 and not np.isnan(emas['ema_9'][-2]) else None,
            'prev_ema_21': emas['ema_21'][-2] if len(emas['ema_21']) >= 2 and not np.isnan(emas['ema_21'][-2]) else None,
            'rsi_avg_gain': float(avg_gain),
            'rsi_avg_loss': float(avg_loss),
            'atr': indicators.atr,
            'adx': indicators.adx if indicators.adx is not None else adx_val,
            'adx_atr': float(adx_atr) if adx_atr is not None else None,
            'adx_plus_di': float(adx_plus_di) if adx_plus_di is not None else None,
            'adx_minus_di': float(adx_minus_di) if adx_minus_di is not None else None,
            'prev_close': float(last_candle[4]),
            'prev_high': float(last_candle[2]),
            'prev_low': float(last_candle[3]),
            'data_len': len(data),
            'min_len': max(200, self.atr_period * 2 + 1),
        }
        self._prev_len[symbol] = len(data)
        
        logger.debug(f"Full calc for {symbol}: EMA9={indicators.ema_9}, RSI={indicators.rsi}, ATR={indicators.atr}")
        
        return indicators
    
    def _incremental_calculate(self, symbol: str, data: List[List[float]], state: Dict[str, Any]) -> IndicatorValues:
        """O(1) incremental update using running state from previous candle."""
        candle = data[-1]
        close = candle[4]
        high = candle[2]
        low = candle[3]
        
        prev_close = state['prev_close']
        prev_high = state['prev_high']
        prev_low = state['prev_low']
        period = self.atr_period  # 14
        
        indicators = IndicatorValues(symbol=symbol)
        indicators.timestamp = datetime.utcnow()
        
        # --- EMAs: O(1) update ---
        prev_ema_9 = state['ema_9']
        prev_ema_21 = state['ema_21']
        new_ema_9 = None
        new_ema_21 = None
        new_ema_50 = None
        new_ema_200 = None
        
        m = self._EMA_MULTIPLIERS
        if state['ema_9'] is not None:
            new_ema_9 = close * m[9] + state['ema_9'] * (1 - m[9])
        if state['ema_21'] is not None:
            new_ema_21 = close * m[21] + state['ema_21'] * (1 - m[21])
        if state['ema_50'] is not None:
            new_ema_50 = close * m[50] + state['ema_50'] * (1 - m[50])
        if state['ema_200'] is not None:
            new_ema_200 = close * m[200] + state['ema_200'] * (1 - m[200])
        
        indicators.ema_9 = new_ema_9
        indicators.ema_21 = new_ema_21
        indicators.ema_50 = new_ema_50
        indicators.ema_200 = new_ema_200
        
        # --- RSI: O(1) update ---
        delta = close - prev_close
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        
        new_avg_gain = (state['rsi_avg_gain'] * (period - 1) + gain) / period
        new_avg_loss = (state['rsi_avg_loss'] * (period - 1) + loss) / period
        
        if new_avg_loss == 0:
            indicators.rsi = 100.0
        else:
            rs = new_avg_gain / new_avg_loss
            indicators.rsi = 100.0 - (100.0 / (1.0 + rs))
        indicators.rsi_period = self.rsi_period
        
        # --- ATR: O(1) update ---
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        tr = max(tr1, tr2, tr3)
        
        prev_atr = state['atr']
        if prev_atr is not None:
            new_atr = (prev_atr * (period - 1) + tr) / period
            indicators.atr = new_atr
        else:
            indicators.atr = None
            new_atr = None
        indicators.atr_period = self.atr_period
        
        # --- ADX: O(1) update ---
        new_adx = state['adx']
        new_adx_atr = state['adx_atr']
        new_adx_plus_di = state['adx_plus_di']
        new_adx_minus_di = state['adx_minus_di']
        
        if new_adx_atr is not None:
            up_move = high - prev_high
            down_move = prev_low - low
            plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
            minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
            
            new_adx_atr = (new_adx_atr * (period - 1) + tr) / period
            new_adx_plus_di = (new_adx_plus_di * (period - 1) + plus_dm) / period
            new_adx_minus_di = (new_adx_minus_di * (period - 1) + minus_dm) / period
            
            denom = new_adx_plus_di + new_adx_minus_di
            dx = 100 * abs(new_adx_plus_di - new_adx_minus_di) / denom if denom > 0 else 0.0
            
            if new_adx is not None:
                new_adx = (new_adx * (period - 1) + dx) / period
        
        indicators.adx = new_adx
        indicators.adx_period = self.atr_period
        
        # --- Pivot Points: O(lookback) from cached list ---
        lb = self.pivot_lookback
        n = len(data)
        if n >= lb:
            start = n - lb
            p_high = max(c[2] for c in data[start:])
            p_low = min(c[3] for c in data[start:])
            p_close = close
            pp = (p_high + p_low + p_close) / 3.0
            indicators.pivot = pp
            indicators.r1 = (2 * pp) - p_low
            indicators.s1 = (2 * pp) - p_high
            indicators.r2 = pp + (p_high - p_low)
            indicators.s2 = pp - (p_high - p_low)
        
        # --- Trend ---
        if indicators.ema_50 is not None and indicators.ema_200 is not None:
            if indicators.ema_50 > indicators.ema_200:
                indicators.trend = 'uptrend'
            elif indicators.ema_50 < indicators.ema_200:
                indicators.trend = 'downtrend'
            else:
                indicators.trend = 'neutral'
        
        # --- Crossover: O(1) from previous and current EMA values ---
        if prev_ema_9 is not None and prev_ema_21 is not None and new_ema_9 is not None and new_ema_21 is not None:
            if prev_ema_9 <= prev_ema_21 and new_ema_9 > new_ema_21:
                indicators.last_crossover = CrossoverType.BULLISH
                indicators.crossover_index = len(data) - 1
            elif prev_ema_9 >= prev_ema_21 and new_ema_9 < new_ema_21:
                indicators.last_crossover = CrossoverType.BEARISH
                indicators.crossover_index = len(data) - 1
            else:
                indicators.last_crossover = CrossoverType.NONE
                indicators.crossover_index = None
        
        self._indicator_cache[symbol] = indicators
        
        # Update incremental state
        state['prev_ema_9'] = state['ema_9']
        state['prev_ema_21'] = state['ema_21']
        state['ema_9'] = new_ema_9
        state['ema_21'] = new_ema_21
        state['ema_50'] = new_ema_50
        state['ema_200'] = new_ema_200
        state['rsi_avg_gain'] = new_avg_gain
        state['rsi_avg_loss'] = new_avg_loss
        state['atr'] = new_atr
        state['adx'] = new_adx
        state['adx_atr'] = new_adx_atr
        state['adx_plus_di'] = new_adx_plus_di
        state['adx_minus_di'] = new_adx_minus_di
        state['prev_close'] = close
        state['prev_high'] = high
        state['prev_low'] = low
        state['data_len'] = len(data)
        self._prev_len[symbol] = len(data)
        
        return indicators
    
    def get_latest(self, symbol: str) -> Optional[IndicatorValues]:
        return self._indicator_cache.get(symbol)
    
    def get_ema_arrays(self, symbol: str) -> Optional[Dict[str, np.ndarray]]:
        return self._ema_cache.get(symbol)
    
    def get_ohlcv(self, symbol: str) -> Optional[List[List[float]]]:
        return self._ohlcv_cache.get(symbol)
    
    def get_price_arrays(self, symbol: str) -> Optional[Dict[str, np.ndarray]]:
        data = self._ohlcv_cache.get(symbol)
        if not data:
            return None
        
        cur_len = len(data)
        cached_len = self._price_arrays_len.get(symbol, -1)
        if cached_len == cur_len and symbol in self._price_arrays_cache:
            return self._price_arrays_cache[symbol]
        
        arrays = {
            'timestamps': np.array([c[0] for c in data]),
            'opens': np.array([c[1] for c in data]),
            'highs': np.array([c[2] for c in data]),
            'lows': np.array([c[3] for c in data]),
            'closes': np.array([c[4] for c in data]),
            'volumes': np.array([c[5] for c in data])
        }
        self._price_arrays_cache[symbol] = arrays
        self._price_arrays_len[symbol] = cur_len
        return arrays
    
    def clear_cache(self, symbol: Optional[str] = None) -> None:
        if symbol:
            self._ohlcv_cache.pop(symbol, None)
            self._indicator_cache.pop(symbol, None)
            self._ema_cache.pop(symbol, None)
            self._inc_state.pop(symbol, None)
            self._prev_len.pop(symbol, None)
            self._price_arrays_cache.pop(symbol, None)
            self._price_arrays_len.pop(symbol, None)
            logger.info(f"Cleared cache for {symbol}")
        else:
            self._ohlcv_cache.clear()
            self._indicator_cache.clear()
            self._ema_cache.clear()
            self._inc_state.clear()
            self._prev_len.clear()
            self._price_arrays_cache.clear()
            self._price_arrays_len.clear()
            logger.info("Cleared all indicator caches")
    
    def get_all_symbols(self) -> List[str]:
        return list(self._ohlcv_cache.keys())
    
    def is_data_sufficient(self, symbol: str, min_candles: int = 200) -> bool:
        data = self._ohlcv_cache.get(symbol, [])
        return len(data) >= min_candles

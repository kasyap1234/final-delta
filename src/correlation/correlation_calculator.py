"""Correlation calculator for cryptocurrency pairs.

This module provides the CorrelationCalculator class for computing
Pearson correlation coefficients between cryptocurrency pairs,
with support for rolling window calculations and hedge asset selection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

import numpy as np

from .price_history import PriceHistory

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of a correlation calculation."""
    symbol1: str
    symbol2: str
    correlation: float
    window: int
    timestamp: datetime
    confidence: Optional[float] = None


class CorrelationCalculator:
    """Calculates correlations between cryptocurrency pairs.
    
    This class provides methods to calculate Pearson correlation coefficients
    between price series, maintain a correlation matrix, and identify
    optimal hedge assets based on correlation.
    
    Attributes:
        price_history: PriceHistory instance for accessing price data
        default_window: Default rolling window size (days)
        min_correlation_threshold: Minimum correlation for hedge selection
    """
    
    def __init__(
        self,
        price_history: PriceHistory,
        default_window: int = 30,
        min_correlation_threshold: float = 0.5
    ):
        """Initialize the correlation calculator.
        
        Args:
            price_history: PriceHistory instance for price data access
            default_window: Default rolling window in days
            min_correlation_threshold: Minimum correlation for hedge assets
        """
        self.price_history = price_history
        self.default_window = default_window
        self.min_correlation_threshold = min_correlation_threshold
        
        # Correlation matrix storage: {(sym1, sym2): CorrelationResult}
        self._correlation_matrix: Dict[Tuple[str, str], CorrelationResult] = {}
        
        # Cache for recent calculations
        self._cache: Dict[Tuple[str, str, int], Tuple[float, datetime]] = {}
        self._cache_ttl_seconds = 300  # 5 minutes
        
        logger.info(
            f"CorrelationCalculator initialized with window={default_window}, "
            f"min_threshold={min_correlation_threshold}"
        )
    
    def calculate_correlation(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray
    ) -> float:
        """Calculate Pearson correlation coefficient between two price series.
        
        Formula: covariance(X, Y) / (std_dev(X) * std_dev(Y))
        
        Args:
            prices1: First price series array
            prices2: Second price series array
            
        Returns:
            Pearson correlation coefficient (-1 to 1)
            
        Raises:
            ValueError: If arrays have different lengths or insufficient data
        """
        if len(prices1) != len(prices2):
            raise ValueError(
                f"Price series must have same length: "
                f"{len(prices1)} vs {len(prices2)}"
            )
        
        if len(prices1) < 2:
            raise ValueError("Need at least 2 data points for correlation")
        
        # Remove any NaN values
        mask = ~(np.isnan(prices1) | np.isnan(prices2))
        clean_p1 = prices1[mask]
        clean_p2 = prices2[mask]
        
        if len(clean_p1) < 2:
            raise ValueError("Insufficient valid data points after removing NaN")
        
        # Calculate means
        mean1 = np.mean(clean_p1)
        mean2 = np.mean(clean_p2)
        
        # Calculate deviations
        dev1 = clean_p1 - mean1
        dev2 = clean_p2 - mean2
        
        # Calculate covariance and standard deviations
        covariance = np.sum(dev1 * dev2) / (len(clean_p1) - 1)
        std1 = np.sqrt(np.sum(dev1 ** 2) / (len(clean_p1) - 1))
        std2 = np.sqrt(np.sum(dev2 ** 2) / (len(clean_p2) - 1))
        
        # Handle zero standard deviation
        if std1 == 0 or std2 == 0:
            logger.warning("Zero standard deviation in price series, returning 0")
            return 0.0
        
        correlation = covariance / (std1 * std2)
        
        # Clamp to [-1, 1] to handle floating point errors
        return float(np.clip(correlation, -1.0, 1.0))
    
    def calculate_rolling_correlation(
        self,
        prices1: np.ndarray,
        prices2: np.ndarray,
        window: int
    ) -> np.ndarray:
        """Calculate rolling correlation over a window.
        
        Args:
            prices1: First price series array
            prices2: Second price series array
            window: Rolling window size
            
        Returns:
            Array of rolling correlation values
            
        Raises:
            ValueError: If window is larger than data length
        """
        if window > len(prices1):
            raise ValueError(
                f"Window size {window} larger than data length {len(prices1)}"
            )
        
        if window < 2:
            raise ValueError("Window must be at least 2")
        
        correlations = []
        
        for i in range(window, len(prices1) + 1):
            window_p1 = prices1[i - window:i]
            window_p2 = prices2[i - window:i]
            
            try:
                corr = self.calculate_correlation(window_p1, window_p2)
                correlations.append(corr)
            except ValueError as e:
                logger.warning(f"Could not calculate correlation at index {i}: {e}")
                correlations.append(np.nan)
        
        return np.array(correlations)
    
    def update_correlation_matrix(
        self,
        symbols: List[str],
        price_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[Tuple[str, str], CorrelationResult]:
        """Update correlation matrix for all symbol pairs.
        
        Args:
            symbols: List of symbols to calculate correlations for
            price_data: Optional pre-fetched price data (symbol -> prices array)
            
        Returns:
            Updated correlation matrix dictionary
        """
        if len(symbols) < 2:
            logger.warning("Need at least 2 symbols for correlation matrix")
            return self._correlation_matrix
        
        timestamp = datetime.utcnow()
        new_matrix = {}
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1:]:
                try:
                    # Get price data
                    if price_data and sym1 in price_data and sym2 in price_data:
                        prices1 = price_data[sym1]
                        prices2 = price_data[sym2]
                        
                        # Align lengths if necessary
                        min_len = min(len(prices1), len(prices2))
                        prices1 = prices1[-min_len:]
                        prices2 = prices2[-min_len:]
                    else:
                        # Fetch from price history with alignment
                        _, prices1, prices2 = self.price_history.get_aligned_price_series(
                            sym1, sym2, self.default_window
                        )
                    
                    # Calculate correlation
                    correlation = self.calculate_correlation(prices1, prices2)
                    
                    # Store result
                    result = CorrelationResult(
                        symbol1=sym1,
                        symbol2=sym2,
                        correlation=correlation,
                        window=len(prices1),
                        timestamp=timestamp
                    )
                    
                    new_matrix[(sym1, sym2)] = result
                    new_matrix[(sym2, sym1)] = CorrelationResult(
                        symbol1=sym2,
                        symbol2=sym1,
                        correlation=correlation,
                        window=len(prices1),
                        timestamp=timestamp
                    )
                    
                except (KeyError, ValueError) as e:
                    logger.warning(
                        f"Could not calculate correlation for {sym1}-{sym2}: {e}"
                    )
                    continue
        
        self._correlation_matrix = new_matrix
        
        logger.info(
            f"Updated correlation matrix with {len(symbols)} symbols, "
            f"{len(new_matrix) // 2} unique pairs"
        )
        
        return self._correlation_matrix
    
    def get_correlation(
        self,
        symbol1: str,
        symbol2: str,
        use_cache: bool = True
    ) -> Optional[float]:
        """Get correlation between two symbols.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            use_cache: Whether to use cached value if available
            
        Returns:
            Correlation coefficient or None if not available
        """
        if symbol1 == symbol2:
            return 1.0
        
        cache_key = (symbol1, symbol2, self.default_window)
        
        # Check cache
        if use_cache and cache_key in self._cache:
            corr, cached_time = self._cache[cache_key]
            age = (datetime.utcnow() - cached_time).total_seconds()
            if age < self._cache_ttl_seconds:
                return corr
        
        # Check matrix
        if (symbol1, symbol2) in self._correlation_matrix:
            corr = self._correlation_matrix[(symbol1, symbol2)].correlation
            self._cache[cache_key] = (corr, datetime.utcnow())
            return corr
        
        # Try to calculate on the fly
        try:
            _, prices1, prices2 = self.price_history.get_aligned_price_series(
                symbol1, symbol2, self.default_window
            )
            corr = self.calculate_correlation(prices1, prices2)
            self._cache[cache_key] = (corr, datetime.utcnow())
            return corr
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not get correlation for {symbol1}-{symbol2}: {e}")
            return None
    
    def get_most_correlated(
        self,
        symbol: str,
        candidates: Optional[List[str]] = None,
        min_correlation: Optional[float] = None,
        top_n: int = 1
    ) -> List[Dict[str, Any]]:
        """Find the most correlated assets for hedge selection.
        
        Args:
            symbol: The base symbol to find correlations for
            candidates: List of candidate symbols (None = all available)
            min_correlation: Minimum correlation threshold
            top_n: Number of top results to return
            
        Returns:
            List of dictionaries with 'symbol' and 'correlation' keys,
            sorted by correlation (highest first)
        """
        min_corr = min_correlation or self.min_correlation_threshold
        
        # Get candidates
        if candidates is None:
            candidates = self.price_history.get_available_symbols()
        
        candidates = [c for c in candidates if c != symbol]
        
        if not candidates:
            logger.warning(f"No candidate symbols available for {symbol}")
            return []
        
        # Calculate correlations
        correlations = []
        for candidate in candidates:
            corr = self.get_correlation(symbol, candidate)
            if corr is not None and abs(corr) >= min_corr:
                correlations.append({
                    'symbol': candidate,
                    'correlation': corr,
                    'absolute_correlation': abs(corr)
                })
        
        # Sort by absolute correlation (descending)
        correlations.sort(key=lambda x: x['absolute_correlation'], reverse=True)
        
        # Return top N without the absolute_correlation field
        results = [
            {'symbol': c['symbol'], 'correlation': c['correlation']}
            for c in correlations[:top_n]
        ]
        
        return results
    
    def get_best_hedge_asset(
        self,
        symbol: str,
        candidates: Optional[List[str]] = None,
        min_correlation: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Find the best hedge asset for a given symbol.
        
        This returns the single most correlated asset.
        
        Args:
            symbol: The base symbol
            candidates: List of candidate symbols
            min_correlation: Minimum correlation threshold
            
        Returns:
            Dictionary with 'symbol' and 'correlation' or None
        """
        results = self.get_most_correlated(symbol, candidates, min_correlation, top_n=1)
        return results[0] if results else None
    
    def get_correlation_matrix(
        self,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Get the full correlation matrix as a nested dictionary.
        
        Args:
            symbols: Optional list of symbols to include (None = all)
            
        Returns:
            Nested dict: {symbol1: {symbol2: correlation, ...}, ...}
        """
        if symbols is None:
            symbols = self.price_history.get_available_symbols()
        
        matrix = defaultdict(dict)
        
        for sym1 in symbols:
            matrix[sym1][sym1] = 1.0
            for sym2 in symbols:
                if sym1 != sym2:
                    corr = self.get_correlation(sym1, sym2)
                    if corr is not None:
                        matrix[sym1][sym2] = corr
        
        return dict(matrix)
    
    def get_correlation_stats(
        self,
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Get correlation statistics for a symbol.
        
        Args:
            symbol: The symbol to analyze
            
        Returns:
            Dictionary with correlation statistics
        """
        correlations = []
        
        for (sym1, sym2), result in self._correlation_matrix.items():
            if sym1 == symbol:
                correlations.append(result.correlation)
        
        if not correlations:
            return None
        
        correlations = np.array(correlations)
        
        return {
            'symbol': symbol,
            'num_pairs': len(correlations),
            'mean_correlation': float(np.mean(correlations)),
            'median_correlation': float(np.median(correlations)),
            'min_correlation': float(np.min(correlations)),
            'max_correlation': float(np.max(correlations)),
            'std_correlation': float(np.std(correlations))
        }
    
    def clear_cache(self) -> None:
        """Clear the correlation cache."""
        self._cache.clear()
        logger.debug("Cleared correlation cache")
    
    def invalidate_symbol(self, symbol: str) -> None:
        """Invalidate all cached correlations for a symbol.
        
        Args:
            symbol: The symbol to invalidate
        """
        # Remove from matrix
        keys_to_remove = [
            key for key in self._correlation_matrix.keys()
            if symbol in key
        ]
        for key in keys_to_remove:
            del self._correlation_matrix[key]
        
        # Remove from cache
        cache_keys_to_remove = [
            key for key in self._cache.keys()
            if symbol in key[:2]
        ]
        for key in cache_keys_to_remove:
            del self._cache[key]
        
        logger.debug(f"Invalidated correlations for {symbol}")

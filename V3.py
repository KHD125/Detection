"""
Wave Detection Ultimate 3.0 - FINAL ENHANCED PRODUCTION VERSION
==================================================================
Professional Stock Ranking System with Advanced Market State Analytics
All bugs fixed, optimized for Streamlit Community Cloud
Enhanced with intelligent market regime awareness and adaptive scoring

Version: 3.1.0-PROFESSIONAL
Last Updated: September 2025
Status: PRODUCTION READY - Market State Integration Complete

MARKET STATE SYSTEM:
- Intelligent regime detection (8 states: STRONG_UPTREND, UPTREND, PULLBACK, etc.)
- Dynamic component score adjustments based on market conditions
- Smart bonuses for exceptional market state patterns
- Comprehensive error handling and logging
"""

# ============================================
# STREAMLIT CONFIGURATION - Prevent File Watcher Issues
# ============================================
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

# ============================================
# IMPORTS AND SETUP
# ============================================

# Standard library imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union  # Add Union back for type hints
from dataclasses import dataclass, field
from functools import wraps  # Remove lru_cache
import time
from io import BytesIO
import warnings
import gc
import re

# Suppress warnings for clean production output.
warnings.filterwarnings('ignore')

# Set NumPy to ignore floating point errors for robust calculations.
np.seterr(all='ignore')

# Set random seed for reproducibility of any random-based operations.
np.random.seed(42)

# ============================================
# SAFE DIVISION UTILITIES - CRITICAL BUG FIXES
# ============================================

def safe_divide(numerator: Union[float, int, np.ndarray, pd.Series],
                denominator: Union[float, int, np.ndarray, pd.Series],
                default: Union[float, int] = 0,
                handle_inf: bool = True,
                warn_on_zero: bool = False) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely perform division operations, handling division by zero and other edge cases.

    Parameters:
    -----------
    numerator : float, int, np.ndarray, or pd.Series
        The numerator value(s)
    denominator : float, int, np.ndarray, or pd.Series
        The denominator value(s)
    default : float or int, default=0
        Value to return when division by zero occurs
    handle_inf : bool, default=True
        Whether to replace infinity values with default
    warn_on_zero : bool, default=False
        Whether to emit warnings when division by zero occurs

    Returns:
    --------
    float, np.ndarray, or pd.Series
        Result of safe division operation
    """
    # Handle pandas Series and numpy arrays
    if isinstance(numerator, (pd.Series, np.ndarray)) or isinstance(denominator, (pd.Series, np.ndarray)):
        # Convert to numpy arrays for consistent handling
        num_array = np.asarray(numerator)
        den_array = np.asarray(denominator)

        # Create result array initialized with default values
        result = np.full_like(num_array, default, dtype=float)

        # Handle different cases
        if isinstance(denominator, (pd.Series, np.ndarray)):
            # Element-wise division with zero checking
            valid_mask = (den_array != 0) & pd.notna(den_array) & pd.notna(num_array)

            if warn_on_zero and np.any(den_array == 0):
                warnings.warn("Division by zero encountered in safe_divide", RuntimeWarning)

            # Perform division only for valid elements
            result[valid_mask] = num_array[valid_mask] / den_array[valid_mask]
        else:
            # Scalar denominator
            if denominator != 0 and pd.notna(denominator):
                valid_mask = pd.notna(num_array)
                result[valid_mask] = num_array[valid_mask] / denominator
            elif warn_on_zero:
                warnings.warn("Division by zero encountered in safe_divide", RuntimeWarning)

        # Handle infinity values if requested
        if handle_inf:
            result = np.where(np.isinf(result), default, result)

        # Return same type as input
        if isinstance(numerator, pd.Series):
            return pd.Series(result, index=numerator.index)
        else:
            return result

    else:
        # Handle scalar values
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            if warn_on_zero and denominator == 0:
                warnings.warn("Division by zero encountered in safe_divide", RuntimeWarning)
            return default

        result = numerator / denominator

        # Handle infinity
        if handle_inf and np.isinf(result):
            return default

        return result


def safe_percentage(numerator: Union[float, int, np.ndarray, pd.Series],
                   denominator: Union[float, int, np.ndarray, pd.Series],
                   default: float = 0.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate percentage: (numerator / denominator) * 100
    """
    return safe_divide(numerator, denominator, default) * 100


def safe_ratio(numerator: Union[float, int, np.ndarray, pd.Series],
               denominator: Union[float, int, np.ndarray, pd.Series],
               default: float = 1.0) -> Union[float, np.ndarray, pd.Series]:
    """
    Safely calculate ratio, with default of 1.0 for ratios
    """
    return safe_divide(numerator, denominator, default)


def safe_normalize(values: Union[np.ndarray, pd.Series],
                  total: Union[float, np.ndarray, pd.Series],
                  default: float = 0.0) -> Union[np.ndarray, pd.Series]:
    """
    Safely normalize values by their sum or a total
    """
    return safe_divide(values, total, default)

# ============================================
# LOGGING CONFIGURATION
# ============================================

# Configure production-ready logging with a clear format.
log_level = logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION AND CONSTANTS
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration with validated weights and thresholds"""
    
    # Data source - Default configuration
    DEFAULT_SHEET_URL: str = ""
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings - Dynamic refresh
    CACHE_TTL: int = 3600  
    STALE_DATA_HOURS: int = 24
    
    # Master Score 3.0 weights (total = 100%)
    POSITION_WEIGHT: float = 0.28
    VOLUME_WEIGHT: float = 0.20
    MOMENTUM_WEIGHT: float = 0.22
    ACCELERATION_WEIGHT: float = 0.02
    BREAKOUT_WEIGHT: float = 0.20
    RVOL_WEIGHT: float = 0.08
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Critical columns (app fails without these)
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    # Important columns (degraded experience without) - ENHANCED with all return periods
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'category', 'sector', 'industry',
        'rvol', 'pe', 'eps_current', 'eps_change_pct',
        'sma_20d', 'sma_50d', 'sma_200d',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'from_low_pct', 'from_high_pct',
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # All percentage columns for consistent handling
    PERCENTAGE_COLUMNS: List[str] = field(default_factory=lambda: [
        'from_low_pct', 'from_high_pct',
        'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 
        'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y',
        'eps_change_pct'
    ])
    
    # Volume ratio columns
    VOLUME_RATIO_COLUMNS: List[str] = field(default_factory=lambda: [
        'vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d',
        'vol_ratio_1d_180d', 'vol_ratio_7d_180d', 'vol_ratio_30d_180d',
        'vol_ratio_90d_180d'
    ])
    
    # Pattern thresholds
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 85,        # Reduced from 90 - more realistic for category leaders
        "hidden_gem": 80,
        "acceleration": 85,
        "institutional": 75,
        "vol_explosion": 95,          # Keep high - explosive volume should be rare
        "market_leader": 90,          # Reduced from 95 - top 10% is more realistic
        "momentum_wave": 75,
        "liquid_leader": 80,
        "long_strength": 80,
        "52w_high_approach": 85,      # Reduced from 90 - more opportunities near highs
        "52w_low_bounce": 85,
        "golden_zone": 85,
        "vol_accumulation": 80,
        "momentum_diverge": 85,       # Reduced from 90 - divergences are valuable signals
        "range_compress": 75,
        "stealth": 70,
        "perfect_storm": 80,
        "bull_trap": 85,              # Reduced from 90 - more trap detection
        "capitulation": 90,           # Reduced from 95 - still extreme but more realistic
        "runaway_gap": 85,
        "rotation_leader": 80,
        "distribution_top": 85,
        "velocity_squeeze": 85,
        "volume_divergence": 85,      # Reduced from 90 - important early warning signal
        "golden_cross": 75,           # Reduced from 80 - classic technical pattern
        "exhaustion": 85,             # Reduced from 90 - exhaustion patterns are valuable
        "pyramid": 75,
        "vacuum": 85,
    })
    
    # Market State Filtering Configuration
    MARKET_STATE_FILTERS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        'MOMENTUM': {
            'allowed_states': ['STRONG_UPTREND', 'UPTREND', 'PULLBACK'],
            'description': 'Focus on stocks in uptrends and pullbacks - optimal for swing trading'
        },
        'AGGRESSIVE': {
            'allowed_states': ['STRONG_UPTREND'],
            'description': 'Only the strongest uptrending stocks - highest risk/reward'
        },
        'VALUE': {
            'allowed_states': ['PULLBACK', 'BOUNCE', 'SIDEWAYS'],
            'description': 'Stocks in correction or consolidation - value opportunities'
        },
        'DEFENSIVE': {
            'allowed_states': ['STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'SIDEWAYS', 'BOUNCE'],
            'description': 'Avoid strong downtrends - conservative risk management'
        },
        'ALL': {
            'allowed_states': ['STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'ROTATION', 
                             'SIDEWAYS', 'DOWNTREND', 'STRONG_DOWNTREND', 'BOUNCE'],
            'description': 'No filtering - all market states included'
        }
    })
    
    # Default filter for swing/momentum trading
    DEFAULT_MARKET_FILTER: str = 'MOMENTUM'
    ENABLE_MARKET_STATE_FILTER: bool = True  # Can be toggled
    
    # Value bounds for data validation
    VALUE_BOUNDS: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'price': (0.01, 1_000_000),
        'rvol': (0.01, 1_000_000.0),
        'pe': (-10000, 10000),
        'returns': (-99.99, 9999.99),
        'volume': (0, 1e12)
    })
    
    # Performance thresholds
    PERFORMANCE_TARGETS: Dict[str, float] = field(default_factory=lambda: {
        'data_processing': 2.0,
        'filtering': 0.2,
        'pattern_detection': 0.5,
        'export_generation': 1.0,
        'search': 0.05
    })
    
    # Market categories (Indian market specific)
    MARKET_CATEGORIES: List[str] = field(default_factory=lambda: [
        'Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap'
    ])
    
    # Tier definitions with proper boundaries
    TIERS: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "eps": {
            "Loss": (-float('inf'), 0),
            "0-5": (0, 5),
            "5-10": (5, 10),
            "10-20": (10, 20),
            "20-50": (20, 50),
            "50-100": (50, 100),
            "100+": (100, float('inf'))
        },
        "pe": {
            "Negative/NA": (-float('inf'), 0),
            "0-10": (0, 10),
            "10-15": (10, 15),
            "15-20": (15, 20),
            "20-30": (20, 30),
            "30-50": (30, 50),
            "50+": (50, float('inf'))
        },
        "price": {
            "0-100": (0, 100),
            "100-250": (100, 250),
            "250-500": (250, 500),
            "500-1000": (500, 1000),
            "1000-2500": (1000, 2500),
            "2500-5000": (2500, 5000),
            "5000+": (5000, float('inf'))
        },
        "eps_change_pct": {
            "Heavy Loss (< -50%)": (-float('inf'), -50),
            "Loss (-50% to -10%)": (-50, -10),
            "Slight Decline (-10% to 0%)": (-10, 0),
            "Low Growth (0% to 20%)": (0, 20),
            "Good Growth (20% to 50%)": (20, 50),
            "High Growth (50% to 100%)": (50, 100),
            "Explosive Growth (> 100%)": (100, float('inf'))
        },
        "position_tiers": {
            "💎 Near Lows (0-20%)": (0, 20),
            "🏗️ Lower Range (20-40%)": (20, 40),
            "🏞️ Middle Range (40-60%)": (40, 60),
            "⛰️ Upper Range (60-80%)": (60, 80),
            "🏔️ Near Highs (80-100%)": (80, 100)
        },
        "performance_tiers": {
            # Short-term momentum (Practical thresholds for Indian markets)
            "🚀 Strong Gainers (>3% 1D)": ("ret_1d", 3),
            "⚡ Power Moves (>7% 1D)": ("ret_1d", 7),
            "💥 Explosive (>15% 1D)": ("ret_1d", 15),
            "🌟 3-Day Surge (>6% 3D)": ("ret_3d", 6),
            "📈 Weekly Winners (>12% 7D)": ("ret_7d", 12),
            
            # Medium-term growth (More realistic thresholds)
            "🏆 Monthly Champions (>25% 30D)": ("ret_30d", 25),
            "🎯 Quarterly Stars (>25% 3M)": ("ret_3m", 25),
            "💎 Half-Year Heroes (>35% 6M)": ("ret_6m", 35),
            
            # Long-term performance (More practical thresholds)
            "🌙 Annual Winners (>50% 1Y)": ("ret_1y", 50),
            "👑 Multi-Year Champions (>100% 3Y)": ("ret_3y", 100),
            "🏛️ Long-Term Legends (>150% 5Y)": ("ret_5y", 150)
        },
        "volume_tiers": {
            "📈 Growing Interest (RVOL >1.5x)": ("rvol", 1.5),
            "🔥 High Activity (RVOL >2x)": ("rvol", 2.0),
            "💥 Explosive Volume (RVOL >5x)": ("rvol", 5.0),
            "🌋 Volcanic Volume (RVOL >10x)": ("rvol", 10.0),
            "😴 Low Activity (RVOL <0.5x)": ("rvol", 0.5, "below")
        },
        "vmi_tiers": {
            "🌙 Hibernating (VMI <0.3)": ("vmi", 0.3, "below"),
            "😴 Sleepy (VMI 0.3-0.6)": ("vmi", 0.3, 0.6),
            "🚶 Walking (VMI 0.6-1.0)": ("vmi", 0.6, 1.0),
            "🏃 Running (VMI 1.0-1.5)": ("vmi", 1.0, 1.5),
            "🚀 Flying (VMI 1.5-2.5)": ("vmi", 1.5, 2.5),
            "🌋 Volcanic (VMI >2.5)": ("vmi", 2.5)
        },
        "momentum_harmony_tiers": {
            "💔 Broken (Score 0)": ("momentum_harmony", 0, 0),
            "🌧️ Conflicted (Score 1)": ("momentum_harmony", 1, 1),
            "⛅ Mixed (Score 2)": ("momentum_harmony", 2, 2),
            "🌤️ Aligned (Score 3)": ("momentum_harmony", 3, 3),
            "☀️ Perfect Harmony (Score 4)": ("momentum_harmony", 4, 4)
        }
    })
    
    # Metric Tooltips for better UX
    METRIC_TOOLTIPS: Dict[str, str] = field(default_factory=lambda: {
        'vmi': 'Volume Momentum Index: Weighted volume trend score (higher = stronger volume momentum)',
        'position_tension': 'Range position stress: Distance from 52W low + distance from 52W high',
        'momentum_harmony': 'Multi-timeframe alignment: 0-4 score showing consistency across periods',
        'overall_market_strength': 'Composite market score: Combined momentum, acceleration, RVOL & breakout',
        'market_state': 'Market momentum regime: STRONG_UPTREND, UPTREND, PULLBACK, SIDEWAYS, DOWNTREND, etc.',
        'money_flow_mm': 'Money Flow in millions: Price × Volume × RVOL / 1M',
        'master_score': 'Overall ranking score (0-100) combining all factors',
        'acceleration_score': 'Rate of momentum change (0-100)',
        'breakout_score': 'Probability of price breakout (0-100)',
        'trend_quality': 'SMA alignment quality (0-100)',
        'liquidity_score': 'Trading liquidity measure (0-100)',
        'from_high_pct': 'Distance from 52-week high: 0% = at high, negative values = below high',
        'from_low_pct': 'Distance from 52-week low: 0% = at low, positive values = above low',
        'vmi_tier': 'Volume Momentum Index tier: Weighted volume trend classification',
        'momentum_harmony_tier': 'Multi-timeframe momentum alignment tier: 0-4 consistency score'
    })

# Global configuration instance
CONFIG = Config()

# ============================================
# PERFORMANCE MONITORING
# ============================================

class PerformanceMonitor:
    """Track and report performance metrics"""
    
    @staticmethod
    def timer(target_time: Optional[float] = None):
        """Performance timing decorator with target comparison"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.perf_counter() - start
                    
                    # Log if exceeds target
                    if target_time and elapsed > target_time:
                        logger.warning(f"{func.__name__} took {elapsed:.2f}s (target: {target_time}s)")
                    elif elapsed > 1.0:
                        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
                    
                    # Store timing
                    if 'performance_metrics' not in st.session_state:
                        st.session_state.performance_metrics = {}
                    st.session_state.performance_metrics[func.__name__] = elapsed
                    
                    return result
                except Exception as e:
                    elapsed = time.perf_counter() - start
                    logger.error(f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}")
                    raise
            return wrapper
        return decorator

# ============================================
# DATA VALIDATION AND SANITIZATION
# ============================================

class DataValidator:
    """
    Comprehensive data validation and sanitization.
    This class ensures data integrity, handles missing or invalid values gracefully,
    and reports on all correction actions taken.
    """
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str], context: str) -> Tuple[bool, str]:
        """
        Validates the structure and basic quality of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to validate.
            required_cols (List[str]): A list of columns that must be present.
            context (str): A descriptive string for logging and error messages.

        Returns:
            Tuple[bool, str]: A boolean indicating validity and a message.
        """
        if df is None:
            return False, f"{context}: DataFrame is None"
        
        if df.empty:
            return False, f"{context}: DataFrame is empty"
        
        # Check for critical columns defined in CONFIG
        missing_critical = [col for col in CONFIG.CRITICAL_COLUMNS if col not in df.columns]
        if missing_critical:
            return False, f"{context}: Missing critical columns: {missing_critical}"
        
        # Check for duplicate tickers
        duplicates = df['ticker'].duplicated().sum()
        if duplicates > 0:
            logger.warning(f"{context}: Found {duplicates} duplicate tickers")
        
        # Calculate data completeness
        total_cells = len(df) * len(df.columns)
        filled_cells = df.notna().sum().sum()
        completeness = safe_percentage(filled_cells, total_cells, default=0.0)
        
        if completeness < 50:
            logger.warning(f"{context}: Low data completeness ({completeness:.1f}%)")
        
        # Update session state with data quality metrics
        if 'data_quality' not in st.session_state:
            st.session_state.data_quality = {}
        
        st.session_state.data_quality.update({
            'completeness': completeness,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'duplicate_tickers': duplicates,
            'context': context,
            'timestamp': datetime.now(timezone.utc)
        })
        
        logger.info(f"{context}: Validated {len(df)} rows, {len(df.columns)} columns, {completeness:.1f}% complete")
        return True, "Valid"

    @staticmethod
    def clean_numeric_value(value: Any, is_percentage: bool = False, bounds: Optional[Tuple[float, float]] = None) -> Optional[float]:
        """
        Cleans, converts, and validates a single numeric value.
        
        Args:
            value (Any): The value to clean.
            is_percentage (bool): Flag to handle percentage symbols.
            bounds (Optional[Tuple[float, float]]): A tuple (min, max) to clip the value.
            
        Returns:
            Optional[float]: The cleaned float value, or np.nan if invalid.
        """
        # ENHANCED INPUT VALIDATION
        if pd.isna(value) or value == '' or value is None:
            return np.nan
        
        try:
            # Convert to string for cleaning
            cleaned = str(value).strip()
            
            # COMPREHENSIVE invalid string detection
            invalid_strings = ['', '-', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', 'NIL', 
                             '#VALUE!', '#ERROR!', '#DIV/0!', '#N/A', '#REF!', '#NAME?',
                             'INF', '-INF', 'INFINITY', '-INFINITY', '∞', '-∞']
            if cleaned.upper() in invalid_strings:
                return np.nan
            
            # Remove symbols and spaces - ENHANCED CLEANING
            cleaned = cleaned.replace('₹', '').replace('$', '').replace(',', '').replace(' ', '').replace('%', '')
            
            # Convert to float
            result = float(cleaned)
            
            # Apply bounds if specified
            if bounds:
                min_val, max_val = bounds
                if result < min_val or result > max_val:
                    logger.debug(f"Value {result} outside bounds [{min_val}, {max_val}]")
                    result = np.clip(result, min_val, max_val)
            
            # Check for unreasonable values
            if np.isnan(result) or np.isinf(result):
                return np.nan
            
            return result
            
        except (ValueError, TypeError, AttributeError):
            return np.nan
    
    @staticmethod
    def sanitize_string(value: Any, default: str = "Unknown") -> str:
        """
        Cleans and sanitizes a string value, returning a default if invalid.
        
        Args:
            value (Any): The value to sanitize.
            default (str): The default value to return if invalid.
            
        Returns:
            str: The sanitized string.
        """
        if pd.isna(value) or value is None:
            return default
        
        cleaned = str(value).strip()
        if cleaned.upper() in ['', 'N/A', 'NA', 'NAN', 'NONE', 'NULL', '-']:
            return default
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
    @staticmethod
    def validate_numeric_columns(df: pd.DataFrame, columns: List[str]) -> Dict[str, int]:
        """
        Validates numeric columns and returns a count of invalid values per column.
        
        Args:
            df (pd.DataFrame): The DataFrame to validate.
            columns (List[str]): List of columns to validate.
            
        Returns:
            Dict[str, int]: Dictionary mapping column names to invalid value counts.
        """
        invalid_counts = {}
        
        for col in columns:
            if col in df.columns:
                # Count non-numeric values
                try:
                    pd.to_numeric(df[col], errors='coerce')
                    invalid_count = df[col].apply(
                        lambda x: not isinstance(x, (int, float, np.number)) and pd.notna(x)
                    ).sum()
                    
                    if invalid_count > 0:
                        invalid_counts[col] = invalid_count
                        logger.warning(f"Column '{col}' has {invalid_count} non-numeric values")
                except Exception as e:
                    logger.error(f"Error validating column '{col}': {str(e)}")
        
        return invalid_counts
        
# ============================================
# SMART CACHING WITH VERSIONING
# ============================================

def extract_spreadsheet_id(url_or_id: str) -> str:
    """
    Extracts the spreadsheet ID from a Google Sheets URL or returns the ID if it's already in the correct format.

    Args:
        url_or_id (str): A Google Sheets URL or just the spreadsheet ID.

    Returns:
        str: The extracted spreadsheet ID, or an empty string if not found.
    """
    if not url_or_id:
        return ""
    
    # If it's already just an ID (no slashes), return it
    if '/' not in url_or_id:
        return url_or_id.strip()
    
    # Try to extract from URL using a regular expression
    pattern = r'/spreadsheets/d/([a-zA-Z0-9-_]+)'
    match = re.search(pattern, url_or_id)
    if match:
        return match.group(1)
    
    # If no match, return as is.
    return url_or_id.strip()

@st.cache_data(ttl=3600, show_spinner=False)  # 1 hour TTL to prevent stale data
def load_and_process_data(source_type: str = "sheet", file_data=None, 
                         sheet_id: str = None, gid: str = None,
                         data_version: str = "1.0") -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """
    Loads and processes data from a Google Sheet or CSV file with caching and versioning.

    Args:
        source_type (str): Specifies the data source, either "sheet" or "upload".
        file_data (Optional): The uploaded CSV file object if `source_type` is "upload".
        sheet_id (str): The Google Spreadsheet ID.
        gid (str): The Google Sheet tab ID.
        data_version (str): A unique key to bust the cache (e.g., hash of date + sheet ID).

    Returns:
        Tuple[pd.DataFrame, datetime, Dict[str, Any]]: A tuple containing the processed DataFrame,
        the processing timestamp, and metadata about the process.
    
    Raises:
        ValueError: If a valid Google Sheets ID is not provided.
        Exception: If data loading or processing fails.
    """
    
    start_time = time.perf_counter()
    metadata = {
        'source_type': source_type,
        'data_version': data_version,
        'processing_start': datetime.now(timezone.utc),
        'errors': [],
        'warnings': []
    }
    
    try:
        # Load data based on source
        if source_type == "upload" and file_data is not None:
            logger.info("Loading data from uploaded CSV")
            try:
                df = pd.read_csv(file_data, low_memory=False)
                metadata['source'] = "User Upload"
            except UnicodeDecodeError:
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        file_data.seek(0)
                        df = pd.read_csv(file_data, low_memory=False, encoding=encoding)
                        metadata['warnings'].append(f"Used {encoding} encoding")
                        break
                    except (UnicodeDecodeError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                        logger.warning(f"Failed to decode with {encoding}: {e}")
                        continue
                else:
                    raise ValueError("Unable to decode CSV file")
        else:
            # Use defaults if not provided
            if not sheet_id:
                raise ValueError("Please enter a Google Sheets ID")
            if not gid:
                gid = CONFIG.DEFAULT_GID
            
            # Construct CSV URL
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            
            logger.info(f"Loading data from Google Sheets ID: {sheet_id}")
            
            try:
                df = pd.read_csv(csv_url, low_memory=False)
                metadata['source'] = "Google Sheets"
            except Exception as e:
                logger.error(f"Failed to load from Google Sheets: {str(e)}")
                metadata['errors'].append(f"Sheet load error: {str(e)}")
                
                # Try to use cached data as fallback
                if 'last_good_data' in st.session_state:
                    logger.info("Using cached data as fallback")
                    df, timestamp, old_metadata = st.session_state.last_good_data
                    metadata['warnings'].append("Using cached data due to load failure")
                    metadata['cache_used'] = True
                    return df, timestamp, metadata
                raise
        
        # Validate loaded data
        is_valid, validation_msg = DataValidator.validate_dataframe(df, CONFIG.CRITICAL_COLUMNS, "Initial load")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Process the data
        df = DataProcessor.process_dataframe(df, metadata)
        
        # Calculate all scores and rankings
        df = RankingEngine.calculate_all_scores(df)
        
        # Corrected method call here
        df = PatternDetector.detect_all_patterns_optimized(df)
        
        # Add advanced metrics
        df = AdvancedMetrics.calculate_all_metrics(df)
        
        # Final validation
        is_valid, validation_msg = DataValidator.validate_dataframe(df, ['master_score', 'rank'], "Final processed")
        if not is_valid:
            raise ValueError(validation_msg)
        
        # Store as last good data - MEMORY OPTIMIZED
        # Only store essential columns to reduce memory footprint
        timestamp = datetime.now(timezone.utc)
        essential_cols = ['ticker', 'company_name', 'price', 'master_score', 'market_state'] + \
                        [col for col in df.columns if col.endswith(('_score', '_pct', 'ret_'))]
        essential_cols = [col for col in essential_cols if col in df.columns]
        st.session_state.last_good_data = (df[essential_cols].copy() if len(essential_cols) > 0 else df.copy(), 
                                         timestamp, metadata)
        
        # Record processing time
        processing_time = time.perf_counter() - start_time
        metadata['processing_time'] = processing_time
        metadata['processing_end'] = datetime.now(timezone.utc)
        
        logger.info(f"Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        # Periodic cleanup
        if 'last_cleanup' not in st.session_state:
            st.session_state.last_cleanup = datetime.now(timezone.utc)
        
        if (datetime.now(timezone.utc) - st.session_state.last_cleanup).total_seconds() > 300:
            gc.collect()
            st.session_state.last_cleanup = datetime.now(timezone.utc)
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Failed to load and process data: {str(e)}")
        metadata['errors'].append(str(e))
        raise
        
# ============================================
# DATA PROCESSING ENGINE
# ============================================

class DataProcessor:
    """
    Handles the entire data processing pipeline, from raw data ingestion to a clean,
    ready-for-analysis DataFrame. This class is optimized for performance and robustness.
    """
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def process_dataframe(df: pd.DataFrame, metadata: Dict[str, Any]) -> pd.DataFrame:
        """
        Main pipeline to validate, clean, and prepare the raw DataFrame.

        Args:
            df (pd.DataFrame): The raw DataFrame to be processed.
            metadata (Dict[str, Any]): A dictionary to log warnings and changes.

        Returns:
            pd.DataFrame: A clean, processed DataFrame ready for scoring.
        """
        df = df.copy()
        initial_count = len(df)
        
        # 1. Process numeric columns with intelligent cleaning
        numeric_cols = [col for col in df.columns if col not in ['ticker', 'company_name', 'category', 'sector', 'industry', 'year', 'market_cap']]
        
        for col in numeric_cols:
            if col in df.columns:
                is_pct = col in CONFIG.PERCENTAGE_COLUMNS
                
                # Dynamically determine bounds based on column name
                bounds = None
                if 'volume' in col.lower():
                    bounds = CONFIG.VALUE_BOUNDS['volume']
                elif col == 'rvol':
                    bounds = CONFIG.VALUE_BOUNDS['rvol']
                elif col == 'pe':
                    bounds = CONFIG.VALUE_BOUNDS['pe']
                elif is_pct:
                    bounds = CONFIG.VALUE_BOUNDS['returns']
                else:
                    bounds = CONFIG.VALUE_BOUNDS.get('price', None)
                
                # Apply vectorized cleaning
                df[col] = df[col].apply(lambda x: DataValidator.clean_numeric_value(x, is_pct, bounds))
        
        # 2. Process categorical columns with robust sanitization
        string_cols = ['ticker', 'company_name', 'category', 'sector', 'industry']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.sanitize_string)
        
        # 3. Handle volume ratios with safety
        for col in CONFIG.VOLUME_RATIO_COLUMNS:
            if col in df.columns:
                df[col] = (100 + df[col]) / 100
                df[col] = df[col].clip(0.01, 1000.0)
                df[col] = df[col].fillna(1.0)
        
        # 4. Critical data validation and removal of duplicates
        df = df.dropna(subset=['ticker', 'price'], how='any')
        df = df[df['price'] > 0.01]
        
        before_dedup = len(df)
        df = df.drop_duplicates(subset=['ticker'], keep='first')
        if before_dedup > len(df):
            metadata['warnings'].append(f"Removed {before_dedup - len(df)} duplicate tickers")
        
        # 5. Fill missing values and add tier classifications
        df = DataProcessor._fill_missing_values(df)
        df = DataProcessor._add_tier_classifications(df)
        
        # 6. Log final data quality metrics
        removed_count = initial_count - len(df)
        if removed_count > 0:
            metadata['warnings'].append(f"Removed {removed_count} invalid rows during processing.")
        
        logger.info(f"Processed {len(df)} valid stocks from {initial_count} initial rows.")
        
        return df

    @staticmethod
    def _fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values in key columns with sensible defaults.
        This is a final defensive step to ensure downstream calculations don't fail due to NaNs.
        """
        # Default for position metrics
        if 'from_low_pct' in df.columns:
            df['from_low_pct'] = df['from_low_pct'].fillna(0)
        
        if 'from_high_pct' in df.columns:
            # FIXED: Calculate proper from_high_pct - positive when above 52w high, negative when below
            if all(col in df.columns for col in ['price', 'high_52w']):
                # Recalculate from_high_pct properly: (current_price / 52w_high - 1) * 100
                df['from_high_pct'] = ((df['price'] / df['high_52w']) - 1) * 100
                # Fill any remaining NaN values (division by zero cases) with 0
                df['from_high_pct'] = df['from_high_pct'].fillna(0)
            else:
                # Fallback if no price/high_52w data
                df['from_high_pct'] = df['from_high_pct'].fillna(0)
        
        # Default for Relative Volume (RVOL)
        if 'rvol' in df.columns:
            df['rvol'] = df['rvol'].fillna(1.0)
        
        # Defaults for price returns
        return_cols = [col for col in df.columns if col.startswith('ret_')]
        for col in return_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Defaults for volume columns
        volume_cols = [col for col in df.columns if col.startswith('volume_')]
        for col in volume_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Defaults for categorical columns
        for col in ['category', 'sector', 'industry']:
            if col not in df.columns:
                df[col] = 'Unknown'
            else:
                df[col] = df[col].fillna('Unknown')
        
        return df
    
    @staticmethod
    def _add_tier_classifications(df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a classification tier to numerical data (e.g., price, PE)
        based on predefined ranges in the `Config` class.
        This is a bug-fixed and robust version of the logic from earlier files.
        """
        def classify_tier(value: float, tier_dict: Dict[str, Tuple[float, float]]) -> str:
            """Helper function to map a value to its tier."""
            if pd.isna(value):
                return "Unknown"
            
            for tier_name, (min_val, max_val) in tier_dict.items():
                if min_val < value <= max_val:
                    return tier_name
                if min_val == -float('inf') and value <= max_val:
                    return tier_name
                if max_val == float('inf') and value > min_val:
                    return tier_name
            
            return "Unknown"
        
        if 'eps_current' in df.columns:
            df['eps_tier'] = df['eps_current'].apply(lambda x: classify_tier(x, CONFIG.TIERS['eps']))
        
        if 'pe' in df.columns:
            df['pe_tier'] = df['pe'].apply(
                lambda x: "Negative/NA" if pd.isna(x) or x <= 0 else classify_tier(x, CONFIG.TIERS['pe'])
            )
        
        if 'price' in df.columns:
            df['price_tier'] = df['price'].apply(lambda x: classify_tier(x, CONFIG.TIERS['price']))
        
        if 'eps_change_pct' in df.columns:
            df['eps_change_tier'] = df['eps_change_pct'].apply(
                lambda x: "Unknown" if pd.isna(x) else classify_tier(x, CONFIG.TIERS['eps_change_pct'])
            )
        
        # Position tier classification (based on from_low_pct or calculate from price data)
        if 'from_low_pct' in df.columns:
            # Use existing from_low_pct column (already 0-100%)
            df['position_pct'] = df['from_low_pct'].apply(
                lambda x: min(100, max(0, x)) if pd.notna(x) else None
            )
            df['position_tier'] = df['position_pct'].apply(
                lambda x: "Unknown" if pd.isna(x) else classify_tier(x, CONFIG.TIERS['position_tiers'])
            )
            logger.info(f"Position tiers created from from_low_pct. Sample tiers: {df['position_tier'].value_counts().head()}")
        
        # Performance tier classifications - Unified approach
        # Enhanced performance tier classification with ALL return periods
        available_return_cols = [col for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'] if col in df.columns]
        if available_return_cols:
            # Convert to percentage for classification (if needed - most are already in %)
            for col in available_return_cols:
                if col in df.columns:
                    # Check if values are in decimal format (e.g., 0.05 for 5%)
                    sample_vals = df[col].dropna().head(100)
                    if len(sample_vals) > 0 and sample_vals.abs().max() < 1.0:
                        df[f'{col}_pct'] = df[col] * 100
                    else:
                        df[f'{col}_pct'] = df[col]  # Already in percentage
            
            # Enhanced performance tier classification with ALL timeframes
            def classify_performance(row):
                # Get all return values (handle both percentage and decimal formats)
                returns = {}
                for col in available_return_cols:
                    val = row.get(col, 0) if pd.notna(row.get(col)) else 0
                    # Use percentage version if available, otherwise raw value
                    pct_col = f'{col}_pct'
                    if pct_col in row:
                        returns[col] = row.get(pct_col, 0) if pd.notna(row.get(pct_col)) else 0
                    else:
                        returns[col] = val
                
                # Priority-based classification (explosive short-term first, then longer-term)
                
                # Explosive short-term moves (highest priority)
                if 'ret_1d' in returns and returns['ret_1d'] > 15:
                    return "💥 Explosive (>15% 1D)"
                elif 'ret_1d' in returns and returns['ret_1d'] > 7:
                    return "⚡ Power Moves (>7% 1D)"
                elif 'ret_1d' in returns and returns['ret_1d'] > 3:
                    return "🚀 Strong Gainers (>3% 1D)"
                
                # Short-term momentum
                elif 'ret_3d' in returns and returns['ret_3d'] > 6:
                    return "🌟 3-Day Surge (>6% 3D)"
                elif 'ret_7d' in returns and returns['ret_7d'] > 12:
                    return "📈 Weekly Winners (>12% 7D)"
                elif 'ret_30d' in returns and returns['ret_30d'] > 25:
                    return "🏆 Monthly Champions (>25% 30D)"
                
                # Medium-term performance
                elif 'ret_3m' in returns and returns['ret_3m'] > 25:
                    return "🎯 Quarterly Stars (>25% 3M)"
                elif 'ret_6m' in returns and returns['ret_6m'] > 35:
                    return "💎 Half-Year Heroes (>35% 6M)"
                
                # Long-term performance
                elif 'ret_1y' in returns and returns['ret_1y'] > 50:
                    return "🌙 Annual Winners (>50% 1Y)"
                elif 'ret_3y' in returns and returns['ret_3y'] > 100:
                    return "👑 Multi-Year Champions (>100% 3Y)"
                elif 'ret_5y' in returns and returns['ret_5y'] > 150:
                    return "🏛️ Long-Term Legends (>150% 5Y)"
                
                else:
                    return "Standard"
            
            df['performance_tier'] = df.apply(classify_performance, axis=1)
            logger.info(f"Enhanced performance tiers created with {len(available_return_cols)} timeframes. Sample tiers: {df['performance_tier'].value_counts().head()}")
            
        # Volume tier classification
        if 'rvol' in df.columns:
            def classify_volume(row):
                rvol = row.get('rvol', 1.0) if pd.notna(row.get('rvol')) else 1.0
                
                if rvol >= 10.0:
                    return "🌋 Volcanic Volume (RVOL >10x)"
                elif rvol >= 5.0:
                    return "💥 Explosive Volume (RVOL >5x)"
                elif rvol >= 2.0:
                    return "🔥 High Activity (RVOL >2x)"
                elif rvol >= 1.5:
                    return "📈 Growing Interest (RVOL >1.5x)"
                elif rvol < 0.5:
                    return "😴 Low Activity (RVOL <0.5x)"
                else:
                    return "Standard Volume"
            
            df['volume_tier'] = df.apply(classify_volume, axis=1)
            logger.info(f"Volume tiers created. Sample tiers: {df['volume_tier'].value_counts().head()}")
            
        elif 'position_tension' in df.columns:
            # Convert position_tension to percentage from 52W low (0-100%)
            df['position_pct'] = df['position_tension'].apply(
                lambda x: min(100, max(0, x * 100)) if pd.notna(x) else None
            )
            df['position_tier'] = df['position_pct'].apply(
                lambda x: "Unknown" if pd.isna(x) else classify_tier(x, CONFIG.TIERS['position_tiers'])
            )
        elif all(col in df.columns for col in ['price', 'low_52w', 'high_52w']):
            # Calculate position percentage from price data with division by zero protection
            range_52w = df['high_52w'] - df['low_52w']
            df['position_pct'] = np.where(
                range_52w > 0,
                safe_percentage(df['price'] - df['low_52w'], range_52w, default=50.0).clip(0, 100),
                50  # Default to middle position when high equals low
            )
            df['position_tier'] = df['position_pct'].apply(
                lambda x: "Unknown" if pd.isna(x) else classify_tier(x, CONFIG.TIERS['position_tiers'])
            )
        
        return df
        
# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """
    Calculates advanced metrics and indicators using a combination of price,
    volume, and algorithmically derived scores. Ensures robust calculation
    by handling potential missing data (NaNs) gracefully.
    """
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates a comprehensive set of advanced metrics for the DataFrame.
        All calculations are designed to be vectorized and handle missing data
        without raising errors.
        FIXED: Money flow overflow prevention added.
    
        Args:
            df (pd.DataFrame): The DataFrame with raw data and core scores.
    
        Returns:
            pd.DataFrame: The DataFrame with all calculated advanced metrics added.
        """
        if df.empty:
            return df
        
        # Money Flow (in millions) - FIXED WITH OVERFLOW PREVENTION
        if all(col in df.columns for col in ['price', 'volume_1d', 'rvol']):
            # Clip RVOL to prevent extreme multiplications - More realistic upper bound
            safe_rvol = df['rvol'].fillna(1.0).clip(0, 50)  # Reduced from 100 to 50
            
            # Calculate money flow with overflow prevention
            money_flow_raw = df['price'].fillna(0) * df['volume_1d'].fillna(0) * safe_rvol
            
            # Clip to prevent overflow (max 1 trillion)
            df['money_flow'] = np.clip(money_flow_raw, 0, 1e12)
            
            # Convert to millions for display
            df['money_flow_mm'] = df['money_flow'] / 1_000_000
            
            # Additional safety: clip the millions version too
            df['money_flow_mm'] = df['money_flow_mm'].clip(0, 1e6)  # Max 1 million millions
        else:
            df['money_flow_mm'] = pd.Series(np.nan, index=df.index)
        
        # Volume Momentum Index (VMI) - Already safe (dividing by constant 10)
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d']):
            df['vmi'] = (
                df['vol_ratio_1d_90d'].fillna(1.0) * 4 +
                df['vol_ratio_7d_90d'].fillna(1.0) * 3 +
                df['vol_ratio_30d_90d'].fillna(1.0) * 2 +
                df['vol_ratio_90d_180d'].fillna(1.0) * 1
            ) / 10
        else:
            df['vmi'] = pd.Series(np.nan, index=df.index)
        
        # Position Tension
        if all(col in df.columns for col in ['from_low_pct', 'from_high_pct']):
            # FIXED: NaN from_low_pct means at low (0%), NaN from_high_pct means at high (0%)
            df['position_tension'] = df['from_low_pct'].fillna(0) + abs(df['from_high_pct'].fillna(0))
        else:
            df['position_tension'] = pd.Series(np.nan, index=df.index)
        
        # Momentum Harmony
        df['momentum_harmony'] = pd.Series(0, index=df.index, dtype=int)
        
        if 'ret_1d' in df.columns:
            df['momentum_harmony'] += (df['ret_1d'].fillna(0) > 0).astype(int)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                # ENHANCED NaN HANDLING with explicit checks
                safe_ret_7d = df['ret_7d'].fillna(0)
                safe_ret_30d = df['ret_30d'].fillna(0)
                daily_ret_7d = safe_divide(safe_ret_7d, 7, default=0.0)
                daily_ret_7d = pd.Series(daily_ret_7d, index=df.index)
                daily_ret_30d = safe_divide(safe_ret_30d, 30, default=0.0)
                daily_ret_30d = pd.Series(daily_ret_30d, index=df.index)
            df['momentum_harmony'] += ((daily_ret_7d > daily_ret_30d)).astype(int)
        
        if all(col in df.columns for col in ['ret_30d', 'ret_3m']):
            with np.errstate(divide='ignore', invalid='ignore'):
                # ENHANCED NaN HANDLING with explicit checks
                safe_ret_30d = df['ret_30d'].fillna(0)
                safe_ret_3m = df['ret_3m'].fillna(0)
                daily_ret_30d_comp = safe_divide(safe_ret_30d, 30, default=0.0)
                daily_ret_30d_comp = pd.Series(daily_ret_30d_comp, index=df.index)
                daily_ret_3m_comp = safe_divide(safe_ret_3m, 90, default=0.0)
                daily_ret_3m_comp = pd.Series(daily_ret_3m_comp, index=df.index)
            df['momentum_harmony'] += ((daily_ret_30d_comp.fillna(-np.inf) > daily_ret_3m_comp.fillna(-np.inf))).astype(int)
        
        if 'ret_3m' in df.columns:
            df['momentum_harmony'] += (df['ret_3m'].fillna(0) > 0).astype(int)
        
        # Market State Analysis
        df['market_state'] = df.apply(AdvancedMetrics._get_market_state_from_row, axis=1)
    
        # Overall Market Strength - FOUNDATION FOCUS  
        # This measures "How solid is the setup?"
        if all(col in df.columns for col in ['position_score', 'volume_score']):
            if 'momentum_score' in df.columns:
                df['overall_market_strength'] = (
                    df['position_score'] * 0.40 +
                    df['volume_score'] * 0.35 +
                    df['momentum_score'] * 0.25
                )
            else:
                df['overall_market_strength'] = (
                    df['position_score'] * 0.55 +
                    df['volume_score'] * 0.45
                )
        else:
            df['overall_market_strength'] = np.nan
        
        # VMI and Momentum Harmony Tier Classifications
        def classify_advanced_tier(value: float, tier_config: tuple) -> str:
            """Enhanced tier classifier for VMI and Momentum Harmony with range support."""
            if pd.isna(value):
                return "Unknown"
            
            if len(tier_config) == 2:
                # Single threshold: ("column", threshold) - above threshold
                _, threshold = tier_config
                return "Above" if value > threshold else "Below"
            elif len(tier_config) == 3:
                if tier_config[2] == "below":
                    # Below threshold: ("column", threshold, "below")
                    _, threshold, _ = tier_config
                    return "Below" if value < threshold else "Above"
                else:
                    # Range: ("column", min_val, max_val)
                    _, min_val, max_val = tier_config
                    return "InRange" if min_val <= value <= max_val else "OutOfRange"
            
            return "Unknown"
        
        # VMI Tier Classification
        if 'vmi' in df.columns:
            def classify_vmi_tier(value: float) -> str:
                if pd.isna(value):
                    return "Unknown"
                if value < 0.3:
                    return "🌙 Hibernating (VMI <0.3)"
                elif 0.3 <= value < 0.6:
                    return "😴 Sleepy (VMI 0.3-0.6)"
                elif 0.6 <= value < 1.0:
                    return "🚶 Walking (VMI 0.6-1.0)"
                elif 1.0 <= value < 1.5:
                    return "🏃 Running (VMI 1.0-1.5)"
                elif 1.5 <= value < 2.5:
                    return "🚀 Flying (VMI 1.5-2.5)"
                else:  # value >= 2.5
                    return "🌋 Volcanic (VMI >2.5)"
            
            df['vmi_tier'] = df['vmi'].apply(classify_vmi_tier)
        else:
            df['vmi_tier'] = pd.Series("Unknown", index=df.index)
        
        # Momentum Harmony Tier Classification
        if 'momentum_harmony' in df.columns:
            def classify_momentum_harmony_tier(value: int) -> str:
                if pd.isna(value):
                    return "Unknown"
                harmony_map = {
                    0: "💔 Broken (Score 0)",
                    1: "🌧️ Conflicted (Score 1)", 
                    2: "⛅ Mixed (Score 2)",
                    3: "🌤️ Aligned (Score 3)",
                    4: "☀️ Perfect Harmony (Score 4)"
                }
                return harmony_map.get(int(value), "Unknown")
            
            df['momentum_harmony_tier'] = df['momentum_harmony'].apply(classify_momentum_harmony_tier)
        else:
            df['momentum_harmony_tier'] = pd.Series("Unknown", index=df.index)
        
        return df
    
    @staticmethod
    def _get_market_state_from_row(row: pd.Series) -> str:
        """
        Helper function to get market state from a pandas row.
        Converts row data to returns dict and gets market state.
        """
        returns_dict = {
            '1d': row.get('ret_1d', 0),
            '3d': row.get('ret_3d', 0),
            '7d': row.get('ret_7d', 0),
            '30d': row.get('ret_30d', 0),
            '3m': row.get('ret_3m', 0),
            '6m': row.get('ret_6m', 0)
        }
        
        market_state = AdvancedMetrics.get_market_state(returns_dict)
        return market_state['state']

    @staticmethod
    def get_market_state(returns_dict: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify current market momentum state based on multi-timeframe analysis.
        FIXED: Honest momentum regime detection, not fake Elliott Waves.
        
        Core Philosophy:
        - Multi-timeframe momentum alignment
        - Identify tradeable regimes
        - Clear, actionable states
        - No pretense of Elliott Wave complexity
        
        States:
        - STRONG_UPTREND: All timeframes aligned up
        - UPTREND: Generally up with minor pullbacks
        - PULLBACK: Correction in uptrend
        - ROTATION: Trend change in progress
        - SIDEWAYS: No clear direction
        - DOWNTREND: Generally down
        - STRONG_DOWNTREND: All timeframes aligned down
        - BOUNCE: Relief rally in downtrend
        
        Returns:
            Dictionary with state, strength, and trading implications
        """
        state = {
            'state': 'UNKNOWN',
            'strength': 0,
            'confidence': 0,
            'trend_alignment': 0,
            'momentum_score': 0,
            'volatility': 'NORMAL',
            'action_bias': 'NEUTRAL',
            'description': ''
        }
        
        # Extract returns with defaults
        ret_1d = returns_dict.get('1d', 0)
        ret_3d = returns_dict.get('3d', 0)
        ret_7d = returns_dict.get('7d', 0)
        ret_30d = returns_dict.get('30d', 0)
        ret_90d = returns_dict.get('3m', 0)
        ret_180d = returns_dict.get('6m', 0)
        
        # Calculate momentum at different scales
        # Daily-equivalent rates for comparison
        very_short = ret_1d  # 1-day momentum
        # ENHANCED DIVISION PROTECTION with safe division functions
        short = safe_divide(ret_7d, 7, default=0.0)  # Daily rate over week
        medium = safe_divide(ret_30d, 30, default=0.0)  # Daily rate over month
        long = safe_divide(ret_90d, 90, default=0.0)  # Daily rate over quarter
        
        # Count positive periods (direction consistency) - ENHANCED NULL HANDLING
        periods = [ret_1d, ret_3d, ret_7d, ret_30d, ret_90d, ret_180d]
        valid_periods = [r for r in periods if pd.notna(r)]
        positive_count = sum(1 for r in valid_periods if r > 0)
        negative_count = sum(1 for r in valid_periods if r < 0)
        
        # Trend alignment score (-100 to 100) - SAFE DIVISION IMPLEMENTATION
        period_count = len(valid_periods) if valid_periods else 1  # Prevent division by zero
        trend_alignment = safe_percentage(positive_count - negative_count, period_count, default=0.0)
        state['trend_alignment'] = trend_alignment
        
        # Overall momentum score
        # Weighted average with recent periods more important
        momentum_score = (
            very_short * 0.3 +
            short * 7 * 0.25 +  # Convert back to period return
            medium * 30 * 0.25 +
            long * 90 * 0.20
        ) / 100 * 50 + 50  # Normalize to 0-100
        
        state['momentum_score'] = np.clip(momentum_score, 0, 100)
        
        # Volatility assessment
        returns_std = np.std([very_short, short * 7, medium * 30])
        if returns_std > 15:
            state['volatility'] = 'HIGH'
        elif returns_std > 7:
            state['volatility'] = 'ELEVATED'
        elif returns_std < 3:
            state['volatility'] = 'LOW'
        else:
            state['volatility'] = 'NORMAL'
        
        # STATE CLASSIFICATION LOGIC
        
        # 1. STRONG UPTREND
        if (positive_count >= 5 and ret_30d > 15 and ret_7d > 3 and ret_1d > 0):
            state['state'] = 'STRONG_UPTREND'
            state['strength'] = min(100, 70 + ret_30d)
            state['confidence'] = 85
            state['action_bias'] = 'STRONG_BUY'
            state['description'] = 'All timeframes aligned bullish. Momentum accelerating.'
        
        # 2. UPTREND
        elif (positive_count >= 4 and ret_30d > 5):
            state['state'] = 'UPTREND'
            state['strength'] = min(100, 50 + ret_30d)
            state['confidence'] = 70
            state['action_bias'] = 'BUY'
            state['description'] = 'Established uptrend. Minor corrections are buying opportunities.'
        
        # 3. PULLBACK IN UPTREND
        elif (ret_30d > 10 and ret_90d > 15 and ret_7d < 0):
            state['state'] = 'PULLBACK'
            state['strength'] = 50 - abs(ret_7d)
            state['confidence'] = 60
            state['action_bias'] = 'BUY_DIP'
            state['description'] = 'Healthy pullback in uptrend. Watch for support.'
        
        # 4. STRONG DOWNTREND
        elif (negative_count >= 5 and ret_30d < -15 and ret_7d < -3 and ret_1d < 0):
            state['state'] = 'STRONG_DOWNTREND'
            state['strength'] = min(100, 70 + abs(ret_30d))
            state['confidence'] = 85
            state['action_bias'] = 'STRONG_SELL'
            state['description'] = 'All timeframes aligned bearish. Avoid catching falling knife.'
        
        # 5. DOWNTREND
        elif (negative_count >= 4 and ret_30d < -5):
            state['state'] = 'DOWNTREND'
            state['strength'] = min(100, 50 + abs(ret_30d))
            state['confidence'] = 70
            state['action_bias'] = 'SELL'
            state['description'] = 'Established downtrend. Rallies are selling opportunities.'
        
        # 6. BOUNCE IN DOWNTREND
        elif (ret_30d < -10 and ret_90d < -15 and ret_7d > 0):
            state['state'] = 'BOUNCE'
            state['strength'] = 50 - abs(ret_30d - ret_7d)
            state['confidence'] = 50
            state['action_bias'] = 'CAUTIOUS'
            state['description'] = 'Relief rally in downtrend. Could be dead cat bounce.'
        
        # 7. ROTATION/TRANSITION
        elif (abs(ret_30d - safe_divide(ret_90d, 3, default=0)) > 10 or 
              (positive_count == 3 and negative_count == 3)):
            state['state'] = 'ROTATION'
            state['strength'] = 50
            state['confidence'] = 40
            state['action_bias'] = 'WAIT'
            state['description'] = 'Trend transition in progress. Wait for clarity.'
        
        # 8. SIDEWAYS/RANGE-BOUND
        else:
            state['state'] = 'SIDEWAYS'
            state['strength'] = 30
            state['confidence'] = 50
            state['action_bias'] = 'NEUTRAL'
            state['description'] = 'No clear trend. Range-trading environment.'
        
        # ADDITIONAL CONTEXT FLAGS
        
        # Momentum divergence detection
        if ret_30d > 10 and ret_7d < -5:
            state['warning'] = 'NEGATIVE_DIVERGENCE'
        elif ret_30d < -10 and ret_7d > 5:
            state['warning'] = 'POSITIVE_DIVERGENCE'
        
        # Extreme conditions
        if ret_30d > 30:  # Reduced from 50% to 30%
            state['extreme'] = 'OVERBOUGHT_MONTHLY'
        elif ret_30d < -25:  # Adjusted from -30% to -25%
            state['extreme'] = 'OVERSOLD_MONTHLY'
        
        if ret_7d > 20:
            state['extreme_short'] = 'OVERBOUGHT_WEEKLY'
        elif ret_7d < -15:
            state['extreme_short'] = 'OVERSOLD_WEEKLY'
        
        # Acceleration/Deceleration
        if short > medium and medium > long:
            state['momentum_phase'] = 'ACCELERATING'
        elif short < medium and medium < long:
            state['momentum_phase'] = 'DECELERATING'
        else:
            state['momentum_phase'] = 'STEADY'
        
        # TRADING RECOMMENDATIONS
        
        recommendations = []
        
        if state['state'] == 'STRONG_UPTREND':
            recommendations.append('Hold longs, add on dips')
            recommendations.append('Use trailing stops')
        elif state['state'] == 'PULLBACK':
            recommendations.append('Look for support levels')
            recommendations.append('Prepare to buy')
        elif state['state'] == 'STRONG_DOWNTREND':
            recommendations.append('Avoid longs')
            recommendations.append('Consider shorts or stay out')
        elif state['state'] == 'BOUNCE':
            recommendations.append('Take quick profits')
            recommendations.append('Don\'t chase')
        elif state['state'] == 'SIDEWAYS':
            recommendations.append('Range trade')
            recommendations.append('Buy support, sell resistance')
        elif state['state'] == 'ROTATION':
            recommendations.append('Reduce position size')
            recommendations.append('Wait for trend confirmation')
        
        state['recommendations'] = recommendations
        
        # CONFIDENCE ADJUSTMENTS
        
        # Higher confidence if volatility is normal
        if state['volatility'] == 'LOW':
            state['confidence'] *= 1.1
        elif state['volatility'] == 'HIGH':
            state['confidence'] *= 0.8
        
        # Ensure confidence is 0-100
        state['confidence'] = np.clip(state['confidence'], 0, 100)
        
        # FINAL SCORING
        
        # Overall state score (0-100)
        # Combines state strength with confidence
        state['overall_score'] = (state['strength'] * 0.7 + state['confidence'] * 0.3)
        
        return state

    @staticmethod
    def get_market_regime_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze overall market regime from multiple stocks.
        
        Args:
            df: DataFrame with return columns
        
        Returns:
            Market regime summary
        """
        if df.empty or 'ret_30d' not in df.columns:
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        # Get valid stocks - ENHANCED VALIDATION
        valid_mask = df['ret_30d'].notna()
        if not valid_mask.any():
            return {'regime': 'UNKNOWN', 'confidence': 0, 'description': 'No valid data available'}
        
        valid_df = df[valid_mask]
        valid_count = len(valid_df)
        
        # Prevent division by zero
        if valid_count == 0:
            return {'regime': 'UNKNOWN', 'confidence': 0, 'description': 'No valid stocks found'}
        
        # Calculate market statistics
        median_return_30d = valid_df['ret_30d'].median()
        mean_return_30d = valid_df['ret_30d'].mean()
        
        # Percentage of stocks in different states - SAFE DIVISION IMPLEMENTATION
        strong_up = safe_percentage((valid_df['ret_30d'] > 20).sum(), valid_count, default=0.0)
        up = safe_percentage((valid_df['ret_30d'] > 5).sum(), valid_count, default=0.0)
        down = safe_percentage((valid_df['ret_30d'] < -5).sum(), valid_count, default=0.0)
        strong_down = safe_percentage((valid_df['ret_30d'] < -20).sum(), valid_count, default=0.0)
        
        # Determine regime
        regime = {
            'regime': 'UNKNOWN',
            'strength': 0,
            'breadth': up - down,
            'median_return': median_return_30d,
            'mean_return': mean_return_30d,
            'confidence': 0,
            'description': ''
        }
        
        # Classification logic
        if up > 70 and median_return_30d > 10:
            regime['regime'] = 'BULL_MARKET'
            regime['strength'] = min(100, up)
            regime['confidence'] = 85
            regime['description'] = f'{up:.0f}% of stocks in uptrend. Strong bull market.'
        
        elif up > 55 and median_return_30d > 5:
            regime['regime'] = 'MILD_BULL'
            regime['strength'] = 60 + (up - 55)
            regime['confidence'] = 70
            regime['description'] = f'{up:.0f}% of stocks up. Moderate bullish bias.'
        
        elif down > 70 and median_return_30d < -10:
            regime['regime'] = 'BEAR_MARKET'
            regime['strength'] = min(100, down)
            regime['confidence'] = 85
            regime['description'] = f'{down:.0f}% of stocks in downtrend. Bear market conditions.'
        
        elif down > 55 and median_return_30d < -5:
            regime['regime'] = 'MILD_BEAR'
            regime['strength'] = 60 + (down - 55)
            regime['confidence'] = 70
            regime['description'] = f'{down:.0f}% of stocks down. Moderate bearish bias.'
        
        elif abs(median_return_30d) < 5 and abs(up - down) < 20:
            regime['regime'] = 'SIDEWAYS'
            regime['strength'] = 50
            regime['confidence'] = 60
            regime['description'] = 'Market range-bound. No clear direction.'
        
        else:
            regime['regime'] = 'MIXED'
            regime['strength'] = 50
            regime['confidence'] = 40
            regime['description'] = 'Mixed signals. Market in transition.'
        
        # Add distribution details
        regime['distribution'] = {
            'strong_up_pct': strong_up,
            'up_pct': up,
            'down_pct': down,
            'strong_down_pct': strong_down,
            'sideways_pct': 100 - up - down
        }
        
        return regime
        
# ============================================
# RANKING ENGINE
# Only 3 critical improvements to your already perfect system
# ============================================

class RankingEngine:
    """
    Core ranking calculations using a multi-factor model.
    FIXED VERSION: Smoothed acceleration, better thresholds, distribution detection.
    Maintains all your original genius logic and weights.
    """

    @staticmethod
    @PerformanceMonitor.timer(target_time=0.5)
    def calculate_all_scores(df: pd.DataFrame) -> pd.DataFrame:
        """
        Master orchestration function for all score calculations.
        BULLETPROOF: Guaranteed to complete and return valid DataFrame.
        
        Architecture:
        - Modular score calculation with error recovery
        - Dynamic weight normalization (no fake values)
        - Comprehensive data quality tracking
        - Full market context analysis
        - Performance optimized with vectorization
        
        Guarantees:
        - Always returns DataFrame with all expected columns
        - Never crashes (full error handling)
        - Preserves original data integrity
        - Complete audit trail via logging
        """
        # CRITICAL: Work on copy to preserve original
        df = df.copy()
        
        if df.empty:
            logger.warning("Empty dataframe provided to calculate_all_scores")
            return df
        
        # Initialize tracking
        start_time = time.time()
        timing_breakdown = {}
        calculation_metadata = {
            'start_time': start_time,
            'initial_rows': len(df),
            'initial_columns': len(df.columns)
        }
        
        logger.info(f"="*60)
        logger.info(f"Starting comprehensive ranking for {len(df)} stocks")
        logger.info(f"="*60)
        
        # ============================================================
        # PHASE 1: COMPONENT SCORE CALCULATION
        # ============================================================
        component_start = time.time()
        
        # Define all score calculators with metadata
        score_definitions = {
            'primary': {
                'position_score': {
                    'func': RankingEngine._calculate_position_score,
                    'weight': getattr(CONFIG, 'POSITION_WEIGHT', 0.28),
                    'required': True
                },
                'volume_score': {
                    'func': RankingEngine._calculate_volume_score,
                    'weight': getattr(CONFIG, 'VOLUME_WEIGHT', 0.20),
                    'required': True
                },
                'momentum_score': {
                    'func': RankingEngine._calculate_momentum_score,
                    'weight': getattr(CONFIG, 'MOMENTUM_WEIGHT', 0.22),
                    'required': True
                },
                'acceleration_score': {
                    'func': RankingEngine._calculate_acceleration_score,
                    'weight': getattr(CONFIG, 'ACCELERATION_WEIGHT', 0.02),
                    'required': True
                },
                'breakout_score': {
                    'func': RankingEngine._calculate_breakout_score,
                    'weight': getattr(CONFIG, 'BREAKOUT_WEIGHT', 0.20),
                    'required': True
                },
                'rvol_score': {
                    'func': RankingEngine._calculate_rvol_score,
                    'weight': getattr(CONFIG, 'RVOL_WEIGHT', 0.08),
                    'required': True
                }
            },
            'auxiliary': {
                'trend_quality': {
                    'func': RankingEngine._calculate_trend_quality,
                    'required': False
                },
                'long_term_strength': {
                    'func': RankingEngine._calculate_long_term_strength,
                    'required': False
                },
                'liquidity_score': {
                    'func': RankingEngine._calculate_liquidity_score,
                    'required': False
                }
            }
        }
        
        # Calculate all scores with error handling
        calculation_results = {}
        
        for score_type, scores in score_definitions.items():
            for score_name, score_config in scores.items():
                try:
                    # Execute score calculation
                    result = score_config['func'](df)
                    
                    # Validate result
                    if result is None:
                        logger.warning(f"{score_name}: Returned None, creating NaN column")
                        df[score_name] = np.nan
                        calculation_results[score_name] = 'failed_none'
                        
                    elif not isinstance(result, pd.Series):
                        logger.warning(f"{score_name}: Invalid type {type(result)}, creating NaN column")
                        df[score_name] = np.nan
                        calculation_results[score_name] = 'failed_type'
                        
                    elif len(result) != len(df):
                        logger.warning(f"{score_name}: Length mismatch ({len(result)} vs {len(df)})")
                        df[score_name] = np.nan
                        calculation_results[score_name] = 'failed_length'
                        
                    else:
                        # Valid result
                        df[score_name] = result
                        valid_count = result.notna().sum()
                        calculation_results[score_name] = f'success_{valid_count}/{len(df)}'
                        
                except Exception as e:
                    logger.error(f"{score_name}: Exception - {str(e)[:100]}")
                    df[score_name] = np.nan
                    calculation_results[score_name] = f'exception_{type(e).__name__}'
        
        timing_breakdown['component_calculation'] = time.time() - component_start
        
        # ============================================================
        # PHASE 2: DATA QUALITY ASSESSMENT
        # ============================================================
        quality_start = time.time()
        
        # Extract primary scores and weights
        primary_score_names = list(score_definitions['primary'].keys())
        primary_weights = np.array([score_definitions['primary'][s]['weight'] for s in primary_score_names])
        
        # Ensure weights sum to 1 - SAFE DIVISION IMPLEMENTATION
        if primary_weights.sum() != 1.0:
            logger.warning(f"Weights sum to {primary_weights.sum():.3f}, normalizing...")
            default_weight = 1.0/len(primary_weights) if len(primary_weights) > 0 else 1.0
            primary_weights = safe_normalize(primary_weights, primary_weights.sum(), default=default_weight)
        
        # Calculate data availability matrix
        scores_matrix = df[primary_score_names].values
        data_available = ~np.isnan(scores_matrix)
        components_per_stock = data_available.sum(axis=1)
        
        # Data quality metrics
        df['components_available'] = components_per_stock
        df['data_completeness'] = (components_per_stock / len(primary_score_names)) * 100
        df['has_minimum_data'] = components_per_stock >= 4  # Minimum threshold
        
        # Quality categories
        df['data_quality_category'] = pd.cut(
            df['components_available'],
            bins=[-0.1, 2.5, 3.5, 4.5, 5.5, 6.1],
            labels=['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
        )
        
        timing_breakdown['quality_assessment'] = time.time() - quality_start
        
        # ============================================================
        # PHASE 3: MASTER SCORE CALCULATION
        # ============================================================
        calculation_start = time.time()
        
        # Constants
        MIN_REQUIRED_COMPONENTS = 4
        
        # Initialize master scores
        master_scores = np.full(len(df), np.nan, dtype=float)
        weights_used = np.full((len(df), len(primary_score_names)), np.nan, dtype=float)
        
        # Vectorized calculation for all stocks at once
        for idx in range(len(df)):
            if components_per_stock[idx] >= MIN_REQUIRED_COMPONENTS:
                # Get valid components for this stock
                valid_mask = data_available[idx]
                
                if valid_mask.any():
                    # Extract valid scores and weights
                    valid_scores = scores_matrix[idx][valid_mask]
                    valid_weights = primary_weights[valid_mask]
                    
                    # Normalize weights - SAFE DIVISION IMPLEMENTATION
                    if valid_weights.sum() > 0:
                        default_weight = 1.0/len(valid_weights) if len(valid_weights) > 0 else 1.0
                        normalized_weights = safe_normalize(valid_weights, valid_weights.sum(), default=default_weight)
                        # Calculate weighted average
                        master_scores[idx] = np.dot(valid_scores, normalized_weights)
                        # Store weights for transparency
                        weights_used[idx][valid_mask] = normalized_weights
        
        # Assign raw scores
        df['master_score_raw'] = master_scores
        
        # Calculate quality multiplier (no linear penalty, use curve)
        # 6/6 = 1.00, 5/6 = 0.88, 4/6 = 0.72
        MAX_COMPONENTS = 6  # Make configurable instead of hard-coded
        df['quality_multiplier'] = np.where(
            df['components_available'] < MIN_REQUIRED_COMPONENTS,
            np.nan,  # No score if insufficient data
            0.5 + 0.5 * safe_divide(df['components_available'], MAX_COMPONENTS, default=1.0) ** 1.5  # Exponential curve
        )
        
        # Apply quality adjustment
        df['master_score_before_bonus'] = df['master_score_raw'] * df['quality_multiplier']
        df['master_score'] = df['master_score_before_bonus'].clip(0, 100)
        
        timing_breakdown['master_calculation'] = time.time() - calculation_start
        
        # ============================================================
        # PHASE 3.5: MARKET REGIME ADJUSTMENTS
        # ============================================================
        regime_start = time.time()
        
        # Detect overall market regime and apply adjustments
        try:
            # Get market regime summary
            market_regime = AdvancedMetrics.get_market_regime_summary(df)
            logger.info(f"Market regime detected: {market_regime['regime']} "
                       f"(confidence: {market_regime['confidence']:.0f}%)")
            logger.info(f"Market regime description: {market_regime['description']}")
            
            # Apply regime-based score adjustments
            valid_scores = df['master_score'].notna()
            adjustments_applied = 0
            
            if valid_scores.any():
                # BULL MARKET: Boost momentum and breakout scores by 5%
                if market_regime['regime'] in ['BULL_MARKET', 'MILD_BULL']:
                    bull_boost_mask = valid_scores & (
                        (df['momentum_score'].fillna(0) > 50) | 
                        (df['breakout_score'].fillna(0) > 50)
                    )
                    if bull_boost_mask.any():
                        # Boost component scores as specified, not master score
                        if 'momentum_score' in df.columns:
                            df.loc[bull_boost_mask, 'momentum_score'] *= 1.05
                        if 'breakout_score' in df.columns:
                            df.loc[bull_boost_mask, 'breakout_score'] *= 1.05
                        # Clip to ensure scores stay within bounds
                        if 'momentum_score' in df.columns:
                            df['momentum_score'] = df['momentum_score'].clip(0, 100)
                        if 'breakout_score' in df.columns:
                            df['breakout_score'] = df['breakout_score'].clip(0, 100)
                        adjustments_applied += bull_boost_mask.sum()
                        logger.info(f"Applied bull market component score boost to {bull_boost_mask.sum()} stocks")
                
                # BEAR MARKET: Boost position scores for stocks near 52w lows by 10%
                elif market_regime['regime'] in ['BEAR_MARKET', 'MILD_BEAR']:
                    if 'from_low_pct' in df.columns and 'position_score' in df.columns:
                        # FIXED: Use 999 to exclude NaN values from boost (they're at the low)
                        bear_boost_mask = valid_scores & (df['from_low_pct'].fillna(999) < 20)
                        if bear_boost_mask.any():
                            # Boost position score as specified, not master score
                            df.loc[bear_boost_mask, 'position_score'] *= 1.10
                            df['position_score'] = df['position_score'].clip(0, 100)
                            adjustments_applied += bear_boost_mask.sum()
                            logger.info(f"Applied bear market position score boost to {bear_boost_mask.sum()} stocks")
                
                # SIDEWAYS: No adjustments (as specified)
                elif market_regime['regime'] == 'SIDEWAYS':
                    logger.info("Sideways market detected - no regime adjustments applied")
            
            # Store regime info for later use
            df.attrs['market_regime'] = market_regime
            
            if adjustments_applied > 0:
                logger.info(f"Market regime adjustments applied to {adjustments_applied} stocks")
                # Recalculate master score since component scores changed
                logger.info("Recalculating master scores after regime adjustments...")
                df = RankingEngine._recalculate_master_score_after_adjustments(df)
            else:
                logger.info("No market regime adjustments applied")
                
        except Exception as e:
            logger.error(f"Market regime detection failed: {e}")
            df.attrs['market_regime'] = {'regime': 'UNKNOWN', 'confidence': 0}
        
        # Ensure master score stays in bounds after regime adjustments
        df['master_score'] = df['master_score'].clip(0, 100)
        
        timing_breakdown['regime_adjustments'] = time.time() - regime_start
        
        # ============================================================
        # PHASE 3.6: MARKET STATE FILTERING
        # ============================================================
        filter_start = time.time()
        
        try:
            # Apply market state filtering based on trading strategy
            if getattr(CONFIG, 'ENABLE_MARKET_STATE_FILTER', True):
                filter_name = getattr(CONFIG, 'DEFAULT_MARKET_FILTER', 'MOMENTUM')
                logger.info(f"Applying {filter_name} market state filter...")
                df = RankingEngine.apply_market_state_filter(df, filter_name)
            else:
                logger.info("Market state filtering disabled - keeping all stocks")
                
        except Exception as e:
            logger.error(f"Market state filtering failed: {e}")
            logger.error("Continuing with unfiltered dataset")
        
        timing_breakdown['market_state_filtering'] = time.time() - filter_start
        
        # ============================================================
        # PHASE 4: SMART BONUSES
        # ============================================================
        bonus_start = time.time()
        
        try:
            df = RankingEngine._apply_smart_bonuses(df)
            df['bonuses_applied'] = True
        except Exception as e:
            logger.error(f"Smart bonuses failed: {e}")
            df['bonuses_applied'] = False
            df['bonus_points'] = 0
            df['bonus_reasons'] = ''
        
        # Ensure master score stays in bounds
        df['master_score'] = df['master_score'].clip(0, 100)
        
        timing_breakdown['bonuses'] = time.time() - bonus_start
        
        # ============================================================
        # PHASE 5: RANKING CALCULATIONS
        # ============================================================
        ranking_start = time.time()
        
        # Overall rankings
        valid_scores_exist = df['master_score'].notna().any()
        
        if valid_scores_exist:
            # Standard rankings
            df['rank'] = df['master_score'].rank(
                method='first', 
                ascending=False, 
                na_option='bottom'
            )
            
            df['percentile'] = df['master_score'].rank(
                method='average',
                ascending=True,
                pct=True,
                na_option='bottom'
            ) * 100
            
            # Decile groups
            df['decile'] = pd.qcut(
                df['master_score'].dropna(),
                q=10,
                labels=range(10, 0, -1),
                duplicates='drop'
            ).reindex(df.index)
            
        else:
            df['rank'] = np.nan
            df['percentile'] = 0
            df['decile'] = np.nan
        
        # Fill NaN ranks
        df['rank'] = df['rank'].fillna(len(df) + 1)
        df['percentile'] = df['percentile'].fillna(0)
        
        # Score grades
        if valid_scores_exist:
            df['score_grade'] = pd.cut(
                df['master_score'],
                bins=[0, 25, 40, 50, 60, 70, 80, 90, 100],
                labels=['F', 'E', 'D', 'C', 'B', 'A', 'AA', 'AAA']
            )
        else:
            df['score_grade'] = pd.Categorical([np.nan] * len(df))
        
        # Category rankings
        try:
            df = RankingEngine._calculate_category_ranks(df)
        except Exception as e:
            logger.error(f"Category ranking failed: {e}")
            df['category_rank'] = np.nan
            df['category_percentile'] = np.nan
        
        timing_breakdown['rankings'] = time.time() - ranking_start
        
        # ============================================================
        # PHASE 6: CONFIDENCE METRICS
        # ============================================================
        confidence_start = time.time()
        
        # Multi-factor confidence score
        df['confidence_score'] = np.where(
            df['has_minimum_data'],
            (
                df['data_completeness'] * 0.40 +  # Data availability
                np.minimum(df['liquidity_score'].fillna(0), 100) * 0.30 +  # FIXED: Use 0 for missing liquidity
                (100 - np.abs(df['percentile'] - 50)) * 0.30  # Rank stability
            ) / 100,
            0
        ).clip(0, 100)
        
        # Confidence categories
        df['confidence_level'] = pd.cut(
            df['confidence_score'],
            bins=[0, 30, 50, 70, 85, 100],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        timing_breakdown['confidence'] = time.time() - confidence_start
        
        # ============================================================
        # PHASE 7: MARKET CONTEXT
        # ============================================================
        context_start = time.time()
        
        # Calculate market statistics
        scored_stocks = df[df['master_score'].notna()]
        
        market_stats = {
            'total_stocks': len(df),
            'scored_stocks': len(scored_stocks),
            'coverage_pct': (len(scored_stocks) / len(df) * 100) if len(df) > 0 else 0
        }
        
        if len(scored_stocks) > 0:
            # Score statistics
            market_stats.update({
                'mean_score': scored_stocks['master_score'].mean(),
                'median_score': scored_stocks['master_score'].median(),
                'std_score': scored_stocks['master_score'].std(),
                'q1_score': scored_stocks['master_score'].quantile(0.25),
                'q3_score': scored_stocks['master_score'].quantile(0.75)
            })
            
            # Component statistics
            for component in primary_score_names:
                if component in df.columns:
                    market_stats[f'mean_{component}'] = df[component].mean()
            
            # Market regime
            momentum_avg = df['momentum_score'].mean() if 'momentum_score' in df.columns else np.nan
            if pd.notna(momentum_avg):
                if momentum_avg > 65:
                    market_regime = 'strong_bull'
                elif momentum_avg > 55:
                    market_regime = 'bull'
                elif momentum_avg > 45:
                    market_regime = 'neutral'
                elif momentum_avg > 35:
                    market_regime = 'bear'
                else:
                    market_regime = 'strong_bear'
            else:
                market_regime = 'unknown'
            
            # Z-scores
            df['z_score'] = safe_divide(
                df['master_score'] - market_stats['mean_score'],
                market_stats['std_score'],
                default=0.0
            ).clip(-3, 3)
                
        else:
            market_stats.update({
                'mean_score': np.nan,
                'median_score': np.nan,
                'std_score': np.nan,
                'q1_score': np.nan,
                'q3_score': np.nan
            })
            market_regime = 'no_data'
            df['z_score'] = np.nan
        
        df['market_regime'] = market_regime
        
        timing_breakdown['context'] = time.time() - context_start
        
        # ============================================================
        # PHASE 8: FINAL VALIDATION
        # ============================================================
        validation_start = time.time()
        
        # Ensure all expected columns exist
        required_columns = {
            # Scores
            **{name: np.nan for name in primary_score_names},
            **{name: np.nan for name in score_definitions['auxiliary'].keys()},
            # Master scores
            'master_score': np.nan,
            'master_score_raw': np.nan,
            'master_score_before_bonus': np.nan,
            'quality_multiplier': np.nan,
            # Rankings
            'rank': len(df) + 1,
            'percentile': 0,
            'decile': np.nan,
            'score_grade': np.nan,
            # Quality metrics
            'components_available': 0,
            'data_completeness': 0,
            'has_minimum_data': False,
            'data_quality_category': 'Unknown',
            # Confidence
            'confidence_score': 0,
            'confidence_level': 'Unknown',
            # Market context
            'market_regime': 'unknown',
            'z_score': 0,
            # Bonuses
            'bonus_points': 0,
            'bonus_reasons': '',
            'bonuses_applied': False
        }
        
        for col, default_value in required_columns.items():
            if col not in df.columns:
                logger.debug(f"Adding missing column: {col}")
                df[col] = default_value
        
        timing_breakdown['validation'] = time.time() - validation_start
        
        # ============================================================
        # FINAL REPORTING
        # ============================================================
        total_time = time.time() - start_time
        
        # Comprehensive logging
        logger.info("="*60)
        logger.info("RANKING CALCULATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total execution time: {total_time:.3f}s")
        logger.info(f"Stocks processed: {market_stats['scored_stocks']}/{market_stats['total_stocks']} "
                   f"({market_stats['coverage_pct']:.1f}% coverage)")
        
        if market_stats['scored_stocks'] > 0:
            logger.info(f"Score distribution: Mean={market_stats['mean_score']:.1f}, "
                       f"Median={market_stats['median_score']:.1f}, "
                       f"Std={market_stats['std_score']:.1f}")
            logger.info(f"Market regime: {market_regime.upper()}")
        
        # Performance breakdown
        logger.info("Performance breakdown:")
        for phase, duration in timing_breakdown.items():
            pct = (duration / total_time * 100) if total_time > 0 else 0
            status = "⚠️" if duration > 0.1 else "✓"
            logger.info(f"  {status} {phase}: {duration:.3f}s ({pct:.1f}%)")
        
        # Data quality report
        quality_dist = df['data_quality_category'].value_counts()
        if len(quality_dist) > 0:
            logger.info("Data quality distribution:")
            for category, count in quality_dist.items():
                logger.info(f"  {category}: {count} stocks")
        
        # Top performers
        if 'ticker' in df.columns and market_stats['scored_stocks'] >= 5:
            top_5 = df.nlargest(5, 'master_score')
            logger.info("Top 5 stocks:")
            for rank, (_, row) in enumerate(top_5.iterrows(), 1):
                logger.info(f"  {rank}. {row['ticker']}: {row['master_score']:.1f} "
                           f"({row['components_available']}/6 components)")
        
        # Warnings
        if total_time > 0.5:
            logger.warning(f"⚠️ Exceeded target time of 0.5s by {total_time - 0.5:.3f}s")
        
        if market_stats['coverage_pct'] < 50:
            logger.warning(f"⚠️ Low data coverage: {market_stats['coverage_pct']:.1f}%")
        
        # Component calculation summary
        logger.debug("Component calculation results:")
        for component, status in calculation_results.items():
            logger.debug(f"  {component}: {status}")
        
        logger.info("="*60)
        
        # GUARANTEE: Return valid DataFrame
        return df
        
    @staticmethod
    def _recalculate_master_score_after_adjustments(df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate master score after regime adjustments to component scores.
        This ensures the master score reflects the adjusted component scores.
        """
        # Get available score columns and their weights
        score_cols = {
            'position_score': getattr(CONFIG, 'POSITION_WEIGHT', 0.28),
            'volume_score': getattr(CONFIG, 'VOLUME_WEIGHT', 0.20),
            'momentum_score': getattr(CONFIG, 'MOMENTUM_WEIGHT', 0.22),
            'acceleration_score': getattr(CONFIG, 'ACCELERATION_WEIGHT', 0.02),
            'breakout_score': getattr(CONFIG, 'BREAKOUT_WEIGHT', 0.20),
            'rvol_score': getattr(CONFIG, 'RVOL_WEIGHT', 0.08)
        }
        
        # Initialize new master scores
        new_master_scores = pd.Series(np.nan, index=df.index)
        
        for idx in df.index:
            row = df.loc[idx]
            
            # Get available scores
            available_scores = []
            available_weights = []
            
            for col, weight in score_cols.items():
                if col in df.columns and pd.notna(row[col]):
                    available_scores.append(row[col])
                    available_weights.append(weight)
            
            # Calculate weighted score if we have minimum components
            if len(available_scores) >= 3:  # Minimum required components
                # Normalize weights to sum to 1
                available_weights = np.array(available_weights)
                available_weights = available_weights / available_weights.sum()
                
                # Calculate weighted score
                weighted_score = np.dot(available_scores, available_weights)
                new_master_scores[idx] = weighted_score
        
        # Update master score with new calculation
        # Apply the same quality multiplier that was used before
        df['master_score_raw'] = new_master_scores
        df['master_score_before_bonus'] = df['master_score_raw'] * df['quality_multiplier']
        df['master_score'] = df['master_score_before_bonus'].clip(0, 100)
        
        return df
        
    @staticmethod
    def apply_market_state_filter(df: pd.DataFrame, filter_name: str = None) -> pd.DataFrame:
        """
        Apply market state filtering based on trading strategy preferences.
        
        Args:
            df: DataFrame with stocks and their market states
            filter_name: Filter to apply (MOMENTUM, AGGRESSIVE, VALUE, DEFENSIVE, ALL)
                        If None, uses CONFIG.DEFAULT_MARKET_FILTER
        
        Returns:
            Filtered DataFrame optimized for the specified trading strategy
        """
        # Check if filtering is enabled
        if not getattr(CONFIG, 'ENABLE_MARKET_STATE_FILTER', True):
            logger.info("Market state filtering is disabled in configuration")
            return df
        
        # Use default filter if none specified
        if filter_name is None:
            filter_name = getattr(CONFIG, 'DEFAULT_MARKET_FILTER', 'MOMENTUM')
        
        # Get the filter configuration
        filter_config = getattr(CONFIG, 'MARKET_STATE_FILTERS', {}).get(filter_name)
        if not filter_config:
            logger.warning(f"Unknown market state filter: {filter_name}, using ALL")
            filter_name = 'ALL'
            filter_config = getattr(CONFIG, 'MARKET_STATE_FILTERS', {}).get('ALL', {
                'allowed_states': ['STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'ROTATION', 
                                 'SIDEWAYS', 'DOWNTREND', 'STRONG_DOWNTREND', 'BOUNCE'],
                'description': 'No filtering - all market states included'
            })
        
        # MEMORY OPTIMIZATION: Work on view first, only copy if changes needed
        original_count = len(df)
        
        logger.info(f"="*50)
        logger.info(f"Applying {filter_name} market state filter")
        logger.info(f"Original dataset: {original_count} stocks")
        
        # Check if we have market state data
        if 'market_state' not in df.columns:
            logger.warning("No market_state column found - filter cannot be applied")
            logger.info(f"="*50)
            return df
        
        # Get allowed states for this filter
        allowed_states = filter_config.get('allowed_states', [])
        if not allowed_states:
            logger.warning(f"No allowed states defined for {filter_name} filter")
            logger.info(f"="*50)
            return df
        
        # Apply the filter
        try:
            # Create filter mask
            state_mask = df['market_state'].isin(allowed_states)
            
            # MEMORY OPTIMIZATION: Only copy if filtering is actually needed
            if state_mask.all():
                # No filtering needed, return original
                filtered_df = df
                logger.info("No filtering applied - all stocks match criteria")
            else:
                filtered_df = df[state_mask].copy()
            
            # Log filtering results
            filtered_count = len(filtered_df)
            filtered_pct = (filtered_count / original_count * 100) if original_count > 0 else 0
            
            logger.info(f"Filter allowed states: {', '.join(allowed_states)}")
            logger.info(f"Filtered dataset: {filtered_count} stocks ({filtered_pct:.1f}% retained)")
            
            # Show state breakdown before filtering - DIVISION PROTECTED
            state_counts = df['market_state'].value_counts()
            logger.info("Market state distribution before filtering:")
            for state, count in state_counts.items():
                pct = (count / original_count * 100) if original_count > 0 else 0
                status = "✓ ALLOWED" if state in allowed_states else "✗ FILTERED"
                logger.info(f"  {state}: {count} stocks ({pct:.1f}%) - {status}")
            
            # Store filter information for reference
            filtered_df.attrs['market_filter_applied'] = filter_name
            filtered_df.attrs['market_filter_allowed_states'] = allowed_states
            filtered_df.attrs['original_count'] = original_count
            filtered_df.attrs['filtered_count'] = filtered_count
            filtered_df.attrs['filter_retention_pct'] = filtered_pct
            
            # Emergency backup: If filter is too aggressive (less than 5% retention), keep top stocks
            if filtered_count < max(1, original_count * 0.05):
                logger.warning(f"Filter too aggressive ({filtered_pct:.1f}% retention)")
                logger.warning("Falling back to top 20% of stocks by master_score")
                
                # Sort by master_score and keep top 20%
                if 'master_score' in df.columns:
                    df_sorted = df.dropna(subset=['master_score']).sort_values('master_score', ascending=False)
                    backup_count = max(1, int(len(df_sorted) * 0.20))
                    backup_df = df_sorted.head(backup_count).copy()
                    
                    backup_df.attrs['market_filter_applied'] = f"{filter_name}_BACKUP"
                    backup_df.attrs['backup_reason'] = "Filter too aggressive"
                    backup_df.attrs['backup_retention_pct'] = (backup_count / original_count * 100)
                    
                    logger.info(f"Backup filter: keeping top {backup_count} stocks ({backup_df.attrs['backup_retention_pct']:.1f}%)")
                    logger.info(f"="*50)
                    
                    return backup_df
            
            logger.info(f"Market state filtering completed successfully")
            logger.info(f"="*50)
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Market state filtering failed: {e}")
            logger.error("Returning unfiltered dataset")
            logger.info(f"="*50)
            return df
    
    @staticmethod
    def _calculate_position_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate position score based on actual market dynamics.
        FIXED: No arbitrary sweet spots, clear breakout/reversal logic.
        
        Core Philosophy:
        - Extremes matter, middle doesn't
        - Near 52w high = Breakout potential (momentum play)
        - Near 52w low = Reversal potential (value play)
        - Middle range = Dead zone (no edge)
        
        Scoring Strategy:
        - This implementation favors MOMENTUM (high position = high score)
        - For value strategy, simply invert the final score
        
        Score Range:
        - 85-100: Breaking out or above 52w high
        - 70-85: Near 52w high (strength zone)
        - 30-70: Middle range (neutral/dead zone)
        - 15-30: Near 52w low (weakness)
        - 0-15: At/below 52w low (extreme weakness)
        """
        # Initialize with NaN
        position_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Check required data
        if 'from_low_pct' not in df.columns or 'from_high_pct' not in df.columns:
            logger.warning("Missing position data (from_low_pct/from_high_pct)")
            return position_score
        
        from_low = df['from_low_pct']
        from_high = df['from_high_pct']
        
        # Only calculate where both values exist
        valid_mask = from_low.notna() & from_high.notna()
        
        if not valid_mask.any():
            logger.warning("No valid position data available")
            return position_score
        
        # CORE CALCULATION: Simple position in range
        # This is the true position (0-100% of 52w range)
        position_in_range = pd.Series(np.nan, index=df.index)
        
        # For stocks below 52w high (normal case)
        below_high = valid_mask & (from_high <= 0)
        if below_high.any():
            # Direct position calculation
            position_in_range[below_high] = from_low[below_high].clip(0, 100)
        
        # For stocks above 52w high (breakout)
        above_high = valid_mask & (from_high > 0)
        if above_high.any():
            # Above high = 100+ (can go to 110 for 10% above high)
            position_in_range[above_high] = 100 + from_high[above_high].clip(0, 20)
        
        # SCORING PHILOSOPHY: Favor extremes, penalize middle
        # Research shows stocks at extremes have edge, middle range doesn't
        
        if valid_mask.any():
            # Non-linear scoring that emphasizes extremes
            for idx in df[valid_mask].index:
                pos = position_in_range[idx]
                
                if pd.isna(pos):
                    continue
                    
                # BREAKOUT ZONE (>90%): High scores
                if pos >= 90:
                    if pos > 100:  # Above 52w high
                        # Diminishing returns above high
                        position_score[idx] = 90 + np.minimum(10, (pos - 100) * 0.5)
                    else:  # 90-100%
                        position_score[idx] = 80 + (pos - 90) * 1.0
                
                # STRENGTH ZONE (70-90%): Good scores
                elif pos >= 70:
                    position_score[idx] = 60 + (pos - 70) * 1.0
                
                # DEAD ZONE (30-70%): Reduced scores
                # This is where your "sweet spot" was, but it's actually dead money
                elif pos >= 30:
                    position_score[idx] = 30 + (pos - 30) * 0.75  # Compressed scoring
                
                # VALUE ZONE (10-30%): Lower scores (but valuable for mean reversion)
                elif pos >= 10:
                    position_score[idx] = 15 + (pos - 10) * 0.75
                
                # EXTREME LOW (<10%): Minimum scores
                else:
                    position_score[idx] = pos * 1.5
        
        # CONTEXT ADJUSTMENTS
        
        # 1. MOMENTUM CONFIRMATION
        # High position needs momentum to confirm
        if 'ret_30d' in df.columns and position_score.notna().any():
            ret_30d = df['ret_30d']
            
            # Breakout with momentum = confirmed
            breakout_confirmed = (
                (position_in_range > 85) & 
                (ret_30d > 10) & 
                position_score.notna()
            )
            if breakout_confirmed.any():
                position_score[breakout_confirmed] = np.minimum(
                    position_score[breakout_confirmed] * 1.1,
                    100
                )
            
            # High position but negative momentum = false breakout
            false_breakout = (
                (position_in_range > 85) & 
                (ret_30d < -5) & 
                position_score.notna()
            )
            if false_breakout.any():
                position_score[false_breakout] *= 0.8
            
            # Low position with positive momentum = potential reversal
            reversal_starting = (
                (position_in_range < 20) & 
                (ret_30d > 5) & 
                position_score.notna()
            )
            if reversal_starting.any():
                position_score[reversal_starting] += 10
        
        # 2. VOLUME CONFIRMATION
        # Breakouts need volume
        if 'rvol' in df.columns and position_score.notna().any():
            rvol = df['rvol']
            
            # High position with volume = strong
            high_with_volume = (
                (position_in_range > 80) & 
                (rvol > 1.5) & 
                position_score.notna()
            )
            if high_with_volume.any():
                position_score[high_with_volume] = np.minimum(
                    position_score[high_with_volume] + 5,
                    100
                )
            
            # High position without volume = weak
            high_no_volume = (
                (position_in_range > 80) & 
                (rvol < 0.8) & 
                position_score.notna()
            )
            if high_no_volume.any():
                position_score[high_no_volume] -= 10
        
        # 3. TIME AT EXTREME
        # How long has it been near high/low?
        if 'sma_20d' in df.columns and 'price' in df.columns:
            price = df['price']
            sma_20 = df['sma_20d']
            
            valid_time = price.notna() & sma_20.notna() & position_score.notna()
            
            if valid_time.any():
                # If near high and price > SMA20 = sustained strength
                sustained_high = (
                    valid_time & 
                    (position_in_range > 80) & 
                    (price > sma_20)
                )
                if sustained_high.any():
                    position_score[sustained_high] += 3
                
                # If near low and price < SMA20 = sustained weakness
                sustained_low = (
                    valid_time & 
                    (position_in_range < 20) & 
                    (price < sma_20)
                )
                if sustained_low.any():
                    position_score[sustained_low] -= 3
        
        # 4. MARKET CAP CONTEXT
        if 'category' in df.columns and position_score.notna().any():
            category = df['category']
            
            # Large caps at extremes are more significant
            is_large = category.isin(['Large Cap', 'Mega Cap'])
            large_extreme = (
                is_large & 
                ((position_in_range > 90) | (position_in_range < 10)) & 
                position_score.notna()
            )
            if large_extreme.any():
                position_score[large_extreme] = (
                    50 + (position_score[large_extreme] - 50) * 1.2
                ).clip(0, 100)
            
            # Small caps above high need caution
            is_small = category.isin(['Micro Cap', 'Small Cap'])
            small_extended = (
                is_small & 
                (position_in_range > 110) & 
                position_score.notna()
            )
            if small_extended.any():
                position_score[small_extended] = np.minimum(
                    position_score[small_extended],
                    85
                )
        
        # 5. MARKET REGIME
        # In bull markets, high position better. In bear, low position better.
        if position_score.notna().any() and len(df[valid_mask]) > 100:
            # Calculate market regime from average position
            avg_position = position_in_range[valid_mask].median()
            
            if pd.notna(avg_position):
                if avg_position > 65:
                    # Bull market: Boost high positions
                    bull_high = (position_in_range > 70) & position_score.notna()
                    if bull_high.any():
                        position_score[bull_high] += 5
                        
                elif avg_position < 35:
                    # Bear market: Value in low positions
                    bear_low = (position_in_range < 30) & position_score.notna()
                    if bear_low.any():
                        position_score[bear_low] += 5
        
        # Final clipping
        position_score = position_score.clip(0, 100)
        
        # NEVER FILL NaN!
        
        # COMPREHENSIVE LOGGING
        valid_scores = position_score.notna().sum()
        total_stocks = len(df)
        
        logger.info(f"Position scores: {valid_scores}/{total_stocks} calculated")
        
        if valid_scores > 0:
            score_dist = position_score[position_score.notna()]
            logger.info(f"Distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Position distribution
            if valid_mask.any():
                pos_dist = position_in_range[valid_mask]
                at_highs = (pos_dist > 90).sum()
                at_lows = (pos_dist < 10).sum()
                in_middle = ((pos_dist >= 30) & (pos_dist <= 70)).sum()
                
                logger.info(f"Position distribution: {at_highs} near highs, "
                           f"{at_lows} near lows, {in_middle} in dead zone")
                
                if in_middle > len(pos_dist) * 0.5:
                    logger.warning(f"Warning: {in_middle} stocks in dead zone (30-70%)")
        
        return position_score
    
    @staticmethod
    def _calculate_volume_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate volume score focusing on what actually matters.
        SIMPLIFIED: Volume confirms, doesn't predict. Focus on liquidity and relative volume.
        
        Core Philosophy:
        - Volume is a CONFIRMING indicator, not predictive
        - Relative volume matters more than absolute
        - Liquidity (turnover) matters for tradability
        - Extreme volume often signals problems
        
        Components:
        - 50% Relative volume (vs historical average)
        - 30% Liquidity (can you actually trade it?)
        - 20% Price-volume harmony (does volume confirm price action?)
        
        Score Range:
        - 70-100: High relative volume with price confirmation
        - 50-70: Above average activity
        - 30-50: Normal activity
        - 10-30: Below average (concerning)
        - 0-10: Dead (no liquidity)
        """
        # Initialize with NaN
        volume_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Check minimum data
        has_rvol = 'rvol' in df.columns
        has_volume = 'volume_1d' in df.columns
        has_price = 'price' in df.columns
        
        if not has_rvol and not has_volume:
            logger.warning("No volume data available")
            return volume_score
        
        # COMPONENT 1: RELATIVE VOLUME (50% weight)
        # This is the most important - is volume unusual today?
        relative_component = pd.Series(np.nan, index=df.index, dtype=float)
        
        if has_rvol:
            rvol = df['rvol']
            valid_rvol = rvol.notna() & (rvol >= 0)
            
            if valid_rvol.any():
                # Simple but effective scoring
                # < 0.5x = dead (20)
                # 0.5-1x = below normal (20-50)
                # 1-2x = normal to elevated (50-70)
                # 2-5x = high interest (70-85)
                # > 5x = extreme (85-90, capped due to potential manipulation)
                
                relative_component[valid_rvol] = np.where(
                    rvol[valid_rvol] < 0.5,
                    rvol[valid_rvol] * 40,  # 0-20
                    np.where(
                        rvol[valid_rvol] < 1,
                        20 + (rvol[valid_rvol] - 0.5) * 60,  # 20-50
                        np.where(
                            rvol[valid_rvol] < 2,
                            50 + (rvol[valid_rvol] - 1) * 20,  # 50-70
                            np.where(
                                rvol[valid_rvol] < 5,
                                70 + np.log(rvol[valid_rvol]) * 10,  # 70-85 (log scale)
                                np.minimum(90, 85 + np.log(rvol[valid_rvol] / 5) * 5)  # 85-90 cap
                            )
                        )
                    )
                )
        
        # COMPONENT 2: LIQUIDITY (30% weight)
        # Can you actually trade this stock?
        liquidity_component = pd.Series(np.nan, index=df.index, dtype=float)
        
        if has_volume and has_price:
            volume = df['volume_1d']
            price = df['price']
            
            valid_liquidity = volume.notna() & price.notna() & (volume >= 0) & (price > 0)
            
            if valid_liquidity.any():
                # Calculate turnover in lakhs (100,000s)
                turnover_lakhs = (volume[valid_liquidity] * price[valid_liquidity]) / 100000
                
                # Scoring based on Indian market standards
                # < 10 lakhs = illiquid (0-30)
                # 10-100 lakhs = low liquidity (30-50)
                # 100-1000 lakhs = decent (50-70)
                # 1000-10000 lakhs = good (70-85)
                # > 10000 lakhs = excellent (85-100)
                
                log_turnover = np.log10(turnover_lakhs + 1)  # Add 1 to handle zero
                
                liquidity_component[valid_liquidity] = np.where(
                    log_turnover < 1,  # < 10 lakhs
                    log_turnover * 30,
                    np.where(
                        log_turnover < 2,  # 10-100 lakhs
                        30 + (log_turnover - 1) * 20,
                        np.where(
                            log_turnover < 3,  # 100-1000 lakhs
                            50 + (log_turnover - 2) * 20,
                            np.where(
                                log_turnover < 4,  # 1000-10000 lakhs
                                70 + (log_turnover - 3) * 15,
                                np.minimum(100, 85 + (log_turnover - 4) * 10)
                            )
                        )
                    )
                )
        
        # COMPONENT 3: PRICE-VOLUME HARMONY (20% weight)
        # Does volume confirm price action?
        harmony_component = pd.Series(50, index=df.index, dtype=float)  # Default neutral
        
        if has_rvol and 'ret_1d' in df.columns:
            rvol = df['rvol']
            ret_1d = df['ret_1d']
            
            valid_harmony = rvol.notna() & ret_1d.notna()
            
            if valid_harmony.any():
                # Best: High volume + strong price move (either direction)
                strong_move_confirmed = valid_harmony & (rvol > 1.5) & (np.abs(ret_1d) > 3)
                harmony_component[strong_move_confirmed] = 80
                
                # Good: Above average volume + price move
                normal_confirmed = valid_harmony & (rvol > 1) & (np.abs(ret_1d) > 1)
                harmony_component[normal_confirmed] = 65
                
                # Bad: High volume + no price move (distribution/accumulation)
                high_vol_no_move = valid_harmony & (rvol > 2) & (np.abs(ret_1d) < 0.5)
                harmony_component[high_vol_no_move] = 30
                
                # Suspicious: Big price move + no volume
                move_no_volume = valid_harmony & (np.abs(ret_1d) > 5) & (rvol < 0.7)
                harmony_component[move_no_volume] = 20
                
                # Neutral: Everything else stays at 50
        
        # COMBINE COMPONENTS
        components = [
            (relative_component, 0.50),
            (liquidity_component, 0.30),
            (harmony_component, 0.20)
        ]
        
        for idx in df.index:
            valid_components = []
            valid_weights = []
            
            for component, weight in components:
                if pd.notna(component[idx]):
                    valid_components.append(component[idx])
                    valid_weights.append(weight)
            
            if valid_components:
                total_weight = sum(valid_weights)
                if total_weight > 0:
                    normalized_weights = [w/total_weight for w in valid_weights]
                    volume_score[idx] = sum(c * w for c, w in zip(valid_components, normalized_weights))
        
        # CONTEXT ADJUSTMENTS (Simplified)
        
        # Market cap context
        if 'category' in df.columns and volume_score.notna().any():
            category = df['category']
            
            # Penny stocks: High volume more common, reduce score
            is_penny = category.isin(['Micro Cap', 'Small Cap'])
            if has_rvol:
                rvol = df['rvol']
                penny_high = is_penny & (rvol > 3) & volume_score.notna()
                if penny_high.any():
                    volume_score[penny_high] *= 0.85
            
            # Large caps: High volume more significant
            is_large = category.isin(['Large Cap', 'Mega Cap'])
            if has_rvol:
                large_high = is_large & (rvol > 2) & volume_score.notna()
                if large_high.any():
                    volume_score[large_high] = np.minimum(
                        volume_score[large_high] * 1.1,
                        100
                    )
        
        # Extreme volume investigation
        if has_rvol:
            rvol = df['rvol']
            
            # Very extreme volume (>10x) is usually bad news
            extreme = (rvol > 10) & volume_score.notna()
            if extreme.any():
                # Check if it's with bad price action
                if 'ret_1d' in df.columns:
                    ret_1d = df['ret_1d']
                    extreme_negative = extreme & (ret_1d < -5)
                    if extreme_negative.any():
                        volume_score[extreme_negative] = np.minimum(volume_score[extreme_negative], 60)
                        logger.warning(f"Extreme volume with negative price action in {extreme_negative.sum()} stocks")
        
        # Final clipping
        volume_score = volume_score.clip(0, 100)
        
        # NEVER FILL NaN!
        
        # COMPREHENSIVE LOGGING
        valid_scores = volume_score.notna().sum()
        total_stocks = len(df)
        
        logger.info(f"Volume scores: {valid_scores}/{total_stocks} calculated")
        
        if valid_scores > 0:
            score_dist = volume_score[volume_score.notna()]
            logger.info(f"Distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Volume categories
            if has_rvol:
                rvol = df['rvol']
                dead = (rvol < 0.5).sum()
                normal = ((rvol >= 0.5) & (rvol < 2)).sum()
                elevated = ((rvol >= 2) & (rvol < 5)).sum()
                extreme = (rvol >= 5).sum()
                
                logger.debug(f"Volume distribution: Dead={dead}, Normal={normal}, "
                            f"Elevated={elevated}, Extreme={extreme}")
        
        return volume_score
        
    @staticmethod
    def _calculate_momentum_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum score with refined scaling and true volatility adjustment.
        REFINED: Better scaling calibration, proper volatility, cleaner logic.
        
        Core Philosophy:
        - Momentum = Recent price performance with quality adjustments
        - Scale based on actual market return distributions
        - True volatility measurement when possible
        - Multi-component for robustness
        
        Components:
        - 60% Raw momentum (market-calibrated scaling)
        - 20% Consistency (direction and magnitude)
        - 20% Quality (true Sharpe-like ratio)
        
        Score Range:
        - 80-100: Exceptional momentum
        - 60-80: Strong momentum
        - 40-60: Neutral momentum
        - 20-40: Weak momentum
        - 0-20: Severe negative momentum
        """
        # Initialize with NaN - NEVER fill with defaults
        momentum_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Check data availability
        has_30d = 'ret_30d' in df.columns
        has_7d = 'ret_7d' in df.columns
        has_1d = 'ret_1d' in df.columns
        
        if not has_30d and not has_7d:
            logger.warning("No return data for momentum calculation")
            return momentum_score
        
        # COMPONENT 1: RAW MOMENTUM (60% weight)
        # Calibrated based on actual market distributions
        raw_momentum = pd.Series(np.nan, index=df.index, dtype=float)
        
        if has_30d:
            ret_30d = df['ret_30d']
            valid_30d = ret_30d.notna()
            
            if valid_30d.any():
                # REFINED SCALING: Based on empirical market data
                # Indian market monthly returns typically:
                # 95th percentile: ~25%
                # 75th percentile: ~10%
                # Median: ~1%
                # 25th percentile: ~-5%
                # 5th percentile: ~-15%
                
                # Use sigmoid with market-calibrated center and scale
                # Center at 1% (typical median), scale for proper spread
                raw_momentum[valid_30d] = 50 / (1 + np.exp(-(ret_30d[valid_30d] - 1) / 10))
                
                # Enhance extremes (non-linear at tails)
                very_strong = valid_30d & (ret_30d > 30)
                if very_strong.any():
                    raw_momentum[very_strong] = 90 + np.tanh((ret_30d[very_strong] - 30) / 20) * 10
                
                very_weak = valid_30d & (ret_30d < -20)
                if very_weak.any():
                    raw_momentum[very_weak] = 10 * (1 + ret_30d[very_weak] / 20)
        
        # Fallback to 7-day (properly scaled)
        needs_7d = raw_momentum.isna() & has_7d
        if needs_7d.any() and has_7d:
            ret_7d = df['ret_7d']
            valid_7d = ret_7d.notna() & needs_7d
            
            if valid_7d.any():
                # Convert to monthly equivalent using compound rate
                # (1 + r_monthly) = (1 + r_weekly)^4.3
                monthly_equivalent = np.sign(ret_7d[valid_7d]) * (
                    (np.abs(1 + ret_7d[valid_7d]/100) ** 4.3 - 1) * 100
                )
                
                # Apply same sigmoid scaling
                raw_momentum[valid_7d] = 50 / (1 + np.exp(-(monthly_equivalent - 1) / 10))
        
        # COMPONENT 2: CONSISTENCY FACTOR (20% weight)
        # Improved measurement of momentum consistency
        consistency_factor = pd.Series(50, index=df.index, dtype=float)
        
        if all([has_1d, has_7d, has_30d]):
            ret_1d = df['ret_1d']
            ret_7d = df['ret_7d']
            ret_30d = df['ret_30d']
            
            valid_all = ret_1d.notna() & ret_7d.notna() & ret_30d.notna()
            
            if valid_all.any():
                # Calculate momentum alignment score
                # All positive with increasing strength = best
                # Mixed signals = neutral
                # All negative with worsening = worst
                
                # Sign alignment (direction consistency)
                signs = np.sign(np.column_stack([ret_1d[valid_all], 
                                                ret_7d[valid_all], 
                                                ret_30d[valid_all]]))
                sign_consistency = np.abs(signs.sum(axis=1)) / 3  # 0 to 1
                
                # Magnitude progression (is momentum building?)
                daily_equiv_7d = ret_7d[valid_all] / 7
                daily_equiv_30d = ret_30d[valid_all] / 30
                
                building = (ret_1d[valid_all] > daily_equiv_7d) & (daily_equiv_7d > daily_equiv_30d)
                steady = np.abs(ret_1d[valid_all] - daily_equiv_7d) < 2
                fading = (ret_1d[valid_all] < daily_equiv_7d) & (daily_equiv_7d < daily_equiv_30d)
                
                # Combine sign and magnitude
                consistency_factor[valid_all] = 50  # Base
                consistency_factor[valid_all] += sign_consistency * 20  # ±20 for direction
                consistency_factor[building] += 20  # Bonus for building
                consistency_factor[fading] -= 20  # Penalty for fading
                consistency_factor[steady] += 10  # Small bonus for steady
        
        # COMPONENT 3: QUALITY FACTOR (20% weight)
        # True Sharpe-like ratio using actual volatility
        quality_factor = pd.Series(50, index=df.index, dtype=float)
        
        # Try to use actual daily data if available
        if 'daily_returns' in df.columns:
            # If we have actual daily returns series
            daily_returns = df['daily_returns']
            valid_daily = daily_returns.notna()
            
            if valid_daily.any():
                # Calculate true volatility
                volatility = daily_returns[valid_daily].apply(lambda x: np.std(x) if len(x) > 1 else np.nan)
                returns = df.loc[valid_daily, 'ret_30d'] if has_30d else df.loc[valid_daily, 'ret_7d']
                
                # Sharpe-like ratio
                sharpe = returns / (volatility * np.sqrt(30) + 1)  # Annualized
                quality_factor[valid_daily] = 50 + np.tanh(sharpe / 2) * 30
        else:
            # Fallback: Estimate volatility from available returns
            if has_30d and has_7d and has_1d:
                ret_30d = df['ret_30d']
                ret_7d = df['ret_7d']
                ret_1d = df['ret_1d']
                
                valid_vol = ret_30d.notna() & ret_7d.notna() & ret_1d.notna()
                
                if valid_vol.any():
                    # Estimate volatility from return dispersion
                    # Better than std of averages
                    expected_1d = ret_30d[valid_vol] / 30
                    expected_7d = ret_30d[valid_vol] * 7 / 30
                    
                    deviation_1d = np.abs(ret_1d[valid_vol] - expected_1d)
                    deviation_7d = np.abs(ret_7d[valid_vol] - expected_7d)
                    
                    # Average deviation as volatility proxy
                    avg_deviation = (deviation_1d + deviation_7d / 7) / 2
                    
                    # Quality score based on return/risk
                    risk_adjusted = ret_30d[valid_vol] / (avg_deviation + 1)
                    quality_factor[valid_vol] = 50 + np.tanh(risk_adjusted / 10) * 30
        
        # COMBINE COMPONENTS
        has_components = raw_momentum.notna()
        
        if has_components.any():
            # Start with raw momentum
            momentum_score[has_components] = raw_momentum[has_components] * 0.6
            
            # Add consistency if available
            if consistency_factor.notna().any():
                valid_both = has_components & consistency_factor.notna()
                momentum_score[valid_both] += consistency_factor[valid_both] * 0.2
            else:
                # If no consistency data, give more weight to raw
                momentum_score[has_components] = raw_momentum[has_components] * 0.8
            
            # Add quality if available
            if quality_factor.notna().any():
                valid_all = has_components & quality_factor.notna()
                momentum_score[valid_all] += quality_factor[valid_all] * 0.2
        
        # CONTEXT ADJUSTMENTS
        
        # Market cap adjustments
        if 'category' in df.columns and momentum_score.notna().any():
            category = df['category']
            
            # Small cap pump & dump detection
            is_small = category.isin(['Micro Cap', 'Small Cap'])
            
            if has_30d:
                ret_30d = df['ret_30d']
                
                # Progressive penalty for extreme returns - More realistic threshold
                extreme_small = is_small & momentum_score.notna() & (ret_30d > 35)  # Reduced from 50%
                if extreme_small.any():
                    penalty_factor = np.exp(-(ret_30d[extreme_small] - 35) / 35)  # Adjusted calculation
                    momentum_score[extreme_small] *= penalty_factor
                    logger.debug(f"Applied pump penalty to {extreme_small.sum()} small caps")
            
            # Large cap momentum significance
            is_large = category.isin(['Large Cap', 'Mega Cap'])
            strong_large = is_large & momentum_score.notna() & (momentum_score > 70)
            if strong_large.any():
                momentum_score[strong_large] = np.minimum(
                    momentum_score[strong_large] * 1.08,
                    100
                )
        
        # Volume confirmation
        if 'rvol' in df.columns and momentum_score.notna().any():
            rvol = df['rvol']
            
            # Momentum-volume harmony
            high_momentum = momentum_score > 70
            
            # Strong momentum with volume = confirmed
            with_volume = high_momentum & (rvol > 1.5) & momentum_score.notna()
            if with_volume.any():
                momentum_score[with_volume] = np.minimum(
                    momentum_score[with_volume] + 5,
                    100
                )
            
            # Strong momentum without volume = suspicious
            no_volume = high_momentum & (rvol < 0.7) & momentum_score.notna()
            if no_volume.any():
                momentum_score[no_volume] -= 10
        
        # Final clipping
        momentum_score = momentum_score.clip(0, 100)
        
        # NEVER FILL NaN!
        
        # COMPREHENSIVE LOGGING
        valid_scores = momentum_score.notna().sum()
        total_stocks = len(df)
        
        logger.info(f"Momentum scores: {valid_scores}/{total_stocks} calculated")
        
        if valid_scores > 0:
            score_dist = momentum_score[momentum_score.notna()]
            logger.info(f"Distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Check market momentum
            if score_dist.mean() > 65:
                logger.info("Strong market-wide momentum detected")
            elif score_dist.mean() < 35:
                logger.warning("Weak market-wide momentum detected")
        
        return momentum_score
        
    @staticmethod
    def _calculate_acceleration_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate momentum acceleration using proper rate of change analysis.
        FIXED: Correct math, no division issues, clear scoring logic.
        
        Core Concept:
        - Acceleration = Change in momentum over time
        - Compare recent vs historical momentum rates
        - Use log returns for proper compounding
        - Smooth transitions, no arbitrary bins
        
        Method:
        - Calculate momentum over different periods
        - Compare short vs long momentum slopes
        - Score based on momentum improvement/deterioration
        
        Score Range:
        - 80-100: Strong positive acceleration (momentum building rapidly)
        - 60-80: Moderate acceleration (steady improvement)
        - 40-60: Neutral (constant momentum)
        - 20-40: Deceleration (momentum fading)
        - 0-20: Strong deceleration (momentum collapsing)
        """
        # Initialize with NaN
        acceleration_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Check minimum required columns
        required_cols = ['ret_1d', 'ret_7d', 'ret_30d']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning(f"Insufficient data for acceleration: only {len(available_cols)} return columns")
            return acceleration_score
        
        # METHOD 1: Momentum Slope Comparison
        # Calculate momentum at different time points and compare slopes
        
        momentum_scores = pd.DataFrame(index=df.index)
        
        # Current momentum (most recent)
        if 'ret_7d' in df.columns:
            # Weekly momentum as baseline
            momentum_scores['current'] = df['ret_7d']
        
        # Historical momentum (for comparison)
        if 'ret_30d' in df.columns:
            # Monthly momentum
            momentum_scores['historical'] = df['ret_30d']
        
        # Very short-term momentum
        if 'ret_1d' in df.columns:
            momentum_scores['immediate'] = df['ret_1d']
        
        # Calculate acceleration as momentum improvement
        valid_data = pd.Series(False, index=df.index)
        
        # Primary calculation: 7d vs 30d momentum
        if 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            ret_7d = df['ret_7d']
            ret_30d = df['ret_30d']
            
            valid = ret_7d.notna() & ret_30d.notna()
            valid_data |= valid
            
            if valid.any():
                # Calculate average daily momentum for each period
                # Use compound rate for accuracy: (1 + r)^(1/n) - 1
                
                # Safe calculation avoiding negative final values
                safe_7d = np.where(ret_7d > -99, ret_7d, -99)
                safe_30d = np.where(ret_30d > -99, ret_30d, -99)
                
                # Daily equivalent rates
                daily_7d = np.sign(safe_7d) * (np.abs(1 + safe_7d/100) ** (1/7) - 1) * 100
                daily_30d = np.sign(safe_30d) * (np.abs(1 + safe_30d/100) ** (1/30) - 1) * 100
                
                # Acceleration factor: recent momentum vs historical
                # Positive = accelerating, Negative = decelerating
                momentum_change = daily_7d - daily_30d
                
                # Convert to 0-100 score using sigmoid-like function
                # Center at 0 (no acceleration) = 50 score
                # Use tanh for smooth transitions
                acceleration_score[valid] = 50 + 30 * np.tanh(momentum_change[valid] / 5)
        
        # Enhanced calculation with 1-day data
        if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
            ret_1d = df['ret_1d']
            ret_7d = df['ret_7d']
            
            valid_short = ret_1d.notna() & ret_7d.notna()
            
            if valid_short.any():
                # Very short-term acceleration
                daily_7d_rate = np.sign(ret_7d) * (np.abs(1 + ret_7d/100) ** (1/7) - 1) * 100
                
                # Compare today vs weekly average
                short_acceleration = ret_1d - daily_7d_rate
                
                # Blend with main score if available
                if acceleration_score.notna().any():
                    valid_blend = valid_short & acceleration_score.notna()
                    # 70% weight on primary, 30% on short-term
                    acceleration_score[valid_blend] = (
                        acceleration_score[valid_blend] * 0.7 +
                        (50 + 30 * np.tanh(short_acceleration[valid_blend] / 3)) * 0.3
                    )
                else:
                    # Use short-term as primary if no other data
                    acceleration_score[valid_short] = 50 + 30 * np.tanh(short_acceleration[valid_short] / 3)
                    valid_data |= valid_short
        
        # METHOD 2: Momentum Consistency Bonus
        # Reward consistent acceleration across timeframes
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
            ret_1d = df['ret_1d']
            ret_7d = df['ret_7d']  
            ret_30d = df['ret_30d']
            
            valid_all = ret_1d.notna() & ret_7d.notna() & ret_30d.notna() & acceleration_score.notna()
            
            if valid_all.any():
                # Check for consistent improvement pattern
                # Each period better than the longer one (normalized)
                improving = (
                    (ret_1d > ret_7d / 7) & 
                    (ret_7d / 7 > ret_30d / 30) &
                    (ret_1d > 0)
                )
                
                # Add bonus for consistent acceleration
                consistent_accel = valid_all & improving
                if consistent_accel.any():
                    acceleration_score[consistent_accel] = np.minimum(
                        acceleration_score[consistent_accel] + 10,
                        95
                    )
                
                # Penalty for consistent deceleration
                declining = (
                    (ret_1d < ret_7d / 7) & 
                    (ret_7d / 7 < ret_30d / 30) &
                    (ret_1d < 0)
                )
                
                consistent_decel = valid_all & declining
                if consistent_decel.any():
                    acceleration_score[consistent_decel] = np.maximum(
                        acceleration_score[consistent_decel] - 10,
                        5
                    )
        
        # CONTEXT ADJUSTMENTS
        
        # Volume confirmation
        if 'rvol' in df.columns and acceleration_score.notna().any():
            rvol = df['rvol']
            
            # Strong acceleration needs volume
            strong_accel = (acceleration_score > 70) & rvol.notna()
            
            # With volume = confirmed
            with_volume = strong_accel & (rvol > 1.5)
            if with_volume.any():
                acceleration_score[with_volume] = np.minimum(
                    acceleration_score[with_volume] * 1.05,
                    100
                )
            
            # Without volume = suspicious
            no_volume = strong_accel & (rvol < 0.8)
            if no_volume.any():
                acceleration_score[no_volume] *= 0.90
        
        # Market cap adjustment
        if 'category' in df.columns and acceleration_score.notna().any():
            category = df['category']
            
            # Small caps: More volatile, normalize extremes
            is_small = category.isin(['Micro Cap', 'Small Cap'])
            small_mask = is_small & acceleration_score.notna()
            
            if small_mask.any():
                # Pull extremes toward center
                acceleration_score[small_mask] = 50 + (acceleration_score[small_mask] - 50) * 0.8
            
            # Large caps: Acceleration more significant
            is_large = category.isin(['Large Cap', 'Mega Cap'])
            large_accel = is_large & (acceleration_score > 65) & acceleration_score.notna()
            
            if large_accel.any():
                acceleration_score[large_accel] = np.minimum(
                    acceleration_score[large_accel] * 1.05,
                    100
                )
        
        # Handle edge cases
        # Extreme positive returns might break calculations
        if 'ret_30d' in df.columns:
            extreme_returns = (df['ret_30d'] > 200) & acceleration_score.notna()
            if extreme_returns.any():
                # Cap acceleration score for extreme movers
                acceleration_score[extreme_returns] = np.minimum(
                    acceleration_score[extreme_returns],
                    85
                )
        
        # Final clipping
        acceleration_score = acceleration_score.clip(0, 100)
        
        # COMPREHENSIVE LOGGING
        valid_scores = acceleration_score.notna().sum()
        total_stocks = len(df)
        
        logger.info(f"Acceleration scores: {valid_scores}/{total_stocks} calculated")
        
        if valid_scores > 0:
            score_dist = acceleration_score[acceleration_score.notna()]
            logger.info(f"Distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Categories
            strong_accel = (acceleration_score > 80).sum()
            moderate_accel = ((acceleration_score > 60) & (acceleration_score <= 80)).sum()
            neutral = ((acceleration_score >= 40) & (acceleration_score <= 60)).sum()
            moderate_decel = ((acceleration_score >= 20) & (acceleration_score < 40)).sum()
            strong_decel = (acceleration_score < 20).sum()
            
            logger.debug(f"Acceleration breakdown: Strong Accel={strong_accel}, "
                        f"Moderate Accel={moderate_accel}, Neutral={neutral}, "
                        f"Moderate Decel={moderate_decel}, Strong Decel={strong_decel}")
            
            # Check for market-wide acceleration
            if score_dist.mean() > 65:
                logger.info("Market-wide positive acceleration detected")
            elif score_dist.mean() < 35:
                logger.warning("Market-wide deceleration detected")
        
        return acceleration_score
        
    @staticmethod
    def _calculate_breakout_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate breakout probability using FULLY VECTORIZED operations.
        FIXED: No loops, consistent weighting, proper breakout patterns.
        
        Breakout Methodology:
        - Distance from 52w high with exponential weighting
        - Volume accumulation patterns (not just surge)
        - Trend strength with proper SMA hierarchy
        - Breakout pattern recognition (cup & handle, flag, consolidation)
        - Volume-price confirmation
        
        Score Components:
        - 35% Distance from high (exponential decay)
        - 30% Volume pattern (accumulation vs distribution)
        - 20% Trend alignment (SMA structure)
        - 15% Price consolidation (volatility compression)
        
        Bonuses Applied:
        - Cup & Handle: +15 points
        - Flag Pattern: +12 points
        - Ascending Triangle: +10 points
        - Volume Dry-up: +8 points (pre-breakout signal)
        """
        breakout_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Component 1: DISTANCE FROM HIGH (35% weight)
        # Uses exponential decay - being at high is exponentially better than being far
        distance_component = pd.Series(50, index=df.index, dtype=float)
        
        if 'from_high_pct' in df.columns:
            from_high = pd.Series(df['from_high_pct'].values, index=df.index)
            valid_high = from_high.notna()
            
            if valid_high.any():
                # from_high is negative (e.g., -5 means 5% below high)
                # Convert to positive distance for calculation
                distance = -from_high[valid_high]
                
                # Exponential scoring: Closer to high = exponentially better
                # Using tanh for smooth S-curve that handles all ranges well
                # Distance 0% = 100, 5% = 85, 10% = 70, 20% = 45, 50% = 10
                distance_component[valid_high] = 100 * (1 - np.tanh(distance / 20))
                
                # Special handling for stocks ABOVE 52w high (breakout already happened)
                above_high = valid_high & (from_high > 0)
                if above_high.any():
                    # Recent breakout (0-10% above) = Good
                    recent_breakout = above_high & (from_high <= 10)
                    distance_component[recent_breakout] = 90 - from_high[recent_breakout] * 2  # 90 to 70
                    
                    # Extended breakout (10-30% above) = Careful
                    extended = above_high & (from_high > 10) & (from_high <= 30)
                    distance_component[extended] = 70 - (from_high[extended] - 10) * 1.5  # 70 to 40
                    
                    # Overextended (>30% above) = Dangerous
                    overextended = above_high & (from_high > 30)
                    distance_component[overextended] = 30  # Flat low score
        
        # Component 2: VOLUME PATTERN (30% weight)
        # Not just current volume, but pattern over time
        volume_component = pd.Series(50, index=df.index, dtype=float)
        
        # Sub-component 2a: Volume trend (is volume building?)
        if all(col in df.columns for col in ['vol_ratio_1d_90d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            vol_1d = pd.Series(df['vol_ratio_1d_90d'].values, index=df.index)
            vol_7d = pd.Series(df['vol_ratio_7d_90d'].values, index=df.index)
            vol_30d = pd.Series(df['vol_ratio_30d_90d'].values, index=df.index)
            
            valid_vol = vol_1d.notna() & vol_7d.notna() & vol_30d.notna()
            
            if valid_vol.any():
                # Calculate volume progression score
                # Best: Steady increase (30d < 7d < 1d)
                volume_progression = pd.Series(50, index=df.index, dtype=float)
                
                # Accumulation pattern: Volume building over time
                accumulation = valid_vol & (vol_1d > vol_7d) & (vol_7d > vol_30d)
                volume_progression[accumulation] = 70 + (vol_1d[accumulation] - 1) * 15  # 70-100
                
                # Steady volume: Consistent across timeframes
                steady = valid_vol & (np.abs(vol_1d - vol_7d) < 0.3) & (np.abs(vol_7d - vol_30d) < 0.3)
                volume_progression[steady] = 60
                
                # Distribution pattern: Volume decreasing (bearish for breakout)
                distribution = valid_vol & (vol_1d < vol_7d) & (vol_7d < vol_30d)
                volume_progression[distribution] = 30
                
                # Spike only: Just 1-day spike, no build-up (unreliable)
                spike_only = valid_vol & (vol_1d > 2) & (vol_7d < 1.2) & (vol_30d < 1.1)
                volume_progression[spike_only] = 40
                
                volume_component[valid_vol] = volume_progression[valid_vol]
        
        # Sub-component 2b: Volume dry-up detection (often precedes breakout)
        if 'rvol' in df.columns and 'from_high_pct' in df.columns:
            rvol = pd.Series(df['rvol'].values, index=df.index)
            from_high = pd.Series(df['from_high_pct'].values, index=df.index)
            
            # Volume dry-up near resistance = spring loading
            volume_dryup = (
                rvol.notna() & 
                from_high.notna() &
                (rvol < 0.7) &  # Low current volume
                (from_high > -10) & (from_high <= 0)  # Near high
            )
            if volume_dryup.any():
                # This is actually bullish - like a coiled spring
                volume_component[volume_dryup] = 65
        
        # Component 3: TREND ALIGNMENT (20% weight)
        # SMA structure with proper weighting
        trend_component = pd.Series(50, index=df.index, dtype=float)
        
        if 'price' in df.columns:
            price = pd.Series(df['price'].values, index=df.index)
            price_valid = price.notna() & (price > 0)
            
            if price_valid.any():
                trend_score = pd.Series(0, index=df.index, dtype=float)
                max_possible = pd.Series(0, index=df.index, dtype=float)
                
                # SMA hierarchy: 200 > 50 > 20 in importance for breakouts
                sma_weights = {
                    'sma_200d': 50,  # Most important - long-term trend
                    'sma_50d': 30,   # Medium-term trend
                    'sma_20d': 20    # Short-term trend
                }
                
                for sma_col, weight in sma_weights.items():
                    if sma_col in df.columns:
                        sma_values = pd.Series(df[sma_col].values, index=df.index)
                        valid_sma = price_valid & sma_values.notna() & (sma_values > 0)
                        
                        # Price above SMA
                        above_sma = valid_sma & (price > sma_values)
                        trend_score[above_sma] += weight
                        max_possible[valid_sma] += weight
                
                # Normalize to 0-100
                has_sma = max_possible > 0
                trend_component[has_sma] = (trend_score[has_sma] / max_possible[has_sma]) * 100
                
                # Golden alignment bonus: Price > SMA20 > SMA50 > SMA200
                if all(col in df.columns for col in ['sma_20d', 'sma_50d', 'sma_200d']):
                    sma_20 = pd.Series(df['sma_20d'].values, index=df.index)
                    sma_50 = pd.Series(df['sma_50d'].values, index=df.index)
                    sma_200 = pd.Series(df['sma_200d'].values, index=df.index)
                    
                    golden_alignment = (
                        price_valid & 
                        sma_20.notna() & sma_50.notna() & sma_200.notna() &
                        (price > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
                    )
                    if golden_alignment.any():
                        trend_component[golden_alignment] = 100  # Perfect score
        
        # Component 4: PRICE CONSOLIDATION (15% weight)
        # Tight consolidation near high = higher breakout probability
        consolidation_component = pd.Series(50, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'from_high_pct']):
            ret_7d = pd.Series(df['ret_7d'].values, index=df.index)
            ret_30d = pd.Series(df['ret_30d'].values, index=df.index)
            from_high = pd.Series(df['from_high_pct'].values, index=df.index)
            
            valid_consol = ret_7d.notna() & ret_30d.notna() & from_high.notna()
            
            if valid_consol.any():
                # Calculate volatility/range
                volatility = np.abs(ret_7d)
                
                # Tight consolidation near high (best setup)
                tight_consol = valid_consol & (volatility < 3) & (from_high > -10) & (from_high <= 0)
                consolidation_component[tight_consol] = 85
                
                # Normal consolidation
                normal_consol = valid_consol & (volatility >= 3) & (volatility < 7) & (from_high > -20)
                consolidation_component[normal_consol] = 60
                
                # Wide/volatile consolidation (less reliable)
                wide_consol = valid_consol & (volatility >= 7)
                consolidation_component[wide_consol] = 35
        
        # COMBINE ALL COMPONENTS (with fixed weights)
        # Calculate weighted average only where we have data
        components = {
            'distance': (distance_component, 0.35),
            'volume': (volume_component, 0.30),
            'trend': (trend_component, 0.20),
            'consolidation': (consolidation_component, 0.15)
        }
        
        # Vectorized combination
        weighted_sum = pd.Series(0, index=df.index, dtype=float)
        weight_sum = pd.Series(0, index=df.index, dtype=float)
        
        for name, (component, weight) in components.items():
            valid = component.notna() & (component != 50)  # 50 is default, means no data
            weighted_sum[valid] += component[valid] * weight
            weight_sum[valid] += weight
        
        # Calculate final score
        has_data = weight_sum > 0
        breakout_score[has_data] = weighted_sum[has_data] / weight_sum[has_data]
        
        # PATTERN RECOGNITION BONUSES
        # These are additive bonuses for specific bullish patterns
        
        # Pattern 1: Cup & Handle
        if all(col in df.columns for col in ['ret_3m', 'ret_1y', 'ret_30d', 'from_high_pct']):
            ret_3m = pd.Series(df['ret_3m'].values, index=df.index)
            ret_1y = pd.Series(df['ret_1y'].values, index=df.index)
            ret_30d = pd.Series(df['ret_30d'].values, index=df.index)
            from_high = pd.Series(df['from_high_pct'].values, index=df.index)
            
            cup_handle = (
                breakout_score.notna() &
                ret_1y.notna() & ret_3m.notna() & ret_30d.notna() & from_high.notna() &
                (ret_1y > 20) &  # Good yearly performance (left side of cup)
                (ret_3m < 10) &  # Consolidation (bottom of cup)
                (ret_30d > -5) & (ret_30d < 5) &  # Handle formation
                (from_high > -15) & (from_high <= 0)  # Near resistance
            )
            if cup_handle.any():
                breakout_score[cup_handle] += 15
                logger.debug(f"Cup & Handle pattern detected in {cup_handle.sum()} stocks")
        
        # Pattern 2: Flag Pattern (consolidation after strong move)
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'from_high_pct', 'rvol']):
            ret_30d = pd.Series(df['ret_30d'].values, index=df.index)
            ret_7d = pd.Series(df['ret_7d'].values, index=df.index)
            from_high = pd.Series(df['from_high_pct'].values, index=df.index)
            rvol = pd.Series(df['rvol'].values, index=df.index)
            
            flag_pattern = (
                breakout_score.notna() &
                ret_30d.notna() & ret_7d.notna() & from_high.notna() & rvol.notna() &
                (ret_30d > 15) &  # Strong prior move (pole)
                (ret_7d > -3) & (ret_7d < 3) &  # Tight consolidation (flag)
                (from_high > -8) & (from_high <= 0) &  # Near high
                (rvol < 1.5)  # Volume contraction during flag
            )
            if flag_pattern.any():
                breakout_score[flag_pattern] += 12
                logger.debug(f"Flag pattern detected in {flag_pattern.sum()} stocks")
        
        # Pattern 3: Ascending Triangle (higher lows, same highs)
        if all(col in df.columns for col in ['low_52w', 'high_52w', 'price', 'from_high_pct']):
            low_52w = pd.Series(df['low_52w'].values, index=df.index)
            high_52w = pd.Series(df['high_52w'].values, index=df.index)
            price = pd.Series(df['price'].values, index=df.index)
            from_high = pd.Series(df['from_high_pct'].values, index=df.index)
            
            # Check if lows are rising while testing same high
            valid_triangle = (
                low_52w.notna() & high_52w.notna() & 
                price.notna() & from_high.notna() & 
                (low_52w > 0) & (high_52w > 0)
            )
            
            if valid_triangle.any():
                # Price well above 52w low but struggling at high
                from_low_ratio = (price - low_52w) / low_52w * 100
                ascending_triangle = (
                    valid_triangle &
                    (from_low_ratio > 30) &  # Well above low
                    (from_high > -5) & (from_high <= 0) &  # Testing high repeatedly
                    breakout_score.notna()
                )
                if ascending_triangle.any():
                    breakout_score[ascending_triangle] += 10
                    logger.debug(f"Ascending triangle detected in {ascending_triangle.sum()} stocks")
        
        # Pattern 4: Volatility Contraction Pattern (VCP)
        if 'ret_3d' in df.columns and 'ret_7d' in df.columns and 'ret_30d' in df.columns:
            ret_3d = pd.Series(df['ret_3d'].values, index=df.index)
            ret_7d = pd.Series(df['ret_7d'].values, index=df.index)
            ret_30d = pd.Series(df['ret_30d'].values, index=df.index)
            
            # Volatility decreasing over time
            vol_3d = np.abs(ret_3d)
            vol_7d = np.abs(ret_7d)
            vol_30d = np.abs(ret_30d)
            
            vcp_pattern = (
                breakout_score.notna() &
                vol_3d.notna() & vol_7d.notna() & vol_30d.notna() &
                (vol_3d < vol_7d * 0.7) &  # Recent volatility much lower
                (vol_7d < vol_30d * 0.8) &  # Progressive contraction
                (vol_3d < 3)  # Very tight recent range
            )
            if vcp_pattern.any():
                breakout_score[vcp_pattern] += 8
                logger.debug(f"VCP pattern detected in {vcp_pattern.sum()} stocks")
        
        # CONTEXT ADJUSTMENTS
        
        # Market cap adjustment
        if 'category' in df.columns and breakout_score.notna().any():
            category = pd.Series(df['category'].values, index=df.index)
            
            # Large cap breakouts are more reliable
            large_cap = category.isin(['Large Cap', 'Mega Cap'])
            large_breakout = large_cap & breakout_score.notna() & (breakout_score > 70)
            if large_breakout.any():
                breakout_score[large_breakout] *= 1.05
                logger.debug(f"Large cap breakout bonus applied to {large_breakout.sum()} stocks")
            
            # Small cap breakouts need stronger confirmation
            small_cap = category.isin(['Small Cap', 'Micro Cap'])
            small_breakout = small_cap & breakout_score.notna() & (breakout_score > 60)
            if small_breakout.any():
                # Require volume confirmation for small caps
                if 'rvol' in df.columns:
                    rvol = pd.Series(df['rvol'].values, index=df.index)
                    needs_volume = small_breakout & (rvol < 1.5)
                    breakout_score[needs_volume] *= 0.85  # Penalty without volume
        
        # Sector momentum adjustment
        if 'sector' in df.columns and 'ret_30d' in df.columns:
            # If sector is strong, individual breakouts more likely
            sector_returns = df.groupby('sector')['ret_30d'].transform('mean')
            strong_sector = sector_returns > 10  # Sector up >10% in month
            
            sector_boost = strong_sector & breakout_score.notna() & (breakout_score > 60)
            if sector_boost.any():
                breakout_score[sector_boost] *= 1.03
                logger.debug(f"Strong sector bonus applied to {sector_boost.sum()} stocks")
        
        # Final clipping
        breakout_score = breakout_score.clip(0, 100)
        
        # Fill remaining NaN with default
        still_nan = breakout_score.isna()
        if still_nan.any():
            # Check if they have any price data at all
            has_price = 'price' in df.columns and df['price'].notna()
            breakout_score[still_nan & has_price] = 40  # Below average default
            breakout_score[still_nan & ~has_price] = np.nan  # Keep NaN if no data
        
        # COMPREHENSIVE LOGGING
        valid_scores = breakout_score.notna().sum()
        if valid_scores > 0:
            logger.info(f"Breakout scores calculated: {valid_scores} valid out of {len(df)} stocks")
            
            # Distribution
            score_dist = breakout_score[breakout_score.notna()]
            logger.info(f"Score distribution - Min: {score_dist.min():.1f}, "
                       f"Max: {score_dist.max():.1f}, "
                       f"Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}")
            
            # Breakout categories
            imminent = (breakout_score > 80).sum()
            probable = ((breakout_score > 65) & (breakout_score <= 80)).sum()
            possible = ((breakout_score > 50) & (breakout_score <= 65)).sum()
            unlikely = (breakout_score <= 50).sum()
            
            logger.debug(f"Breakout probability: Imminent={imminent}, Probable={probable}, "
                        f"Possible={possible}, Unlikely={unlikely}")
            
            # Pattern detection summary
            if 'from_high_pct' in df.columns:
                near_high = ((df['from_high_pct'] > -5) & (df['from_high_pct'] <= 0)).sum()
                above_high = (df['from_high_pct'] > 0).sum()
                logger.debug(f"Position: {near_high} near 52w high, {above_high} above high")
        
        return breakout_score
    
    @staticmethod
    def _calculate_rvol_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate RVOL score using smooth logarithmic scaling.
        FIXED: No category overrides, proper logarithmic curve, context-aware.
        
        Core Philosophy:
        - RVOL is exponential by nature - needs log scaling
        - NO discrete categories that create jumps
        - Higher volume should never score lower (no inversions!)
        - Context determines if high volume is good or bad
        
        Score Formula:
        - Below normal (0-1x): Linear scaling 0-50
        - Above normal (1x+): Logarithmic scaling 50-100
        - Context adjustments for price action and market cap
        
        Score Range:
        - 85-100: Extreme volume (major event/news)
        - 70-85: High volume (strong interest)
        - 50-70: Elevated volume (above average)
        - 30-50: Normal range (typical)
        - 10-30: Low volume (lack of interest)
        - 0-10: Dead volume (no activity)
        """
        # Initialize with NaN
        rvol_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        if 'rvol' not in df.columns:
            logger.warning("RVOL data not available")
            return rvol_score  # Return NaN, not default!
        
        rvol = df['rvol']
        valid_rvol = rvol.notna() & (rvol >= 0)
        
        if not valid_rvol.any():
            logger.warning("No valid RVOL data found")
            return rvol_score
        
        # BASE SCORE: Smooth continuous function (NO CATEGORIES!)
        
        # Handle zero volume separately
        zero_vol = valid_rvol & (rvol == 0)
        if zero_vol.any():
            rvol_score[zero_vol] = 0  # Dead stocks
        
        # Below normal volume (0 < RVOL < 1)
        below_normal = valid_rvol & (rvol > 0) & (rvol < 1)
        if below_normal.any():
            # Linear scaling: 0.1x = 5, 0.5x = 25, 0.9x = 45
            rvol_score[below_normal] = rvol[below_normal] * 50
        
        # Normal to elevated (RVOL >= 1)
        above_normal = valid_rvol & (rvol >= 1)
        if above_normal.any():
            # Logarithmic scaling for smooth progression
            # log(1) = 0 → 50
            # log(2) = 0.69 → 65
            # log(5) = 1.61 → 75
            # log(10) = 2.30 → 82
            # log(20) = 3.00 → 87
            # log(50) = 3.91 → 92
            
            # Using natural log with adjusted multiplier
            log_rvol = np.log(rvol[above_normal])
            
            # Smooth formula that never decreases
            rvol_score[above_normal] = 50 + 30 * (1 - np.exp(-log_rvol * 0.7))
            
            # Cap at 95 for extreme values
            rvol_score[above_normal] = rvol_score[above_normal].clip(50, 95)
        
        # CONTEXT LAYER 1: Price Action Harmony
        # Volume means different things with different price action
        if 'ret_1d' in df.columns and rvol_score.notna().any():
            ret_1d = df['ret_1d']
            valid_context = valid_rvol & ret_1d.notna() & rvol_score.notna()
            
            if valid_context.any():
                # High volume + big move = confirmation (good)
                strong_move = valid_context & (rvol > 2) & (ret_1d.abs() > 5)
                if strong_move.any():
                    rvol_score[strong_move] = np.minimum(rvol_score[strong_move] * 1.1, 100)
                
                # High volume + no move = distribution (bad)
                no_move = valid_context & (rvol > 2) & (ret_1d.abs() < 1)
                if no_move.any():
                    rvol_score[no_move] *= 0.8
                
                # Low volume + big move = suspicious
                suspicious = valid_context & (rvol < 0.5) & (ret_1d.abs() > 10)
                if suspicious.any():
                    rvol_score[suspicious] *= 0.7
        
        # CONTEXT LAYER 2: Sustained vs Spike
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
            vol_7d = df['vol_ratio_7d_90d']
            vol_30d = df['vol_ratio_30d_90d']
            
            valid_sustain = valid_rvol & vol_7d.notna() & vol_30d.notna() & rvol_score.notna()
            
            if valid_sustain.any():
                # Sustained elevation (building volume) = better
                sustained = valid_sustain & (rvol > 2) & (vol_7d > 1.5) & (vol_30d > 1.2)
                if sustained.any():
                    rvol_score[sustained] = np.minimum(rvol_score[sustained] * 1.05, 100)
                    logger.debug(f"Sustained volume bonus for {sustained.sum()} stocks")
                
                # Spike only (no build-up) = suspicious
                spike_only = valid_sustain & (rvol > 3) & (vol_7d < 1.2) & (vol_30d < 1.1)
                if spike_only.any():
                    rvol_score[spike_only] *= 0.85
                    logger.debug(f"Spike-only penalty for {spike_only.sum()} stocks")
        
        # CONTEXT LAYER 3: Market Cap Adjustments
        if 'category' in df.columns and rvol_score.notna().any():
            category = df['category']
            
            # Small/Micro caps: High RVOL is more common
            is_penny = category.isin(['Micro Cap', 'Small Cap'])
            penny_valid = is_penny & valid_rvol & rvol_score.notna()
            
            if penny_valid.any():
                # Progressive adjustment based on RVOL level
                # 2-3x is normal for penny stocks
                penny_moderate = penny_valid & (rvol > 2) & (rvol <= 3)
                if penny_moderate.any():
                    rvol_score[penny_moderate] *= 0.9
                
                # 3-5x is elevated but not unusual
                penny_elevated = penny_valid & (rvol > 3) & (rvol <= 5)
                if penny_elevated.any():
                    rvol_score[penny_elevated] *= 0.85
                
                # 5-10x needs scrutiny
                penny_high = penny_valid & (rvol > 5) & (rvol <= 10)
                if penny_high.any():
                    rvol_score[penny_high] *= 0.75
                
                # >10x is likely manipulation
                penny_extreme = penny_valid & (rvol > 10)
                if penny_extreme.any():
                    rvol_score[penny_extreme] = np.minimum(rvol_score[penny_extreme] * 0.6, 70)
                    logger.warning(f"Extreme RVOL in {penny_extreme.sum()} penny stocks")
            
            # Large/Mega caps: High RVOL is more significant
            is_large = category.isin(['Large Cap', 'Mega Cap'])
            large_valid = is_large & valid_rvol & rvol_score.notna()
            
            if large_valid.any():
                # Even moderate elevation is significant
                large_elevated = large_valid & (rvol > 1.5)
                if large_elevated.any():
                    rvol_score[large_elevated] = np.minimum(rvol_score[large_elevated] * 1.1, 100)
                
                # High RVOL in large caps = major event
                large_high = large_valid & (rvol > 3)
                if large_high.any():
                    rvol_score[large_high] = np.minimum(rvol_score[large_high] * 1.15, 100)
                    logger.info(f"Significant RVOL in {large_high.sum()} large cap stocks")
        
        # CONTEXT LAYER 4: Position in Range
        # Volume at extremes has different meaning
        if 'from_high_pct' in df.columns and rvol_score.notna().any():
            from_high = df['from_high_pct']
            valid_position = valid_rvol & from_high.notna() & rvol_score.notna()
            
            if valid_position.any():
                # High volume near 52w high = breakout attempt
                near_high = valid_position & (from_high > -5) & (from_high <= 0) & (rvol > 2)
                if near_high.any():
                    rvol_score[near_high] = np.minimum(rvol_score[near_high] + 5, 100)
                
                # High volume near 52w low = potential reversal
                if 'from_low_pct' in df.columns:
                    from_low = df['from_low_pct']
                    near_low = valid_position & from_low.notna() & (from_low < 10) & (rvol > 2)
                    if near_low.any():
                        rvol_score[near_low] = np.minimum(rvol_score[near_low] + 3, 100)
        
        # Final clipping
        rvol_score = rvol_score.clip(0, 100)
        
        # DO NOT FILL NaN!
        
        # COMPREHENSIVE LOGGING
        valid_scores = rvol_score.notna().sum()
        total_stocks = len(df)
        
        logger.info(f"RVOL scores: {valid_scores}/{total_stocks} calculated")
        
        if valid_scores > 0:
            score_dist = rvol_score[rvol_score.notna()]
            logger.info(f"Distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # RVOL distribution
            if valid_rvol.any():
                rvol_values = rvol[valid_rvol]
                dead = (rvol_values == 0).sum()
                low = ((rvol_values > 0) & (rvol_values < 0.5)).sum()
                below_avg = ((rvol_values >= 0.5) & (rvol_values < 1)).sum()
                normal = ((rvol_values >= 1) & (rvol_values < 2)).sum()
                elevated = ((rvol_values >= 2) & (rvol_values < 5)).sum()
                high = ((rvol_values >= 5) & (rvol_values < 10)).sum()
                extreme = (rvol_values >= 10).sum()
                
                logger.debug(f"RVOL ranges: Dead={dead}, Low={low}, Below={below_avg}, "
                            f"Normal={normal}, Elevated={elevated}, High={high}, Extreme={extreme}")
                
                if rvol_values.median() > 1.5:
                    logger.info(f"Market-wide elevated volume (median RVOL: {rvol_values.median():.2f})")
        
        return rvol_score
        
    @staticmethod
    def _calculate_trend_quality(df: pd.DataFrame) -> pd.Series:
        """
        Calculate trend quality based on proper SMA hierarchy and alignment.
        FIXED: Correct SMA weights, no overrides, proper cross detection.
        
        Core Philosophy:
        - SMA200 > SMA50 > SMA20 in importance (not reversed!)
        - Trend quality = alignment + consistency + strength
        - No arbitrary overrides that defeat calculations
        - Golden/Death cross needs actual crossover, not just proximity
        
        Components:
        - 50% SMA Alignment (200=50%, 50=30%, 20=20% of this)
        - 25% Trend Consistency (how well SMAs are ordered)
        - 15% Trend Strength (separation between SMAs)
        - 10% Special Patterns (golden cross, squeeze, etc.)
        
        Score Range:
        - 85-100: Perfect bullish alignment
        - 70-85: Strong bullish trend
        - 50-70: Mild bullish/neutral
        - 30-50: Weak/mixed trend
        - 15-30: Bearish trend
        - 0-15: Strong bearish alignment
        """
        # Initialize with NaN
        trend_quality = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Check minimum requirements
        if 'price' not in df.columns:
            logger.warning("No price data for trend quality calculation")
            return trend_quality  # Return NaN, not default!
        
        price = df['price']
        price_valid = price.notna() & (price > 0)
        
        if not price_valid.any():
            logger.warning("No valid price data for trend quality")
            return trend_quality
        
        # COMPONENT 1: SMA ALIGNMENT (50% weight)
        # FIXED: Proper hierarchy - SMA200 most important!
        alignment_score = pd.Series(0, index=df.index, dtype=float)
        alignment_max = pd.Series(0, index=df.index, dtype=float)
        
        # SMA200 - PRIMARY TREND (50% of alignment score)
        if 'sma_200d' in df.columns:
            sma_200 = df['sma_200d']
            valid_200 = price_valid & sma_200.notna() & (sma_200 > 0)
            
            if valid_200.any():
                # Above SMA200 = bullish primary trend
                above_200 = valid_200 & (price > sma_200)
                alignment_score[above_200] += 50
                alignment_max[valid_200] += 50
                
                # Distance bonus/penalty (up to ±10 points)
                distance_200 = ((price - sma_200) / sma_200 * 100).clip(-20, 20)
                alignment_score[valid_200] += distance_200[valid_200] * 0.5
        
        # SMA50 - SECONDARY TREND (30% of alignment score)
        if 'sma_50d' in df.columns:
            sma_50 = df['sma_50d']
            valid_50 = price_valid & sma_50.notna() & (sma_50 > 0)
            
            if valid_50.any():
                above_50 = valid_50 & (price > sma_50)
                alignment_score[above_50] += 30
                alignment_max[valid_50] += 30
                
                # Distance bonus/penalty
                distance_50 = ((price - sma_50) / sma_50 * 100).clip(-15, 15)
                alignment_score[valid_50] += distance_50[valid_50] * 0.3
        
        # SMA20 - MINOR TREND (20% of alignment score)
        if 'sma_20d' in df.columns:
            sma_20 = df['sma_20d']
            valid_20 = price_valid & sma_20.notna() & (sma_20 > 0)
            
            if valid_20.any():
                above_20 = valid_20 & (price > sma_20)
                alignment_score[above_20] += 20
                alignment_max[valid_20] += 20
                
                # Distance bonus/penalty
                distance_20 = ((price - sma_20) / sma_20 * 100).clip(-10, 10)
                alignment_score[valid_20] += distance_20[valid_20] * 0.2
        
        # Normalize alignment score
        has_alignment = alignment_max > 0
        alignment_component = pd.Series(50, index=df.index, dtype=float)
        if has_alignment.any():
            # Convert to 0-100 scale
            alignment_component[has_alignment] = (
                (alignment_score[has_alignment] / alignment_max[has_alignment]) * 100
            ).clip(0, 100)
        
        # COMPONENT 2: TREND CONSISTENCY (25% weight)
        # How well are SMAs ordered?
        consistency_component = pd.Series(50, index=df.index, dtype=float)
        
        # Check if we have multiple SMAs
        sma_count = sum([
            'sma_20d' in df.columns,
            'sma_50d' in df.columns,
            'sma_200d' in df.columns
        ])
        
        if sma_count >= 2:
            # Perfect bullish: Price > SMA20 > SMA50 > SMA200
            if all(col in df.columns for col in ['sma_20d', 'sma_50d', 'sma_200d']):
                sma_20 = df['sma_20d']
                sma_50 = df['sma_50d']
                sma_200 = df['sma_200d']
                
                all_valid = (
                    price_valid & 
                    sma_20.notna() & sma_50.notna() & sma_200.notna() &
                    (sma_20 > 0) & (sma_50 > 0) & (sma_200 > 0)
                )
                
                if all_valid.any():
                    # Perfect bullish alignment
                    perfect_bull = all_valid & (price > sma_20) & (sma_20 > sma_50) & (sma_50 > sma_200)
                    consistency_component[perfect_bull] = 100
                    
                    # Good bullish (price above all, but SMAs mixed)
                    good_bull = all_valid & (price > sma_20) & (price > sma_50) & (price > sma_200) & ~perfect_bull
                    consistency_component[good_bull] = 75
                    
                    # Mixed (price between SMAs)
                    mixed = all_valid & ~perfect_bull & ~good_bull
                    price_above_count = (
                        (price > sma_20).astype(int) +
                        (price > sma_50).astype(int) +
                        (price > sma_200).astype(int)
                    )
                    consistency_component[mixed] = 25 + (price_above_count[mixed] * 16.67)
                    
                    # Perfect bearish alignment
                    perfect_bear = all_valid & (price < sma_20) & (sma_20 < sma_50) & (sma_50 < sma_200)
                    consistency_component[perfect_bear] = 0
            
            # Two SMAs available
            elif sma_count == 2:
                if 'sma_50d' in df.columns and 'sma_200d' in df.columns:
                    sma_50 = df['sma_50d']
                    sma_200 = df['sma_200d']
                    
                    valid_two = (
                        price_valid & 
                        sma_50.notna() & sma_200.notna() &
                        (sma_50 > 0) & (sma_200 > 0)
                    )
                    
                    if valid_two.any():
                        bullish_two = valid_two & (price > sma_50) & (sma_50 > sma_200)
                        consistency_component[bullish_two] = 80
                        
                        bearish_two = valid_two & (price < sma_50) & (sma_50 < sma_200)
                        consistency_component[bearish_two] = 20
        
        # COMPONENT 3: TREND STRENGTH (15% weight)
        # Separation between SMAs indicates trend strength
        strength_component = pd.Series(50, index=df.index, dtype=float)
        
        if sma_count >= 2:
            separation_scores = []
            
            if 'sma_50d' in df.columns and 'sma_200d' in df.columns:
                sma_50 = df['sma_50d']
                sma_200 = df['sma_200d']
                
                valid = sma_50.notna() & sma_200.notna() & (sma_200 > 0)
                if valid.any():
                    # Calculate % separation
                    separation = ((sma_50 - sma_200) / sma_200 * 100).clip(-30, 30)
                    
                    # Positive separation (50 > 200) = bullish strength
                    # Negative separation (50 < 200) = bearish strength
                    # Convert to 0-100 score
                    sep_score = pd.Series(50, index=df.index)
                    sep_score[valid] = 50 + separation[valid] * 1.67
                    separation_scores.append(sep_score)
            
            if separation_scores:
                strength_component = pd.concat(separation_scores, axis=1).mean(axis=1)
        
        # COMPONENT 4: SPECIAL PATTERNS (10% weight)
        pattern_component = pd.Series(50, index=df.index, dtype=float)
        
        # Golden/Death Cross Detection (FIXED)
        if 'sma_50d' in df.columns and 'sma_200d' in df.columns:
            sma_50 = df['sma_50d']
            sma_200 = df['sma_200d']
            
            valid_cross = sma_50.notna() & sma_200.notna() & (sma_50 > 0) & (sma_200 > 0)
            
            if valid_cross.any():
                # Golden cross: SMA50 above SMA200 and close together (recent cross)
                sma_50_above = sma_50 > sma_200
                proximity = (np.abs(sma_50 - sma_200) / sma_200) < 0.03  # Within 3%
                
                # Need momentum confirmation for actual cross
                if 'ret_30d' in df.columns:
                    ret_30d = df['ret_30d']
                    
                    # Golden cross: 50 > 200, close together, positive momentum
                    golden_cross = valid_cross & sma_50_above & proximity & (ret_30d > 5)
                    pattern_component[golden_cross] = 80
                    
                    # Death cross: 50 < 200, close together, negative momentum
                    death_cross = valid_cross & ~sma_50_above & proximity & (ret_30d < -5)
                    pattern_component[death_cross] = 20
                else:
                    # Without momentum, just use position
                    potential_golden = valid_cross & sma_50_above & proximity
                    pattern_component[potential_golden] = 70
                    
                    potential_death = valid_cross & ~sma_50_above & proximity
                    pattern_component[potential_death] = 30
        
        # SMA Squeeze Pattern (volatility contraction)
        if all(col in df.columns for col in ['sma_20d', 'sma_50d']):
            sma_20 = df['sma_20d']
            sma_50 = df['sma_50d']
            
            valid_squeeze = sma_20.notna() & sma_50.notna() & (sma_50 > 0)
            if valid_squeeze.any():
                # SMAs converging = potential breakout
                convergence = (np.abs(sma_20 - sma_50) / sma_50) < 0.02  # Within 2%
                pattern_component[convergence] = 60  # Neutral with potential
        
        # COMBINE COMPONENTS
        components = [
            (alignment_component, 0.50),
            (consistency_component, 0.25),
            (strength_component, 0.15),
            (pattern_component, 0.10)
        ]
        
        # Calculate weighted average
        for idx in df.index:
            valid_components = []
            valid_weights = []
            
            for component, weight in components:
                if pd.notna(component[idx]) and component[idx] != 50:  # 50 is default
                    valid_components.append(component[idx])
                    valid_weights.append(weight)
            
            if valid_components:
                total_weight = sum(valid_weights)
                if total_weight > 0:
                    normalized_weights = [w/total_weight for w in valid_weights]
                    trend_quality[idx] = sum(c * w for c, w in zip(valid_components, normalized_weights))
        
        # NO OVERRIDES! Let the calculation stand
        
        # Final clipping
        trend_quality = trend_quality.clip(0, 100)
        
        # COMPREHENSIVE LOGGING
        valid_scores = trend_quality.notna().sum()
        total_stocks = len(df)
        
        logger.info(f"Trend quality scores: {valid_scores}/{total_stocks} calculated")
        
        if valid_scores > 0:
            score_dist = trend_quality[trend_quality.notna()]
            logger.info(f"Distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Trend categories
            strong_bull = (trend_quality > 85).sum()
            bull = ((trend_quality > 70) & (trend_quality <= 85)).sum()
            neutral = ((trend_quality >= 30) & (trend_quality <= 70)).sum()
            bear = ((trend_quality >= 15) & (trend_quality < 30)).sum()
            strong_bear = (trend_quality < 15).sum()
            
            logger.debug(f"Trend breakdown: Strong Bull={strong_bull}, Bull={bull}, "
                        f"Neutral={neutral}, Bear={bear}, Strong Bear={strong_bear}")
        
        return trend_quality
    
    @staticmethod
    def _calculate_long_term_strength(df: pd.DataFrame) -> pd.Series:
        """
        Calculate long-term strength based on multi-year performance.
        GUARANTEED to return a Series of same length as input DataFrame.
        
        Core Philosophy:
        - ALWAYS returns pd.Series with same index as df
        - Fully vectorized (no loops)
        - Only scores what can be measured (NaN for insufficient data)
        - Long-term = 1y, 3y, 5y ONLY
        - No penalties, just honest assessment
        
        Score Components:
        - 1-year: 20% weight (when available)
        - 3-year: 35% weight (when available)
        - 5-year: 45% weight (when available)
        - Weights auto-normalize based on available data
        """
        # CRITICAL: Initialize with correct index and length
        long_term_strength = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Early return if empty DataFrame
        if df.empty:
            logger.warning("Empty DataFrame provided to calculate_long_term_strength")
            return long_term_strength
        
        # Define periods and weights
        periods = {
            'ret_1y': 0.20,
            'ret_3y': 0.35,
            'ret_5y': 0.45
        }
        
        # Check which columns exist
        available_periods = {}
        for period, weight in periods.items():
            if period in df.columns:
                available_periods[period] = weight
        
        # If no long-term data columns exist at all
        if not available_periods:
            logger.warning("No long-term return columns (ret_1y, ret_3y, ret_5y) found in DataFrame")
            # Return Series of NaN (not error, just no data)
            return long_term_strength
        
        # VECTORIZED CALCULATION
        # Collect scores for each period
        period_scores = pd.DataFrame(index=df.index)
        period_weights = pd.DataFrame(index=df.index)
        
        for period, base_weight in available_periods.items():
            # Get return data
            returns = pd.Series(df[period].values, index=df.index)
            valid = returns.notna()
            
            if not valid.any():
                continue
            
            # Calculate annualized return based on period
            years = float(period.replace('ret_', '').replace('y', ''))
            
            # Initialize score for this period
            score = pd.Series(np.nan, index=df.index)
            
            # Calculate CAGR for valid returns
            # Handle negative returns properly
            positive_final = valid & (returns > -100)
            if positive_final.any():
                # CAGR = ((1 + return/100)^(1/years) - 1) * 100
                cagr = ((1 + returns[positive_final]/100) ** (1/years) - 1) * 100
                
                # Convert CAGR to score (0-100)
                # Indian market context:
                # <0% → 20-50
                # 0-15% → 50-70  
                # 15-25% → 70-85
                # >25% → 85-100
                score[positive_final] = np.where(
                    cagr < 0,
                    np.maximum(20, 50 + cagr * 1.5),  # Negative: scale from 20 to 50
                    np.where(
                        cagr <= 15,
                        50 + (cagr / 15) * 20,  # 0-15%: scale from 50 to 70
                        np.where(
                            cagr <= 25,
                            70 + ((cagr - 15) / 10) * 15,  # 15-25%: scale from 70 to 85
                            np.minimum(100, 85 + (cagr - 25) * 0.5)  # >25%: scale from 85 to 100
                        )
                    )
                )
            
            # Handle total loss cases
            total_loss = valid & (returns <= -100)
            if total_loss.any():
                score[total_loss] = 0
            
            # Store score and weight for this period
            period_scores[period] = score
            period_weights[period] = pd.Series(
                np.where(score.notna(), base_weight, 0),
                index=df.index
            )
        
        # Calculate weighted average where we have data
        if not period_scores.empty:
            # Sum weighted scores
            weighted_sum = pd.Series(0, index=df.index, dtype=float)
            weight_sum = pd.Series(0, index=df.index, dtype=float)
            
            for period in period_scores.columns:
                valid = period_scores[period].notna()
                weighted_sum[valid] += period_scores[period][valid] * period_weights[period][valid]
                weight_sum[valid] += period_weights[period][valid]
            
            # Calculate final score where weights > 0
            has_score = weight_sum > 0
            long_term_strength[has_score] = weighted_sum[has_score] / weight_sum[has_score]
        
        # ADD CONSISTENCY BONUS (if multiple periods available)
        if len(period_scores.columns) >= 2:
            # Check for consistency across periods
            scores_array = period_scores.values
            valid_rows = ~np.isnan(scores_array).all(axis=1)
            
            if valid_rows.any():
                # Count how many periods have data for each stock
                periods_with_data = (~np.isnan(scores_array)).sum(axis=1)
                
                # Calculate standard deviation across periods
                score_std = np.nanstd(scores_array, axis=1)
                score_mean = np.nanmean(scores_array, axis=1)
                
                # Low std relative to mean = consistent
                # Add small bonus for consistency
                consistent = valid_rows & (periods_with_data >= 2) & (score_std < 10) & (score_mean > 50)
                if consistent.any():
                    long_term_strength[consistent] = np.minimum(
                        long_term_strength[consistent] * 1.05, 
                        100
                    )
                
                # High std = inconsistent (volatile performance)
                inconsistent = valid_rows & (periods_with_data >= 2) & (score_std > 20)
                if inconsistent.any():
                    long_term_strength[inconsistent] *= 0.95
        
        # SPECIAL CASES
        # Check for improvement pattern if all 3 periods available
        if all(period in period_scores.columns for period in ['ret_1y', 'ret_3y', 'ret_5y']):
            ret_1y = pd.Series(df['ret_1y'].values, index=df.index)
            ret_3y = pd.Series(df['ret_3y'].values, index=df.index)
            ret_5y = pd.Series(df['ret_5y'].values, index=df.index)
            
            all_valid = ret_1y.notna() & ret_3y.notna() & ret_5y.notna()
            
            if all_valid.any():
                # Annualized returns
                ann_1y = ret_1y[all_valid]
                ann_3y = ret_3y[all_valid] / 3
                ann_5y = ret_5y[all_valid] / 5
                
                # Accelerating growth pattern
                accelerating = all_valid & (ann_1y > ann_3y * 1.1) & (ann_3y > ann_5y * 1.1)
                if accelerating.any():
                    long_term_strength[accelerating] = np.minimum(
                        long_term_strength[accelerating] + 5, 
                        100
                    )
                
                # Decelerating pattern
                decelerating = all_valid & (ann_1y < ann_3y * 0.9) & (ann_3y < ann_5y * 0.9)
                if decelerating.any():
                    long_term_strength[decelerating] = np.maximum(
                        long_term_strength[decelerating] - 5,
                        0
                    )
        
        # MINIMUM DATA REQUIREMENT
        # Require at least 1 period with data for a score
        # This is already handled by the weight_sum > 0 check above
        
        # Final clipping to ensure valid range
        long_term_strength = long_term_strength.clip(0, 100)
        
        # DO NOT FILL NaN!
        # Stocks without sufficient data remain NaN
        # This is intentional and correct
        
        # VERIFICATION - Critical for debugging
        assert len(long_term_strength) == len(df), f"Length mismatch: {len(long_term_strength)} vs {len(df)}"
        assert long_term_strength.index.equals(df.index), "Index mismatch!"
        
        # LOGGING
        total_stocks = len(df)
        scored_stocks = long_term_strength.notna().sum()
        
        logger.info(f"Long-term strength: {scored_stocks}/{total_stocks} stocks scored")
        
        if scored_stocks > 0:
            score_dist = long_term_strength[long_term_strength.notna()]
            logger.info(f"Score distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Check data availability
            for period in ['ret_1y', 'ret_3y', 'ret_5y']:
                if period in df.columns:
                    available = df[period].notna().sum()
                    pct = (available / total_stocks) * 100
                    logger.debug(f"{period}: {available}/{total_stocks} ({pct:.1f}%) stocks have data")
            
            # Distribution categories
            excellent = (long_term_strength > 80).sum()
            good = ((long_term_strength > 65) & (long_term_strength <= 80)).sum()
            average = ((long_term_strength > 50) & (long_term_strength <= 65)).sum()
            poor = (long_term_strength <= 50).sum()
            
            if excellent > 0 or poor > 0:
                logger.debug(f"Performance breakdown: Excellent={excellent}, Good={good}, "
                            f"Average={average}, Poor={poor}")
        else:
            logger.warning("No stocks had sufficient data for long-term strength scoring")
        
        # GUARANTEED RETURN
        # Always return Series of same length as input
        return long_term_strength
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """
        Calculate true liquidity score based on turnover value and tradability.
        FIXED: Uses turnover (volume × price), float ratios, proper scaling.
        
        Liquidity Philosophy:
        - Liquidity = Ability to trade without impacting price
        - Turnover value matters more than share volume
        - Consistency across timeframes is crucial
        - Float-adjusted metrics show true liquidity
        - Market cap relative thresholds
        
        Score Components:
        - 40% Turnover value (volume × price)
        - 25% Turnover ratio (turnover / market cap)
        - 20% Volume consistency
        - 15% Trading frequency (active days)
        
        Score Interpretation:
        - 85-100: Highly liquid (institutional grade)
        - 70-85: Good liquidity (easily tradable)
        - 50-70: Moderate liquidity (retail friendly)
        - 30-50: Low liquidity (caution needed)
        - 0-30: Illiquid (difficult to trade)
        """
        liquidity_score = pd.Series(np.nan, index=df.index, dtype=float)
        
        # Component 1: TURNOVER VALUE (40% weight)
        # Actual money traded, not just shares
        turnover_component = pd.Series(np.nan, index=df.index, dtype=float)
        
        if 'volume_1d' in df.columns and 'price' in df.columns:
            volume_1d = pd.Series(df['volume_1d'].values, index=df.index)
            price = pd.Series(df['price'].values, index=df.index)
            
            valid_turnover = volume_1d.notna() & price.notna() & (volume_1d >= 0) & (price > 0)
            
            if valid_turnover.any():
                # Calculate daily turnover in currency (millions for scaling)
                # Assuming price is in rupees and volume in shares
                turnover = (volume_1d[valid_turnover] * price[valid_turnover]) / 1_000_000  # In millions
                
                # Log scale for turnover (spans many orders of magnitude)
                # Avoid log(0) with small addition
                log_turnover = np.log10(turnover + 0.001)
                
                # Dynamic scaling based on actual data range
                # Don't assume arbitrary ranges
                log_min = log_turnover.min()
                log_max = log_turnover.max()
                log_range = log_max - log_min
                
                if log_range > 0:
                    # Normalize to 0-100 based on actual range
                    turnover_component[valid_turnover] = ((log_turnover - log_min) / log_range) * 100
                else:
                    # All same value
                    turnover_component[valid_turnover] = 50
                
                # Apply market context thresholds
                # Indian market context (in millions INR daily turnover)
                # <0.1M = 20, 0.1-1M = 20-40, 1-10M = 40-60, 10-100M = 60-80, >100M = 80-100
                turnover_component[valid_turnover] = np.where(
                    turnover < 0.1, 20,
                    np.where(
                        turnover < 1, 20 + (np.log10(turnover + 0.1) + 1) * 20,
                        np.where(
                            turnover < 10, 40 + np.log10(turnover) * 20,
                            np.where(
                                turnover < 100, 60 + np.log10(turnover/10) * 20,
                                np.where(
                                    turnover < 1000, 80 + np.log10(turnover/100) * 10,
                                    95  # Cap very high turnover
                                )
                            )
                        )
                    )
                )
        
        # Component 2: TURNOVER RATIO (25% weight)
        # Turnover as % of market cap (velocity)
        velocity_component = pd.Series(np.nan, index=df.index, dtype=float)
        
        if all(col in df.columns for col in ['volume_1d', 'price', 'market_cap']):
            volume_1d = pd.Series(df['volume_1d'].values, index=df.index)
            price = pd.Series(df['price'].values, index=df.index)
            market_cap = pd.Series(df['market_cap'].values, index=df.index)
            
            # Parse market cap (could be string like "1.5B")
            market_cap_numeric = pd.Series(np.nan, index=df.index)
            for idx in df.index:
                mc_val = market_cap[idx]
                if pd.notna(mc_val):
                    if isinstance(mc_val, str):
                        # Parse strings like "1.5B", "500M", "75K"
                        try:
                            if 'T' in str(mc_val):
                                market_cap_numeric[idx] = float(mc_val.replace('T', '').replace(',', '')) * 1_000_000
                            elif 'B' in str(mc_val):
                                market_cap_numeric[idx] = float(mc_val.replace('B', '').replace(',', '')) * 1_000
                            elif 'M' in str(mc_val):
                                market_cap_numeric[idx] = float(mc_val.replace('M', '').replace(',', ''))
                            elif 'K' in str(mc_val):
                                market_cap_numeric[idx] = float(mc_val.replace('K', '').replace(',', '')) / 1_000
                            else:
                                market_cap_numeric[idx] = float(str(mc_val).replace(',', ''))
                        except (ValueError, TypeError, AttributeError) as e:
                            logger.debug(f"Failed to parse market cap value {mc_val}: {e}")
                            market_cap_numeric[idx] = np.nan
                    else:
                        market_cap_numeric[idx] = float(mc_val)
            
            valid_velocity = (
                volume_1d.notna() & price.notna() & market_cap_numeric.notna() &
                (volume_1d >= 0) & (price > 0) & (market_cap_numeric > 0)
            )
            
            if valid_velocity.any():
                # Daily turnover as % of market cap
                daily_turnover = (volume_1d[valid_velocity] * price[valid_velocity]) / 1_000_000
                turnover_ratio = (daily_turnover / market_cap_numeric[valid_velocity]) * 100
                
                # Score based on turnover ratio
                # 0-0.1% = 30, 0.1-0.5% = 30-60, 0.5-2% = 60-80, >2% = 80-100
                velocity_component[valid_velocity] = np.where(
                    turnover_ratio < 0.1, 30 + turnover_ratio * 300,
                    np.where(
                        turnover_ratio < 0.5, 30 + (turnover_ratio - 0.1) * 75,
                        np.where(
                            turnover_ratio < 2, 60 + (turnover_ratio - 0.5) * 13.33,
                            np.where(
                                turnover_ratio < 5, 80 + (turnover_ratio - 2) * 5,
                                95  # Cap at very high ratios (might be manipulation)
                            )
                        )
                    )
                )
        
        # Component 3: VOLUME CONSISTENCY (20% weight)
        # How stable is the liquidity across time
        consistency_component = pd.Series(np.nan, index=df.index, dtype=float)
        
        vol_columns = ['volume_1d', 'volume_7d', 'volume_30d', 'volume_90d']
        available_vols = [col for col in vol_columns if col in df.columns]
        
        if len(available_vols) >= 2:
            # Collect volume data
            vol_data = pd.DataFrame()
            for col in available_vols:
                vol_data[col] = pd.Series(df[col].values, index=df.index)
            
            # Calculate coefficient of variation for each stock
            valid_rows = vol_data.notna().all(axis=1)
            
            if valid_rows.any():
                # Normalize by average to get daily equivalents
                vol_normalized = vol_data.copy()
                if 'volume_7d' in vol_normalized.columns:
                    vol_normalized['volume_7d'] = vol_normalized['volume_7d'] / 7
                if 'volume_30d' in vol_normalized.columns:
                    vol_normalized['volume_30d'] = vol_normalized['volume_30d'] / 30
                if 'volume_90d' in vol_normalized.columns:
                    vol_normalized['volume_90d'] = vol_normalized['volume_90d'] / 90
                
                # Calculate consistency
                vol_mean = vol_normalized[valid_rows].mean(axis=1)
                vol_std = vol_normalized[valid_rows].std(axis=1)
                
                # Avoid division by zero
                vol_cv = pd.Series(np.nan, index=df.index)
                non_zero = valid_rows & (vol_mean > 0)
                vol_cv[non_zero] = vol_std[non_zero] / vol_mean[non_zero]
                
                # Score based on consistency (lower CV = higher score)
                # CV < 0.3 = 80, 0.3-0.6 = 60-80, 0.6-1.0 = 40-60, >1.0 = 20-40
                consistency_component[non_zero] = np.where(
                    vol_cv[non_zero] < 0.3, 80,
                    np.where(
                        vol_cv[non_zero] < 0.6, 80 - (vol_cv[non_zero] - 0.3) * 66.67,
                        np.where(
                            vol_cv[non_zero] < 1.0, 60 - (vol_cv[non_zero] - 0.6) * 50,
                            np.maximum(20, 40 - (vol_cv[non_zero] - 1.0) * 10)
                        )
                    )
                )
        
        # Component 4: TRADING FREQUENCY (15% weight)
        # How often does it trade (vs halted/inactive days)
        frequency_component = pd.Series(np.nan, index=df.index, dtype=float)
        
        if 'volume_30d' in df.columns and 'volume_1d' in df.columns:
            volume_30d = pd.Series(df['volume_30d'].values, index=df.index)
            volume_1d = pd.Series(df['volume_1d'].values, index=df.index)
            
            valid_freq = volume_30d.notna() & volume_1d.notna() & (volume_30d >= 0)
            
            if valid_freq.any():
                # Estimate active trading days
                # If daily average equals total/30, then trades every day
                # If daily volume >> average, then sporadic trading
                avg_daily = volume_30d[valid_freq] / 30
                
                # Avoid division by zero
                safe_avg = avg_daily.copy()
                safe_avg[safe_avg == 0] = 1
                
                # Ratio indicates consistency
                ratio = volume_1d[valid_freq] / safe_avg
                
                # Score based on trading regularity
                # Ratio near 1 = regular (good)
                # Ratio >> 1 = sporadic (bad)
                # Ratio << 1 = declining (bad)
                frequency_component[valid_freq] = np.where(
                    ratio < 0.5, 40,  # Much lower than average
                    np.where(
                        ratio < 0.8, 50,  # Somewhat lower
                        np.where(
                            ratio < 1.5, 70,  # Normal range
                            np.where(
                                ratio < 3, 60,  # Somewhat sporadic
                                40  # Very sporadic
                            )
                        )
                    )
                )
                
                # Bonus for very consistent trading
                very_consistent = valid_freq & (ratio > 0.8) & (ratio < 1.2)
                frequency_component[very_consistent] = 85
        
        # COMBINE COMPONENTS
        components = {
            'turnover': (turnover_component, 0.40),
            'velocity': (velocity_component, 0.25),
            'consistency': (consistency_component, 0.20),
            'frequency': (frequency_component, 0.15)
        }
        
        # Calculate weighted average
        for idx in df.index:
            component_scores = []
            component_weights = []
            
            for name, (score_series, weight) in components.items():
                if pd.notna(score_series[idx]):
                    component_scores.append(score_series[idx])
                    component_weights.append(weight)
            
            if component_scores:
                # Normalize weights
                total_weight = sum(component_weights)
                normalized_weights = [w/total_weight for w in component_weights]
                
                # Calculate weighted score
                liquidity_score[idx] = sum(
                    score * weight 
                    for score, weight in zip(component_scores, normalized_weights)
                )
        
        # MARKET CAP ADJUSTMENTS
        # Different liquidity expectations for different caps
        if 'category' in df.columns and liquidity_score.notna().any():
            category = pd.Series(df['category'].values, index=df.index)
            
            # Large/Mega caps
            is_large = category.isin(['Large Cap', 'Mega Cap'])
            large_mask = is_large & liquidity_score.notna()
            
            if large_mask.any():
                # Large caps should have high liquidity
                # Penalize if low, bonus if high
                liquidity_score[large_mask] = np.where(
                    liquidity_score[large_mask] < 50,
                    liquidity_score[large_mask] * 0.8,  # Penalty for illiquid large cap
                    np.minimum(liquidity_score[large_mask] * 1.1, 100)  # Bonus for liquid
                )
            
            # Small/Micro caps
            is_small = category.isin(['Small Cap', 'Micro Cap'])
            small_mask = is_small & liquidity_score.notna()
            
            if small_mask.any():
                # Small caps naturally less liquid - adjust expectations
                # Boost scores slightly to normalize
                liquidity_score[small_mask] = np.minimum(
                    liquidity_score[small_mask] + 10, 
                    100
                )
        
        # Clip final scores
        liquidity_score = liquidity_score.clip(0, 100)
        
        # NO DEFAULT FILLING
        # Stocks without data remain NaN
        
        # LOGGING
        valid_scores = liquidity_score.notna().sum()
        if valid_scores > 0:
            logger.info(f"Liquidity scores calculated: {valid_scores}/{len(df)} stocks")
            
            score_dist = liquidity_score[liquidity_score.notna()]
            logger.info(f"Liquidity distribution - Mean: {score_dist.mean():.1f}, "
                       f"Median: {score_dist.median():.1f}, "
                       f"Std: {score_dist.std():.1f}")
            
            # Liquidity categories
            highly_liquid = (liquidity_score > 85).sum()
            good_liquid = ((liquidity_score > 70) & (liquidity_score <= 85)).sum()
            moderate = ((liquidity_score > 50) & (liquidity_score <= 70)).sum()
            low = ((liquidity_score > 30) & (liquidity_score <= 50)).sum()
            illiquid = (liquidity_score <= 30).sum()
            
            logger.debug(f"Liquidity breakdown: Highly={highly_liquid}, Good={good_liquid}, "
                        f"Moderate={moderate}, Low={low}, Illiquid={illiquid}")
            
            # Check if turnover data was available
            if turnover_component.notna().any():
                logger.debug(f"Turnover data available for {turnover_component.notna().sum()} stocks")
            else:
                logger.warning("No turnover data available - liquidity scores may be incomplete")
        
        return liquidity_score
        
    @staticmethod
    def _apply_smart_bonuses(df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smart bonuses based on confirmed exceptional patterns.
        FIXED: All additive, order-independent, transparent tracking, justified amounts.
        
        Bonus Philosophy:
        - Only reward EXCEPTIONAL patterns (top 5% cases)
        - All bonuses are ADDITIVE (no multiplicative chaos)
        - Maximum total bonus capped at 10 points
        - Each bonus requires multiple confirmations
        - Full transparency with bonus tracking
        - Never modify NaN scores
        
        Bonus Categories:
        1. Perfect Alignment (max 4 points)
        2. Momentum Confirmation (max 3 points)
        3. Breakout Validation (max 3 points)
        4. Volume Surge Quality (max 2 points)
        5. Recovery Pattern (max 2 points)
        
        Maximum cumulative bonus: 10 points (prevents score inflation)
        """
        # Don't modify original
        df = df.copy()
        
        # Initialize bonus tracking
        df['bonus_points'] = 0.0
        df['bonus_reasons'] = ''
        
        # Only apply bonuses to stocks with valid master scores
        valid_scores = df['master_score'].notna()
        
        if not valid_scores.any():
            logger.info("No valid master scores for bonus application")
            return df
        
        # Track individual bonuses for transparency
        perfect_alignment_bonus = pd.Series(0, index=df.index, dtype=float)
        momentum_bonus = pd.Series(0, index=df.index, dtype=float)
        breakout_bonus = pd.Series(0, index=df.index, dtype=float)
        volume_bonus = pd.Series(0, index=df.index, dtype=float)
        recovery_bonus = pd.Series(0, index=df.index, dtype=float)
        
        # BONUS 1: PERFECT ALIGNMENT (max 4 points)
        # All technical indicators aligned perfectly
        if all(col in df.columns for col in ['position_score', 'momentum_score', 'trend_quality', 'volume_score']):
            perfect_alignment = (
                valid_scores &
                (df['position_score'] > 80) &
                (df['momentum_score'] > 70) &
                (df['trend_quality'] > 80) &
                (df['volume_score'] > 70)
            )
            
            if perfect_alignment.any():
                perfect_alignment_bonus[perfect_alignment] = 4
                df.loc[perfect_alignment, 'bonus_reasons'] += 'PerfectAlign(+4) '
                logger.debug(f"Perfect alignment bonus applied to {perfect_alignment.sum()} stocks")
        
        # BONUS 2: MOMENTUM CONFIRMATION (max 3 points)
        # Strong momentum with acceleration and volume
        if all(col in df.columns for col in ['momentum_score', 'acceleration_score', 'rvol_score']):
            # Check for confirmed momentum
            momentum_confirmed = (
                valid_scores &
                (df['momentum_score'] > 75) &
                (df['acceleration_score'] > 70) &
                (df['rvol_score'] > 60)
            )
            
            if momentum_confirmed.any():
                # Graduated bonus based on strength
                super_momentum = momentum_confirmed & (df['momentum_score'] > 85)
                regular_momentum = momentum_confirmed & ~super_momentum
                
                momentum_bonus[super_momentum] = 3
                momentum_bonus[regular_momentum] = 2
                
                df.loc[super_momentum, 'bonus_reasons'] += 'SuperMomentum(+3) '
                df.loc[regular_momentum, 'bonus_reasons'] += 'Momentum(+2) '
                logger.debug(f"Momentum bonus applied to {momentum_confirmed.sum()} stocks")
        
        # BONUS 3: BREAKOUT VALIDATION (max 3 points)
        # Confirmed breakout with volume
        if all(col in df.columns for col in ['breakout_score', 'position_score', 'rvol_score']):
            breakout_confirmed = (
                valid_scores &
                (df['breakout_score'] > 75) &
                (df['position_score'] > 70) &  # Near highs
                (df['rvol_score'] > 65)  # With volume
            )
            
            if breakout_confirmed.any():
                # Extra bonus for extreme breakout scores
                extreme_breakout = breakout_confirmed & (df['breakout_score'] > 90)
                regular_breakout = breakout_confirmed & ~extreme_breakout
                
                breakout_bonus[extreme_breakout] = 3
                breakout_bonus[regular_breakout] = 2
                
                df.loc[extreme_breakout, 'bonus_reasons'] += 'ExtremeBreakout(+3) '
                df.loc[regular_breakout, 'bonus_reasons'] += 'Breakout(+2) '
                logger.debug(f"Breakout bonus applied to {breakout_confirmed.sum()} stocks")
        
        # BONUS 4: VOLUME SURGE QUALITY (max 2 points)
        # Exceptional volume with price confirmation
        if all(col in df.columns for col in ['volume_score', 'rvol_score', 'momentum_score']):
            volume_surge = (
                valid_scores &
                (df['volume_score'] > 80) &
                (df['rvol_score'] > 75) &
                (df['momentum_score'] > 50)  # Positive momentum
            )
            
            if volume_surge.any():
                volume_bonus[volume_surge] = 2
                df.loc[volume_surge, 'bonus_reasons'] += 'VolumeSurge(+2) '
                logger.debug(f"Volume surge bonus applied to {volume_surge.sum()} stocks")
        
        # BONUS 5: RECOVERY PATTERN (max 2 points)
        # Strong recovery from oversold
        if all(col in df.columns for col in ['position_score', 'momentum_score', 'acceleration_score']):
            recovery_pattern = (
                valid_scores &
                (df['position_score'] < 40) &  # Still relatively low
                (df['momentum_score'] > 60) &  # But strong momentum
                (df['acceleration_score'] > 70)  # And accelerating
            )
            
            if recovery_pattern.any():
                recovery_bonus[recovery_pattern] = 2
                df.loc[recovery_pattern, 'bonus_reasons'] += 'Recovery(+2) '
                logger.debug(f"Recovery bonus applied to {recovery_pattern.sum()} stocks")
        
        # BONUS 6: MARKET STATE MOMENTUM (max 3 points)
        # Individual stock in STRONG_UPTREND with high momentum
        market_state_bonus = pd.Series(0, index=df.index, dtype=float)
        if 'market_state' in df.columns and 'momentum_score' in df.columns:
            strong_uptrend_momentum = (
                valid_scores &
                (df['market_state'] == 'STRONG_UPTREND') &
                (df['momentum_score'] > 70)
            )
            
            if strong_uptrend_momentum.any():
                market_state_bonus[strong_uptrend_momentum] = 3
                df.loc[strong_uptrend_momentum, 'bonus_reasons'] += 'StrongMomentum(+3) '
                logger.debug(f"Strong uptrend momentum bonus applied to {strong_uptrend_momentum.sum()} stocks")
        
        # BONUS 7: MARKET STATE VALUE OPPORTUNITY (max 2 points)
        # Stock in PULLBACK state with low position score (value opportunity)
        if 'market_state' in df.columns and 'position_score' in df.columns:
            pullback_value = (
                valid_scores &
                (df['market_state'] == 'PULLBACK') &
                (df['position_score'] < 30)
            )
            
            if pullback_value.any():
                market_state_bonus[pullback_value] = 2
                df.loc[pullback_value, 'bonus_reasons'] += 'PullbackValue(+2) '
                logger.debug(f"Pullback value opportunity bonus applied to {pullback_value.sum()} stocks")
        
        # PENALTY PATTERNS (negative bonuses)
        penalty = pd.Series(0, index=df.index, dtype=float)
        
        # Penalty 1: Divergence (high score components but low others)
        if all(col in df.columns for col in ['momentum_score', 'volume_score']):
            divergence = (
                valid_scores &
                ((df['momentum_score'] > 80) & (df['volume_score'] < 30)) |
                ((df['momentum_score'] < 30) & (df['volume_score'] > 80))
            )
            
            if divergence.any():
                penalty[divergence] = -2
                df.loc[divergence, 'bonus_reasons'] += 'Divergence(-2) '
                logger.debug(f"Divergence penalty applied to {divergence.sum()} stocks")
        
        # Penalty 2: Low confidence scores
        if 'confidence_score' in df.columns:
            low_confidence = (
                valid_scores &
                (df['confidence_score'] < 30) &
                (df['master_score'] > 70)  # High score but low confidence
            )
            
            if low_confidence.any():
                penalty[low_confidence] = -3
                df.loc[low_confidence, 'bonus_reasons'] += 'LowConfidence(-3) '
                logger.debug(f"Low confidence penalty applied to {low_confidence.sum()} stocks")
        
        # CALCULATE TOTAL BONUS
        # Sum all bonuses (including negative)
        total_bonus = (
            perfect_alignment_bonus +
            momentum_bonus +
            breakout_bonus +
            volume_bonus +
            recovery_bonus +
            market_state_bonus +
            penalty
        )
        
        # Cap total bonus to prevent score inflation
        MAX_POSITIVE_BONUS = 10
        MAX_NEGATIVE_PENALTY = -5
        
        total_bonus = total_bonus.clip(MAX_NEGATIVE_PENALTY, MAX_POSITIVE_BONUS)
        
        # Apply bonuses only to valid scores
        df.loc[valid_scores, 'bonus_points'] = total_bonus[valid_scores]
        
        # Calculate final score with bonuses
        df['master_score_before_bonus'] = df['master_score'].copy()
        df.loc[valid_scores, 'master_score'] = (
            df.loc[valid_scores, 'master_score'] + total_bonus[valid_scores]
        ).clip(0, 100)
        
        # SPECIAL ADJUSTMENTS (applied after bonuses)
        # Market cap adjustments
        if 'category' in df.columns:
            # Mega caps with high scores get stability bonus
            mega_caps = df['category'] == 'Mega Cap'
            mega_high = mega_caps & valid_scores & (df['master_score'] > 80)
            if mega_high.any():
                df.loc[mega_high, 'master_score'] = np.minimum(
                    df.loc[mega_high, 'master_score'] + 2, 
                    100
                )
                df.loc[mega_high, 'bonus_reasons'] += 'MegaCap(+2) '
            
            # Penny stocks with extreme scores get capped
            penny_stocks = df['category'].isin(['Micro Cap', 'Small Cap'])
            penny_extreme = penny_stocks & valid_scores & (df['master_score'] > 90)
            if penny_extreme.any():
                df.loc[penny_extreme, 'master_score'] = np.minimum(
                    df.loc[penny_extreme, 'master_score'],
                    88  # Cap at 88
                )
                df.loc[penny_extreme, 'bonus_reasons'] += 'PennyCap(capped) '
        
        # COMPREHENSIVE LOGGING
        bonuses_applied = (df['bonus_points'] != 0).sum()
        
        if bonuses_applied > 0:
            logger.info(f"Smart bonuses applied to {bonuses_applied} stocks")
            
            # Bonus distribution
            positive_bonuses = (df['bonus_points'] > 0).sum()
            negative_penalties = (df['bonus_points'] < 0).sum()
            
            logger.info(f"Bonus distribution: {positive_bonuses} bonuses, {negative_penalties} penalties")
            
            # Average bonus impact
            avg_bonus = df.loc[df['bonus_points'] != 0, 'bonus_points'].mean()
            max_bonus = df['bonus_points'].max()
            min_bonus = df['bonus_points'].min()
            
            logger.debug(f"Bonus impact - Avg: {avg_bonus:.1f}, Max: {max_bonus:.1f}, Min: {min_bonus:.1f}")
            
            # Score changes
            score_changes = df.loc[valid_scores, 'master_score'] - df.loc[valid_scores, 'master_score_before_bonus']
            if (score_changes != 0).any():
                avg_change = score_changes[score_changes != 0].mean()
                logger.debug(f"Average score change from bonuses: {avg_change:.1f} points")
            
            # Top bonus recipients
            if 'ticker' in df.columns:
                top_bonus = df.nlargest(3, 'bonus_points')
                if len(top_bonus) > 0:
                    logger.debug("Top bonus recipients:")
                    for _, row in top_bonus.iterrows():
                        if row['bonus_points'] > 0:
                            logger.debug(f"  {row['ticker']}: +{row['bonus_points']:.1f} ({row['bonus_reasons'].strip()})")
        else:
            logger.info("No stocks qualified for smart bonuses")
        
        # Verify score integrity
        if (df['master_score'] > 100).any():
            over_100 = (df['master_score'] > 100).sum()
            logger.error(f"ERROR: {over_100} stocks have scores > 100 after bonuses!")
            df['master_score'] = df['master_score'].clip(0, 100)
        
        if (df['master_score'] < 0).any():
            below_0 = (df['master_score'] < 0).sum()
            logger.error(f"ERROR: {below_0} stocks have scores < 0 after bonuses!")
            df['master_score'] = df['master_score'].clip(0, 100)
        
        return df

    @staticmethod
    def _calculate_category_ranks(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive category-based rankings with full error handling.
        FIXED: Fully vectorized, handles all edge cases, provides multiple metrics.
        
        Category Ranking Philosophy:
        - Ranks within peer groups (Large Cap, Mid Cap, etc.)
        - Provides both rank and percentile
        - Handles small categories intelligently
        - Adds category statistics for context
        - Size-adjusted rankings for fairness
        - Completely vectorized (no loops)
        
        Outputs:
        - category_rank: Rank within category (1 = best)
        - category_percentile: Percentile within category (100 = best)
        - category_size: Number of stocks in category
        - category_decile: Which decile within category (1-10)
        - category_relative_score: Score vs category average
        - peer_group_rank: Rank within similar-sized categories
        """
        # Don't modify original
        df = df.copy()
        
        # Initialize all category columns with NaN
        category_columns = [
            'category_rank',
            'category_percentile', 
            'category_size',
            'category_decile',
            'category_relative_score',
            'peer_group_rank',
            'category_avg_score',
            'category_median_score',
            'category_std_score'
        ]
        
        for col in category_columns:
            df[col] = np.nan
        
        # Check if category column exists
        if 'category' not in df.columns:
            logger.warning("No 'category' column found, skipping category rankings")
            return df
        
        # Check if master_score exists
        if 'master_score' not in df.columns:
            logger.warning("No 'master_score' column found, skipping category rankings")
            return df
        
        # Get unique categories
        categories = df['category'].unique()
        valid_categories = [c for c in categories if pd.notna(c)]
        
        if len(valid_categories) == 0:
            logger.warning("No valid categories found")
            return df
        
        logger.info(f"Calculating category ranks for {len(valid_categories)} categories")
        
        # VECTORIZED CATEGORY RANKING - No loops!
        # Use groupby for all operations at once
        
        # 1. Basic category rankings
        category_groups = df.groupby('category')['master_score']
        
        # Rank within category (1 = best)
        df['category_rank'] = category_groups.rank(
            method='first',  # First occurrence wins ties
            ascending=False,  # Higher score = better rank
            na_option='bottom'  # NaN scores go to bottom
        )
        
        # Percentile within category (100 = best)
        df['category_percentile'] = category_groups.rank(
            method='average',  # Average for ties
            ascending=True,  # Lower score = lower percentile
            pct=True,  # Return as percentile
            na_option='bottom'
        ) * 100
        
        # 2. Category size (for context)
        category_sizes = df.groupby('category').size()
        df['category_size'] = df['category'].map(category_sizes)
        
        # 3. Category statistics
        category_stats = df.groupby('category')['master_score'].agg([
            ('category_avg_score', 'mean'),
            ('category_median_score', 'median'),
            ('category_std_score', 'std'),
            ('category_min_score', 'min'),
            ('category_max_score', 'max'),
            ('category_valid_count', 'count')
        ])
        
        # Map statistics back to dataframe
        for stat_col in category_stats.columns:
            df[stat_col] = df['category'].map(category_stats[stat_col])
        
        # 4. Relative score vs category average
        df['category_relative_score'] = safe_divide(
            df['master_score'] - df['category_avg_score'],
            df['category_std_score'],
            default=0.0
        )
        
        # 5. Decile within category (1-10, 1 = best)
        df['category_decile'] = pd.cut(
            df['category_percentile'],
            bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1],  # Reverse order (1 = best)
            include_lowest=True
        )
        
        # 6. Handle small categories (less than 10 stocks)
        small_categories = df['category_size'] < 10
        if small_categories.any():
            # Mark small category ranks as less reliable
            df.loc[small_categories, 'category_rank_reliability'] = 'Low'
            logger.info(f"{small_categories.sum()} stocks in small categories (<10 stocks)")
        else:
            df['category_rank_reliability'] = 'Normal'
        
        # Very small categories (less than 3 stocks)
        tiny_categories = df['category_size'] < 3
        if tiny_categories.any():
            # Don't rank if category too small
            df.loc[tiny_categories, 'category_rank_reliability'] = 'Insufficient'
            logger.warning(f"{tiny_categories.sum()} stocks in tiny categories (<3 stocks)")
        
        # 7. Peer group ranking (compare similar market cap categories)
        # Define peer groups
        peer_groups = {
            'Large_Mega': ['Large Cap', 'Mega Cap'],
            'Mid': ['Mid Cap'],
            'Small_Micro': ['Small Cap', 'Micro Cap']
        }
        
        df['peer_group'] = 'Other'  # Default
        for group_name, categories_in_group in peer_groups.items():
            mask = df['category'].isin(categories_in_group)
            df.loc[mask, 'peer_group'] = group_name
        
        # Rank within peer groups
        peer_groups_data = df.groupby('peer_group')['master_score']
        df['peer_group_rank'] = peer_groups_data.rank(
            method='first',
            ascending=False,
            na_option='bottom'
        )
        
        df['peer_group_size'] = df.groupby('peer_group')['peer_group'].transform('size')
        
        df['peer_group_percentile'] = peer_groups_data.rank(
            method='average',
            ascending=True,
            pct=True,
            na_option='bottom'
        ) * 100
        
        # 8. Category performance tiers
        # Classify categories by average score
        category_performance = category_stats['category_avg_score'].to_dict()
        
        def classify_category_performance(cat):
            if cat not in category_performance or pd.isna(category_performance[cat]):
                return 'Unknown'
            avg_score = category_performance[cat]
            if avg_score >= 70:
                return 'Strong'
            elif avg_score >= 50:
                return 'Average'
            elif avg_score >= 30:
                return 'Weak'
            else:
                return 'Very Weak'
        
        df['category_performance'] = df['category'].apply(classify_category_performance)
        
        # 9. Best in category flags
        # Mark top stocks in each category
        df['is_category_leader'] = df['category_rank'] == 1
        df['is_category_top5'] = df['category_rank'] <= 5
        df['is_category_top10'] = df['category_rank'] <= 10
        df['is_category_top_decile'] = df['category_decile'] == 1
        
        # 10. Category momentum (if historical data available)
        if 'momentum_score' in df.columns:
            category_momentum = df.groupby('category')['momentum_score'].mean()
            df['category_momentum'] = df['category'].map(category_momentum)
            
            # Flag if stock momentum better than category
            df['beats_category_momentum'] = df['momentum_score'] > df['category_momentum']
        
        # COMPREHENSIVE LOGGING
        for category in valid_categories:
            cat_data = df[df['category'] == category]
            valid_scores = cat_data['master_score'].notna().sum()
            
            if valid_scores > 0:
                avg_score = cat_data['master_score'].mean()
                median_score = cat_data['master_score'].median()
                
                logger.debug(f"{category}: {valid_scores} stocks, "
                            f"Avg: {avg_score:.1f}, Median: {median_score:.1f}")
        
        # Log category distribution
        category_dist = df['category'].value_counts()
        logger.info("Category distribution:")
        for cat, count in category_dist.head().items():
            logger.info(f"  {cat}: {count} stocks")
        
        # Find category leaders
        if 'ticker' in df.columns:
            leaders = df[df['is_category_leader'] == True]
            if len(leaders) > 0:
                logger.info(f"Category leaders identified: {len(leaders)}")
                for _, leader in leaders.head(5).iterrows():
                    if pd.notna(leader['master_score']):
                        logger.debug(f"  {leader['category']}: {leader['ticker']} "
                                   f"(Score: {leader['master_score']:.1f})")
        
        # Verify rankings integrity
        for category in valid_categories:
            cat_mask = df['category'] == category
            cat_ranks = df.loc[cat_mask, 'category_rank']
            
            if cat_ranks.notna().any():
                # Check for duplicate ranks
                rank_counts = cat_ranks.value_counts()
                duplicates = rank_counts[rank_counts > 1]
                if len(duplicates) > 0:
                    logger.error(f"Duplicate ranks in {category}: {duplicates.to_dict()}")
                
                # Check for gaps in ranking
                valid_ranks = cat_ranks.dropna().sort_values()
                expected_ranks = range(1, len(valid_ranks) + 1)
                if not all(r in valid_ranks.values for r in expected_ranks):
                    logger.warning(f"Ranking gaps detected in {category}")
        
        # Performance statistics
        logger.info("Category ranking summary:")
        logger.info(f"  Categories processed: {len(valid_categories)}")
        logger.info(f"  Stocks with category ranks: {df['category_rank'].notna().sum()}")
        logger.info(f"  Category leaders: {df['is_category_leader'].sum()}")
        logger.info(f"  Top decile stocks: {df['is_category_top_decile'].sum()}")
        
        # Small category warning
        small_cat_count = (df['category_size'] < 10).sum()
        if small_cat_count > 0:
            small_cat_pct = (small_cat_count / len(df)) * 100
            logger.warning(f"{small_cat_count} stocks ({small_cat_pct:.1f}%) in small categories")
        
        # Cross-category comparison
        if len(valid_categories) > 1:
            best_category = df.groupby('category')['master_score'].mean().idxmax()
            worst_category = df.groupby('category')['master_score'].mean().idxmin()
            
            if pd.notna(best_category) and pd.notna(worst_category):
                logger.info(f"Best performing category: {best_category}")
                logger.info(f"Worst performing category: {worst_category}")
        
        return df
# ============================================
# PATTERN DETECTION ENGINE - FULLY OPTIMIZED & FIXED
# ============================================

class PatternDetector:
    """
    Advanced pattern detection using vectorized operations for maximum performance.
    This class identifies a comprehensive set of 41 technical, fundamental,
    and intelligent trading patterns.
    FIXED: Pattern confidence calculation now works correctly.
    """

    # Pattern metadata for intelligent confidence scoring
    PATTERN_METADATA = {
            '🐱 CAT LEADER': {'importance_weight': 10, 'category': 'momentum'},
            '💎 HIDDEN GEM': {'importance_weight': 10, 'category': 'value'},
            '🏦 INSTITUTIONAL': {'importance_weight': 10, 'category': 'volume'},
            '⚡ VOL EXPLOSION': {'importance_weight': 15, 'category': 'volume'},
            '👑 MARKET LEADER': {'importance_weight': 10, 'category': 'leadership'},
            '🌊 MOMENTUM WAVE': {'importance_weight': 10, 'category': 'momentum'},
            '💰 LIQUID LEADER': {'importance_weight': 10, 'category': 'liquidity'},
            '🔥 PREMIUM MOMENTUM': {'importance_weight': 15, 'category': 'premium'},
            '🧩 ENTROPY COMPRESSION': {'importance_weight': 20, 'category': 'mathematical'},
            '🚀 VELOCITY BREAKOUT': {'importance_weight': 15, 'category': 'acceleration'},
            '🌋 INSTITUTIONAL TSUNAMI': {'importance_weight': 25, 'category': 'institutional'},
            '📈 VALUE MOMENTUM': {'importance_weight': 10, 'category': 'fundamental'},
            '🎯 EARNINGS ROCKET': {'importance_weight': 10, 'category': 'fundamental'},
            '🏆 QUALITY LEADER': {'importance_weight': 10, 'category': 'fundamental'},
            '🔄 TURNAROUND': {'importance_weight': 10, 'category': 'fundamental'},
            '⚠️ HIGH PE': {'importance_weight': -5, 'category': 'warning'},
            '🎲 52W HIGH APPROACH': {'importance_weight': 10, 'category': 'range'},
            '↗️ 52W LOW BOUNCE': {'importance_weight': 10, 'category': 'range'},
            '🔀 MOMENTUM DIVERGE': {'importance_weight': 10, 'category': 'divergence'},
            '🤏 RANGE COMPRESS': {'importance_weight': 5, 'category': 'range'},
            '🤫 STEALTH': {'importance_weight': 10, 'category': 'hidden'},
            '🏎️ ACCELERATION': {'importance_weight': 10, 'category': 'aggressive'},
            '⛈️ PERFECT STORM': {'importance_weight': 20, 'category': 'extreme'},
            '🪤 BULL TRAP': {'importance_weight': 15, 'category': 'reversal'},
            '💣 CAPITULATION': {'importance_weight': 20, 'category': 'reversal'},
            '🏃 RUNAWAY GAP': {'importance_weight': 12, 'category': 'continuation'},
            '🔃 ROTATION LEADER': {'importance_weight': 10, 'category': 'rotation'},
            '📊 DISTRIBUTION': {'importance_weight': 15, 'category': 'warning'},
            '🗜️ VELOCITY SQUEEZE': {'importance_weight': 15, 'category': 'coiled'},
            '🔉 VOLUME DIVERGENCE': {'importance_weight': -10, 'category': 'warning'},
            '✨ GOLDEN CROSS': {'importance_weight': 12, 'category': 'bullish'},
            '📉 EXHAUSTION': {'importance_weight': -15, 'category': 'bearish'},
            '🔺 PYRAMID': {'importance_weight': 8, 'category': 'accumulation'},
            '🌪️ VACUUM': {'importance_weight': 18, 'category': 'reversal'},
            '🎆 EARNINGS SURPRISE LEADER': {'importance_weight': 22, 'category': 'fundamental'},
            '🕰️ INFORMATION DECAY ARBITRAGE': {'importance_weight': 25, 'category': 'mathematical'},
            '🐦 PHOENIX RISING': {'importance_weight': 28, 'category': 'transformation'},
            '⚛️ ATOMIC DECAY MOMENTUM': {'importance_weight': 20, 'category': 'physics'},
            '💹 GARP LEADER': {'importance_weight': 18, 'category': 'fundamental'},
            '🛡️ PULLBACK SUPPORT': {'importance_weight': 12, 'category': 'technical'},
            '💳 OVERSOLD QUALITY': {'importance_weight': 15, 'category': 'value'}
    }

    @staticmethod
    @PerformanceMonitor.timer(target_time=0.3)
    def detect_all_patterns_optimized(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects all trading patterns using highly efficient vectorized operations.
        Returns a DataFrame with 'patterns' column and 'pattern_confidence' score.
        """
        if df.empty:
            df['patterns'] = ''
            df['pattern_confidence'] = 0.0
            df['pattern_count'] = 0
            df['pattern_categories'] = ''
            return df
        
        logger.info(f"Starting pattern detection for {len(df)} stocks...")
        
        # Get all pattern definitions
        patterns_with_masks = PatternDetector._get_all_pattern_definitions(df)
        
        # Create pattern matrix for vectorized processing
        pattern_names = [name for name, _ in patterns_with_masks]
        pattern_matrix = pd.DataFrame(False, index=df.index, columns=pattern_names)
        
        # Fill pattern matrix with detection results
        patterns_detected = 0
        for name, mask in patterns_with_masks:
            if mask is not None:
                # Convert mask to pandas Series if it's a numpy array
                if isinstance(mask, np.ndarray):
                    mask = pd.Series(mask, index=df.index)
                elif not isinstance(mask, pd.Series):
                    mask = pd.Series(mask, index=df.index)
                
                # Check if mask has any data
                if len(mask) > 0:
                    pattern_matrix[name] = mask.reindex(df.index, fill_value=False)
                    detected_count = mask.sum() if hasattr(mask, 'sum') else np.sum(mask)
                    if detected_count > 0:
                        patterns_detected += 1
                        logger.debug(f"Pattern '{name}' detected in {detected_count} stocks")
        
        # Combine patterns into string column
        df['patterns'] = pattern_matrix.apply(
            lambda row: ' | '.join(row.index[row].tolist()), axis=1
        )
        df['patterns'] = df['patterns'].fillna('')
        
        # Count patterns per stock
        df['pattern_count'] = pattern_matrix.sum(axis=1)
        
        # Calculate pattern categories
        df['pattern_categories'] = pattern_matrix.apply(
            lambda row: PatternDetector._get_pattern_categories(row), axis=1
        )
        
        # Calculate confidence score with FIXED calculation
        df = PatternDetector._calculate_pattern_confidence(df)
        
        # Log summary
        stocks_with_patterns = (df['patterns'] != '').sum()
        avg_patterns_per_stock = df['pattern_count'].mean()
        logger.info(f"Pattern detection complete: {patterns_detected} patterns found, "
                   f"{stocks_with_patterns} stocks with patterns, "
                   f"avg {avg_patterns_per_stock:.1f} patterns/stock")
        
        return df

    @staticmethod
    def _calculate_pattern_confidence(df: pd.DataFrame) -> pd.DataFrame:
        """
        FIXED: Calculate confidence score based on pattern importance weights.
        Now properly calculates max_possible_score.
        """
        
        # Calculate maximum possible score for normalization
        all_positive_weights = [
            abs(meta['importance_weight']) 
            for meta in PatternDetector.PATTERN_METADATA.values()
            if meta['importance_weight'] > 0
        ]
        max_possible_score = sum(sorted(all_positive_weights, reverse=True)[:5])  # Top 5 patterns
        
        def calculate_confidence(patterns_str):
            """Calculate confidence for a single stock's patterns"""
            if pd.isna(patterns_str) or patterns_str == '':
                return 0.0
            
            patterns = [p.strip() for p in patterns_str.split(' | ')]
            total_weight = 0
            pattern_categories = set()
            
            for pattern in patterns:
                # Match pattern with metadata (handle emoji differences)
                for key, meta in PatternDetector.PATTERN_METADATA.items():
                    if pattern == key or pattern.replace(' ', '') == key.replace(' ', ''):
                        total_weight += meta['importance_weight']
                        pattern_categories.add(meta.get('category', 'unknown'))
                        break
            
            # Bonus for diverse categories
            category_bonus = len(pattern_categories) * 2
            
            # Calculate final confidence
            if max_possible_score > 0:
                raw_confidence = (abs(total_weight) + category_bonus) / max_possible_score * 100
                # Apply sigmoid smoothing for better distribution
                confidence = 100 * (2 / (1 + np.exp(-raw_confidence/50)) - 1)
                return min(100, max(0, confidence))
            return 0.0
        
        # Apply calculation to all rows
        df['pattern_confidence'] = df['patterns'].apply(calculate_confidence).round(2)
        
        # Add confidence tier
        df['confidence_tier'] = pd.cut(
            df['pattern_confidence'],
            bins=[0, 25, 50, 75, 100],
            labels=['Low', 'Medium', 'High', 'Very High'],
            include_lowest=True
        )
        
        return df
    
    @staticmethod
    def _get_pattern_categories(row: pd.Series) -> str:
        """Get unique categories for detected patterns"""
        categories = set()
        for pattern_name in row.index[row]:
            for key, meta in PatternDetector.PATTERN_METADATA.items():
                if pattern_name == key or pattern_name.replace(' ', '') == key.replace(' ', ''):
                    categories.add(meta.get('category', 'unknown'))
                    break
        return ', '.join(sorted(categories)) if categories else ''

    @staticmethod
    def _get_all_pattern_definitions(df: pd.DataFrame) -> List[Tuple[str, pd.Series]]:
        """
        Defines all 41 patterns using vectorized boolean masks.
        FIXED: Uses raw data instead of scores that don't exist yet.
        Returns list of (pattern_name, mask) tuples.
        """
        patterns = []
        
        # Helper function to safely get column data
        def get_col_safe(col_name: str, default_value: Any = np.nan) -> pd.Series:
            if col_name in df.columns:
                return df[col_name].copy()
            return pd.Series(default_value, index=df.index)
        
        # Helper function to ensure mask is a pandas Series
        def ensure_series(mask: Any) -> pd.Series:
            if isinstance(mask, pd.Series):
                return mask
            elif isinstance(mask, np.ndarray):
                return pd.Series(mask, index=df.index)
            else:
                return pd.Series(mask, index=df.index)
    
        # ========== MOMENTUM & LEADERSHIP PATTERNS (1-11) ==========
        
        # 1. Category Leader - FIXED: category_percentile might not exist yet
        # The column exists, but we'll validate the threshold is working
        cat_percentile = get_col_safe('category_percentile', 0)
        threshold = CONFIG.PATTERN_THRESHOLDS.get('category_leader', 90)
        
        # Create mask
        mask = ensure_series(cat_percentile >= threshold)
        
        # Log for debugging (optional - remove in production)
        leaders_count = mask.sum()
        if leaders_count > 0:
            logger.debug(f"CAT LEADER: Found {leaders_count} category leaders (threshold: {threshold})")
        
        patterns.append(('🐱 CAT LEADER', mask))
        
        # 2. Hidden Gem - FIXED: percentile might not exist yet
        mask = ensure_series(
            (get_col_safe('category_percentile', 0) >= CONFIG.PATTERN_THRESHOLDS.get('hidden_gem', 80)) & 
            (get_col_safe('percentile', 100) < 70)
        )
        patterns.append(('💎 HIDDEN GEM', mask))
        
        # 3. Institutional - FIXED: Use raw volume ratios
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'ret_3m', 'from_low_pct', 'from_high_pct']):
            mask = (
                (get_col_safe('vol_ratio_90d_180d', 1) > 1.2) &
                (get_col_safe('vol_ratio_30d_90d', 1).between(0.9, 1.1)) &
                (get_col_safe('ret_3m', 0).between(5, 25)) &
                (get_col_safe('from_low_pct', 0) > 30) &
                (get_col_safe('from_high_pct', -100) > -30)
            )
        else:
            mask = pd.Series(False, index=df.index)
        patterns.append(('🏦 INSTITUTIONAL', mask))
        
        # 4. Volume Explosion - Works with raw data
        mask = get_col_safe('rvol', 0) > 3
        patterns.append(('⚡ VOL EXPLOSION', mask))
        
        # 5. Market Leader - FIXED: percentile might not exist
        if 'percentile' in df.columns:
            mask = get_col_safe('percentile', 0) >= CONFIG.PATTERN_THRESHOLDS.get('market_leader', 95)
        else:
            mask = pd.Series(False, index=df.index)
        patterns.append(('👑 MARKET LEADER', mask))
        
        # 6. Momentum Wave - FIXED: Use return data directly
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'rvol']):
            ret_7d = get_col_safe('ret_7d', 0)
            ret_30d = get_col_safe('ret_30d', 0)
            
            # Calculate acceleration
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d = ret_7d / 7
                daily_30d = ret_30d / 30
                accelerating = (daily_7d > daily_30d * 1.2)  # 20% faster pace
            
            mask = (
                (ret_30d >= 15) &  # Strong 30-day momentum
                (ret_7d >= 5) &    # Good 7-day momentum
                accelerating &      # Acceleration confirmed
                (get_col_safe('rvol', 1) > 1.5)  # Volume confirmation
            )
        else:
            mask = pd.Series(False, index=df.index)
        patterns.append(('🌊 MOMENTUM WAVE', mask))
        
        # 7. Liquid Leader - FIXED: Check if liquidity_score exists
        if 'liquidity_score' in df.columns and 'percentile' in df.columns:
            mask = (
                (get_col_safe('liquidity_score', 0) >= CONFIG.PATTERN_THRESHOLDS.get('liquid_leader', 80)) & 
                (get_col_safe('percentile', 0) >= CONFIG.PATTERN_THRESHOLDS.get('liquid_leader', 80))
            )
        else:
            # Use volume as proxy for liquidity
            mask = (
                (get_col_safe('volume_1d', 0) > df['volume_1d'].median() * 2) &
                (get_col_safe('rvol', 1) > 1.5)
            )
        patterns.append(('💰 LIQUID LEADER', mask))
        
        # 8. Premium Momentum Grade - FIXED: Use master_score if available
        try:
            ret_1d = get_col_safe('ret_1d', 0)
            ret_7d = get_col_safe('ret_7d', 0)
            ret_30d = get_col_safe('ret_30d', 0)
            
            # Build momentum DNA without master_score
            momentum_dna_score = (
                np.where((ret_1d > 0) & (ret_7d > 0) & (ret_30d > 0), 25, 0) +
                np.where(ret_7d > safe_divide(ret_30d, 4, default=0), 30, 0) +
                np.where(get_col_safe('rvol', 1) > 1.2, 20, 0) +
                np.where(ret_30d > 20, 25, 0)  # Use return instead of score
            )
            
            mask = (
                (momentum_dna_score >= 75) &
                (get_col_safe('from_low_pct', 0) > 15) &
                (get_col_safe('from_high_pct', 0) > -15) &
                (get_col_safe('vol_ratio_7d_90d', 1) > 1.1)
            )
        except Exception:
            mask = pd.Series(False, index=df.index)
        patterns.append(('🔥 PREMIUM MOMENTUM', mask))

        # 9. Entropy Compression - Volatility breakout prediction using information theory
        # FIXED: Properly implements entropy compression detection with actual data structure
        try:
            if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d', 'volume_1d', 'volume_30d', 'volume_90d', 'rvol']):
                ret_1d = get_col_safe('ret_1d', 0)
                ret_7d = get_col_safe('ret_7d', 0)
                ret_30d = get_col_safe('ret_30d', 0)
                
                # FIXED: Calculate volatility compression (not rolling - we have point-in-time data)
                # Entropy Metric 1: Return volatility compression
                # Compare short-term vs long-term volatility using actual returns
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Daily equivalent volatility
                    daily_vol_7d = np.abs(ret_7d / 7)    # Average daily volatility over 7 days
                    daily_vol_30d = np.abs(ret_30d / 30)  # Average daily volatility over 30 days
                    
                    # Volatility compression: short-term vol < long-term vol (tightening range)
                    volatility_compressed = (
                        (daily_vol_7d < daily_vol_30d * 0.7) &  # 7-day vol is 30% lower than 30-day
                        (daily_vol_7d < 2)  # And absolute volatility is low (< 2% daily)
                    )
                
                # Entropy Metric 2: Volume stability (low entropy in volume)
                volume_1d = get_col_safe('volume_1d', 0)
                volume_30d = get_col_safe('volume_30d', 0)
                volume_90d = get_col_safe('volume_90d', 0)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Volume consistency check
                    vol_ratio_1d_30d = np.where(volume_30d > 0, volume_1d / volume_30d, 1)
                    vol_ratio_30d_90d = np.where(volume_90d > 0, volume_30d / volume_90d, 1)
                    
                    # Stable volume = ratios close to 1 (low entropy)
                    volume_stable = (
                        (vol_ratio_1d_30d > 0.7) & (vol_ratio_1d_30d < 1.5) &  # Daily volume within normal range
                        (vol_ratio_30d_90d > 0.8) & (vol_ratio_30d_90d < 1.2)   # Monthly volume stable
                    )
                
                # Entropy Metric 3: Price range compression
                high_52w = get_col_safe('high_52w', 0)
                low_52w = get_col_safe('low_52w', 0)
                price = get_col_safe('price', 0)
                from_low_pct = get_col_safe('from_low_pct', 0)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Calculate 52-week range as percentage
                    range_pct = np.where(low_52w > 0, ((high_52w - low_52w) / low_52w) * 100, 100)
                    
                    # Price compression: tight range + middle position
                    price_compressed = (
                        (range_pct < 50) &  # Less than 50% range in 52 weeks
                        (from_low_pct > 30) & (from_low_pct < 70)  # In middle 40% of range
                    )
                
                # Entropy Metric 4: Technical structure alignment (order from chaos)
                sma_20d = get_col_safe('sma_20d', 0)
                sma_50d = get_col_safe('sma_50d', 0)
                sma_200d = get_col_safe('sma_200d', 0)
                
                # Price structure shows order (bullish alignment)
                price_structure_aligned = (
                    (price > sma_20d) &
                    (sma_20d > sma_50d) &
                    (sma_50d > sma_200d)
                )
                
                # Catalyst Detection: Signs of energy building
                rvol = get_col_safe('rvol', 1)
                eps_change_pct = get_col_safe('eps_change_pct', 0)
                vol_ratio_7d_90d = get_col_safe('vol_ratio_7d_90d', 1)
                
                # Calculate catalyst score (energy building up)
                catalyst_indicators = (
                    np.where(rvol > 1.2, 1, 0) +  # Volume picking up
                    np.where(eps_change_pct > 10, 1, 0) +  # Earnings momentum
                    np.where(vol_ratio_7d_90d > 1.1, 1, 0) +  # Recent volume trend
                    np.where(ret_7d > 0, 1, 0)  # Positive recent momentum
                )
                
                # FINAL ENTROPY COMPRESSION DETECTION
                # Low entropy (compressed state) + energy building = potential breakout
                mask = ensure_series(
                    volatility_compressed &  # Volatility is compressed
                    volume_stable &  # Volume patterns are stable
                    price_compressed &  # Price range is tight
                    price_structure_aligned &  # Technical structure is ordered
                    (catalyst_indicators >= 2)  # At least 2 catalyst signals
                )
                
                # Log detection stats for debugging
                compression_count = mask.sum() if hasattr(mask, 'sum') else 0
                if compression_count > 0:
                    logger.debug(f"ENTROPY COMPRESSION: {compression_count} stocks detected")
                    
            else:
                # Missing required columns
                logger.warning("ENTROPY COMPRESSION: Missing required columns")
                mask = pd.Series(False, index=df.index)
                
        except Exception as e:
            logger.error(f"Error in ENTROPY COMPRESSION pattern: {str(e)}")
            mask = pd.Series(False, index=df.index)
        
        patterns.append(('🧩 ENTROPY COMPRESSION', mask))
        
        # 10. Velocity Breakout - Multi-timeframe momentum acceleration
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'rvol', 'from_high_pct']):
            mask = (
                (get_col_safe('ret_1d', 0) > 3) &                              # Recent significant pop
                (get_col_safe('ret_7d', 0) > get_col_safe('ret_30d', 1) * 0.5) &  # Weekly pace exceeds monthly
                (get_col_safe('ret_30d', 0) > get_col_safe('ret_3m', 1) * 0.7) &  # Monthly pace exceeds quarterly
                (get_col_safe('rvol', 0) > 2) &                                # Strong volume confirmation
                (get_col_safe('from_high_pct', -100) > -15)                    # Near highs for continuation
            )
        else:
            mask = pd.Series(False, index=df.index)
        patterns.append(('🚀 VELOCITY BREAKOUT', mask))

        # 11. Institutional Tsunami - Multi-dimensional confluence scoring
        try:
            if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_1y', 'from_high_pct', 'ret_7d', 'ret_30d', 'pe', 'eps_change_pct']):
                ret_7d = get_col_safe('ret_7d', 0)
                ret_30d = get_col_safe('ret_30d', 0)
                ret_1y = get_col_safe('ret_1y', 0)
                pe = get_col_safe('pe')
                eps_change_pct = get_col_safe('eps_change_pct')
                
                # Multi-timeframe volume tsunami scoring (75 points max)
                vol_tsunami_score = (
                    np.where(get_col_safe('vol_ratio_7d_90d', 1) > 2.0, 30, 0) +
                    np.where(get_col_safe('vol_ratio_30d_90d', 1) > 1.5, 25, 0) +
                    np.where(get_col_safe('vol_ratio_90d_180d', 1) > 1.3, 20, 0)
                )
                
                # Hidden strength - institutions accumulating quietly (25 points)
                hidden_strength = np.where(
                    (ret_1y > 50) & (get_col_safe('from_high_pct', -100) < -20), 
                    25, 0
                )
                
                # Fresh acceleration confirmation (15 points)
                with np.errstate(divide='ignore', invalid='ignore'):
                    fresh_accel = np.where(ret_7d > ret_30d / 4, 15, 0)
                
                # Quality confirmation - profitable growth (15 points)
                quality_conf = np.where(
                    pe.notna() & (pe < 30) & eps_change_pct.notna() & (eps_change_pct > 20), 
                    15, 0
                )
                
                # Total tsunami score (max 130 points)
                tsunami_score = vol_tsunami_score + hidden_strength + fresh_accel + quality_conf
                
                # Requires ≥90 points for institutional tsunami pattern
                mask = (tsunami_score >= 90)
            else:
                mask = pd.Series(False, index=df.index)
        except Exception as e:
            mask = pd.Series(False, index=df.index)
            logger.warning(f"Error in INSTITUTIONAL TSUNAMI pattern: {e}")
        patterns.append(('🌋 INSTITUTIONAL TSUNAMI', mask))

        # ========== FUNDAMENTAL PATTERNS (12-16) ==========
    
        # 12. Value Momentum - FIXED: Check master_score existence
        pe = get_col_safe('pe')
        if 'master_score' in df.columns:
            mask = pe.notna() & (pe > 0) & (pe < 15) & (get_col_safe('master_score', 0) >= 70)
        else:
            # Use returns as proxy for score
            mask = pe.notna() & (pe > 0) & (pe < 15) & (get_col_safe('ret_30d', 0) > 20)
        patterns.append(('📈 VALUE MOMENTUM', mask))
        
        # 13. Earnings Rocket - FIXED: acceleration_score doesn't exist
        eps_change_pct = get_col_safe('eps_change_pct')
        if 'acceleration_score' in df.columns:
            mask = eps_change_pct.notna() & (eps_change_pct > 50) & (get_col_safe('acceleration_score', 0) >= 70)
        else:
            # Calculate acceleration directly
            if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d']):
                with np.errstate(divide='ignore', invalid='ignore'):
                    accel = (get_col_safe('ret_1d', 0) > get_col_safe('ret_7d', 0)/7) & \
                           (get_col_safe('ret_7d', 0)/7 > get_col_safe('ret_30d', 0)/30)
                mask = eps_change_pct.notna() & (eps_change_pct > 50) & accel
            else:
                mask = eps_change_pct.notna() & (eps_change_pct > 50)
        patterns.append(('🎯 EARNINGS ROCKET', mask))

        # 14. Quality Leader - Good PE, EPS growth, and percentile
        if all(col in df.columns for col in ['pe', 'eps_change_pct', 'percentile']):
            pe, eps_change_pct, percentile = get_col_safe('pe'), get_col_safe('eps_change_pct'), get_col_safe('percentile')
            mask = pe.notna() & eps_change_pct.notna() & (pe.between(10, 25)) & (eps_change_pct > 20) & (percentile >= 80)
            patterns.append(('🏆 QUALITY LEADER', mask))
        
        # 15. Turnaround Play - Enhanced 5-factor confirmation
        if all(col in df.columns for col in ['eps_change_pct', 'ret_30d', 'vol_ratio_30d_90d', 'from_low_pct', 'pe']):
            eps_change_pct = get_col_safe('eps_change_pct')
            pe = get_col_safe('pe')
            mask = (
                eps_change_pct.notna() & pe.notna() &
                (eps_change_pct > 100) &                                       # Massive improvement
                (get_col_safe('ret_30d', 0) > 15) &                           # Recent momentum confirmation
                (get_col_safe('vol_ratio_30d_90d', 1) > 1.5) &                # Volume confirmation
                (get_col_safe('from_low_pct', 0) < 60) &                      # Still reasonable entry point
                (pe < 30)                                                      # Valuation protection
            )
        else:
            # Fallback to basic version if enhanced columns not available
            eps_change_pct = get_col_safe('eps_change_pct')
            mask = eps_change_pct.notna() & (eps_change_pct > 100) & (get_col_safe('volume_score', 0) >= 60)
        patterns.append(('🔄 TURNAROUND', mask))
        
        # 16. High PE Warning - Realistic threshold for Indian markets
        pe = get_col_safe('pe')
        mask = pe.notna() & (pe > 50)  # Lowered from 100 to 50 for practical screening
        patterns.append(('⚠️ HIGH PE', mask))

        # ========== RANGE PATTERNS (17-20) ==========
        
        # 17. 52W High Approach
        mask = (
            (get_col_safe('from_high_pct', -100) > -5) & 
            (get_col_safe('volume_score', 0) >= 70) & 
            (get_col_safe('momentum_score', 0) >= 60)
        )
        patterns.append(('🎲 52W HIGH APPROACH', mask))
        
        # 18. 52W Low Bounce
        mask = (
            (get_col_safe('from_low_pct', 100) < 20) & 
            (get_col_safe('acceleration_score', 0) >= 80) & 
            (get_col_safe('ret_30d', 0) > 10)
        )
        patterns.append(('↗️ 52W LOW BOUNCE', mask))
        
        # 19. Momentum Divergence
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'acceleration_score', 'rvol']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d_pace = np.where(df['ret_7d'].fillna(0) != 0, df['ret_7d'].fillna(0) / 7, np.nan)
                daily_30d_pace = np.where(df['ret_30d'].fillna(0) != 0, df['ret_30d'].fillna(0) / 30, np.nan)
            mask = (
                pd.Series(daily_7d_pace > daily_30d_pace * 1.5, index=df.index).fillna(False) & 
                (get_col_safe('acceleration_score', 0) >= 85) & 
                (get_col_safe('rvol', 0) > 2)
            )
            patterns.append(('🔀 MOMENTUM DIVERGE', mask))
        
        # 20. Range Compression - Smart 20-day volatility approach
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'from_low_pct', 'rvol']):
            # Calculate recent volatility as proxy for range compression
            volatility_7d = abs(get_col_safe('ret_7d', 0))
            volatility_30d = abs(get_col_safe('ret_30d', 0)) / 4  # Daily equivalent
            from_low_pct = get_col_safe('from_low_pct', 0)
            
            # Range compression: low recent volatility + not at extremes + building volume
            mask = (
                (volatility_7d < 5) &          # Low 7-day volatility (tight range)
                (volatility_30d < 3) &         # Low daily volatility over month  
                (from_low_pct > 20) &          # Not at 52W lows
                (from_low_pct < 80) &          # Not at 52W highs
                (get_col_safe('rvol', 0) > 0.8)  # Some volume interest
            )
            patterns.append(('🤏 RANGE COMPRESS', mask))

        # ========== INTELLIGENCE PATTERNS (21-23) ==========
        
        # 21. Stealth Accumulator
        if all(col in df.columns for col in ['vol_ratio_90d_180d', 'vol_ratio_30d_90d', 'from_low_pct', 'ret_7d', 'ret_30d']):
            ret_7d, ret_30d = get_col_safe('ret_7d'), get_col_safe('ret_30d')
            with np.errstate(divide='ignore', invalid='ignore'):
                ret_ratio = pd.Series(
                    np.where(ret_30d != 0, ret_7d / (ret_30d / 4), np.nan), 
                    index=df.index
                ).fillna(0)
            mask = (
                (get_col_safe('vol_ratio_90d_180d', 1) > 1.1) & 
                (get_col_safe('vol_ratio_30d_90d', 1).between(0.9, 1.1)) & 
                (get_col_safe('from_low_pct', 0) > 40) & 
                (ret_ratio > 1)
            )
            patterns.append(('🤫 STEALTH', mask))

        # 22. 🏎️ ACCELERATION - Momentum acceleration pattern
        if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'rvol', 'from_high_pct', 'category']):
            ret_1d, ret_7d = get_col_safe('ret_1d'), get_col_safe('ret_7d')
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_pace_ratio = pd.Series(
                    np.where(ret_7d != 0, ret_1d / (ret_7d / 7), np.nan), 
                    index=df.index
                ).fillna(0)
            mask = (
                (daily_pace_ratio > 2) & 
                (get_col_safe('rvol', 0) > 3) & 
                (get_col_safe('from_high_pct', -100) > -15) & 
                (df['category'].isin(['Small Cap', 'Micro Cap']))
            )
            patterns.append(('🏎️ ACCELERATION', mask))
        
        # 23. Perfect Storm
        if 'momentum_harmony' in df.columns and 'master_score' in df.columns:
            mask = (
                (get_col_safe('momentum_harmony', 0) == 4) & 
                (get_col_safe('ret_30d', 0) > 20) &  # Use return data instead
                (get_col_safe('rvol', 0) > 2)  # Add volume confirmation
            )
            patterns.append(('⛈️ PERFECT STORM', mask))

        # ========== REVERSAL & CONTINUATION PATTERNS (24-34) ==========
        
        # 24. BULL TRAP - Failed breakout/shorting opportunity
        if all(col in df.columns for col in ['from_high_pct', 'ret_7d', 'volume_7d', 'volume_30d']):
            mask = (
                (get_col_safe('from_high_pct', -100) > -5) &     # Was near 52W high
                (get_col_safe('ret_7d', 0) < -10) &              # Now falling hard
                (get_col_safe('volume_7d', 0) > get_col_safe('volume_30d', 1))  # High volume selling
            )
            patterns.append(('🪤 BULL TRAP', mask))
        
        # 25. CAPITULATION BOTTOM - Panic selling exhaustion
        if all(col in df.columns for col in ['ret_1d', 'from_low_pct', 'rvol', 'volume_1d', 'volume_90d']):
            mask = (
                (get_col_safe('ret_1d', 0) < -7) &               # Huge down day
                (get_col_safe('from_low_pct', 100) < 20) &       # Near 52W low
                (get_col_safe('rvol', 0) > 5) &                  # Extreme volume
                (get_col_safe('volume_1d', 0) > get_col_safe('volume_90d', 1) * 3)  # Panic volume
            )
            patterns.append(('💣 CAPITULATION', mask))
        
        # 26. RUNAWAY GAP - Continuation pattern
        if all(col in df.columns for col in ['price', 'prev_close', 'ret_30d', 'rvol', 'from_high_pct']):
            price = get_col_safe('price', 0)
            prev_close = get_col_safe('prev_close', 1)
            
            # Calculate gap percentage safely
            with np.errstate(divide='ignore', invalid='ignore'):
                gap = np.where(prev_close > 0, 
                              ((price - prev_close) / prev_close) * 100,
                              0)
            gap_series = pd.Series(gap, index=df.index)
            
            mask = (
                (gap_series > 5) &                               # Big gap up
                (get_col_safe('ret_30d', 0) > 20) &             # Already trending
                (get_col_safe('rvol', 0) > 3) &                 # Institutional volume
                (get_col_safe('from_high_pct', -100) > -3)      # Making new highs
            )
            patterns.append(('🏃 RUNAWAY GAP', mask))
        
        # 27. ROTATION LEADER - First mover in sector rotation
        if all(col in df.columns for col in ['ret_7d', 'sector', 'rvol']):
            ret_7d = get_col_safe('ret_7d', 0)
            
            # Calculate sector average return safely
            if 'sector' in df.columns:
                sector_avg = df.groupby('sector')['ret_7d'].transform('mean').fillna(0)
            else:
                sector_avg = pd.Series(0, index=df.index)
            
            mask = (
                (ret_7d > sector_avg + 5) &                      # Beating sector by 5%
                (ret_7d > 0) &                                   # Positive absolute return
                (sector_avg < 0) &                               # Sector still negative
                (get_col_safe('rvol', 0) > 2)                   # Volume confirmation
            )
            patterns.append(('🔃 ROTATION LEADER', mask))
        
        # 28. DISTRIBUTION TOP - Smart money selling
        if all(col in df.columns for col in ['from_high_pct', 'rvol', 'ret_1d', 'ret_30d', 'volume_7d', 'volume_30d']):
            mask = (
                (get_col_safe('from_high_pct', -100) > -10) &    # Near highs
                (get_col_safe('rvol', 0) > 2) &                  # High volume
                (get_col_safe('ret_1d', 0) < 2) &                # Price not moving up
                (get_col_safe('ret_30d', 0) > 25) &              # After reasonable rally (reduced from 50%)
                (get_col_safe('volume_7d', 0) > get_col_safe('volume_30d', 1) * 1.5)  # Volume spike
            )
            patterns.append(('📊 DISTRIBUTION', mask))

        # 29. VELOCITY SQUEEZE
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'from_high_pct', 'from_low_pct', 'high_52w', 'low_52w']):
            with np.errstate(divide='ignore', invalid='ignore'):
                daily_7d = np.where(df['ret_7d'] != 0, df['ret_7d'] / 7, 0)
                daily_30d = np.where(df['ret_30d'] != 0, df['ret_30d'] / 30, 0)
                range_pct = np.where(df['low_52w'] > 0, 
                                   (df['high_52w'] - df['low_52w']) / df['low_52w'], 
                                   np.inf)
            
            mask = (
                (daily_7d > daily_30d) &  # Velocity increasing
                (abs(df['from_high_pct']) + df['from_low_pct'] < 30) &  # Middle of range
                (range_pct < 0.5)  # Tight range
            )
            patterns.append(('🗜️ VELOCITY SQUEEZE', mask))
        
        # 30. VOLUME DIVERGENCE WARNING - Clarified bearish divergence logic
        if all(col in df.columns for col in ['ret_7d', 'ret_30d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'from_high_pct']):
            # Classic bearish divergence: price making new highs but volume declining
            mask = (
                (df['ret_7d'] > 5) &              # Recent price strength
                (df['ret_30d'] > 15) &            # Monthly price advance
                (df['vol_ratio_7d_90d'] < 0.8) &  # Recent volume declining
                (df['vol_ratio_30d_90d'] < 0.9) & # Monthly volume declining
                (df['from_high_pct'] > -10)       # Near highs but volume not confirming
            )
            patterns.append(('🔉 VOLUME DIVERGENCE', mask))
        
        # 31. GOLDEN CROSS MOMENTUM
        if all(col in df.columns for col in ['sma_20d', 'sma_50d', 'sma_200d', 'rvol', 'ret_7d', 'ret_30d']):
            mask = (
                (df['sma_20d'] > df['sma_50d']) &
                (df['sma_50d'] > df['sma_200d']) &
                ((df['sma_20d'] - df['sma_50d']) / df['sma_50d'] > 0.02) &
                (df['rvol'] > 1.5) &
                (df['ret_7d'] > df['ret_30d'] / 4)
            )
            patterns.append(('✨ GOLDEN CROSS', mask))
        
        # 32. MOMENTUM EXHAUSTION
        if all(col in df.columns for col in ['ret_7d', 'ret_1d', 'rvol', 'from_low_pct', 'price', 'sma_20d']):
            with np.errstate(divide='ignore', invalid='ignore'):
                sma_deviation = np.where(df['sma_20d'] > 0,
                                        (df['price'] - df['sma_20d']) / df['sma_20d'],
                                        0)
            
            # Handle RVOL shift safely
            rvol_shifted = df['rvol'].shift(1).fillna(df['rvol'].median())
            
            mask = (
                (df['ret_7d'] > 25) &
                (df['ret_1d'] < 0) &
                (df['rvol'] < rvol_shifted) &
                (df['from_low_pct'] > 80) &
                (sma_deviation > 0.15)
            )
            patterns.append(('📉 EXHAUSTION', mask))
        
        # 33. PYRAMID ACCUMULATION
        if all(col in df.columns for col in ['vol_ratio_7d_90d', 'vol_ratio_30d_90d', 'vol_ratio_90d_180d', 'ret_30d', 'from_low_pct']):
            mask = (
                (df['vol_ratio_7d_90d'] > 1.1) &
                (df['vol_ratio_30d_90d'] > 1.05) &
                (df['vol_ratio_90d_180d'] > 1.02) &
                (df['ret_30d'].between(5, 15)) &
                (df['from_low_pct'] < 50)
            )
            patterns.append(('🔺 PYRAMID', mask))
        
        # 34. MOMENTUM VACUUM
        if all(col in df.columns for col in ['ret_30d', 'ret_7d', 'ret_1d', 'rvol', 'from_low_pct']):
            mask = (
                (df['ret_30d'] < -20) &
                (df['ret_7d'] > 0) &
                (df['ret_1d'] > 2) &
                (df['rvol'] > 3) &
                (df['from_low_pct'] < 10)
            )
            patterns.append(('🌪️ VACUUM', mask))

        # 35. EARNINGS SURPRISE LEADER - Multi-timeframe earnings acceleration with pace calculation
        try:
            if all(col in df.columns for col in ['eps_change_pct', 'ret_1d', 'ret_7d', 'ret_30d', 'pe', 'vol_ratio_30d_90d', 'price', 'sma_20d']):
                eps_change_pct = get_col_safe('eps_change_pct', 0)
                pe = get_col_safe('pe', 100)
                price = get_col_safe('price', 0)
                sma_20d = get_col_safe('sma_20d', 0)
                
                # Calculate sector median PE safely
                if 'sector' in df.columns:
                    sector_pe_median = df.groupby('sector')['pe'].transform('median').fillna(20)
                else:
                    sector_pe_median = pd.Series(20, index=df.index)
                
                # Safe pace calculation with sophisticated velocity analysis
                with np.errstate(divide='ignore', invalid='ignore'):
                    ret_7d = get_col_safe('ret_7d', 0)
                    ret_30d = get_col_safe('ret_30d', 0)
                    weekly_vs_monthly_pace = np.where(ret_30d != 0, ret_7d / (ret_30d / 4), 0)
                
                mask = (
                    # Fundamental acceleration - EPS growth exceeding price growth
                    eps_change_pct.notna() & (eps_change_pct > 50) &
                    (eps_change_pct > ret_30d) &                                 # EPS > price growth
                    
                    # Valuation opportunity - below sector median
                    pe.notna() & (pe < sector_pe_median) &
                    
                    # Multi-timeframe momentum building
                    (get_col_safe('ret_1d', 0) > 0) &
                    (weekly_vs_monthly_pace > 1) &                               # Accelerating pace
                    
                    # Volume confirmation
                    (get_col_safe('vol_ratio_30d_90d', 1) > 1.2) &
                    
                    # Technical position above trend
                    price.notna() & sma_20d.notna() & (price > sma_20d)
                )
                patterns.append(('🎆 EARNINGS SURPRISE LEADER', mask))
        except Exception as e:
            logger.warning(f"Error in EARNINGS SURPRISE LEADER pattern: {e}")
            patterns.append(('🎆 EARNINGS SURPRISE LEADER', pd.Series(False, index=df.index)))

        # 36. INFORMATION DECAY ARBITRAGE - Advanced information theory application
        try:
            if all(col in df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'vol_ratio_7d_90d', 'vol_ratio_30d_90d']):
                # Information decay modeling with empirically derived half-lives
                SHORT_HALFLIFE = 3.5    # days - market microstructure response
                MEDIUM_HALFLIFE = 14    # days - institutional response time
                LONG_HALFLIFE = 45      # days - fundamental analysis integration
                
                ret_1d = get_col_safe('ret_1d', 0)
                ret_3d = get_col_safe('ret_3d', 0)
                ret_7d = get_col_safe('ret_7d', 0)
                ret_30d = get_col_safe('ret_30d', 0)
                
                # Calculate theoretical decay curves using exponential decay model
                with np.errstate(divide='ignore', invalid='ignore'):
                    theoretical_3d = ret_1d * np.exp(-3/SHORT_HALFLIFE)
                    theoretical_7d = ret_1d * np.exp(-7/SHORT_HALFLIFE)
                    theoretical_30d = ret_1d * np.exp(-30/MEDIUM_HALFLIFE)
                    
                    # Calculate normalized decay rate anomalies
                    decay_anomaly_3d = np.where(theoretical_3d != 0, (ret_3d - theoretical_3d) / abs(theoretical_3d), 0)
                    decay_anomaly_7d = np.where(theoretical_7d != 0, (ret_7d - theoretical_7d) / abs(theoretical_7d), 0)
                    decay_anomaly_30d = np.where(theoretical_30d != 0, (ret_30d - theoretical_30d) / abs(theoretical_30d), 0)
                
                # Multi-dimensional decay score with adaptive weighting
                decay_score = (
                    (decay_anomaly_3d > 0.15).astype(int) * 25 +      # Short-term underpricing
                    (decay_anomaly_7d > 0.10).astype(int) * 35 +      # Medium-term persistence
                    (decay_anomaly_30d > 0.05).astype(int) * 40       # Long-term sustainability
                )
                
                # Volume stealth confirmation - building interest but not explosive
                stealth_volume = (
                    (get_col_safe('vol_ratio_7d_90d', 1) > 1.1) &     # Building interest
                    (get_col_safe('vol_ratio_7d_90d', 1) < 2.0) &     # But not explosive yet
                    (get_col_safe('vol_ratio_30d_90d', 1) > 1.05)     # Consistent trend
                )
                
                # Position and quality filters for implementation
                arbitrage_setup = (
                    (get_col_safe('from_high_pct', 0) < -8) &         # Room to move up
                    (get_col_safe('from_low_pct', 0) > 25) &          # Not at lows
                    (get_col_safe('price', 0) > get_col_safe('sma_50d', 0) * 0.95) &  # Technical support
                    (get_col_safe('pe', 100) < 40) &                 # Not extremely overvalued
                    (get_col_safe('eps_change_pct', 0) > -5)          # Earnings not collapsing
                )
                
                # Final information decay arbitrage detection
                mask = (
                    (decay_score >= 60) &
                    stealth_volume &
                    arbitrage_setup
                )
                patterns.append(('🕰️ INFORMATION DECAY ARBITRAGE', mask))
        except Exception as e:
            logger.warning(f"Error in INFORMATION DECAY ARBITRAGE pattern: {e}")
            patterns.append(('🕰️ INFORMATION DECAY ARBITRAGE', pd.Series(False, index=df.index)))

        # 37. PHOENIX RISING - Epic comeback with dramatic transformation
        try:
            if all(col in df.columns for col in ['from_low_pct', 'eps_change_pct', 'rvol', 'vol_ratio_90d_180d', 'pe']):
                from_low_pct = get_col_safe('from_low_pct', 0)
                eps_change_pct = get_col_safe('eps_change_pct', 0)
                pe = get_col_safe('pe', 100)
                
                mask = (
                    # Strong recovery from significant lows - realistic comeback
                    (from_low_pct > 40) &
                    
                    # Substantial fundamental turnaround - achievable transformation
                    eps_change_pct.notna() & (eps_change_pct > 75) &
                    
                    # Notable volume interest - institutional recognition
                    (get_col_safe('rvol', 0) > 2.5) &
                    (get_col_safe('vol_ratio_90d_180d', 1) > 1.4) &
                    
                    # Quality confirmation - reasonable valuation after turnaround
                    pe.notna() & (pe > 0) & (pe < 60)
                )
                patterns.append(('🐦 PHOENIX RISING', ensure_series(mask)))
        except Exception as e:
            logger.warning(f"Error in PHOENIX RISING pattern: {e}")
            patterns.append(('🐦 PHOENIX RISING', pd.Series(False, index=df.index)))

        # 38. ATOMIC DECAY MOMENTUM - Physics-based momentum timing using radioactive decay mathematics
        try:
            if all(col in df.columns for col in ['ret_1d', 'ret_7d', 'ret_30d', 'rvol', 'from_low_pct', 'acceleration_score']):
                ret_1d = get_col_safe('ret_1d', 0)
                ret_7d = get_col_safe('ret_7d', 0)
                ret_30d = get_col_safe('ret_30d', 0)
                
                # Calculate momentum half-life using atomic decay physics (t½ = ln(2)/λ)
                with np.errstate(divide='ignore', invalid='ignore'):
                    # Momentum decay rate calculation - ratio of short to long momentum
                    momentum_ratio = np.where(ret_30d != 0, abs(ret_7d / ret_30d), 0.1)
                    momentum_decay_rate = np.log(2) / np.maximum(momentum_ratio, 0.1)  # Half-life calculation
                
                # Atomic momentum strength criteria (undecayed energy state)
                atomic_strength = (
                    (momentum_decay_rate > 2) &                      # Slow decay rate (sustained momentum)
                    (ret_7d > 3) &                                   # Strong recent momentum
                    (ret_30d > 8) &                                  # Sustained longer momentum
                    (get_col_safe('rvol', 1) > 1.3) &               # Volume confirmation
                    (get_col_safe('from_low_pct', 0) > 15) &        # Off lows
                    (get_col_safe('acceleration_score', 0) >= 75)    # Accelerating
                )
                
                patterns.append(('⚛️ ATOMIC DECAY MOMENTUM', ensure_series(atomic_strength)))
        except Exception as e:
            logger.warning(f"Error in ATOMIC DECAY MOMENTUM pattern: {e}")
            patterns.append(('⚛️ ATOMIC DECAY MOMENTUM', pd.Series(False, index=df.index)))

        # 39. GARP LEADER - Growth At Reasonable Price methodology
        try:
            if all(col in df.columns for col in ['eps_change_pct', 'pe', 'ret_6m', 'from_low_pct']):
                eps_change_pct = get_col_safe('eps_change_pct')
                pe = get_col_safe('pe')
                
                mask = (
                    eps_change_pct.notna() & pe.notna() &
                    (eps_change_pct > 20) &                                        # Strong growth
                    (pe.between(8, 25)) &                                          # Reasonable valuation
                    (get_col_safe('ret_6m', 0) > 10) &                            # Market recognition
                    (get_col_safe('from_low_pct', 0) > 40)                        # Not oversold
                )
                patterns.append(('💹 GARP LEADER', ensure_series(mask)))
        except Exception as e:
            logger.warning(f"Error in GARP LEADER pattern: {e}")
            patterns.append(('💹 GARP LEADER', pd.Series(False, index=df.index)))

        # 40. PULLBACK SUPPORT - Realistic support bounce detection
        try:
            if all(col in df.columns for col in ['price', 'sma_20d', 'sma_200d', 'ret_1d', 'rvol']):
                price = get_col_safe('price', 0)
                sma_200d = get_col_safe('sma_200d', 0)
                sma_20d = get_col_safe('sma_20d', 0)
                
                # Calculate realistic support zone around 20-day SMA (±3% for practical trading)
                with np.errstate(divide='ignore', invalid='ignore'):
                    support_zone_low = sma_20d * 0.97   # Widened from 0.98 to 0.97
                    support_zone_high = sma_20d * 1.03  # Widened from 1.02 to 1.03
                
                mask = (
                    price.notna() & sma_200d.notna() & sma_20d.notna() &
                    (price > sma_200d) &                                          # Above long-term trend
                    (price >= support_zone_low) & (price <= support_zone_high) &  # Near 20-day SMA (±3%)
                    (get_col_safe('ret_1d', 0) > 0) &                            # Bouncing
                    (get_col_safe('rvol', 0) > 1.2)                              # Lowered volume threshold for practicality
                )
                patterns.append(('🛡️ PULLBACK SUPPORT', ensure_series(mask)))
        except Exception as e:
            logger.warning(f"Error in PULLBACK SUPPORT pattern: {e}")
            patterns.append(('🛡️ PULLBACK SUPPORT', pd.Series(False, index=df.index)))

        # 41. OVERSOLD QUALITY - Value opportunity identification
        try:
            if all(col in df.columns for col in ['from_low_pct', 'eps_change_pct', 'pe', 'ret_1d', 'rvol']):
                eps_change_pct = get_col_safe('eps_change_pct')
                pe = get_col_safe('pe')
                
                mask = (
                    (get_col_safe('from_low_pct', 0) < 25) &                      # Oversold
                    eps_change_pct.notna() & (eps_change_pct > 0) &               # Still growing
                    pe.notna() & (pe < 20) &                                      # Reasonable valuation
                    (get_col_safe('ret_1d', 0) > 1) &                            # Starting to bounce
                    (get_col_safe('rvol', 0) > 1.5)                              # Interest building
                )
                patterns.append(('💳 OVERSOLD QUALITY', ensure_series(mask)))
        except Exception as e:
            logger.warning(f"Error in OVERSOLD QUALITY pattern: {e}")
            patterns.append(('💳 OVERSOLD QUALITY', pd.Series(False, index=df.index)))
        
        # Ensure all patterns have Series masks
        patterns = [(name, ensure_series(mask)) for name, mask in patterns]
        
        return patterns
    
    @staticmethod
    def get_pattern_summary(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a summary of pattern detections
        """
        if 'patterns' not in df.columns:
            return pd.DataFrame()
        
        pattern_counts = {}
        pattern_stocks = {}
        
        for idx, patterns_str in df['patterns'].items():
            if patterns_str:
                for pattern in patterns_str.split(' | '):
                    pattern = pattern.strip()
                    if pattern:
                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                        if pattern not in pattern_stocks:
                            pattern_stocks[pattern] = []
                        pattern_stocks[pattern].append(df.loc[idx, 'ticker'])
        
        if not pattern_counts:
            return pd.DataFrame()
        
        # Create summary dataframe
        summary_data = []
        for pattern, count in pattern_counts.items():
            meta = PatternDetector.PATTERN_METADATA.get(pattern, {})
            top_stocks = pattern_stocks[pattern][:3]
            
            summary_data.append({
                'Pattern': pattern,
                'Count': count,
                'Weight': meta.get('importance_weight', 0),
                'Category': meta.get('category', 'unknown'),
                'Top Stocks': ', '.join(top_stocks)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Count', ascending=False)
        
        return summary_df 
        
# ============================================
# LEADERSHIP DENSITY ENGINE - ALL-TIME BEST APPROACH
# ============================================

class LeadershipDensityEngine:
    """
    Revolutionary Leadership Density Index (LDI) approach for sector/industry/category analysis.
    
    This approach measures sector strength through leadership density rather than sampling bias.
    LDI = (Number of Market Leaders in Group) / (Total Stocks in Group) × 100
    
    Key Benefits:
    - No sampling bias - uses entire universe
    - Fair comparison across all group sizes  
    - Market-relative leadership assessment
    - True sector strength measurement
    """
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _calculate_global_threshold_cached(df_json: str, percentile: float = 90.0) -> float:
        """Cached calculation of global market leadership threshold"""
        df = pd.read_json(df_json)
        
        if df.empty or 'master_score' not in df.columns:
            return 75.0  # Default threshold
        
        # Calculate global threshold (top 10% of entire market)
        threshold = df['master_score'].quantile(percentile / 100.0)
        return max(threshold, 60.0)  # Minimum threshold of 60
    
    @staticmethod
    def calculate_global_threshold(df: pd.DataFrame, percentile: float = 90.0) -> float:
        """Calculate global market leadership threshold (top 10% default)"""
        if df.empty or 'master_score' not in df.columns:
            return 75.0
        
        try:
            # Use only relevant columns for caching
            cache_df = df[['master_score']].copy()
            df_json = cache_df.to_json()
            return LeadershipDensityEngine._calculate_global_threshold_cached(df_json, percentile)
        except Exception as e:
            logger.warning(f"Cache failed for global threshold: {str(e)}")
            # Fallback calculation
            threshold = df['master_score'].quantile(percentile / 100.0)
            return max(threshold, 60.0)
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _calculate_sector_ldi_cached(df_json: str, threshold: float) -> pd.DataFrame:
        """Cached calculation of sector Leadership Density Index"""
        df = pd.read_json(df_json)
        
        if df.empty or 'sector' not in df.columns or 'master_score' not in df.columns:
            return pd.DataFrame()
        
        # Calculate LDI for each sector
        sector_stats = []
        
        for sector in df['sector'].unique():
            if sector == 'Unknown':
                continue
                
            sector_df = df[df['sector'] == sector]
            total_stocks = len(sector_df)
            
            if total_stocks == 0:
                continue
            
            # Count market leaders in this sector
            leaders = sector_df[sector_df['master_score'] >= threshold]
            leader_count = len(leaders)
            
            # Calculate Leadership Density Index
            ldi = (leader_count / total_stocks) * 100
            
            # Additional metrics for enhanced analysis
            avg_score = sector_df['master_score'].mean()
            median_score = sector_df['master_score'].median()
            top_10_pct_count = max(1, int(total_stocks * 0.1))
            top_performers = sector_df.nlargest(top_10_pct_count, 'master_score')
            elite_avg_score = top_performers['master_score'].mean()
            
            # Calculate momentum and volume metrics if available
            avg_momentum = sector_df['momentum_score'].mean() if 'momentum_score' in sector_df.columns else 50.0
            avg_volume = sector_df['volume_score'].mean() if 'volume_score' in sector_df.columns else 50.0
            avg_rvol = sector_df['rvol'].mean() if 'rvol' in sector_df.columns else 1.0
            avg_ret_30d = sector_df['ret_30d'].mean() if 'ret_30d' in sector_df.columns else 0.0
            
            # Money flow if available
            total_money_flow = sector_df['money_flow_mm'].sum() if 'money_flow_mm' in sector_df.columns else 0.0
            
            sector_stats.append({
                'sector': sector,
                'ldi_score': round(ldi, 2),
                'leader_count': leader_count,
                'total_stocks': total_stocks,
                'avg_score': round(avg_score, 2),
                'median_score': round(median_score, 2),
                'elite_avg_score': round(elite_avg_score, 2),
                'avg_momentum': round(avg_momentum, 2),
                'avg_volume': round(avg_volume, 2),
                'avg_rvol': round(avg_rvol, 2),
                'avg_ret_30d': round(avg_ret_30d, 2),
                'total_money_flow': round(total_money_flow, 2),
                'leadership_density': f"{ldi:.1f}%"
            })
        
        if not sector_stats:
            return pd.DataFrame()
        
        # Create DataFrame and sort by LDI score
        ldi_df = pd.DataFrame(sector_stats)
        ldi_df = ldi_df.sort_values('ldi_score', ascending=False)
        ldi_df.set_index('sector', inplace=True)
        
        return ldi_df
    
    @staticmethod
    def calculate_sector_ldi(df: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
        """Calculate Leadership Density Index for all sectors"""
        if df.empty or 'sector' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Calculate global threshold
            threshold = LeadershipDensityEngine.calculate_global_threshold(df, percentile)
            
            # Use only relevant columns for caching
            cache_cols = ['sector', 'master_score']
            optional_cols = ['momentum_score', 'volume_score', 'rvol', 'ret_30d', 'money_flow_mm']
            cache_cols.extend([col for col in optional_cols if col in df.columns])
            
            cache_df = df[cache_cols].copy()
            df_json = cache_df.to_json()
            
            return LeadershipDensityEngine._calculate_sector_ldi_cached(df_json, threshold)
        except Exception as e:
            logger.error(f"Error calculating sector LDI: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _calculate_industry_ldi_cached(df_json: str, threshold: float) -> pd.DataFrame:
        """Cached calculation of industry Leadership Density Index"""
        df = pd.read_json(df_json)
        
        if df.empty or 'industry' not in df.columns or 'master_score' not in df.columns:
            return pd.DataFrame()
        
        # Calculate LDI for each industry
        industry_stats = []
        
        for industry in df['industry'].unique():
            if industry == 'Unknown':
                continue
                
            industry_df = df[df['industry'] == industry]
            total_stocks = len(industry_df)
            
            if total_stocks == 0:
                continue
            
            # Count market leaders in this industry
            leaders = industry_df[industry_df['master_score'] >= threshold]
            leader_count = len(leaders)
            
            # Calculate Leadership Density Index
            ldi = (leader_count / total_stocks) * 100
            
            # Additional metrics
            avg_score = industry_df['master_score'].mean()
            median_score = industry_df['master_score'].median()
            
            # Quality assessment
            quality_flag = ''
            if total_stocks < 5:
                quality_flag = '⚠️ Small Sample'
            elif ldi == 0 and total_stocks > 10:
                quality_flag = '📉 No Leaders'
            elif ldi > 20:
                quality_flag = '🔥 High Density'
            
            # Calculate momentum and volume metrics if available
            avg_momentum = industry_df['momentum_score'].mean() if 'momentum_score' in industry_df.columns else 50.0
            avg_volume = industry_df['volume_score'].mean() if 'volume_score' in industry_df.columns else 50.0
            
            industry_stats.append({
                'industry': industry,
                'ldi_score': round(ldi, 2),
                'leader_count': leader_count,
                'total_stocks': total_stocks,
                'avg_score': round(avg_score, 2),
                'median_score': round(median_score, 2),
                'avg_momentum': round(avg_momentum, 2),
                'avg_volume': round(avg_volume, 2),
                'quality_flag': quality_flag,
                'leadership_density': f"{ldi:.1f}%"
            })
        
        if not industry_stats:
            return pd.DataFrame()
        
        # Create DataFrame and sort by LDI score
        ldi_df = pd.DataFrame(industry_stats)
        ldi_df = ldi_df.sort_values('ldi_score', ascending=False)
        ldi_df.set_index('industry', inplace=True)
        
        return ldi_df
    
    @staticmethod
    def calculate_industry_ldi(df: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
        """Calculate Leadership Density Index for all industries"""
        if df.empty or 'industry' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Calculate global threshold
            threshold = LeadershipDensityEngine.calculate_global_threshold(df, percentile)
            
            # Use only relevant columns for caching
            cache_cols = ['industry', 'master_score']
            optional_cols = ['momentum_score', 'volume_score']
            cache_cols.extend([col for col in optional_cols if col in df.columns])
            
            cache_df = df[cache_cols].copy()
            df_json = cache_df.to_json()
            
            return LeadershipDensityEngine._calculate_industry_ldi_cached(df_json, threshold)
        except Exception as e:
            logger.error(f"Error calculating industry LDI: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _calculate_category_ldi_cached(df_json: str, threshold: float) -> pd.DataFrame:
        """Cached calculation of category Leadership Density Index"""
        df = pd.read_json(df_json)
        
        if df.empty or 'category' not in df.columns or 'master_score' not in df.columns:
            return pd.DataFrame()
        
        # Calculate LDI for each category
        category_stats = []
        
        for category in df['category'].unique():
            if category == 'Unknown':
                continue
                
            category_df = df[df['category'] == category]
            total_stocks = len(category_df)
            
            if total_stocks == 0:
                continue
            
            # Count market leaders in this category
            leaders = category_df[category_df['master_score'] >= threshold]
            leader_count = len(leaders)
            
            # Calculate Leadership Density Index
            ldi = (leader_count / total_stocks) * 100
            
            # Additional metrics
            avg_score = category_df['master_score'].mean()
            avg_percentile = category_df['category_percentile'].mean() if 'category_percentile' in category_df.columns else 50.0
            total_money_flow = category_df['money_flow_mm'].sum() if 'money_flow_mm' in category_df.columns else 0.0
            
            category_stats.append({
                'category': category,
                'ldi_score': round(ldi, 2),
                'leader_count': leader_count,
                'total_stocks': total_stocks,
                'avg_score': round(avg_score, 2),
                'avg_percentile': round(avg_percentile, 2),
                'total_money_flow': round(total_money_flow, 2),
                'leadership_density': f"{ldi:.1f}%"
            })
        
        if not category_stats:
            return pd.DataFrame()
        
        # Create DataFrame and sort by LDI score
        ldi_df = pd.DataFrame(category_stats)
        ldi_df = ldi_df.sort_values('ldi_score', ascending=False)
        ldi_df.set_index('category', inplace=True)
        
        return ldi_df
    
    @staticmethod
    def calculate_category_ldi(df: pd.DataFrame, percentile: float = 90.0) -> pd.DataFrame:
        """Calculate Leadership Density Index for all categories"""
        if df.empty or 'category' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Calculate global threshold
            threshold = LeadershipDensityEngine.calculate_global_threshold(df, percentile)
            
            # Use only relevant columns for caching
            cache_cols = ['category', 'master_score']
            optional_cols = ['category_percentile', 'money_flow_mm']
            cache_cols.extend([col for col in optional_cols if col in df.columns])
            
            cache_df = df[cache_cols].copy()
            df_json = cache_df.to_json()
            
            return LeadershipDensityEngine._calculate_category_ldi_cached(df_json, threshold)
        except Exception as e:
            logger.error(f"Error calculating category LDI: {str(e)}")
            return pd.DataFrame()

# ============================================
# MARKET INTELLIGENCE
# ============================================

class MarketIntelligence:
    """Professional market regime detection and sector analysis"""
    
    @staticmethod
    def detect_market_regime(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Detect current market regime with supporting data"""
        
        if df.empty:
            return "NO DATA", {}
        
        metrics = {}
        
        # Calculate category-based metrics
        if 'category' in df.columns and 'master_score' in df.columns:
            category_scores = df.groupby('category')['master_score'].mean()
            
            micro_small_avg = category_scores[category_scores.index.isin(['Micro Cap', 'Small Cap'])].mean() if any(category_scores.index.isin(['Micro Cap', 'Small Cap'])) else 50
            large_mega_avg = category_scores[category_scores.index.isin(['Large Cap', 'Mega Cap'])].mean() if any(category_scores.index.isin(['Large Cap', 'Mega Cap'])) else 50
            
            metrics['micro_small_avg'] = micro_small_avg if pd.notna(micro_small_avg) else 50
            metrics['large_mega_avg'] = large_mega_avg if pd.notna(large_mega_avg) else 50
            metrics['category_spread'] = metrics['micro_small_avg'] - metrics['large_mega_avg']
        else:
            metrics['micro_small_avg'] = 50
            metrics['large_mega_avg'] = 50
            metrics['category_spread'] = 0
        
        # Calculate market breadth
        if 'ret_30d' in df.columns:
            breadth = len(df[df['ret_30d'] > 0]) / len(df) if len(df) > 0 else 0.5
            metrics['breadth'] = breadth
        else:
            breadth = 0.5
            metrics['breadth'] = breadth
        
        # Calculate volatility metrics
        if 'rvol' in df.columns:
            avg_rvol = df['rvol'].median()
            metrics['avg_rvol'] = avg_rvol if pd.notna(avg_rvol) else 1.0
        else:
            metrics['avg_rvol'] = 1.0
        
        # Determine market regime professionally
        if metrics['micro_small_avg'] > metrics['large_mega_avg'] + 10 and breadth > 0.6:
            regime = "🔥 RISK-ON BULL"
        elif metrics['large_mega_avg'] > metrics['micro_small_avg'] + 10 and breadth < 0.4:
            regime = "🛡️ RISK-OFF DEFENSIVE"
        elif metrics['avg_rvol'] > 1.5 and breadth > 0.5:
            regime = "⚡VOLATILE OPPORTUNITY"
        else:
            regime = "😴 RANGE-BOUND"
        
        metrics['regime'] = regime
        
        return regime, metrics
    
    @staticmethod
    def calculate_advance_decline_ratio(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate advance/decline ratio and related metrics"""
        
        if 'ret_1d' not in df.columns or df.empty:
            return {'advancing': 0, 'declining': 0, 'unchanged': 0, 'ad_ratio': 1.0, 'ad_line': 0, 'breadth_pct': 0}
        
        advancing = len(df[df['ret_1d'] > 0])
        declining = len(df[df['ret_1d'] < 0])
        unchanged = len(df[df['ret_1d'] == 0])
        
        ad_metrics = {
            'advancing': advancing,
            'declining': declining,
            'unchanged': unchanged,
            'ad_ratio': safe_divide(advancing, declining, default=1.0) if declining > 0 else (float('inf') if advancing > 0 else 1.0),
            'ad_line': advancing - declining,
            'breadth_pct': safe_percentage(advancing, len(df), default=0.0)
        }
        
        return ad_metrics
    
    @staticmethod
    def detect_sector_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced sector rotation detection using Leadership Density Index"""
        
        if df.empty or 'sector' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Primary approach: Use LDI for accurate analysis
            ldi_df = LeadershipDensityEngine.calculate_sector_ldi(df)
            
            if ldi_df.empty:
                return pd.DataFrame()
            
            # Calculate enhanced flow score
            ldi_df['flow_score'] = (
                ldi_df['ldi_score'] * 0.4 +
                ldi_df['avg_score'] * 0.3 +
                ldi_df['avg_momentum'] * 0.15 +
                ldi_df['avg_volume'] * 0.15
            )
            
            ldi_df['rank'] = ldi_df['flow_score'].rank(ascending=False)
            
            # Ensure compatibility with existing UI
            display_df = ldi_df.rename(columns={
                'leader_count': 'analyzed_stocks'
            }).copy()
            
            display_df['sampling_pct'] = 100.0
            
            # Professional quality indicators
            display_df['ldi_quality'] = display_df['ldi_score'].apply(
                lambda x: 'Elite' if x >= 20 else 
                         'Strong' if x >= 10 else 
                         'Moderate' if x >= 5 else 
                         'Weak'
            )
            
            return display_df.sort_values('flow_score', ascending=False)
            
        except Exception as e:
            logger.error(f"Error in LDI sector rotation: {str(e)}")
            # Fallback to traditional method
            return MarketIntelligence._calculate_traditional_sectors(df)
    
    @staticmethod
    def detect_industry_rotation(df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced industry rotation detection using Leadership Density Index"""
        
        if df.empty or 'industry' not in df.columns:
            return pd.DataFrame()
        
        try:
            # Primary approach: Use LDI for accurate analysis
            ldi_df = LeadershipDensityEngine.calculate_industry_ldi(df)
            
            if ldi_df.empty:
                return pd.DataFrame()
            
            # Calculate enhanced flow score
            ldi_df['flow_score'] = (
                ldi_df['ldi_score'] * 0.4 +
                ldi_df['avg_score'] * 0.3 +
                ldi_df['avg_momentum'] * 0.15 +
                ldi_df['avg_volume'] * 0.15
            )
            
            ldi_df['rank'] = ldi_df['flow_score'].rank(ascending=False)
            
            # Ensure compatibility with existing UI
            display_df = ldi_df.rename(columns={
                'leader_count': 'analyzed_stocks'
            }).copy()
            
            display_df['sampling_pct'] = 100.0
            
            return display_df.sort_values('flow_score', ascending=False)
            
        except Exception as e:
            logger.error(f"Error in LDI industry rotation: {str(e)}")
            # Fallback to traditional method
            return MarketIntelligence._calculate_traditional_industries(df)
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _calculate_traditional_sectors(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback traditional sector analysis with optimized sampling"""
        
        if 'sector' not in df.columns or df.empty:
            return pd.DataFrame()
        
        sector_data = []
        
        for sector in df['sector'].unique():
            if sector == 'Unknown':
                continue
                
            sector_df = df[df['sector'] == sector].copy()
            total_count = len(sector_df)
            
            if total_count == 0:
                continue
            
            # Simplified sampling logic
            if total_count <= 10:
                sample_count = total_count
            elif total_count <= 50:
                sample_count = max(8, int(total_count * 0.6))
            else:
                sample_count = min(30, int(total_count * 0.3))
            
            # Take top performers
            sampled_df = sector_df.nlargest(sample_count, 'master_score')
            sector_data.append(sampled_df)
        
        if not sector_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(sector_data, ignore_index=True)
        
        # Calculate aggregated metrics
        sector_metrics = combined_df.groupby('sector').agg({
            'master_score': ['mean', 'median', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in combined_df.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        sector_metrics.columns = [
            'avg_score', 'median_score', 'analyzed_stocks',
            'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow'
        ]
        
        # Add total stock counts and sampling percentage
        total_counts = df.groupby('sector').size().rename('total_stocks')
        sector_metrics = sector_metrics.join(total_counts, how='left')
        sector_metrics['sampling_pct'] = (sector_metrics['analyzed_stocks'] / sector_metrics['total_stocks'] * 100).round(1)
        
        # Calculate flow score
        sector_metrics['flow_score'] = (
            sector_metrics['avg_score'] * 0.3 +
            sector_metrics['median_score'] * 0.2 +
            sector_metrics['avg_momentum'] * 0.25 +
            sector_metrics['avg_volume'] * 0.25
        )
        
        sector_metrics['rank'] = sector_metrics['flow_score'].rank(ascending=False)
        
        return sector_metrics.sort_values('flow_score', ascending=False)
    
    @staticmethod
    @st.cache_data(ttl=300, show_spinner=False)
    def _calculate_traditional_industries(df: pd.DataFrame) -> pd.DataFrame:
        """Fallback traditional industry analysis with optimized sampling"""
        
        if 'industry' not in df.columns or df.empty:
            return pd.DataFrame()
        
        industry_data = []
        
        for industry in df['industry'].unique():
            if industry == 'Unknown':
                continue
                
            industry_df = df[df['industry'] == industry].copy()
            total_count = len(industry_df)
            
            if total_count == 0:
                continue
            
            # Simplified sampling logic for industries
            if total_count <= 5:
                sample_count = total_count
            elif total_count <= 25:
                sample_count = max(5, int(total_count * 0.7))
            elif total_count <= 100:
                sample_count = max(10, int(total_count * 0.4))
            else:
                sample_count = min(40, int(total_count * 0.2))
            
            # Take top performers
            sampled_df = industry_df.nlargest(sample_count, 'master_score')
            industry_data.append(sampled_df)
        
        if not industry_data:
            return pd.DataFrame()
        
        combined_df = pd.concat(industry_data, ignore_index=True)
        
        # Calculate aggregated metrics
        industry_metrics = combined_df.groupby('industry').agg({
            'master_score': ['mean', 'median', 'count'],
            'momentum_score': 'mean',
            'volume_score': 'mean',
            'rvol': 'mean',
            'ret_30d': 'mean',
            'money_flow_mm': 'sum' if 'money_flow_mm' in combined_df.columns else lambda x: 0
        }).round(2)
        
        # Flatten column names
        industry_metrics.columns = [
            'avg_score', 'median_score', 'analyzed_stocks',
            'avg_momentum', 'avg_volume', 'avg_rvol', 'avg_ret_30d', 'total_money_flow'
        ]
        
        # Add total stock counts and sampling percentage
        total_counts = df.groupby('industry').size().rename('total_stocks')
        industry_metrics = industry_metrics.join(total_counts, how='left')
        industry_metrics['sampling_pct'] = (industry_metrics['analyzed_stocks'] / industry_metrics['total_stocks'] * 100).round(1)
        
        # Calculate flow score
        industry_metrics['flow_score'] = (
            industry_metrics['avg_score'] * 0.3 +
            industry_metrics['median_score'] * 0.2 +
            industry_metrics['avg_momentum'] * 0.25 +
            industry_metrics['avg_volume'] * 0.25
        )
        
        industry_metrics['rank'] = industry_metrics['flow_score'].rank(ascending=False)
        
        return industry_metrics.sort_values('flow_score', ascending=False)


# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create all visualizations with proper error handling"""
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame) -> go.Figure:
        """Create score distribution chart"""
        fig = go.Figure()
        
        if df.empty:
            fig.add_annotation(
                text="No data available for visualization",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        scores = [
            ('position_score', 'Position', '#3498db'),
            ('volume_score', 'Volume', '#e74c3c'),
            ('momentum_score', 'Momentum', '#2ecc71'),
            ('acceleration_score', 'Acceleration', '#f39c12'),
            ('breakout_score', 'Breakout', '#9b59b6'),
            ('rvol_score', 'RVOL', '#e67e22')
        ]
        
        for score_col, label, color in scores:
            if score_col in df.columns:
                score_data = df[score_col].dropna()
                if len(score_data) > 0:
                    fig.add_trace(go.Box(
                        y=score_data,
                        name=label,
                        marker_color=color,
                        boxpoints='outliers',
                        hovertemplate=f'{label}<br>Score: %{{y:.1f}}<extra></extra>'
                    ))
        
        fig.update_layout(
            title="Score Component Distribution",
            yaxis_title="Score (0-100)",
            template='plotly_white',
            height=400,
            showlegend=False
        )
        
        return fig

    @staticmethod
    def create_acceleration_profiles(df: pd.DataFrame, n: int = 10) -> go.Figure:
        """Create acceleration profiles showing momentum over time"""
        try:
            accel_df = df.nlargest(min(n, len(df)), 'acceleration_score')
            
            if len(accel_df) == 0:
                return go.Figure()
            
            fig = go.Figure()
            
            for _, stock in accel_df.iterrows():
                x_points = []
                y_points = []
                
                x_points.append('Start')
                y_points.append(0)
                
                if 'ret_30d' in stock.index and pd.notna(stock['ret_30d']):
                    x_points.append('30D')
                    y_points.append(stock['ret_30d'])
                
                if 'ret_7d' in stock.index and pd.notna(stock['ret_7d']):
                    x_points.append('7D')
                    y_points.append(stock['ret_7d'])
                
                if 'ret_1d' in stock.index and pd.notna(stock['ret_1d']):
                    x_points.append('Today')
                    y_points.append(stock['ret_1d'])
                
                if len(x_points) > 1:
                    if stock['acceleration_score'] >= 85:
                        line_style = dict(width=3, dash='solid')
                        marker_style = dict(size=10, symbol='star')
                    elif stock['acceleration_score'] >= 70:
                        line_style = dict(width=2, dash='solid')
                        marker_style = dict(size=8)
                    else:
                        line_style = dict(width=2, dash='dot')
                        marker_style = dict(size=6)
                    
                    fig.add_trace(go.Scatter(
                        x=x_points,
                        y=y_points,
                        mode='lines+markers',
                        name=f"{stock['ticker']} ({stock['acceleration_score']:.0f})",
                        line=line_style,
                        marker=marker_style,
                        hovertemplate=(
                            f"<b>{stock['ticker']}</b><br>" +
                            "%{x}: %{y:.1f}%<br>" +
                            f"Accel Score: {stock['acceleration_score']:.0f}<extra></extra>"
                        )
                    ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            
            fig.update_layout(
                title=f"Acceleration Profiles - Top {len(accel_df)} Momentum Builders",
                xaxis_title="Time Frame",
                yaxis_title="Return %",
                height=400,
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating acceleration profiles: {str(e)}")
            return go.Figure()

# ============================================
# FILTER ENGINE - OPTIMIZED VERSION
# ============================================

class FilterEngine:
    """
    Centralized filter management with single state object.
    This eliminates 15+ separate session state keys.
    FIXED: Now properly cleans up ALL dynamic widget keys.
    """
    
    @staticmethod
    def initialize_filters():
        """Initialize single filter state object"""
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {
                'categories': [],
                'sectors': [],
                'industries': [],
                'min_score': 0,
                'patterns': [],
                'trend_filter': "All Trends",
                'trend_range': (0, 100),
                'trend_custom_range': (0, 100),
                'eps_tiers': [],
                'pe_tiers': [],
                'price_tiers': [],
                'eps_change_tiers': [],
                'position_tiers': [],
                'position_range': (0, 100),
                'performance_tiers': [],
                'performance_custom_range': (-100, 500),
                'ret_1d_range': (2.0, 25.0),
                'ret_3d_range': (3.0, 50.0),
                'ret_7d_range': (5.0, 75.0),
                'ret_30d_range': (10.0, 150.0),
                'ret_3m_range': (15.0, 200.0),
                'ret_6m_range': (20.0, 500.0),
                'ret_1y_range': (25.0, 1000.0),
                'ret_3y_range': (50.0, 2000.0),
                'ret_5y_range': (75.0, 5000.0),
                'min_pe': None,
                'max_pe': None,
                'require_fundamental_data': False,
                'market_states': [],
                'market_strength_range': (0, 100),
                'long_term_strength_range': (0, 100),
                'position_score_range': (0, 100),
                'volume_score_range': (0, 100),
                'momentum_score_range': (0, 100),
                'acceleration_score_range': (0, 100),
                'breakout_score_range': (0, 100),
                'rvol_score_range': (0, 100),
                'position_score_selection': "All Scores",
                'volume_score_selection': "All Scores",
                'momentum_score_selection': "All Scores",
                'acceleration_score_selection': "All Scores",
                'breakout_score_selection': "All Scores",
                'rvol_score_selection': "All Scores",
                'quick_filter': None,
                'quick_filter_applied': False,
                'volume_tiers': [],
                'rvol_range': (0.1, 20.0),
                'vmi_tiers': [],
                'custom_vmi_range': (0.5, 3.0),
                'momentum_harmony_tiers': [],
                # Performance filter selections
                'ret_1d_selection': "All Returns",
                'ret_3d_selection': "All Returns", 
                'ret_7d_selection': "All Returns",
                'ret_30d_selection': "All Returns",
                'ret_3m_selection': "All Returns",
                'ret_6m_selection': "All Returns",
                'ret_1y_selection': "All Returns",
                'ret_3y_selection': "All Returns",
                'ret_5y_selection': "All Returns"
            }
        
        # CRITICAL FIX: Clean up any corrupted performance_tiers values
        # This prevents the "default value not in options" error
        valid_performance_options = [
            "🚀 Strong Gainers (>3% 1D)",
            "⚡ Power Moves (>7% 1D)",
            "💥 Explosive (>15% 1D)",
            "🌟 3-Day Surge (>6% 3D)",
            "📈 Weekly Winners (>12% 7D)",
            "🏆 Monthly Champions (>25% 30D)",
            "🎯 Quarterly Stars (>40% 3M)",
            "💎 Half-Year Heroes (>60% 6M)",
            "🌙 Annual Winners (>80% 1Y)",
            "👑 Multi-Year Champions (>150% 3Y)",
            "🏛️ Long-Term Legends (>250% 5Y)",
            "🎯 Custom Range"
        ]
        
        # Clean up performance_tiers to only include valid options
        if 'performance_tiers' in st.session_state.filter_state:
            current_tiers = st.session_state.filter_state['performance_tiers']
            if isinstance(current_tiers, list):
                # Remove any invalid options (like those with extra spaces)
                cleaned_tiers = [tier for tier in current_tiers if tier in valid_performance_options]
                st.session_state.filter_state['performance_tiers'] = cleaned_tiers
    
    @staticmethod
    def get_filter(key: str, default: Any = None) -> Any:
        """Get filter value from centralized state"""
        FilterEngine.initialize_filters()
        return st.session_state.filter_state.get(key, default)
    
    @staticmethod
    def set_filter(key: str, value: Any) -> None:
        """Set filter value in centralized state"""
        FilterEngine.initialize_filters()
        st.session_state.filter_state[key] = value
    
    @staticmethod
    def get_active_count() -> int:
        """Count active filters"""
        FilterEngine.initialize_filters()
        count = 0
        filters = st.session_state.filter_state
        
        # Check each filter type
        if filters.get('categories'): count += 1
        if filters.get('sectors'): count += 1
        if filters.get('industries'): count += 1
        if filters.get('min_score', 0) > 0: count += 1
        if filters.get('patterns'): count += 1
        if filters.get('trend_filter') != "All Trends": count += 1
        if filters.get('eps_tiers'): count += 1
        if filters.get('pe_tiers'): count += 1
        if filters.get('price_tiers'): count += 1
        if filters.get('eps_change_tiers'): count += 1
        if filters.get('min_pe') is not None: count += 1
        if filters.get('max_pe') is not None: count += 1
        if filters.get('require_fundamental_data'): count += 1
        if filters.get('market_states'): count += 1
        if filters.get('market_strength_range') != (0, 100): count += 1
        if filters.get('long_term_strength_range') != (0, 100): count += 1
        if filters.get('performance_tiers'): count += 1
        if filters.get('position_tiers'): count += 1
        if filters.get('volume_tiers'): count += 1
        if filters.get('vmi_tiers'): count += 1
        if filters.get('momentum_harmony_tiers'): count += 1
        if filters.get('custom_vmi_range') and filters.get('custom_vmi_range') != (0.5, 3.0): count += 1
        if filters.get('position_score_range') != (0, 100): count += 1
        if filters.get('volume_score_range') != (0, 100): count += 1
        if filters.get('momentum_score_range') != (0, 100): count += 1
        if filters.get('acceleration_score_range') != (0, 100): count += 1
        if filters.get('breakout_score_range') != (0, 100): count += 1
        if filters.get('rvol_score_range') != (0, 100): count += 1
        
        return count
    
    @staticmethod
    def clear_all_filters():
        """
        Reset all filters to defaults and clear widget states.
        FIXED: Now properly deletes ALL dynamic widget keys to prevent memory leaks.
        """
        # Reset centralized filter state
        st.session_state.filter_state = {
            'categories': [],
            'sectors': [],
            'industries': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': "All Trends",
            'trend_range': (0, 100),
            'trend_custom_range': (0, 100),
            'eps_tiers': [],
            'pe_tiers': [],
            'price_tiers': [],
            'eps_change_tiers': [],
            'min_pe': None,
            'max_pe': None,
            'require_fundamental_data': False,
            'market_states': [],
            'market_strength_range': (0, 100),
            'long_term_strength_range': (0, 100),
            'quick_filter': None,
            'quick_filter_applied': False,
            'performance_tiers': [],
            'performance_custom_range': (-100, 500),
            'ret_1d_range': (2.0, 25.0),
            'ret_3d_range': (3.0, 50.0),
            'ret_7d_range': (5.0, 75.0),
            'ret_30d_range': (10.0, 150.0),
            'ret_3m_range': (15.0, 200.0),
            'ret_6m_range': (20.0, 500.0),
            'ret_1y_range': (25.0, 1000.0),
            'ret_3y_range': (50.0, 2000.0),
            'ret_5y_range': (75.0, 5000.0),
            'position_tiers': [],
            'position_range': (0, 100),
            'volume_tiers': [],
            'rvol_range': (0.1, 20.0),
            'vmi_tiers': [],
            'custom_vmi_range': (0.5, 3.0),
            'momentum_harmony_tiers': [],
            'position_score_range': (0, 100),
            'volume_score_range': (0, 100),
            'momentum_score_range': (0, 100),
            'acceleration_score_range': (0, 100),
            'breakout_score_range': (0, 100),
            'rvol_score_range': (0, 100),
            'position_score_selection': "All Scores",
            'volume_score_selection': "All Scores",
            'momentum_score_selection': "All Scores",
            'acceleration_score_selection': "All Scores",
            'breakout_score_selection': "All Scores",
            'rvol_score_selection': "All Scores",
            # Performance filter selections
            'ret_1d_selection': "All Returns",
            'ret_3d_selection': "All Returns", 
            'ret_7d_selection': "All Returns",
            'ret_30d_selection': "All Returns",
            'ret_3m_selection': "All Returns",
            'ret_6m_selection': "All Returns",
            'ret_1y_selection': "All Returns",
            'ret_3y_selection': "All Returns",
            'ret_5y_selection': "All Returns"
        }
        
        # CRITICAL FIX: Delete all widget keys to force UI reset
        # First, delete known widget keys
        widget_keys_to_delete = [
            # Multiselect widgets
            'category_multiselect', 'sector_multiselect', 'industry_multiselect',
            'patterns_multiselect', 'market_states_multiselect',
            'eps_tier_multiselect', 'pe_tier_multiselect', 'price_tier_multiselect',
            'eps_change_tiers_widget', 'performance_tier_multiselect', 'position_tier_multiselect',
            'volume_tier_multiselect',
            'performance_tier_multiselect_intelligence', 'volume_tier_multiselect_intelligence',
            'position_tier_multiselect_intelligence',
            
            # Slider widgets
            'min_score_slider', 'market_strength_slider', 'performance_custom_range_slider',
            'trend_custom_range_slider',
            'ret_1d_range_slider', 'ret_3d_range_slider', 'ret_7d_range_slider', 'ret_30d_range_slider',
            'ret_3m_range_slider', 'ret_6m_range_slider', 'ret_1y_range_slider', 'ret_3y_range_slider', 'ret_5y_range_slider',
            'position_range_slider', 'rvol_range_slider',
            'position_score_slider', 'volume_score_slider', 'momentum_score_slider',
            'acceleration_score_slider', 'breakout_score_slider', 'rvol_score_slider',
            
            # Score dropdown widgets
            'position_score_dropdown', 'volume_score_dropdown', 'momentum_score_dropdown',
            'acceleration_score_dropdown', 'breakout_score_dropdown', 'rvol_score_dropdown',
            
            # Performance dropdown widgets
            'ret_1d_dropdown', 'ret_3d_dropdown', 'ret_7d_dropdown', 'ret_30d_dropdown',
            'ret_3m_dropdown', 'ret_6m_dropdown', 'ret_1y_dropdown', 'ret_3y_dropdown', 'ret_5y_dropdown',
            
            # Selectbox widgets
            'trend_selectbox', 'wave_timeframe_select',
            
            # Text input widgets
            'eps_change_input', 'min_pe_input', 'max_pe_input',
            
            # Checkbox widgets
            'require_fundamental_checkbox',
            
            # Additional filter-related keys
            'display_count_select', 'sort_by_select', 'export_template_radio',
            'wave_sensitivity', 'show_sensitivity_details', 'show_market_regime',
            'score_component_expander'
        ]
        
        # Delete each known widget key if it exists
        deleted_count = 0
        for key in widget_keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
                deleted_count += 1
                
        # ==== MEMORY LEAK FIX - START ====
        # Now clean up ANY dynamically created widget keys
        # This is crucial for preventing memory leaks
        
        # Define all possible widget suffixes used by Streamlit
        widget_suffixes = [
            '_multiselect', '_slider', '_selectbox', '_checkbox',
            '_input', '_radio', '_button', '_expander', '_toggle',
            '_number_input', '_text_area', '_date_input', '_time_input',
            '_color_picker', '_file_uploader', '_camera_input', '_select_slider'
        ]
        
        # Also check for common prefixes used in dynamic widgets
        widget_prefixes = [
            'FormSubmitter', 'temp_', 'dynamic_', 'filter_', 'widget_'
        ]
        
        # Collect all keys to delete (can't modify dict during iteration)
        dynamic_keys_to_delete = []
        
        # Check all session state keys
        for key in list(st.session_state.keys()):
            # Skip if already deleted
            if key in widget_keys_to_delete:
                continue
            
            # Check if key has widget suffix
            for suffix in widget_suffixes:
                if key.endswith(suffix):
                    dynamic_keys_to_delete.append(key)
                    break
            
            # Check if key has widget prefix
            for prefix in widget_prefixes:
                if key.startswith(prefix) and key not in dynamic_keys_to_delete:
                    dynamic_keys_to_delete.append(key)
                    break
        
        # Delete all collected dynamic keys
        for key in dynamic_keys_to_delete:
            try:
                del st.session_state[key]
                deleted_count += 1
                logger.debug(f"Deleted dynamic widget key: {key}")
            except KeyError:
                # Key might have been deleted already
                pass
        
        # ==== MEMORY LEAK FIX - END ====
        
        # Also clear legacy filter keys for backward compatibility
        legacy_keys = [
            'category_filter', 'sector_filter', 'industry_filter',
            'min_score', 'patterns', 'trend_filter',
            'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter',
            'min_eps_change', 'min_pe', 'max_pe',
            'require_fundamental_data', 'market_states_filter',
            'market_strength_range_slider'
        ]
        
        for key in legacy_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple):
                    if key == 'market_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    if key == 'min_score':
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = None
                else:
                    st.session_state[key] = None
        
        # Reset active filter count
        st.session_state.active_filter_count = 0
        
        # Clear quick filter
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        # Clear any cached filter results
        if 'user_preferences' in st.session_state:
            st.session_state.user_preferences['last_filters'] = {}
        
        # Clean up any cached data related to filters
        cache_keys_to_clear = []
        for key in list(st.session_state.keys()):
            if key.startswith('filter_cache_') or key.startswith('filtered_'):
                cache_keys_to_clear.append(key)
        
        for key in cache_keys_to_clear:
            del st.session_state[key]
            deleted_count += 1
        
        logger.info(f"All filters and widget states cleared successfully. Deleted {deleted_count} keys total.")
    
    @staticmethod
    def sync_widget_to_filter(widget_key: str, filter_key: str):
        """Sync widget state to filter state - used in callbacks"""
        if widget_key in st.session_state:
            st.session_state.filter_state[filter_key] = st.session_state[widget_key]
    
    @staticmethod
    def build_filter_dict() -> Dict[str, Any]:
        """Build filter dictionary for apply_filters method"""
        FilterEngine.initialize_filters()
        filters = {}
        state = st.session_state.filter_state
        
        # Map internal state to filter dict format
        if state.get('categories'):
            filters['categories'] = state['categories']
        if state.get('sectors'):
            filters['sectors'] = state['sectors']
        if state.get('industries'):
            filters['industries'] = state['industries']
        if state.get('min_score', 0) > 0:
            filters['min_score'] = state['min_score']
        if state.get('patterns'):
            filters['patterns'] = state['patterns']
        if state.get('trend_filter') != "All Trends":
            filters['trend_filter'] = state['trend_filter']
            filters['trend_range'] = state.get('trend_range', (0, 100))
        if state.get('eps_tiers'):
            filters['eps_tiers'] = state['eps_tiers']
        if state.get('pe_tiers'):
            filters['pe_tiers'] = state['pe_tiers']
        if state.get('price_tiers'):
            filters['price_tiers'] = state['price_tiers']
        if state.get('eps_change_tiers'):
            filters['eps_change_tiers'] = state['eps_change_tiers']
        if state.get('min_pe') is not None:
            filters['min_pe'] = state['min_pe']
        if state.get('max_pe') is not None:
            filters['max_pe'] = state['max_pe']
        if state.get('require_fundamental_data'):
            filters['require_fundamental_data'] = True
        if state.get('market_states'):
            filters['market_states'] = state['market_states']
        if state.get('market_strength_range') != (0, 100):
            filters['market_strength_range'] = state['market_strength_range']
        if state.get('long_term_strength_range') != (0, 100):
            filters['long_term_strength_range'] = state['long_term_strength_range']
            
        return filters
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.1)
    def apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Apply all filters to dataframe efficiently using vectorized operations.
        If no filters provided, get from centralized state.
        """
        if df.empty:
            return df
        
        # Use provided filters or get from state
        if filters is None:
            filters = FilterEngine.build_filter_dict()
        
        if not filters:
            return df
        
        # Create boolean masks for each filter
        masks = []
        
        # Helper function for isin filters
        def create_mask_from_isin(column: str, values: List[Any]) -> Optional[pd.Series]:
            if values and column in df.columns:
                return df[column].isin(values)
            return None
        
        # 1. Category filters
        if 'categories' in filters:
            mask = create_mask_from_isin('category', filters['categories'])
            if mask is not None:
                masks.append(mask)

        if 'sectors' in filters:
            mask = create_mask_from_isin('sector', filters['sectors'])
            if mask is not None:
                masks.append(mask)

        if 'industries' in filters:
            mask = create_mask_from_isin('industry', filters['industries'])
            if mask is not None:
                masks.append(mask)
        
        # 2. Score filter
        if filters.get('min_score', 0) > 0 and 'master_score' in df.columns:
            masks.append(df['master_score'] >= filters['min_score'])
        
        # 2.1. Intelligence Score Filters
        # Position Score Filter
        if 'position_score_range' in filters and 'position_score' in df.columns:
            position_range = filters['position_score_range']
            if position_range != (0, 100):
                min_pos, max_pos = position_range
                masks.append((df['position_score'] >= min_pos) & (df['position_score'] <= max_pos))
        
        # Volume Score Filter
        if 'volume_score_range' in filters and 'volume_score' in df.columns:
            volume_range = filters['volume_score_range']
            if volume_range != (0, 100):
                min_vol, max_vol = volume_range
                masks.append((df['volume_score'] >= min_vol) & (df['volume_score'] <= max_vol))
        
        # Momentum Score Filter
        if 'momentum_score_range' in filters and 'momentum_score' in df.columns:
            momentum_range = filters['momentum_score_range']
            if momentum_range != (0, 100):
                min_mom, max_mom = momentum_range
                masks.append((df['momentum_score'] >= min_mom) & (df['momentum_score'] <= max_mom))
        
        # Acceleration Score Filter
        if 'acceleration_score_range' in filters and 'acceleration_score' in df.columns:
            acceleration_range = filters['acceleration_score_range']
            if acceleration_range != (0, 100):
                min_acc, max_acc = acceleration_range
                masks.append((df['acceleration_score'] >= min_acc) & (df['acceleration_score'] <= max_acc))
        
        # Breakout Score Filter
        if 'breakout_score_range' in filters and 'breakout_score' in df.columns:
            breakout_range = filters['breakout_score_range']
            if breakout_range != (0, 100):
                min_brk, max_brk = breakout_range
                masks.append((df['breakout_score'] >= min_brk) & (df['breakout_score'] <= max_brk))
        
        # RVOL Score Filter
        if 'rvol_score_range' in filters and 'rvol_score' in df.columns:
            rvol_range = filters['rvol_score_range']
            if rvol_range != (0, 100):
                min_rvol, max_rvol = rvol_range
                masks.append((df['rvol_score'] >= min_rvol) & (df['rvol_score'] <= max_rvol))
        
        # 3. Pattern filter
        if filters.get('patterns') and 'patterns' in df.columns:
            pattern_mask = pd.Series(False, index=df.index)
            for pattern in filters['patterns']:
                pattern_mask |= df['patterns'].str.contains(pattern, na=False, regex=False)
            masks.append(pattern_mask)
        
        # 4. Trend filter
        trend_range = filters.get('trend_range')
        if trend_range and trend_range != (0, 100) and 'trend_quality' in df.columns:
            min_trend, max_trend = trend_range
            masks.append((df['trend_quality'] >= min_trend) & (df['trend_quality'] <= max_trend))
        
        # 5. EPS change filter - Now using tiers instead of min value
        if 'eps_change_tiers' in filters:
            masks.append(create_mask_from_isin('eps_change_tier', filters['eps_change_tiers']))
        
        # 5.5. Position Intelligence filters
        if 'position_tiers' in filters:
            selected_tiers = filters['position_tiers']
            if selected_tiers and "🎯 Custom Range" not in selected_tiers:
                masks.append(create_mask_from_isin('position_tier', selected_tiers))
        
        # Custom position range filter (only if "🎯 Custom Range" is selected)
        if 'position_tiers' in filters and "🎯 Custom Range" in filters['position_tiers']:
            if 'position_range' in filters:
                position_range = filters['position_range']
                if 'position_pct' in df.columns:
                    masks.append(df['position_pct'].between(position_range[0], position_range[1], inclusive='both'))
        
        # 5.6. Performance Intelligence filters
        if 'performance_tiers' in filters:
            selected_tiers = filters['performance_tiers']
            # Only apply preset tier filtering if "Custom Range" is not selected
            custom_range_in_tiers = any("Custom Range" in tier for tier in selected_tiers) if selected_tiers else False
            if selected_tiers and not custom_range_in_tiers:
                masks.append(create_mask_from_isin('performance_tier', selected_tiers))
        
        # Custom performance range filter (only apply if actual range filters are modified from defaults)
        custom_range_selected = 'performance_tiers' in filters and any("Custom Range" in tier for tier in filters['performance_tiers'])
        
        if custom_range_selected:
            # Define default ranges for each timeframe (MUST MATCH UI EXACTLY) - REALISTIC INDIAN MARKET THRESHOLDS
            default_ranges = {
                'ret_1d_range': (2.0, 25.0),
                'ret_3d_range': (3.0, 50.0),
                'ret_7d_range': (5.0, 75.0),
                'ret_30d_range': (10.0, 150.0),
                'ret_3m_range': (15.0, 200.0),
                'ret_6m_range': (20.0, 500.0),
                'ret_1y_range': (25.0, 1000.0),
                'ret_3y_range': (50.0, 2000.0),
                'ret_5y_range': (75.0, 5000.0)
            }
            
            # Only apply filters for ranges that have been modified from defaults
            active_custom_ranges = []
            
            # Short-term performance ranges
            if 'ret_1d_range' in filters and 'ret_1d' in df.columns:
                range_val = filters['ret_1d_range']
                if range_val != default_ranges['ret_1d_range']:
                    masks.append(df['ret_1d'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_1d_range')
            
            if 'ret_3d_range' in filters and 'ret_3d' in df.columns:
                range_val = filters['ret_3d_range']
                if range_val != default_ranges['ret_3d_range']:
                    masks.append(df['ret_3d'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_3d_range')
            
            if 'ret_7d_range' in filters and 'ret_7d' in df.columns:
                range_val = filters['ret_7d_range']
                if range_val != default_ranges['ret_7d_range']:
                    masks.append(df['ret_7d'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_7d_range')
            
            if 'ret_30d_range' in filters and 'ret_30d' in df.columns:
                range_val = filters['ret_30d_range']
                if range_val != default_ranges['ret_30d_range']:
                    masks.append(df['ret_30d'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_30d_range')
            
            # Medium-term performance ranges
            if 'ret_3m_range' in filters and 'ret_3m' in df.columns:
                range_val = filters['ret_3m_range']
                if range_val != default_ranges['ret_3m_range']:
                    masks.append(df['ret_3m'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_3m_range')
            
            if 'ret_6m_range' in filters and 'ret_6m' in df.columns:
                range_val = filters['ret_6m_range']
                if range_val != default_ranges['ret_6m_range']:
                    masks.append(df['ret_6m'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_6m_range')
            
            # Long-term performance ranges
            if 'ret_1y_range' in filters and 'ret_1y' in df.columns:
                range_val = filters['ret_1y_range']
                if range_val != default_ranges['ret_1y_range']:
                    masks.append(df['ret_1y'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_1y_range')
            
            if 'ret_3y_range' in filters and 'ret_3y' in df.columns:
                range_val = filters['ret_3y_range']
                if range_val != default_ranges['ret_3y_range']:
                    masks.append(df['ret_3y'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_3y_range')
            
            if 'ret_5y_range' in filters and 'ret_5y' in df.columns:
                range_val = filters['ret_5y_range']
                if range_val != default_ranges['ret_5y_range']:
                    masks.append(df['ret_5y'].between(range_val[0], range_val[1], inclusive='both'))
                    active_custom_ranges.append('ret_5y_range')
            
            # CRITICAL FIX: If Custom Range is selected but no ranges are modified,
            # don't apply any performance filtering (show all stocks)
            # This prevents the "0 stocks" issue when just selecting Custom Range
            if not active_custom_ranges:
                # No custom ranges are active, so don't filter by performance
                pass  # This effectively shows all stocks for performance filtering
            else:
                logger.info(f"Active custom ranges: {active_custom_ranges}")
            
            # Legacy support for old performance_custom_range
            if 'performance_custom_range' in filters:
                perf_range = filters['performance_custom_range']
                # Apply Custom Range to any available return percentage column
                perf_masks = []
                for col in ['ret_1d', 'ret_7d', 'ret_30d']:
                    if col in df.columns:
                        perf_masks.append(df[col].between(perf_range[0], perf_range[1], inclusive='both'))
                if perf_masks:
                    # Use OR logic - stock qualifies if it meets the range in ANY timeframe
                    combined_mask = perf_masks[0]
                    for mask in perf_masks[1:]:
                        combined_mask = combined_mask | mask
                    masks.append(combined_mask)
        
        # 5.6.1. NEW Individual Performance Period Filters (V9 Enhancement)
        # Handle individual return period filters from the new Performance Filter UI
        individual_performance_ranges = {
            'ret_1d_range': 'ret_1d',
            'ret_3d_range': 'ret_3d', 
            'ret_7d_range': 'ret_7d',
            'ret_30d_range': 'ret_30d',
            'ret_3m_range': 'ret_3m',
            'ret_6m_range': 'ret_6m',
            'ret_1y_range': 'ret_1y',
            'ret_3y_range': 'ret_3y',
            'ret_5y_range': 'ret_5y'
        }
        
        for range_key, column_name in individual_performance_ranges.items():
            if range_key in filters and column_name in df.columns:
                range_val = filters[range_key]
                if isinstance(range_val, tuple) and len(range_val) == 2:
                    min_val, max_val = range_val
                    masks.append(df[column_name].between(min_val, max_val, inclusive='both'))
                    logger.info(f"Applied {range_key} filter: {range_val} on column {column_name}")
        
        # 5.7. Volume Intelligence filters
        if 'volume_tiers' in filters:
            selected_tiers = filters['volume_tiers']
            if selected_tiers and "🎯 Custom RVOL Range" not in selected_tiers:
                masks.append(create_mask_from_isin('volume_tier', selected_tiers))
        
        # Custom RVOL range filter (only if "🎯 Custom RVOL Range" is selected)
        if 'volume_tiers' in filters and "🎯 Custom RVOL Range" in filters['volume_tiers']:
            if 'rvol_range' in filters and 'rvol' in df.columns:
                rvol_range_val = filters['rvol_range']
                masks.append(df['rvol'].between(rvol_range_val[0], rvol_range_val[1], inclusive='both'))
        
        # 5.8. VMI (Volume Momentum Index) filters
        if 'vmi_tiers' in filters:
            selected_tiers = filters['vmi_tiers']
            if selected_tiers and "🎯 Custom VMI Range" not in selected_tiers:
                masks.append(create_mask_from_isin('vmi_tier', selected_tiers))
        
        # Custom VMI range filter (only if "🎯 Custom VMI Range" is selected)
        if 'vmi_tiers' in filters and "🎯 Custom VMI Range" in filters['vmi_tiers']:
            if 'custom_vmi_range' in filters and 'vmi' in df.columns:
                vmi_range_val = filters['custom_vmi_range']
                masks.append(df['vmi'].between(vmi_range_val[0], vmi_range_val[1], inclusive='both'))
        
        # 5.9. Momentum Harmony filters
        if 'momentum_harmony_tiers' in filters:
            selected_tiers = filters['momentum_harmony_tiers']
            if selected_tiers:
                masks.append(create_mask_from_isin('momentum_harmony_tier', selected_tiers))
        
        # 6. PE filters
        if filters.get('min_pe') is not None and 'pe' in df.columns:
            masks.append(df['pe'] >= filters['min_pe'])
        
        if filters.get('max_pe') is not None and 'pe' in df.columns:
            masks.append(df['pe'] <= filters['max_pe'])
        
        # 7. Tier filters
        if 'eps_tiers' in filters:
            masks.append(create_mask_from_isin('eps_tier', filters['eps_tiers']))
        if 'pe_tiers' in filters:
            masks.append(create_mask_from_isin('pe_tier', filters['pe_tiers']))
        if 'price_tiers' in filters:
            masks.append(create_mask_from_isin('price_tier', filters['price_tiers']))
        
        # 8. Data completeness filter
        if filters.get('require_fundamental_data', False):
            if all(col in df.columns for col in ['pe', 'eps_change_pct']):
                masks.append(df['pe'].notna() & (df['pe'] > 0) & df['eps_change_pct'].notna())
        
        # 9. Market State filters
        if 'market_states' in filters:
            selected_states = filters['market_states']
            if selected_states:
                # Handle preset filters
                preset_mapping = {
                    "🎯 MOMENTUM (Default)": ['STRONG_UPTREND', 'UPTREND', 'PULLBACK'],
                    "⚡ AGGRESSIVE": ['STRONG_UPTREND'],
                    "💎 VALUE": ['PULLBACK', 'BOUNCE', 'SIDEWAYS'],
                    "🛡️ DEFENSIVE": ['STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'SIDEWAYS', 'BOUNCE'],
                    "🌍 ALL": ['STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'ROTATION', 'SIDEWAYS', 'DOWNTREND', 'STRONG_DOWNTREND', 'BOUNCE']
                }
                
                # Valid individual market states
                valid_market_states = {'STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'ROTATION', 'SIDEWAYS', 'DOWNTREND', 'STRONG_DOWNTREND', 'BOUNCE'}
                
                # Collect all allowed states from presets and custom selections
                allowed_states = []
                custom_selection_active = "📊 Custom Selection" in selected_states
                
                for state in selected_states:
                    if state in preset_mapping:
                        # Handle preset selection
                        allowed_states.extend(preset_mapping[state])
                    elif state in valid_market_states:
                        # Handle individual state selection (always include individual states if selected)
                        if custom_selection_active:
                            allowed_states.append(state)
                        else:
                            # If custom selection is not active, but individual states are selected,
                            # this might be a legacy case - still include them
                            allowed_states.append(state)
                    # Skip "📊 Custom Selection" itself as it's just an enabler
                
                # Debug logging for troubleshooting
                if custom_selection_active:
                    individual_states = [s for s in selected_states if s in valid_market_states]
                    logger.info(f"Market State Filter Debug:")
                    logger.info(f"  - Selected states: {selected_states}")
                    logger.info(f"  - Individual states: {individual_states}")
                    logger.info(f"  - Final allowed states: {allowed_states}")
                    
                    # Additional debug: check if market_state column exists and has expected values
                    if 'market_state' in df.columns:
                        unique_values = df['market_state'].value_counts().head(10)
                        logger.info(f"  - Market state values in data: {unique_values.to_dict()}")
                
                # Remove duplicates and apply filter - but only if we have allowed states
                if allowed_states:
                    unique_states = list(set(allowed_states))
                    logger.info(f"  - Applying filter for states: {unique_states}")
                    mask = create_mask_from_isin('market_state', unique_states)
                    if mask is not None:
                        masks.append(mask)
                        logger.info(f"  - Filter mask created successfully, {mask.sum()} stocks match")
                    else:
                        logger.warning("  - Failed to create market state filter mask")
                elif custom_selection_active:
                    # User selected "📊 Custom Selection" but no individual states
                    logger.info("  - Custom selection active but no individual states selected, skipping filter")
                else:
                    logger.warning(f"  - No allowed states found for selection: {selected_states}")
        
        # Market Strength Filter - Professional Implementation
        if 'market_strength_range' in filters and 'overall_market_strength' in df.columns:
            strength_range = filters['market_strength_range']
            if strength_range != (0, 100):
                min_strength, max_strength = strength_range
                strength_mask = (df['overall_market_strength'] >= min_strength) & (df['overall_market_strength'] <= max_strength)
                masks.append(strength_mask)
                logger.info(f"Applied Market Strength filter: {strength_range}, {strength_mask.sum()} stocks match")
        
        # Long Term Strength Filter - Professional Implementation
        if 'long_term_strength_range' in filters and 'long_term_strength' in df.columns:
            lts_range = filters['long_term_strength_range']
            if lts_range != (0, 100):
                min_lts, max_lts = lts_range
                lts_mask = (df['long_term_strength'] >= min_lts) & (df['long_term_strength'] <= max_lts)
                masks.append(lts_mask)
                logger.info(f"Applied Long Term Strength filter: {lts_range}, {lts_mask.sum()} stocks match")
        
        # Combine all masks
        masks = [mask for mask in masks if mask is not None]
        
        if masks:
            combined_mask = np.logical_and.reduce(masks)
            filtered_df = df[combined_mask].copy()
        else:
            filtered_df = df.copy()
        
        logger.info(f"Filters reduced {len(df)} to {len(filtered_df)} stocks")
        
        return filtered_df
    
    @staticmethod
    def get_filter_options(df: pd.DataFrame, column: str, current_filters: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Get available options for a filter based on other active filters.
        This creates BIDIRECTIONAL SMART INTERCONNECTED filters for Category/Sector/Industry.
        """
        if df.empty or column not in df.columns:
            return []
        
        # Use current filters or get from state
        if current_filters is None:
            current_filters = FilterEngine.build_filter_dict()
        
        # BIDIRECTIONAL SMART INTERCONNECTION LOGIC
        temp_filters = current_filters.copy()
        
        # For Category/Sector/Industry interconnection - BIDIRECTIONAL FILTERING
        if column == 'category':
            # For categories, apply sector AND industry filters to show only relevant categories
            temp_filters.pop('categories', None)  # Don't apply category filter when getting categories
            # Keep sectors and industries filters to show only categories that have those sectors/industries
        elif column == 'sector':
            # For sectors, apply category AND industry filters to show only relevant sectors
            temp_filters.pop('sectors', None)  # Don't apply sector filter when getting sectors  
            # Keep categories and industries filters to show only sectors that have those categories/industries
        elif column == 'industry':
            # For industries, apply category AND sector filters to show only relevant industries
            temp_filters.pop('industries', None)  # Don't apply industry filter when getting industries
            # Keep categories and sectors filters to show only industries that have those categories/sectors
        else:
            # For non-interconnected filters, remove only the current filter
            filter_key_map = {
                'eps_tier': 'eps_tiers',
                'pe_tier': 'pe_tiers',
                'price_tier': 'price_tiers',
                'eps_change_tier': 'eps_change_tiers',
                'market_state': 'market_states',
                'position_tier': 'position_tiers',
                'volume_tier': 'volume_tiers',
                'performance_tier': 'performance_tiers',
                'vmi_tier': 'vmi_tiers',
                'momentum_harmony_tier': 'momentum_harmony_tiers'
            }
            
            if column in filter_key_map:
                temp_filters.pop(filter_key_map[column], None)
        
        # Apply remaining filters to get the filtered dataset
        filtered_df = FilterEngine.apply_filters(df, temp_filters)
        
        # Get unique values from the filtered dataset
        values = filtered_df[column].dropna().unique()
        
        # Filter out invalid values
        values = [v for v in values if str(v).strip() not in ['Unknown', 'unknown', '', 'nan', 'NaN', 'None', 'N/A', '-']]
        
        # Sort appropriately
        try:
            values = sorted(values, key=lambda x: float(str(x).replace(',', '')))
        except (ValueError, TypeError):
            values = sorted(values, key=str)
        
        return values
    
    @staticmethod
    def reset_to_defaults():
        """Reset filters to default state but keep widget keys"""
        FilterEngine.initialize_filters()
        
        # Reset only the filter values, not the widgets
        st.session_state.filter_state = {
            'categories': [],
            'sectors': [],
            'industries': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': "All Trends",
            'trend_range': (0, 100),
            'eps_tiers': [],
            'pe_tiers': [],
            'price_tiers': [],
            'eps_change_tiers': [],
            'min_pe': None,
            'max_pe': None,
            'require_fundamental_data': False,
            'market_states': [],
            'market_strength_range': (0, 100),
            'long_term_strength_range': (0, 100),
            'quick_filter': None,
            'quick_filter_applied': False
        }

        # Clean up ALL dynamically created widget keys
        all_widget_patterns = [
            '_multiselect', '_slider', '_selectbox', '_checkbox', 
            '_input', '_radio', '_button', '_expander', '_toggle',
            '_number_input', '_text_area', '_date_input', '_time_input',
            '_color_picker', '_file_uploader', '_camera_input'
        ]
        
        # Collect keys to delete (can't modify dict during iteration)
        dynamic_keys_to_delete = []
        
        for key in list(st.session_state.keys()):
            # Check if this key ends with any widget pattern
            for pattern in all_widget_patterns:
                if pattern in key:
                    dynamic_keys_to_delete.append(key)
                    break
        
        # Delete the dynamic keys
        for key in dynamic_keys_to_delete:
            try:
                del st.session_state[key]
                logger.debug(f"Deleted dynamic widget key: {key}")
            except KeyError:
                # Key might have been deleted already
                pass
        
        # Also clean up any keys that start with 'FormSubmitter'
        form_keys_to_delete = [key for key in st.session_state.keys() if key.startswith('FormSubmitter')]
        for key in form_keys_to_delete:
            try:
                del st.session_state[key]
            except KeyError:
                pass
        # ==== COMPREHENSIVE WIDGET CLEANUP - END ====
        st.session_state.active_filter_count = 0
        logger.info("Filters reset to defaults")
        
# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Optimized search functionality"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=0.05)
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks with optimized performance"""
        
        if not query or df.empty:
            return pd.DataFrame()
        
        try:
            query = query.upper().strip()
            df['ticker'].str.upper().str.contains(query.upper())
            
            # Method 1: Direct ticker match
            ticker_exact = df[df['ticker'].str.upper() == query]
            if not ticker_exact.empty:
                return ticker_exact
            
            # Method 2: Ticker contains
            ticker_contains = df[df['ticker'].str.upper().str.contains(query, na=False, regex=False)]
            
            # Method 3: Company name contains
            if 'company_name' in df.columns:
                company_contains = df[df['company_name'].str.upper().str.contains(query, na=False, regex=False)]
            else:
                company_contains = pd.DataFrame()
            
            # Method 4: Word match
            def word_starts_with(company_name_str):
                if pd.isna(company_name_str):
                    return False
                words = str(company_name_str).upper().split()
                return any(word.startswith(query) for word in words)
            
            if 'company_name' in df.columns:
                company_word_match = df[df['company_name'].apply(word_starts_with)]
            else:
                company_word_match = pd.DataFrame()
            
            # Combine results
            all_matches = pd.concat([
                ticker_contains,
                company_contains,
                company_word_match
            ]).drop_duplicates()
            
            # Sort by relevance
            if not all_matches.empty:
                all_matches['relevance'] = 0
                all_matches.loc[all_matches['ticker'].str.upper() == query, 'relevance'] = 100
                all_matches.loc[all_matches['ticker'].str.upper().str.startswith(query), 'relevance'] += 50
                
                if 'company_name' in all_matches.columns:
                    all_matches.loc[all_matches['company_name'].str.upper().str.startswith(query), 'relevance'] += 30
                
                return all_matches.sort_values(['relevance', 'master_score'], ascending=[False, False]).drop('relevance', axis=1)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return pd.DataFrame()

# ============================================
# EXPORT ENGINE
# ============================================

class ExportEngine:
    """Handle all export operations"""
    
    @staticmethod
    @PerformanceMonitor.timer(target_time=1.0)
    def create_excel_report(df: pd.DataFrame, template: str = 'full') -> BytesIO:
        """Create comprehensive Excel report"""
        
        output = BytesIO()
        
        templates = {
            'day_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'rvol', 
                           'momentum_score', 'acceleration_score', 'ret_1d', 'ret_7d', 
                           'volume_score', 'vmi', 'market_state', 'patterns', 'category', 'industry'],
                'focus': 'Intraday momentum and volume'
            },
            'swing_trader': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 
                           'breakout_score', 'position_score', 'position_tension',
                           'from_high_pct', 'from_low_pct', 'trend_quality', 
                           'momentum_harmony', 'patterns', 'industry'],
                'focus': 'Position and breakout setups'
            },
            'investor': {
                'columns': ['rank', 'ticker', 'company_name', 'master_score', 'pe', 
                           'eps_current', 'eps_change_pct', 'ret_1y', 'ret_3y', 
                           'long_term_strength', 'money_flow_mm', 'category', 'sector', 'industry'],
                'focus': 'Fundamentals and long-term performance'
            },
            'full': {
                'columns': None,
                'focus': 'Complete analysis'
            }
        }
        
        try:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                workbook = writer.book
                
                header_format = workbook.add_format({
                    'bold': True,
                    'bg_color': '#3498db',
                    'font_color': 'white',
                    'border': 1
                })
                
                # 1. Top 100 Stocks
                top_100 = df.nlargest(min(100, len(df)), 'master_score')
                
                if template in templates and templates[template]['columns']:
                    export_cols = [col for col in templates[template]['columns'] if col in top_100.columns]
                else:
                    export_cols = None
                
                if export_cols:
                    top_100_export = top_100[export_cols]
                else:
                    top_100_export = top_100
                
                top_100_export.to_excel(writer, sheet_name='Top 100', index=False)
                
                worksheet = writer.sheets['Top 100']
                for i, col in enumerate(top_100_export.columns):
                    worksheet.write(0, i, col, header_format)
                
                # 2. Market Intelligence
                intel_data = []
                
                regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
                intel_data.append({
                    'Metric': 'Market Regime',
                    'Value': regime,
                    'Details': f"Breadth: {regime_metrics.get('breadth', 0):.1%}"
                })
                
                ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
                intel_data.append({
                    'Metric': 'Advance/Decline',
                    'Value': f"{ad_metrics.get('advancing', 0)}/{ad_metrics.get('declining', 0)}",
                    'Details': f"Ratio: {ad_metrics.get('ad_ratio', 1):.2f}"
                })
                
                intel_df = pd.DataFrame(intel_data)
                intel_df.to_excel(writer, sheet_name='Market Intelligence', index=False)
                
                # 3. Sector Rotation
                sector_rotation = MarketIntelligence.detect_sector_rotation(df)
                if not sector_rotation.empty:
                    sector_rotation.to_excel(writer, sheet_name='Sector Rotation')
                
                # 4. Industry Rotation
                industry_rotation = MarketIntelligence.detect_industry_rotation(df)
                if not industry_rotation.empty:
                    industry_rotation.to_excel(writer, sheet_name='Industry Rotation')
                
                # 5. Pattern Analysis
                pattern_counts = {}
                for patterns in df['patterns'].dropna():
                    if patterns:
                        for p in patterns.split(' | '):
                            pattern_counts[p] = pattern_counts.get(p, 0) + 1
                
                if pattern_counts:
                    pattern_df = pd.DataFrame(
                        list(pattern_counts.items()),
                        columns=['Pattern', 'Count']
                    ).sort_values('Count', ascending=False)
                    pattern_df.to_excel(writer, sheet_name='Pattern Analysis', index=False)
                
                # 6. Wave Radar Signals
                wave_signals = df[
                    (df['momentum_score'] >= 60) & 
                    (df['acceleration_score'] >= 70) &
                    (df['rvol'] >= 2)
                ].head(50)
                
                if len(wave_signals) > 0:
                    wave_cols = ['ticker', 'company_name', 'master_score', 
                                'momentum_score', 'acceleration_score', 'rvol',
                                'market_state', 'patterns', 'category', 'industry']
                    available_wave_cols = [col for col in wave_cols if col in wave_signals.columns]
                    
                    wave_signals[available_wave_cols].to_excel(
                        writer, sheet_name='Wave Radar', index=False
                    )
                
                # 7. Summary Statistics
                summary_stats = {
                    'Total Stocks': len(df),
                    'Average Master Score': df['master_score'].mean() if 'master_score' in df.columns else 0,
                    'Stocks with Patterns': (df['patterns'] != '').sum() if 'patterns' in df.columns else 0,
                    'High RVOL (>2x)': (df['rvol'] > 2).sum() if 'rvol' in df.columns else 0,
                    'Positive 30D Returns': (df['ret_30d'] > 0).sum() if 'ret_30d' in df.columns else 0,
                    'Template Used': template,
                    'Export Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                logger.info(f"Excel report created successfully with {len(writer.sheets)} sheets")
                
        except Exception as e:
            logger.error(f"Error creating Excel report: {str(e)}")
            raise
        
        output.seek(0)
        return output
    
    @staticmethod
    def create_csv_export(df: pd.DataFrame) -> str:
        """Create CSV export efficiently"""
        
        export_cols = [
            'rank', 'ticker', 'company_name', 'master_score',
            'position_score', 'volume_score', 'momentum_score',
            'acceleration_score', 'breakout_score', 'rvol_score',
            'trend_quality', 'price', 'pe', 'eps_current', 'eps_change_pct',
            'from_low_pct', 'from_high_pct',
            'ret_1d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y',
            'rvol', 'vmi', 'money_flow_mm', 'position_tension',
            'momentum_harmony', 'market_state', 'patterns', 
            'category', 'sector', 'industry', 'eps_tier', 'pe_tier', 'overall_market_strength'
        ]
        
        available_cols = [col for col in export_cols if col in df.columns]
        
        export_df = df[available_cols].copy()
        
        # Convert volume ratios back to percentage
        vol_ratio_cols = [col for col in export_df.columns if 'vol_ratio' in col]
        for col in vol_ratio_cols:
            with np.errstate(divide='ignore', invalid='ignore'):
                export_df[col] = (export_df[col] - 1) * 100
                export_df[col] = export_df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        return export_df.to_csv(index=False)

# ============================================
# UI COMPONENTS
# ============================================

class UIComponents:
    """Reusable UI components with proper tooltips"""
    
    @staticmethod
    def render_metric_card(label: str, value: Any, delta: Optional[str] = None, 
                          help_text: Optional[str] = None) -> None:
        """Render a styled metric card with tooltips"""
        # Add tooltip from CONFIG if available
        metric_key = label.lower().replace(' ', '_')
        if not help_text and metric_key in CONFIG.METRIC_TOOLTIPS:
            help_text = CONFIG.METRIC_TOOLTIPS[metric_key]
        
        if help_text:
            st.metric(label, value, delta, help=help_text)
        else:
            st.metric(label, value, delta)
    
    @staticmethod
    def render_summary_section(df: pd.DataFrame) -> None:
        """🚀 ALL TIME BEST REVOLUTIONARY SUMMARY DASHBOARD - INSTITUTIONAL GRADE INTELLIGENCE 🚀"""
        
        if df.empty:
            st.warning("No data available for summary")
            return
        
        # ================================================================================================
        # 🎯 EXECUTIVE COMMAND CENTER - REAL-TIME MARKET INTELLIGENCE
        # ================================================================================================
        
        # Market Regime Detection
        regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
        
        # ================================================================================================
        # 💎 TIER 1: CRITICAL MARKET PULSE INDICATORS
        # ================================================================================================
        
        st.markdown("### 📊 Market Pulse")
        
        pulse_col1, pulse_col2, pulse_col3, pulse_col4, pulse_col5 = st.columns(5)
        
        with pulse_col1:
            # Advanced A/D Ratio with Institutional Logic
            ad_metrics = MarketIntelligence.calculate_advance_decline_ratio(df)
            ad_ratio = ad_metrics.get('ad_ratio', 1.0)
            advancing = ad_metrics.get('advancing', 0)
            declining = ad_metrics.get('declining', 0)
            
            if ad_ratio == float('inf'):
                ad_status = "🚀💥 EXPLOSIVE"
                ad_display = "∞"
                market_signal = "EXTREME BULLISH"
            elif ad_ratio > 3:
                ad_status = "⚡ POWERFUL"
                ad_display = f"{ad_ratio:.1f}"
                market_signal = "STRONG BULLISH"
            elif ad_ratio > 2:
                ad_status = "🔥 STRONG"
                ad_display = f"{ad_ratio:.1f}"
                market_signal = "BULLISH"
            elif ad_ratio > 1.5:
                ad_status = "📈 POSITIVE"
                ad_display = f"{ad_ratio:.1f}"
                market_signal = "MILD BULLISH"
            elif ad_ratio > 0.8:
                ad_status = "⚖️ NEUTRAL"
                ad_display = f"{ad_ratio:.1f}"
                market_signal = "MIXED"
            else:
                ad_status = "📉 WEAK"
                ad_display = f"{ad_ratio:.1f}"
                market_signal = "BEARISH"
            
            UIComponents.render_metric_card(
                "Market Breadth",
                f"{ad_status}",
                f"A/D: {ad_display} • {market_signal}",
                f"Advance/Decline Ratio: {advancing} advancing vs {declining} declining stocks. Critical market breadth indicator."
            )
        
        with pulse_col2:
            # Institutional Momentum Intelligence
            if 'momentum_score' in df.columns:
                momentum_stats = df['momentum_score'].describe()
                high_momentum = len(df[df['momentum_score'] >= 80])
                elite_momentum = len(df[df['momentum_score'] >= 90])
                momentum_pct = (high_momentum / len(df) * 100) if len(df) > 0 else 0
                
                if momentum_pct > 25:
                    momentum_status = "🚀 EXPLOSIVE"
                    momentum_quality = "INSTITUTIONAL GRADE"
                elif momentum_pct > 15:
                    momentum_status = "🔥 STRONG"
                    momentum_quality = "PROFESSIONAL GRADE"
                elif momentum_pct > 8:
                    momentum_status = "📈 BUILDING"
                    momentum_quality = "DEVELOPING"
                else:
                    momentum_status = "⚖️ WEAK"
                    momentum_quality = "CONSOLIDATING"
                
                UIComponents.render_metric_card(
                    "Momentum Engine",
                    f"{momentum_status}",
                    f"{momentum_pct:.0f}% • {elite_momentum} Elite • {momentum_quality}",
                    f"Momentum analysis: {high_momentum} stocks with institutional-grade momentum (≥80). Avg: {momentum_stats['mean']:.1f}"
                )
            else:
                UIComponents.render_metric_card(
                    "Momentum Engine", 
                    "⚠️ NO DATA",
                    "Momentum data unavailable",
                    "Momentum scoring requires momentum_score column in dataset"
                )
        
        with pulse_col3:
            # Advanced Volume Intelligence
            if 'rvol' in df.columns:
                volume_stats = df['rvol'].describe()
                avg_rvol = df['rvol'].median()
                volume_surge = len(df[df['rvol'] > 3])
                extreme_volume = len(df[df['rvol'] > 5])
                institutional_volume = len(df[df['rvol'] > 2])
                
                if avg_rvol > 2.0:
                    vol_status = "🌊🌊 TSUNAMI"
                    vol_quality = "EXTREME ACTIVITY"
                elif avg_rvol > 1.5:
                    vol_status = "🌊 SURGE"
                    vol_quality = "HIGH ACTIVITY"
                elif avg_rvol > 1.2:
                    vol_status = "💧 ELEVATED"
                    vol_quality = "ACTIVE"
                else:
                    vol_status = "🏜️ QUIET"
                    vol_quality = "LOW ACTIVITY"
                
                UIComponents.render_metric_card(
                    "Volume Intelligence",
                    f"{vol_status}",
                    f"{avg_rvol:.1f}x • {volume_surge} Surges • {vol_quality}",
                    f"Volume analysis: {institutional_volume} stocks with institutional volume (>2x). {extreme_volume} extreme volume spikes."
                )
            else:
                UIComponents.render_metric_card(
                    "Volume Intelligence",
                    "⚠️ NO DATA",
                    "Volume data unavailable",
                    "Volume analysis requires rvol column in dataset"
                )
        
        with pulse_col4:
            # Professional Risk Assessment
            risk_score = 0
            risk_factors = []
            
            # Factor 1: Overextension Risk
            if 'from_high_pct' in df.columns and 'momentum_score' in df.columns:
                overextended = len(df[(df['from_high_pct'] >= -2) & (df['momentum_score'] < 50)])
                if overextended > len(df) * 0.15:
                    risk_score += 25
                    risk_factors.append(f"{overextended} overextended")
            
            # Factor 2: Pump & Dump Risk
            if 'rvol' in df.columns and 'master_score' in df.columns:
                pump_risk = len(df[(df['rvol'] > 8) & (df['master_score'] < 60)])
                if pump_risk > 10:
                    risk_score += 20
                    risk_factors.append(f"{pump_risk} pump suspects")
            
            # Factor 3: Trend Deterioration
            if 'trend_quality' in df.columns:
                weak_trends = len(df[df['trend_quality'] < 40])
                if weak_trends > len(df) * 0.25:
                    risk_score += 15
                    risk_factors.append(f"{weak_trends} weak trends")
            
            # Factor 4: Pattern Quality
            if 'patterns' in df.columns:
                warning_patterns = len(df[df['patterns'].str.contains('🪤|📉|🔉|📊', na=False)])
                if warning_patterns > len(df) * 0.1:
                    risk_score += 20
                    risk_factors.append(f"{warning_patterns} warning patterns")
            
            # Factor 5: Market Regime Risk
            regime_risk = 0
            if regime in ['WEAK_DOWNTREND', 'STRONG_DOWNTREND', 'CRASH']:
                regime_risk = 30
            elif regime in ['DISTRIBUTION', 'PULLBACK']:
                regime_risk = 15
            
            risk_score += regime_risk
            
            # Risk Classification
            if risk_score >= 70:
                risk_status = "🔴 EXTREME"
                risk_action = "DEFENSIVE MODE"
            elif risk_score >= 50:
                risk_status = "🟠 HIGH"
                risk_action = "REDUCE EXPOSURE"
            elif risk_score >= 30:
                risk_status = "🟡 MODERATE"
                risk_action = "SELECTIVE"
            else:
                risk_status = "🟢 LOW"
                risk_action = "OPPORTUNITY MODE"
            
            risk_detail = " • ".join(risk_factors[:2]) if risk_factors else "Minimal risks detected"
            
            UIComponents.render_metric_card(
                "Risk Assessment",
                f"{risk_status}",
                f"Score: {risk_score}/100 • {risk_action}",
                f"Professional risk analysis: {risk_detail}. Market regime: {regime}"
            )
        
        with pulse_col5:
            # Market Regime Intelligence
            regime_strength = regime_metrics.get('strength', 50)
            regime_confidence = regime_metrics.get('confidence', 50)
            
            if regime in ['STRONG_UPTREND', 'PARABOLIC']:
                regime_emoji = "🚀🚀"
                regime_action = "AGGRESSIVE LONG"
            elif regime in ['UPTREND', 'MOMENTUM_BUILD']:
                regime_emoji = "📈"
                regime_action = "LONG BIAS"
            elif regime in ['PULLBACK', 'HEALTHY_CORRECTION']:
                regime_emoji = "🔄"
                regime_action = "BUY DIPS"
            elif regime in ['CONSOLIDATION', 'SIDEWAYS']:
                regime_emoji = "⚖️"
                regime_action = "RANGE TRADE"
            elif regime in ['DISTRIBUTION', 'TOPPING']:
                regime_emoji = "⚠️"
                regime_action = "REDUCE LONGS"
            else:
                regime_emoji = "📉"
                regime_action = "DEFENSIVE"
            
            UIComponents.render_metric_card(
                "Market Regime",
                f"{regime_emoji} {regime.replace('_', ' ')}",
                f"Strength: {regime_strength:.0f}% • {regime_action}",
                f"Institutional market regime analysis. Confidence: {regime_confidence:.0f}%. Strategic positioning: {regime_action}"
            )
        
        # ================================================================================================
        # 🎯 TIER 2: TODAY'S ELITE OPPORTUNITIES - INSTITUTIONAL GRADE
        # ================================================================================================
        
        st.markdown("---")
        st.markdown("### 🎯 Today's Best Opportunities")
        
        opp_col1, opp_col2, opp_col3, opp_col4 = st.columns(4)
        
        with opp_col1:
            # 🌋 Institutional Tsunami - Enhanced Detection
            if 'patterns' in df.columns:
                institutional_tsunami = df[df['patterns'].str.contains('🌋 INSTITUTIONAL TSUNAMI', na=False)]
                if len(institutional_tsunami) > 0:
                    top_tsunami = institutional_tsunami.nlargest(3, 'master_score')
                    
                    st.markdown("**🌋 INSTITUTIONAL TSUNAMI**")
                    for _, stock in top_tsunami.iterrows():
                        company_name = stock.get('company_name', 'N/A')[:20] + "..." if len(stock.get('company_name', '')) > 20 else stock.get('company_name', 'N/A')
                        rvol_val = stock.get('rvol', 0)
                        momentum_val = stock.get('momentum_score', 0)
                        
                        # Enhanced display with multiple metrics
                        st.write(f"🎯 **{stock['ticker']}** - {company_name}")
                        st.caption(f"Score: {stock['master_score']:.1f} | RVOL: {rvol_val:.1f}x | Mom: {momentum_val:.0f}")
                    
                    # Add tsunami strength indicator
                    avg_strength = top_tsunami['master_score'].mean()
                    tsunami_strength = "🔥🔥 MEGA" if avg_strength > 85 else "🔥 STRONG" if avg_strength > 75 else "📈 MODERATE"
                    st.info(f"**Tsunami Strength:** {tsunami_strength}")
                else:
                    st.markdown("**🌋 INSTITUTIONAL TSUNAMI**")
                    st.info("⚖️ No tsunami patterns detected")
                    st.caption("Monitor for institutional accumulation")
            else:
                st.markdown("**🌋 INSTITUTIONAL TSUNAMI**")
                st.warning("Pattern data unavailable")
        
        with opp_col2:
            # 🕰️ Information Decay Arbitrage - Advanced
            if 'patterns' in df.columns:
                info_decay = df[df['patterns'].str.contains('🕰️ INFORMATION DECAY ARBITRAGE', na=False)]
                if len(info_decay) > 0:
                    top_decay = info_decay.nlargest(3, 'master_score')
                    
                    st.markdown("**🕰️ INFO DECAY ARBITRAGE**")
                    for _, stock in top_decay.iterrows():
                        company_name = stock.get('company_name', 'N/A')[:20] + "..." if len(stock.get('company_name', '')) > 20 else stock.get('company_name', 'N/A')
                        vol_score = stock.get('volume_score', 0)
                        position = stock.get('position_score', 0)
                        
                        st.write(f"🎯 **{stock['ticker']}** - {company_name}")
                        st.caption(f"Score: {stock['master_score']:.1f} | Vol: {vol_score:.0f} | Pos: {position:.0f}")
                    
                    # Add decay efficiency indicator  
                    avg_efficiency = top_decay['master_score'].mean()
                    decay_efficiency = "⚡ OPTIMAL" if avg_efficiency > 80 else "📈 GOOD" if avg_efficiency > 70 else "⚖️ FAIR"
                    st.info(f"**Decay Efficiency:** {decay_efficiency}")
                else:
                    st.markdown("**🕰️ INFO DECAY ARBITRAGE**")
                    st.info("⚖️ No decay opportunities")
                    st.caption("Awaiting information asymmetries")
            else:
                st.markdown("**🕰️ INFO DECAY ARBITRAGE**")
                st.warning("Pattern data unavailable")
        
        with opp_col3:
            # 🎆 Earnings Surprise Leaders - Enhanced
            if 'patterns' in df.columns:
                earnings_surprise = df[df['patterns'].str.contains('🎆 EARNINGS SURPRISE LEADER', na=False)]
                if len(earnings_surprise) > 0:
                    top_earnings = earnings_surprise.nlargest(3, 'master_score')
                    
                    st.markdown("**🎆 EARNINGS ROCKETS**")
                    for _, stock in top_earnings.iterrows():
                        company_name = stock.get('company_name', 'N/A')[:20] + "..." if len(stock.get('company_name', '')) > 20 else stock.get('company_name', 'N/A')
                        eps_growth = stock.get('eps_change_pct', 0)
                        trend_quality = stock.get('trend_quality', 0)
                        
                        st.write(f"🎯 **{stock['ticker']}** - {company_name}")
                        st.caption(f"EPS: {eps_growth:.0f}% | Score: {stock['master_score']:.1f} | Trend: {trend_quality:.0f}")
                    
                    # Add earnings momentum indicator
                    avg_eps_growth = top_earnings['eps_change_pct'].mean() if 'eps_change_pct' in top_earnings.columns and len(top_earnings) > 0 else 0
                    earnings_power = "🚀 EXPLOSIVE" if avg_eps_growth > 100 else "🔥 STRONG" if avg_eps_growth > 50 else "📈 SOLID"
                    st.info(f"**Earnings Power:** {earnings_power}")
                else:
                    st.markdown("**🎆 EARNINGS ROCKETS**")
                    st.info("⚖️ No earnings surprises")
                    st.caption("Monitor upcoming earnings")
            else:
                st.markdown("**🎆 EARNINGS ROCKETS**")
                st.warning("Pattern data unavailable")
        
        with opp_col4:
            # 🐦 Phoenix Rising - Turnaround Stories
            if 'patterns' in df.columns:
                phoenix_rising = df[df['patterns'].str.contains('🐦 PHOENIX RISING', na=False)]
                if len(phoenix_rising) > 0:
                    top_phoenix = phoenix_rising.nlargest(3, 'master_score')
                    
                    st.markdown("**🐦 PHOENIX RISING**")
                    for _, stock in top_phoenix.iterrows():
                        company_name = stock.get('company_name', 'N/A')[:20] + "..." if len(stock.get('company_name', '')) > 20 else stock.get('company_name', 'N/A')
                        from_low = stock.get('from_low_pct', 0)
                        acceleration = stock.get('acceleration_score', 0)
                        
                        st.write(f"🎯 **{stock['ticker']}** - {company_name}")
                        st.caption(f"From Low: {from_low:.0f}% | Accel: {acceleration:.0f} | Score: {stock['master_score']:.1f}")
                    
                    # Add transformation strength
                    avg_score = top_phoenix['master_score'].mean()
                    transformation_power = "🔥🔥 MEGA" if avg_score > 85 else "🔥 STRONG" if avg_score > 75 else "📈 EMERGING"
                    st.info(f"**Transformation:** {transformation_power}")
                else:
                    st.markdown("**🐦 PHOENIX RISING**")
                    st.info("⚖️ No phoenix patterns")
                    st.caption("Scanning for turnarounds")
            else:
                st.markdown("**🐦 PHOENIX RISING**")
                st.warning("Pattern data unavailable")
        
        # ================================================================================================
        # ADVANCED MARKET INTELLIGENCE & SECTOR ROTATION
        # ================================================================================================
        
        st.markdown("---")
        st.markdown("### 🧠 **Market Intelligence**")
        
        intel_col1, intel_col2 = st.columns([3, 2])
        
        with intel_col1:
            # Revolutionary Sector Rotation Analysis with LDI Integration
            st.markdown("**🔄 SECTOR ROTATION INTELLIGENCE**")
            
            sector_rotation = MarketIntelligence.detect_sector_rotation(df)
            
            if not sector_rotation.empty:
                # Enhanced visualization with institutional perspective
                fig = go.Figure()
                
                top_12 = sector_rotation.head(12)  # Show more sectors for better analysis
                
                # Create color mapping based on flow score and LDI
                colors = []
                for score in top_12['flow_score']:
                    if score > 75:
                        colors.append('#00ff00')  # Bright green for hot sectors
                    elif score > 60:
                        colors.append('#32cd32')  # Green for strong sectors
                    elif score > 45:
                        colors.append('#ffd700')  # Gold for neutral
                    elif score > 30:
                        colors.append('#ff8c00')  # Orange for weak
                    else:
                        colors.append('#ff4500')  # Red for very weak
                
                fig.add_trace(go.Bar(
                    x=top_12.index,
                    y=top_12['flow_score'],
                    text=[f"{val:.1f}" for val in top_12['flow_score']],
                    textposition='outside',
                    marker_color=colors,
                    marker_line=dict(color='rgba(0,0,0,0.3)', width=1),
                    hovertemplate=(
                        '<b>%{x}</b><br>'
                        'Flow Score: %{y:.1f}<br>'
                        'LDI Score: %{customdata[0]:.1f}%<br>'
                        'Market Leaders: %{customdata[1]} of %{customdata[2]}<br>'
                        'Leadership Density: %{customdata[3]}<br>'
                        'Elite Avg Score: %{customdata[4]:.1f}<br>'
                        'Intelligence: %{customdata[5]}<extra></extra>'
                    ) if all(col in top_12.columns for col in ['ldi_score', 'elite_avg_score', 'ldi_quality']) else (
                        '<b>%{x}</b><br>'
                        'Flow Score: %{y:.1f}<br>'
                        'Stocks Analyzed: %{customdata[0]} of %{customdata[1]}<br>'
                        'Coverage: %{customdata[2]:.1f}%<br>'
                        'Avg Score: %{customdata[3]:.1f}<extra></extra>'
                    ),
                    customdata=np.column_stack((
                        top_12['ldi_score'] if 'ldi_score' in top_12.columns else [0] * len(top_12),
                        top_12['analyzed_stocks'],
                        top_12['total_stocks'],
                        top_12['leadership_density'] if 'leadership_density' in top_12.columns else ['N/A'] * len(top_12),
                        top_12['elite_avg_score'] if 'elite_avg_score' in top_12.columns else top_12['avg_score'],
                        top_12['ldi_quality'] if 'ldi_quality' in top_12.columns else ['Traditional'] * len(top_12)
                    )) if all(col in top_12.columns for col in ['ldi_score', 'elite_avg_score']) else np.column_stack((
                        top_12['analyzed_stocks'],
                        top_12['total_stocks'],
                        top_12['sampling_pct'] if 'sampling_pct' in top_12.columns else [100] * len(top_12),
                        top_12['avg_score']
                    ))
                ))
                
                # Enhanced layout with institutional styling
                fig.update_layout(
                    title="🎯 INSTITUTIONAL SECTOR ROTATION MAP",
                    xaxis_title="Sector",
                    yaxis_title="Enhanced Flow Score",
                    height=450,
                    template='plotly_white',
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(size=12, color='#2c3e50'),
                    title_font=dict(size=16, color='#2c3e50')
                )
                
                # Add horizontal reference lines
                fig.add_hline(y=75, line_dash="dash", line_color="green", annotation_text="Hot Zone")
                fig.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="Neutral Zone")
                fig.add_hline(y=25, line_dash="dash", line_color="red", annotation_text="Cold Zone")
                
                st.plotly_chart(fig, width="stretch", theme="streamlit")
                
                # Add sector insights
                hot_sectors = top_12[top_12['flow_score'] > 75]
                if len(hot_sectors) > 0:
                    st.success(f"🔥 **HOT SECTORS**: {', '.join(hot_sectors.index[:3])}")
                
                cold_sectors = top_12[top_12['flow_score'] < 35]
                if len(cold_sectors) > 0:
                    st.warning(f"❄️ **AVOID SECTORS**: {', '.join(cold_sectors.index[-2:])}")
                    
            else:
                st.info("📊 Sector rotation data processing...")
                st.caption("Ensure sector data is available in your dataset")
        
        with intel_col2:
            # Enhanced Market Regime Detection with Action Items
            regime, regime_metrics = MarketIntelligence.detect_market_regime(df)
            
            st.markdown("**🎯 MARKET REGIME ANALYSIS**")
            
            # Regime display with enhanced styling
            regime_colors = {
                'STRONG_UPTREND': '🟢',
                'UPTREND': '🟢',
                'PULLBACK': '🟡',
                'SIDEWAYS': '🟠',
                'WEAK_UPTREND': '🟡',
                'WEAK_DOWNTREND': '🟠',
                'DOWNTREND': '🔴',
                'STRONG_DOWNTREND': '🔴'
            }
            
            regime_emoji = regime_colors.get(regime, '⚪')
            st.markdown(f"### {regime_emoji} **{regime}**")
            
            # Detailed regime metrics
            st.markdown("**📡 INTELLIGENCE SIGNALS**")
            
            signals = []
            signal_strength = 0
            
            # Breadth Analysis
            breadth = regime_metrics.get('breadth', 0.5)
            if breadth > 0.7:
                signals.append("✅ Exceptional breadth")
                signal_strength += 3
            elif breadth > 0.6:
                signals.append("✅ Strong breadth")
                signal_strength += 2
            elif breadth > 0.4:
                signals.append("⚖️ Moderate breadth")
                signal_strength += 1
            else:
                signals.append("⚠️ Weak breadth")
            
            # Category Leadership Analysis
            category_spread = regime_metrics.get('category_spread', 0)
            if category_spread > 15:
                signals.append("🔄 Small caps LEADING")
                signal_strength += 2
            elif category_spread > 5:
                signals.append("🔄 Small caps active")
                signal_strength += 1
            elif category_spread < -15:
                signals.append("🛡️ Large caps DEFENSIVE")
            elif category_spread < -5:
                signals.append("🛡️ Large caps preferred")
                signal_strength += 1
            
            # Volume Analysis
            avg_rvol = regime_metrics.get('avg_rvol', 1.0)
            if avg_rvol > 2.0:
                signals.append("🌊🌊 MASSIVE volume")
                signal_strength += 3
            elif avg_rvol > 1.5:
                signals.append("🌊 High volume activity")
                signal_strength += 2
            elif avg_rvol > 1.2:
                signals.append("💧 Normal volume")
                signal_strength += 1
            else:
                signals.append("🏜️ Low volume")
            
            # Pattern Emergence
            if 'patterns' in df.columns:
                pattern_count = (df['patterns'] != '').sum()
                pattern_density = pattern_count / len(df) if len(df) > 0 else 0
                if pattern_density > 0.3:
                    signals.append("🎯 MANY patterns emerging")
                    signal_strength += 2
                elif pattern_density > 0.2:
                    signals.append("🎯 Patterns developing")
                    signal_strength += 1
                elif pattern_density > 0.1:
                    signals.append("⚖️ Few patterns")
                else:
                    signals.append("📉 Pattern scarcity")
            
            for signal in signals:
                st.write(signal)
            
            # Market Strength Meter
            st.markdown("**💪 MARKET STRENGTH COMPOSITE**")
            
            strength_score = min((
                (breadth * 40) +
                (min(avg_rvol, 3) / 3 * 30) +
                (max(0, category_spread) / 20 * 15) +
                ((pattern_count / len(df)) * 15 if 'patterns' in df.columns and len(df) > 0 else 0)
            ), 100)
            
            # Enhanced strength visualization
            if strength_score > 85:
                strength_meter = "🟢🟢🟢🟢🟢"
                strength_label = "EXCEPTIONAL"
            elif strength_score > 70:
                strength_meter = "🟢🟢🟢🟢⚪"
                strength_label = "STRONG"
            elif strength_score > 55:
                strength_meter = "🟢🟢🟢⚪⚪"
                strength_label = "MODERATE"
            elif strength_score > 40:
                strength_meter = "🟢🟢⚪⚪⚪"
                strength_label = "WEAK"
            else:
                strength_meter = "🟢⚪⚪⚪⚪"
                strength_label = "VERY WEAK"
            
            st.write(f"{strength_meter}")
            st.write(f"**{strength_label}** ({strength_score:.0f}/100)")
            
            # Action Items based on regime
            st.markdown("**🎯 ACTION ITEMS**")
            if regime in ['STRONG_UPTREND', 'UPTREND']:
                st.info("• Focus on momentum leaders\n• Increase position sizes\n• Look for breakout patterns")
            elif regime in ['PULLBACK']:
                st.warning("• Prepare shopping lists\n• Watch for reversal patterns\n• Reduce position sizes")
            elif regime in ['SIDEWAYS']:
                st.info("• Trade range-bound setups\n• Focus on earnings plays\n• Neutral position sizing")
            else:
                st.error("• Defensive positioning\n• Cash preservation\n• Avoid new positions")

        # ================================================================================================
        # PERFORMANCE ATTRIBUTION & ADVANCED METRICS
        # ================================================================================================
        
        st.markdown("---")
        st.markdown("### 🏆PERFORMANCE ATTRIBUTION ANALYSIS")
        
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.markdown("**📊 SCORE COMPONENT BREAKDOWN**")
            
            # Advanced score component analysis
            score_components = {}
            if 'momentum_score' in df.columns:
                score_components['Momentum'] = df['momentum_score'].mean()
            if 'acceleration_score' in df.columns:
                score_components['Acceleration'] = df['acceleration_score'].mean()
            if 'breakout_score' in df.columns:
                score_components['Breakout'] = df['breakout_score'].mean()
            if 'position_score' in df.columns:
                score_components['Position'] = df['position_score'].mean()
            if 'volume_score' in df.columns:
                score_components['Volume'] = df['volume_score'].mean()
            if 'rvol_score' in df.columns:
                score_components['RVOL'] = df['rvol_score'].mean()
            
            if score_components:
                # Create component performance dataframe
                component_df = pd.DataFrame(list(score_components.items()), columns=['Component', 'Score'])
                component_df['Quality'] = component_df['Score'].apply(
                    lambda x: '🔥 Elite' if x >= 75 else '📈 Strong' if x >= 65 else '⚖️ Average' if x >= 50 else '📉 Weak'
                )
                component_df = component_df.sort_values('Score', ascending=False)
                
                st.dataframe(
                    component_df,
                    width='stretch',
                    hide_index=True,
                    column_config={
                        'Component': st.column_config.TextColumn("Component", width="medium"),
                        'Score': st.column_config.ProgressColumn("Average Score", min_value=0, max_value=100, format="%.1f"),
                        'Quality': st.column_config.TextColumn("Quality", width="small")
                    }
                )
                
                # Highlight best component
                best_component = component_df.iloc[0]['Component']
                best_score = component_df.iloc[0]['Score']
                st.success(f"🏆 **Strongest Factor**: {best_component} ({best_score:.1f})")
            else:
                st.info("📊 Score breakdown unavailable")
        
        with perf_col2:
            st.markdown("**💰 RETURN ATTRIBUTION MATRIX**")
            
            # Multi-timeframe return analysis
            return_periods = {
                '1D': 'ret_1d',
                '7D': 'ret_7d', 
                '30D': 'ret_30d',
                '3M': 'ret_3m',
                '1Y': 'ret_1y'
            }
            
            return_analysis = {}
            for period, column in return_periods.items():
                if column in df.columns:
                    positive_count = len(df[df[column] > 0])
                    total_count = len(df[df[column].notna()])
                    win_rate = (positive_count / total_count * 100) if total_count > 0 else 0
                    avg_return = df[column].mean()
                    
                    return_analysis[period] = {
                        'Win Rate': win_rate,
                        'Avg Return': avg_return,
                        'Quality': '🔥' if win_rate > 60 and avg_return > 2 else '📈' if win_rate > 50 else '⚖️' if win_rate > 40 else '📉'
                    }
            
            if return_analysis:
                return_df = pd.DataFrame(return_analysis).T
                
                st.dataframe(
                    return_df,
                    width='stretch',
                    column_config={
                        'Win Rate': st.column_config.ProgressColumn("Win Rate %", min_value=0, max_value=100, format="%.1f"),
                        'Avg Return': st.column_config.NumberColumn("Avg Return %", format="%.2f%%"),
                        'Quality': st.column_config.TextColumn("Signal", width="small")
                    }
                )
                
                # Find best performing timeframe
                if len(return_df) > 0:
                    best_period = return_df['Avg Return'].idxmax()
                    best_return = return_df.loc[best_period, 'Avg Return']
                    st.success(f"🎯 **Best Timeframe**: {best_period} ({best_return:.2f}%)")
            else:
                st.info("📊 Return data unavailable")
        
        with perf_col3:
            st.markdown("**🎯 PATTERN EFFECTIVENESS**")
            
            # Advanced pattern analysis
            if 'patterns' in df.columns and 'master_score' in df.columns:
                pattern_performance = {}
                
                # Extract and analyze individual patterns
                for _, row in df.iterrows():
                    if pd.notna(row['patterns']) and row['patterns']:
                        patterns = row['patterns'].split(' | ')
                        for pattern in patterns:
                            pattern = pattern.strip()
                            if pattern:
                                if pattern not in pattern_performance:
                                    pattern_performance[pattern] = {'scores': [], 'count': 0}
                                pattern_performance[pattern]['scores'].append(row['master_score'])
                                pattern_performance[pattern]['count'] += 1
                
                # Calculate effectiveness metrics
                pattern_stats = []
                for pattern, data in pattern_performance.items():
                    if data['count'] >= 3:  # Only patterns with sufficient sample size
                        avg_score = np.mean(data['scores'])
                        pattern_stats.append({
                            'Pattern': pattern[:20] + '...' if len(pattern) > 20 else pattern,
                            'Count': data['count'],
                            'Avg Score': avg_score,
                            'Effectiveness': '🔥 Elite' if avg_score > 80 else '📈 Strong' if avg_score > 70 else '⚖️ Average'
                        })
                
                if pattern_stats:
                    pattern_df = pd.DataFrame(pattern_stats).sort_values('Avg Score', ascending=False).head(8)
                    
                    st.dataframe(
                        pattern_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'Pattern': st.column_config.TextColumn("Pattern", width="medium"),
                            'Count': st.column_config.NumberColumn("Count", width="small"),
                            'Avg Score': st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=100, format="%.1f"),
                            'Effectiveness': st.column_config.TextColumn("Quality", width="small")
                        }
                    )
                    
                    # Highlight most effective pattern
                    best_pattern = pattern_df.iloc[0]['Pattern']
                    best_effectiveness = pattern_df.iloc[0]['Avg Score']
                    st.success(f"🏆 **Most Effective**: {best_pattern} ({best_effectiveness:.1f})")
                else:
                    st.info("📊 Insufficient pattern data")
            else:
                st.info("📊 Pattern analysis unavailable")

# ============================================
# SESSION STATE MANAGER
# ============================================

class SessionStateManager:
    """
    Unified session state manager for Streamlit.
    This class ensures all state variables are properly initialized,
    preventing runtime errors and managing filter states consistently.
    """

    @staticmethod
    def initialize():
        """
        Initializes all necessary session state variables with explicit defaults.
        This prevents KeyErrors when accessing variables for the first time.
        """
        defaults = {
            # Core Application State
            'search_query': "",
            'last_refresh': datetime.now(timezone.utc),
            'data_source': "sheet",
            'user_preferences': {
                'default_top_n': CONFIG.DEFAULT_TOP_N,
                'display_mode': 'Hybrid (Technical + Fundamentals)',
                'last_filters': {}
            },
            'active_filter_count': 0,
            'quick_filter': None,
            'quick_filter_applied': False,
            'show_debug': False,
            'performance_metrics': {},
            'data_quality': {},
            
            # Legacy filter keys (for backward compatibility)
            'display_count': CONFIG.DEFAULT_TOP_N,
            'sort_by': 'Rank',
            'export_template': 'Full Analysis (All Data)',
            'category_filter': [],
            'sector_filter': [],
            'industry_filter': [],
            'min_score': 0,
            'patterns': [],
            'trend_filter': "All Trends",
            'eps_tier_filter': [],
            'pe_tier_filter': [],
            'price_tier_filter': [],
            'eps_change_tiers': [],
            'position_tiers': [],
            'position_range': (0, 100),
            'performance_tiers': [],
            'performance_custom_range': (-100, 500),
            'min_eps_change': "",
            'min_pe': None,
            'max_pe': None,
            'require_fundamental_data': False,
            
            # Wave Radar specific filters
            'market_states_filter': [],
            'market_strength_range_slider': (0, 100),
            'long_term_strength_range_slider': (0, 100),
            'show_sensitivity_details': False,
            'show_market_regime': True,
            'wave_timeframe_select': "All Waves",
            'wave_sensitivity': "Balanced",
            
            # Sheet configuration
            'sheet_id': '',
            'gid': CONFIG.DEFAULT_GID
        }
        
        # Initialize default values
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Initialize centralized filter state (NEW)
        if 'filter_state' not in st.session_state:
            st.session_state.filter_state = {
                'categories': [],
                'sectors': [],
                'industries': [],
                'min_score': 0,
                'patterns': [],
                'trend_filter': "All Trends",
                'trend_range': (0, 100),
                'eps_tiers': [],
                'pe_tiers': [],
                'price_tiers': [],
                'eps_change_tiers': [],
                'position_tiers': [],
                'position_range': (0, 100),
                'performance_tiers': [],
                'performance_custom_range': (-100, 500),
                'volume_tiers': [],
                'rvol_range': (0.1, 20.0),
                'vmi_tiers': [],
                'custom_vmi_range': (0.5, 3.0),
                'momentum_harmony_tiers': [],
                'ret_1d_range': (2.0, 25.0),
                'ret_3d_range': (3.0, 50.0),
                'ret_7d_range': (5.0, 75.0),
                'ret_30d_range': (10.0, 150.0),
                'ret_3m_range': (15.0, 200.0),
                'ret_6m_range': (20.0, 500.0),
                'ret_1y_range': (25.0, 1000.0),
                'ret_3y_range': (50.0, 2000.0),
                'ret_5y_range': (75.0, 5000.0),
                'min_eps_change': None,
                'min_pe': None,
                'max_pe': None,
                'require_fundamental_data': False,
                'market_states': [],
                'market_strength_range': (0, 100),
                'long_term_strength_range': (0, 100),
                'position_score_range': (0, 100),
                'volume_score_range': (0, 100),
                'momentum_score_range': (0, 100),
                'acceleration_score_range': (0, 100),
                'breakout_score_range': (0, 100),
                'rvol_score_range': (0, 100),
                'position_score_selection': "All Scores",
                'volume_score_selection': "All Scores",
                'momentum_score_selection': "All Scores",
                'acceleration_score_selection': "All Scores",
                'breakout_score_selection': "All Scores",
                'rvol_score_selection': "All Scores",
                'ret_1d_selection': "All Returns",
                'ret_3d_selection': "All Returns",
                'ret_7d_selection': "All Returns",
                'ret_30d_selection': "All Returns",
                'ret_3m_selection': "All Returns",
                'ret_6m_selection': "All Returns",
                'ret_1y_selection': "All Returns",
                'ret_3y_selection': "All Returns",
                'ret_5y_selection': "All Returns",
                'quick_filter': None,
                'quick_filter_applied': False
            }
        
        # CRITICAL FIX: Clean up any corrupted performance_tiers values
        # This prevents the "default value not in options" error
        valid_performance_options = [
            "🚀 Strong Gainers (>3% 1D)",
            "⚡ Power Moves (>7% 1D)",
            "💥 Explosive (>15% 1D)",
            "🌟 3-Day Surge (>6% 3D)",
            "📈 Weekly Winners (>12% 7D)",
            "🏆 Monthly Champions (>25% 30D)",
            "🎯 Quarterly Stars (>40% 3M)",
            "💎 Half-Year Heroes (>60% 6M)",
            "🌙 Annual Winners (>80% 1Y)",
            "👑 Multi-Year Champions (>150% 3Y)",
            "🏛️ Long-Term Legends (>250% 5Y)",
            "🎯 Custom Range"
        ]
        
        # Clean up performance_tiers in both filter_state and legacy state
        for state_key in ['filter_state', 'performance_tiers']:
            if state_key == 'filter_state' and 'filter_state' in st.session_state:
                if 'performance_tiers' in st.session_state.filter_state:
                    current_tiers = st.session_state.filter_state['performance_tiers']
                    if isinstance(current_tiers, list):
                        cleaned_tiers = [tier for tier in current_tiers if tier in valid_performance_options]
                        st.session_state.filter_state['performance_tiers'] = cleaned_tiers
            elif state_key == 'performance_tiers' and 'performance_tiers' in st.session_state:
                current_tiers = st.session_state['performance_tiers']
                if isinstance(current_tiers, list):
                    cleaned_tiers = [tier for tier in current_tiers if tier in valid_performance_options]
                    st.session_state['performance_tiers'] = cleaned_tiers

    @staticmethod
    def build_filter_dict() -> Dict[str, Any]:
        """
        Builds a comprehensive filter dictionary from the current session state.
        This centralizes filter data for easy consumption by the FilterEngine.
        
        Returns:
            Dict[str, Any]: A dictionary of all active filter settings.
        """
        filters = {}
        
        # Use centralized filter state if available
        if 'filter_state' in st.session_state:
            state = st.session_state.filter_state
            
            # Map centralized state to filter dict
            if state.get('categories'):
                filters['categories'] = state['categories']
            if state.get('sectors'):
                filters['sectors'] = state['sectors']
            if state.get('industries'):
                filters['industries'] = state['industries']
            if state.get('min_score', 0) > 0:
                filters['min_score'] = state['min_score']
            if state.get('patterns'):
                filters['patterns'] = state['patterns']
            if state.get('trend_filter') != "All Trends":
                filters['trend_filter'] = state['trend_filter']
                filters['trend_range'] = state.get('trend_range', (0, 100))
            if state.get('eps_tiers'):
                filters['eps_tiers'] = state['eps_tiers']
            if state.get('pe_tiers'):
                filters['pe_tiers'] = state['pe_tiers']
            if state.get('price_tiers'):
                filters['price_tiers'] = state['price_tiers']
            if state.get('eps_change_tiers'):
                filters['eps_change_tiers'] = state['eps_change_tiers']
            if state.get('min_pe') is not None:
                filters['min_pe'] = state['min_pe']
            if state.get('max_pe') is not None:
                filters['max_pe'] = state['max_pe']
            if state.get('require_fundamental_data'):
                filters['require_fundamental_data'] = True
            if state.get('market_states'):
                filters['market_states'] = state['market_states']
            if state.get('market_strength_range') != (0, 100):
                filters['market_strength_range'] = state['market_strength_range']
            if state.get('long_term_strength_range') != (0, 100):
                filters['long_term_strength_range'] = state['long_term_strength_range']
            if state.get('performance_tiers'):
                filters['performance_tiers'] = state['performance_tiers']
            if state.get('position_tiers'):
                filters['position_tiers'] = state['position_tiers']
            if state.get('volume_tiers'):
                filters['volume_tiers'] = state['volume_tiers']
            if state.get('vmi_tiers'):
                filters['vmi_tiers'] = state['vmi_tiers']
            if state.get('momentum_harmony_tiers'):
                filters['momentum_harmony_tiers'] = state['momentum_harmony_tiers']
            if state.get('custom_vmi_range') != (0.5, 3.0):
                filters['custom_vmi_range'] = state['custom_vmi_range']
            if state.get('position_score_range') != (0, 100):
                filters['position_score_range'] = state['position_score_range']
            if state.get('volume_score_range') != (0, 100):
                filters['volume_score_range'] = state['volume_score_range']
            if state.get('momentum_score_range') != (0, 100):
                filters['momentum_score_range'] = state['momentum_score_range']
            if state.get('acceleration_score_range') != (0, 100):
                filters['acceleration_score_range'] = state['acceleration_score_range']
            if state.get('breakout_score_range') != (0, 100):
                filters['breakout_score_range'] = state['breakout_score_range']
            if state.get('rvol_score_range') != (0, 100):
                filters['rvol_score_range'] = state['rvol_score_range']
            
            # Individual Performance Return Period Filters
            # Define default ranges to compare against
            default_ranges = {
                'ret_1d_range': (2.0, 25.0),
                'ret_3d_range': (3.0, 50.0),
                'ret_7d_range': (5.0, 75.0),
                'ret_30d_range': (10.0, 150.0),
                'ret_3m_range': (15.0, 200.0),
                'ret_6m_range': (20.0, 500.0),
                'ret_1y_range': (25.0, 1000.0),
                'ret_3y_range': (50.0, 2000.0),
                'ret_5y_range': (75.0, 5000.0)
            }
            
            # Add individual return period filters if they differ from defaults
            for ret_col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']:
                range_key = f'{ret_col}_range'
                if state.get(range_key) and state[range_key] != default_ranges.get(range_key):
                    filters[range_key] = state[range_key]
                
        else:
            # Fallback to legacy individual keys
            # Categorical filters
            for key, filter_name in [
                ('category_filter', 'categories'), 
                ('sector_filter', 'sectors'), 
                ('industry_filter', 'industries')
            ]:
                if st.session_state.get(key) and st.session_state[key]:
                    filters[filter_name] = st.session_state[key]
            
            # Numeric filters
            if st.session_state.get('min_score', 0) > 0:
                filters['min_score'] = st.session_state['min_score']
            
            # PE filters
            if st.session_state.get('min_pe'):
                value = st.session_state['min_pe']
                if isinstance(value, str) and value.strip():
                    try:
                        filters['min_pe'] = float(value)
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)):
                    filters['min_pe'] = float(value)
            
            if st.session_state.get('max_pe'):
                value = st.session_state['max_pe']
                if isinstance(value, str) and value.strip():
                    try:
                        filters['max_pe'] = float(value)
                    except ValueError:
                        pass
                elif isinstance(value, (int, float)):
                    filters['max_pe'] = float(value)

            # Multi-select filters
            if st.session_state.get('patterns') and st.session_state['patterns']:
                filters['patterns'] = st.session_state['patterns']
            
            # Tier filters
            for key, filter_name in [
                ('eps_tier_filter', 'eps_tiers'),
                ('pe_tier_filter', 'pe_tiers'),
                ('price_tier_filter', 'price_tiers'),
                ('eps_change_tier_filter', 'eps_change_tiers')
            ]:
                if st.session_state.get(key) and st.session_state[key]:
                    filters[filter_name] = st.session_state[key]
            
            # Trend filter
            if st.session_state.get('trend_filter') != "All Trends":
                trend_options = {
                    "🔥 Exceptional (85+)": (85, 100),
                    "🚀 Strong (70-84)": (70, 84),
                    "✅ Good (55-69)": (55, 69),
                    "➡️ Neutral (40-54)": (40, 54),
                    "⚠️ Weak (25-39)": (25, 39),
                    "🔻 Poor (<25)": (0, 24)
                }
                filters['trend_filter'] = st.session_state['trend_filter']
                filters['trend_range'] = trend_options.get(st.session_state['trend_filter'], (0, 100))
            
            # Market filters
            if st.session_state.get('market_strength_range_slider') != (0, 100):
                filters['market_strength_range'] = st.session_state['market_strength_range_slider']
            
            if st.session_state.get('long_term_strength_range_slider') != (0, 100):
                filters['long_term_strength_range'] = st.session_state['long_term_strength_range_slider']
            
            if st.session_state.get('market_states_filter') and st.session_state['market_states_filter']:
                filters['market_states'] = st.session_state['market_states_filter']
            
            # Checkbox filters
            if st.session_state.get('require_fundamental_data', False):
                filters['require_fundamental_data'] = True
            
        return filters

    @staticmethod
    def clear_filters():
        """
        Resets all filter-related session state keys to their default values.
        This provides a clean slate for the user.
        FIXED: Now properly cleans ALL dynamic widget keys.
        """
        # Clear the centralized filter state
        if 'filter_state' in st.session_state:
            st.session_state.filter_state = {
                'categories': [],
                'sectors': [],
                'industries': [],
                'min_score': 0,
                'patterns': [],
                'trend_filter': "All Trends",
                'trend_range': (0, 100),
                'trend_custom_range': (0, 100),
                'eps_tiers': [],
                'pe_tiers': [],
                'price_tiers': [],
                'eps_change_tiers': [],
                'position_tiers': [],
                'position_range': (0, 100),
                'performance_tiers': [],
                'performance_custom_range': (-100, 500),
                'volume_tiers': [],
                'rvol_range': (0.1, 20.0),
                'min_eps_change': None,
                'min_pe': None,
                'max_pe': None,
                'require_fundamental_data': False,
                'market_states': [],
                'market_strength_range': (0, 100),
                'long_term_strength_range': (0, 100),
                'position_score_range': (0, 100),
                'volume_score_range': (0, 100),
                'momentum_score_range': (0, 100),
                'acceleration_score_range': (0, 100),
                'breakout_score_range': (0, 100),
                'rvol_score_range': (0, 100),
                'position_score_selection': "All Scores",
                'volume_score_selection': "All Scores",
                'momentum_score_selection': "All Scores",
                'acceleration_score_selection': "All Scores",
                'breakout_score_selection': "All Scores",
                'rvol_score_selection': "All Scores",
                # Performance filter selections
                'ret_1d_selection': "All Returns",
                'ret_3d_selection': "All Returns", 
                'ret_7d_selection': "All Returns",
                'ret_30d_selection': "All Returns",
                'ret_3m_selection': "All Returns",
                'ret_6m_selection': "All Returns",
                'ret_1y_selection': "All Returns",
                'ret_3y_selection': "All Returns",
                'ret_5y_selection': "All Returns",
                # Performance filter ranges
                'ret_1d_range': (2.0, 25.0),
                'ret_3d_range': (3.0, 50.0),
                'ret_7d_range': (5.0, 75.0),
                'ret_30d_range': (10.0, 150.0),
                'ret_3m_range': (15.0, 200.0),
                'ret_6m_range': (20.0, 500.0),
                'ret_1y_range': (25.0, 1000.0),
                'ret_3y_range': (50.0, 2000.0),
                'ret_5y_range': (75.0, 5000.0),
                'quick_filter': None,
                'quick_filter_applied': False
            }
        
        # Clear individual legacy filter keys
        filter_keys = [
            'category_filter', 'sector_filter', 'industry_filter', 'eps_tier_filter',
            'pe_tier_filter', 'price_tier_filter', 'eps_change_tier_filter', 'patterns', 'min_score', 'trend_filter',
            'min_pe', 'max_pe', 'require_fundamental_data',
            'quick_filter', 'quick_filter_applied', 'market_states_filter',
            'market_strength_range_slider', 'long_term_strength_range_slider', 'show_sensitivity_details', 'show_market_regime',
            'wave_timeframe_select', 'wave_sensitivity'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    elif key == 'wave_timeframe_select':
                        st.session_state[key] = "All Waves"
                    elif key == 'wave_sensitivity':
                        st.session_state[key] = "Balanced"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple):
                    if key == 'market_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    if key == 'min_score':
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = None if key in ['min_pe', 'max_pe'] else 0
                else:
                    st.session_state[key] = None
        
        # CRITICAL FIX: Delete all widget keys to force UI reset
        widget_keys_to_delete = [
            # Multiselect widgets
            'category_multiselect', 'sector_multiselect', 'industry_multiselect',
            'patterns_multiselect', 'market_states_multiselect',
            'eps_tier_multiselect', 'pe_tier_multiselect', 'price_tier_multiselect',
            'eps_change_tiers_widget', 'performance_tier_multiselect', 'position_tier_multiselect',
            'volume_tier_multiselect',
            'performance_tier_multiselect_intelligence', 'volume_tier_multiselect_intelligence',
            'position_tier_multiselect_intelligence',
            
            # Slider widgets
            'min_score_slider', 'market_strength_slider', 'performance_custom_range_slider',
            'trend_custom_range_slider',
            'ret_1d_range_slider', 'ret_3d_range_slider', 'ret_7d_range_slider', 'ret_30d_range_slider',
            'ret_3m_range_slider', 'ret_6m_range_slider', 'ret_1y_range_slider', 'ret_3y_range_slider', 'ret_5y_range_slider',
            'position_range_slider', 'rvol_range_slider',
            'position_score_slider', 'volume_score_slider', 'momentum_score_slider',
            'acceleration_score_slider', 'breakout_score_slider', 'rvol_score_slider',
            
            # Score dropdown widgets
            'position_score_dropdown', 'volume_score_dropdown', 'momentum_score_dropdown',
            'acceleration_score_dropdown', 'breakout_score_dropdown', 'rvol_score_dropdown',
            
            # Selectbox widgets
            'trend_selectbox', 'wave_timeframe_select', 'display_mode_toggle',
            
            # Text input widgets
            'eps_change_input', 'min_pe_input', 'max_pe_input',
            
            # Checkbox widgets
            'require_fundamental_checkbox', 'show_sensitivity_details', 'show_market_regime',
            
            # Additional keys
            'display_count_select', 'sort_by_select', 'export_template_radio',
            'wave_sensitivity', 'search_input', 'sheet_input', 'gid_input'
        ]
        
        # Delete each widget key if it exists
        deleted_count = 0
        for key in widget_keys_to_delete:
            if key in st.session_state:
                del st.session_state[key]
                deleted_count += 1
        
        # ==== MEMORY LEAK FIX - START ====
        # Clean up ANY dynamically created widget keys that weren't in the predefined list
        # This catches widgets created on the fly or with dynamic keys
        
        all_widget_patterns = [
            '_multiselect', '_slider', '_selectbox', '_checkbox', 
            '_input', '_radio', '_button', '_expander', '_toggle',
            '_number_input', '_text_area', '_date_input', '_time_input',
            '_color_picker', '_file_uploader', '_camera_input'
        ]
        
        # Collect keys to delete (can't modify dict during iteration)
        dynamic_keys_to_delete = []
        
        for key in list(st.session_state.keys()):
            # Check if this key ends with any widget pattern
            for pattern in all_widget_patterns:
                if pattern in key and key not in widget_keys_to_delete:
                    dynamic_keys_to_delete.append(key)
                    break
        
        # Delete the dynamic keys
        for key in dynamic_keys_to_delete:
            try:
                del st.session_state[key]
                deleted_count += 1
                logger.debug(f"Deleted dynamic widget key: {key}")
            except KeyError:
                # Key might have been deleted already
                pass
        
        # Also clean up any keys that start with 'FormSubmitter'
        form_keys_to_delete = [key for key in st.session_state.keys() if key.startswith('FormSubmitter')]
        for key in form_keys_to_delete:
            try:
                del st.session_state[key]
                deleted_count += 1
            except KeyError:
                pass
        
        # ==== MEMORY LEAK FIX - END ====
        
        # Also clear legacy filter keys for backward compatibility
        legacy_keys = [
            'category_filter', 'sector_filter', 'industry_filter',
            'min_score', 'patterns', 'trend_filter',
            'eps_tier_filter', 'pe_tier_filter', 'price_tier_filter',
            'min_eps_change', 'min_pe', 'max_pe',
            'require_fundamental_data', 'market_states_filter',
            'market_strength_range_slider'
        ]
        
        for key in legacy_keys:
            if key in st.session_state:
                if isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                elif isinstance(st.session_state[key], bool):
                    st.session_state[key] = False
                elif isinstance(st.session_state[key], str):
                    if key == 'trend_filter':
                        st.session_state[key] = "All Trends"
                    else:
                        st.session_state[key] = ""
                elif isinstance(st.session_state[key], tuple):
                    if key == 'market_strength_range_slider':
                        st.session_state[key] = (0, 100)
                elif isinstance(st.session_state[key], (int, float)):
                    if key == 'min_score':
                        st.session_state[key] = 0
                    else:
                        st.session_state[key] = None
                else:
                    st.session_state[key] = None
        
        # Reset active filter count
        st.session_state.active_filter_count = 0
        
        # Clear quick filter
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        # Clear any cached filter results
        if 'user_preferences' in st.session_state:
            st.session_state.user_preferences['last_filters'] = {}
        
        logger.info(f"All filters and widget states cleared successfully. Deleted {deleted_count} widget keys.")
    
    @staticmethod
    def sync_filter_states():
        """
        Synchronizes legacy individual filter keys with centralized filter state.
        This ensures backward compatibility during transition.
        """
        if 'filter_state' not in st.session_state:
            return
        
        state = st.session_state.filter_state
        
        # Sync from centralized to individual (for widgets that still use old keys)
        mappings = [
            ('categories', 'category_filter'),
            ('sectors', 'sector_filter'),
            ('industries', 'industry_filter'),
            ('min_score', 'min_score'),
            ('patterns', 'patterns'),
            ('trend_filter', 'trend_filter'),
            ('eps_tiers', 'eps_tier_filter'),
            ('pe_tiers', 'pe_tier_filter'),
            ('price_tiers', 'price_tier_filter'),
            ('eps_change_tiers', 'eps_change_tier_filter'),
            ('min_pe', 'min_pe'),
            ('max_pe', 'max_pe'),
            ('require_fundamental_data', 'require_fundamental_data'),
            ('market_states', 'market_states_filter'),
            ('market_strength_range', 'market_strength_range_slider'),
        ]
        
        for state_key, session_key in mappings:
            if state_key in state:
                st.session_state[session_key] = state[state_key]
    
    @staticmethod
    def get_active_filter_count() -> int:
        """
        Counts the number of active filters.
        
        Returns:
            int: Number of active filters.
        """
        count = 0
        
        if 'filter_state' in st.session_state:
            state = st.session_state.filter_state
            
            if state.get('categories'): count += 1
            if state.get('sectors'): count += 1
            if state.get('industries'): count += 1
            if state.get('min_score', 0) > 0: count += 1
            if state.get('patterns'): count += 1
            if state.get('trend_filter') != "All Trends": count += 1
            if state.get('eps_tiers'): count += 1
            if state.get('pe_tiers'): count += 1
            if state.get('price_tiers'): count += 1
            if state.get('eps_change_tiers'): count += 1
            if state.get('min_pe') is not None: count += 1
            if state.get('max_pe') is not None: count += 1
            if state.get('require_fundamental_data'): count += 1
            if state.get('market_states'): count += 1
            if state.get('market_strength_range') != (0, 100): count += 1
            if state.get('long_term_strength_range') != (0, 100): count += 1
        else:
            # Fallback to old method
            filter_checks = [
                ('category_filter', lambda x: x and len(x) > 0),
                ('sector_filter', lambda x: x and len(x) > 0),
                ('industry_filter', lambda x: x and len(x) > 0),
                ('min_score', lambda x: x > 0),
                ('patterns', lambda x: x and len(x) > 0),
                ('trend_filter', lambda x: x != 'All Trends'),
                ('eps_tier_filter', lambda x: x and len(x) > 0),
                ('pe_tier_filter', lambda x: x and len(x) > 0),
                ('price_tier_filter', lambda x: x and len(x) > 0),
                ('eps_change_tier_filter', lambda x: x and len(x) > 0),
                ('min_pe', lambda x: x is not None and str(x).strip() != ''),
                ('max_pe', lambda x: x is not None and str(x).strip() != ''),
                ('require_fundamental_data', lambda x: x),
                ('market_states_filter', lambda x: x and len(x) > 0),
                ('market_strength_range_slider', lambda x: x != (0, 100)),
                ('long_term_strength_range_slider', lambda x: x != (0, 100))
            ]
            
            for key, check_func in filter_checks:
                value = st.session_state.get(key)
                if value is not None and check_func(value):
                    count += 1
        
        return count
    
    @staticmethod
    def safe_get(key: str, default: Any = None) -> Any:
        """
        Safely get a session state value with fallback.
        
        Args:
            key (str): The session state key.
            default (Any): Default value if key doesn't exist.
            
        Returns:
            Any: The value from session state or default.
        """
        if key not in st.session_state:
            st.session_state[key] = default
        return st.session_state[key]
    
    @staticmethod
    def safe_set(key: str, value: Any) -> None:
        """
        Safely set a session state value.
        
        Args:
            key (str): The session state key.
            value (Any): The value to set.
        """
        st.session_state[key] = value
    
    @staticmethod
    def reset_quick_filters():
        """Reset quick filter states"""
        st.session_state.quick_filter = None
        st.session_state.quick_filter_applied = False
        
        if 'filter_state' in st.session_state:
            st.session_state.filter_state['quick_filter'] = None
            st.session_state.filter_state['quick_filter_applied'] = False
        
# ============================================
# MARKET STATE FILTER VALIDATION
# ============================================

def validate_market_state_filters():
    """
    Quick validation test for all market state filter presets.
    Ensures configuration is properly set up and filters work correctly.
    """
    try:
        logger.info("="*60)
        logger.info("VALIDATING MARKET STATE FILTERS")
        logger.info("="*60)
        
        # Check configuration exists
        if not hasattr(CONFIG, 'MARKET_STATE_FILTERS'):
            logger.error("MARKET_STATE_FILTERS not found in CONFIG")
            return False
        
        # Check default filter
        default_filter = getattr(CONFIG, 'DEFAULT_MARKET_FILTER', None)
        if not default_filter:
            logger.error("DEFAULT_MARKET_FILTER not set in CONFIG")
            return False
        
        # Test each filter preset
        filters = CONFIG.MARKET_STATE_FILTERS
        all_states = ['STRONG_UPTREND', 'UPTREND', 'PULLBACK', 'ROTATION', 
                     'SIDEWAYS', 'DOWNTREND', 'STRONG_DOWNTREND', 'BOUNCE']
        
        for filter_name, filter_config in filters.items():
            logger.info(f"Testing {filter_name} filter...")
            
            # Check allowed_states exists and is valid
            allowed_states = filter_config.get('allowed_states', [])
            if not allowed_states:
                logger.error(f"  ✗ {filter_name}: No allowed_states defined")
                return False
            
            # Check all states are valid
            invalid_states = [s for s in allowed_states if s not in all_states]
            if invalid_states:
                logger.error(f"  ✗ {filter_name}: Invalid states: {invalid_states}")
                return False
            
            # Check description exists
            description = filter_config.get('description', '')
            if not description:
                logger.warning(f"  ⚠ {filter_name}: No description provided")
            
            # Test strategy focus
            state_count = len(allowed_states)
            if filter_name == 'MOMENTUM' and state_count < 2:
                logger.warning(f"  ⚠ {filter_name}: May be too restrictive ({state_count} states)")
            elif filter_name == 'ALL' and state_count != len(all_states):
                logger.warning(f"  ⚠ {filter_name}: Should include all states ({state_count}/{len(all_states)})")
            
            logger.info(f"  ✓ {filter_name}: {state_count} allowed states - {', '.join(allowed_states)}")
        
        # Test default filter exists
        if default_filter not in filters:
            logger.error(f"DEFAULT_MARKET_FILTER '{default_filter}' not found in MARKET_STATE_FILTERS")
            return False
        
        logger.info(f"✓ Default filter: {default_filter}")
        logger.info(f"✓ All {len(filters)} filter presets validated successfully")
        logger.info("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Market state filter validation failed: {e}")
        return False

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main Streamlit application - Final Perfected Production Version"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize robust session state
    SessionStateManager.initialize()
    
    # Custom CSS for production UI
    st.markdown("""
    <style>
    /* Production-ready CSS */
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: rgba(28, 131, 225, 0.1);
        border: 1px solid rgba(28, 131, 225, 0.2);
        padding: 5% 5% 5% 10%;
        border-radius: 5px;
        overflow-wrap: break-word;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 5px;
    }
    /* Button styling */
    div.stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    /* Mobile responsive */
    @media (max-width: 768px) {
        .stDataFrame {font-size: 12px;}
        div[data-testid="metric-container"] {padding: 3%;}
        .main {padding: 0rem 0.5rem;}
    }
    /* Table optimization */
    .stDataFrame > div {overflow-x: auto;}
    /* Loading animation */
    .stSpinner > div {
        border-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div style="
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="margin: 0; font-size: 2.5rem;">🌊 Wave Detection Ultimate 3.0</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
            Professional Stock Ranking System
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### 🎯 Quick Actions")
        
        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Data", type="primary", width="stretch"):
                st.cache_data.clear()
                st.session_state.last_refresh = datetime.now(timezone.utc)
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", width="stretch"):
                st.cache_data.clear()
                gc.collect()  # Force garbage collection
                st.success("Cache cleared!")
                time.sleep(0.5)
                st.rerun()
        
        # Data source selection
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        data_source_col1, data_source_col2 = st.columns(2)
        
        with data_source_col1:
            if st.button("📊 Google Sheets", 
                        type="primary" if st.session_state.data_source == "sheet" else "secondary", 
                        width="stretch"):
                st.session_state.data_source = "sheet"
                st.rerun()
        
        with data_source_col2:
            if st.button("📁 Upload CSV", 
                        type="primary" if st.session_state.data_source == "upload" else "secondary", 
                        width="stretch"):
                st.session_state.data_source = "upload"
                st.rerun()

        uploaded_file = None
        sheet_id = None
        gid = None
        
        if st.session_state.data_source == "upload":
            uploaded_file = st.file_uploader(
                "Choose CSV file", 
                type="csv",
                help="Upload a CSV file with stock data. Must contain 'ticker' and 'price' columns."
            )
            if uploaded_file is None:
                st.info("Please upload a CSV file to continue")
        else:
            # Google Sheets input
            st.markdown("#### 📊 Google Sheets Configuration")
            
            sheet_input = st.text_input(
                "Google Sheets ID or URL",
                value=st.session_state.get('sheet_id', ''),
                placeholder="Enter Sheet ID or full URL",
                help="Example: 1OEQ_qxL4lzlO9LlKnDGlDku2yQC1iYvOYeXF0mTQlJM or the full Google Sheets URL"
            )
            
            if sheet_input:
                sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_input)
                if sheet_id_match:
                    sheet_id = sheet_id_match.group(1)
                else:
                    sheet_id = sheet_input.strip()
            
                st.session_state.sheet_id = sheet_id
            
            gid_input = st.text_input(
                "Sheet Tab GID (Optional)",
                value=st.session_state.get('gid', CONFIG.DEFAULT_GID),
                placeholder=f"Default: {CONFIG.DEFAULT_GID}",
                help="The GID identifies specific sheet tab. Found in URL after #gid="
            )
            
            if gid_input:
                gid = gid_input.strip()
            else:
                gid = CONFIG.DEFAULT_GID
            
            if not sheet_id:
                st.warning("Please enter a Google Sheets ID to continue")
        
        # Data quality indicator
        data_quality = st.session_state.get('data_quality', {})
        if data_quality:
            with st.expander("📊 Data Quality", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    completeness = data_quality.get('completeness', 0)
                    if completeness > 80:
                        emoji = "🟢"
                    elif completeness > 60:
                        emoji = "🟡"
                    else:
                        emoji = "🔴"
                    
                    st.metric("Completeness", f"{emoji} {completeness:.1f}%")
                    st.metric("Total Stocks", f"{data_quality.get('total_rows', 0):,}")
                
                with col2:
                    if 'timestamp' in data_quality:
                        age = datetime.now(timezone.utc) - data_quality['timestamp']
                        hours = age.total_seconds() / 3600
                        
                        if hours < 1:
                            freshness = "🟢 Fresh"
                        elif hours < 24:
                            freshness = "🟡 Recent"
                        else:
                            freshness = "🔴 Stale"
                        
                        st.metric("Data Age", freshness)
                    
                    duplicates = data_quality.get('duplicate_tickers', 0)
                    if duplicates > 0:
                        st.metric("Duplicates", f"⚠️ {duplicates}")
        
        # Performance metrics
        perf_metrics = st.session_state.get('performance_metrics', {})
        if perf_metrics:
            with st.expander("⚡ Performance"):
                total_time = sum(perf_metrics.values())
                if total_time < 3:
                    perf_emoji = "🟢"
                elif total_time < 5:
                    perf_emoji = "🟡"
                else:
                    perf_emoji = "🔴"
                
                st.metric("Load Time", f"{perf_emoji} {total_time:.1f}s")
                
                # Show slowest operations
                if len(perf_metrics) > 0:
                    slowest = sorted(perf_metrics.items(), key=lambda x: x[1], reverse=True)[:3]
                    for func_name, elapsed in slowest:
                        if elapsed > 0.001:
                            st.caption(f"{func_name}: {elapsed:.4f}s")
        
        st.markdown("---")
        st.markdown("### 🔍 Smart Filters")
        
        active_filter_count = 0
        
        if st.session_state.get('quick_filter_applied', False):
            active_filter_count += 1
        
        filter_checks = [
            ('category_filter', lambda x: x and len(x) > 0),
            ('sector_filter', lambda x: x and len(x) > 0),
            ('industry_filter', lambda x: x and len(x) > 0),
            ('min_score', lambda x: x > 0),
            ('patterns', lambda x: x and len(x) > 0),
            ('trend_filter', lambda x: x != 'All Trends'),
            ('eps_tier_filter', lambda x: x and len(x) > 0),
            ('pe_tier_filter', lambda x: x and len(x) > 0),
            ('price_tier_filter', lambda x: x and len(x) > 0),
            ('min_eps_change', lambda x: x is not None and str(x).strip() != ''),
            ('min_pe', lambda x: x is not None and str(x).strip() != ''),
            ('max_pe', lambda x: x is not None and str(x).strip() != ''),
            ('require_fundamental_data', lambda x: x),
            ('market_states_filter', lambda x: x and len(x) > 0),
            ('market_strength_range_slider', lambda x: x != (0, 100)),
            ('long_term_strength_range_slider', lambda x: x != (0, 100))
        ]
        
        for key, check_func in filter_checks:
            value = st.session_state.get(key)
            if value is not None and check_func(value):
                active_filter_count += 1
        
        st.session_state.active_filter_count = active_filter_count
        
        if active_filter_count > 0:
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        
        if st.button("🗑️ Clear All Filters", 
                    width="stretch", 
                    type="primary" if active_filter_count > 0 else "secondary"):
            SessionStateManager.clear_filters()
            st.success("✅ All filters cleared!")
            st.rerun()
        
        st.markdown("---")
        show_debug = st.checkbox("🐛 Show Debug Info", 
                               value=st.session_state.get('show_debug', False),
                               key="show_debug")
    
    try:
        if st.session_state.data_source == "upload" and uploaded_file is None:
            st.warning("Please upload a CSV file to continue")
            st.stop()
        
        if st.session_state.data_source == "sheet" and not sheet_id:
            st.warning("Please enter a Google Sheets ID to continue")
            st.stop()
        
        with st.spinner("📥 Loading and processing data..."):
            try:
                if st.session_state.data_source == "upload" and uploaded_file is not None:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "upload", file_data=uploaded_file
                    )
                else:
                    ranked_df, data_timestamp, metadata = load_and_process_data(
                        "sheet", 
                        sheet_id=sheet_id,
                        gid=gid
                    )
                
                st.session_state.ranked_df = ranked_df
                st.session_state.data_timestamp = data_timestamp
                st.session_state.last_refresh = datetime.now(timezone.utc)
                
                if metadata.get('warnings'):
                    for warning in metadata['warnings']:
                        st.warning(warning)
                
                if metadata.get('errors'):
                    for error in metadata['errors']:
                        st.error(error)
                
            except Exception as e:
                logger.error(f"Failed to load data: {str(e)}")
                
                last_good_data = st.session_state.get('last_good_data')
                if last_good_data:
                    ranked_df, data_timestamp, metadata = last_good_data
                    st.warning("Failed to load fresh data, using cached version")
                else:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Common issues:\n- Invalid Google Sheets ID\n- Sheet not publicly accessible\n- Network connectivity\n- Invalid CSV format")
                    st.stop()
        
    except Exception as e:
        st.error(f"❌ Critical Error: {str(e)}")
        with st.expander("🔍 Error Details"):
            st.code(str(e))
        st.stop()
    
    # Quick Action Buttons
    st.markdown("### ⚡ Quick Actions")
    qa_col1, qa_col2, qa_col3, qa_col4, qa_col5 = st.columns(5)
    
    quick_filter_applied = st.session_state.get('quick_filter_applied', False)
    quick_filter = st.session_state.get('quick_filter', None)
    
    with qa_col1:
        if st.button("📈 Top Gainers", width="stretch"):
            st.session_state['quick_filter'] = 'top_gainers'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col2:
        if st.button("🔥 Volume Surges", width="stretch"):
            st.session_state['quick_filter'] = 'volume_surges'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col3:
        if st.button("🚀 High Velocity", width="stretch"):
            st.session_state['quick_filter'] = 'velocity_breakout'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col4:
        if st.button("🌋 Tsunami", width="stretch"):
            st.session_state['quick_filter'] = 'institutional_tsunami'
            st.session_state['quick_filter_applied'] = True
            st.rerun()
    
    with qa_col5:
        if st.button("🌊 Show All", width="stretch"):
            st.session_state['quick_filter'] = None
            st.session_state['quick_filter_applied'] = False
            st.rerun()
    
    # Get ranked_df from session state
    ranked_df = st.session_state.get('ranked_df')
    if ranked_df is None:
        st.error("No data available. Please check your data source configuration.")
        st.stop()
    
    if quick_filter:
        if quick_filter == 'top_gainers':
            ranked_df_display = ranked_df[ranked_df['momentum_score'] >= 80]
            st.info(f"Showing {len(ranked_df_display)} stocks with momentum score ≥ 80")
        elif quick_filter == 'volume_surges':
            ranked_df_display = ranked_df[ranked_df['rvol'] >= 3]
            st.info(f"Showing {len(ranked_df_display)} stocks with RVOL ≥ 3x")
        elif quick_filter == 'velocity_breakout':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('VELOCITY BREAKOUT', na=False)]
            st.info(f"Showing {len(ranked_df_display)} stocks with Velocity Breakout pattern")
        elif quick_filter == 'institutional_tsunami':
            ranked_df_display = ranked_df[ranked_df['patterns'].str.contains('🌋 INSTITUTIONAL TSUNAMI', na=False)]
            st.info(f"Showing {len(ranked_df_display)} stocks with Institutional Tsunami pattern")
    else:
        ranked_df_display = ranked_df
    
    # Sidebar filters
    with st.sidebar:
        # Initialize centralized filter state
        FilterEngine.initialize_filters()
        
        # Initialize filters dict for current frame
        filters = {}
        
        # Display Mode
        st.markdown("### 📊 Display Mode")
        
        # Ensure user_preferences exists and safe access to display_mode with fallback
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {'display_mode': 'Hybrid (Technical + Fundamentals)'}
        current_display_mode = st.session_state.user_preferences.get('display_mode', 'Hybrid (Technical + Fundamentals)')
        
        display_mode = st.radio(
            "Choose your view:",
            options=["Technical", "Hybrid (Technical + Fundamentals)"],
            index=0 if current_display_mode == 'Technical' else 1,
            help="Technical: Pure momentum analysis | Hybrid: Adds PE & EPS data",
            key="display_mode_toggle"
        )
        
        st.session_state.user_preferences['display_mode'] = display_mode
        show_fundamentals = (display_mode == "Hybrid (Technical + Fundamentals)")
        
        # DEBUG: Show available columns when in Hybrid mode
        if show_fundamentals and 'ranked_df' in st.session_state:
            available_fund_cols = [col for col in ['pe', 'eps_current', 'eps_change_pct'] 
                                 if col in st.session_state.ranked_df.columns]
            if available_fund_cols:
                st.sidebar.success(f"✅ Fundamental data available: {', '.join(available_fund_cols)}")
            else:
                st.sidebar.warning("⚠️ No fundamental data found in current dataset")
        
        st.markdown("---")
        
        # CRITICAL: Define callback functions BEFORE widgets
        def sync_categories():
            if 'category_multiselect' in st.session_state:
                st.session_state.filter_state['categories'] = st.session_state.category_multiselect
        
        def sync_sectors():
            if 'sector_multiselect' in st.session_state:
                st.session_state.filter_state['sectors'] = st.session_state.sector_multiselect
        
        def sync_industries():
            if 'industry_multiselect' in st.session_state:
                st.session_state.filter_state['industries'] = st.session_state.industry_multiselect
        
        def sync_min_score():
            if 'min_score_slider' in st.session_state:
                st.session_state.filter_state['min_score'] = st.session_state.min_score_slider
        
        def sync_position_tier():
            if 'position_tier_multiselect_intelligence' in st.session_state:
                st.session_state.filter_state['position_tiers'] = st.session_state.position_tier_multiselect_intelligence
        
        def sync_position_range():
            if 'position_range_slider' in st.session_state:
                st.session_state.filter_state['position_range'] = st.session_state.position_range_slider
        
        def sync_performance_tier():
            if 'performance_tier_multiselect_intelligence' in st.session_state:
                st.session_state.filter_state['performance_tiers'] = st.session_state.performance_tier_multiselect_intelligence
        
        def sync_performance_custom_range():
            # Sync individual range sliders for all timeframes
            if 'ret_1d_range_slider' in st.session_state:
                st.session_state.filter_state['ret_1d_range'] = st.session_state.ret_1d_range_slider
            if 'ret_3d_range_slider' in st.session_state:
                st.session_state.filter_state['ret_3d_range'] = st.session_state.ret_3d_range_slider
            if 'ret_7d_range_slider' in st.session_state:
                st.session_state.filter_state['ret_7d_range'] = st.session_state.ret_7d_range_slider
            if 'ret_30d_range_slider' in st.session_state:
                st.session_state.filter_state['ret_30d_range'] = st.session_state.ret_30d_range_slider
            if 'ret_3m_range_slider' in st.session_state:
                st.session_state.filter_state['ret_3m_range'] = st.session_state.ret_3m_range_slider
            if 'ret_6m_range_slider' in st.session_state:
                st.session_state.filter_state['ret_6m_range'] = st.session_state.ret_6m_range_slider
            if 'ret_1y_range_slider' in st.session_state:
                st.session_state.filter_state['ret_1y_range'] = st.session_state.ret_1y_range_slider
            if 'ret_3y_range_slider' in st.session_state:
                st.session_state.filter_state['ret_3y_range'] = st.session_state.ret_3y_range_slider
            if 'ret_5y_range_slider' in st.session_state:
                st.session_state.filter_state['ret_5y_range'] = st.session_state.ret_5y_range_slider
            # Legacy support
            if 'performance_custom_range_slider' in st.session_state:
                st.session_state.filter_state['performance_custom_range'] = st.session_state.performance_custom_range_slider
        
        def sync_volume_tier():
            if 'volume_tier_multiselect_intelligence' in st.session_state:
                st.session_state.filter_state['volume_tiers'] = st.session_state.volume_tier_multiselect_intelligence
        
        def sync_vmi_tier():
            if 'vmi_tier_multiselect_intelligence' in st.session_state:
                st.session_state.filter_state['vmi_tiers'] = st.session_state.vmi_tier_multiselect_intelligence
        
        def sync_momentum_harmony_tier():
            if 'momentum_harmony_tier_multiselect_intelligence' in st.session_state:
                st.session_state.filter_state['momentum_harmony_tiers'] = st.session_state.momentum_harmony_tier_multiselect_intelligence
        
        def sync_rvol_range():
            if 'rvol_range_slider' in st.session_state:
                st.session_state.filter_state['rvol_range'] = st.session_state.rvol_range_slider
        
        # Performance Filter Sync Functions
        def sync_performance_dropdowns():
            # Sync all performance dropdown selections
            for ret_col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']:
                dropdown_key = f'{ret_col}_dropdown'
                if dropdown_key in st.session_state:
                    st.session_state.filter_state[f'{ret_col}_selection'] = st.session_state[dropdown_key]
        
        def sync_performance_sliders():
            # Sync all performance range sliders
            for ret_col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y']:
                slider_key = f'{ret_col}_range_slider'
                if slider_key in st.session_state:
                    st.session_state.filter_state[f'{ret_col}_range'] = st.session_state[slider_key]
        
        def sync_patterns():
            if 'patterns_multiselect' in st.session_state:
                st.session_state.filter_state['patterns'] = st.session_state.patterns_multiselect

        def sync_market_states():
            if 'market_states_multiselect' in st.session_state:
                st.session_state.filter_state['market_states'] = st.session_state.market_states_multiselect
        
        # Intelligence Score Dropdown and Slider Sync Functions
        def sync_position_score_dropdown():
            if 'position_score_dropdown' in st.session_state:
                st.session_state.filter_state['position_score_selection'] = st.session_state.position_score_dropdown
        
        def sync_position_score_slider():
            if 'position_score_slider' in st.session_state:
                st.session_state.filter_state['position_score_range'] = st.session_state.position_score_slider
        
        def sync_volume_score_dropdown():
            if 'volume_score_dropdown' in st.session_state:
                st.session_state.filter_state['volume_score_selection'] = st.session_state.volume_score_dropdown
        
        def sync_volume_score_slider():
            if 'volume_score_slider' in st.session_state:
                st.session_state.filter_state['volume_score_range'] = st.session_state.volume_score_slider
        
        def sync_market_strength():
            if 'market_strength_slider' in st.session_state:
                st.session_state.filter_state['market_strength_range'] = st.session_state.market_strength_slider
        
        def sync_long_term_strength():
            if 'long_term_strength_slider' in st.session_state:
                st.session_state.filter_state['long_term_strength_range'] = st.session_state.long_term_strength_slider
        
        def sync_momentum_score_dropdown():
            if 'momentum_score_dropdown' in st.session_state:
                st.session_state.filter_state['momentum_score_selection'] = st.session_state.momentum_score_dropdown
        
        def sync_momentum_score_slider():
            if 'momentum_score_slider' in st.session_state:
                st.session_state.filter_state['momentum_score_range'] = st.session_state.momentum_score_slider
        
        def sync_acceleration_score_dropdown():
            if 'acceleration_score_dropdown' in st.session_state:
                st.session_state.filter_state['acceleration_score_selection'] = st.session_state.acceleration_score_dropdown
        
        def sync_acceleration_score_slider():
            if 'acceleration_score_slider' in st.session_state:
                st.session_state.filter_state['acceleration_score_range'] = st.session_state.acceleration_score_slider
        
        def sync_breakout_score_dropdown():
            if 'breakout_score_dropdown' in st.session_state:
                st.session_state.filter_state['breakout_score_selection'] = st.session_state.breakout_score_dropdown
        
        def sync_breakout_score_slider():
            if 'breakout_score_slider' in st.session_state:
                st.session_state.filter_state['breakout_score_range'] = st.session_state.breakout_score_slider
        
        def sync_rvol_score_dropdown():
            if 'rvol_score_dropdown' in st.session_state:
                st.session_state.filter_state['rvol_score_selection'] = st.session_state.rvol_score_dropdown
        
        def sync_rvol_score_slider():
            if 'rvol_score_slider' in st.session_state:
                st.session_state.filter_state['rvol_score_range'] = st.session_state.rvol_score_slider
        
        # BIDIRECTIONAL SMART INTERCONNECTED FILTERS: Category ↔ Sector ↔ Industry
        # Any selection affects the other two filters bidirectionally
        st.markdown("#### 🏢 Company Classification")
        
        # Category filter with callback
        categories = FilterEngine.get_filter_options(ranked_df_display, 'category', filters)
        
        # Clean default values to only include available options (INTERCONNECTION FIX)
        stored_categories = st.session_state.filter_state.get('categories', [])
        valid_category_defaults = [cat for cat in stored_categories if cat in categories]
        
        selected_categories = st.multiselect(
            f"Market Cap Category ({len(categories)} available)",
            options=categories,
            default=valid_category_defaults,
            placeholder="Select categories (empty = All)",
            help="📊 Filter by market cap category. Updates based on selected sectors and industries.",
            key="category_multiselect",
            on_change=sync_categories  # SYNC ON CHANGE
        )
        
        if selected_categories:
            filters['categories'] = selected_categories
        
        # Sector filter with callback
        sectors = FilterEngine.get_filter_options(ranked_df_display, 'sector', filters)
        
        # Clean default values to only include available options (INTERCONNECTION FIX)
        stored_sectors = st.session_state.filter_state.get('sectors', [])
        valid_sector_defaults = [sec for sec in stored_sectors if sec in sectors]
        
        selected_sectors = st.multiselect(
            f"Sector ({len(sectors)} available)",
            options=sectors,
            default=valid_sector_defaults,
            placeholder="Select sectors (empty = All)",
            help="🏭 Filter by business sector. Updates based on selected categories and industries.",
            key="sector_multiselect",
            on_change=sync_sectors  # SYNC ON CHANGE
        )
        
        if selected_sectors:
            filters['sectors'] = selected_sectors
        
        # Industry filter with callback
        industries = FilterEngine.get_filter_options(ranked_df_display, 'industry', filters)
        
        # Clean default values to only include available options (INTERCONNECTION FIX)
        stored_industries = st.session_state.filter_state.get('industries', [])
        valid_industry_defaults = [ind for ind in stored_industries if ind in industries]
        
        selected_industries = st.multiselect(
            f"Industry ({len(industries)} available)",
            options=industries,
            default=valid_industry_defaults,
            placeholder="Select industries (empty = All)",
            help="🏢 Filter by specific industry. Updates categories and sectors to show only relevant options.",
            key="industry_multiselect",
            on_change=sync_industries  # SYNC ON CHANGE
        )
        
        if selected_industries:
            filters['industries'] = selected_industries

        st.markdown("#### ✨ Pattern Detector")
        
        # Pattern filter with callback
        all_patterns = set()
        for patterns in ranked_df_display['patterns'].dropna():
            if patterns:
                all_patterns.update(patterns.split(' | '))
        
        if all_patterns:
            # Clean default values to only include available options (INTERCONNECTION FIX)
            stored_patterns = st.session_state.filter_state.get('patterns', [])
            valid_pattern_defaults = [pat for pat in stored_patterns if pat in sorted(all_patterns)]
            
            selected_patterns = st.multiselect(
                "Patterns",
                options=sorted(all_patterns),
                default=valid_pattern_defaults,
                placeholder="Select patterns (empty = All)",
                help="Filter by specific patterns",
                key="patterns_multiselect",
                on_change=sync_patterns  # SYNC ON CHANGE
            )
            
            if selected_patterns:
                filters['patterns'] = selected_patterns
        
        # Trend filter with callback and custom range
        st.markdown("#### 📈 Trend Strength")
        trend_options = {
            "All Trends": (0, 100),
            "🔥 Exceptional (85+)": (85, 100),
            "🚀 Strong (70-84)": (70, 84),
            "✅ Good (55-69)": (55, 69),
            "➡️ Neutral (40-54)": (40, 54),
            "⚠️ Weak (25-39)": (25, 39),
            "🔻 Poor (<25)": (0, 24),
            "🎯 Custom Range": None  # Special option for custom range
        }
        
        current_trend = st.session_state.filter_state.get('trend_filter', "All Trends")
        if current_trend not in trend_options:
            current_trend = "All Trends"
        
        # Custom sync function for trend with custom range support
        def sync_trend_with_custom():
            if 'trend_selectbox' in st.session_state:
                selected = st.session_state.trend_selectbox
                st.session_state.filter_state['trend_filter'] = selected
                
                if selected == "🎯 Custom Range":
                    # Don't set range here, will be set by slider
                    pass
                else:
                    st.session_state.filter_state['trend_range'] = trend_options[selected]
        
        def sync_trend_custom_slider():
            if 'trend_custom_range_slider' in st.session_state:
                st.session_state.filter_state['trend_custom_range'] = st.session_state.trend_custom_range_slider
        
        selected_trend = st.selectbox(
            "Trend Quality",
            options=list(trend_options.keys()),
            index=list(trend_options.keys()).index(current_trend),
            help="Filter stocks by trend strength based on SMA alignment. Choose Custom Range for precise control.",
            key="trend_selectbox",
            on_change=sync_trend_with_custom
        )
        
        # Show custom range slider when Custom Range is selected
        if selected_trend == "🎯 Custom Range":
            # Get current custom range from session state, default to (0, 100)
            current_custom_range = st.session_state.filter_state.get('trend_custom_range', (0, 100))
            
            custom_range = st.slider(
                "Custom Trend Quality Range",
                min_value=0,
                max_value=100,
                value=current_custom_range,
                step=1,
                help="🔥 85+: Exceptional | 🚀 70-84: Strong | ✅ 55-69: Good | ➡️ 40-54: Neutral | ⚠️ 25-39: Weak | 🔻 <25: Poor",
                key="trend_custom_range_slider",
                on_change=sync_trend_custom_slider
            )
            
            # Set the filter range to custom slider value
            filters['trend_filter'] = selected_trend
            filters['trend_range'] = custom_range
            # Also update session state for consistency
            st.session_state.filter_state['trend_range'] = custom_range
            
            # Show indicator for selected range
            def get_trend_indicator_for_range(min_val, max_val):
                """Get appropriate indicator for the selected range"""
                if min_val >= 85:
                    return "🔥"
                elif min_val >= 70:
                    return "🚀"
                elif min_val >= 55:
                    return "✅"
                elif min_val >= 40:
                    return "➡️"
                elif min_val >= 25:
                    return "⚠️"
                else:
                    return "🔻"
            
            range_indicator = get_trend_indicator_for_range(custom_range[0], custom_range[1])
            st.caption(f"{range_indicator} **Selected Range**: {custom_range[0]}-{custom_range[1]} | Filtering stocks with trend quality in this range")
            
        elif selected_trend != "All Trends":
            filters['trend_filter'] = selected_trend
            filters['trend_range'] = trend_options[selected_trend]
        
        # Market State filters with callbacks
        st.markdown("#### 📈 Market State Filters")
        
        # Add filter presets and custom selection option (no individual states in main multiselect)
        preset_options = ["🎯 MOMENTUM (Default)", "⚡ AGGRESSIVE", "💎 VALUE", "🛡️ DEFENSIVE", "🌍 ALL"]
        market_state_with_presets = preset_options + ["📊 Custom Selection"]
        
        selected_market_states = st.multiselect(
            "Market State",
            options=market_state_with_presets,
            default=st.session_state.filter_state.get('market_states', []),
            placeholder="Select market states or use preset strategy",
            help="Filter by market momentum state. Use presets for different trading strategies or select Custom Selection for individual states",
            key="market_states_multiselect",
            on_change=sync_market_states  # SYNC ON CHANGE
        )
        
        # Show custom market states dropdown when Custom Selection is active
        custom_selection_active = "📊 Custom Selection" in selected_market_states
        custom_states_selection = []
        
        if custom_selection_active:
            # Define the 8 specific market states for custom selection
            custom_market_states = [
                "BOUNCE",
                "DOWNTREND", 
                "PULLBACK",
                "ROTATION",
                "SIDEWAYS",
                "STRONG_DOWNTREND",
                "STRONG_UPTREND",
                "UPTREND"
            ]
            
            # Add sync function for custom states
            def sync_custom_market_states():
                if 'custom_market_states_multiselect' in st.session_state:
                    st.session_state.filter_state['custom_market_states'] = st.session_state.custom_market_states_multiselect
            
            custom_states_selection = st.multiselect(
                "Select Individual Market States",
                options=custom_market_states,
                default=st.session_state.filter_state.get('custom_market_states', []),
                placeholder="Choose specific market states to filter",
                help="Select one or more market states to include in your filter",
                key="custom_market_states_multiselect",
                on_change=sync_custom_market_states
            )
            
            # Long Term Strength Slider - Professional Implementation
            st.markdown("**🏆 Long Term Strength Filter**")
            long_term_strength_range = st.slider(
                "Long Term Strength Range",
                min_value=0,
                max_value=100,
                value=st.session_state.filter_state.get('long_term_strength_range', (0, 100)),
                step=5,
                help="Filter by long term strength score (0-100). Higher values indicate stronger long-term trend consistency, momentum harmony, and sustained performance characteristics.",
                key="long_term_strength_slider",
                on_change=sync_long_term_strength
            )
            
            # Market Strength Slider - Professional Implementation
            st.markdown("**📊 Market Strength Filter**")
            market_strength_range = st.slider(
                "Market Strength Range",
                min_value=0,
                max_value=100,
                value=st.session_state.filter_state.get('market_strength_range', (0, 100)),
                step=5,
                help="Filter by overall market strength score (0-100). Higher values indicate stronger market conditions with better momentum, acceleration, and volume characteristics.",
                key="market_strength_slider",
                on_change=sync_market_strength
            )
            
            # Combine selections for filtering
            if custom_states_selection:
                # Use custom states selection when available
                filters['market_states'] = ["📊 Custom Selection"] + custom_states_selection
            else:
                # Just custom selection flag without individual states
                filters['market_states'] = selected_market_states
        else:
            # Regular preset selection
            if selected_market_states:
                filters['market_states'] = selected_market_states
        
        # Add Market Strength filter when Custom Selection is active
        if custom_selection_active:
            if st.session_state.filter_state.get('market_strength_range', (0, 100)) != (0, 100):
                filters['market_strength_range'] = st.session_state.filter_state['market_strength_range']
            if st.session_state.filter_state.get('long_term_strength_range', (0, 100)) != (0, 100):
                filters['long_term_strength_range'] = st.session_state.filter_state['long_term_strength_range']
        
        # 🎯 Score Component - Professional Expandable Section
        with st.expander("🎯 Score Component", expanded=False):
            
            # Minimum Master Score
            min_score = st.slider(
                "Minimum Master Score",
                min_value=0,
                max_value=100,
                value=st.session_state.filter_state.get('min_score', 0),
                step=5,
                help="Filter stocks by minimum master score (0-100). Higher scores indicate better overall ranking.",
                key="min_score_slider",
                on_change=sync_min_score
            )
            
            if min_score > 0:
                filters['min_score'] = min_score
            
            # Position Score Dropdown with Custom Range
            if 'position_score' in ranked_df_display.columns:
                position_score_options = [
                    "All Scores",
                    "🟢 Strong (>= 80)",
                    "🟡 Good (>= 60)",
                    "🟠 Fair (>= 40)",
                    "🔴 Weak (< 40)",
                    "🎯 Custom Range"
                ]
                
                current_position_selection = st.session_state.filter_state.get('position_score_selection', "All Scores")
                if current_position_selection not in position_score_options:
                    current_position_selection = "All Scores"
                
                position_score_selection = st.selectbox(
                    "Position Score",
                    options=position_score_options,
                    index=position_score_options.index(current_position_selection),
                    help="Filter stocks by position score strength",
                    key="position_score_dropdown",
                    on_change=sync_position_score_dropdown
                )
                
                # Show custom range slider when "🎯 Custom Range" is selected
                if position_score_selection == "🎯 Custom Range":
                    position_score_range = st.slider(
                        "Position Score Custom Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('position_score_range', (0, 100)),
                        step=5,
                        help="Filter stocks by position score custom range (0-100)",
                        key="position_score_slider",
                        on_change=sync_position_score_slider
                    )
                    
                    if position_score_range != (0, 100):
                        filters['position_score_range'] = position_score_range
                elif position_score_selection != "All Scores":
                    # Map selection to range
                    if position_score_selection == "🟢 Strong (>= 80)":
                        filters['position_score_range'] = (80, 100)
                    elif position_score_selection == "🟡 Good (>= 60)":
                        filters['position_score_range'] = (60, 100)
                    elif position_score_selection == "🟠 Fair (>= 40)":
                        filters['position_score_range'] = (40, 100)
                    elif position_score_selection == "🔴 Weak (< 40)":
                        filters['position_score_range'] = (0, 39)
            
            # Volume Score Dropdown with Custom Range
            if 'volume_score' in ranked_df_display.columns:
                volume_score_options = [
                    "All Scores",
                    "🟢 Strong (>= 80)",
                    "🟡 Good (>= 60)",
                    "🟠 Fair (>= 40)",
                    "🔴 Weak (< 40)",
                    "🎯 Custom Range"
                ]
                
                current_volume_selection = st.session_state.filter_state.get('volume_score_selection', "All Scores")
                if current_volume_selection not in volume_score_options:
                    current_volume_selection = "All Scores"
                
                volume_score_selection = st.selectbox(
                    "Volume Score",
                    options=volume_score_options,
                    index=volume_score_options.index(current_volume_selection),
                    help="Filter stocks by volume score strength",
                    key="volume_score_dropdown",
                    on_change=sync_volume_score_dropdown
                )
                
                # Show custom range slider when "🎯 Custom Range" is selected
                if volume_score_selection == "🎯 Custom Range":
                    volume_score_range = st.slider(
                        "Volume Score Custom Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('volume_score_range', (0, 100)),
                        step=5,
                        help="Filter stocks by volume score custom range (0-100)",
                        key="volume_score_slider",
                        on_change=sync_volume_score_slider
                    )
                    
                    if volume_score_range != (0, 100):
                        filters['volume_score_range'] = volume_score_range
                elif volume_score_selection != "All Scores":
                    # Map selection to range
                    if volume_score_selection == "🟢 Strong (>= 80)":
                        filters['volume_score_range'] = (80, 100)
                    elif volume_score_selection == "🟡 Good (>= 60)":
                        filters['volume_score_range'] = (60, 100)
                    elif volume_score_selection == "🟠 Fair (>= 40)":
                        filters['volume_score_range'] = (40, 100)
                    elif volume_score_selection == "🔴 Weak (< 40)":
                        filters['volume_score_range'] = (0, 39)
            
            # Momentum Score Dropdown with Custom Range
            if 'momentum_score' in ranked_df_display.columns:
                momentum_score_options = [
                    "All Scores",
                    "🟢 Strong (>= 80)",
                    "🟡 Good (>= 60)",
                    "🟠 Fair (>= 40)",
                    "🔴 Weak (< 40)",
                    "🎯 Custom Range"
                ]
                
                current_momentum_selection = st.session_state.filter_state.get('momentum_score_selection', "All Scores")
                if current_momentum_selection not in momentum_score_options:
                    current_momentum_selection = "All Scores"
                
                momentum_score_selection = st.selectbox(
                    "Momentum Score",
                    options=momentum_score_options,
                    index=momentum_score_options.index(current_momentum_selection),
                    help="Filter stocks by momentum score strength",
                    key="momentum_score_dropdown",
                    on_change=sync_momentum_score_dropdown
                )
                
                # Show custom range slider when "🎯 Custom Range" is selected
                if momentum_score_selection == "🎯 Custom Range":
                    momentum_score_range = st.slider(
                        "Momentum Score Custom Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('momentum_score_range', (0, 100)),
                        step=5,
                        help="Filter stocks by momentum score custom range (0-100)",
                        key="momentum_score_slider",
                        on_change=sync_momentum_score_slider
                    )
                    
                    if momentum_score_range != (0, 100):
                        filters['momentum_score_range'] = momentum_score_range
                elif momentum_score_selection != "All Scores":
                    # Map selection to range
                    if momentum_score_selection == "🟢 Strong (>= 80)":
                        filters['momentum_score_range'] = (80, 100)
                    elif momentum_score_selection == "🟡 Good (>= 60)":
                        filters['momentum_score_range'] = (60, 100)
                    elif momentum_score_selection == "🟠 Fair (>= 40)":
                        filters['momentum_score_range'] = (40, 100)
                    elif momentum_score_selection == "🔴 Weak (< 40)":
                        filters['momentum_score_range'] = (0, 39)
            
            # Acceleration Score Dropdown with Custom Range
            if 'acceleration_score' in ranked_df_display.columns:
                acceleration_score_options = [
                    "All Scores",
                    "🟢 Strong (>= 80)",
                    "🟡 Good (>= 60)",
                    "🟠 Fair (>= 40)",
                    "🔴 Weak (< 40)",
                    "🎯 Custom Range"
                ]
                
                current_acceleration_selection = st.session_state.filter_state.get('acceleration_score_selection', "All Scores")
                if current_acceleration_selection not in acceleration_score_options:
                    current_acceleration_selection = "All Scores"
                
                acceleration_score_selection = st.selectbox(
                    "Acceleration Score",
                    options=acceleration_score_options,
                    index=acceleration_score_options.index(current_acceleration_selection),
                    help="Filter stocks by acceleration score strength",
                    key="acceleration_score_dropdown",
                    on_change=sync_acceleration_score_dropdown
                )
                
                # Show custom range slider when "🎯 Custom Range" is selected
                if acceleration_score_selection == "🎯 Custom Range":
                    acceleration_score_range = st.slider(
                        "Acceleration Score Custom Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('acceleration_score_range', (0, 100)),
                        step=5,
                        help="Filter stocks by acceleration score custom range (0-100)",
                        key="acceleration_score_slider",
                        on_change=sync_acceleration_score_slider
                    )
                    
                    if acceleration_score_range != (0, 100):
                        filters['acceleration_score_range'] = acceleration_score_range
                elif acceleration_score_selection != "All Scores":
                    # Map selection to range
                    if acceleration_score_selection == "🟢 Strong (>= 80)":
                        filters['acceleration_score_range'] = (80, 100)
                    elif acceleration_score_selection == "🟡 Good (>= 60)":
                        filters['acceleration_score_range'] = (60, 100)
                    elif acceleration_score_selection == "🟠 Fair (>= 40)":
                        filters['acceleration_score_range'] = (40, 100)
                    elif acceleration_score_selection == "🔴 Weak (< 40)":
                        filters['acceleration_score_range'] = (0, 39)
            
            # Breakout Score Dropdown with Custom Range
            if 'breakout_score' in ranked_df_display.columns:
                breakout_score_options = [
                    "All Scores",
                    "🟢 Strong (>= 80)",
                    "🟡 Good (>= 60)",
                    "🟠 Fair (>= 40)",
                    "🔴 Weak (< 40)",
                    "🎯 Custom Range"
                ]
                
                current_breakout_selection = st.session_state.filter_state.get('breakout_score_selection', "All Scores")
                if current_breakout_selection not in breakout_score_options:
                    current_breakout_selection = "All Scores"
                
                breakout_score_selection = st.selectbox(
                    "Breakout Score",
                    options=breakout_score_options,
                    index=breakout_score_options.index(current_breakout_selection),
                    help="Filter stocks by breakout score strength",
                    key="breakout_score_dropdown",
                    on_change=sync_breakout_score_dropdown
                )
                
                # Show custom range slider when "🎯 Custom Range" is selected
                if breakout_score_selection == "🎯 Custom Range":
                    breakout_score_range = st.slider(
                        "Breakout Score Custom Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('breakout_score_range', (0, 100)),
                        step=5,
                        help="Filter stocks by breakout score custom range (0-100)",
                        key="breakout_score_slider",
                        on_change=sync_breakout_score_slider
                    )
                    
                    if breakout_score_range != (0, 100):
                        filters['breakout_score_range'] = breakout_score_range
                elif breakout_score_selection != "All Scores":
                    # Map selection to range
                    if breakout_score_selection == "🟢 Strong (>= 80)":
                        filters['breakout_score_range'] = (80, 100)
                    elif breakout_score_selection == "🟡 Good (>= 60)":
                        filters['breakout_score_range'] = (60, 100)
                    elif breakout_score_selection == "🟠 Fair (>= 40)":
                        filters['breakout_score_range'] = (40, 100)
                    elif breakout_score_selection == "🔴 Weak (< 40)":
                        filters['breakout_score_range'] = (0, 39)
            
            # RVOL Score Dropdown with Custom Range
            if 'rvol_score' in ranked_df_display.columns:
                rvol_score_options = [
                    "All Scores",
                    "🟢 Strong (>= 80)",
                    "🟡 Good (>= 60)",
                    "🟠 Fair (>= 40)",
                    "🔴 Weak (< 40)",
                    "🎯 Custom Range"
                ]
                
                current_rvol_selection = st.session_state.filter_state.get('rvol_score_selection', "All Scores")
                if current_rvol_selection not in rvol_score_options:
                    current_rvol_selection = "All Scores"
                
                rvol_score_selection = st.selectbox(
                    "RVOL Score",
                    options=rvol_score_options,
                    index=rvol_score_options.index(current_rvol_selection),
                    help="Filter stocks by RVOL score strength",
                    key="rvol_score_dropdown",
                    on_change=sync_rvol_score_dropdown
                )
                
                # Show custom range slider when "🎯 Custom Range" is selected
                if rvol_score_selection == "🎯 Custom Range":
                    rvol_score_range = st.slider(
                        "RVOL Score Custom Range",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('rvol_score_range', (0, 100)),
                        step=5,
                        help="Filter stocks by RVOL score custom range (0-100)",
                        key="rvol_score_slider",
                        on_change=sync_rvol_score_slider
                    )
                    
                    if rvol_score_range != (0, 100):
                        filters['rvol_score_range'] = rvol_score_range
                elif rvol_score_selection != "All Scores":
                    # Map selection to range
                    if rvol_score_selection == "🟢 Strong (>= 80)":
                        filters['rvol_score_range'] = (80, 100)
                    elif rvol_score_selection == "🟡 Good (>= 60)":
                        filters['rvol_score_range'] = (60, 100)
                    elif rvol_score_selection == "🟠 Fair (>= 40)":
                        filters['rvol_score_range'] = (40, 100)
                    elif rvol_score_selection == "🔴 Weak (< 40)":
                        filters['rvol_score_range'] = (0, 39)
        
        # 📈 Performance Filter - Professional Expandable Section
        with st.expander("📈 Performance Filter", expanded=False):
            
            # Check for available return columns
            available_return_cols = [col for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y', 'ret_5y'] if col in ranked_df_display.columns]
            
            if available_return_cols:
                # Individual Return Period Filters
                st.markdown("**📊 Individual Return Period Filters**")
                
                # Define performance configuration with your exact specifications
                performance_config = {
                    'ret_1d': {
                        'name': '1 Day Return',
                        'presets': [
                            {'label': '💥 Explosive', 'min': 10, 'max': None, 'description': '>10% 1D'},
                            {'label': '🚀 Strong Rise', 'min': 5, 'max': 10, 'description': '5-10% 1D'},
                            {'label': '📈 Positive', 'min': 2, 'max': 5, 'description': '2-5% 1D'},
                            {'label': '➡️ Flat', 'min': -2, 'max': 2, 'description': '-2% to 2% 1D'},
                            {'label': '📉 Negative', 'min': -5, 'max': -2, 'description': '-5% to -2% 1D'},
                            {'label': '💣 Crash', 'min': None, 'max': -5, 'description': '<-5% 1D'}
                        ],
                        'slider_range': (-50, 50),
                        'slider_step': 0.5,
                        'default_min': -50,
                        'default_max': 50
                    },
                    'ret_3d': {
                        'name': '3 Day Return',
                        'presets': [
                            {'label': '🌟 3-Day Surge', 'min': 15, 'max': None, 'description': '>15% 3D'},
                            {'label': '⚡ Strong Momentum', 'min': 8, 'max': 15, 'description': '8-15% 3D'},
                            {'label': '📊 Steady Rise', 'min': 3, 'max': 8, 'description': '3-8% 3D'},
                            {'label': '⏸️ Consolidating', 'min': -3, 'max': 3, 'description': '-3% to 3% 3D'},
                            {'label': '⚠️ Weakening', 'min': -8, 'max': -3, 'description': '-8% to -3% 3D'},
                            {'label': '🔻 Sharp Decline', 'min': None, 'max': -8, 'description': '<-8% 3D'}
                        ],
                        'slider_range': (-75, 75),
                        'slider_step': 1,
                        'default_min': -75,
                        'default_max': 75
                    },
                    'ret_7d': {
                        'name': 'Weekly Return',
                        'presets': [
                            {'label': '📈 Weekly Winners', 'min': 20, 'max': None, 'description': '>20% 7D'},
                            {'label': '💪 Strong Week', 'min': 12, 'max': 20, 'description': '12-20% 7D'},
                            {'label': '✅ Good Week', 'min': 5, 'max': 12, 'description': '5-12% 7D'},
                            {'label': '😐 Flat Week', 'min': -5, 'max': 5, 'description': '-5% to 5% 7D'},
                            {'label': '😟 Bad Week', 'min': -12, 'max': -5, 'description': '-12% to -5% 7D'},
                            {'label': '💔 Terrible Week', 'min': None, 'max': -12, 'description': '<-12% 7D'}
                        ],
                        'slider_range': (-100, 100),
                        'slider_step': 1,
                        'default_min': -100,
                        'default_max': 100
                    },
                    'ret_30d': {
                        'name': 'Monthly Return',
                        'presets': [
                            {'label': '🏆 Monthly Champions', 'min': 30, 'max': None, 'description': '>30% 30D'},
                            {'label': '🎯 Top Performers', 'min': 20, 'max': 30, 'description': '20-30% 30D'},
                            {'label': '📈 Outperformers', 'min': 10, 'max': 20, 'description': '10-20% 30D'},
                            {'label': '🔄 Market Performers', 'min': -10, 'max': 10, 'description': '-10% to 10% 30D'},
                            {'label': '📉 Underperformers', 'min': -20, 'max': -10, 'description': '-20% to -10% 30D'},
                            {'label': '☠️ Monthly Losers', 'min': None, 'max': -20, 'description': '<-20% 30D'}
                        ],
                        'slider_range': (-100, 200),
                        'slider_step': 2,
                        'default_min': -100,
                        'default_max': 200
                    },
                    'ret_3m': {
                        'name': '3 Month Return',
                        'presets': [
                            {'label': '🎯 Quarterly Stars', 'min': 50, 'max': None, 'description': '>50% 3M'},
                            {'label': '⭐ Strong Quarter', 'min': 30, 'max': 50, 'description': '30-50% 3M'},
                            {'label': '👍 Good Quarter', 'min': 15, 'max': 30, 'description': '15-30% 3M'},
                            {'label': '➖ Flat Quarter', 'min': -15, 'max': 15, 'description': '-15% to 15% 3M'},
                            {'label': '👎 Weak Quarter', 'min': -30, 'max': -15, 'description': '-30% to -15% 3M'},
                            {'label': '🚨 Quarterly Disaster', 'min': None, 'max': -30, 'description': '<-30% 3M'}
                        ],
                        'slider_range': (-100, 300),
                        'slider_step': 5,
                        'default_min': -100,
                        'default_max': 300
                    },
                    'ret_6m': {
                        'name': '6 Month Return',
                        'presets': [
                            {'label': '💎 Half-Year Heroes', 'min': 80, 'max': None, 'description': '>80% 6M'},
                            {'label': '🌟 Semi-Annual Stars', 'min': 60, 'max': 80, 'description': '60-80% 6M'},
                            {'label': '📈 Strong Half', 'min': 30, 'max': 60, 'description': '30-60% 6M'},
                            {'label': '〰️ Sideways', 'min': -20, 'max': 20, 'description': '-20% to 20% 6M'},
                            {'label': '📉 Weak Half', 'min': -50, 'max': -20, 'description': '-50% to -20% 6M'},
                            {'label': '💀 Half-Year Collapse', 'min': None, 'max': -50, 'description': '<-50% 6M'}
                        ],
                        'slider_range': (-100, 500),
                        'slider_step': 10,
                        'default_min': -100,
                        'default_max': 500
                    },
                    'ret_1y': {
                        'name': '1 Year Return',
                        'presets': [
                            {'label': '🌙 Annual Winners', 'min': 100, 'max': None, 'description': '>100% 1Y'},
                            {'label': '🏅 Year Stars', 'min': 80, 'max': 100, 'description': '80-100% 1Y'},
                            {'label': '💪 Strong Year', 'min': 50, 'max': 80, 'description': '50-80% 1Y'},
                            {'label': '📊 Market Year', 'min': -30, 'max': 30, 'description': '-30% to 30% 1Y'},
                            {'label': '😔 Disappointing Year', 'min': -60, 'max': -30, 'description': '-60% to -30% 1Y'},
                            {'label': '🔴 Annual Disasters', 'min': None, 'max': -60, 'description': '<-60% 1Y'}
                        ],
                        'slider_range': (-100, 1000),
                        'slider_step': 10,
                        'default_min': -100,
                        'default_max': 1000
                    },
                    'ret_3y': {
                        'name': '3 Year Return',
                        'presets': [
                            {'label': '👑 Multi-Year Champions', 'min': 200, 'max': None, 'description': '>200% 3Y'},
                            {'label': '🚀 3Y Rockets', 'min': 150, 'max': 200, 'description': '150-200% 3Y'},
                            {'label': '⭐ 3Y Stars', 'min': 100, 'max': 150, 'description': '100-150% 3Y'},
                            {'label': '📈 3Y Growth', 'min': 0, 'max': 50, 'description': '0-50% 3Y'},
                            {'label': '📉 3Y Decline', 'min': -50, 'max': 0, 'description': '-50% to 0% 3Y'},
                            {'label': '💣 3Y Destruction', 'min': None, 'max': -50, 'description': '<-50% 3Y'}
                        ],
                        'slider_range': (-100, 2000),
                        'slider_step': 25,
                        'default_min': -100,
                        'default_max': 2000
                    },
                    'ret_5y': {
                        'name': '5 Year Return',
                        'presets': [
                            {'label': '🏛️ Long-Term Legends', 'min': 300, 'max': None, 'description': '>300% 5Y'},
                            {'label': '💎 5Y Diamonds', 'min': 250, 'max': 300, 'description': '250-300% 5Y'},
                            {'label': '🌟 5Y Winners', 'min': 150, 'max': 250, 'description': '150-250% 5Y'},
                            {'label': '📊 5Y Average', 'min': 0, 'max': 100, 'description': '0-100% 5Y'},
                            {'label': '⚠️ 5Y Laggards', 'min': -75, 'max': 0, 'description': '-75% to 0% 5Y'},
                            {'label': '☠️ 5Y Wipeout', 'min': None, 'max': -75, 'description': '<-75% 5Y'}
                        ],
                        'slider_range': (-100, 5000),
                        'slider_step': 50,
                        'default_min': -100,
                        'default_max': 5000
                    }
                }
                
                # Create single column layout for sequential order
                # Process each available return column in proper time sequence
                for ret_col in available_return_cols:
                    if ret_col in performance_config:
                        config = performance_config[ret_col]
                        
                        # Create preset options
                        preset_options = ["All Returns"] + [preset['label'] for preset in config['presets']] + ["🎯 Custom Range"]
                        
                        # Get current selection
                        current_selection = st.session_state.filter_state.get(f'{ret_col}_selection', "All Returns")
                        if current_selection not in preset_options:
                            current_selection = "All Returns"
                        
                        # Performance period dropdown
                        selection = st.selectbox(
                            config['name'],
                            options=preset_options,
                            index=preset_options.index(current_selection),
                            help=f"Filter stocks by {config['name'].lower()}",
                            key=f"{ret_col}_dropdown",
                            on_change=sync_performance_dropdowns
                        )
                        
                        # Update session state with current selection
                        st.session_state.filter_state[f'{ret_col}_selection'] = selection
                        
                        # Handle custom range selection
                        if selection == "🎯 Custom Range":
                            # Ensure proper type conversion for slider value parameter
                            default_value = (float(config['default_min']), float(config['default_max']))
                            current_value = st.session_state.filter_state.get(f'{ret_col}_range', default_value)
                            
                            # Ensure current_value is also a tuple of floats
                            if isinstance(current_value, tuple) and len(current_value) == 2:
                                current_value = (float(current_value[0]), float(current_value[1]))
                            else:
                                current_value = default_value
                            
                            range_value = st.slider(
                                f"{config['name']} Range (%)",
                                min_value=float(config['slider_range'][0]),
                                max_value=float(config['slider_range'][1]),
                                value=current_value,
                                step=float(config['slider_step']),
                                help=f"Custom range for {config['name'].lower()}",
                                key=f"{ret_col}_range_slider",
                                on_change=sync_performance_sliders
                            )
                            filters[f'{ret_col}_range'] = range_value
                        elif selection != "All Returns":
                            # Handle preset selection
                            for preset in config['presets']:
                                if preset['label'] == selection:
                                    min_val = preset['min'] if preset['min'] is not None else config['default_min']
                                    max_val = preset['max'] if preset['max'] is not None else config['default_max']
                                    filters[f'{ret_col}_range'] = (min_val, max_val)
                                    break
            else:
                st.info("📊 No return data available for performance filtering")
        
        # 🧠 Intelligence Filter - Combined Section
        with st.expander("🧠 Intelligence Filter", expanded=False):
            # VMI (Volume Momentum Index) Filter
            if 'vmi_tier' in ranked_df_display.columns or 'vmi' in ranked_df_display.columns:
                vmi_tier_options = list(CONFIG.TIERS['vmi_tiers'].keys()) + ["🎯 Custom VMI Range"]
                vmi_tiers = st.multiselect(
                    "VMI (Volume Momentum Index) Tiers",
                    options=vmi_tier_options,
                    default=st.session_state.filter_state.get('vmi_tiers', []),
                    key='vmi_tier_multiselect_intelligence',
                    on_change=sync_vmi_tier,
                    help="Volume Momentum Index: Weighted volume trend score classification"
                )
                
                if vmi_tiers:
                    filters['vmi_tiers'] = vmi_tiers
                
                # Show custom VMI range slider when "🎯 Custom VMI Range" is selected
                custom_vmi_range_selected = any("Custom VMI Range" in tier for tier in vmi_tiers) if vmi_tiers else False
                if custom_vmi_range_selected:
                    st.write("📊 **Custom VMI Range Filter**")
                    
                    vmi_range = st.slider(
                        "VMI Range",
                        min_value=0.0,
                        max_value=5.0,
                        value=(0.5, 3.0),
                        step=0.1,
                        key='custom_vmi_range_intelligence',
                        help="VMI typically ranges from 0.1 (very low volume) to 3.0+ (very high volume)"
                    )
                    filters['custom_vmi_range'] = vmi_range
            
            # Momentum Harmony Filter  
            if 'momentum_harmony_tier' in ranked_df_display.columns or 'momentum_harmony' in ranked_df_display.columns:
                momentum_harmony_tier_options = list(CONFIG.TIERS['momentum_harmony_tiers'].keys())
                momentum_harmony_tiers = st.multiselect(
                    "Momentum Harmony Tiers",
                    options=momentum_harmony_tier_options,
                    default=st.session_state.filter_state.get('momentum_harmony_tiers', []),
                    key='momentum_harmony_tier_multiselect_intelligence',
                    on_change=sync_momentum_harmony_tier,
                    help="Multi-timeframe momentum alignment: 0-4 score showing consistency across periods"
                )
                
                if momentum_harmony_tiers:
                    filters['momentum_harmony_tiers'] = momentum_harmony_tiers
            
            #  Volume Intelligence
            if 'volume_tier' in ranked_df_display.columns or 'rvol' in ranked_df_display.columns:
                # Volume tier multiselect with custom range option
                volume_tier_options = list(CONFIG.TIERS['volume_tiers'].keys()) + ["🎯 Custom RVOL Range"]
                volume_tiers = st.multiselect(
                    "Volume Activity Tiers",
                    options=volume_tier_options,
                    default=st.session_state.filter_state.get('volume_tiers', []),
                    key='volume_tier_multiselect_intelligence',
                    on_change=sync_volume_tier,
                    help="Select volume activity tiers or use Custom RVOL Range for precise control"
                )
                
                if volume_tiers:
                    filters['volume_tiers'] = volume_tiers
                
                # Show custom RVOL range slider when "🎯 Custom RVOL Range" is selected
                custom_rvol_range_selected = any("Custom RVOL Range" in tier for tier in volume_tiers) if volume_tiers else False
                if custom_rvol_range_selected:
                    st.write("📊 **Custom RVOL Range Filter**")
                    
                    rvol_range = st.slider(
                        "RVOL Range",
                        min_value=0.1,
                        max_value=20.0,
                        value=st.session_state.filter_state.get('rvol_range', (0.1, 20.0)),
                        step=0.1,
                        help="Filter by Relative Volume (RVOL) range",
                        key="rvol_range_slider",
                        on_change=sync_rvol_range
                    )
                    if rvol_range != (0.1, 20.0):
                        filters['rvol_range'] = rvol_range
            
            # 🎯 Position Intelligence
            if 'position_tier' in ranked_df_display.columns:
                # Position tier multiselect with custom range option
                position_tier_options = list(CONFIG.TIERS['position_tiers'].keys()) + ["🎯 Custom Position Range"]
                position_tiers = st.multiselect(
                    "Position Tiers",
                    options=position_tier_options,
                    default=st.session_state.filter_state.get('position_tiers', []),
                    key='position_tier_multiselect_intelligence',
                    on_change=sync_position_tier,
                    help="Select position tiers or use Custom Position Range for precise control"
                )
                
                if position_tiers:
                    filters['position_tiers'] = position_tiers
                
                # Show custom position range slider when "🎯 Custom Position Range" is selected
                custom_position_range_selected = any("Custom Position Range" in tier for tier in position_tiers) if position_tiers else False
                if custom_position_range_selected:
                    st.write("📊 **Custom Position Range Filter**")
                    
                    position_range = st.slider(
                        "Position Range (%)",
                        min_value=0,
                        max_value=100,
                        value=st.session_state.filter_state.get('position_range', (0, 100)),
                        step=1,
                        help="Filter by position percentage range (distance from 52-week low)",
                        key="position_range_slider",
                        on_change=sync_position_range
                    )
                    if position_range != (0, 100):
                        filters['position_range'] = position_range
        
        # Advanced filters with callbacks
        with st.expander("🔧 Advanced Filters"):
            # Define callbacks for advanced filters
            def sync_eps_tier():
                if 'eps_tier_multiselect' in st.session_state:
                    st.session_state.filter_state['eps_tiers'] = st.session_state.eps_tier_multiselect
            
            def sync_pe_tier():
                if 'pe_tier_multiselect' in st.session_state:
                    st.session_state.filter_state['pe_tiers'] = st.session_state.pe_tier_multiselect
            
            def sync_price_tier():
                if 'price_tier_multiselect' in st.session_state:
                    st.session_state.filter_state['price_tiers'] = st.session_state.price_tier_multiselect
            
            def sync_eps_change_tier():
                if 'eps_change_tiers_widget' in st.session_state:
                    st.session_state.filter_state['eps_change_tiers'] = st.session_state.eps_change_tiers_widget
            
            def sync_min_pe():
                if 'min_pe_input' in st.session_state:
                    value = st.session_state.min_pe_input
                    if value.strip():
                        try:
                            st.session_state.filter_state['min_pe'] = float(value)
                        except ValueError:
                            st.session_state.filter_state['min_pe'] = None
                    else:
                        st.session_state.filter_state['min_pe'] = None
            
            def sync_max_pe():
                if 'max_pe_input' in st.session_state:
                    value = st.session_state.max_pe_input
                    if value.strip():
                        try:
                            st.session_state.filter_state['max_pe'] = float(value)
                        except ValueError:
                            st.session_state.filter_state['max_pe'] = None
                    else:
                        st.session_state.filter_state['max_pe'] = None
            
            def sync_fundamental():
                if 'require_fundamental_checkbox' in st.session_state:
                    st.session_state.filter_state['require_fundamental_data'] = st.session_state.require_fundamental_checkbox
            
            # Tier filters
            for tier_type, col_name, filter_key, sync_func in [
                ('eps_tiers', 'eps_tier', 'eps_tiers', sync_eps_tier),
                ('pe_tiers', 'pe_tier', 'pe_tiers', sync_pe_tier),
                ('price_tiers', 'price_tier', 'price_tiers', sync_price_tier)
            ]:
                if col_name in ranked_df_display.columns:
                    tier_options = FilterEngine.get_filter_options(ranked_df_display, col_name, filters)
                    
                    selected_tiers = st.multiselect(
                        f"{col_name.replace('_', ' ').title()}",
                        options=tier_options,
                        default=st.session_state.filter_state.get(filter_key, []),
                        placeholder=f"Select {col_name.replace('_', ' ')}s (empty = All)",
                        key=f"{col_name}_multiselect",
                        on_change=sync_func  # SYNC ON CHANGE
                    )
                    
                    if selected_tiers:
                        filters[tier_type] = selected_tiers
            
            # EPS change tier filter
            if 'eps_change_tier' in ranked_df_display.columns:
                # EPS Change Tier Filter
                eps_change_tiers = st.multiselect(
                    "EPS Change Tier",
                    options=list(CONFIG.TIERS['eps_change_pct'].keys()),
                    default=st.session_state.filter_state.get('eps_change_tiers', []),
                    key='eps_change_tiers_widget',
                    on_change=sync_eps_change_tier,
                    help="Select EPS change tiers to include"
                )
                
                if eps_change_tiers:
                    filters['eps_change_tiers'] = eps_change_tiers
            
            # PE filters (only in hybrid mode)
            if show_fundamentals and 'pe' in ranked_df_display.columns:
                st.markdown("**🔍 Fundamental Filters**")
                
                col1, col2 = st.columns(2)
                with col1:
                    current_min_pe = st.session_state.filter_state.get('min_pe')
                    min_pe_str = str(current_min_pe) if current_min_pe is not None else ""
                    
                    min_pe_input = st.text_input(
                        "Min PE Ratio",
                        value=min_pe_str,
                        placeholder="e.g. 10",
                        key="min_pe_input",
                        on_change=sync_min_pe  # SYNC ON CHANGE
                    )
                    
                    if min_pe_input.strip():
                        try:
                            min_pe_val = float(min_pe_input)
                            filters['min_pe'] = min_pe_val
                        except ValueError:
                            st.error("Invalid Min PE")
                
                with col2:
                    current_max_pe = st.session_state.filter_state.get('max_pe')
                    max_pe_str = str(current_max_pe) if current_max_pe is not None else ""
                    
                    max_pe_input = st.text_input(
                        "Max PE Ratio",
                        value=max_pe_str,
                        placeholder="e.g. 30",
                        key="max_pe_input",
                        on_change=sync_max_pe  # SYNC ON CHANGE
                    )
                    
                    if max_pe_input.strip():
                        try:
                            max_pe_val = float(max_pe_input)
                            filters['max_pe'] = max_pe_val
                        except ValueError:
                            st.error("Invalid Max PE")
                
                # Data completeness filter
                require_fundamental = st.checkbox(
                    "Only show stocks with PE and EPS data",
                    value=st.session_state.filter_state.get('require_fundamental_data', False),
                    key="require_fundamental_checkbox",
                    on_change=sync_fundamental  # SYNC ON CHANGE
                )
                
                if require_fundamental:
                    filters['require_fundamental_data'] = True
        
        # Count active filters using FilterEngine method
        active_filter_count = FilterEngine.get_active_count()
        st.session_state.active_filter_count = active_filter_count
        
        if active_filter_count > 0:
            st.info(f"🔍 **{active_filter_count} filter{'s' if active_filter_count > 1 else ''} active**")
        
        # Clear filters button - ENHANCED VERSION
        if st.button("🗑️ Clear All Filters", 
                    width="stretch", 
                    type="primary" if active_filter_count > 0 else "secondary",
                    key="clear_filters_sidebar_btn"):
            
            # Use both FilterEngine and SessionStateManager clear methods
            FilterEngine.clear_all_filters()
            SessionStateManager.clear_filters()
            
            st.success("✅ All filters cleared!")
            time.sleep(0.3)
            st.rerun()
    
    # Apply filters (outside sidebar)
    if quick_filter_applied:
        filtered_df = FilterEngine.apply_filters(ranked_df_display, filters)
    else:
        filtered_df = FilterEngine.apply_filters(ranked_df, filters)
    
    filtered_df = filtered_df.sort_values('rank')
    
    # Save current filters
    st.session_state.user_preferences['last_filters'] = filters
    
    # Debug info (OPTIONAL)
    if show_debug:
        with st.sidebar.expander("🐛 Debug Info", expanded=True):
            st.write("**Active Filters:**")
            for key, value in filters.items():
                if value is not None and value != [] and value != 0 and \
                   (not (isinstance(value, tuple) and value == (0,100))):
                    st.write(f"• {key}: {value}")
            
            st.write(f"\n**Filter State:**")
            st.write(st.session_state.filter_state)
            
            st.write(f"\n**Filter Result:**")
            st.write(f"Before: {len(ranked_df)} stocks")
            st.write(f"After: {len(filtered_df)} stocks")
            
            if st.session_state.performance_metrics:
                st.write(f"\n**Performance:**")
                for func, time_taken in st.session_state.performance_metrics.items():
                    if time_taken > 0.001:
                        st.write(f"• {func}: {time_taken:.4f}s")
    
    active_filter_count = st.session_state.get('active_filter_count', 0)
    if quick_filter:
        filter_status_col1, filter_status_col2 = st.columns([5, 1])
        with filter_status_col1:
            quick_filter_names = {
                'top_gainers': '📈 Top Gainers',
                'volume_surges': '🔥 Volume Surges',
                'velocity_breakout': '🚀 High Velocity',
                'institutional_tsunami': '🌋 Tsunami'
            }
            filter_display = quick_filter_names.get(quick_filter, 'Filtered')
            
            if active_filter_count > 1:
                st.info(f"**Viewing:** {filter_display} + {active_filter_count - 1} other filter{'s' if active_filter_count > 2 else ''} | **{len(filtered_df):,} stocks** shown")
            else:
                st.info(f"**Viewing:** {filter_display} | **{len(filtered_df):,} stocks** shown")
        
        with filter_status_col2:
            if st.button("Clear Filters", type="secondary", key="clear_filters_main_btn"):
                FilterEngine.clear_all_filters()
                SessionStateManager.clear_filters()
                st.rerun()
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        total_stocks = len(filtered_df)
        total_original = len(ranked_df)
        pct_of_all = (total_stocks/total_original*100) if total_original > 0 else 0
        
        UIComponents.render_metric_card(
            "Total Stocks",
            f"{total_stocks:,}",
            f"{pct_of_all:.0f}% of {total_original:,}"
        )
    
    with col2:
        if not filtered_df.empty and 'master_score' in filtered_df.columns:
            avg_score = filtered_df['master_score'].mean()
            std_score = filtered_df['master_score'].std()
            UIComponents.render_metric_card(
                "Avg Score",
                f"{avg_score:.1f}",
                f"σ={std_score:.1f}"
            )
        else:
            UIComponents.render_metric_card("Avg Score", "N/A")
    
    with col3:
        if show_fundamentals and 'pe' in filtered_df.columns:
            valid_pe = filtered_df['pe'].notna() & (filtered_df['pe'] > 0) & (filtered_df['pe'] < 10000)
            pe_coverage = valid_pe.sum()
            pe_pct = (pe_coverage / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
            
            if pe_coverage > 0:
                median_pe = filtered_df.loc[valid_pe, 'pe'].median()
                UIComponents.render_metric_card(
                    "Median PE",
                    f"{median_pe:.1f}x",
                    f"{pe_pct:.0f}% have data"
                )
            else:
                UIComponents.render_metric_card("PE Data", "Limited", "No PE data")
        else:
            if not filtered_df.empty and 'master_score' in filtered_df.columns:
                min_score = filtered_df['master_score'].min()
                max_score = filtered_df['master_score'].max()
                score_range = f"{min_score:.1f}-{max_score:.1f}"
            else:
                score_range = "N/A"
            UIComponents.render_metric_card("Score Range", score_range)
    
    with col4:
        if show_fundamentals and 'eps_change_pct' in filtered_df.columns:
            valid_eps_change = filtered_df['eps_change_pct'].notna()
            positive_eps_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 0)
            strong_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 50)
            mega_growth = valid_eps_change & (filtered_df['eps_change_pct'] > 100)
            
            growth_count = positive_eps_growth.sum()
            strong_count = strong_growth.sum()
            
            if mega_growth.sum() > 0:
                UIComponents.render_metric_card(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{strong_count} >50% | {mega_growth.sum()} >100%"
                )
            else:
                UIComponents.render_metric_card(
                    "EPS Growth +ve",
                    f"{growth_count}",
                    f"{valid_eps_change.sum()} have data"
                )
        else:
            if 'acceleration_score' in filtered_df.columns:
                accelerating = (filtered_df['acceleration_score'] >= 80).sum()
            else:
                accelerating = 0
            UIComponents.render_metric_card("Accelerating", f"{accelerating}")
    
    with col5:
        if 'rvol' in filtered_df.columns:
            high_rvol = (filtered_df['rvol'] > 2).sum()
        else:
            high_rvol = 0
        UIComponents.render_metric_card("High RVOL", f"{high_rvol}")
    
    with col6:
        if 'trend_quality' in filtered_df.columns:
            strong_trends = (filtered_df['trend_quality'] >= 70).sum()
            total = len(filtered_df)
            UIComponents.render_metric_card(
                "Strong Trends", 
                f"{strong_trends}",
                f"{strong_trends/total*100:.0f}%" if total > 0 else "0%"
            )
        else:
            if 'patterns' in filtered_df.columns:
                with_patterns = (filtered_df['patterns'] != '').sum()
                UIComponents.render_metric_card("With Patterns", f"{with_patterns}")
            else:
                UIComponents.render_metric_card("With Patterns", "N/A")
    
    tabs = st.tabs([
        "📊 Summary", "🏆 Rankings", "📈 Market Radar", "📊 Analysis", "🔍 Search", "📥 Export", "ℹ️ About"
    ])
    
    with tabs[0]:
        st.markdown("### 📊 Executive Summary Dashboard")
        
        if not filtered_df.empty:
            UIComponents.render_summary_section(filtered_df)
            
            st.markdown("---")
            st.markdown("#### 💾 Download Clean Processed Data")
            
            download_cols = st.columns(3)
            
            with download_cols[0]:
                st.markdown("**📊 Current View Data**")
                st.write(f"Includes {len(filtered_df)} stocks matching current filters")
                
                csv_filtered = ExportEngine.create_csv_export(filtered_df)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv_filtered,
                    file_name=f"wave_detection_filtered_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download currently filtered stocks with all scores and indicators"
                )
            
            with download_cols[1]:
                st.markdown("**🏆 Top 100 Stocks**")
                st.write("Elite stocks ranked by Master Score")
                
                top_100 = filtered_df.nlargest(100, 'master_score')
                csv_top100 = ExportEngine.create_csv_export(top_100)
                st.download_button(
                    label="📥 Download Top 100 (CSV)",
                    data=csv_top100,
                    file_name=f"wave_detection_top100_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download top 100 stocks by Master Score"
                )
            
            with download_cols[2]:
                st.markdown("**🎯 Pattern Stocks Only**")
                if 'patterns' in filtered_df.columns:
                    pattern_stocks = filtered_df[filtered_df['patterns'] != '']
                    st.write(f"Includes {len(pattern_stocks)} stocks with patterns")
                    
                    if len(pattern_stocks) > 0:
                        csv_patterns = ExportEngine.create_csv_export(pattern_stocks)
                        st.download_button(
                            label="📥 Download Pattern Stocks (CSV)",
                            data=csv_patterns,
                            file_name=f"wave_detection_patterns_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            help="Download stocks with technical patterns"
                        )
                    else:
                        st.info("📊 No pattern stocks in current selection")
                else:
                    st.warning("⚠️ Pattern data not available in dataset")
        
        else:
            st.warning("No data available for summary. Please adjust filters.")
    
    # Tab 1: Rankings - ALL TIME BEST PROFESSIONAL RANKING TAB
    with tabs[1]:
        st.markdown("### 🏆 Top Ranked Stocks")
        st.markdown("*Complete analysis with all critical metrics for professional stock ranking*")
        
        # Enhanced Control Panel
        ranking_controls = st.columns([2, 2, 2, 2, 2])
        
        with ranking_controls[0]:
            display_count = st.selectbox(
                "🔢 Show Top",
                options=CONFIG.AVAILABLE_TOP_N,
                index=CONFIG.AVAILABLE_TOP_N.index(st.session_state.user_preferences['default_top_n']),
                key="display_count_select"
            )
            st.session_state.user_preferences['default_top_n'] = display_count
        
        with ranking_controls[1]:
            sort_options = ['Master Score', 'Momentum Score', 'Volume Score', 'Position Score', 
                          'Acceleration Score', 'Breakout Score', 'RVOL Score', 'Trend Quality',
                          'Long Term Strength', 'Liquidity Score', 'Overall Market Strength']
            
            sort_by = st.selectbox(
                "📊 Primary Sort",
                options=sort_options, 
                index=0,
                key="sort_by_select"
            )
        
        with ranking_controls[2]:
            view_mode = st.selectbox(
                "📋 View Mode",
                options=["Essential", "Technical", "Complete", "Custom"],
                index=0,
                key="ranking_view_mode",
                help="Essential: Core metrics | Technical: All scores | Complete: Everything"
            )
        
        with ranking_controls[3]:
            performance_timeframe = st.selectbox(
                "⏱️ Performance Period",
                options=["1D", "3D", "7D", "30D", "All"],
                index=4,
                key="perf_timeframe",
                help="Focus on specific performance timeframe"
            )
        
        with ranking_controls[4]:
            export_format = st.selectbox(
                "💾 Export Format",
                options=["None", "CSV", "Excel", "JSON"],
                index=0,
                key="export_format"
            )
        
        # 🔧 PERFORMANCE PERIOD FILTERING IMPLEMENTATION
        # Apply performance-based filtering before display
        performance_filtered_df = filtered_df.copy()
        
        if performance_timeframe != "All":
            # Map timeframe to return column
            timeframe_mapping = {
                "1D": "ret_1d",
                "3D": "ret_3d", 
                "7D": "ret_7d",
                "30D": "ret_30d"
            }
            
            primary_return_col = timeframe_mapping.get(performance_timeframe)
            
            if primary_return_col and primary_return_col in performance_filtered_df.columns:
                # Adjust master score based on selected timeframe performance
                # Give higher weight to the selected timeframe
                valid_returns = performance_filtered_df[primary_return_col].notna()
                
                if valid_returns.any():
                    # Create performance-adjusted score
                    timeframe_multiplier = pd.Series(1.0, index=performance_filtered_df.index)
                    
                    # Boost scores for strong performers in selected timeframe
                    strong_performers = (performance_filtered_df[primary_return_col] > 10) & valid_returns
                    moderate_performers = (performance_filtered_df[primary_return_col] > 5) & (performance_filtered_df[primary_return_col] <= 10) & valid_returns
                    weak_performers = (performance_filtered_df[primary_return_col] < -5) & valid_returns
                    
                    timeframe_multiplier[strong_performers] = 1.15  # 15% boost for strong performers
                    timeframe_multiplier[moderate_performers] = 1.05  # 5% boost for moderate performers  
                    timeframe_multiplier[weak_performers] = 0.95    # 5% penalty for weak performers
                    
                    # Apply timeframe adjustment to master score
                    performance_filtered_df['master_score_adjusted'] = (
                        performance_filtered_df['master_score'] * timeframe_multiplier
                    ).clip(0, 100)
                    
                    # Resort by adjusted score for this timeframe
                    performance_filtered_df = performance_filtered_df.sort_values('master_score_adjusted', ascending=False)
                    
                    logger.info(f"Applied {performance_timeframe} performance weighting: "
                              f"{strong_performers.sum()} strong, {moderate_performers.sum()} moderate, "
                              f"{weak_performers.sum()} weak performers")
        
        display_df = performance_filtered_df.head(display_count).copy()
        
        
        # Enhanced sorting logic
        sort_mapping = {
            'Master Score': 'master_score',
            'Momentum Score': 'momentum_score',
            'Volume Score': 'volume_score', 
            'Position Score': 'position_score',
            'Acceleration Score': 'acceleration_score',
            'Breakout Score': 'breakout_score',
            'RVOL Score': 'rvol_score',
            'Trend Quality': 'trend_quality',
            'Long Term Strength': 'long_term_strength',
            'Liquidity Score': 'liquidity_score',
            'Overall Market Strength': 'overall_market_strength'
        }
        
        if sort_by in sort_mapping and sort_mapping[sort_by] in display_df.columns:
            display_df = display_df.sort_values(sort_mapping[sort_by], ascending=False)
        else:
            display_df = display_df.sort_values('master_score', ascending=False)
        
        if not display_df.empty:
            
            # PROFESSIONAL COLUMN CONFIGURATION BASED ON VIEW MODE
            # Use adjusted score if performance timeframe filtering is applied
            score_column = 'master_score_adjusted' if (performance_timeframe != "All" and 'master_score_adjusted' in display_df.columns) else 'master_score'
            score_label = f'{performance_timeframe} Score' if performance_timeframe != "All" and 'master_score_adjusted' in display_df.columns else 'Score'
            
            if view_mode == "Essential":
                # Core metrics every trader needs
                display_cols = {
                    'rank': 'Rank',
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    score_column: score_label,
                    'market_state': 'State',
                    'price': 'Price',
                    'from_low_pct': 'Low%',
                    'from_high_pct': 'High%',
                    'ret_1d': '1D%',
                    'ret_7d': '7D%',
                    'ret_30d': '30D%',
                    'rvol': 'RVOL',
                    'vmi': 'VMI',
                    'volume_1d': 'Vol(Cr)',
                    'patterns': 'Patterns',
                    'category': 'Category'
                }
                
                # Add fundamentals if in Hybrid mode
                if show_fundamentals:
                    display_cols.update({
                        'pe': 'PE',
                        'eps_current': 'EPS',
                        'eps_change_pct': 'EPS Δ%'
                    })
                
            elif view_mode == "Technical":
                # All technical scores and indicators
                display_cols = {
                    'rank': 'Rank',
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    score_column: 'Master' if performance_timeframe == "All" else f'{performance_timeframe} Master',
                    'position_score': 'Position',
                    'momentum_score': 'Momentum',
                    'volume_score': 'Volume',
                    'acceleration_score': 'Accel',
                    'breakout_score': 'Breakout',
                    'rvol_score': 'RVOL Scr',
                    'trend_quality': 'Trend',
                    'long_term_strength': 'LT Str',
                    'liquidity_score': 'Liquid',
                    'overall_market_strength': 'Mkt Str',
                    'price': 'Price',
                    'vmi': 'VMI',
                    'patterns': 'Patterns'
                }
                
                # Add fundamentals if in Hybrid mode
                if show_fundamentals:
                    display_cols.update({
                        'pe': 'PE',
                        'eps_current': 'EPS',
                        'eps_change_pct': 'EPS Δ%'
                    })
                
            elif view_mode == "Complete":
                # Everything - ultimate professional view
                display_cols = {
                    'rank': 'Rank',
                    'ticker': 'Ticker', 
                    'company_name': 'Company',
                    score_column: 'Master' if performance_timeframe == "All" else f'{performance_timeframe} Master',
                    'position_score': 'Pos',
                    'momentum_score': 'Mom',
                    'volume_score': 'Vol',
                    'acceleration_score': 'Acc',
                    'breakout_score': 'Brk',
                    'rvol_score': 'RVS',
                    'trend_quality': 'Trd',
                    'long_term_strength': 'LTS',
                    'liquidity_score': 'Liq',
                    'overall_market_strength': 'MktS',
                    'market_state': 'State',
                    'price': 'Price',
                    'from_low_pct': 'Low%',
                    'from_high_pct': 'High%',
                    'ret_1d': '1D%',
                    'ret_3d': '3D%',
                    'ret_7d': '7D%',
                    'ret_30d': '30D%',
                    'rvol': 'RVOL',
                    'vmi': 'VMI',
                    'volume_1d': 'Vol(Cr)',
                    'money_flow_mm': 'MF(MM)',
                    'patterns': 'Patterns',
                    'category': 'Category',
                    'sector': 'Sector',
                    'industry': 'Industry'
                }
                
                # Add fundamentals if available - ONLY WHAT EXISTS IN V9.PY
                if show_fundamentals:
                    display_cols.update({
                        'pe': 'PE',
                        'eps_current': 'EPS',
                        'eps_change_pct': 'EPS Δ%'
                    })
                    
            else:  # Custom view
                # Let user select what they want to see
                st.markdown("#### 🛠️ Customize Your View")
                
                available_cols = [
                    'rank', 'ticker', 'company_name', 'master_score', 'position_score', 'momentum_score',
                    'volume_score', 'acceleration_score', 'breakout_score', 'rvol_score', 'trend_quality',
                    'long_term_strength', 'liquidity_score', 'overall_market_strength', 'market_state',
                    'price', 'from_low_pct', 'from_high_pct', 'ret_1d', 'ret_3d', 'ret_7d', 'ret_30d',
                    'rvol', 'vmi', 'volume_1d', 'money_flow_mm', 'patterns',
                    'category', 'sector', 'industry'
                ]
                
                # Add fundamentals if available - ONLY WHAT EXISTS IN V9.PY  
                if show_fundamentals:
                    available_cols.extend(['pe', 'eps_current', 'eps_change_pct'])
                
                # Filter to only available columns
                available_cols = [col for col in available_cols if col in display_df.columns]
                
                # Smart default selection - only columns that exist
                default_cols = ['rank', 'ticker', 'company_name', 'master_score', 'price']
                if 'ret_30d' in available_cols:
                    default_cols.append('ret_30d')
                if 'rvol' in available_cols:
                    default_cols.append('rvol')
                if 'patterns' in available_cols:
                    default_cols.append('patterns')
                
                custom_cols = st.multiselect(
                    "Select columns to display:",
                    options=available_cols,
                    default=default_cols,
                    key="custom_columns_select"
                )
                
                # Create display_cols dict for custom selection
                display_cols = {col: col.replace('_', ' ').title() for col in custom_cols}
        
            # ROBUST COLUMN FILTERING - Only include columns that actually exist
            available_display_cols = [c for c in display_cols.keys() if c in display_df.columns]
            final_display_cols = {k: display_cols[k] for k in available_display_cols}
            
            # Check if fundamental columns were requested but missing
            if show_fundamentals:
                requested_fund_cols = [c for c in ['pe', 'eps_current', 'eps_change_pct'] if c in display_cols.keys()]
                missing_fund_cols = [c for c in requested_fund_cols if c not in display_df.columns]
                
                if missing_fund_cols:
                    st.warning(f"⚠️ **Hybrid Mode**: Some fundamental data missing from dataset: {', '.join(missing_fund_cols)}")
                    st.info("💡 **Tip**: Upload data with PE, EPS columns or switch to Technical mode for full experience")
            
            # SAFETY CHECK: Ensure no duplicate column names
            column_names = list(final_display_cols.values())
            if len(column_names) != len(set(column_names)):
                st.error(f"🚨 **CRITICAL ERROR**: Duplicate column names detected: {column_names}")
                st.error("This will cause the application to crash. Please report this bug.")
                return
            
            # Create formatted dataframe for display - PERFORMANCE OPTIMIZED
            # Only copy columns that will be displayed to reduce memory usage
            display_cols_list = list(final_display_cols.keys())
            display_df_formatted = display_df[display_cols_list].copy()
            
            # PROFESSIONAL FORMATTING RULES
            format_rules = {
                'master_score': lambda x: f"{x:.1f}" if pd.notna(x) else '-',
                'master_score_adjusted': lambda x: f"{x:.1f}" if pd.notna(x) else '-',  # Add adjusted score formatting
                'position_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'momentum_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'volume_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'acceleration_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'breakout_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'rvol_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'trend_quality': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'long_term_strength': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'liquidity_score': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'overall_market_strength': lambda x: f"{x:.0f}" if pd.notna(x) else '-',
                'price': lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-',
                'from_low_pct': lambda x: f"{x:.0f}%" if pd.notna(x) else '-',
                'from_high_pct': lambda x: f"{x:+.1f}%" if pd.notna(x) else '-',
                'ret_1d': lambda x: f"{x:+.1f}%" if pd.notna(x) else '-',
                'ret_3d': lambda x: f"{x:+.1f}%" if pd.notna(x) else '-',
                'ret_7d': lambda x: f"{x:+.1f}%" if pd.notna(x) else '-',
                'ret_30d': lambda x: f"{x:+.1f}%" if pd.notna(x) else '-',
                'rvol': lambda x: f"{x:.1f}x" if pd.notna(x) else '-',
                'vmi': lambda x: f"{x:.2f}" if pd.notna(x) else '-',
                'volume_1d': lambda x: f"{x:.1f}" if pd.notna(x) else '-',
                'money_flow_mm': lambda x: f"₹{x:.0f}M" if pd.notna(x) else '-',
                # Fundamental columns formatting
                'pe': lambda x: f"{x:.1f}x" if pd.notna(x) and x > 0 else 'N/A',
                'eps_current': lambda x: f"₹{x:.2f}" if pd.notna(x) else 'N/A',
                'eps_change_pct': lambda x: f"{x:+.1f}%" if pd.notna(x) else 'N/A',
                'market_cap_cr': lambda x: f"₹{x:.0f}Cr" if pd.notna(x) else '-'
            }
            
            # Apply formatting
            for col, formatter in format_rules.items():
                if col in display_df_formatted.columns:
                    display_df_formatted[col] = display_df[col].apply(formatter)
            
            # Format PE with professional logic
            def format_pe_professional(value):
                try:
                    if pd.isna(value) or value == 'N/A':
                        return '-'
                    
                    val = float(value)
                    
                    if val <= 0:
                        return 'Loss'
                    elif val > 10000:
                        return '>10K'
                    elif val > 1000:
                        return f"{val:.0f}"
                    elif val > 100:
                        return f"{val:.1f}"
                    else:
                        return f"{val:.1f}"
                except (ValueError, TypeError, AttributeError):
                    return '-'
            
            # Format EPS Change with professional logic
            def format_eps_change_professional(value):
                try:
                    if pd.isna(value):
                        return '-'
                    
                    val = float(value)
                    
                    if abs(val) >= 1000:
                        return f"{val/1000:+.1f}K%"
                    elif abs(val) >= 100:
                        return f"{val:+.0f}%"
                    else:
                        return f"{val:+.1f}%"
                except (ValueError, TypeError, AttributeError):
                    return '-'
            
            # Apply professional fundamental formatting - ONLY EXISTING COLUMNS
            if show_fundamentals:
                if 'pe' in display_df_formatted.columns:
                    display_df_formatted['pe'] = display_df['pe'].apply(format_pe_professional)
                
                if 'eps_change_pct' in display_df_formatted.columns:
                    display_df_formatted['eps_change_pct'] = display_df['eps_change_pct'].apply(format_eps_change_professional)
                
                # Format EPS current (earnings per share)
                if 'eps_current' in display_df_formatted.columns:
                    display_df_formatted['eps_current'] = display_df['eps_current'].apply(lambda x: f"₹{x:.1f}" if pd.notna(x) and x > 0 else '-')
            
            # Add trend indicators for better visualization
            if 'trend_quality' in display_df.columns:
                def get_trend_indicator_professional(score):
                    if pd.isna(score):
                        return "➖"
                    elif score >= 85:
                        return "🔥"  # Exceptional
                    elif score >= 70:
                        return "🚀"  # Strong
                    elif score >= 55:
                        return "✅"  # Good
                    elif score >= 40:
                        return "➡️"  # Neutral
                    elif score >= 25:
                        return "⚠️"  # Weak
                    else:
                        return "🔻"  # Poor
                
                display_df_formatted['trend_indicator'] = display_df['trend_quality'].apply(get_trend_indicator_professional)
                if view_mode in ["Essential", "Complete"]:
                    final_display_cols['trend_indicator'] = 'Trend'
            
            # Add momentum strength indicators
            if 'momentum_score' in display_df.columns:
                def get_momentum_indicator(score):
                    if pd.isna(score):
                        return "➖"
                    elif score >= 80:
                        return "🎯"  # Excellent
                    elif score >= 60:
                        return "📈"  # Strong
                    elif score >= 40:
                        return "📊"  # Moderate
                    else:
                        return "📉"  # Weak
                
                display_df_formatted['momentum_indicator'] = display_df['momentum_score'].apply(get_momentum_indicator)
                if view_mode == "Essential":
                    final_display_cols['momentum_indicator'] = 'Mom'
            
            # Select and rename columns for final display
            final_display_df = display_df_formatted[list(final_display_cols.keys())]
            final_display_df.columns = list(final_display_cols.values())
            
            # PROFESSIONAL COLUMN CONFIGURATION FOR STREAMLIT
            column_config = {}
            
            # Define comprehensive column configurations
            config_map = {
                "Rank": st.column_config.NumberColumn("Rank", help="Overall ranking position", format="%d", width="tiny"),
                "Ticker": st.column_config.TextColumn("Ticker", help="Stock symbol", width="small"),
                "Company": st.column_config.TextColumn("Company", help="Company name", width="medium", max_chars=40),
                "Master": st.column_config.TextColumn("Master", help="Master Score (0-100)", width="small"),
                "Score": st.column_config.TextColumn("Score", help="Master Score (0-100)", width="small"),
                "Pos": st.column_config.TextColumn("Pos", help="Position Score", width="tiny"),
                "Position": st.column_config.TextColumn("Position", help="Position Score", width="small"),
                "Mom": st.column_config.TextColumn("Mom", help="Momentum Score", width="tiny"),
                "Momentum": st.column_config.TextColumn("Momentum", help="Momentum Score", width="small"),
                "Vol": st.column_config.TextColumn("Vol", help="Volume Score", width="tiny"),
                "Volume": st.column_config.TextColumn("Volume", help="Volume Score", width="small"),
                "Acc": st.column_config.TextColumn("Acc", help="Acceleration Score", width="tiny"),
                "Accel": st.column_config.TextColumn("Accel", help="Acceleration Score", width="small"),
                "Brk": st.column_config.TextColumn("Brk", help="Breakout Score", width="tiny"),
                "Breakout": st.column_config.TextColumn("Breakout", help="Breakout Score", width="small"),
                "RVS": st.column_config.TextColumn("RVS", help="RVOL Score", width="tiny"),
                "RVOL Scr": st.column_config.TextColumn("RVOL Scr", help="RVOL Score (0-100)", width="small"),
                "RVOL": st.column_config.TextColumn("RVOL", help="Relative Volume vs Avg", width="small"),
                "Trd": st.column_config.TextColumn("Trd", help="Trend Quality Score", width="tiny"),
                "Trend": st.column_config.TextColumn("Trend", help="Trend Quality Indicator", width="small"),
                "LTS": st.column_config.TextColumn("LTS", help="Long Term Strength", width="tiny"),
                "LT Str": st.column_config.TextColumn("LT Str", help="Long Term Strength", width="small"),
                "Liq": st.column_config.TextColumn("Liq", help="Liquidity Score", width="tiny"),
                "Liquid": st.column_config.TextColumn("Liquid", help="Liquidity Score", width="small"),
                "MktS": st.column_config.TextColumn("MktS", help="Overall Market Strength", width="tiny"),
                "Mkt Str": st.column_config.TextColumn("Mkt Str", help="Overall Market Strength", width="small"),
                "State": st.column_config.TextColumn("State", help="Current Market State", width="medium"),
                "Price": st.column_config.TextColumn("Price", help="Current stock price", width="small"),
                "Low%": st.column_config.TextColumn("Low%", help="Distance from 52W low", width="small"),
                "High%": st.column_config.TextColumn("High%", help="Distance from 52W high", width="small"),
                "1D%": st.column_config.TextColumn("1D%", help="1-day return", width="small"),
                "3D%": st.column_config.TextColumn("3D%", help="3-day return", width="small"),
                "7D%": st.column_config.TextColumn("7D%", help="7-day return", width="small"),
                "30D%": st.column_config.TextColumn("30D%", help="30-day return", width="small"),
                "VMI": st.column_config.TextColumn("VMI", help="Volume Momentum Index", width="small"),
                "Vol(Cr)": st.column_config.TextColumn("Vol(Cr)", help="Volume in Crores", width="small"),
                "MF(MM)": st.column_config.TextColumn("MF(MM)", help="Money Flow in MM", width="small"),
                "ATR%": st.column_config.TextColumn("ATR%", help="Average True Range %", width="small"),
                "RSI": st.column_config.TextColumn("RSI", help="Relative Strength Index", width="small"),
                "Patterns": st.column_config.TextColumn("Patterns", help="Technical Patterns", width="large", max_chars=80),
                "Category": st.column_config.TextColumn("Category", help="Market Cap Category", width="medium"),
                "Sector": st.column_config.TextColumn("Sector", help="Sector Classification", width="medium"),
                "Industry": st.column_config.TextColumn("Industry", help="Industry Classification", width="medium", max_chars=40),
                
                # Fundamental columns - ONLY WHAT EXISTS IN V9.PY
                "PE": st.column_config.TextColumn("PE", help="Price to Earnings Ratio", width="small"),
                "EPS": st.column_config.TextColumn("EPS", help="Earnings Per Share (Current)", width="small"),
                "EPS Δ%": st.column_config.TextColumn("EPS Δ%", help="EPS Change %", width="small"),
                
                # Indicator columns
                "Mom": st.column_config.TextColumn("Mom", help="Momentum Indicator", width="tiny"),
            }
            
            # Apply configurations for columns that exist in our dataframe
            for col_name in final_display_df.columns:
                if col_name in config_map:
                    column_config[col_name] = config_map[col_name]
                else:
                    # Default configuration for any missing columns
                    column_config[col_name] = st.column_config.TextColumn(col_name, width="medium")
            
            # ENHANCED MAIN DATAFRAME DISPLAY
            st.markdown("#### 📊 Main Rankings Table")
            
            # Enhanced performance timeframe display with detailed info
            if performance_timeframe != "All":
                timeframe_info_cols = st.columns([3, 1])
                
                with timeframe_info_cols[0]:
                    # Show performance impact details
                    if 'master_score_adjusted' in display_df.columns:
                        adjustment_stats = {
                            'boosted': (display_df['master_score_adjusted'] > display_df['master_score']).sum(),
                            'penalized': (display_df['master_score_adjusted'] < display_df['master_score']).sum(),
                            'unchanged': (display_df['master_score_adjusted'] == display_df['master_score']).sum()
                        }
                        
                        st.info(f"📅 **{performance_timeframe} Performance Focus Active** | "
                               f"🚀 {adjustment_stats['boosted']} boosted • "
                               f"⚠️ {adjustment_stats['penalized']} penalized • "
                               f"➖ {adjustment_stats['unchanged']} unchanged")
                    else:
                        st.info(f"📅 **Performance Focus**: {performance_timeframe} timeframe analysis")
                
                with timeframe_info_cols[1]:
                    if st.button("🔄 Reset to All", key="reset_timeframe", help="Show all timeframes"):
                        st.rerun()
            
            st.dataframe(
                final_display_df,
                width='stretch',
                height=min(800, len(final_display_df) * 35 + 100),
                hide_index=True,
                column_config=column_config
            )
            
            # PROFESSIONAL EXPORT FUNCTIONALITY
            if export_format != "None":
                st.markdown("---")
                export_cols = st.columns([3, 1])
                
                with export_cols[0]:
                    st.markdown(f"#### 💾 Export Data ({export_format})")
                    
                with export_cols[1]:
                    if export_format == "CSV":
                        csv_data = ExportEngine.create_csv_export(display_df)
                        st.download_button(
                            label=f"📥 Download Top {display_count} ({export_format})",
                            data=csv_data,
                            file_name=f"professional_rankings_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "Excel":
                        # Create Excel export with multiple sheets
                        excel_buffer = BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            display_df.to_excel(writer, sheet_name='Rankings', index=False)
                            
                            # Add summary sheet - USING ACTUAL V9.PY COLUMNS
                            summary_data = {
                                'Metric': ['Total Stocks', 'Avg Master Score', 'Avg 30D Return', 'Top Category', 'Avg PE'],
                                'Value': [
                                    len(display_df),
                                    f"{display_df['master_score'].mean():.1f}",
                                    f"{display_df['ret_30d'].mean():.1f}%" if 'ret_30d' in display_df.columns else 'N/A',
                                    display_df['category'].mode().iloc[0] if 'category' in display_df.columns else 'N/A',
                                    f"{display_df['pe'].median():.1f}" if 'pe' in display_df.columns else 'N/A'
                                ]
                            }
                            pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
                        
                        excel_buffer.seek(0)
                        st.download_button(
                            label=f"📥 Download Top {display_count} (Excel)",
                            data=excel_buffer.getvalue(),
                            file_name=f"professional_rankings_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    elif export_format == "JSON":
                        json_data = display_df.to_json(orient='records', indent=2)
                        st.download_button(
                            label=f"📥 Download Top {display_count} (JSON)",
                            data=json_data,
                            file_name=f"professional_rankings_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
            
            
            # PROFESSIONAL ANALYTICS SECTION - COMPREHENSIVE INSIGHTS
            st.markdown("---")
            st.markdown("### 📊 PROFESSIONAL ANALYTICS & INSIGHTS")
            
            # Create analytics tabs for better organization
            analytics_tabs = st.tabs(["📈 Score Analysis", "💰 Performance Metrics", "🔍 Technical Analysis", "🏭 Sector Analysis", "⚡ Quick Stats"])
            
            with analytics_tabs[0]:  # Score Analysis
                st.markdown("#### 📈 Score Distribution Analysis")
                
                score_cols = st.columns(4)
                
                with score_cols[0]:
                    st.markdown("**🎯 Master Score Distribution**")
                    if 'master_score' in display_df.columns:
                        score_stats = {
                            'Elite (90+)': f"{(display_df['master_score'] >= 90).sum()}",
                            'Excellent (80-89)': f"{((display_df['master_score'] >= 80) & (display_df['master_score'] < 90)).sum()}",
                            'Good (70-79)': f"{((display_df['master_score'] >= 70) & (display_df['master_score'] < 80)).sum()}",
                            'Average (60-69)': f"{((display_df['master_score'] >= 60) & (display_df['master_score'] < 70)).sum()}",
                            'Below Avg (<60)': f"{(display_df['master_score'] < 60).sum()}",
                            'Mean Score': f"{display_df['master_score'].mean():.1f}",
                            'Median Score': f"{display_df['master_score'].median():.1f}"
                        }
                        
                        score_df = pd.DataFrame(list(score_stats.items()), columns=['Range', 'Count'])
                        st.dataframe(score_df, width='stretch', hide_index=True)
                
                with score_cols[1]:
                    st.markdown("**🚀 Momentum Analysis**")
                    if 'momentum_score' in display_df.columns:
                        momentum_stats = {
                            'Explosive (80+)': f"{(display_df['momentum_score'] >= 80).sum()}",
                            'Strong (60-79)': f"{((display_df['momentum_score'] >= 60) & (display_df['momentum_score'] < 80)).sum()}",
                            'Moderate (40-59)': f"{((display_df['momentum_score'] >= 40) & (display_df['momentum_score'] < 60)).sum()}",
                            'Weak (<40)': f"{(display_df['momentum_score'] < 40).sum()}",
                            'Avg Momentum': f"{display_df['momentum_score'].mean():.1f}",
                            'Top 10 Avg': f"{display_df.head(10)['momentum_score'].mean():.1f}"
                        }
                        
                        momentum_df = pd.DataFrame(list(momentum_stats.items()), columns=['Level', 'Value'])
                        st.dataframe(momentum_df, width='stretch', hide_index=True)
                
                with score_cols[2]:
                    st.markdown("**📊 Volume Analysis**") 
                    if 'volume_score' in display_df.columns:
                        volume_stats = {
                            'Exceptional (80+)': f"{(display_df['volume_score'] >= 80).sum()}",
                            'High (60-79)': f"{((display_df['volume_score'] >= 60) & (display_df['volume_score'] < 80)).sum()}",
                            'Normal (40-59)': f"{((display_df['volume_score'] >= 40) & (display_df['volume_score'] < 60)).sum()}",
                            'Low (<40)': f"{(display_df['volume_score'] < 40).sum()}",
                            'Avg Vol Score': f"{display_df['volume_score'].mean():.1f}",
                            'RVOL > 3x': f"{(display_df['rvol'] > 3).sum()}" if 'rvol' in display_df.columns else 'N/A'
                        }
                        
                        volume_df = pd.DataFrame(list(volume_stats.items()), columns=['Level', 'Count'])
                        st.dataframe(volume_df, width='stretch', hide_index=True)
                
                with score_cols[3]:
                    st.markdown("**📍 Position Analysis**")
                    if 'position_score' in display_df.columns:
                        position_stats = {
                            'Near Highs (80+)': f"{(display_df['position_score'] >= 80).sum()}",
                            'Strong (60-79)': f"{((display_df['position_score'] >= 60) & (display_df['position_score'] < 80)).sum()}",
                            'Mid Range (40-59)': f"{((display_df['position_score'] >= 40) & (display_df['position_score'] < 60)).sum()}",
                            'Near Lows (<40)': f"{(display_df['position_score'] < 40).sum()}",
                            'Avg Position': f"{display_df['position_score'].mean():.1f}",
                            'From Low Avg': f"{display_df['from_low_pct'].mean():.0f}%" if 'from_low_pct' in display_df.columns else 'N/A'
                        }
                        
                        position_df = pd.DataFrame(list(position_stats.items()), columns=['Range', 'Value'])
                        st.dataframe(position_df, width='stretch', hide_index=True)
            
            with analytics_tabs[1]:  # Performance Metrics
                st.markdown("#### 💰 Performance & Returns Analysis")
                
                perf_cols = st.columns(3)
                
                with perf_cols[0]:
                    st.markdown("**📈 Returns Distribution**")
                    return_stats = {}
                    
                    if performance_timeframe == "1D" and 'ret_1d' in display_df.columns:
                        ret_col = 'ret_1d'
                    elif performance_timeframe == "3D" and 'ret_3d' in display_df.columns:
                        ret_col = 'ret_3d'
                    elif performance_timeframe == "7D" and 'ret_7d' in display_df.columns:
                        ret_col = 'ret_7d'
                    elif 'ret_30d' in display_df.columns:
                        ret_col = 'ret_30d'
                    else:
                        ret_col = None
                    
                    if ret_col:
                        return_stats = {
                            'Best Performer': f"{display_df[ret_col].max():.1f}%",
                            'Worst Performer': f"{display_df[ret_col].min():.1f}%",
                            'Average Return': f"{display_df[ret_col].mean():.1f}%",
                            'Median Return': f"{display_df[ret_col].median():.1f}%",
                            'Positive Returns': f"{(display_df[ret_col] > 0).sum()}",
                            'Negative Returns': f"{(display_df[ret_col] < 0).sum()}",
                            'Win Rate': f"{(display_df[ret_col] > 0).sum() / len(display_df) * 100:.0f}%",
                            'Strong Gains (>10%)': f"{(display_df[ret_col] > 10).sum()}",
                            'Big Gains (>20%)': f"{(display_df[ret_col] > 20).sum()}"
                        }
                    else:
                        return_stats = {'No Data': 'Returns data not available'}
                    
                    ret_df = pd.DataFrame(list(return_stats.items()), columns=['Metric', 'Value'])
                    st.dataframe(ret_df, width='stretch', hide_index=True)
                
                with perf_cols[1]:
                    st.markdown("**💎 Quality Metrics**")
                    if show_fundamentals:
                        fund_stats = {}
                        
                        # ONLY USE COLUMNS THAT EXIST IN V9.PY
                        if 'pe' in display_df.columns:
                            valid_pe = display_df['pe'].notna() & (display_df['pe'] > 0) & (display_df['pe'] < 1000)
                            if valid_pe.any():
                                fund_stats.update({
                                    'Median PE': f"{display_df.loc[valid_pe, 'pe'].median():.1f}x",
                                    'Value Stocks (PE<15)': f"{(display_df['pe'] < 15).sum()}",
                                    'Growth Stocks (PE>30)': f"{(display_df['pe'] > 30).sum()}"
                                })
                        
                        if 'eps_change_pct' in display_df.columns:
                            valid_eps = display_df['eps_change_pct'].notna()
                            if valid_eps.any():
                                fund_stats.update({
                                    'EPS Growth +ve': f"{(display_df['eps_change_pct'] > 0).sum()}",
                                    'Strong Growth (>25%)': f"{(display_df['eps_change_pct'] > 25).sum()}",
                                    'Avg EPS Growth': f"{display_df['eps_change_pct'].mean():.1f}%"
                                })
                        
                        if 'eps_current' in display_df.columns:
                            valid_eps_current = display_df['eps_current'].notna() & (display_df['eps_current'] > 0)
                            if valid_eps_current.any():
                                fund_stats.update({
                                    'Profitable Stocks': f"{(display_df['eps_current'] > 0).sum()}",
                                    'Avg EPS': f"₹{display_df.loc[valid_eps_current, 'eps_current'].mean():.1f}"
                                })
                        
                        if fund_stats:
                            fund_df = pd.DataFrame(list(fund_stats.items()), columns=['Metric', 'Value'])
                            st.dataframe(fund_df, width='stretch', hide_index=True)
                        else:
                            st.info("No fundamental data available")
                    else:
                        liquidity_stats = {}
                        if 'liquidity_score' in display_df.columns:
                            liquidity_stats = {
                                'High Liquidity (80+)': f"{(display_df['liquidity_score'] >= 80).sum()}",
                                'Good Liquidity (60-79)': f"{((display_df['liquidity_score'] >= 60) & (display_df['liquidity_score'] < 80)).sum()}",
                                'Average Liquidity': f"{display_df['liquidity_score'].mean():.1f}",
                                'Top 10 Avg Liq': f"{display_df.head(10)['liquidity_score'].mean():.1f}"
                            }
                        
                        if 'volume_1d' in display_df.columns:
                            liquidity_stats.update({
                                'Avg Volume (Cr)': f"{display_df['volume_1d'].mean():.1f}",
                                'Max Volume (Cr)': f"{display_df['volume_1d'].max():.1f}"
                            })
                        
                        liq_df = pd.DataFrame(list(liquidity_stats.items()), columns=['Metric', 'Value'])
                        st.dataframe(liq_df, width='stretch', hide_index=True)
                
                with perf_cols[2]:
                    st.markdown("**⚡ Risk & Volatility**")
                    risk_stats = {}
                    
                    if 'from_high_pct' in display_df.columns:
                        risk_stats.update({
                            'Near Highs (>-5%)': f"{(display_df['from_high_pct'] > -5).sum()}",
                            'Deep Correction (<-20%)': f"{(display_df['from_high_pct'] < -20).sum()}"
                        })
                    
                    if risk_stats:
                        risk_df = pd.DataFrame(list(risk_stats.items()), columns=['Metric', 'Value'])
                        st.dataframe(risk_df, width='stretch', hide_index=True)
                    else:
                        st.info("Risk metrics not available")
            
            with analytics_tabs[2]:  # Technical Analysis
                st.markdown("#### 🔍 Advanced Technical Analysis")
                
                tech_cols = st.columns(3)
                
                with tech_cols[0]:
                    st.markdown("**🎯 Pattern Analysis**")
                    if 'patterns' in display_df.columns:
                        # Count pattern occurrences
                        pattern_counts = {}
                        for patterns_str in display_df['patterns'].dropna():
                            if patterns_str and patterns_str.strip():
                                for pattern in patterns_str.split(' | '):
                                    pattern = pattern.strip()
                                    if pattern:
                                        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                        
                        if pattern_counts:
                            # Get top 8 patterns
                            top_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:8]
                            pattern_data = [{'Pattern': p, 'Count': c} for p, c in top_patterns]
                            patterns_df = pd.DataFrame(pattern_data)
                            st.dataframe(patterns_df, width='stretch', hide_index=True)
                        else:
                            st.info("No patterns detected")
                    else:
                        st.info("Pattern data not available")
                
                with tech_cols[1]:
                    st.markdown("**📊 Breakout Analysis**")
                    if 'breakout_score' in display_df.columns:
                        breakout_stats = {
                            'Strong Breakouts (80+)': f"{(display_df['breakout_score'] >= 80).sum()}",
                            'Potential Breakouts (60-79)': f"{((display_df['breakout_score'] >= 60) & (display_df['breakout_score'] < 80)).sum()}",
                            'Building (40-59)': f"{((display_df['breakout_score'] >= 40) & (display_df['breakout_score'] < 60)).sum()}",
                            'No Breakout (<40)': f"{(display_df['breakout_score'] < 40).sum()}",
                            'Avg Breakout Score': f"{display_df['breakout_score'].mean():.1f}",
                            'Top 10 Avg': f"{display_df.head(10)['breakout_score'].mean():.1f}"
                        }
                        
                        breakout_df = pd.DataFrame(list(breakout_stats.items()), columns=['Level', 'Count'])
                        st.dataframe(breakout_df, width='stretch', hide_index=True)
                
                with tech_cols[2]:
                    st.markdown("**⚡ Acceleration Signals**")
                    if 'acceleration_score' in display_df.columns:
                        accel_stats = {
                            'High Acceleration (80+)': f"{(display_df['acceleration_score'] >= 80).sum()}",
                            'Moderate (60-79)': f"{((display_df['acceleration_score'] >= 60) & (display_df['acceleration_score'] < 80)).sum()}",
                            'Building (40-59)': f"{((display_df['acceleration_score'] >= 40) & (display_df['acceleration_score'] < 60)).sum()}",
                            'Low (<40)': f"{(display_df['acceleration_score'] < 40).sum()}",
                            'Avg Acceleration': f"{display_df['acceleration_score'].mean():.1f}",
                            'Momentum Leaders': f"{((display_df['acceleration_score'] >= 80) & (display_df['momentum_score'] >= 80)).sum()}"
                        }
                        
                        accel_df = pd.DataFrame(list(accel_stats.items()), columns=['Level', 'Count'])
                        st.dataframe(accel_df, width='stretch', hide_index=True)
            
            with analytics_tabs[3]:  # Sector Analysis
                st.markdown("#### 🏭 Sector & Category Performance")
                
                sector_cols = st.columns(2)
                
                with sector_cols[0]:
                    st.markdown("**📈 Category Performance**")
                    if 'category' in display_df.columns:
                        cat_performance = display_df.groupby('category').agg({
                            'master_score': ['mean', 'count'],
                            'ret_30d': 'mean' if 'ret_30d' in display_df.columns else lambda x: 0,
                            'rvol': 'mean' if 'rvol' in display_df.columns else lambda x: 0
                        }).round(2)
                        
                        # Flatten column names
                        cat_performance.columns = ['Avg Score', 'Count', 'Avg 30D Ret', 'Avg RVOL']
                        cat_performance = cat_performance.sort_values('Avg Score', ascending=False)
                        
                        # Format the display
                        cat_display = cat_performance.copy()
                        if 'Avg 30D Ret' in cat_display.columns:
                            cat_display['Avg 30D Ret'] = cat_display['Avg 30D Ret'].apply(lambda x: f"{x:.1f}%")
                        if 'Avg RVOL' in cat_display.columns:
                            cat_display['Avg RVOL'] = cat_display['Avg RVOL'].apply(lambda x: f"{x:.1f}x")
                        
                        st.dataframe(cat_display, width='stretch')
                    else:
                        st.info("Category data not available")
                
                with sector_cols[1]:
                    st.markdown("**🏢 Sector Distribution**")
                    if 'sector' in display_df.columns:
                        sector_counts = display_df['sector'].value_counts().head(10)
                        sector_df = pd.DataFrame({
                            'Sector': sector_counts.index,
                            'Count': sector_counts.values
                        })
                        st.dataframe(sector_df, width='stretch', hide_index=True)
                    else:
                        st.info("Sector data not available")
            
            with analytics_tabs[4]:  # Quick Stats
                st.markdown("#### ⚡ Quick Professional Stats")
                
                quick_cols = st.columns(4)
                
                with quick_cols[0]:
                    UIComponents.render_metric_card(
                        "💎 Elite Stocks", 
                        f"{(display_df['master_score'] >= 90).sum()}",
                        f"{(display_df['master_score'] >= 90).sum()/len(display_df)*100:.0f}% of total"
                    )
                
                with quick_cols[1]:
                    if 'patterns' in display_df.columns:
                        with_patterns = (display_df['patterns'] != '').sum()
                        UIComponents.render_metric_card(
                            "🎯 With Patterns", 
                            f"{with_patterns}",
                            f"{with_patterns/len(display_df)*100:.0f}% have signals"
                        )
                
                with quick_cols[2]:
                    if 'rvol' in display_df.columns:
                        high_volume = (display_df['rvol'] > 2).sum()
                        UIComponents.render_metric_card(
                            "🔊 High Volume", 
                            f"{high_volume}",
                            f"RVOL > 2x"
                        )
                
                with quick_cols[3]:
                    if 'ret_30d' in display_df.columns:
                        positive_returns = (display_df['ret_30d'] > 0).sum()
                        UIComponents.render_metric_card(
                            "📈 Positive Returns", 
                            f"{positive_returns}",
                            f"{positive_returns/len(display_df)*100:.0f}% winners"
                        )
        
        else:
            st.warning("No stocks match the selected filters.")
            
            # Show filter summary
            st.markdown("#### Current Filters Applied:")
            active_filter_count = FilterEngine.get_active_count()
            if active_filter_count > 0:
                filter_summary = []
                
                if st.session_state.filter_state.get('categories'):
                    filter_summary.append(f"Categories: {', '.join(st.session_state.filter_state['categories'])}")
                if st.session_state.filter_state.get('sectors'):
                    filter_summary.append(f"Sectors: {', '.join(st.session_state.filter_state['sectors'])}")
                if st.session_state.filter_state.get('industries'):
                    filter_summary.append(f"Industries: {', '.join(st.session_state.filter_state['industries'][:5])}...")
                if st.session_state.filter_state.get('min_score', 0) > 0:
                    filter_summary.append(f"Min Score: {st.session_state.filter_state['min_score']}")
                if st.session_state.filter_state.get('patterns'):
                    filter_summary.append(f"Patterns: {len(st.session_state.filter_state['patterns'])} selected")
                
                for filter_text in filter_summary:
                    st.write(f"• {filter_text}")
                
                if st.button("Clear All Filters", type="primary", key="clear_filters_ranking_btn"):
                    FilterEngine.clear_all_filters()
                    SessionStateManager.clear_filters()
                    st.rerun()
            else:
                st.info("No filters applied. All stocks should be visible unless there's no data loaded.")
        
    # Tab 2: Ultimate Market Radar - ALL TIME BEST IMPLEMENTATION
    with tabs[2]:
        st.markdown("### 🎯 MARKET RADAR - TRADING INTELLIGENCE")
        st.markdown("*Advanced multi-dimensional market analysis for professional traders*")
        
        # ================================================================================================
        # 🔥 PROFESSIONAL CONTROL PANEL
        # ================================================================================================
        
        # Main Control Row
        radar_controls = st.columns([2, 2, 2, 2, 2])
        
        with radar_controls[0]:
            radar_mode = st.selectbox(
                "🎯 Radar Mode",
                options=[
                    "🌊 Wave Hunter", 
                    "⚡ Breakout Scanner", 
                    "🏗️ Pattern Recognition",
                    "💰 Institutional Flow",
                    "🔥 Momentum Surge",
                    "📊 Full Spectrum"
                ],
                index=5,  # Default to Full Spectrum
                key="radar_mode_select",
                help="Choose your market analysis focus"
            )
        
        with radar_controls[1]:
            timeframe_focus = st.selectbox(
                "⏰ Timeframe Focus",
                options=[
                    "🚀 Intraday (1D)",
                    "📈 Short-term (3D)", 
                    "🌊 Medium-term (7D)",
                    "💪 Long-term (30D)",
                    "🔄 Multi-timeframe"
                ],
                index=4,  # Default to Multi-timeframe
                key="timeframe_focus_select",
                help="Primary analysis timeframe"
            )
        
        with radar_controls[2]:
            sensitivity_level = st.select_slider(
                "🎚️ Signal Sensitivity",
                options=["🛡️ Ultra Conservative", "🔒 Conservative", "⚖️ Balanced", "🚀 Aggressive", "🔥 Ultra Aggressive"],
                value="⚖️ Balanced",
                key="sensitivity_level_select",
                help="Signal detection sensitivity level"
            )
        
        with radar_controls[3]:
            risk_filter = st.selectbox(
                "⚖️ Risk Profile",
                options=[
                    "🛡️ Low Risk Only",
                    "⚖️ Balanced Risk", 
                    "🚀 High Risk/Reward",
                    "🔥 Maximum Alpha"
                ],
                index=1,
                key="risk_filter_select",
                help="Filter opportunities by risk profile"
            )
        
        with radar_controls[4]:
            market_regime = st.selectbox(
                "📊 Market Regime",
                options=[
                    "🐂 Bull Market",
                    "🐻 Bear Market", 
                    "🔄 Sideways/Choppy",
                    "📊 Auto-Detect"
                ],
                index=3,
                key="market_regime_select",
                help="Adjust analysis for market conditions"
            )
        
        # Advanced Controls Row
        advanced_controls = st.columns([2, 2, 2, 2, 2])
        
        with advanced_controls[0]:
            enable_ai_signals = st.checkbox(
                "🧠 AI Pattern Recognition",
                value=True,
                key="enable_ai_signals",
                help="Enable AI-powered pattern detection"
            )
        
        with advanced_controls[1]:
            show_institutional = st.checkbox(
                "🏦 Institutional Analysis",
                value=True, 
                key="show_institutional",
                help="Show institutional flow analysis"
            )
        
        with advanced_controls[2]:
            enable_alerts = st.checkbox(
                "🚨 Smart Alerts",
                value=True,
                key="enable_alerts", 
                help="Enable intelligent trading alerts"
            )
        
        with advanced_controls[3]:
            show_correlations = st.checkbox(
                "🔗 Cross-Asset Analysis",
                value=False,
                key="show_correlations",
                help="Show sector/asset correlations"
            )
        
        with advanced_controls[4]:
            export_signals = st.checkbox(
                "📤 Export Signals",
                value=False,
                key="export_signals",
                help="Enable signal export functionality"
            )
        
        # Additional Control Variables
        show_sensitivity_details = st.checkbox(
            "📊 Show Sensitivity Details",
            value=False,
            key="show_sensitivity_details",
            help="Display detailed sensitivity threshold information"
        )
        
        show_market_regime = st.checkbox(
            "📊 Show Market Regime",
            value=True,
            key="show_market_regime",
            help="Display market regime and category rotation analysis"
        )
        
        st.markdown("---")
        
        # ================================================================================================
        # 🧠 AI-POWERED MARKET INTELLIGENCE ENGINE
        # ================================================================================================
        
        # Initialize the filtered dataframe
        radar_df = filtered_df.copy()
        
        # ================================================================================================
        # 🚨 CRITICAL FIX: APPLY RADAR MODE FILTERING (PREVIOUSLY MISSING!)
        # ================================================================================================
        
        # Apply Radar Mode filtering
        original_count = len(radar_df)
        
        if radar_mode == "🌊 Wave Hunter":
            radar_df = radar_df[
                (radar_df.get('momentum_score', 0) >= 50) &
                (radar_df.get('rvol', 0) >= 1.5) &
                (radar_df.get('acceleration_score', 0) >= 40)
            ]
            st.info(f"🌊 Wave Hunter Mode: {len(radar_df)}/{original_count} stocks meet momentum+volume criteria")
            
        elif radar_mode == "⚡ Breakout Scanner":
            radar_df = radar_df[radar_df.get('breakout_score', 0) >= 65]
            st.info(f"⚡ Breakout Scanner Mode: {len(radar_df)}/{original_count} stocks have breakout score ≥65")
            
        elif radar_mode == "🏗️ Pattern Recognition":
            if 'patterns' in radar_df.columns:
                radar_df = radar_df[radar_df['patterns'].str.len() > 0]
                st.info(f"🏗️ Pattern Recognition Mode: {len(radar_df)}/{original_count} stocks have detected patterns")
            else:
                st.warning("🏗️ Pattern Recognition Mode: 'patterns' column not available - using all data")
                
        elif radar_mode == "💰 Institutional Flow":
            flow_filter = (
                (radar_df.get('money_flow_mm', 0).abs() >= 10) |
                (radar_df.get('rvol', 0) >= 3.0)
            )
            radar_df = radar_df[flow_filter]
            st.info(f"💰 Institutional Flow Mode: {len(radar_df)}/{original_count} stocks show institutional activity")
            
        elif radar_mode == "🔥 Momentum Surge":
            momentum_filter = (
                (radar_df.get('momentum_score', 0) >= 60) &
                (radar_df.get('acceleration_score', 0) >= 50)
            )
            radar_df = radar_df[momentum_filter]
            st.info(f"🔥 Momentum Surge Mode: {len(radar_df)}/{original_count} stocks in momentum surge")
            
        # 📊 Full Spectrum uses all data (no additional filtering)
        elif radar_mode == "📊 Full Spectrum":
            pass  # No filtering applied
        
        # ================================================================================================
        # 🚨 CRITICAL FIX: APPLY RISK PROFILE FILTERING (PREVIOUSLY MISSING!)
        # ================================================================================================
        
        pre_risk_count = len(radar_df)
        
        if risk_filter == "🛡️ Low Risk Only":
            risk_filter_condition = (
                (radar_df.get('pe', 999) < 25) &
                (radar_df.get('volatility_score', 100) < 60) &
                (radar_df.get('master_score', 0) >= 50)
            )
            radar_df = radar_df[risk_filter_condition]
            st.info(f"🛡️ Low Risk Filter: {len(radar_df)}/{pre_risk_count} stocks meet low-risk criteria")
            
        elif risk_filter == "🚀 High Risk/Reward":
            risk_filter_condition = (
                (radar_df.get('momentum_score', 0) >= 70) |
                (radar_df.get('from_low_pct', 0) >= 100) |
                (radar_df.get('volatility_score', 0) >= 70)
            )
            radar_df = radar_df[risk_filter_condition]
            st.info(f"🚀 High Risk/Reward Filter: {len(radar_df)}/{pre_risk_count} high-potential stocks")
            
        elif risk_filter == "🔥 Maximum Alpha":
            risk_filter_condition = (
                (radar_df.get('master_score', 0) >= 80) &
                (radar_df.get('rvol', 0) >= 2.0) &
                (radar_df.get('momentum_score', 0) >= 65)
            )
            radar_df = radar_df[risk_filter_condition]
            st.info(f"🔥 Maximum Alpha Filter: {len(radar_df)}/{pre_risk_count} elite alpha-generating stocks")
            
        # ⚖️ Balanced Risk uses current data (no additional filtering)
        elif risk_filter == "⚖️ Balanced Risk":
            pass  # No filtering applied
        
        # ================================================================================================
        # 🚨 CRITICAL FIX: APPLY MARKET REGIME FILTERING (PREVIOUSLY MISSING!)
        # ================================================================================================
        
        pre_regime_count = len(radar_df)
        
        if market_regime == "🐂 Bull Market":
            regime_filter = (
                (radar_df.get('trend_score', 0) >= 60) &
                (radar_df.get('momentum_score', 0) >= 50)
            )
            radar_df = radar_df[regime_filter]
            st.info(f"🐂 Bull Market Filter: {len(radar_df)}/{pre_regime_count} stocks aligned with bull market")
            
        elif market_regime == "🐻 Bear Market":
            regime_filter = (
                (radar_df.get('from_high_pct', 0) < -20) &
                (radar_df.get('value_score', 0) >= 60)
            )
            radar_df = radar_df[regime_filter]
            st.info(f"🐻 Bear Market Filter: {len(radar_df)}/{pre_regime_count} defensive/value stocks")
            
        elif market_regime == "🔄 Sideways/Choppy":
            regime_filter = (
                (radar_df.get('volatility_score', 50) >= 40) &
                (radar_df.get('volatility_score', 50) <= 70)
            )
            radar_df = radar_df[regime_filter]
            st.info(f"🔄 Sideways Market Filter: {len(radar_df)}/{pre_regime_count} range-bound opportunities")
            
        # 📊 Auto-Detect uses current data
        elif market_regime == "📊 Auto-Detect":
            pass  # No filtering applied
        
        # ================================================================================================
        # 🚨 CRITICAL FIX: CREATE SENSITIVITY THRESHOLD FUNCTION (PREVIOUSLY MISSING!)
        # ================================================================================================
        
        def get_sensitivity_thresholds(sensitivity_level):
            """Get thresholds based on sensitivity level for consistent application across all tabs"""
            if "Conservative" in sensitivity_level:
                return {
                    'momentum': 70, 'acceleration': 80, 'rvol': 2.5,
                    'breakout': 80, 'pattern': 75, 'trend': 80, 'confluence': 70,
                    'institutional': 50, 'probability': 75
                }
            elif "Balanced" in sensitivity_level:
                return {
                    'momentum': 60, 'acceleration': 70, 'rvol': 2.0, 
                    'breakout': 70, 'pattern': 65, 'trend': 70, 'confluence': 60,
                    'institutional': 25, 'probability': 65
                }
            else:  # Aggressive
                return {
                    'momentum': 50, 'acceleration': 60, 'rvol': 1.5,
                    'breakout': 60, 'pattern': 55, 'trend': 60, 'confluence': 50,
                    'institutional': 10, 'probability': 55
                }
        
        # Get sensitivity thresholds for use throughout all analysis
        thresholds = get_sensitivity_thresholds(sensitivity_level)
        
        if not radar_df.empty:
            
            # ============================================================================================
            # 📊 REAL-TIME MARKET OVERVIEW DASHBOARD
            # ============================================================================================
            
            st.markdown("### 📊 REAL-TIME MARKET INTELLIGENCE")
            
            # Market Overview Metrics
            overview_cols = st.columns([2, 2, 2, 2, 2])
            
            try:
                with overview_cols[0]:
                    total_stocks = len(radar_df)
                    strong_stocks = len(radar_df[radar_df['master_score'] >= 70]) if 'master_score' in radar_df.columns else 0
                    strength_pct = (strong_stocks / total_stocks * 100) if total_stocks > 0 else 0
                    
                    UIComponents.render_metric_card(
                        "Market Strength",
                        f"{strength_pct:.1f}%",
                        f"{strong_stocks}/{total_stocks} stocks"
                    )
                
                with overview_cols[1]:
                    if 'rvol' in radar_df.columns:
                        high_volume = len(radar_df[radar_df['rvol'] >= 2.0])
                        volume_pct = (high_volume / total_stocks * 100) if total_stocks > 0 else 0
                        volume_status = "🔥 Active" if volume_pct > 30 else "📊 Normal" if volume_pct > 15 else "💤 Quiet"
                        
                        UIComponents.render_metric_card(
                            "Volume Activity",
                            f"{volume_pct:.1f}%",
                            f"{volume_status}"
                        )
                    else:
                        UIComponents.render_metric_card("Volume Activity", "N/A", "Data not available")
                
                with overview_cols[2]:
                    if 'momentum_score' in radar_df.columns:
                        momentum_avg = radar_df['momentum_score'].mean()
                        momentum_trend = "🚀 Bullish" if momentum_avg > 60 else "📊 Neutral" if momentum_avg > 40 else "🐻 Bearish"
                        
                        UIComponents.render_metric_card(
                            "Momentum Regime",
                            f"{momentum_avg:.0f}",
                            momentum_trend
                        )
                    else:
                        UIComponents.render_metric_card("Momentum Regime", "N/A", "Data not available")
                
                with overview_cols[3]:
                    if 'patterns' in radar_df.columns:
                        pattern_stocks = len(radar_df[radar_df['patterns'].str.len() > 0])
                        pattern_pct = (pattern_stocks / total_stocks * 100) if total_stocks > 0 else 0
                        pattern_activity = "🎯 High" if pattern_pct > 20 else "📈 Moderate" if pattern_pct > 10 else "📊 Low"
                        
                        UIComponents.render_metric_card(
                            "Pattern Activity",
                            f"{pattern_pct:.1f}%",
                            f"{pattern_activity}"
                        )
                    else:
                        UIComponents.render_metric_card("Pattern Activity", "N/A", "Data not available")
                
                with overview_cols[4]:
                    if 'breakout_score' in radar_df.columns:
                        breakout_candidates = len(radar_df[radar_df['breakout_score'] >= 75])
                        breakout_pct = (breakout_candidates / total_stocks * 100) if total_stocks > 0 else 0
                        breakout_status = "🚀 Explosive" if breakout_pct > 15 else "📈 Active" if breakout_pct > 8 else "📊 Quiet"
                        
                        UIComponents.render_metric_card(
                            "Breakout Potential",
                            f"{breakout_pct:.1f}%",
                            breakout_status
                        )
                    else:
                        UIComponents.render_metric_card("Breakout Potential", "N/A", "Data not available")
            
            except Exception as e:
                logger.error(f"Error in market overview: {str(e)}")
                st.error("Error calculating market overview metrics")
            
            st.markdown("---")
        
        if show_sensitivity_details:
            with st.expander("📊 Current Sensitivity Thresholds", expanded=True):
                if "Conservative" in sensitivity_level:
                    st.markdown("""
                    **Conservative Settings** 🛡️
                    - **Momentum Shifts:** Score ≥ 60, Acceleration ≥ 70
                    - **Emerging Patterns:** Within 5% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 3.0x (extreme volumes only)
                    - **Acceleration Alerts:** Score ≥ 85 (strongest signals)
                    - **Pattern Distance:** 5% from qualification
                    """)
                elif "Balanced" in sensitivity_level:
                    st.markdown("""
                    **Balanced Settings** ⚖️
                    - **Momentum Shifts:** Score ≥ 50, Acceleration ≥ 60
                    - **Emerging Patterns:** Within 10% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 2.0x (standard threshold)
                    - **Acceleration Alerts:** Score ≥ 70 (good acceleration)
                    - **Pattern Distance:** 10% from qualification
                    """)
                else:  # Aggressive
                    st.markdown("""
                    **Aggressive Settings** 🚀
                    - **Momentum Shifts:** Score ≥ 40, Acceleration ≥ 50
                    - **Emerging Patterns:** Within 15% of qualifying threshold
                    - **Volume Surges:** RVOL ≥ 1.5x (building volume)
                    - **Acceleration Alerts:** Score ≥ 60 (early signals)
                    - **Pattern Distance:** 15% from qualification
                    """)
                
                st.info("💡 **Tip**: Start with Balanced, then adjust based on market conditions and your risk tolerance.")
        
        # Convert timeframe_focus to wave_timeframe for backward compatibility
        wave_timeframe_map = {
            "🚀 Intraday (1D)": "Intraday Surge",
            "📈 Short-term (3D)": "3-Day Buildup", 
            "🌊 Medium-term (7D)": "Weekly Breakout",
            "💪 Long-term (30D)": "Monthly Trend",
            "🔄 Multi-timeframe": "All Waves"
        }
        wave_timeframe = wave_timeframe_map.get(timeframe_focus, "All Waves")
        
        # Initialize wave_filtered_df
        wave_filtered_df = radar_df.copy()
        
        if wave_timeframe != "All Waves":
            try:
                if wave_timeframe == "Intraday Surge":
                    required_cols = ['rvol', 'ret_1d', 'price', 'prev_close']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['rvol'] >= 2.5) &
                            (wave_filtered_df['ret_1d'] > 2) &
                            (wave_filtered_df['price'] > wave_filtered_df['prev_close'] * 1.02)
                        ]
                    
                elif wave_timeframe == "3-Day Buildup":
                    required_cols = ['ret_3d', 'vol_ratio_7d_90d', 'price', 'sma_20d']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_3d'] > 5) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 1.5) &
                            (wave_filtered_df['price'] > wave_filtered_df['sma_20d'])
                        ]
                
                elif wave_timeframe == "Weekly Breakout":
                    required_cols = ['ret_7d', 'vol_ratio_7d_90d', 'from_high_pct']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_7d'] > 8) &
                            (wave_filtered_df['vol_ratio_7d_90d'] > 2.0) &
                            (wave_filtered_df['from_high_pct'] > -10)
                        ]
                
                elif wave_timeframe == "Monthly Trend":
                    required_cols = ['ret_30d', 'vol_ratio_30d_180d', 'from_low_pct']
                    if all(col in wave_filtered_df.columns for col in required_cols):
                        wave_filtered_df = wave_filtered_df[
                            (wave_filtered_df['ret_30d'] > 15) &
                            (wave_filtered_df['vol_ratio_30d_180d'] > 1.2) &
                            (wave_filtered_df['from_low_pct'] > 30)
                        ]
            except Exception as e:
                logger.warning(f"Error applying {wave_timeframe} filter: {str(e)}")
                st.warning(f"Some data not available for {wave_timeframe} filter")
        
        if not wave_filtered_df.empty:
            st.markdown("#### 🚀 Momentum Shifts - Stocks Entering Strength")
            
            if "Conservative" in sensitivity_level:
                momentum_threshold = 60
                acceleration_threshold = 70
                min_rvol = 3.0
            elif "Balanced" in sensitivity_level:
                momentum_threshold = 50
                acceleration_threshold = 60
                min_rvol = 2.0
            else:
                momentum_threshold = 40
                acceleration_threshold = 50
                min_rvol = 1.5
            
            momentum_shifts = wave_filtered_df[
                (wave_filtered_df['momentum_score'] >= momentum_threshold) & 
                (wave_filtered_df['acceleration_score'] >= acceleration_threshold)
            ].copy()
            
            if len(momentum_shifts) > 0:
                momentum_shifts['signal_count'] = 0
                momentum_shifts.loc[momentum_shifts['momentum_score'] >= momentum_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['acceleration_score'] >= acceleration_threshold, 'signal_count'] += 1
                momentum_shifts.loc[momentum_shifts['rvol'] >= min_rvol, 'signal_count'] += 1
                if 'breakout_score' in momentum_shifts.columns:
                    momentum_shifts.loc[momentum_shifts['breakout_score'] >= 75, 'signal_count'] += 1
                if 'vol_ratio_7d_90d' in momentum_shifts.columns:
                    momentum_shifts.loc[momentum_shifts['vol_ratio_7d_90d'] >= 1.5, 'signal_count'] += 1
                
                momentum_shifts['shift_strength'] = (
                    momentum_shifts['momentum_score'] * 0.4 +
                    momentum_shifts['acceleration_score'] * 0.4 +
                    momentum_shifts['rvol_score'] * 0.2
                )
                
                top_shifts = momentum_shifts.sort_values(['signal_count', 'shift_strength'], ascending=[False, False]).head(20)
                
                display_columns = ['ticker', 'company_name', 'master_score', 'momentum_score', 
                                 'acceleration_score', 'rvol', 'signal_count', 'market_state']
                
                if 'ret_7d' in top_shifts.columns:
                    display_columns.insert(-2, 'ret_7d')
                
                display_columns.append('category')
                
                shift_display = top_shifts[[col for col in display_columns if col in top_shifts.columns]].copy()
                
                shift_display['Signals'] = shift_display['signal_count'].apply(
                    lambda x: f"{'🔥' * min(x, 3)} {x}/5"
                )
                
                if 'ret_7d' in shift_display.columns:
                    shift_display['7D Return'] = shift_display['ret_7d'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else '-')
                
                if 'rvol' in shift_display.columns:
                    shift_display['RVOL'] = shift_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                    shift_display = shift_display.drop('rvol', axis=1)
                
                rename_dict = {
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    'master_score': 'Score',
                    'momentum_score': 'Momentum',
                    'acceleration_score': 'Acceleration',
                    'market_state': 'market_state',
                    'category': 'Category'
                }
                
                shift_display = shift_display.rename(columns=rename_dict)
                
                if 'signal_count' in shift_display.columns:
                    shift_display = shift_display.drop('signal_count', axis=1)
                
                # OPTIMIZED DATAFRAME WITH COLUMN_CONFIG
                st.dataframe(
                    shift_display, 
                    width="stretch", 
                    hide_index=True,
                    column_config={
                        'Ticker': st.column_config.TextColumn(
                            'Ticker',
                            help="Stock symbol",
                            width="small"
                        ),
                        'Company': st.column_config.TextColumn(
                            'Company',
                            help="Company name",
                            width="medium"
                        ),
                        'Score': st.column_config.ProgressColumn(
                            'Score',
                            help="Master Score",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'Momentum': st.column_config.ProgressColumn(
                            'Momentum',
                            help="Momentum Score",
                            format="%.0f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'Acceleration': st.column_config.ProgressColumn(
                            'Acceleration',
                            help="Acceleration Score",
                            format="%.0f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'RVOL': st.column_config.TextColumn(
                            'RVOL',
                            help="Relative Volume",
                            width="small"
                        ),
                        'Signals': st.column_config.TextColumn(
                            'Signals',
                            help="Signal strength indicator",
                            width="small"
                        ),
                        '7D Return': st.column_config.TextColumn(
                            '7D Return',
                            help="7-day return percentage",
                            width="small"
                        ),
                        'Market State': st.column_config.TextColumn(
                            'Market State',
                            help="Current market state",
                            width="medium"
                        ),
                        'Category': st.column_config.TextColumn(
                            'Category',
                            help="Market cap category",
                            width="medium"
                        )
                    }
                )
                
                multi_signal = len(top_shifts[top_shifts['signal_count'] >= 3])
                if multi_signal > 0:
                    st.success(f"🏆 Found {multi_signal} stocks with 3+ signals (strongest momentum)")
                
                super_signals = top_shifts[top_shifts['signal_count'] >= 4]
                if len(super_signals) > 0:
                    st.warning(f"🔥🔥 {len(super_signals)} stocks showing EXTREME momentum (4+ signals)!")
            else:
                st.info(f"No momentum shifts detected in {wave_timeframe} timeframe. Try 'Aggressive' sensitivity.")
            
            st.markdown("#### 🚀 Acceleration Profiles - Momentum Building Over Time")
            
            if "Conservative" in sensitivity_level:
                accel_threshold = 85
            elif "Balanced" in sensitivity_level:
                accel_threshold = 70
            else:
                accel_threshold = 60
            
            accelerating_stocks = wave_filtered_df[
                wave_filtered_df['acceleration_score'] >= accel_threshold
            ].nlargest(10, 'acceleration_score')
            
            if len(accelerating_stocks) > 0:
                fig_accel = Visualizer.create_acceleration_profiles(accelerating_stocks, n=10)
                st.plotly_chart(fig_accel, width="stretch", theme="streamlit")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    perfect_accel = len(accelerating_stocks[accelerating_stocks['acceleration_score'] >= 90])
                    st.metric("Perfect Acceleration (90+)", perfect_accel)
                with col2:
                    strong_accel = len(accelerating_stocks[accelerating_stocks['acceleration_score'] >= 80])
                    st.metric("Strong Acceleration (80+)", strong_accel)
                with col3:
                    avg_accel = accelerating_stocks['acceleration_score'].mean()
                    st.metric("Avg Acceleration Score", f"{avg_accel:.1f}")
            else:
                st.info(f"No stocks meet the acceleration threshold ({accel_threshold}+) for {sensitivity_level} sensitivity.")
            
            if show_market_regime:
                st.markdown("#### 💰 Category Rotation - Smart Money Flow")
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    try:
                        if 'category' in wave_filtered_df.columns:
                            category_dfs = []
                            for cat in wave_filtered_df['category'].unique():
                                if cat != 'Unknown':
                                    cat_df = wave_filtered_df[wave_filtered_df['category'] == cat]
                                    
                                    category_size = len(cat_df)
                                    if category_size == 0: 
                                        continue  
                                    if 1 <= category_size <= 5:
                                        sample_count = category_size
                                    elif 6 <= category_size <= 20:
                                        sample_count = max(1, int(category_size * 0.80))
                                    elif 21 <= category_size <= 50:
                                        sample_count = max(1, int(category_size * 0.60))
                                    else:
                                        sample_count = min(50, int(category_size * 0.25))
                                    
                                    if sample_count > 0:
                                        cat_df = cat_df.nlargest(sample_count, 'master_score')
                                    else:
                                        cat_df = pd.DataFrame()
                                        
                                    if not cat_df.empty:
                                        category_dfs.append(cat_df)
                            
                            if category_dfs:
                                normalized_cat_df = pd.concat(category_dfs, ignore_index=True)
                            else:
                                normalized_cat_df = pd.DataFrame()
                            
                            if not normalized_cat_df.empty:
                                category_flow = normalized_cat_df.groupby('category').agg({
                                    'master_score': ['mean', 'count'],
                                    'momentum_score': 'mean',
                                    'volume_score': 'mean',
                                    'rvol': 'mean'
                                }).round(2)
                                
                                if not category_flow.empty:
                                    category_flow.columns = ['Avg Score', 'Count', 'Avg Momentum', 'Avg Volume', 'Avg RVOL']
                                    category_flow['Flow Score'] = (
                                        category_flow['Avg Score'] * 0.4 +
                                        category_flow['Avg Momentum'] * 0.3 +
                                        category_flow['Avg Volume'] * 0.3
                                    )
                                    
                                    category_flow = category_flow.sort_values('Flow Score', ascending=False)
                                    
                                    top_category = category_flow.index[0] if len(category_flow) > 0 else ""
                                    if 'Small' in top_category or 'Micro' in top_category:
                                        flow_direction = "🔥 RISK-ON"
                                    elif 'Large' in top_category or 'Mega' in top_category:
                                        flow_direction = "❄️ RISK-OFF"
                                    else:
                                        flow_direction = "➡️ Neutral"
                                    
                                    fig_flow = go.Figure()
                                    
                                    fig_flow.add_trace(go.Bar(
                                        x=category_flow.index,
                                        y=category_flow['Flow Score'],
                                        text=[f"{val:.1f}" for val in category_flow['Flow Score']],
                                        textposition='outside',
                                        marker_color=['#2ecc71' if score > 60 else '#e74c3c' if score < 40 else '#f39c12' 
                                                     for score in category_flow['Flow Score']],
                                        hovertemplate='Category: %{x}<br>Flow Score: %{y:.1f}<br>Stocks: %{customdata}<extra></extra>',
                                        customdata=category_flow['Count']
                                    ))
                                    
                                    fig_flow.update_layout(
                                        title=f"Smart Money Flow Direction: {flow_direction} (Dynamically Sampled)",
                                        xaxis_title="Market Cap Category",
                                        yaxis_title="Flow Score",
                                        height=300,
                                        template='plotly_white',
                                        showlegend=False
                                    )
                                    
                                    st.plotly_chart(fig_flow, width="stretch", theme="streamlit")
                                else:
                                    st.info("Insufficient data for category flow analysis after sampling.")
                            else:
                                st.info("No valid stocks found in categories for flow analysis after sampling.")
                        else:
                            st.info("Category data not available for flow analysis.")
                            
                    except Exception as e:
                        logger.error(f"Error in category flow analysis: {str(e)}")
                        st.error("Unable to analyze category flow")
                
                with col2:
                    if 'category_flow' in locals() and not category_flow.empty:
                        st.markdown(f"**🎯 Market Regime: {flow_direction}**")
                        
                        st.markdown("**💎 Strongest Categories:**")
                        for i, (cat, row) in enumerate(category_flow.head(3).iterrows()):
                            emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                            st.write(f"{emoji} **{cat}**: Score {row['Flow Score']:.1f}")
                        
                        st.markdown("**🔄 Category Shifts:**")
                        small_caps_score = category_flow[category_flow.index.str.contains('Small|Micro')]['Flow Score'].mean()
                        large_caps_score = category_flow[category_flow.index.str.contains('Large|Mega')]['Flow Score'].mean()
                        
                        if small_caps_score > large_caps_score + 10:
                            st.success("📈 Small Caps Leading - Early Bull Signal!")
                        elif large_caps_score > small_caps_score + 10:
                            st.warning("📉 Large Caps Leading - Defensive Mode")
                        else:
                            st.info("➡️ Balanced Market - No Clear Leader")
                    else:
                        st.info("Category data not available")
            
            st.markdown("#### 🎯 Emerging Patterns - About to Qualify")
            
            pattern_distance_map = {"Conservative": 5, "Balanced": 10, "Aggressive": 15}
            pattern_distance = 10  # Default
            for key in pattern_distance_map:
                if key in sensitivity_level:
                    pattern_distance = pattern_distance_map[key]
                    break
            
            emergence_data = []
            
            if 'category_percentile' in wave_filtered_df.columns:
                close_to_leader = wave_filtered_df[
                    (wave_filtered_df['category_percentile'] >= (90 - pattern_distance)) & 
                    (wave_filtered_df['category_percentile'] < 90)
                ]
                for _, stock in close_to_leader.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🐱 CAT LEADER',
                        'Distance': f"{90 - stock['category_percentile']:.1f}% away",
                        'Current': f"{stock['category_percentile']:.1f}%ile",
                        'Score': stock['master_score']
                    })
            
            if 'breakout_score' in wave_filtered_df.columns:
                close_to_breakout = wave_filtered_df[
                    (wave_filtered_df['breakout_score'] >= (80 - pattern_distance)) & 
                    (wave_filtered_df['breakout_score'] < 80)
                ]
                for _, stock in close_to_breakout.iterrows():
                    emergence_data.append({
                        'Ticker': stock['ticker'],
                        'Company': stock['company_name'],
                        'Pattern': '🎯 HIGH BREAKOUT SCORE',
                        'Distance': f"{80 - stock['breakout_score']:.1f} pts away",
                        'Current': f"{stock['breakout_score']:.1f} score",
                        'Score': stock['master_score']
                    })
            
            if emergence_data:
                emergence_df = pd.DataFrame(emergence_data).sort_values('Score', ascending=False).head(15)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    # OPTIMIZED DATAFRAME WITH COLUMN_CONFIG
                    st.dataframe(
                        emergence_df, 
                        width="stretch", 
                        hide_index=True,
                        column_config={
                            'Ticker': st.column_config.TextColumn(
                                'Ticker',
                                help="Stock symbol",
                                width="small"
                            ),
                            'Company': st.column_config.TextColumn(
                                'Company',
                                help="Company name",
                                width="medium"
                            ),
                            'Pattern': st.column_config.TextColumn(
                                'Pattern',
                                help="Pattern about to emerge",
                                width="medium"
                            ),
                            'Distance': st.column_config.TextColumn(
                                'Distance',
                                help="Distance from pattern qualification",
                                width="small"
                            ),
                            'Current': st.column_config.TextColumn(
                                'Current',
                                help="Current value",
                                width="small"
                            ),
                            'Score': st.column_config.ProgressColumn(
                                'Score',
                                help="Master Score",
                                format="%.1f",
                                min_value=0,
                                max_value=100,
                                width="small"
                            )
                        }
                    )
                with col2:
                    UIComponents.render_metric_card("Emerging Patterns", len(emergence_df))
            else:
                st.info(f"No patterns emerging within {pattern_distance}% threshold.")
            
            st.markdown("#### 🌊 Volume Surges - Unusual Activity NOW")
            
            rvol_threshold_map = {"Conservative": 3.0, "Balanced": 2.0, "Aggressive": 1.5}
            rvol_threshold = 2.0  # Default
            for key in rvol_threshold_map:
                if key in sensitivity_level:
                    rvol_threshold = rvol_threshold_map[key]
                    break
            
            volume_surges = wave_filtered_df[wave_filtered_df['rvol'] >= rvol_threshold].copy()
            
            if len(volume_surges) > 0:
                volume_surges['surge_score'] = (
                    volume_surges['rvol_score'] * 0.5 +
                    volume_surges['volume_score'] * 0.3 +
                    volume_surges['momentum_score'] * 0.2
                )
                
                top_surges = volume_surges.nlargest(15, 'surge_score')
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    display_cols = ['ticker', 'company_name', 'rvol', 'price', 'money_flow_mm', 'market_state', 'category']
                    
                    if 'ret_1d' in top_surges.columns:
                        display_cols.insert(3, 'ret_1d')
                    
                    surge_display = top_surges[[col for col in display_cols if col in top_surges.columns]].copy()
                    
                    surge_display['Type'] = surge_display['rvol'].apply(
                        lambda x: "🔥🔥🔥" if x > 5 else "🔥🔥" if x > 3 else "🔥"
                    )
                    
                    if 'ret_1d' in surge_display.columns:
                        surge_display['ret_1d'] = surge_display['ret_1d'].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else '-')
                    
                    if 'money_flow_mm' in surge_display.columns:
                        surge_display['money_flow_mm'] = surge_display['money_flow_mm'].apply(lambda x: f"₹{x:.1f}M" if pd.notna(x) else '-')
                    
                    surge_display['price'] = surge_display['price'].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-')
                    surge_display['rvol'] = surge_display['rvol'].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else '-')
                    
                    rename_dict = {
                        'ticker': 'Ticker',
                        'company_name': 'Company',
                        'rvol': 'RVOL',
                        'price': 'Price',
                        'money_flow_mm': 'Money Flow',
                        'market_state': 'Market State',
                        'category': 'Category',
                        'ret_1d': '1D Ret'
                    }
                    surge_display = surge_display.rename(columns=rename_dict)
                    
                    # OPTIMIZED DATAFRAME WITH COLUMN_CONFIG
                    st.dataframe(
                        surge_display, 
                        width="stretch", 
                        hide_index=True,
                        column_config={
                            'Type': st.column_config.TextColumn(
                                'Type',
                                help="Volume surge intensity",
                                width="small"
                            ),
                            'Ticker': st.column_config.TextColumn(
                                'Ticker',
                                help="Stock symbol",
                                width="small"
                            ),
                            'Company': st.column_config.TextColumn(
                                'Company',
                                help="Company name",
                                width="medium"
                            ),
                            'RVOL': st.column_config.TextColumn(
                                'RVOL',
                                help="Relative Volume",
                                width="small"
                            ),
                            'Price': st.column_config.TextColumn(
                                'Price',
                                help="Current price",
                                width="small"
                            ),
                            '1D Ret': st.column_config.TextColumn(
                                '1D Ret',
                                help="1-day return",
                                width="small"
                            ),
                            'Money Flow': st.column_config.TextColumn(
                                'Money Flow',
                                help="Money flow in millions",
                                width="small"
                            ),
                            'Market State': st.column_config.TextColumn(
                                'Market State',
                                help="Current market state",
                                width="medium"
                            ),
                            'Category': st.column_config.TextColumn(
                                'Category',
                                help="Market cap category",
                                width="medium"
                            )
                        }
                    )
                
                with col2:
                    UIComponents.render_metric_card("Active Surges", len(volume_surges))
                    UIComponents.render_metric_card("Extreme (>5x)", len(volume_surges[volume_surges['rvol'] > 5]))
                    UIComponents.render_metric_card("High (>3x)", len(volume_surges[volume_surges['rvol'] > 3]))
                    
                    if 'category' in volume_surges.columns:
                        st.markdown("**📊 Surge by Category:**")
                        surge_categories = volume_surges['category'].value_counts()
                        if len(surge_categories) > 0:
                            for cat, count in surge_categories.head(3).items():
                                st.caption(f"• {cat}: {count} stocks")
            else:
                st.info(f"No volume surges detected with {sensitivity_level} sensitivity (requires RVOL ≥ {min_rvol}x).")
        
        # ================================================================================================
        # 🎯 ADVANCED ANALYSIS TABS - 6 SPECIALIZED INTELLIGENCE MODULES
        # ================================================================================================
        
        st.markdown("---")
        st.markdown("### 🎯 SPECIALIZED MARKET INTELLIGENCE MODULES")
        
        radar_analysis_tabs = st.tabs([
            "🚀 Momentum Surge", 
            "🎯 Pattern Recognition", 
            "🏦 Institutional Flow",
            "⚡ Breakout Scanner", 
            "🎲 High Probability", 
            "🔄 Multi-Timeframe"
        ])
        
        # Tab 1: Momentum Surge Analysis
        with radar_analysis_tabs[0]:
            st.markdown("#### 🚀 Momentum Surge Detection")
            
            try:
                momentum_criteria = {
                    "Conservative": {"score": 70, "accel": 80, "rvol": 2.5},
                    "Balanced": {"score": 60, "accel": 70, "rvol": 2.0},
                    "Aggressive": {"score": 50, "accel": 60, "rvol": 1.5}
                }
                
                current_criteria = momentum_criteria.get("Balanced")
                for key in momentum_criteria:
                    if key in sensitivity_level:
                        current_criteria = momentum_criteria[key]
                        break
                
                momentum_stocks = radar_df[
                    (radar_df.get('momentum_score', 0) >= current_criteria["score"]) &
                    (radar_df.get('acceleration_score', 0) >= current_criteria["accel"]) &
                    (radar_df.get('rvol', 0) >= current_criteria["rvol"])
                ].copy()
                
                if not momentum_stocks.empty:
                    # 🚨 CRITICAL FIX: Proper momentum strength calculation with normalized values
                    momentum_stocks['momentum_strength'] = (
                        momentum_stocks.get('momentum_score', 0) * 0.5 +
                        momentum_stocks.get('acceleration_score', 0) * 0.3 +
                        np.clip(momentum_stocks.get('rvol', 0) * 4, 0, 20)  # Cap RVOL contribution at 20 points
                    )
                    
                    top_momentum = momentum_stocks.nlargest(15, 'momentum_strength')
                    
                    # Display metrics
                    momentum_cols = st.columns(4)
                    with momentum_cols[0]:
                        UIComponents.render_metric_card("Total Momentum", len(momentum_stocks))
                    with momentum_cols[1]:
                        avg_score = top_momentum['momentum_score'].mean() if 'momentum_score' in top_momentum.columns and len(top_momentum) > 0 else 0
                        UIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}")
                    with momentum_cols[2]:
                        avg_rvol = top_momentum['rvol'].mean() if 'rvol' in top_momentum.columns and len(top_momentum) > 0 else 0
                        UIComponents.render_metric_card("Avg RVOL", f"{avg_rvol:.1f}x")
                    with momentum_cols[3]:
                        strength_rating = "🔥🔥🔥" if avg_score > 70 else "🔥🔥" if avg_score > 60 else "🔥"
                        UIComponents.render_metric_card("Strength", strength_rating)
                    
                    # Display dataframe
                    display_cols = ['ticker', 'company_name', 'momentum_score', 'acceleration_score', 'rvol', 'momentum_strength']
                    display_momentum = top_momentum[[col for col in display_cols if col in top_momentum.columns]]
                    
                    st.dataframe(
                        display_momentum,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'ticker': st.column_config.TextColumn("Ticker", width="small"),
                            'company_name': st.column_config.TextColumn("Company", width="medium"),
                            'momentum_score': st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.1f"),
                            'acceleration_score': st.column_config.ProgressColumn("Acceleration", min_value=0, max_value=100, format="%.1f"),
                            'rvol': st.column_config.NumberColumn("RVOL", format="%.1fx"),
                            'momentum_strength': st.column_config.ProgressColumn("Strength", min_value=0, max_value=120, format="%.1f")
                        }
                    )
                else:
                    st.info("No momentum surge stocks found with current criteria")
                    
            except Exception as e:
                logger.error(f"Momentum surge analysis error: {str(e)}")
                st.error("Error in momentum surge analysis")
        
        # Tab 2: Pattern Recognition
        with radar_analysis_tabs[1]:
            st.markdown("#### 🎯 Advanced Pattern Recognition")
            
            try:
                pattern_analysis = radar_df.copy()
                
                # 🚨 CRITICAL FIX: Use sensitivity-based pattern thresholds
                pattern_thresholds = {
                    'breakout': thresholds['pattern'],
                    'reversal': thresholds['pattern'],
                    'trend': thresholds['trend']
                }
                
                st.info(f"🎯 Using {sensitivity_level} pattern thresholds: Breakout/Reversal ≥{pattern_thresholds['breakout']}%, Trend ≥{pattern_thresholds['trend']}%")
                
                # Pattern scoring system
                pattern_signals = []
                
                # Breakout patterns
                if 'breakout_score' in pattern_analysis.columns:
                    breakout_stocks = pattern_analysis[pattern_analysis['breakout_score'] >= pattern_thresholds['breakout']]
                    for _, stock in breakout_stocks.iterrows():
                        reliability_threshold = pattern_thresholds['breakout'] + 15
                        pattern_signals.append({
                            'Ticker': stock['ticker'],
                            'Company': stock.get('company_name', 'Unknown'),
                            'Pattern': '📈 Breakout',
                            'Score': stock['breakout_score'],
                            'Reliability': 'High' if stock['breakout_score'] > reliability_threshold else 'Medium'
                        })
                
                # Reversal patterns
                if 'reversal_score' in pattern_analysis.columns:
                    reversal_stocks = pattern_analysis[pattern_analysis['reversal_score'] >= pattern_thresholds['reversal']]
                    for _, stock in reversal_stocks.iterrows():
                        reliability_threshold = pattern_thresholds['reversal'] + 15
                        pattern_signals.append({
                            'Ticker': stock['ticker'],
                            'Company': stock.get('company_name', 'Unknown'),
                            'Pattern': '🔄 Reversal',
                            'Score': stock['reversal_score'],
                            'Reliability': 'High' if stock['reversal_score'] > reliability_threshold else 'Medium'
                        })
                
                # Continuation patterns
                if 'trend_score' in pattern_analysis.columns:
                    trend_stocks = pattern_analysis[pattern_analysis['trend_score'] >= pattern_thresholds['trend']]
                    for _, stock in trend_stocks.iterrows():
                        reliability_threshold = pattern_thresholds['trend'] + 10
                        pattern_signals.append({
                            'Ticker': stock['ticker'],
                            'Company': stock.get('company_name', 'Unknown'),
                            'Pattern': '➡️ Continuation',
                            'Score': stock['trend_score'],
                            'Reliability': 'High' if stock['trend_score'] > reliability_threshold else 'Medium'
                        })
                
                if pattern_signals:
                    pattern_df = pd.DataFrame(pattern_signals)
                    
                    # Pattern statistics
                    pattern_cols = st.columns(4)
                    with pattern_cols[0]:
                        UIComponents.render_metric_card("Total Patterns", len(pattern_df))
                    with pattern_cols[1]:
                        high_rel = len(pattern_df[pattern_df['Reliability'] == 'High'])
                        UIComponents.render_metric_card("High Reliability", high_rel)
                    with pattern_cols[2]:
                        avg_score = pattern_df['Score'].mean()
                        UIComponents.render_metric_card("Avg Score", f"{avg_score:.1f}")
                    with pattern_cols[3]:
                        top_pattern = pattern_df['Pattern'].mode().iloc[0] if not pattern_df.empty else "None"
                        UIComponents.render_metric_card("Top Pattern", top_pattern)
                    
                    # Display patterns
                    st.dataframe(
                        pattern_df.sort_values('Score', ascending=False),
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'Ticker': st.column_config.TextColumn("Ticker", width="small"),
                            'Company': st.column_config.TextColumn("Company", width="medium"),
                            'Pattern': st.column_config.TextColumn("Pattern", width="medium"),
                            'Score': st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                            'Reliability': st.column_config.TextColumn("Reliability", width="small")
                        }
                    )
                else:
                    st.info("No significant patterns detected")
                    
            except Exception as e:
                logger.error(f"Pattern recognition error: {str(e)}")
                st.error("Error in pattern recognition analysis")
        
        # Tab 3: Institutional Flow Analysis
        with radar_analysis_tabs[2]:
            st.markdown("#### 🏦 Institutional Flow Intelligence")
            
            try:
                institutional_analysis = radar_df.copy()
                
                # 🚨 CRITICAL FIX: Use sensitivity-based flow thresholds
                flow_threshold = thresholds['institutional']
                rvol_threshold = 3.0 if "Conservative" in sensitivity_level else 2.5 if "Balanced" in sensitivity_level else 2.0
                
                st.info(f"🎯 Using {sensitivity_level} flow sensitivity: Volume ≥{rvol_threshold}x, Institutional threshold: {flow_threshold}%")
                
                # Money flow analysis
                flow_signals = []
                
                if 'money_flow_mm' in institutional_analysis.columns:
                    # Use sensitivity-based percentile threshold
                    percentile_threshold = 0.9 if "Conservative" in sensitivity_level else 0.8 if "Balanced" in sensitivity_level else 0.7
                    flow_cutoff = institutional_analysis['money_flow_mm'].quantile(percentile_threshold)
                    
                    large_flows = institutional_analysis[
                        abs(institutional_analysis['money_flow_mm']) >= abs(flow_cutoff)
                    ].copy()
                    
                    for _, stock in large_flows.iterrows():
                        flow_value = stock.get('money_flow_mm', 0)
                        flow_type = "🟢 Inflow" if flow_value > 0 else "🔴 Outflow"
                        extreme_threshold = abs(flow_cutoff) * 1.5
                        
                        flow_signals.append({
                            'Ticker': stock['ticker'],
                            'Company': stock.get('company_name', 'Unknown'),
                            'Flow Type': flow_type,
                            'Amount': f"₹{abs(flow_value):.1f}M",
                            'Intensity': 'Extreme' if abs(flow_value) > extreme_threshold else 'High'
                        })
                
                # Volume-based institutional detection with sensitivity threshold
                if 'rvol' in institutional_analysis.columns:
                    inst_volume = institutional_analysis[institutional_analysis['rvol'] >= rvol_threshold].copy()
                    
                    for _, stock in inst_volume.iterrows():
                        if stock['ticker'] not in [s['Ticker'] for s in flow_signals]:
                            extreme_rvol_threshold = rvol_threshold * 2
                            flow_signals.append({
                                'Ticker': stock['ticker'],
                                'Company': stock.get('company_name', 'Unknown'),
                                'Flow Type': '📊 Volume Inst.',
                                'Amount': f"{stock['rvol']:.1f}x RVOL",
                                'Intensity': 'Extreme' if stock['rvol'] > extreme_rvol_threshold else 'High'
                            })
                
                if flow_signals:
                    flow_df = pd.DataFrame(flow_signals)
                    
                    # Flow statistics
                    flow_cols = st.columns(4)
                    with flow_cols[0]:
                        UIComponents.render_metric_card("Total Signals", len(flow_df))
                    with flow_cols[1]:
                        inflows = len(flow_df[flow_df['Flow Type'].str.contains('Inflow')])
                        UIComponents.render_metric_card("Inflows", inflows)
                    with flow_cols[2]:
                        outflows = len(flow_df[flow_df['Flow Type'].str.contains('Outflow')])
                        UIComponents.render_metric_card("Outflows", outflows)
                    with flow_cols[3]:
                        extreme_flows = len(flow_df[flow_df['Intensity'] == 'Extreme'])
                        UIComponents.render_metric_card("Extreme", extreme_flows)
                    
                    # Flow direction indicator
                    if inflows > outflows + 2:
                        st.success("🟢 **Net Institutional Buying** - Bullish sentiment")
                    elif outflows > inflows + 2:
                        st.error("🔴 **Net Institutional Selling** - Bearish sentiment")
                    else:
                        st.info("⚖️ **Balanced Flow** - Mixed sentiment")
                    
                    # Display flows
                    st.dataframe(
                        flow_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'Ticker': st.column_config.TextColumn("Ticker", width="small"),
                            'Company': st.column_config.TextColumn("Company", width="medium"),
                            'Flow Type': st.column_config.TextColumn("Flow Type", width="medium"),
                            'Amount': st.column_config.TextColumn("Amount", width="small"),
                            'Intensity': st.column_config.TextColumn("Intensity", width="small")
                        }
                    )
                else:
                    st.info("No significant institutional flow detected")
                    
            except Exception as e:
                logger.error(f"Institutional flow analysis error: {str(e)}")
                st.error("Error in institutional flow analysis")
        
        # Tab 4: Breakout Scanner
        with radar_analysis_tabs[3]:
            st.markdown("#### ⚡ Advanced Breakout Scanner")
            
            try:
                breakout_analysis = radar_df.copy()
                
                # 🚨 CRITICAL FIX: Use sensitivity-based breakout thresholds
                breakout_threshold = thresholds['breakout']
                price_proximity = 0.98 if "Conservative" in sensitivity_level else 0.96 if "Balanced" in sensitivity_level else 0.94
                volume_threshold = 2.5 if "Conservative" in sensitivity_level else 2.0 if "Balanced" in sensitivity_level else 1.5
                
                st.info(f"🎯 Using {sensitivity_level} breakout criteria: Price proximity ≥{price_proximity*100:.0f}%, Volume ≥{volume_threshold}x")
                
                # Multi-timeframe breakout detection
                breakout_candidates = []
                
                # Price-based breakouts with sensitivity adjustment
                if all(col in breakout_analysis.columns for col in ['price', 'high_20d', 'high_50d']):
                    price_breakouts = breakout_analysis[
                        (breakout_analysis['price'] >= breakout_analysis['high_20d'] * price_proximity) |
                        (breakout_analysis['price'] >= breakout_analysis['high_50d'] * (price_proximity - 0.02))
                    ].copy()
                    
                    for _, stock in price_breakouts.iterrows():
                        # Calculate precise distance metrics
                        high_20_distance = (stock['price'] / stock['high_20d'] - 1) * 100 if 'high_20d' in stock and pd.notna(stock['high_20d']) else -999
                        high_50_distance = (stock['price'] / stock['high_50d'] - 1) * 100 if 'high_50d' in stock and pd.notna(stock['high_50d']) else -999
                        
                        # Determine breakout type
                        if high_20_distance >= (price_proximity - 1) * 100:
                            breakout_type = "🚀 20D High"
                            distance = high_20_distance
                        elif high_50_distance >= (price_proximity - 1.02) * 100:
                            breakout_type = "📈 50D High"
                            distance = high_50_distance
                        else:
                            continue
                        
                        # Apply breakout threshold if available
                        if 'breakout_score' in stock and pd.notna(stock['breakout_score']):
                            if stock['breakout_score'] < breakout_threshold:
                                continue
                        
                        strength = 'Strong' if distance > 2 else 'Emerging'
                        
                        breakout_candidates.append({
                            'Ticker': stock['ticker'],
                            'Company': stock.get('company_name', 'Unknown'),
                            'Breakout Type': breakout_type,
                            'Distance': f"{distance:+.1f}%",
                            'Volume': f"{stock.get('rvol', 0):.1f}x" if 'rvol' in stock else "N/A",
                            'Strength': strength
                        })
                
                # Volume breakouts with sensitivity threshold
                if 'rvol' in breakout_analysis.columns:
                    volume_breakouts = breakout_analysis[breakout_analysis['rvol'] >= volume_threshold].copy()
                    
                    for _, stock in volume_breakouts.iterrows():
                        if stock['ticker'] not in [c['Ticker'] for c in breakout_candidates]:
                            vol_extreme_threshold = volume_threshold * 2
                            vol_strength = "🔥 Extreme" if stock['rvol'] > vol_extreme_threshold else "📊 High"
                            strength = 'Strong' if stock['rvol'] > vol_extreme_threshold else 'Medium'
                            
                            breakout_candidates.append({
                                'Ticker': stock['ticker'],
                                'Company': stock.get('company_name', 'Unknown'),
                                'Breakout Type': f'{vol_strength} Volume',
                                'Distance': f"{stock['rvol']:.1f}x",
                                'Volume': f"{stock['rvol']:.1f}x",
                                'Strength': strength
                            })
                
                if breakout_candidates:
                    breakout_df = pd.DataFrame(breakout_candidates)
                    
                    # Breakout statistics
                    breakout_cols = st.columns(4)
                    with breakout_cols[0]:
                        UIComponents.render_metric_card("Total Breakouts", len(breakout_df))
                    with breakout_cols[1]:
                        strong_breakouts = len(breakout_df[breakout_df['Strength'] == 'Strong'])
                        UIComponents.render_metric_card("Strong", strong_breakouts)
                    with breakout_cols[2]:
                        price_breakouts_count = len(breakout_df[breakout_df['Breakout Type'].str.contains('High')])
                        UIComponents.render_metric_card("Price Breakouts", price_breakouts_count)
                    with breakout_cols[3]:
                        volume_breakouts_count = len(breakout_df[breakout_df['Breakout Type'].str.contains('Volume')])
                        UIComponents.render_metric_card("Volume Breakouts", volume_breakouts_count)
                    
                    # Display breakouts
                    st.dataframe(
                        breakout_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'Ticker': st.column_config.TextColumn("Ticker", width="small"),
                            'Company': st.column_config.TextColumn("Company", width="medium"),
                            'Breakout Type': st.column_config.TextColumn("Type", width="medium"),
                            'Distance': st.column_config.TextColumn("Distance", width="small"),
                            'Volume': st.column_config.TextColumn("Volume", width="small"),
                            'Strength': st.column_config.TextColumn("Strength", width="small")
                        }
                    )
                else:
                    st.info("No breakout candidates detected")
                    
            except Exception as e:
                logger.error(f"Breakout scanner error: {str(e)}")
                st.error("Error in breakout scanner")
        
        # Tab 5: High Probability Signals
        with radar_analysis_tabs[4]:
            st.markdown("#### 🎲 High Probability Signal Generator")
            
            try:
                # Multi-factor scoring for high probability signals
                high_prob_analysis = radar_df.copy()
                
                # 🚨 CRITICAL FIX: Use sensitivity-based probability thresholds
                prob_threshold = thresholds['probability']
                st.info(f"🎯 Using {sensitivity_level} probability threshold: ≥{prob_threshold}%")
                
                # Calculate composite probability score
                prob_factors = []
                
                # Technical score (if available)
                if 'master_score' in high_prob_analysis.columns:
                    prob_factors.append(('Technical', 'master_score', 0.3))
                
                # Momentum factor
                if 'momentum_score' in high_prob_analysis.columns:
                    prob_factors.append(('Momentum', 'momentum_score', 0.25))
                
                # Volume factor with sensitivity adjustment - 🚨 FIXED: Proper volume scoring
                if 'rvol' in high_prob_analysis.columns:
                    volume_multiplier = 8 if "Conservative" in sensitivity_level else 10 if "Balanced" in sensitivity_level else 12
                    high_prob_analysis['volume_score'] = np.clip(high_prob_analysis['rvol'] * volume_multiplier, 0, 100)
                    prob_factors.append(('Volume', 'volume_score', 0.25))
                
                # Trend factor
                if 'trend_score' in high_prob_analysis.columns:
                    prob_factors.append(('Trend', 'trend_score', 0.2))
                
                # Calculate probability score
                if prob_factors:
                    high_prob_analysis['probability_score'] = 0
                    for name, col, weight in prob_factors:
                        if col in high_prob_analysis.columns:
                            high_prob_analysis['probability_score'] += high_prob_analysis[col] * weight
                    
                    # Filter high probability stocks using sensitivity threshold
                    high_prob_stocks = high_prob_analysis[
                        high_prob_analysis['probability_score'] >= prob_threshold
                    ].copy()
                    
                    if not high_prob_stocks.empty:
                        # Add confidence intervals with sensitivity-aware thresholds
                        very_high_threshold = prob_threshold + 15
                        high_threshold = prob_threshold + 10
                        good_threshold = prob_threshold + 5
                        
                        high_prob_stocks['confidence'] = high_prob_stocks['probability_score'].apply(
                            lambda x: 'Very High' if x > very_high_threshold 
                            else 'High' if x > high_threshold 
                            else 'Good' if x > good_threshold 
                            else 'Moderate'
                        )
                        
                        top_prob = high_prob_stocks.nlargest(20, 'probability_score')
                        
                        # Probability statistics
                        prob_cols = st.columns(4)
                        with prob_cols[0]:
                            UIComponents.render_metric_card("High Prob Signals", len(high_prob_stocks))
                        with prob_cols[1]:
                            very_high = len(high_prob_stocks[high_prob_stocks['confidence'] == 'Very High'])
                            UIComponents.render_metric_card("Very High", very_high)
                        with prob_cols[2]:
                            avg_prob = top_prob['probability_score'].mean()
                            UIComponents.render_metric_card("Avg Probability", f"{avg_prob:.1f}%")
                        with prob_cols[3]:
                            success_rate = min(avg_prob * 1.1, 95)  # Estimated success rate
                            UIComponents.render_metric_card("Est. Success", f"{success_rate:.0f}%")
                        
                        # Display high probability signals
                        display_cols = ['ticker', 'company_name', 'probability_score', 'confidence']
                        if 'master_score' in top_prob.columns:
                            display_cols.append('master_score')
                        if 'rvol' in top_prob.columns:
                            display_cols.append('rvol')
                        
                        prob_display = top_prob[[col for col in display_cols if col in top_prob.columns]]
                        
                        st.dataframe(
                            prob_display,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'ticker': st.column_config.TextColumn("Ticker", width="small"),
                                'company_name': st.column_config.TextColumn("Company", width="medium"),
                                'probability_score': st.column_config.ProgressColumn("Probability", min_value=0, max_value=100, format="%.1f"),
                                'confidence': st.column_config.TextColumn("Confidence", width="small"),
                                'master_score': st.column_config.ProgressColumn("Technical", min_value=0, max_value=100, format="%.1f"),
                                'rvol': st.column_config.NumberColumn("RVOL", format="%.1fx")
                            }
                        )
                        
                        # Risk warning
                        st.warning("⚠️ **Risk Warning**: High probability signals are statistical indicators. Always use proper risk management and position sizing.")
                        
                    else:
                        st.info(f"No high probability signals found with {sensitivity_level} criteria (≥{prob_threshold}%)")
                else:
                    st.info("Insufficient data for probability analysis")
                    
            except Exception as e:
                logger.error(f"High probability analysis error: {str(e)}")
                st.error("Error in high probability analysis")
        
        # Tab 6: Multi-Timeframe Analysis
        with radar_analysis_tabs[5]:
            st.markdown("#### 🔄 Multi-Timeframe Confluence Analysis")
            
            try:
                # 🚨 CRITICAL FIX: Use sensitivity-based confluence threshold
                confluence_threshold = thresholds['confluence']
                st.info(f"🎯 Using {sensitivity_level} confluence threshold: {confluence_threshold}%")
                
                # Enhanced confluence calculation with better data handling
                confluence_signals = []
                
                for _, stock in radar_df.iterrows():
                    short_score = 0
                    medium_score = 0
                    total_factors = 0
                    
                    # Short-term factors (with fallbacks)
                    if 'ret_1d' in stock and pd.notna(stock.get('ret_1d')):
                        short_score += 30 if stock['ret_1d'] > 3 else 20 if stock['ret_1d'] > 1 else 10 if stock['ret_1d'] > 0 else 0
                        total_factors += 1
                        
                    if 'ret_3d' in stock and pd.notna(stock.get('ret_3d')):
                        short_score += 35 if stock['ret_3d'] > 7 else 25 if stock['ret_3d'] > 3 else 15 if stock['ret_3d'] > 0 else 0
                        total_factors += 1
                        
                    if 'momentum_score' in stock and pd.notna(stock.get('momentum_score')):
                        short_score += stock['momentum_score'] * 0.6
                        total_factors += 1
                        
                    # Medium-term factors (with fallbacks)
                    if 'ret_7d' in stock and pd.notna(stock.get('ret_7d')):
                        medium_score += 40 if stock['ret_7d'] > 15 else 30 if stock['ret_7d'] > 7 else 20 if stock['ret_7d'] > 0 else 0
                        total_factors += 1
                        
                    if 'ret_30d' in stock and pd.notna(stock.get('ret_30d')):
                        medium_score += 45 if stock['ret_30d'] > 30 else 35 if stock['ret_30d'] > 15 else 25 if stock['ret_30d'] > 0 else 0
                        total_factors += 1
                        
                    if 'trend_score' in stock and pd.notna(stock.get('trend_score')):
                        medium_score += stock['trend_score'] * 0.5
                        total_factors += 1
                    
                    # Additional scoring factors for better results
                    if 'acceleration_score' in stock and pd.notna(stock.get('acceleration_score')):
                        short_score += stock['acceleration_score'] * 0.3
                        total_factors += 1
                        
                    if 'rvol' in stock and pd.notna(stock.get('rvol')):
                        volume_boost = min(stock['rvol'] * 8, 30)  # 🚨 FIXED: Reduced multiplier and cap
                        short_score += volume_boost
                        total_factors += 1
                    
                    # Only include stocks with minimum data availability
                    if total_factors >= 3:
                        # Calculate normalized scores
                        short_term_final = short_score / max(1, (total_factors * 0.6))
                        medium_term_final = medium_score / max(1, (total_factors * 0.4))
                        
                        # Calculate confluence with timeframe weighting
                        confluence_score = (short_term_final * 0.6 + medium_term_final * 0.4)
                        
                        # Alignment calculation
                        alignment_diff = abs(short_term_final - medium_term_final)
                        alignment = 'Strong' if alignment_diff < 25 else 'Medium' if alignment_diff < 40 else 'Weak'
                        
                        # Apply sensitivity-based threshold
                        if confluence_score >= confluence_threshold:
                            confluence_signals.append({
                                'ticker': stock['ticker'],
                                'company_name': stock.get('company_name', 'Unknown'),
                                'short_term_score': round(short_term_final, 1),
                                'medium_term_score': round(medium_term_final, 1),
                                'confluence_score': round(confluence_score, 1),
                                'alignment': alignment,
                                'data_factors': total_factors
                            })
                
                if confluence_signals:
                    confluence_df = pd.DataFrame(confluence_signals)
                    confluence_df = confluence_df.sort_values('confluence_score', ascending=False)
                    
                    # Enhanced statistics
                    mtf_cols = st.columns(4)
                    with mtf_cols[0]:
                        UIComponents.render_metric_card("Confluence Signals", len(confluence_df))
                    with mtf_cols[1]:
                        strong_alignment = len(confluence_df[confluence_df['alignment'] == 'Strong'])
                        UIComponents.render_metric_card("Strong Alignment", strong_alignment)
                    with mtf_cols[2]:
                        avg_confluence = confluence_df['confluence_score'].mean()
                        UIComponents.render_metric_card("Avg Confluence", f"{avg_confluence:.1f}%")
                    with mtf_cols[3]:
                        max_confluence = confluence_df['confluence_score'].max()
                        UIComponents.render_metric_card("Max Confluence", f"{max_confluence:.1f}%")
                    
                    # Data quality indicator
                    avg_factors = confluence_df['data_factors'].mean()
                    if avg_factors >= 6:
                        data_quality = "🟢 Excellent"
                    elif avg_factors >= 4:
                        data_quality = "🟡 Good"
                    else:
                        data_quality = "🟠 Limited"
                    
                    st.info(f"📊 Data Quality: {data_quality} (Avg {avg_factors:.1f} factors per stock)")
                    
                    # Enhanced dataframe display
                    display_df = confluence_df.drop('data_factors', axis=1)  # Hide technical column
                    
                    st.dataframe(
                        display_df,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'ticker': st.column_config.TextColumn("Ticker", width="small"),
                            'company_name': st.column_config.TextColumn("Company", width="medium"),
                            'short_term_score': st.column_config.ProgressColumn("Short-term", min_value=0, max_value=100, format="%.1f"),
                            'medium_term_score': st.column_config.ProgressColumn("Medium-term", min_value=0, max_value=100, format="%.1f"),
                            'confluence_score': st.column_config.ProgressColumn("Confluence", min_value=0, max_value=100, format="%.1f"),
                            'alignment': st.column_config.TextColumn("Alignment", width="small")
                        }
                    )
                    
                    # Enhanced analysis feedback
                    strong_signals = len(confluence_df[confluence_df['alignment'] == 'Strong'])
                    total_signals = len(confluence_df)
                    
                    if strong_signals > total_signals * 0.7:
                        st.success("🎯 **Excellent Multi-Timeframe Alignment** - Very high conviction signals!")
                    elif strong_signals > total_signals * 0.5:
                        st.success("✅ **Good Multi-Timeframe Alignment** - High conviction signals")
                    elif strong_signals > total_signals * 0.3:
                        st.info("⚖️ **Mixed Timeframe Signals** - Moderate conviction, selective approach recommended")
                    else:
                        st.warning("⚠️ **Weak Timeframe Alignment** - Lower conviction, careful position sizing advised")
                    
                    # Sensitivity adjustment recommendations
                    if len(confluence_df) < 5:
                        st.info("💡 **Tip**: Try 'Aggressive' sensitivity for more signals, or 'Conservative' for higher quality")
                
                else:
                    # Enhanced "no signals" feedback with actionable advice
                    total_analyzed = len(radar_df)
                    st.warning(f"🔍 **No confluence signals detected** (Analyzed {total_analyzed} stocks with {confluence_threshold}% threshold)")
                    
                    # Provide actionable suggestions
                    st.markdown("**🛠 Troubleshooting Suggestions:**")
                    if "Conservative" in sensitivity_level:
                        st.info("• Try **'Balanced'** or **'Aggressive'** sensitivity for more signals")
                    if radar_mode != "📊 Full Spectrum":
                        st.info(f"• Current mode: **{radar_mode}** - Try **'📊 Full Spectrum'** for broader analysis")
                    if total_analyzed < 50:
                        st.info("• Current filters may be too restrictive - try adjusting Risk Profile or Market Regime")
                    
                    st.info("• Confluence analysis requires multiple timeframe data - some datasets may have limited historical returns")
                    
            except Exception as e:
                logger.error(f"Multi-timeframe analysis error: {str(e)}")
                st.error(f"Error in multi-timeframe analysis: {str(e)}")
                st.info("💡 This might be due to missing return columns (ret_1d, ret_3d, ret_7d, ret_30d) in your dataset")
        
        if filtered_df.empty:
            st.warning("No data available for Ultimate Market Radar analysis")

    # Tab 3: 🏆 ALL TIME BEST ANALYSIS TAB - INSTITUTIONAL GRADE INTELLIGENCE
    with tabs[3]:
        
        if not filtered_df.empty:
            # ================================================================================================
            # 🎯 EXECUTIVE DASHBOARD - TOP-LEVEL MARKET INTELLIGENCE
            # ================================================================================================
            
            st.markdown("### 📊 **EXECUTIVE MARKET DASHBOARD**")
            
            # Calculate comprehensive market intelligence metrics
            total_analyzed = len(filtered_df)
            elite_threshold = 80
            strong_threshold = 70
            
            elite_stocks = len(filtered_df[filtered_df['master_score'] >= elite_threshold]) if 'master_score' in filtered_df.columns else 0
            strong_stocks = len(filtered_df[filtered_df['master_score'] >= strong_threshold]) if 'master_score' in filtered_df.columns else 0
            
            # Market health indicators
            market_health = "🔥 BULLISH" if elite_stocks > total_analyzed * 0.15 else "📈 POSITIVE" if strong_stocks > total_analyzed * 0.25 else "⚖️ NEUTRAL" if strong_stocks > total_analyzed * 0.15 else "⚠️ CAUTIOUS"
            
            # Executive metrics row
            exec_cols = st.columns(6)
            
            with exec_cols[0]:
                UIComponents.render_metric_card("Market Health", market_health, f"{strong_stocks} strong signals")
            
            with exec_cols[1]:
                elite_pct = (elite_stocks / total_analyzed * 100) if total_analyzed > 0 else 0
                UIComponents.render_metric_card("Elite Stocks", f"{elite_pct:.1f}%", f"{elite_stocks}/{total_analyzed}")
            
            with exec_cols[2]:
                avg_momentum = filtered_df['momentum_score'].mean() if 'momentum_score' in filtered_df.columns else 0
                momentum_rating = "🚀" if avg_momentum > 70 else "📈" if avg_momentum > 60 else "➡️"
                UIComponents.render_metric_card("Momentum", f"{avg_momentum:.1f}", momentum_rating)
            
            with exec_cols[3]:
                avg_volume = filtered_df['rvol'].mean() if 'rvol' in filtered_df.columns else 0
                volume_activity = "🔥 HIGH" if avg_volume > 2.0 else "📊 MEDIUM" if avg_volume > 1.5 else "📉 LOW"
                UIComponents.render_metric_card("Volume Activity", volume_activity, f"{avg_volume:.1f}x avg")
            
            with exec_cols[4]:
                breakout_count = len(filtered_df[filtered_df['breakout_score'] >= 70]) if 'breakout_score' in filtered_df.columns else 0
                breakout_pct = (breakout_count / total_analyzed * 100) if total_analyzed > 0 else 0
                UIComponents.render_metric_card("Breakouts", f"{breakout_pct:.1f}%", f"{breakout_count} stocks")
            
            with exec_cols[5]:
                risk_level = "🛡️ LOW" if avg_momentum > 65 and avg_volume < 3 else "⚠️ MEDIUM" if avg_volume < 4 else "🚨 HIGH"
                UIComponents.render_metric_card("Risk Level", risk_level, "Systematic")
            
            # ================================================================================================
            # 📈 ADVANCED VISUALIZATION SUITE - PROFESSIONAL CHARTS
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### 📊 **ADVANCED MARKET VISUALIZATION SUITE**")
            
            viz_tabs = st.tabs([
                "🎯 Score Analytics", 
                "📊 Performance Matrix", 
                "🔥 Momentum Heatmap",
                "💎 Pattern Intelligence",
                "🏢 Sector Analysis",
                "🏭 Industry Analysis",
                "⚡ Risk Dashboard"
            ])
            
            # Tab 1: Score Analytics
            with viz_tabs[0]:
                score_cols = st.columns(2)
                
                with score_cols[0]:
                    st.markdown("#### 📊 **Master Score Distribution**")
                    fig_dist = Visualizer.create_score_distribution(filtered_df)
                    st.plotly_chart(fig_dist, width='stretch', theme="streamlit")
                    
                    # Score quality analysis
                    if 'master_score' in filtered_df.columns:
                        score_stats = {
                            "Elite (80-100)": len(filtered_df[filtered_df['master_score'] >= 80]),
                            "Strong (70-79)": len(filtered_df[(filtered_df['master_score'] >= 70) & (filtered_df['master_score'] < 80)]),
                            "Good (60-69)": len(filtered_df[(filtered_df['master_score'] >= 60) & (filtered_df['master_score'] < 70)]),
                            "Average (50-59)": len(filtered_df[(filtered_df['master_score'] >= 50) & (filtered_df['master_score'] < 60)]),
                            "Below Avg (<50)": len(filtered_df[filtered_df['master_score'] < 50])
                        }
                        
                        st.markdown("**🎯 Score Quality Distribution:**")
                        for category, count in score_stats.items():
                            pct = (count / total_analyzed * 100) if total_analyzed > 0 else 0
                            st.write(f"• {category}: **{count}** stocks ({pct:.1f}%)")
                
                with score_cols[1]:
                    st.markdown("#### 🔥 **Component Score Analysis**")
                    
                    # Check if we have at least some score columns
                    available_score_cols = [col for col in ['momentum_score', 'acceleration_score', 'breakout_score', 'position_score', 'volume_score', 'rvol_score'] if col in filtered_df.columns]
                    
                    if len(available_score_cols) >= 2:  # Need at least 2 score columns
                        # Build component data dynamically based on available columns
                        components = []
                        scores = []
                        
                        if 'momentum_score' in filtered_df.columns:
                            components.append('Momentum')
                            scores.append(filtered_df['momentum_score'].mean())
                        
                        if 'acceleration_score' in filtered_df.columns:
                            components.append('Acceleration')
                            scores.append(filtered_df['acceleration_score'].mean())
                        
                        if 'breakout_score' in filtered_df.columns:
                            components.append('Breakout')
                            scores.append(filtered_df['breakout_score'].mean())
                        
                        if 'position_score' in filtered_df.columns:
                            components.append('Position')
                            scores.append(filtered_df['position_score'].mean())
                        
                        if 'volume_score' in filtered_df.columns:
                            components.append('Volume')
                            scores.append(filtered_df['volume_score'].mean())
                        
                        if 'rvol_score' in filtered_df.columns:
                            components.append('RVOL')
                            scores.append(filtered_df['rvol_score'].mean())
                        elif 'rvol' in filtered_df.columns:
                            components.append('RVOL')
                            scores.append(min(filtered_df['rvol'].mean() * 20, 100))  # Normalized to 0-100, capped at 100
                        
                        # Create component dataframe
                        component_data = {
                            'Component': components,
                            'Avg Score': scores,
                            'Quality': ['🔥 Strong' if score >= 65 else '📈 Good' if score >= 55 else '⚖️ Average' if score >= 40 else '📉 Weak' for score in scores]
                        }
                        
                        component_df = pd.DataFrame(component_data)
                        
                        st.dataframe(
                            component_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'Component': st.column_config.TextColumn("Component", width="medium"),
                                'Avg Score': st.column_config.ProgressColumn("Avg Score", min_value=0, max_value=100, format="%.1f"),
                                'Quality': st.column_config.TextColumn("Quality", width="small")
                            }
                        )
                        
                        # Component insights
                        if not component_df.empty:
                            best_component = component_df.loc[component_df['Avg Score'].idxmax(), 'Component']
                            best_score = component_df['Avg Score'].max()
                            st.success(f"🏆 **Strongest Component**: {best_component} ({best_score:.1f} avg score)")
                            
                            # Show component count
                            st.info(f"📊 Analyzing {len(components)} score components from your dataset")
                    else:
                        st.warning("⚠️ **Insufficient Data**: Need at least 2 score components for analysis")
                        st.info("💡 **Available columns**: " + ", ".join(available_score_cols) if available_score_cols else "No score columns found")
            
            # Tab 2: Performance Matrix
            with viz_tabs[1]:
                perf_cols = st.columns(2)
                
                with perf_cols[0]:
                    st.markdown("#### 💰 **Return Performance Analysis**")
                    
                    if any(col in filtered_df.columns for col in ['ret_1d', 'ret_3d', 'ret_7d', 'ret_30d']):
                        return_analysis = {}
                        
                        for period, col in [('1D', 'ret_1d'), ('3D', 'ret_3d'), ('7D', 'ret_7d'), ('30D', 'ret_30d')]:
                            if col in filtered_df.columns:
                                positive_returns = len(filtered_df[filtered_df[col] > 0])
                                avg_return = filtered_df[col].mean()
                                max_return = filtered_df[col].max()
                                
                                return_analysis[period] = {
                                    'Positive %': (positive_returns / len(filtered_df) * 100),
                                    'Avg Return': avg_return,
                                    'Max Return': max_return,
                                    'Quality': '🔥' if avg_return > 5 else '📈' if avg_return > 2 else '⚖️' if avg_return > 0 else '📉'
                                }
                        
                        if return_analysis:
                            return_df = pd.DataFrame(return_analysis).T
                            
                            st.dataframe(
                                return_df,
                                width='stretch',
                                column_config={
                                    'Positive %': st.column_config.ProgressColumn("Win Rate", min_value=0, max_value=100, format="%.1f"),
                                    'Avg Return': st.column_config.NumberColumn("Avg Return", format="%.2f%%"),
                                    'Max Return': st.column_config.NumberColumn("Max Return", format="%.2f%%"),
                                    'Quality': st.column_config.TextColumn("Quality", width="small")
                                }
                            )
                            
                            # Performance insights
                            best_period = return_df['Avg Return'].idxmax()
                            best_avg = return_df.loc[best_period, 'Avg Return']
                            st.success(f"🎯 **Best Timeframe**: {best_period} with {best_avg:.2f}% avg return")
                    else:
                        st.info("📊 Return data not available in current dataset")
                
                with perf_cols[1]:
                    st.markdown("#### 📊 **Volume & Liquidity Matrix**")
                    
                    if 'rvol' in filtered_df.columns:
                        volume_ranges = {
                            'Extreme (>5x)': len(filtered_df[filtered_df['rvol'] > 5]),
                            'Very High (3-5x)': len(filtered_df[(filtered_df['rvol'] >= 3) & (filtered_df['rvol'] <= 5)]),
                            'High (2-3x)': len(filtered_df[(filtered_df['rvol'] >= 2) & (filtered_df['rvol'] < 3)]),
                            'Above Avg (1.5-2x)': len(filtered_df[(filtered_df['rvol'] >= 1.5) & (filtered_df['rvol'] < 2)]),
                            'Normal (<1.5x)': len(filtered_df[filtered_df['rvol'] < 1.5])
                        }
                        
                        volume_df = pd.DataFrame([
                            {'Range': k, 'Count': v, 'Percentage': (v/len(filtered_df)*100)}
                            for k, v in volume_ranges.items()
                        ])
                        
                        st.dataframe(
                            volume_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'Range': st.column_config.TextColumn("Volume Range", width="medium"),
                                'Count': st.column_config.NumberColumn("Stocks", width="small"),
                                'Percentage': st.column_config.ProgressColumn("% of Total", min_value=0, max_value=100, format="%.1f")
                            }
                        )
                        
                        # Volume insights
                        high_volume = volume_ranges['Extreme (>5x)'] + volume_ranges['Very High (3-5x)'] + volume_ranges['High (2-3x)']
                        high_vol_pct = (high_volume / len(filtered_df) * 100)
                        
                        if high_vol_pct > 30:
                            st.success(f"🔥 **High Activity Market**: {high_vol_pct:.1f}% stocks showing elevated volume")
                        elif high_vol_pct > 15:
                            st.info(f"📊 **Moderate Activity**: {high_vol_pct:.1f}% stocks with high volume")
                        else:
                            st.warning(f"📉 **Low Activity**: Only {high_vol_pct:.1f}% stocks with high volume")
            
            # Tab 3: Momentum Heatmap
            with viz_tabs[2]:
                st.markdown("#### 🔥 **Multi-Dimensional Momentum Analysis**")
                
                # Check for momentum-related columns
                momentum_cols = [col for col in ['momentum_score', 'acceleration_score'] if col in filtered_df.columns]
                
                if len(momentum_cols) >= 1:
                    momentum_viz_cols = st.columns(2)
                    
                    with momentum_viz_cols[0]:
                        if len(momentum_cols) == 2:  # Both momentum and acceleration available
                            st.markdown("**🎯 Momentum vs Acceleration Matrix**")
                            
                            fig_momentum = go.Figure()
                            
                            # Add scatter plot with color coding by master score
                            fig_momentum.add_trace(
                                go.Scatter(
                                    x=filtered_df['momentum_score'],
                                    y=filtered_df['acceleration_score'],
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=filtered_df['master_score'] if 'master_score' in filtered_df.columns else 'blue',
                                        colorscale='RdYlGn',
                                        showscale=True,
                                        colorbar=dict(title="Master Score")
                                    ),
                                    text=filtered_df['ticker'] if 'ticker' in filtered_df.columns else filtered_df.index,
                                    hovertemplate='<b>%{text}</b><br>Momentum: %{x}<br>Acceleration: %{y}<extra></extra>'
                                )
                            )
                            
                            # Add quadrant lines
                            fig_momentum.add_hline(y=60, line_dash="dash", line_color="gray", opacity=0.5)
                            fig_momentum.add_vline(x=60, line_dash="dash", line_color="gray", opacity=0.5)
                            
                            fig_momentum.update_layout(
                                title="Momentum-Acceleration Matrix",
                                xaxis_title="Momentum Score",
                                yaxis_title="Acceleration Score",
                                template='plotly_white',
                                height=400
                            )
                            
                            st.plotly_chart(fig_momentum, width='stretch')
                        else:
                            # Show single momentum metric
                            available_col = momentum_cols[0]
                            st.markdown(f"**📊 {available_col.replace('_', ' ').title()} Distribution**")
                            
                            fig_single = go.Figure()
                            fig_single.add_trace(
                                go.Histogram(
                                    x=filtered_df[available_col],
                                    nbinsx=20,
                                    marker_color='lightblue',
                                    opacity=0.7
                                )
                            )
                            
                            fig_single.update_layout(
                                title=f"{available_col.replace('_', ' ').title()} Distribution",
                                xaxis_title=available_col.replace('_', ' ').title(),
                                yaxis_title="Count",
                                template='plotly_white',
                                height=400
                            )
                            
                            st.plotly_chart(fig_single, width='stretch')
                    
                    with momentum_viz_cols[1]:
                        if len(momentum_cols) == 2:  # Both momentum and acceleration available
                            st.markdown("**📊 Momentum Quadrant Analysis**")
                            
                            # Quadrant analysis
                            high_momentum = filtered_df['momentum_score'] >= 60
                            high_acceleration = filtered_df['acceleration_score'] >= 60
                            
                            quadrants = {
                                '🚀 Explosive (High/High)': len(filtered_df[high_momentum & high_acceleration]),
                                '📈 Building (High/Low)': len(filtered_df[high_momentum & ~high_acceleration]),
                                '⚡ Accelerating (Low/High)': len(filtered_df[~high_momentum & high_acceleration]),
                                '⚖️ Consolidating (Low/Low)': len(filtered_df[~high_momentum & ~high_acceleration])
                            }
                            
                            quadrant_df = pd.DataFrame([
                                {'Quadrant': k, 'Count': v, 'Percentage': (v/len(filtered_df)*100)}
                                for k, v in quadrants.items()
                            ])
                            
                            st.dataframe(
                                quadrant_df,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    'Quadrant': st.column_config.TextColumn("Momentum Quadrant", width="medium"),
                                    'Count': st.column_config.NumberColumn("Stocks", width="small"),
                                    'Percentage': st.column_config.ProgressColumn("% of Total", min_value=0, max_value=100, format="%.1f")
                                }
                            )
                            
                            # Quadrant insights
                            explosive_pct = (quadrants['🚀 Explosive (High/High)'] / len(filtered_df) * 100)
                            if explosive_pct > 20:
                                st.success(f"🚀 **Explosive Market**: {explosive_pct:.1f}% stocks in high momentum/acceleration")
                            elif explosive_pct > 10:
                                st.info(f"📈 **Building Momentum**: {explosive_pct:.1f}% stocks showing explosive potential")
                            else:
                                st.warning(f"⚖️ **Consolidation Phase**: Only {explosive_pct:.1f}% explosive stocks")
                        else:
                            # Show insights for single momentum metric
                            available_col = momentum_cols[0]
                            col_name = available_col.replace('_', ' ').title()
                            st.markdown(f"**📊 {col_name} Analysis**")
                            
                            high_threshold = filtered_df[available_col].quantile(0.7)
                            high_count = len(filtered_df[filtered_df[available_col] >= high_threshold])
                            high_pct = (high_count / len(filtered_df) * 100)
                            
                            metric_analysis = pd.DataFrame([
                                {'Category': f'🔥 High {col_name}', 'Count': high_count, 'Percentage': high_pct},
                                {'Category': f'📈 Medium {col_name}', 'Count': len(filtered_df) - high_count - len(filtered_df[filtered_df[available_col] <= filtered_df[available_col].quantile(0.3)]), 'Percentage': 100 - high_pct - (len(filtered_df[filtered_df[available_col] <= filtered_df[available_col].quantile(0.3)]) / len(filtered_df) * 100)},
                                {'Category': f'⚖️ Low {col_name}', 'Count': len(filtered_df[filtered_df[available_col] <= filtered_df[available_col].quantile(0.3)]), 'Percentage': len(filtered_df[filtered_df[available_col] <= filtered_df[available_col].quantile(0.3)]) / len(filtered_df) * 100}
                            ])
                            
                            st.dataframe(
                                metric_analysis,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    'Category': st.column_config.TextColumn(f"{col_name} Range", width="medium"),
                                    'Count': st.column_config.NumberColumn("Stocks", width="small"),
                                    'Percentage': st.column_config.ProgressColumn("% of Total", min_value=0, max_value=100, format="%.1f")
                                }
                            )
                            
                            if high_pct > 30:
                                st.success(f"🔥 **Strong {col_name}**: {high_pct:.1f}% stocks showing high {available_col.replace('_', ' ')}")
                            elif high_pct > 15:
                                st.info(f"📈 **Moderate {col_name}**: {high_pct:.1f}% stocks with elevated {available_col.replace('_', ' ')}")
                            else:
                                st.warning(f"⚖️ **Low {col_name}**: Only {high_pct:.1f}% stocks with high {available_col.replace('_', ' ')}")
                else:
                    st.warning("⚠️ **Momentum data not available** - No momentum or acceleration scores found in dataset")
            
            # Tab 4: Pattern Intelligence
            with viz_tabs[3]:
                st.markdown("#### 💎 **Advanced Pattern Recognition Intelligence**")
                
                pattern_intel_cols = st.columns(2)
                
                with pattern_intel_cols[0]:
                    st.markdown("**🔍 Pattern Frequency Analysis**")
                    
                    if 'patterns' in filtered_df.columns:
                        pattern_counts = {}
                        for patterns in filtered_df['patterns'].dropna():
                            if patterns:
                                for p in patterns.split(' | '):
                                    pattern_counts[p] = pattern_counts.get(p, 0) + 1
                        
                        if pattern_counts:
                            pattern_df = pd.DataFrame(
                                list(pattern_counts.items()),
                                columns=['Pattern', 'Count']
                            ).sort_values('Count', ascending=False).head(15)
                            
                            pattern_df['Percentage'] = (pattern_df['Count'] / len(filtered_df) * 100)
                            pattern_df['Quality'] = pattern_df['Percentage'].apply(
                                lambda x: '🔥' if x > 15 else '📈' if x > 8 else '⚖️' if x > 3 else '📉'
                            )
                            
                            st.dataframe(
                                pattern_df,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    'Pattern': st.column_config.TextColumn("Pattern Type", width="medium"),
                                    'Count': st.column_config.NumberColumn("Occurrences", width="small"),
                                    'Percentage': st.column_config.ProgressColumn("% of Stocks", min_value=0, max_value=50, format="%.1f"),
                                    'Quality': st.column_config.TextColumn("Quality", width="small")
                                }
                            )
                            
                            # Pattern insights
                            dominant_pattern = pattern_df.iloc[0]['Pattern']
                            dominant_pct = pattern_df.iloc[0]['Percentage']
                            st.success(f"🎯 **Dominant Pattern**: {dominant_pattern} ({dominant_pct:.1f}% of stocks)")
                        else:
                            st.info("📊 **No patterns detected** in current selection")
                    else:
                        st.warning("⚠️ **Pattern data not available** - No pattern column found in dataset")
                
                with pattern_intel_cols[1]:
                    st.markdown("**📊 Pattern Quality Matrix**")
                    
                    if 'breakout_score' in filtered_df.columns:
                        # Pattern quality analysis
                        pattern_quality = {
                            'Elite Patterns (>80)': len(filtered_df[filtered_df['breakout_score'] > 80]),
                            'Strong Patterns (70-80)': len(filtered_df[(filtered_df['breakout_score'] >= 70) & (filtered_df['breakout_score'] <= 80)]),
                            'Good Patterns (60-70)': len(filtered_df[(filtered_df['breakout_score'] >= 60) & (filtered_df['breakout_score'] < 70)]),
                            'Average Patterns (50-60)': len(filtered_df[(filtered_df['breakout_score'] >= 50) & (filtered_df['breakout_score'] < 60)]),
                            'Weak Patterns (<50)': len(filtered_df[filtered_df['breakout_score'] < 50])
                        }
                        
                        quality_df = pd.DataFrame([
                            {'Quality Level': k, 'Count': v, 'Percentage': (v/len(filtered_df)*100)}
                            for k, v in pattern_quality.items()
                        ])
                        
                        st.dataframe(
                            quality_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'Quality Level': st.column_config.TextColumn("Pattern Quality", width="medium"),
                                'Count': st.column_config.NumberColumn("Stocks", width="small"),
                                'Percentage': st.column_config.ProgressColumn("% of Total", min_value=0, max_value=100, format="%.1f")
                            }
                        )
                        
                        # Quality insights
                        elite_patterns = pattern_quality['Elite Patterns (>80)']
                        strong_patterns = pattern_quality['Strong Patterns (70-80)']
                        high_quality_pct = ((elite_patterns + strong_patterns) / len(filtered_df) * 100)
                        
                        if high_quality_pct > 25:
                            st.success(f"💎 **High Pattern Quality**: {high_quality_pct:.1f}% elite/strong patterns")
                        elif high_quality_pct > 15:
                            st.info(f"📈 **Good Pattern Quality**: {high_quality_pct:.1f}% high-quality patterns")
                        else:
                            st.warning(f"⚖️ **Mixed Pattern Quality**: {high_quality_pct:.1f}% high-quality patterns")
            
            # Tab 5: Sector Analysis 
            with viz_tabs[4]:
                st.markdown("#### 🏢 **Comprehensive Sector Intelligence**")
                
                sector_overview_df_local = MarketIntelligence.detect_sector_rotation(filtered_df)
                
                if not sector_overview_df_local.empty:
                    sector_intel_cols = st.columns(2)
                    
                    with sector_intel_cols[0]:
                        st.markdown("**🎯 Sector Leadership Analysis**")
                        
                        # Enhanced sector display
                        display_cols_overview = ['flow_score', 'ldi_score', 'leadership_density', 'analyzed_stocks', 
                                               'total_stocks', 'avg_score', 'elite_avg_score', 'ldi_quality']
                        
                        available_overview_cols = [col for col in display_cols_overview if col in sector_overview_df_local.columns]
                        sector_overview_display = sector_overview_df_local[available_overview_cols].copy()
                        
                        # Add quality indicators
                        if 'ldi_score' in sector_overview_display.columns:
                            sector_overview_display['Sector_Quality'] = sector_overview_display['ldi_score'].apply(
                                lambda x: '💎 Elite' if x > 20 else '🔥 Strong' if x > 15 else '📈 Good' if x > 10 else '⚖️ Average'
                            )
                        
                        st.dataframe(
                            sector_overview_display,
                            width='stretch',
                            column_config={
                                'ldi_score': st.column_config.NumberColumn('LDI Score', format="%.1f%%"),
                                'flow_score': st.column_config.ProgressColumn('Flow Score', min_value=0, max_value=100, format="%.1f"),
                                'avg_score': st.column_config.ProgressColumn('Avg Score', min_value=0, max_value=100, format="%.1f"),
                                'Sector_Quality': st.column_config.TextColumn('Quality', width="small")
                            }
                        )
                        
                        # Sector insights
                        if 'ldi_score' in sector_overview_df_local.columns and len(sector_overview_df_local) > 0:
                            top_sector = sector_overview_df_local.index[0]
                            top_ldi = sector_overview_df_local['ldi_score'].iloc[0]
                            st.success(f"🏆 **Leading Sector**: {top_sector} (LDI: {top_ldi:.1f}%)")
                    
                    with sector_intel_cols[1]:
                        st.markdown("**📊 Sector Distribution Matrix**")
                        
                        if 'sector' in filtered_df.columns:
                            sector_dist = filtered_df['sector'].value_counts().head(10)
                            sector_pct = (sector_dist / len(filtered_df) * 100)
                            
                            sector_matrix = pd.DataFrame({
                                'Sector': sector_dist.index,
                                'Stock Count': sector_dist.values,
                                'Percentage': sector_pct.values,
                                'Representation': sector_pct.map(
                                    lambda x: '🔥 Dominant' if x > 20 else '📈 Strong' if x > 10 else '⚖️ Moderate' if x > 5 else '📉 Light'
                                ).values
                            })
                            
                            st.dataframe(
                                sector_matrix,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    'Sector': st.column_config.TextColumn("Sector", width="medium"),
                                    'Stock Count': st.column_config.NumberColumn("Stocks", width="small"),
                                    'Percentage': st.column_config.ProgressColumn("% of Total", min_value=0, max_value=50, format="%.1f"),
                                    'Representation': st.column_config.TextColumn("Level", width="small")
                                }
                            )
                        else:
                            st.info("Sector data not available")
                else:
                    st.info("No sector data available for analysis")
            
            # Tab 6: Industry Analysis
            with viz_tabs[5]:
                st.markdown("#### 🏭 **Comprehensive Industry Intelligence Platform**")
                
                industry_intel_cols = st.columns(2)
                
                with industry_intel_cols[0]:
                    st.markdown("**🔍 Industry Performance Rankings**")
                    
                    if 'industry' in filtered_df.columns:
                        # Industry performance analysis
                        industry_metrics = filtered_df.groupby('industry').agg({
                            'master_score': ['mean', 'std', 'count'],
                            'momentum_score': 'mean' if 'momentum_score' in filtered_df.columns else lambda x: None,
                            'acceleration_score': 'mean' if 'acceleration_score' in filtered_df.columns else lambda x: None,
                            'volume_score': 'mean' if 'volume_score' in filtered_df.columns else lambda x: None,
                            'price': 'mean' if 'price' in filtered_df.columns else lambda x: None
                        }).round(2)
                        
                        # Flatten column names
                        industry_metrics.columns = ['avg_score', 'score_volatility', 'stock_count', 
                                                  'momentum_avg', 'acceleration_avg', 'volume_avg', 'avg_price']
                        
                        # Calculate industry strength index
                        industry_metrics['strength_index'] = (
                            (industry_metrics['avg_score'] * 0.4) +
                            (industry_metrics['momentum_avg'].fillna(50) * 0.2) +
                            (industry_metrics['acceleration_avg'].fillna(50) * 0.2) +
                            (industry_metrics['volume_avg'].fillna(50) * 0.2)
                        ).round(1)
                        
                        # Add quality indicators
                        industry_metrics['quality'] = industry_metrics['strength_index'].apply(
                            lambda x: '🚀 Elite' if x > 80 else '🔥 Strong' if x > 70 else '📈 Good' if x > 60 else '⚖️ Average' if x > 50 else '📉 Weak'
                        )
                        
                        # Sort by strength index
                        industry_ranking = industry_metrics.sort_values('strength_index', ascending=False)
                        
                        st.dataframe(
                            industry_ranking[['avg_score', 'strength_index', 'stock_count', 'score_volatility', 'quality']],
                            width='stretch',
                            column_config={
                                'avg_score': st.column_config.ProgressColumn('Avg Score', min_value=0, max_value=100, format="%.1f"),
                                'strength_index': st.column_config.ProgressColumn('Strength Index', min_value=0, max_value=100, format="%.1f"),
                                'stock_count': st.column_config.NumberColumn('Stock Count', width="small"),
                                'score_volatility': st.column_config.NumberColumn('Volatility', width="small", format="%.1f"),
                                'quality': st.column_config.TextColumn('Quality', width="small")
                            }
                        )
                        
                        # Industry insights
                        if len(industry_ranking) > 0:
                            top_industry = industry_ranking.index[0]
                            top_strength = industry_ranking['strength_index'].iloc[0]
                            top_count = industry_ranking['stock_count'].iloc[0]
                            st.success(f"🏆 **Leading Industry**: {top_industry} (Strength: {top_strength:.1f}, {top_count} stocks)")
                    else:
                        st.warning("⚠️ **Industry data not available** - No industry column found in dataset")
                
                with industry_intel_cols[1]:
                    st.markdown("**📊 Industry Concentration Analysis**")
                    
                    if 'industry' in filtered_df.columns:
                        industry_dist = filtered_df['industry'].value_counts().head(12)
                        industry_pct = (industry_dist / len(filtered_df) * 100)
                        
                        # Create concentration matrix
                        concentration_matrix = pd.DataFrame({
                            'Industry': industry_dist.index,
                            'Stock Count': industry_dist.values,
                            'Market Share': industry_pct.values,
                            'Concentration': industry_pct.apply(
                                lambda x: '🔥 Dominant' if x > 15 else '📈 Major' if x > 8 else '⚖️ Moderate' if x > 4 else '📉 Niche'
                            ).values
                        })
                        
                        st.dataframe(
                            concentration_matrix,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'Industry': st.column_config.TextColumn("Industry", width="medium"),
                                'Stock Count': st.column_config.NumberColumn("Stocks", width="small"),
                                'Market Share': st.column_config.ProgressColumn("% of Market", min_value=0, max_value=50, format="%.1f"),
                                'Concentration': st.column_config.TextColumn("Level", width="small")
                            }
                        )
                        
                        # Concentration insights
                        total_industries = len(filtered_df['industry'].unique())
                        top_5_concentration = industry_pct.head(5).sum()
                        
                        if top_5_concentration > 60:
                            st.warning(f"⚠️ **High Concentration**: Top 5 industries control {top_5_concentration:.1f}% of market")
                        elif top_5_concentration > 40:
                            st.info(f"📊 **Moderate Concentration**: Top 5 industries hold {top_5_concentration:.1f}% of market")
                        else:
                            st.success(f"🎯 **Diversified Market**: Well-distributed across {total_industries} industries")
                        
                        # Industry momentum analysis
                        if 'momentum_score' in filtered_df.columns:
                            st.markdown("**🚀 Industry Momentum Leaders**")
                            
                            industry_momentum = filtered_df.groupby('industry')['momentum_score'].mean().sort_values(ascending=False).head(8)
                            momentum_leaders = pd.DataFrame({
                                'Industry': industry_momentum.index,
                                'Avg Momentum': industry_momentum.values.round(1),
                                'Momentum Level': industry_momentum.apply(
                                    lambda x: '🚀 Explosive' if x > 80 else '🔥 Strong' if x > 70 else '📈 Building' if x > 60 else '⚖️ Stable'
                                ).values
                            })
                            
                            st.dataframe(
                                momentum_leaders,
                                width='stretch',
                                hide_index=True,
                                column_config={
                                    'Industry': st.column_config.TextColumn("Industry", width="medium"),
                                    'Avg Momentum': st.column_config.ProgressColumn("Momentum Score", min_value=0, max_value=100, format="%.1f"),
                                    'Momentum Level': st.column_config.TextColumn("Status", width="small")
                                }
                            )
                    else:
                        st.warning("⚠️ **Industry data not available** - No industry column found in dataset")
            
            # Tab 7: Risk Dashboard
            with viz_tabs[6]:
                st.markdown("#### ⚡ **Advanced Risk Management Dashboard**")
                
                risk_cols = st.columns(2)
                
                with risk_cols[0]:
                    st.markdown("**🛡️ Risk Distribution Analysis**")
                    
                    # Calculate risk metrics
                    if 'rvol' in filtered_df.columns and 'master_score' in filtered_df.columns:
                        # Risk categorization based on volume and score
                        high_risk = (filtered_df['rvol'] > 4) | (filtered_df['master_score'] < 50)
                        medium_risk = ((filtered_df['rvol'] >= 2) & (filtered_df['rvol'] <= 4)) & (filtered_df['master_score'] >= 50)
                        low_risk = (filtered_df['rvol'] < 2) & (filtered_df['master_score'] >= 70)
                        
                        risk_distribution = {
                            '🚨 High Risk': len(filtered_df[high_risk]),
                            '⚠️ Medium Risk': len(filtered_df[medium_risk]),
                            '🛡️ Low Risk': len(filtered_df[low_risk])
                        }
                        
                        risk_df = pd.DataFrame([
                            {'Risk Level': k, 'Count': v, 'Percentage': (v/len(filtered_df)*100)}
                            for k, v in risk_distribution.items()
                        ])
                        
                        st.dataframe(
                            risk_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'Risk Level': st.column_config.TextColumn("Risk Category", width="medium"),
                                'Count': st.column_config.NumberColumn("Stocks", width="small"),
                                'Percentage': st.column_config.ProgressColumn("% of Portfolio", min_value=0, max_value=100, format="%.1f")
                            }
                        )
                        
                        # Risk insights
                        high_risk_pct = (risk_distribution['🚨 High Risk'] / len(filtered_df) * 100)
                        if high_risk_pct > 40:
                            st.error(f"🚨 **High Risk Portfolio**: {high_risk_pct:.1f}% high-risk positions")
                        elif high_risk_pct > 25:
                            st.warning(f"⚠️ **Moderate Risk**: {high_risk_pct:.1f}% high-risk positions")
                        else:
                            st.success(f"🛡️ **Controlled Risk**: Only {high_risk_pct:.1f}% high-risk positions")
                
                with risk_cols[1]:
                    st.markdown("**📊 Position Sizing Recommendations**")
                    
                    if 'master_score' in filtered_df.columns and 'rvol' in filtered_df.columns:
                        # Position sizing based on score and volatility
                        filtered_df_copy = filtered_df.copy()
                        
                        # Calculate recommended position size (1-5% of portfolio)
                        filtered_df_copy['position_size'] = (
                            (filtered_df_copy['master_score'] / 100 * 3) +  # Base on score (0-3%)
                            np.where(filtered_df_copy['rvol'] < 2, 2, 1)     # Bonus for low volatility
                        ).clip(0.5, 5.0)  # Min 0.5%, Max 5%
                        
                        # Position categories
                        position_categories = pd.cut(
                            filtered_df_copy['position_size'], 
                            bins=[0, 1, 2, 3, 5], 
                            labels=['🔸 Small (0.5-1%)', '🔹 Medium (1-2%)', '🔷 Large (2-3%)', '💎 Max (3-5%)']
                        ).value_counts()
                        
                        position_df = pd.DataFrame({
                            'Position Size': position_categories.index,
                            'Count': position_categories.values,
                            'Percentage': (position_categories.values / len(filtered_df) * 100)
                        })
                        
                        st.dataframe(
                            position_df,
                            width='stretch',
                            hide_index=True,
                            column_config={
                                'Position Size': st.column_config.TextColumn("Recommended Size", width="medium"),
                                'Count': st.column_config.NumberColumn("Stocks", width="small"),
                                'Percentage': st.column_config.ProgressColumn("% Allocation", min_value=0, max_value=100, format="%.1f")
                            }
                        )
                        
                        # Position sizing insights
                        max_positions = position_categories.get('💎 Max (3-5%)', 0)
                        max_pct = (max_positions / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
                        
                        if max_pct > 15:
                            st.success(f"💎 **High Conviction Opportunities**: {max_pct:.1f}% qualify for maximum position size")
                        elif max_pct > 8:
                            st.info(f"🔷 **Solid Opportunities**: {max_pct:.1f}% qualify for large positions")
                        else:
                            st.warning(f"🔸 **Conservative Allocation**: Only {max_pct:.1f}% qualify for large positions")
            
            # ================================================================================================
            # 🎯 EXECUTIVE SUMMARY AND ACTIONABLE INSIGHTS
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### 🎯 **EXECUTIVE SUMMARY & ACTIONABLE INSIGHTS**")
            
            summary_cols = st.columns(2)
            
            with summary_cols[0]:
                st.markdown("#### 📋 **Key Findings**")
                
                findings = []
                
                # Market sentiment finding
                if elite_stocks > total_analyzed * 0.15:
                    findings.append(f"🔥 **Strong Market**: {elite_pct:.1f}% elite stocks indicate bullish conditions")
                elif strong_stocks > total_analyzed * 0.25:
                    findings.append(f"📈 **Positive Market**: {(strong_stocks/total_analyzed*100):.1f}% strong stocks")
                else:
                    findings.append(f"⚖️ **Neutral Market**: {(strong_stocks/total_analyzed*100):.1f}% strong stocks - selective approach needed")
                
                # Volume finding
                if 'rvol' in filtered_df.columns:
                    high_vol_stocks = len(filtered_df[filtered_df['rvol'] > 2.5])
                    high_vol_pct = (high_vol_stocks / total_analyzed * 100)
                    if high_vol_pct > 25:
                        findings.append(f"📊 **High Activity**: {high_vol_pct:.1f}% stocks show elevated volume")
                    else:
                        findings.append(f"📉 **Low Activity**: Only {high_vol_pct:.1f}% stocks with high volume")
                
                # Momentum finding
                if 'momentum_score' in filtered_df.columns:
                    avg_momentum = filtered_df['momentum_score'].mean()
                    if avg_momentum > 65:
                        findings.append(f"🚀 **Strong Momentum**: {avg_momentum:.1f} average momentum score")
                    elif avg_momentum > 55:
                        findings.append(f"📈 **Building Momentum**: {avg_momentum:.1f} average momentum score")
                    else:
                        findings.append(f"⚖️ **Weak Momentum**: {avg_momentum:.1f} average momentum score")
                
                # Risk finding
                if 'rvol' in filtered_df.columns:
                    high_risk_count = len(filtered_df[filtered_df['rvol'] > 4])
                    risk_pct = (high_risk_count / total_analyzed * 100)
                    if risk_pct > 30:
                        findings.append(f"🚨 **High Risk Environment**: {risk_pct:.1f}% highly volatile stocks")
                    elif risk_pct > 15:
                        findings.append(f"⚠️ **Moderate Risk**: {risk_pct:.1f}% highly volatile stocks")
                    else:
                        findings.append(f"🛡️ **Low Risk Environment**: Only {risk_pct:.1f}% highly volatile stocks")
                
                for finding in findings:
                    st.write(f"• {finding}")
            
            with summary_cols[1]:
                st.markdown("#### 🎯 **Strategic Recommendations**")
                
                recommendations = []
                
                # Position sizing recommendation
                if elite_stocks > total_analyzed * 0.1:
                    recommendations.append("💎 **Increase Position Sizes** on elite-scoring stocks (80+)")
                else:
                    recommendations.append("🔸 **Conservative Sizing** - limited high-conviction opportunities")
                
                # Diversification recommendation
                if 'sector' in filtered_df.columns:
                    sector_concentration = filtered_df['sector'].value_counts().iloc[0] / len(filtered_df)
                    if sector_concentration > 0.4:
                        recommendations.append("🌐 **Increase Diversification** - high sector concentration detected")
                    else:
                        recommendations.append("✅ **Good Diversification** - balanced sector exposure")
                
                # Risk management recommendation
                if 'rvol' in filtered_df.columns:
                    avg_rvol = filtered_df['rvol'].mean()
                    if avg_rvol > 3:
                        recommendations.append("🛡️ **Implement Stop Losses** - high volatility environment")
                    elif avg_rvol > 2:
                        recommendations.append("⚠️ **Monitor Positions Closely** - elevated volatility")
                    else:
                        recommendations.append("📈 **Normal Risk Management** - stable environment")
                
                # Timing recommendation
                if 'momentum_score' in filtered_df.columns:
                    momentum_trend = filtered_df['momentum_score'].mean()
                    if momentum_trend > 65:
                        recommendations.append("⚡ **Aggressive Entry** - strong momentum environment")
                    elif momentum_trend > 55:
                        recommendations.append("📈 **Selective Entry** - building momentum")
                    else:
                        recommendations.append("⏳ **Wait for Setup** - weak momentum environment")
                
                for rec in recommendations:
                    st.write(f"• {rec}")
            
            # ================================================================================================
            # 🏆 TOP PICKS SUMMARY
            # ================================================================================================
            
            st.markdown("---")
            st.markdown("### 🏆 **INSTITUTIONAL TOP PICKS**")
            
            if 'master_score' in filtered_df.columns:
                # Get top picks based on multiple criteria
                top_picks = filtered_df[
                    (filtered_df['master_score'] >= 75) &
                    (filtered_df['rvol'] >= 1.5 if 'rvol' in filtered_df.columns else True)
                ].nlargest(10, 'master_score')
                
                if not top_picks.empty:
                    # Display top picks with key metrics
                    display_columns = ['ticker', 'company_name', 'master_score']
                    if 'momentum_score' in top_picks.columns:
                        display_columns.append('momentum_score')
                    if 'rvol' in top_picks.columns:
                        display_columns.append('rvol')
                    if 'ret_1d' in top_picks.columns:
                        display_columns.append('ret_1d')
                    
                    picks_display = top_picks[display_columns].copy()
                    
                    # Add quality rating
                    picks_display['Quality'] = picks_display['master_score'].apply(
                        lambda x: '💎 Elite' if x >= 85 else '🔥 Premium' if x >= 80 else '📈 Strong'
                    )
                    
                    st.dataframe(
                        picks_display,
                        width='stretch',
                        hide_index=True,
                        column_config={
                            'ticker': st.column_config.TextColumn("Ticker", width="small"),
                            'company_name': st.column_config.TextColumn("Company", width="medium"),
                            'master_score': st.column_config.ProgressColumn("Master Score", min_value=0, max_value=100, format="%.1f"),
                            'momentum_score': st.column_config.ProgressColumn("Momentum", min_value=0, max_value=100, format="%.1f"),
                            'rvol': st.column_config.NumberColumn("RVOL", format="%.1fx"),
                            'ret_1d': st.column_config.NumberColumn("1D Return", format="%.2f%%"),
                            'Quality': st.column_config.TextColumn("Quality", width="small")
                        }
                    )
                    
                    # Top pick insights
                    elite_picks = len(picks_display[picks_display['Quality'] == '💎 Elite'])
                    premium_picks = len(picks_display[picks_display['Quality'] == '🔥 Premium'])
                    
                    if elite_picks > 0:
                        st.success(f"💎 **{elite_picks} Elite-Grade Opportunities** identified for institutional allocation")
                    if premium_picks > 0:
                        st.success(f"🔥 **{premium_picks} Premium-Grade Opportunities** suitable for aggressive positions")
                else:
                    st.info("📊 No stocks meet institutional-grade criteria in current selection")
        
        else:
            st.warning("📊 No data available for comprehensive analysis. Please adjust your filters.")
            st.info("💡 **Tip**: Try expanding your filters or checking different tabs for data availability")
    
    # Tab 4: Search
    # Tab 4: Search
    with tabs[4]:
        st.markdown("### 🔍 Advanced Stock Search")
        
        # Search interface
        col1, col2 = st.columns([4, 1])
        
        with col1:
            search_query = st.text_input(
                "Search stocks",
                placeholder="Enter ticker or company name...",
                help="Search by ticker symbol or company name",
                key="search_input"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            search_clicked = st.button("🔎 Search", type="primary", width="stretch", key="search_btn")
        
        # Perform search
        if search_query or search_clicked:
            with st.spinner("Searching..."):
                search_results = SearchEngine.search_stocks(filtered_df, search_query)

            if not search_results.empty:
                # ENSURE PATTERN CONFIDENCE IS CALCULATED FOR SEARCH RESULTS
                if 'patterns' in search_results.columns and 'pattern_confidence' not in search_results.columns:
                    search_results = PatternDetector._calculate_pattern_confidence(search_results)
            
            if not search_results.empty:
                st.success(f"Found {len(search_results)} matching stock(s)")
                
                # Create summary dataframe for search results
                summary_columns = ['ticker', 'company_name', 'rank', 'master_score', 'price', 
                                  'ret_30d', 'rvol', 'market_state', 'category']
                
                available_summary_cols = [col for col in summary_columns if col in search_results.columns]
                search_summary = search_results[available_summary_cols].copy()
                
                # Format the summary data
                if 'price' in search_summary.columns:
                    search_summary['price_display'] = search_summary['price'].apply(
                        lambda x: f"₹{x:,.0f}" if pd.notna(x) else '-'
                    )
                    search_summary = search_summary.drop('price', axis=1)
                
                if 'ret_30d' in search_summary.columns:
                    search_summary['ret_30d_display'] = search_summary['ret_30d'].apply(
                        lambda x: f"{x:+.1f}%" if pd.notna(x) else '-'
                    )
                    search_summary = search_summary.drop('ret_30d', axis=1)
                
                if 'rvol' in search_summary.columns:
                    search_summary['rvol_display'] = search_summary['rvol'].apply(
                        lambda x: f"{x:.1f}x" if pd.notna(x) else '-'
                    )
                    search_summary = search_summary.drop('rvol', axis=1)
                
                # Rename columns for display
                column_rename = {
                    'ticker': 'Ticker',
                    'company_name': 'Company',
                    'rank': 'Rank',
                    'master_score': 'Score',
                    'price_display': 'Price',
                    'ret_30d_display': '30D Return',
                    'rvol_display': 'RVOL',
                    'market_state': 'Market State',
                    'category': 'Category'
                }
                
                search_summary = search_summary.rename(columns=column_rename)
                
                # Display search results summary with optimized column_config
                st.markdown("#### 📊 Search Results Overview")
                st.dataframe(
                    search_summary,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        'Ticker': st.column_config.TextColumn(
                            'Ticker',
                            help="Stock symbol - Click expander below for details",
                            width="small"
                        ),
                        'Company': st.column_config.TextColumn(
                            'Company',
                            help="Company name",
                            width="large"
                        ),
                        'Rank': st.column_config.NumberColumn(
                            'Rank',
                            help="Overall ranking position",
                            format="%d",
                            width="small"
                        ),
                        'Score': st.column_config.ProgressColumn(
                            'Score',
                            help="Master Score (0-100)",
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        'Price': st.column_config.TextColumn(
                            'Price',
                            help="Current stock price",
                            width="small"
                        ),
                        '30D Return': st.column_config.TextColumn(
                            '30D Return',
                            help="30-day return percentage",
                            width="small"
                        ),
                        'RVOL': st.column_config.TextColumn(
                            'RVOL',
                            help="Relative Volume",
                            width="small"
                        ),
                        'Market State': st.column_config.TextColumn(
                            'Market State',
                            help="Current momentum market state",
                            width="medium"
                        ),
                        'Category': st.column_config.TextColumn(
                            'Category',
                            help="Market cap category",
                            width="medium"
                        )
                    }
                )
                
                st.markdown("---")
                st.markdown("#### 📋 Detailed Stock Information")
                
                # Display each result in expandable sections
                for idx, stock in search_results.iterrows():
                    with st.expander(
                        f"📊 {stock['ticker']} - {stock['company_name']} "
                        f"(Rank #{int(stock['rank'])})",
                        expanded=(len(search_results) == 1)  # Auto-expand if only one result
                    ):
                        # Header metrics
                        metric_cols = st.columns(6)
                        
                        with metric_cols[0]:
                            UIComponents.render_metric_card(
                                "Master Score",
                                f"{stock['master_score']:.1f}",
                                f"Rank #{int(stock['rank'])}"
                            )
                        
                        with metric_cols[1]:
                            price_value = f"₹{stock['price']:,.0f}" if pd.notna(stock.get('price')) else "N/A"
                            ret_1d_value = f"{stock['ret_1d']:+.1f}%" if pd.notna(stock.get('ret_1d')) else None
                            UIComponents.render_metric_card("Price", price_value, ret_1d_value)
                        
                        with metric_cols[2]:
                            UIComponents.render_metric_card(
                                "From Low",
                                f"{stock['from_low_pct']:.0f}%" if pd.notna(stock.get('from_low_pct')) else "N/A",
                                "52-week range position"
                            )
                        
                        with metric_cols[3]:
                            ret_30d = stock.get('ret_30d', 0)
                            UIComponents.render_metric_card(
                                "30D Return",
                                f"{ret_30d:+.1f}%" if pd.notna(ret_30d) else "N/A",
                                "↑" if ret_30d > 0 else "↓" if ret_30d < 0 else "→"
                            )
                        
                        with metric_cols[4]:
                            rvol = stock.get('rvol', 1)
                            UIComponents.render_metric_card(
                                "RVOL",
                                f"{rvol:.1f}x" if pd.notna(rvol) else "N/A",
                                "High" if rvol > 2 else "Normal" if rvol > 0.5 else "Low"
                            )
                        
                        with metric_cols[5]:
                            UIComponents.render_metric_card(
                                "Market State",
                                stock.get('market_state', 'N/A'),
                                stock.get('category', 'N/A')
                            )
                        
                        # Score breakdown with optimized display
                        st.markdown("#### 📈 Score Components")
                        
                        # Create score breakdown dataframe
                        score_data = {
                            'Component': ['Position', 'Volume', 'Momentum', 'Acceleration', 'Breakout', 'RVOL'],
                            'Score': [
                                stock.get('position_score', 0),
                                stock.get('volume_score', 0),
                                stock.get('momentum_score', 0),
                                stock.get('acceleration_score', 0),
                                stock.get('breakout_score', 0),
                                stock.get('rvol_score', 0)
                            ],
                            'Weight': [
                                f"{CONFIG.POSITION_WEIGHT:.0%}",
                                f"{CONFIG.VOLUME_WEIGHT:.0%}",
                                f"{CONFIG.MOMENTUM_WEIGHT:.0%}",
                                f"{CONFIG.ACCELERATION_WEIGHT:.0%}",
                                f"{CONFIG.BREAKOUT_WEIGHT:.0%}",
                                f"{CONFIG.RVOL_WEIGHT:.0%}"
                            ],
                            'Contribution': [
                                stock.get('position_score', 0) * CONFIG.POSITION_WEIGHT,
                                stock.get('volume_score', 0) * CONFIG.VOLUME_WEIGHT,
                                stock.get('momentum_score', 0) * CONFIG.MOMENTUM_WEIGHT,
                                stock.get('acceleration_score', 0) * CONFIG.ACCELERATION_WEIGHT,
                                stock.get('breakout_score', 0) * CONFIG.BREAKOUT_WEIGHT,
                                stock.get('rvol_score', 0) * CONFIG.RVOL_WEIGHT
                            ]
                        }
                        
                        score_df = pd.DataFrame(score_data)
                        
                        # Add quality indicator
                        score_df['Quality'] = score_df['Score'].apply(
                            lambda x: '🟢 Strong' if x >= 80 
                            else '🟡 Good' if x >= 60 
                            else '🟠 Fair' if x >= 40 
                            else '🔴 Weak'
                        )
                        
                        # Display score breakdown with column_config
                        st.dataframe(
                            score_df,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                'Component': st.column_config.TextColumn(
                                    'Component',
                                    help="Score component name",
                                    width="medium"
                                ),
                                'Score': st.column_config.ProgressColumn(
                                    'Score',
                                    help="Component score (0-100)",
                                    format="%.1f",
                                    min_value=0,
                                    max_value=100,
                                    width="small"
                                ),
                                'Weight': st.column_config.TextColumn(
                                    'Weight',
                                    help="Component weight in master score",
                                    width="small"
                                ),
                                'Contribution': st.column_config.NumberColumn(
                                    'Contribution',
                                    help="Points contributed to master score",
                                    format="%.1f",
                                    width="small"
                                ),
                                'Quality': st.column_config.TextColumn(
                                    'Quality',
                                    help="Component strength indicator",
                                    width="small"
                                )
                            }
                        )
                        
                        # Patterns
                        if stock.get('patterns'):
                            st.markdown(f"**🎯 Patterns Detected:**")
                            patterns_list = stock['patterns'].split(' | ')
                            pattern_cols = st.columns(min(3, len(patterns_list)))
                            for i, pattern in enumerate(patterns_list):
                                with pattern_cols[i % 3]:
                                    st.info(pattern)
                        
                        # Additional details in organized tabs
                        detail_tabs = st.tabs(["📊 Classification", "📈 Performance", "💰 Fundamentals", "🔍 Technicals", "📊 Volume", "🎯 Advanced"])
                        
                        with detail_tabs[0]:  # Classification
                            class_col1, class_col2 = st.columns(2)
                            
                            with class_col1:
                                st.markdown("**📊 Stock Classification**")
                                classification_data = {
                                    'Attribute': ['Sector', 'Industry', 'Category', 'Market Cap'],
                                    'Value': [
                                        stock.get('sector', 'Unknown'),
                                        stock.get('industry', 'Unknown'),
                                        stock.get('category', 'Unknown'),
                                        stock.get('market_cap', 'N/A')
                                    ]
                                }
                                class_df = pd.DataFrame(classification_data)
                                st.dataframe(
                                    class_df,
                                    width="stretch",
                                    hide_index=True,
                                    column_config={
                                        'Attribute': st.column_config.TextColumn('Attribute', width="medium"),
                                        'Value': st.column_config.TextColumn('Value', width="large")
                                    }
                                )
                            
                            with class_col2:
                                st.markdown("**📈 Tier Classifications**")
                                tier_data = {
                                    'Tier Type': [],
                                    'Classification': []
                                }
                                
                                if 'price_tier' in stock.index:
                                    tier_data['Tier Type'].append('Price Tier')
                                    tier_data['Classification'].append(stock.get('price_tier', 'N/A'))
                                
                                if 'eps_tier' in stock.index:
                                    tier_data['Tier Type'].append('EPS Tier')
                                    tier_data['Classification'].append(stock.get('eps_tier', 'N/A'))
                                
                                if 'pe_tier' in stock.index:
                                    tier_data['Tier Type'].append('PE Tier')
                                    tier_data['Classification'].append(stock.get('pe_tier', 'N/A'))

                                if 'eps_change_tier' in stock.index:
                                    tier_data['Tier Type'].append('EPS Growth')
                                    tier_data['Classification'].append(stock.get('eps_change_tier', 'N/A'))
                                
                                if tier_data['Tier Type']:
                                    tier_df = pd.DataFrame(tier_data)
                                    st.dataframe(
                                        tier_df,
                                        width="stretch",
                                        hide_index=True,
                                        column_config={
                                            'Tier Type': st.column_config.TextColumn('Type', width="medium"),
                                            'Classification': st.column_config.TextColumn('Class', width="medium")
                                        }
                                    )
                                else:
                                    st.info("No tier data available")
                        
                        with detail_tabs[1]:  # Performance
                            st.markdown("**📈 Historical Performance**")
                            
                            perf_data = {
                                'Period': [],
                                'Return': [],
                                'Status': []
                            }
                            
                            periods = [
                                ('1 Day', 'ret_1d'),
                                ('3 Days', 'ret_3d'),
                                ('7 Days', 'ret_7d'),
                                ('30 Days', 'ret_30d'),
                                ('3 Months', 'ret_3m'),
                                ('6 Months', 'ret_6m'),
                                ('1 Year', 'ret_1y'),
                                ('3 Years', 'ret_3y'),
                                ('5 Years', 'ret_5y')
                            ]
                            
                            for period_name, col_name in periods:
                                if col_name in stock.index and pd.notna(stock[col_name]):
                                    perf_data['Period'].append(period_name)
                                    ret_val = stock[col_name]
                                    perf_data['Return'].append(f"{ret_val:+.1f}%")
                                    
                                    if ret_val > 10:
                                        perf_data['Status'].append('🟢 Strong')
                                    elif ret_val > 0:
                                        perf_data['Status'].append('🟡 Positive')
                                    elif ret_val > -10:
                                        perf_data['Status'].append('🟠 Negative')
                                    else:
                                        perf_data['Status'].append('🔴 Weak')
                            
                            if perf_data['Period']:
                                perf_df = pd.DataFrame(perf_data)
                                st.dataframe(
                                    perf_df,
                                    width="stretch",
                                    hide_index=True,
                                    column_config={
                                        'Period': st.column_config.TextColumn('Period', width="medium"),
                                        'Return': st.column_config.TextColumn('Return', width="small"),
                                        'Status': st.column_config.TextColumn('Status', width="small")
                                    }
                                )
                            else:
                                st.info("No performance data available")
                        
                        with detail_tabs[2]:  # Fundamentals
                            if show_fundamentals:
                                st.markdown("**💰 Fundamental Analysis**")
                                
                                fund_data = {
                                    'Metric': [],
                                    'Value': [],
                                    'Assessment': []
                                }
                                
                                # PE Ratio
                                if 'pe' in stock.index and pd.notna(stock['pe']):
                                    fund_data['Metric'].append('PE Ratio')
                                    pe_val = stock['pe']
                                    
                                    if pe_val <= 0:
                                        fund_data['Value'].append('Loss/Negative')
                                        fund_data['Assessment'].append('🔴 No Earnings')
                                    elif pe_val < 15:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🟢 Undervalued')
                                    elif pe_val < 25:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🟡 Fair Value')
                                    elif pe_val < 50:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🟠 Expensive')
                                    else:
                                        fund_data['Value'].append(f"{pe_val:.1f}x")
                                        fund_data['Assessment'].append('🔴 Very Expensive')
                                
                                # EPS
                                if 'eps_current' in stock.index and pd.notna(stock['eps_current']):
                                    fund_data['Metric'].append('Current EPS')
                                    fund_data['Value'].append(f"₹{stock['eps_current']:.2f}")
                                    fund_data['Assessment'].append('📊 Earnings/Share')
                                
                                # EPS Change
                                if 'eps_change_pct' in stock.index and pd.notna(stock['eps_change_pct']):
                                    fund_data['Metric'].append('EPS Growth')
                                    eps_chg = stock['eps_change_pct']
                                    
                                    if eps_chg >= 100:
                                        fund_data['Value'].append(f"{eps_chg:+.0f}%")
                                        fund_data['Assessment'].append('🚀 Explosive Growth')
                                    elif eps_chg >= 50:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🔥 High Growth')
                                    elif eps_chg >= 20:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🟢 Good Growth')
                                    elif eps_chg >= 0:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🟡 Modest Growth')
                                    else:
                                        fund_data['Value'].append(f"{eps_chg:+.1f}%")
                                        fund_data['Assessment'].append('🔴 Declining')
                                
                                if fund_data['Metric']:
                                    fund_df = pd.DataFrame(fund_data)
                                    st.dataframe(
                                        fund_df,
                                        width="stretch",
                                        hide_index=True,
                                        column_config={
                                            'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                            'Value': st.column_config.TextColumn('Value', width="small"),
                                            'Assessment': st.column_config.TextColumn('Assessment', width="medium")
                                        }
                                    )
                                else:
                                    st.info("No fundamental data available")
                            else:
                                st.info("Enable 'Hybrid' display mode to see fundamental data")
                        
                        with detail_tabs[3]:  # Technicals
                            st.markdown("**🔍 Technical Analysis**")
                            
                            tech_col1, tech_col2 = st.columns(2)
                            
                            with tech_col1:
                                st.markdown("**📊 52-Week Range**")
                                range_data = {
                                    'Metric': [],
                                    'Value': []
                                }
                                
                                if 'low_52w' in stock.index and pd.notna(stock['low_52w']):
                                    range_data['Metric'].append('52W Low')
                                    range_data['Value'].append(f"₹{stock['low_52w']:,.0f}")
                                
                                if 'high_52w' in stock.index and pd.notna(stock['high_52w']):
                                    range_data['Metric'].append('52W High')
                                    range_data['Value'].append(f"₹{stock['high_52w']:,.0f}")
                                
                                if 'from_low_pct' in stock.index and pd.notna(stock['from_low_pct']):
                                    range_data['Metric'].append('From Low')
                                    range_data['Value'].append(f"{stock['from_low_pct']:.0f}%")
                                
                                if 'from_high_pct' in stock.index and pd.notna(stock['from_high_pct']):
                                    range_data['Metric'].append('From High')
                                    range_data['Value'].append(f"{stock['from_high_pct']:.0f}%")
                                
                                if range_data['Metric']:
                                    range_df = pd.DataFrame(range_data)
                                    st.dataframe(
                                        range_df,
                                        width="stretch",
                                        hide_index=True,
                                        column_config={
                                            'Metric': st.column_config.TextColumn('Metric', width="medium"),
                                            'Value': st.column_config.TextColumn('Value', width="medium")
                                        }
                                    )
                            
                            with tech_col2:
                                st.markdown("**📈 Moving Averages**")
                                sma_data = {
                                    'SMA': [],
                                    'Value': [],
                                    'Position': []
                                }
                                
                                current_price = stock.get('price', 0)
                                
                                for sma_col, sma_label in [('sma_20d', '20 DMA'), ('sma_50d', '50 DMA'), ('sma_200d', '200 DMA')]:
                                    if sma_col in stock.index and pd.notna(stock[sma_col]) and stock[sma_col] > 0:
                                        sma_value = stock[sma_col]
                                        sma_data['SMA'].append(sma_label)
                                        sma_data['Value'].append(f"₹{sma_value:,.0f}")
                                        
                                        if current_price > sma_value:
                                            pct_diff = ((current_price - sma_value) / sma_value) * 100
                                            sma_data['Position'].append(f"🟢 +{pct_diff:.1f}%")
                                        else:
                                            pct_diff = ((sma_value - current_price) / sma_value) * 100
                                            sma_data['Position'].append(f"🔴 -{pct_diff:.1f}%")
                                
                                if sma_data['SMA']:
                                    sma_df = pd.DataFrame(sma_data)
                                    st.dataframe(
                                        sma_df,
                                        width="stretch",
                                        hide_index=True,
                                        column_config={
                                            'SMA': st.column_config.TextColumn('SMA', width="small"),
                                            'Value': st.column_config.TextColumn('Value', width="medium"),
                                            'Position': st.column_config.TextColumn('Position', width="small")
                                        }
                                    )
                            
                            # Trend Analysis
                            if 'trend_quality' in stock.index and pd.notna(stock['trend_quality']):
                                tq = stock['trend_quality']
                                if tq >= 85:
                                    trend_status = f"🔥 Exceptional ({tq:.0f})"
                                    trend_color = "success"
                                elif tq >= 70:
                                    trend_status = f"🚀 Strong ({tq:.0f})"
                                    trend_color = "success"
                                elif tq >= 55:
                                    trend_status = f"✅ Good ({tq:.0f})"
                                    trend_color = "success"
                                elif tq >= 40:
                                    trend_status = f"➡️ Neutral ({tq:.0f})"
                                    trend_color = "warning"
                                elif tq >= 25:
                                    trend_status = f"⚠️ Weak ({tq:.0f})"
                                    trend_color = "warning"
                                else:
                                    trend_status = f"🔻 Poor ({tq:.0f})"
                                    trend_color = "error"
                                
                                getattr(st, trend_color)(f"**Trend Status:** {trend_status}")
                        
                        with detail_tabs[4]:  # Volume Analysis
                            st.markdown("**📊 Volume Analysis**")
                            
                            vol_col1, vol_col2 = st.columns(2)
                            
                            with vol_col1:
                                st.markdown("**📈 Current Volume Metrics**")
                                volume_data = {
                                    'Metric': [],
                                    'Value': []
                                }
                                
                                # Current Volume
                                if 'volume_1d' in stock.index and pd.notna(stock['volume_1d']):
                                    vol_val = stock['volume_1d']
                                    if vol_val >= 1_000_000:
                                        vol_str = f"{vol_val/1_000_000:.1f}M"
                                    elif vol_val >= 1_000:
                                        vol_str = f"{vol_val/1_000:.0f}K" 
                                    else:
                                        vol_str = f"{vol_val:.0f}"
                                    volume_data['Metric'].append('Current Volume')
                                    volume_data['Value'].append(vol_str)
                                
                                # Relative Volume (RVOL)
                                if 'rvol' in stock.index and pd.notna(stock['rvol']):
                                    rvol_val = stock['rvol']
                                    volume_data['Metric'].append('Relative Volume')
                                    volume_data['Value'].append(f"{rvol_val:.2f}x")
                                    
                                    # Volume interpretation
                                    if rvol_val >= 10:
                                        vol_status = "🌋 Volcanic Activity"
                                        vol_color = "error"
                                    elif rvol_val >= 5:
                                        vol_status = "💥 Explosive Volume"
                                        vol_color = "error" 
                                    elif rvol_val >= 2:
                                        vol_status = "🔥 High Activity"
                                        vol_color = "warning"
                                    elif rvol_val >= 1.5:
                                        vol_status = "📈 Growing Interest"
                                        vol_color = "success"
                                    elif rvol_val >= 0.5:
                                        vol_status = "➡️ Normal Activity"
                                        vol_color = "info"
                                    else:
                                        vol_status = "😴 Low Activity"
                                        vol_color = "warning"
                                    
                                    getattr(st, vol_color)(f"**Volume Status:** {vol_status}")
                                
                                # VMI (Volume Momentum Index)
                                if 'vmi' in stock.index and pd.notna(stock['vmi']):
                                    vmi_val = stock['vmi']
                                    volume_data['Metric'].append('VMI')
                                    volume_data['Value'].append(f"{vmi_val:.2f}")
                                
                                # Volume Tier
                                if 'volume_tier' in stock.index and pd.notna(stock['volume_tier']):
                                    volume_data['Metric'].append('Volume Tier')
                                    volume_data['Value'].append(stock['volume_tier'])
                                
                                if volume_data['Metric']:
                                    vol_df = pd.DataFrame(volume_data)
                                    st.dataframe(
                                        vol_df,
                                        width='stretch',
                                        hide_index=True
                                    )
                                else:
                                    st.info("No volume data available")
                            
                            with vol_col2:
                                st.markdown("**📊 Volume Ratios & Trends**")
                                
                                ratio_data = {
                                    'Period': [],
                                    'Ratio': [],
                                    'Status': []
                                }
                                
                                # Volume ratios
                                volume_ratios = [
                                    ('vol_ratio_1d_90d', '1D vs 90D'),
                                    ('vol_ratio_7d_90d', '7D vs 90D'),
                                    ('vol_ratio_30d_90d', '30D vs 90D'),
                                    ('vol_ratio_1d_180d', '1D vs 180D'),
                                    ('vol_ratio_7d_180d', '7D vs 180D'),
                                    ('vol_ratio_30d_180d', '30D vs 180D'),
                                    ('vol_ratio_90d_180d', '90D vs 180D')
                                ]
                                
                                for col_name, display_name in volume_ratios:
                                    if col_name in stock.index and pd.notna(stock[col_name]):
                                        ratio_val = stock[col_name]
                                        ratio_data['Period'].append(display_name)
                                        ratio_data['Ratio'].append(f"{ratio_val:.2f}")
                                        
                                        # Status interpretation
                                        if ratio_val >= 2.0:
                                            status = "🔥 Very High"
                                        elif ratio_val >= 1.5:
                                            status = "📈 High"
                                        elif ratio_val >= 1.2:
                                            status = "➕ Above Normal"
                                        elif ratio_val >= 0.8:
                                            status = "➡️ Normal"
                                        elif ratio_val >= 0.5:
                                            status = "➖ Below Normal"
                                        else:
                                            status = "📉 Low"
                                        
                                        ratio_data['Status'].append(status)
                                
                                if ratio_data['Period']:
                                    ratio_df = pd.DataFrame(ratio_data)
                                    st.dataframe(
                                        ratio_df,
                                        width='stretch',
                                        hide_index=True
                                    )
                                else:
                                    st.info("No volume ratio data available")
                            
                            # Volume Score Section
                            if 'volume_score' in stock.index and pd.notna(stock['volume_score']):
                                st.markdown("---")
                                st.markdown("**🎯 Volume Score Analysis**")
                                
                                vol_score = stock['volume_score']
                                score_col1, score_col2, score_col3 = st.columns(3)
                                
                                with score_col1:
                                    if vol_score >= 80:
                                        score_status = "🔥 Excellent"
                                        score_color = "success"
                                    elif vol_score >= 60:
                                        score_status = "✅ Good"
                                        score_color = "success"
                                    elif vol_score >= 40:
                                        score_status = "⚠️ Average"
                                        score_color = "warning"
                                    else:
                                        score_status = "❌ Poor"
                                        score_color = "error"
                                    
                                    st.metric("Volume Score", f"{vol_score:.0f}/100")
                                    getattr(st, score_color)(f"**Status:** {score_status}")
                                
                                with score_col2:
                                    # VMI Tier Classification
                                    if 'vmi_tier' in stock.index and pd.notna(stock['vmi_tier']):
                                        st.markdown("**VMI Classification**")
                                        st.info(f"📊 {stock['vmi_tier']}")
                                
                                with score_col3:
                                    # Volume Activity Level
                                    if 'rvol' in stock.index and pd.notna(stock['rvol']):
                                        rvol_val = stock['rvol']
                                        st.markdown("**Activity Level**")
                                        
                                        if rvol_val >= 2.0:
                                            activity = "🔥 High Activity"
                                        elif rvol_val >= 1.0:
                                            activity = "📈 Normal+"
                                        elif rvol_val >= 0.5:
                                            activity = "➡️ Normal"
                                        else:
                                            activity = "😴 Low"
                                        
                                        st.info(f"📊 {activity}")
                            
                            # Liquidity Analysis
                            if all(col in stock.index for col in ['volume_1d', 'price']) and all(pd.notna(stock[col]) for col in ['volume_1d', 'price']):
                                st.markdown("---")
                                st.markdown("**💧 Liquidity Analysis**")
                                
                                volume = stock['volume_1d']
                                price = stock['price']
                                turnover = volume * price
                                
                                liq_col1, liq_col2 = st.columns(2)
                                
                                with liq_col1:
                                    st.metric("Daily Turnover", f"₹{turnover/10_000_000:.1f}Cr" if turnover >= 10_000_000 else f"₹{turnover/100_000:.1f}L")
                                
                                with liq_col2:
                                    # Liquidity classification
                                    if turnover >= 100_000_000:  # 10Cr+
                                        liq_status = "🌊 Highly Liquid"
                                        liq_color = "success"
                                    elif turnover >= 10_000_000:  # 1Cr+
                                        liq_status = "💧 Good Liquidity"
                                        liq_color = "success"
                                    elif turnover >= 1_000_000:  # 10L+
                                        liq_status = "💦 Moderate Liquidity"
                                        liq_color = "warning"
                                    else:
                                        liq_status = "🏜️ Low Liquidity"
                                        liq_color = "error"
                                    
                                    getattr(st, liq_color)(f"**Liquidity:** {liq_status}")
                        
                        with detail_tabs[5]:  # Advanced Metrics
                            st.markdown("**🎯 Advanced Metrics**")
                            
                            adv_data = {
                                'Metric': [],
                                'Value': [],
                                'Description': []
                            }
                            
                            # VMI
                            if 'vmi' in stock.index and pd.notna(stock['vmi']):
                                adv_data['Metric'].append('VMI')
                                adv_data['Value'].append(f"{stock['vmi']:.2f}")
                                adv_data['Description'].append('Volume Momentum Index')
                            
                            # Position Tension
                            if 'position_tension' in stock.index and pd.notna(stock['position_tension']):
                                adv_data['Metric'].append('Position Tension')
                                adv_data['Value'].append(f"{stock['position_tension']:.0f}")
                                adv_data['Description'].append('Range position stress')
                            
                            # Momentum Harmony
                            if 'momentum_harmony' in stock.index and pd.notna(stock['momentum_harmony']):
                                harmony_val = int(stock['momentum_harmony'])
                                harmony_emoji = "🟢" if harmony_val >= 3 else "🟡" if harmony_val >= 2 else "🔴"
                                adv_data['Metric'].append('Momentum Harmony')
                                adv_data['Value'].append(f"{harmony_emoji} {harmony_val}/4")
                                adv_data['Description'].append('Multi-timeframe alignment')
                            
                            # Money Flow
                            if 'money_flow_mm' in stock.index and pd.notna(stock['money_flow_mm']):
                                adv_data['Metric'].append('Money Flow')
                                adv_data['Value'].append(f"₹{stock['money_flow_mm']:.1f}M")
                                adv_data['Description'].append('Price × Volume × RVOL')

                            
                            # Overall Market Strength
                            if 'overall_market_strength' in stock.index and pd.notna(stock['overall_market_strength']):
                                adv_data['Metric'].append('Market Strength')
                                adv_data['Value'].append(f"{stock['overall_market_strength']:.1f}")
                                adv_data['Description'].append('Combined momentum, acceleration & breakout strength')

                            # Long Term Strength
                            if 'long_term_strength' in stock.index and pd.notna(stock['long_term_strength']):
                                adv_data['Metric'].append('Long Term Strength')
                                adv_data['Value'].append(f"{stock['long_term_strength']:.1f}")
                                adv_data['Description'].append('Long-term trend consistency & momentum harmony')
                            
                            # Pattern Confidence
                            if 'pattern_confidence' in stock.index and pd.notna(stock['pattern_confidence']):
                                adv_data['Metric'].append('Pattern Confidence')
                                adv_data['Value'].append(f"{stock['pattern_confidence']:.1f}%")
                                adv_data['Description'].append('Pattern strength score')
                            
                            if adv_data['Metric']:
                                adv_df = pd.DataFrame(adv_data)
                                st.dataframe(
                                    adv_df,
                                    width="stretch",
                                    hide_index=True,
                                    column_config={
                                        'Metric': st.column_config.TextColumn(
                                            'Metric',
                                            help="Advanced metric name",
                                            width="medium"
                                        ),
                                        'Value': st.column_config.TextColumn(
                                            'Value',
                                            help="Metric value",
                                            width="small"
                                        ),
                                        'Description': st.column_config.TextColumn(
                                            'Description',
                                            help="What this metric measures",
                                            width="large"
                                        )
                                    }
                                )
                            else:
                                st.info("No advanced metrics available")
            
            else:
                st.warning("No stocks found matching your search criteria.")
                
                # Provide search suggestions
                st.markdown("#### 💡 Search Tips:")
                st.markdown("""
                - **Ticker Search:** Enter exact ticker symbol (e.g., RELIANCE, TCS, INFY)
                - **Company Search:** Enter part of company name (e.g., Tata, Infosys, Reliance)
                - **Partial Match:** Search works with partial text (e.g., 'REL' finds RELIANCE)
                - **Case Insensitive:** Search is not case-sensitive
                """)
        
        else:
            # Show search instructions when no search is active
            st.info("Enter a ticker symbol or company name to search")
            
            # Show top performers as suggestions
            st.markdown("#### 🏆 Today's Top Performers")
            
            if not filtered_df.empty:
                top_performers = filtered_df.nlargest(5, 'master_score')[['ticker', 'company_name', 'master_score', 'ret_1d', 'rvol']]
                
                suggestions_data = []
                for _, row in top_performers.iterrows():
                    suggestions_data.append({
                        'Ticker': row['ticker'],
                        'Company': row['company_name'][:30] + '...' if len(row['company_name']) > 30 else row['company_name'],
                        'Score': row['master_score'],
                        '1D Return': f"{row['ret_1d']:+.1f}%" if pd.notna(row['ret_1d']) else '-',
                        'RVOL': f"{row['rvol']:.1f}x" if pd.notna(row['rvol']) else '-'
                    })
                
                suggestions_df = pd.DataFrame(suggestions_data)
                
                st.dataframe(
                    suggestions_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        'Ticker': st.column_config.TextColumn('Ticker', width="small"),
                        'Company': st.column_config.TextColumn('Company', width="large"),
                        'Score': st.column_config.ProgressColumn(
                            'Score',
                            format="%.1f",
                            min_value=0,
                            max_value=100,
                            width="small"
                        ),
                        '1D Return': st.column_config.TextColumn('1D Return', width="small"),
                        'RVOL': st.column_config.TextColumn('RVOL', width="small")
                    }
                )
                
                st.caption("💡 Tip: Click on any ticker above and copy it to search")    
                
    with tabs[5]:
        st.markdown("### 📥 Export Data")
        
        st.markdown("#### 📋 Export Templates")
        export_template = st.radio(
            "Choose export template:",
            options=[
                "Full Analysis (All Data)",
                "Day Trader Focus",
                "Swing Trader Focus",
                "Investor Focus"
            ],
            key="export_template_radio",
            help="Select a template based on your trading style"
        )
        
        template_map = {
            "Full Analysis (All Data)": "full",
            "Day Trader Focus": "day_trader",
            "Swing Trader Focus": "swing_trader",
            "Investor Focus": "investor"
        }
        
        selected_template = template_map[export_template]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Excel Report")
            st.markdown(
                "Comprehensive multi-sheet report including:\n"
                "- Top 100 stocks with all scores\n"
                "- Market intelligence dashboard\n"
                "- Sector rotation analysis\n"
                "- Pattern frequency analysis\n"
                "- Wave Radar signals\n"
                "- Summary statistics"
            )
            
            if st.button("Generate Excel Report", type="primary", width="stretch"):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    with st.spinner("Creating Excel report..."):
                        try:
                            excel_file = ExportEngine.create_excel_report(
                                filtered_df, template=selected_template
                            )
                            
                            st.download_button(
                                label="📥 Download Excel Report",
                                data=excel_file,
                                file_name=f"wave_detection_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.success("Excel report generated successfully!")
                            
                        except Exception as e:
                            st.error(f"Error generating Excel report: {str(e)}")
                            logger.error(f"Excel export error: {str(e)}", exc_info=True)
        
        with col2:
            st.markdown("#### 📄 CSV Export")
            st.markdown(
                "Enhanced CSV format with:\n"
                "- All ranking scores\n"
                "- Advanced metrics (VMI, Money Flow)\n"
                "- Pattern detections\n"
                "- Market states\n"
                "- Category classifications\n"
                "- Optimized for further analysis"
            )
            
            if st.button("Generate CSV Export", width="stretch"):
                if len(filtered_df) == 0:
                    st.error("No data to export. Please adjust your filters.")
                else:
                    try:
                        csv_data = ExportEngine.create_csv_export(filtered_df)
                        
                        st.download_button(
                            label="📥 Download CSV File",
                            data=csv_data,
                            file_name=f"wave_detection_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        st.success("CSV export generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Error generating CSV: {str(e)}")
                        logger.error(f"CSV export error: {str(e)}", exc_info=True)
        
        st.markdown("---")
        st.markdown("#### 📊 Export Preview")
        
        export_stats = {
            "Total Stocks": len(filtered_df),
            "Average Score": f"{filtered_df['master_score'].mean():.1f}" if not filtered_df.empty else "N/A",
            "Stocks with Patterns": (filtered_df['patterns'] != '').sum() if 'patterns' in filtered_df.columns else 0,
            "High RVOL (>2x)": (filtered_df['rvol'] > 2).sum() if 'rvol' in filtered_df.columns else 0,
            "Positive 30D Returns": (filtered_df['ret_30d'] > 0).sum() if 'ret_30d' in filtered_df.columns else 0,
            "Data Quality": f"{st.session_state.data_quality.get('completeness', 0):.1f}%"
        }
        
        stat_cols = st.columns(3)
        for i, (label, value) in enumerate(export_stats.items()):
            with stat_cols[i % 3]:
                UIComponents.render_metric_card(label, value)
    
    with tabs[6]:
        st.markdown("### ℹ️ About Wave Detection Ultimate 3.0")
        
        # Main content in clean two-column layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎯 System Overview")
            st.markdown("""
            Wave Detection Ultimate 3.0 is a professional-grade stock ranking system designed for institutional-quality market analysis. 
            The system combines advanced technical analysis, volume dynamics, and pattern recognition to identify high-potential 
            investment opportunities.
            
            #### 📊 Master Score 3.0 Algorithm
            Our proprietary ranking algorithm evaluates stocks across six key dimensions:
            
            - **Position Analysis (30%)** - 52-week range positioning and momentum
            - **Volume Dynamics (25%)** - Multi-timeframe volume pattern analysis  
            - **Momentum Tracking (15%)** - 30-day price momentum measurement
            - **Acceleration Detection (10%)** - Momentum acceleration signals
            - **Breakout Probability (10%)** - Technical breakout readiness assessment
            - **RVOL Integration (10%)** - Real-time relative volume analysis
            
            #### 🔍 Pattern Detection System
            The system employs 41 sophisticated pattern detection algorithms organized into seven categories:
            
            **Technical Patterns (13)**
            - Core momentum and volume patterns
            - Mathematical indicators (Premium Momentum, Entropy, Velocity)
            - Advanced institutional detection
            
            **Fundamental Patterns (9)** *(Hybrid Mode)*
            - Value momentum analysis
            - Earnings-based patterns  
            - Quality and growth indicators
            
            **Market Intelligence (19)**
            - Range analysis patterns
            - Reversal and rotation detection
            - Market psychology indicators
            
            #### 🧠 Market State Intelligence
            Advanced 8-regime market detection system that adapts scoring based on current market conditions:
            - **Bull Markets:** STRONG_UPTREND, UPTREND, PULLBACK
            - **Neutral:** NEUTRAL, UNCLEAR  
            - **Bear Markets:** DISTRIBUTION, DOWNTREND, STRONG_DOWNTREND
            """)
        
        with col2:
            st.markdown("#### ⚡ Performance Specifications")
            
            # Clean performance metrics
            UIComponents.render_metric_card("Initial Load Time", "< 2 seconds")
            UIComponents.render_metric_card("Filter Response", "< 200ms") 
            UIComponents.render_metric_card("Search Speed", "< 50ms")
            UIComponents.render_metric_card("Stock Capacity", "2,000+")
            
            st.markdown("#### 🔧 Technical Features")
            st.markdown("""
            - **Data Sources:** Google Sheets, CSV upload
            - **Caching:** Smart 1-hour cache with validation
            - **UI:** Mobile-responsive design
            - **Export:** Professional Excel templates
            - **Error Handling:** Graceful degradation
            - **Memory:** Optimized for large datasets
            """)
            
            st.markdown("#### 📈 Display Modes")
            st.markdown("""
            **Technical Mode** (Default)
            - Pure momentum analysis
            - Volume pattern focus
            - Technical indicators only
            
            **Hybrid Mode**
            - Technical + fundamental analysis
            - PE ratio evaluation
            - EPS growth tracking
            - Value pattern detection
            """)
            
            st.markdown("#### 🇮🇳 Market Optimization")
            st.markdown("""
            - Currency: ₹ (Indian Rupee)
            - Timezone: IST (Indian Standard Time)
            - Exchanges: NSE/BSE categories
            - Number format: Indian conventions
            """)
        
        # System architecture section
        st.markdown("---")
        st.markdown("#### 🏗️ System Architecture")
        
        arch_col1, arch_col2, arch_col3 = st.columns(3)
        
        with arch_col1:
            st.markdown("""
            **Core Engines**
            - RankingEngine: Score calculation
            - PatternDetector: 41-pattern system
            - DataValidator: Quality assurance
            - FilterEngine: Real-time filtering
            """)
        
        with arch_col2:
            st.markdown("""
            **Management Layer**
            - SessionStateManager: State persistence
            - CacheManager: Performance optimization
            - ConfigManager: Dynamic configuration
            - UIComponents: Responsive interface
            """)
        
        with arch_col3:
            st.markdown("""
            **Analytics Modules**
            - MarketStateAnalyzer: Regime detection
            - VolumeAnalyzer: Volume dynamics
            - MomentumEngine: Trend analysis
            - ExportManager: Professional reports
            """)
        
        # Current session statistics
        st.markdown("---")
        st.markdown("#### 📊 Current Session")
        
        session_cols = st.columns(4)
        
        with session_cols[0]:
            UIComponents.render_metric_card(
                "Stocks Loaded",
                f"{len(ranked_df):,}" if 'ranked_df' in locals() else "0"
            )
        
        with session_cols[1]:
            UIComponents.render_metric_card(
                "Filtered Results",
                f"{len(filtered_df):,}" if 'filtered_df' in locals() else "0"
            )
        
        with session_cols[2]:
            data_quality = st.session_state.data_quality.get('completeness', 0)
            quality_emoji = "🟢" if data_quality > 80 else "🟡" if data_quality > 60 else "🔴"
            UIComponents.render_metric_card(
                "Data Quality",
                f"{quality_emoji} {data_quality:.1f}%"
            )
        
        with session_cols[3]:
            cache_time = datetime.now(timezone.utc) - st.session_state.last_refresh
            minutes = int(cache_time.total_seconds() / 60)
            cache_emoji = "🟢" if minutes < 60 else "🔴"
            UIComponents.render_metric_card(
                "Cache Status",
                f"{cache_emoji} {minutes}min ago"
            )
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="
            text-align: center; 
            color: #666; 
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 5px;
            margin-top: 2rem;
        ">
            <strong>🌊 Wave Detection Ultimate 3.0</strong><br>
            <small>Professional Stock Ranking System for Institutional-Grade Market Analysis</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical Application Error: {str(e)}")
        logger.error(f"Application crashed: {str(e)}", exc_info=True)
        
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📧 Report Issue"):
            st.info("Please take a screenshot and report this error.")

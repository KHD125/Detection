"""
Wave Detection Ultimate 3.0 - FINAL PRODUCTION VERSION
======================================================
Professional Stock Ranking System - Clean & Powerful
Based on proven V2 OLD VERSION structure with enhanced features

🎯 PHILOSOPHY: Clean, Direct, Effective
📊 FOCUS: Rankings-first approach with smart filtering
🚀 FEATURES: Advanced patterns, ML insights, production-ready

Version: 3.0.0-FINAL
Last Updated: August 2025
Status: PRODUCTION READY
"""

# ============================================
# IMPORTS AND SETUP
# ============================================

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from functools import wraps
import time
from io import BytesIO
import warnings
import gc
import re
import json

# Production settings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')
np.random.seed(42)

# ============================================
# LOGGING CONFIGURATION
# ============================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================
# CONFIGURATION
# ============================================

@dataclass(frozen=True)
class Config:
    """System configuration - Clean and focused"""
    
    # Data source settings
    DEFAULT_SHEET_URL: str = ""
    DEFAULT_GID: str = "1823439984"
    
    # Cache settings
    CACHE_TTL: int = 3600
    STALE_DATA_HOURS: int = 24
    
    # Master Score weights (total = 100%)
    POSITION_WEIGHT: float = 0.30
    VOLUME_WEIGHT: float = 0.25
    MOMENTUM_WEIGHT: float = 0.15
    ACCELERATION_WEIGHT: float = 0.10
    BREAKOUT_WEIGHT: float = 0.10
    RVOL_WEIGHT: float = 0.10
    
    # Display settings
    DEFAULT_TOP_N: int = 50
    AVAILABLE_TOP_N: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200, 500])
    
    # Critical columns
    CRITICAL_COLUMNS: List[str] = field(default_factory=lambda: ['ticker', 'price', 'volume_1d'])
    
    # Important columns for full functionality
    IMPORTANT_COLUMNS: List[str] = field(default_factory=lambda: [
        'category', 'sector', 'industry', 'rvol', 'pe', 'eps_current', 'eps_change_pct',
        'sma_20d', 'sma_50d', 'sma_200d', 'ret_1d', 'ret_7d', 'ret_30d', 
        'from_low_pct', 'from_high_pct'
    ])
    
    # Pattern thresholds
    PATTERN_THRESHOLDS: Dict[str, float] = field(default_factory=lambda: {
        "category_leader": 90,
        "hidden_gem": 80,
        "acceleration": 85,
        "institutional": 75,
        "vol_explosion": 95,
        "momentum_wave": 80,
        "quality_leader": 85,
        "breakout_ready": 75
    })
    
    def validate_weights(self) -> bool:
        """Validate weights sum to 1.0"""
        total = (self.POSITION_WEIGHT + self.VOLUME_WEIGHT + self.MOMENTUM_WEIGHT + 
                self.ACCELERATION_WEIGHT + self.BREAKOUT_WEIGHT + self.RVOL_WEIGHT)
        return abs(total - 1.0) < 0.001

# Initialize configuration
CONFIG = Config()

# Validate configuration on startup
if not CONFIG.validate_weights():
    logger.error("Configuration validation failed: Weights do not sum to 1.0")
    raise ValueError("Invalid configuration")

logger.info("✅ Wave Detection Ultimate 3.0 - System initialized")

# ============================================
# PERFORMANCE MONITORING
# ============================================

def performance_tracked(operation_name: str):
    """Simple performance tracking decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.debug(f"⚡ {operation_name}: {elapsed:.3f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"❌ {operation_name} failed after {elapsed:.3f}s: {e}")
                raise
        return wrapper
    return decorator

# ============================================
# DATA VALIDATION
# ============================================

class DataValidator:
    """Production-grade data validation"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """Validate dataframe structure and data quality"""
        errors = []
        warnings = []
        
        if df is None or df.empty:
            errors.append("DataFrame is empty or None")
            return False, errors, warnings
        
        # Print column info for debugging
        logger.info(f"DataFrame columns: {list(df.columns)}")
        
        # Find actual column names (flexible matching)
        actual_columns = df.columns.tolist()
        
        # Map common variations
        column_mapping = {
            'ticker': ['ticker', 'symbol', 'stock_symbol', 'scrip_code'],
            'price': ['price', 'ltp', 'last_price', 'close_price', 'current_price'],
            'volume_1d': ['volume_1d', 'volume', 'vol', 'daily_volume']
        }
        
        # Check for critical columns with flexible matching
        found_critical = {}
        for critical_col in CONFIG.CRITICAL_COLUMNS:
            found = False
            if critical_col in column_mapping:
                for variant in column_mapping[critical_col]:
                    matching_cols = [col for col in actual_columns if variant.lower() in col.lower()]
                    if matching_cols:
                        found_critical[critical_col] = matching_cols[0]
                        found = True
                        break
            else:
                if critical_col in actual_columns:
                    found_critical[critical_col] = critical_col
                    found = True
            
            if not found:
                warnings.append(f"Critical column '{critical_col}' not found")
        
        # Data quality checks - be more lenient
        price_col = found_critical.get('price')
        if price_col and price_col in df.columns:
            # First try to clean a sample to see what we're dealing with
            sample_prices = df[price_col].head(10)
            logger.info(f"Sample price values: {sample_prices.tolist()}")
            
            # Clean the price column
            cleaned_prices = df[price_col].apply(DataValidator.clean_numeric_value)
            price_issues = cleaned_prices.isna().sum()
            
            logger.info(f"Price validation: {len(df) - price_issues}/{len(df)} valid prices")
            
            # Only error if more than 50% are invalid (was 10%)
            if price_issues > len(df) * 0.5:
                errors.append(f"Too many invalid price values: {price_issues}/{len(df)}")
            elif price_issues > len(df) * 0.1:
                warnings.append(f"Some invalid price values: {price_issues}/{len(df)}")
        
        return len(errors) == 0, errors, warnings
    
    @staticmethod
    def clean_numeric_value(value, is_percentage: bool = False) -> float:
        """Clean and convert numeric values including Indian formats"""
        if pd.isna(value) or value == '' or value is None:
            return np.nan
            
        if isinstance(value, (int, float)):
            return float(value)
            
        if isinstance(value, str):
            # Remove common prefixes and clean
            cleaned = str(value).strip()
            
            # Handle common non-numeric indicators
            if cleaned.lower() in ['na', 'n/a', '-', 'nil', 'null', '']:
                return np.nan
            
            # Remove currency symbols and spaces
            cleaned = re.sub(r'[₹$,\s]', '', cleaned)
            
            # Handle negative values in parentheses
            if cleaned.startswith('(') and cleaned.endswith(')'):
                cleaned = '-' + cleaned[1:-1]
            
            # Handle Indian formats (Crores, Lakhs)
            if any(x in cleaned.lower() for x in ['cr', 'crore']):
                number = re.sub(r'[Ccrore\s]', '', cleaned, flags=re.IGNORECASE)
                try:
                    return float(number) * 10000000  # 1 Crore = 10 Million
                except:
                    return np.nan
                    
            if any(x in cleaned.lower() for x in ['l', 'lakh']):
                number = re.sub(r'[Llakh\s]', '', cleaned, flags=re.IGNORECASE)
                try:
                    return float(number) * 100000  # 1 Lakh = 100K
                except:
                    return np.nan
            
            # Handle percentages
            if '%' in cleaned:
                number = cleaned.replace('%', '')
                try:
                    return float(number)
                except:
                    return np.nan
            
            # Handle scientific notation
            if 'e' in cleaned.lower():
                try:
                    return float(cleaned)
                except:
                    return np.nan
                    
            # Regular number - try direct conversion
            try:
                return float(cleaned)
            except:
                # Last resort - extract numeric part
                numeric_match = re.search(r'-?\d+\.?\d*', cleaned)
                if numeric_match:
                    try:
                        return float(numeric_match.group())
                    except:
                        return np.nan
                return np.nan
                
        return np.nan

# ============================================
# SESSION STATE MANAGEMENT
# ============================================

class SessionStateManager:
    """Clean session state management"""
    
    @staticmethod
    def initialize():
        """Initialize session state with defaults"""
        defaults = {
            'data_source': 'sheet',
            'ranked_df': pd.DataFrame(),
            'data_timestamp': None,
            'last_refresh': None,
            'show_debug': False,
            'active_filter_count': 0,
            'quick_filter_applied': False,
            'quick_filter': None,
            # Filter states
            'category_filter': [],
            'sector_filter': [],
            'min_score': 0,
            'patterns': [],
            'min_rvol': 1.0,
            'show_fundamentals': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def clear_filters():
        """Clear all active filters"""
        filter_keys = [
            'category_filter', 'sector_filter', 'min_score', 'patterns',
            'min_rvol', 'quick_filter_applied', 'quick_filter'
        ]
        
        for key in filter_keys:
            if key in st.session_state:
                if key in ['category_filter', 'sector_filter', 'patterns']:
                    st.session_state[key] = []
                elif key in ['min_score', 'min_rvol']:
                    st.session_state[key] = 0 if key == 'min_score' else 1.0
                else:
                    st.session_state[key] = False if 'applied' in key else None

# ============================================
# DATA LOADING ENGINE
# ============================================

class DataLoader:
    """Clean data loading from multiple sources"""
    
    @staticmethod
    @performance_tracked("data_loading")
    def load_from_sheets(sheet_id: str, gid: str = None) -> pd.DataFrame:
        """Load data from Google Sheets with improved error handling"""
        try:
            # Clean sheet_id - extract from URL if needed
            if 'docs.google.com' in sheet_id:
                # Extract sheet ID from full URL
                match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_id)
                if match:
                    sheet_id = match.group(1)
                else:
                    raise ValueError("Cannot extract sheet ID from URL")
            
            # Build URL
            if gid and gid != '0':
                url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            else:
                url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
                
            logger.info(f"Loading from URL: {url}")
            
            # Load with timeout and error handling
            import requests
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Check if we got HTML (error page) instead of CSV
            content_type = response.headers.get('content-type', '')
            if 'html' in content_type:
                raise ValueError("Received HTML instead of CSV - check sheet permissions or ID")
            
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            
            if df.empty:
                raise ValueError("Sheet appears to be empty")
                
            logger.info(f"📊 Loaded {len(df)} rows from Google Sheets")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error loading from Google Sheets: {e}")
            raise Exception(f"Network error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to load from Google Sheets: {e}")
            raise Exception(f"Google Sheets loading failed: {str(e)}")
    
    @staticmethod
    @performance_tracked("data_loading")
    def load_from_upload(uploaded_file) -> pd.DataFrame:
        """Load data from uploaded CSV"""
        try:
            df = pd.read_csv(uploaded_file)
            logger.info(f"📊 Loaded {len(df)} rows from uploaded file")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load uploaded file: {e}")
            raise Exception(f"File upload failed: {str(e)}")
    
    @staticmethod
    def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to expected format"""
        
        # Common column mappings
        column_mapping = {
            # Symbol/Ticker variations
            'symbol': ['ticker', 'stock_symbol', 'scrip_code', 'symbol', 'scrip'],
            # Price variations  
            'price': ['ltp', 'last_price', 'close_price', 'current_price', 'price'],
            # Company name variations
            'company_name': ['name', 'company', 'stock_name', 'company_name'],
            # Volume variations
            'volume_1d': ['volume', 'vol', 'daily_volume', 'volume_1d'],
            # Return variations
            'ret_1d': ['1d_return', 'daily_return', 'ret_1d', '1d%'],
            'ret_7d': ['7d_return', 'weekly_return', 'ret_7d', '7d%'],
            'ret_30d': ['30d_return', 'monthly_return', 'ret_30d', '30d%'],
            # Other important fields
            'rvol': ['relative_volume', 'rvol', 'rel_vol'],
            'from_low_pct': ['from_52w_low', 'from_low', 'from_low_pct'],
            'from_high_pct': ['from_52w_high', 'from_high', 'from_high_pct'],
            'pe': ['pe_ratio', 'pe', 'p/e'],
            'category': ['market_cap', 'cap', 'category'],
            'sector': ['sector', 'industry_sector'],
            'industry': ['industry', 'sub_sector']
        }
        
        # Create reverse mapping
        normalized_df = df.copy()
        original_columns = df.columns.tolist()
        
        for target_col, variations in column_mapping.items():
            found = False
            for variation in variations:
                # Look for exact match (case insensitive)
                matching_cols = [col for col in original_columns if col.lower() == variation.lower()]
                if matching_cols:
                    normalized_df = normalized_df.rename(columns={matching_cols[0]: target_col})
                    found = True
                    break
                
                # Look for partial match if no exact match
                if not found:
                    matching_cols = [col for col in original_columns if variation.lower() in col.lower()]
                    if matching_cols:
                        normalized_df = normalized_df.rename(columns={matching_cols[0]: target_col})
                        found = True
                        break
        
        # Log column mapping
        renamed_cols = []
        for old_col, new_col in zip(df.columns, normalized_df.columns):
            if old_col != new_col:
                renamed_cols.append(f"{old_col} → {new_col}")
        
        if renamed_cols:
            logger.info(f"Column mapping: {'; '.join(renamed_cols)}")
        
        return normalized_df

# ============================================
# ADVANCED METRICS CALCULATOR
# ============================================

class AdvancedMetrics:
    """Calculate advanced metrics and indicators from OLD VERSION"""
    
    @staticmethod
    @performance_tracked("advanced_metrics")
    def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive advanced metrics"""
        
        # Momentum Indicators
        df['momentum_strength'] = PatternDetector._calculate_momentum_strength(df)
        df['volume_profile'] = PatternDetector._calculate_volume_profile(df)
        df['price_velocity'] = PatternDetector._calculate_price_velocity(df)
        df['trend_consistency'] = PatternDetector._calculate_trend_consistency(df)
        
        # Risk Metrics
        df['volatility_score'] = PatternDetector._calculate_volatility_score(df)
        df['liquidity_score'] = PatternDetector._calculate_liquidity_score(df)
        
        # Quality Scores
        df['fundamental_score'] = PatternDetector._calculate_fundamental_score(df)
        df['technical_score'] = PatternDetector._calculate_technical_score(df)
        
        logger.info("✅ Advanced metrics calculated")
        return df
    
    @staticmethod
    def _calculate_momentum_strength(df: pd.DataFrame) -> pd.Series:
        """Calculate momentum strength score"""
        momentum_scores = []
        
        for _, row in df.iterrows():
            ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce')
            ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce')
            ret_30d = pd.to_numeric(row.get('ret_30d', 0), errors='coerce')
            rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
            
            # Weighted momentum calculation
            momentum = (ret_1d * 0.4 + ret_7d * 0.4 + ret_30d * 0.2) * np.log(rvol + 1)
            momentum_scores.append(min(100, max(0, momentum + 50)))
            
        return pd.Series(momentum_scores)
    
    @staticmethod
    def _calculate_volume_profile(df: pd.DataFrame) -> pd.Series:
        """Calculate volume profile score"""
        volume_scores = []
        
        for _, row in df.iterrows():
            rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
            volume = pd.to_numeric(row.get('volume_1d', 0), errors='coerce')
            
            # Volume profile calculation
            if volume > 1000000:  # High volume
                volume_score = min(100, rvol * 30)
            elif volume > 100000:  # Medium volume
                volume_score = min(80, rvol * 25)
            else:  # Low volume
                volume_score = min(60, rvol * 20)
                
            volume_scores.append(volume_score)
            
        return pd.Series(volume_scores)
    
    @staticmethod
    def _calculate_price_velocity(df: pd.DataFrame) -> pd.Series:
        """Calculate price velocity (rate of change)"""
        velocity_scores = []
        
        for _, row in df.iterrows():
            ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce')
            ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce')
            
            # Acceleration calculation
            daily_avg = ret_7d / 7
            acceleration = ret_1d - daily_avg
            velocity = min(100, max(0, acceleration * 10 + 50))
            velocity_scores.append(velocity)
            
        return pd.Series(velocity_scores)
    
    @staticmethod
    def _calculate_trend_consistency(df: pd.DataFrame) -> pd.Series:
        """Calculate trend consistency score"""
        consistency_scores = []
        
        for _, row in df.iterrows():
            ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce')
            ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce')
            ret_30d = pd.to_numeric(row.get('ret_30d', 0), errors='coerce')
            
            # Check trend alignment
            positive_trend = sum([ret_1d > 0, ret_7d > 0, ret_30d > 0])
            consistency = (positive_trend / 3) * 100
            consistency_scores.append(consistency)
            
        return pd.Series(consistency_scores)
    
    @staticmethod
    def _calculate_volatility_score(df: pd.DataFrame) -> pd.Series:
        """Calculate volatility score"""
        volatility_scores = []
        
        for _, row in df.iterrows():
            ret_1d = abs(pd.to_numeric(row.get('ret_1d', 0), errors='coerce'))
            ret_7d = abs(pd.to_numeric(row.get('ret_7d', 0), errors='coerce'))
            
            # Volatility calculation (lower volatility = higher score)
            avg_volatility = (ret_1d + ret_7d / 7) / 2
            volatility_score = max(0, 100 - avg_volatility * 5)
            volatility_scores.append(volatility_score)
            
        return pd.Series(volatility_scores)
    
    @staticmethod
    def _calculate_liquidity_score(df: pd.DataFrame) -> pd.Series:
        """Calculate liquidity score"""
        liquidity_scores = []
        
        for _, row in df.iterrows():
            volume = pd.to_numeric(row.get('volume_1d', 0), errors='coerce')
            price = pd.to_numeric(row.get('price', 1), errors='coerce')
            
            # Liquidity based on volume and price
            turnover = volume * price
            if turnover > 100000000:  # 10 Cr+
                liquidity_score = 100
            elif turnover > 10000000:  # 1 Cr+
                liquidity_score = 80
            elif turnover > 1000000:  # 10 L+
                liquidity_score = 60
            else:
                liquidity_score = 40
                
            liquidity_scores.append(liquidity_score)
            
        return pd.Series(liquidity_scores)
    
    @staticmethod
    def _calculate_fundamental_score(df: pd.DataFrame) -> pd.Series:
        """Calculate fundamental score"""
        fundamental_scores = []
        
        for _, row in df.iterrows():
            pe = pd.to_numeric(row.get('pe', 20), errors='coerce')
            eps_change = pd.to_numeric(row.get('eps_change_pct', 0), errors='coerce')
            
            # Fundamental scoring
            pe_score = max(0, 100 - abs(pe - 15) * 2) if pe > 0 else 50
            eps_score = min(100, max(0, eps_change * 2 + 50))
            
            fundamental_score = (pe_score + eps_score) / 2
            fundamental_scores.append(fundamental_score)
            
        return pd.Series(fundamental_scores)
    
    @staticmethod
    def _calculate_technical_score(df: pd.DataFrame) -> pd.Series:
        """Calculate technical score"""
        technical_scores = []
        
        for _, row in df.iterrows():
            from_low = pd.to_numeric(row.get('from_low_pct', 50), errors='coerce')
            from_high = pd.to_numeric(row.get('from_high_pct', 50), errors='coerce')
            rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
            
            # Technical scoring
            position_score = from_low  # Higher from low is better
            breakout_score = 100 - from_high  # Closer to high is better
            volume_score = min(100, rvol * 25)
            
            technical_score = (position_score * 0.4 + breakout_score * 0.3 + volume_score * 0.3)
            technical_scores.append(technical_score)
            
        return pd.Series(technical_scores)

# ============================================
# MARKET INTELLIGENCE ENGINE
# ============================================

class MarketIntelligence:
    """Advanced market analysis and regime detection from OLD VERSION"""
    
    @staticmethod
    @performance_tracked("market_intelligence")
    def analyze_market_regime(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze current market regime"""
        
        if df.empty:
            return {'regime': 'Unknown', 'confidence': 0, 'indicators': {}}
        
        # Calculate market-wide indicators
        avg_return_1d = df['ret_1d'].mean()
        avg_return_7d = df['ret_7d'].mean()
        avg_rvol = df['rvol'].mean()
        high_quality_pct = len(df[df['master_score'] >= 80]) / len(df) * 100
        
        # Pattern distribution
        pattern_counts = PatternDetector.get_pattern_summary(df)
        
        # Regime detection logic
        regime_indicators = {
            'avg_return_1d': avg_return_1d,
            'avg_return_7d': avg_return_7d,
            'avg_rvol': avg_rvol,
            'high_quality_pct': high_quality_pct,
            'total_patterns': len(pattern_counts),
            'top_patterns': dict(list(pattern_counts.items())[:5])
        }
        
        # Determine regime
        if avg_return_7d > 3 and avg_rvol > 1.5 and high_quality_pct > 30:
            regime = 'Bull Market'
            confidence = 85
        elif avg_return_7d < -2 and avg_rvol > 2:
            regime = 'Bear Market'
            confidence = 80
        elif avg_rvol < 1.2 and abs(avg_return_7d) < 1:
            regime = 'Sideways'
            confidence = 75
        elif avg_rvol > 2 and abs(avg_return_1d) > 2:
            regime = 'High Volatility'
            confidence = 78
        else:
            regime = 'Transitional'
            confidence = 60
            
        return {
            'regime': regime,
            'confidence': confidence,
            'indicators': regime_indicators
        }
    
    @staticmethod
    def get_sector_analysis(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sector performance"""
        
        if 'sector' not in df.columns or df.empty:
            return {'sectors': {}, 'top_sectors': [], 'sector_rotation': 'Unknown'}
        
        sector_stats = {}
        
        for sector in df['sector'].unique():
            if pd.isna(sector):
                continue
                
            sector_df = df[df['sector'] == sector]
            
            sector_stats[sector] = {
                'count': len(sector_df),
                'avg_score': sector_df['master_score'].mean(),
                'avg_return_7d': sector_df['ret_7d'].mean(),
                'avg_rvol': sector_df['rvol'].mean(),
                'high_quality_count': len(sector_df[sector_df['master_score'] >= 80])
            }
        
        # Sort sectors by performance
        top_sectors = sorted(sector_stats.items(), 
                           key=lambda x: x[1]['avg_score'], reverse=True)[:5]
        
        return {
            'sectors': sector_stats,
            'top_sectors': [sector[0] for sector in top_sectors],
            'sector_rotation': 'Active' if len(top_sectors) > 0 else 'Inactive'
        }
    
    @staticmethod
    def get_market_breadth(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        
        if df.empty:
            return {}
        
        total_stocks = len(df)
        
        breadth_indicators = {
            'advance_decline_ratio': len(df[df['ret_1d'] > 0]) / total_stocks * 100,
            'high_volume_stocks': len(df[df['rvol'] >= 2]) / total_stocks * 100,
            'momentum_stocks': len(df[df['ret_7d'] > 5]) / total_stocks * 100,
            'quality_leaders': len(df[df['master_score'] >= 90]) / total_stocks * 100,
            'pattern_coverage': len(df[df['patterns'] != '']) / total_stocks * 100
        }
        
        return breadth_indicators
    
    @staticmethod
    def get_risk_assessment(df: pd.DataFrame) -> Dict[str, Any]:
        """Assess market risk levels"""
        
        if df.empty:
            return {'risk_level': 'Unknown', 'risk_factors': []}
        
        risk_factors = []
        risk_score = 0
        
        # Check various risk indicators
        avg_rvol = df['rvol'].mean()
        if avg_rvol > 3:
            risk_factors.append('High volume activity')
            risk_score += 20
            
        volatility = df['ret_1d'].std()
        if volatility > 5:
            risk_factors.append('High volatility')
            risk_score += 25
            
        negative_momentum = len(df[df['ret_7d'] < -5]) / len(df) * 100
        if negative_momentum > 30:
            risk_factors.append('Widespread negative momentum')
            risk_score += 30
            
        concentration = df['master_score'].std()
        if concentration < 10:
            risk_factors.append('Low score dispersion')
            risk_score += 15
            
        # Determine risk level
        if risk_score > 60:
            risk_level = 'High Risk'
        elif risk_score > 35:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'Low Risk'
            
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors
        }

# ============================================
# RANKING ENGINE
# ============================================

class RankingEngine:
    """Core ranking algorithm - Master Score 3.0"""
    
    @staticmethod
    @performance_tracked("ranking_calculation")
    def calculate_master_score(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Master Score using weighted components"""
        
        # Position Score (30%) - How far from 52-week low
        if 'from_low_pct' in df.columns:
            df['position_score'] = pd.to_numeric(df['from_low_pct'], errors='coerce').fillna(0)
        else:
            df['position_score'] = 50  # Default neutral
            
        # Volume Score (25%) - Relative volume
        if 'rvol' in df.columns:
            rvol_clean = pd.to_numeric(df['rvol'], errors='coerce').fillna(1.0)
            df['volume_score'] = np.minimum(100, rvol_clean * 20)  # Cap at 100
        else:
            df['volume_score'] = 50
            
        # Momentum Score (15%) - Recent returns
        if 'ret_7d' in df.columns:
            ret_clean = pd.to_numeric(df['ret_7d'], errors='coerce').fillna(0)
            df['momentum_score'] = np.minimum(100, np.maximum(0, ret_clean * 5 + 50))
        else:
            df['momentum_score'] = 50
            
        # Acceleration Score (10%) - Price momentum acceleration
        if 'ret_1d' in df.columns and 'ret_7d' in df.columns:
            ret_1d = pd.to_numeric(df['ret_1d'], errors='coerce').fillna(0)
            ret_7d = pd.to_numeric(df['ret_7d'], errors='coerce').fillna(0)
            acceleration = ret_1d - (ret_7d / 7)  # Daily vs average
            df['acceleration_score'] = np.minimum(100, np.maximum(0, acceleration * 10 + 50))
        else:
            df['acceleration_score'] = 50
            
        # Breakout Score (10%) - Distance from resistance levels
        if 'from_high_pct' in df.columns:
            from_high = pd.to_numeric(df['from_high_pct'], errors='coerce').fillna(50)
            df['breakout_score'] = 100 - from_high  # Closer to high = higher score
        else:
            df['breakout_score'] = 50
            
        # RVOL Bonus (10%) - Volume confirmation
        if 'rvol' in df.columns:
            rvol_clean = pd.to_numeric(df['rvol'], errors='coerce').fillna(1.0)
            df['rvol_score'] = np.minimum(100, rvol_clean * 25)
        else:
            df['rvol_score'] = 50
            
        # Calculate weighted Master Score
        df['master_score'] = (
            df['position_score'] * CONFIG.POSITION_WEIGHT +
            df['volume_score'] * CONFIG.VOLUME_WEIGHT +
            df['momentum_score'] * CONFIG.MOMENTUM_WEIGHT +
            df['acceleration_score'] * CONFIG.ACCELERATION_WEIGHT +
            df['breakout_score'] * CONFIG.BREAKOUT_WEIGHT +
            df['rvol_score'] * CONFIG.RVOL_WEIGHT
        ).round(1)
        
        # Add rank
        df['rank'] = df['master_score'].rank(method='dense', ascending=False).astype(int)
        
        logger.info(f"✅ Master Score calculated for {len(df)} stocks")
        return df

# ============================================
# PATTERN DETECTION - ENHANCED WITH 36 PATTERNS
# ============================================

class PatternDetector:
    """Advanced pattern detection engine with 36 sophisticated patterns"""
    
    # Pattern metadata for intelligent confidence scoring (from OLD VERSION)
    PATTERN_METADATA = {
        '🔥 CAT LEADER': {'importance_weight': 10, 'category': 'momentum'},
        '💎 HIDDEN GEM': {'importance_weight': 10, 'category': 'value'},
        '🚀 ACCELERATING': {'importance_weight': 10, 'category': 'momentum'},
        '🏦 INSTITUTIONAL': {'importance_weight': 10, 'category': 'volume'},
        '⚡ VOL EXPLOSION': {'importance_weight': 15, 'category': 'volume'},
        '🎯 BREAKOUT': {'importance_weight': 10, 'category': 'technical'},
        '👑 MARKET LEADER': {'importance_weight': 10, 'category': 'leadership'},
        '🌊 MOMENTUM WAVE': {'importance_weight': 10, 'category': 'momentum'},
        '💰 LIQUID LEADER': {'importance_weight': 10, 'category': 'liquidity'},
        '💪 LONG STRENGTH': {'importance_weight': 5, 'category': 'trend'},
        '📈 QUALITY TREND': {'importance_weight': 10, 'category': 'trend'},
        '💎 VALUE MOMENTUM': {'importance_weight': 10, 'category': 'fundamental'},
        '📊 EARNINGS ROCKET': {'importance_weight': 10, 'category': 'fundamental'},
        '🏆 QUALITY LEADER': {'importance_weight': 10, 'category': 'fundamental'},
        '⚡ TURNAROUND': {'importance_weight': 10, 'category': 'fundamental'},
        '⚠️ HIGH PE': {'importance_weight': -5, 'category': 'warning'},
        '🎯 52W HIGH APPROACH': {'importance_weight': 10, 'category': 'range'},
        '🔄 52W LOW BOUNCE': {'importance_weight': 10, 'category': 'range'},
        '👑 GOLDEN ZONE': {'importance_weight': 5, 'category': 'range'},
        '📊 VOL ACCUMULATION': {'importance_weight': 5, 'category': 'volume'},
        '🔀 MOMENTUM DIVERGE': {'importance_weight': 10, 'category': 'divergence'},
        '🎯 RANGE COMPRESS': {'importance_weight': 5, 'category': 'range'},
        '🤫 STEALTH': {'importance_weight': 10, 'category': 'hidden'},
        '🧛 VAMPIRE': {'importance_weight': 10, 'category': 'aggressive'},
        '⛈️ PERFECT STORM': {'importance_weight': 20, 'category': 'extreme'},
        '🪤 BULL TRAP': {'importance_weight': 15, 'category': 'reversal'},
        '💣 CAPITULATION': {'importance_weight': 20, 'category': 'reversal'},
        '🏃 RUNAWAY GAP': {'importance_weight': 12, 'category': 'continuation'},
        '🔄 ROTATION LEADER': {'importance_weight': 10, 'category': 'rotation'},
        '⚠️ DISTRIBUTION': {'importance_weight': 15, 'category': 'warning'},
        '🎯 VELOCITY SQUEEZE': {'importance_weight': 15, 'category': 'coiled'},
        '⚠️ VOLUME DIVERGENCE': {'importance_weight': -10, 'category': 'warning'},
        '⚡ GOLDEN CROSS': {'importance_weight': 12, 'category': 'bullish'},
        '📉 EXHAUSTION': {'importance_weight': -15, 'category': 'bearish'},
        '🔺 PYRAMID': {'importance_weight': 8, 'category': 'accumulation'},
        '🌪️ VACUUM': {'importance_weight': 18, 'category': 'reversal'}
    }
    
    @staticmethod
    @performance_tracked("pattern_detection")
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect 36 sophisticated trading patterns with confidence scoring"""
        
        patterns_list = []
        confidence_list = []
        
        for idx, row in df.iterrows():
            stock_patterns = []
            pattern_confidences = []
            
            # Get cleaned values
            rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
            ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce')
            ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce')
            ret_30d = pd.to_numeric(row.get('ret_30d', 0), errors='coerce')
            from_low = pd.to_numeric(row.get('from_low_pct', 50), errors='coerce')
            from_high = pd.to_numeric(row.get('from_high_pct', 50), errors='coerce')
            master_score = pd.to_numeric(row.get('master_score', 50), errors='coerce')
            pe = pd.to_numeric(row.get('pe', 20), errors='coerce')
            volume = pd.to_numeric(row.get('volume_1d', 0), errors='coerce')
            price = pd.to_numeric(row.get('price', 0), errors='coerce')
            
            # Advanced Pattern Detection (36 patterns)
            
            # 1. Category Leadership Patterns
            if master_score >= 95 and from_low >= 80:
                stock_patterns.append('🔥 CAT LEADER')
                pattern_confidences.append(95)
                
            # 2. Hidden Value Patterns
            if master_score >= 80 and rvol >= 1.5 and from_low >= 20 and from_low <= 60:
                stock_patterns.append('💎 HIDDEN GEM')
                pattern_confidences.append(85)
                
            # 3. Acceleration Patterns
            if ret_1d > ret_7d / 7 * 2 and rvol >= 2:
                stock_patterns.append('🚀 ACCELERATING')
                pattern_confidences.append(90)
                
            # 4. Institutional Activity
            if rvol >= 2.5 and volume > 1000000 and master_score >= 70:
                stock_patterns.append('🏦 INSTITUTIONAL')
                pattern_confidences.append(88)
                
            # 5. Volume Explosion
            if rvol >= 5.0:
                stock_patterns.append('⚡ VOL EXPLOSION')
                pattern_confidences.append(95)
                
            # 6. Breakout Patterns
            if from_high <= 5 and rvol >= 2 and ret_1d > 2:
                stock_patterns.append('🎯 BREAKOUT')
                pattern_confidences.append(92)
                
            # 7. Market Leadership
            if master_score >= 90 and rvol >= 1.5:
                stock_patterns.append('👑 MARKET LEADER')
                pattern_confidences.append(90)
                
            # 8. Momentum Wave
            if ret_7d >= 5 and ret_1d >= 1 and rvol >= 1.5:
                stock_patterns.append('🌊 MOMENTUM WAVE')
                pattern_confidences.append(85)
                
            # 9. Liquid Leader
            if volume > 5000000 and master_score >= 80:
                stock_patterns.append('💰 LIQUID LEADER')
                pattern_confidences.append(82)
                
            # 10. Long-term Strength
            if ret_30d >= 15 and master_score >= 75:
                stock_patterns.append('💪 LONG STRENGTH')
                pattern_confidences.append(78)
                
            # 11. Quality Trend
            if ret_7d > 0 and ret_30d > 0 and master_score >= 80:
                stock_patterns.append('📈 QUALITY TREND')
                pattern_confidences.append(80)
                
            # 12. Value Momentum
            if pe > 0 and pe < 15 and ret_7d > 3:
                stock_patterns.append('💎 VALUE MOMENTUM')
                pattern_confidences.append(83)
                
            # 13. Earnings Rocket
            if ret_1d > 5 and rvol > 3:
                stock_patterns.append('📊 EARNINGS ROCKET')
                pattern_confidences.append(88)
                
            # 14. Quality Leader
            if master_score >= 90:
                stock_patterns.append('🏆 QUALITY LEADER')
                pattern_confidences.append(master_score)
                
            # 15. Turnaround
            if from_low >= 50 and ret_7d > 10:
                stock_patterns.append('⚡ TURNAROUND')
                pattern_confidences.append(85)
                
            # 16. High PE Warning
            if pe > 50:
                stock_patterns.append('⚠️ HIGH PE')
                pattern_confidences.append(75)
                
            # 17. 52W High Approach
            if from_high <= 10 and ret_1d > 0:
                stock_patterns.append('🎯 52W HIGH APPROACH')
                pattern_confidences.append(87)
                
            # 18. 52W Low Bounce
            if from_low >= 90 and ret_7d > 5:
                stock_patterns.append('🔄 52W LOW BOUNCE')
                pattern_confidences.append(89)
                
            # 19. Golden Zone
            if from_low >= 40 and from_low <= 70 and master_score >= 75:
                stock_patterns.append('👑 GOLDEN ZONE')
                pattern_confidences.append(80)
                
            # 20. Volume Accumulation
            if rvol >= 1.5 and rvol <= 3 and ret_1d > 0:
                stock_patterns.append('📊 VOL ACCUMULATION')
                pattern_confidences.append(75)
                
            # 21. Momentum Divergence
            if ret_1d < 0 and ret_7d > 5:
                stock_patterns.append('🔀 MOMENTUM DIVERGE')
                pattern_confidences.append(70)
                
            # 22. Range Compression
            if from_high > 20 and from_low < 80 and rvol < 1.2:
                stock_patterns.append('🎯 RANGE COMPRESS')
                pattern_confidences.append(72)
                
            # 23. Stealth Movement
            if ret_7d > 8 and rvol < 1.5:
                stock_patterns.append('🤫 STEALTH')
                pattern_confidences.append(85)
                
            # 24. Vampire (After Hours Activity)
            if rvol > 2 and ret_1d > 3 and volume > 2000000:
                stock_patterns.append('🧛 VAMPIRE')
                pattern_confidences.append(88)
                
            # 25. Perfect Storm
            if rvol > 5 and ret_1d > 8 and master_score > 90:
                stock_patterns.append('⛈️ PERFECT STORM')
                pattern_confidences.append(98)
                
            # 26. Bull Trap
            if from_high <= 5 and ret_1d < -2 and rvol > 2:
                stock_patterns.append('🪤 BULL TRAP')
                pattern_confidences.append(85)
                
            # 27. Capitulation
            if ret_1d < -8 and rvol > 4:
                stock_patterns.append('💣 CAPITULATION')
                pattern_confidences.append(92)
                
            # 28. Runaway Gap
            if ret_1d > 8 and rvol > 3 and from_high <= 10:
                stock_patterns.append('🏃 RUNAWAY GAP')
                pattern_confidences.append(90)
                
            # 29. Rotation Leader
            if master_score > 85 and rvol > 2 and ret_7d > 5:
                stock_patterns.append('🔄 ROTATION LEADER')
                pattern_confidences.append(87)
                
            # 30. Distribution Warning
            if from_high <= 5 and rvol > 3 and ret_1d < 0:
                stock_patterns.append('⚠️ DISTRIBUTION')
                pattern_confidences.append(80)
                
            # 31. Velocity Squeeze
            if rvol < 0.8 and from_low > 60 and master_score > 80:
                stock_patterns.append('🎯 VELOCITY SQUEEZE')
                pattern_confidences.append(88)
                
            # 32. Volume Divergence Warning
            if ret_1d > 3 and rvol < 1.0:
                stock_patterns.append('⚠️ VOLUME DIVERGENCE')
                pattern_confidences.append(75)
                
            # 33. Golden Cross
            if ret_7d > ret_30d and ret_7d > 5:
                stock_patterns.append('⚡ GOLDEN CROSS')
                pattern_confidences.append(83)
                
            # 34. Exhaustion
            if rvol > 5 and ret_1d < -5 and from_high <= 10:
                stock_patterns.append('📉 EXHAUSTION')
                pattern_confidences.append(85)
                
            # 35. Pyramid Accumulation
            if rvol >= 1.2 and rvol <= 2 and ret_7d > 3 and ret_30d > 10:
                stock_patterns.append('🔺 PYRAMID')
                pattern_confidences.append(80)
                
            # 36. Vacuum Effect
            if rvol > 8 and ret_1d > 10:
                stock_patterns.append('🌪️ VACUUM')
                pattern_confidences.append(95)
            
            # Calculate overall confidence
            if pattern_confidences:
                overall_confidence = sum(pattern_confidences) / len(pattern_confidences)
            else:
                overall_confidence = 0
                
            patterns_list.append(" | ".join(stock_patterns) if stock_patterns else "")
            confidence_list.append(round(overall_confidence, 1))
        
        df['patterns'] = patterns_list
        df['pattern_confidence'] = confidence_list
        
        pattern_count = sum(1 for p in patterns_list if p)
        logger.info(f"🎯 Detected patterns in {pattern_count} stocks with 36-pattern engine")
        
        return df
    
    @staticmethod
    def get_pattern_summary(df: pd.DataFrame) -> Dict[str, int]:
        """Get pattern distribution summary"""
        all_patterns = []
        for patterns in df['patterns'].dropna():
            if patterns and patterns != "":
                all_patterns.extend([p.strip() for p in patterns.split('|')])
        
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        return dict(sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True))

# ============================================
# MAIN DATA PROCESSING PIPELINE
# ============================================

@st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
def load_and_process_data(source_type: str, **kwargs) -> Tuple[pd.DataFrame, datetime, Dict[str, Any]]:
    """Main data processing pipeline with caching"""
    
    start_time = time.time()
    metadata = {'warnings': [], 'errors': []}
    
    try:
        # Step 1: Load raw data
        if source_type == "sheet":
            sheet_id = kwargs.get('sheet_id')
            gid = kwargs.get('gid')
            df = DataLoader.load_from_sheets(sheet_id, gid)
        elif source_type == "upload":
            uploaded_file = kwargs.get('file_data')
            df = DataLoader.load_from_upload(uploaded_file)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Step 2: Validate data
        is_valid, errors, warnings = DataValidator.validate_dataframe(df)
        metadata['warnings'].extend(warnings)
        metadata['errors'].extend(errors)
        
        if not is_valid:
            raise ValueError(f"Data validation failed: {errors}")
        
        # Step 3: Auto-detect and map column names
        df = DataLoader._normalize_column_names(df)
        
        # Step 4: Clean numeric columns
        numeric_columns = ['price', 'ret_1d', 'ret_7d', 'ret_30d', 'rvol', 'volume_1d', 
                          'from_low_pct', 'from_high_pct', 'pe', 'eps_change_pct']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].apply(DataValidator.clean_numeric_value)
        
        # Step 5: Calculate Master Score
        df = RankingEngine.calculate_master_score(df)
        
        # Step 6: Calculate Advanced Metrics
        df = AdvancedMetrics.calculate_advanced_metrics(df)
        
        # Step 7: Detect patterns (enhanced with 36 patterns)
        df = PatternDetector.detect_patterns(df)
        
        # Step 8: Sort by rank
        df = df.sort_values('rank').reset_index(drop=True)
        
        # Step 7: Add processing metadata
        processing_time = time.time() - start_time
        timestamp = datetime.now(timezone.utc)
        
        logger.info(f"✅ Data processing complete: {len(df)} stocks in {processing_time:.2f}s")
        
        return df, timestamp, metadata
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise Exception(f"Processing failed: {str(e)}")

# ============================================
# FILTERING ENGINE
# ============================================

class FilterEngine:
    """Smart filtering system"""
    
    @staticmethod
    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        """Apply all active filters"""
        
        if df.empty:
            return df
            
        filtered_df = df.copy()
        
        # Quick filters
        quick_filter = st.session_state.get('quick_filter')
        if quick_filter == 'top_gainers':
            filtered_df = filtered_df[filtered_df['momentum_score'] >= 80]
        elif quick_filter == 'volume_surges':
            filtered_df = filtered_df[pd.to_numeric(filtered_df['rvol'], errors='coerce') >= 3]
        elif quick_filter == 'breakout_ready':
            filtered_df = filtered_df[pd.to_numeric(filtered_df['from_low_pct'], errors='coerce') >= 70]
        elif quick_filter == 'hidden_gems':
            filtered_df = filtered_df[
                (filtered_df['master_score'] >= 80) & 
                (pd.to_numeric(filtered_df['rvol'], errors='coerce') >= 1.5)
            ]
        
        # Category filter
        category_filter = st.session_state.get('category_filter', [])
        if category_filter and 'category' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
        
        # Sector filter
        sector_filter = st.session_state.get('sector_filter', [])
        if sector_filter and 'sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sector'].isin(sector_filter)]
        
        # Score filter
        min_score = st.session_state.get('min_score', 0)
        if min_score > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= min_score]
        
        # RVOL filter
        min_rvol = st.session_state.get('min_rvol', 1.0)
        if min_rvol > 1.0 and 'rvol' in filtered_df.columns:
            rvol_clean = pd.to_numeric(filtered_df['rvol'], errors='coerce')
            filtered_df = filtered_df[rvol_clean >= min_rvol]
        
        # Pattern filter
        pattern_filter = st.session_state.get('patterns', [])
        if pattern_filter and 'patterns' in filtered_df.columns:
            mask = filtered_df['patterns'].str.contains('|'.join(pattern_filter), na=False, case=False)
            filtered_df = filtered_df[mask]
        
        return filtered_df

# ============================================
# VISUALIZATION ENGINE
# ============================================

class Visualizer:
    """Create advanced visualizations from OLD VERSION"""
    
    @staticmethod
    def create_pattern_distribution_chart(df: pd.DataFrame):
        """Create pattern distribution chart"""
        
        pattern_counts = PatternDetector.get_pattern_summary(df)
        
        if not pattern_counts:
            st.info("No patterns detected")
            return
            
        # Create Plotly chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(pattern_counts.keys())[:10],
                y=list(pattern_counts.values())[:10],
                marker_color='rgba(55, 128, 191, 0.7)',
                marker_line_color='rgba(55, 128, 191, 1.0)',
                marker_line_width=2
            )
        ])
        
        fig.update_layout(
            title="Top 10 Pattern Distribution",
            xaxis_title="Patterns",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_score_distribution(df: pd.DataFrame):
        """Create score distribution histogram"""
        
        if df.empty:
            st.info("No data to display")
            return
            
        fig = go.Figure(data=[
            go.Histogram(
                x=df['master_score'],
                nbinsx=20,
                marker_color='rgba(76, 175, 80, 0.7)',
                marker_line_color='rgba(76, 175, 80, 1.0)',
                marker_line_width=1
            )
        ])
        
        fig.update_layout(
            title="Master Score Distribution",
            xaxis_title="Master Score",
            yaxis_title="Count",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_sector_performance_chart(df: pd.DataFrame):
        """Create sector performance chart"""
        
        if 'sector' not in df.columns or df.empty:
            st.info("Sector data not available")
            return
            
        sector_analysis = MarketIntelligence.get_sector_analysis(df)
        
        if not sector_analysis['sectors']:
            st.info("No sector data to display")
            return
            
        sectors = list(sector_analysis['sectors'].keys())[:10]
        scores = [sector_analysis['sectors'][s]['avg_score'] for s in sectors]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sectors,
                y=scores,
                marker_color='rgba(255, 152, 0, 0.7)',
                marker_line_color='rgba(255, 152, 0, 1.0)',
                marker_line_width=2
            )
        ])
        
        fig.update_layout(
            title="Top 10 Sector Performance",
            xaxis_title="Sectors",
            yaxis_title="Average Score",
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_market_breadth_gauge(df: pd.DataFrame):
        """Create market breadth gauge"""
        
        breadth = MarketIntelligence.get_market_breadth(df)
        
        if not breadth:
            st.info("Market breadth data not available")
            return
            
        # Create gauge chart for advance/decline ratio
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = breadth.get('advance_decline_ratio', 50),
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Advance/Decline Ratio"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgray"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "lightgreen"},
                    {'range': [75, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def create_volume_vs_returns_scatter(df: pd.DataFrame):
        """Create volume vs returns scatter plot"""
        
        if df.empty:
            st.info("No data to display")
            return
            
        fig = go.Figure(data=go.Scatter(
            x=df['rvol'],
            y=df['ret_7d'],
            mode='markers',
            marker=dict(
                size=df['master_score']/5,
                color=df['master_score'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Master Score")
            ),
            text=df['symbol'],
            hovertemplate='<b>%{text}</b><br>' +
                         'RVOL: %{x}<br>' +
                         '7D Return: %{y}%<br>' +
                         'Score: %{marker.color}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Volume vs Returns Analysis",
            xaxis_title="Relative Volume (RVOL)",
            yaxis_title="7-Day Return (%)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# SEARCH ENGINE
# ============================================

class SearchEngine:
    """Intelligent search functionality from OLD VERSION"""
    
    @staticmethod
    @performance_tracked("search")
    def search_stocks(df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Search stocks by symbol, company name, or patterns"""
        
        if df.empty or not query.strip():
            return df
            
        query = query.strip().lower()
        
        # Search in multiple fields
        mask = (
            df['symbol'].str.lower().str.contains(query, na=False) |
            df.get('company_name', pd.Series()).str.lower().str.contains(query, na=False) |
            df['patterns'].str.lower().str.contains(query, na=False) |
            df.get('sector', pd.Series()).str.lower().str.contains(query, na=False) |
            df.get('category', pd.Series()).str.lower().str.contains(query, na=False)
        )
        
        return df[mask]
    
    @staticmethod
    def search_by_criteria(df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Advanced search by multiple criteria"""
        
        if df.empty:
            return df
            
        filtered_df = df.copy()
        
        # Score range
        if 'min_score' in criteria and criteria['min_score'] > 0:
            filtered_df = filtered_df[filtered_df['master_score'] >= criteria['min_score']]
            
        if 'max_score' in criteria and criteria['max_score'] < 100:
            filtered_df = filtered_df[filtered_df['master_score'] <= criteria['max_score']]
        
        # Volume criteria
        if 'min_rvol' in criteria and criteria['min_rvol'] > 1:
            filtered_df = filtered_df[
                pd.to_numeric(filtered_df['rvol'], errors='coerce') >= criteria['min_rvol']
            ]
        
        # Return criteria
        if 'min_return_7d' in criteria:
            filtered_df = filtered_df[
                pd.to_numeric(filtered_df['ret_7d'], errors='coerce') >= criteria['min_return_7d']
            ]
        
        # Pattern criteria
        if 'required_patterns' in criteria and criteria['required_patterns']:
            for pattern in criteria['required_patterns']:
                filtered_df = filtered_df[
                    filtered_df['patterns'].str.contains(pattern, na=False, case=False)
                ]
        
        return filtered_df

# ============================================
# CLEAN UI COMPONENTS
# ============================================

class UIComponents:
    """Clean UI following OLD VERSION philosophy"""
    
    @staticmethod
    def render_header():
        """Render clean header"""
        st.markdown("""
            <div style='text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                       color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h1>🌊 Wave Detection Ultimate 3.0</h1>
                <p style='font-size: 18px; margin: 0;'>Production-Grade Stock Discovery System</p>
            </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar():
        """Render clean sidebar with single filter section"""
        
        with st.sidebar:
            st.markdown("### 🔍 Smart Filters")
            
            # Quick Action Buttons
            st.markdown("#### ⚡ Quick Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🚀 Top Gainers", use_container_width=True):
                    st.session_state['quick_filter'] = 'top_gainers'
                    st.rerun()
                if st.button("💎 Hidden Gems", use_container_width=True):
                    st.session_state['quick_filter'] = 'hidden_gems'
                    st.rerun()
            
            with col2:
                if st.button("📈 Volume Surge", use_container_width=True):
                    st.session_state['quick_filter'] = 'volume_surges'
                    st.rerun()
                if st.button("⭐ Breakout Ready", use_container_width=True):
                    st.session_state['quick_filter'] = 'breakout_ready'
                    st.rerun()
            
            if st.button("🔄 Clear Filters", use_container_width=True):
                # Clear all filters
                for key in ['quick_filter', 'category_filter', 'sector_filter', 'min_score', 'min_rvol', 'patterns']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
            
            st.divider()
            
            # Advanced Filters
            st.markdown("#### 🎛️ Advanced Filters")
            
            # Score filter
            min_score = st.slider("Minimum Score", 0, 100, 
                                st.session_state.get('min_score', 0))
            st.session_state['min_score'] = min_score
            
            # RVOL filter
            min_rvol = st.slider("Minimum RVOL", 1.0, 10.0, 
                               st.session_state.get('min_rvol', 1.0), 0.1)
            st.session_state['min_rvol'] = min_rvol
    
    @staticmethod
    def render_data_status(df: pd.DataFrame, timestamp: datetime):
        """Render data status bar"""
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Stocks", len(df))
        
        with col2:
            avg_score = df['master_score'].mean() if not df.empty else 0
            st.metric("⭐ Avg Score", f"{avg_score:.1f}")
        
        with col3:
            last_update = timestamp.strftime("%H:%M:%S")
            st.metric("🕒 Last Update", last_update)
        
        with col4:
            if not df.empty:
                high_quality = len(df[df['master_score'] >= 80])
                st.metric("🎯 High Quality", high_quality)
    
    @staticmethod
    def render_stock_table(df: pd.DataFrame):
        """Render clean stock table"""
        
        if df.empty:
            st.warning("No stocks match current filters")
            return
        
        # Display columns configuration with fallbacks
        display_columns = {
            'rank': '#',
            'symbol': 'Symbol', 
            'company_name': 'Company',
            'price': 'Price',
            'ret_1d': '1D%',
            'ret_7d': '7D%', 
            'rvol': 'RVOL',
            'master_score': 'Score',
            'patterns': 'Patterns'
        }
        
        # Find available columns with flexible matching
        available_cols = []
        for col_key, display_name in display_columns.items():
            if col_key in df.columns:
                available_cols.append(col_key)
            else:
                # Try to find similar column
                similar_cols = [col for col in df.columns if col_key.lower() in col.lower()]
                if similar_cols:
                    available_cols.append(similar_cols[0])
                    display_columns[similar_cols[0]] = display_name
        
        # If no symbol column found, use first column
        if not any('symbol' in col.lower() for col in available_cols):
            if len(df.columns) > 0:
                first_col = df.columns[0]
                available_cols.insert(1, first_col)
                display_columns[first_col] = 'Symbol'
        
        # Create display dataframe
        display_df = df[available_cols].copy()
        
        # Rename columns
        display_df = display_df.rename(columns=display_columns)
        
        # Format numeric columns
        for col in display_df.columns:
            if 'Price' in col and col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) and x != 0 else "-")
            elif '%' in col and col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "-")
            elif 'RVOL' in col and col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "-")
            elif 'Score' in col and col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
        
        # Display table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    
    @staticmethod
    def render_export_options(df: pd.DataFrame):
        """Render export options"""
        
        if df.empty:
            return
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                "📁 Download CSV",
                csv_data,
                f"wave_detection_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True,
                key="export_csv_main"
            )
        
        with col2:
            top_10 = df.head(10)
            symbols = top_10['symbol'].tolist()
            symbol_list = ", ".join(symbols)
            
            st.download_button(
                "🎯 Top 10 Symbols",
                symbol_list,
                f"top_10_symbols_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                "text/plain",
                use_container_width=True,
                key="export_symbols_main"
            )

# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application - Clean and Simple like OLD VERSION"""
    
    # Page configuration
    st.set_page_config(
        page_title="Wave Detection Ultimate 3.0",
        page_icon="🌊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = True
        st.session_state['quick_filter'] = None
        st.session_state['min_score'] = 0
        st.session_state['min_rvol'] = 1.0
    
    # Render header
    UIComponents.render_header()
    
    # Render sidebar
    UIComponents.render_sidebar()
    
    # Data loading section
    st.markdown("### 📊 Data Source")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        source_type = st.radio(
            "Choose data source:",
            ["Google Sheets", "Upload CSV"],
            horizontal=True
        )
    
    with col2:
        auto_refresh = st.checkbox("Auto refresh (5 min)", value=False)
    
    df = None
    timestamp = None
    
    # Handle data loading
    if source_type == "Google Sheets":
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            sheet_id = st.text_input(
                "Google Sheets ID:",
                value=st.session_state.get('last_sheet_id', '')
            )
        
        with col2:
            gid = st.text_input(
                "GID (optional):",
                value=st.session_state.get('last_gid', '0')
            )
        
        with col3:
            load_button = st.button("🔄 Load Data", type="primary")
        
        if load_button and sheet_id:
            try:
                with st.spinner("Loading data from Google Sheets..."):
                    df, timestamp, metadata = load_and_process_data(
                        "sheet", 
                        sheet_id=sheet_id, 
                        gid=gid
                    )
                    
                    # Store for next time
                    st.session_state['last_sheet_id'] = sheet_id
                    st.session_state['last_gid'] = gid
                    st.session_state['data'] = df
                    st.session_state['timestamp'] = timestamp
                    
                    # Show warnings if any
                    if metadata['warnings']:
                        st.warning(f"Warnings: {'; '.join(metadata['warnings'])}")
                    
                    st.success(f"✅ Loaded {len(df)} stocks successfully!")
                    
            except Exception as e:
                st.error(f"❌ Loading failed: {str(e)}")
    
    elif source_type == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="Upload your stock data CSV file"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processing uploaded file..."):
                    df, timestamp, metadata = load_and_process_data(
                        "upload",
                        file_data=uploaded_file
                    )
                    
                    st.session_state['data'] = df
                    st.session_state['timestamp'] = timestamp
                    
                    # Show warnings if any
                    if metadata['warnings']:
                        st.warning(f"Warnings: {'; '.join(metadata['warnings'])}")
                    
                    st.success(f"✅ Processed {len(df)} stocks successfully!")
                    
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
    
    # Use cached data if available
    if df is None and 'data' in st.session_state:
        df = st.session_state['data']
        timestamp = st.session_state['timestamp']
    
    # Display results
    if df is not None and not df.empty:
        
        # Apply filters
        filtered_df = FilterEngine.apply_filters(df)
        
        # Data status
        UIComponents.render_data_status(filtered_df, timestamp)
        
        st.divider()
        
        # Main content tabs (Enhanced like OLD VERSION with 7 tabs)
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Rankings", 
            "🎯 Patterns", 
            "📈 Analytics", 
            "🔍 Search",
            "🌊 Wave Radar",
            "📁 Export",
            "⚙️ Settings"
        ])
        
        with tab1:
            st.markdown("### 🏆 Stock Rankings")
            UIComponents.render_stock_table(filtered_df)
        
        with tab2:
            st.markdown("### 🎯 Pattern Analysis")
            
            if not filtered_df.empty and 'patterns' in filtered_df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Pattern Distribution")
                    Visualizer.create_pattern_distribution_chart(filtered_df)
                
                with col2:
                    st.markdown("#### Pattern Statistics")
                    pattern_counts = PatternDetector.get_pattern_summary(filtered_df)
                    
                    if pattern_counts:
                        for i, (pattern, count) in enumerate(list(pattern_counts.items())[:5]):
                            st.metric(f"#{i+1} {pattern}", count)
                    else:
                        st.info("No patterns detected in current selection")
                        
                # Pattern confidence analysis
                if 'pattern_confidence' in filtered_df.columns:
                    st.markdown("#### Pattern Confidence Analysis")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_confidence = filtered_df['pattern_confidence'].mean()
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    with col2:
                        high_confidence = len(filtered_df[filtered_df['pattern_confidence'] >= 80])
                        st.metric("High Confidence", high_confidence)
                    
                    with col3:
                        max_confidence = filtered_df['pattern_confidence'].max()
                        st.metric("Max Confidence", f"{max_confidence:.1f}%")
            else:
                st.info("Load data to see pattern analysis")
        
        with tab3:
            st.markdown("### 📈 Advanced Analytics")
            
            if not filtered_df.empty:
                # Market Intelligence Section
                st.markdown("#### 🧠 Market Intelligence")
                
                col1, col2, col3 = st.columns(3)
                
                # Market Regime
                regime_data = MarketIntelligence.analyze_market_regime(filtered_df)
                with col1:
                    st.metric(
                        "Market Regime", 
                        regime_data['regime'],
                        f"{regime_data['confidence']}% confidence"
                    )
                
                # Risk Assessment
                risk_data = MarketIntelligence.get_risk_assessment(filtered_df)
                with col2:
                    st.metric(
                        "Risk Level",
                        risk_data['risk_level'],
                        f"Score: {risk_data['risk_score']}"
                    )
                
                # Sector Analysis
                sector_data = MarketIntelligence.get_sector_analysis(filtered_df)
                with col3:
                    st.metric(
                        "Sector Rotation",
                        sector_data['sector_rotation'],
                        f"Top: {sector_data['top_sectors'][0] if sector_data['top_sectors'] else 'N/A'}"
                    )
                
                st.divider()
                
                # Advanced Charts
                st.markdown("#### 📊 Advanced Visualizations")
                
                chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
                    "Score Distribution", "Sector Performance", "Market Breadth", "Volume Analysis"
                ])
                
                with chart_tab1:
                    Visualizer.create_score_distribution(filtered_df)
                
                with chart_tab2:
                    Visualizer.create_sector_performance_chart(filtered_df)
                
                with chart_tab3:
                    Visualizer.create_market_breadth_gauge(filtered_df)
                    
                    # Market breadth details
                    breadth = MarketIntelligence.get_market_breadth(filtered_df)
                    if breadth:
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Advancing Stocks", f"{breadth.get('advance_decline_ratio', 0):.1f}%")
                        with col2:
                            st.metric("High Volume", f"{breadth.get('high_volume_stocks', 0):.1f}%")
                        with col3:
                            st.metric("Momentum Stocks", f"{breadth.get('momentum_stocks', 0):.1f}%")
                
                with chart_tab4:
                    Visualizer.create_volume_vs_returns_scatter(filtered_df)
                
                st.divider()
                
                # Risk Factors
                if risk_data['risk_factors']:
                    st.markdown("#### ⚠️ Risk Factors")
                    for factor in risk_data['risk_factors']:
                        st.warning(f"• {factor}")
                
                # Advanced Metrics Summary
                if 'momentum_strength' in filtered_df.columns:
                    st.markdown("#### 🎯 Advanced Metrics Summary")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        avg_momentum = filtered_df['momentum_strength'].mean()
                        st.metric("Avg Momentum", f"{avg_momentum:.1f}")
                    
                    with metrics_col2:
                        avg_volume_profile = filtered_df['volume_profile'].mean()
                        st.metric("Avg Volume Profile", f"{avg_volume_profile:.1f}")
                    
                    with metrics_col3:
                        avg_liquidity = filtered_df['liquidity_score'].mean()
                        st.metric("Avg Liquidity", f"{avg_liquidity:.1f}")
                    
                    with metrics_col4:
                        avg_volatility = filtered_df['volatility_score'].mean()
                        st.metric("Avg Volatility", f"{avg_volatility:.1f}")
            else:
                st.info("Load data to see advanced analytics")
        
        with tab4:
            st.markdown("### 🔍 Intelligent Search")
            
            if not filtered_df.empty:
                search_col1, search_col2 = st.columns([2, 1])
                
                with search_col1:
                    search_query = st.text_input(
                        "Search stocks by symbol, company, pattern, or sector:",
                        placeholder="e.g., RELIANCE, HIDDEN GEM, IT, etc."
                    )
                
                with search_col2:
                    search_button = st.button("🔍 Search", type="primary")
                
                # Advanced search criteria
                with st.expander("🎛️ Advanced Search Criteria"):
                    adv_col1, adv_col2, adv_col3 = st.columns(3)
                    
                    with adv_col1:
                        min_search_score = st.slider("Min Score", 0, 100, 0, key="search_min_score")
                        max_search_score = st.slider("Max Score", 0, 100, 100, key="search_max_score")
                    
                    with adv_col2:
                        min_search_rvol = st.slider("Min RVOL", 1.0, 10.0, 1.0, key="search_min_rvol")
                        min_return_7d = st.slider("Min 7D Return", -20.0, 50.0, -20.0, key="search_min_return")
                    
                    with adv_col3:
                        available_patterns = ['VOL EXPLOSION', 'MOMENTUM WAVE', 'HIDDEN GEM', 'BREAKOUT READY']
                        required_patterns = st.multiselect("Required Patterns", available_patterns, key="search_patterns")
                
                # Perform search
                search_df = filtered_df
                
                if search_query and search_button:
                    search_df = SearchEngine.search_stocks(filtered_df, search_query)
                    st.success(f"Found {len(search_df)} stocks matching '{search_query}'")
                
                # Apply advanced criteria
                if any([min_search_score > 0, max_search_score < 100, min_search_rvol > 1.0, 
                       min_return_7d > -20, required_patterns]):
                    
                    criteria = {
                        'min_score': min_search_score,
                        'max_score': max_search_score,
                        'min_rvol': min_search_rvol,
                        'min_return_7d': min_return_7d,
                        'required_patterns': required_patterns
                    }
                    
                    search_df = SearchEngine.search_by_criteria(search_df, criteria)
                    st.info(f"Filtered to {len(search_df)} stocks based on criteria")
                
                # Display search results
                if not search_df.empty:
                    UIComponents.render_stock_table(search_df)
                else:
                    st.warning("No stocks match your search criteria")
            else:
                st.info("Load data to use search functionality")
        
        with tab5:
            st.markdown("### 🌊 Wave Radar - Market Overview")
            
            if not filtered_df.empty:
                # Market Regime Overview
                regime_data = MarketIntelligence.analyze_market_regime(filtered_df)
                
                st.markdown("#### 🎯 Market Pulse")
                pulse_col1, pulse_col2, pulse_col3, pulse_col4 = st.columns(4)
                
                with pulse_col1:
                    st.metric(
                        "Market Regime",
                        regime_data['regime'],
                        f"{regime_data['confidence']}% confidence"
                    )
                
                with pulse_col2:
                    avg_return = regime_data['indicators'].get('avg_return_7d', 0)
                    st.metric(
                        "Market Return (7D)",
                        f"{avg_return:.1f}%",
                        delta=f"vs 1D: {regime_data['indicators'].get('avg_return_1d', 0):.1f}%"
                    )
                
                with pulse_col3:
                    avg_rvol = regime_data['indicators'].get('avg_rvol', 1)
                    st.metric(
                        "Market Volume",
                        f"{avg_rvol:.1f}x",
                        delta="Relative to avg"
                    )
                
                with pulse_col4:
                    quality_pct = regime_data['indicators'].get('high_quality_pct', 0)
                    st.metric(
                        "Quality Stocks",
                        f"{quality_pct:.1f}%",
                        delta="Score ≥ 80"
                    )
                
                st.divider()
                
                # Top Patterns in Market
                st.markdown("#### 🎯 Dominant Patterns")
                top_patterns = regime_data['indicators'].get('top_patterns', {})
                
                if top_patterns:
                    pattern_cols = st.columns(len(top_patterns))
                    for i, (pattern, count) in enumerate(top_patterns.items()):
                        with pattern_cols[i]:
                            st.metric(pattern, count)
                else:
                    st.info("No significant patterns detected")
                
                st.divider()
                
                # Sector Heat Map
                st.markdown("#### 🔥 Sector Heat Map")
                sector_data = MarketIntelligence.get_sector_analysis(filtered_df)
                
                if sector_data['sectors']:
                    sectors = list(sector_data['sectors'].keys())[:8]
                    sector_scores = [sector_data['sectors'][s]['avg_score'] for s in sectors]
                    
                    # Create color-coded sector display
                    sector_grid_cols = st.columns(4)
                    for i, (sector, score) in enumerate(zip(sectors, sector_scores)):
                        col_idx = i % 4
                        with sector_grid_cols[col_idx]:
                            # Color based on performance
                            if score >= 80:
                                color = "🟢"
                            elif score >= 60:
                                color = "🟡"
                            else:
                                color = "🔴"
                            
                            st.metric(
                                f"{color} {sector}",
                                f"{score:.1f}",
                                delta=f"{sector_data['sectors'][sector]['count']} stocks"
                            )
                else:
                    st.info("Sector data not available")
                
                st.divider()
                
                # Wave Detection Summary
                st.markdown("#### 🌊 Wave Detection Summary")
                
                wave_col1, wave_col2, wave_col3 = st.columns(3)
                
                with wave_col1:
                    st.markdown("**🚀 Momentum Waves**")
                    momentum_stocks = len(filtered_df[filtered_df['patterns'].str.contains('MOMENTUM', na=False)])
                    st.metric("Stocks in Wave", momentum_stocks)
                
                with wave_col2:
                    st.markdown("**💎 Hidden Opportunities**")
                    hidden_gems = len(filtered_df[filtered_df['patterns'].str.contains('HIDDEN', na=False)])
                    st.metric("Hidden Gems", hidden_gems)
                
                with wave_col3:
                    st.markdown("**⚡ Volume Explosions**")
                    vol_explosions = len(filtered_df[filtered_df['patterns'].str.contains('EXPLOSION', na=False)])
                    st.metric("Volume Spikes", vol_explosions)
                
            else:
                st.info("Load data to see Wave Radar")
        
        with tab6:
            st.markdown("### 📁 Export Options")
            UIComponents.render_export_options(filtered_df)
            
            if not filtered_df.empty:
                st.markdown("#### 📊 Advanced Export Formats")
                
                export_col1, export_col2 = st.columns(2)
                
                with export_col1:
                    if st.button("📊 Export Full Report"):
                        # Create comprehensive report with all advanced data
                        regime_data = MarketIntelligence.analyze_market_regime(filtered_df)
                        sector_data = MarketIntelligence.get_sector_analysis(filtered_df)
                        risk_data = MarketIntelligence.get_risk_assessment(filtered_df)
                        
                        report_data = {
                            'summary': {
                                'total_stocks': len(filtered_df),
                                'avg_score': filtered_df['master_score'].mean(),
                                'high_quality_count': len(filtered_df[filtered_df['master_score'] >= 80])
                            },
                            'market_intelligence': {
                                'regime': regime_data,
                                'sectors': sector_data,
                                'risk_assessment': risk_data
                            },
                            'top_10': filtered_df.head(10).to_dict('records'),
                            'timestamp': timestamp.isoformat(),
                            'patterns': PatternDetector.get_pattern_summary(filtered_df)
                        }
                        
                        st.download_button(
                            "Download Advanced Report (JSON)",
                            json.dumps(report_data, indent=2),
                            f"wave_detection_advanced_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                            "application/json",
                            key="export_json_radar"
                        )
                
                with export_col2:
                    if st.button("🎯 Export Smart Watchlist"):
                        # Smart watchlist with different categories
                        watchlist_data = {
                            'momentum_leaders': filtered_df[
                                filtered_df['patterns'].str.contains('MOMENTUM', na=False)
                            ]['symbol'].head(10).tolist(),
                            'hidden_gems': filtered_df[
                                filtered_df['patterns'].str.contains('HIDDEN', na=False)
                            ]['symbol'].head(10).tolist(),
                            'volume_spikes': filtered_df[
                                filtered_df['patterns'].str.contains('EXPLOSION', na=False)
                            ]['symbol'].head(10).tolist(),
                            'quality_leaders': filtered_df[
                                filtered_df['master_score'] >= 90
                            ]['symbol'].head(10).tolist()
                        }
                        
                        watchlist_text = ""
                        for category, symbols in watchlist_data.items():
                            if symbols:
                                watchlist_text += f"\n# {category.upper()}\n"
                                watchlist_text += "\n".join(symbols) + "\n"
                        
                        st.download_button(
                            "Download Smart Watchlist",
                            watchlist_text,
                            f"smart_watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                            "text/plain",
                            key="export_watchlist_radar"
                        )
        
        with tab7:
            st.markdown("### ⚙️ System Settings & Information")
            
            settings_tab1, settings_tab2, settings_tab3 = st.tabs([
                "Display Settings", "Performance Settings", "About System"
            ])
            
            with settings_tab1:
                st.markdown("#### 🎨 Display Configuration")
                
                display_col1, display_col2 = st.columns(2)
                
                with display_col1:
                    rows_per_page = st.selectbox("Rows per page", [10, 20, 50, 100], index=1)
                    show_patterns = st.checkbox("Show pattern details", value=True)
                    show_confidence = st.checkbox("Show pattern confidence", value=True)
                    show_advanced_metrics = st.checkbox("Show advanced metrics", value=False)
                
                with display_col2:
                    chart_theme = st.selectbox("Chart theme", ["plotly", "plotly_white", "plotly_dark"])
                    table_height = st.slider("Table height", 300, 800, 400)
                    auto_refresh_interval = st.selectbox("Auto refresh", ["Off", "1 min", "5 min", "15 min"])
            
            with settings_tab2:
                st.markdown("#### ⚡ Performance Configuration")
                
                perf_col1, perf_col2 = st.columns(2)
                
                with perf_col1:
                    cache_enabled = st.checkbox("Enable caching", value=True)
                    parallel_processing = st.checkbox("Parallel processing", value=True)
                    debug_mode = st.checkbox("Debug mode", value=False)
                
                with perf_col2:
                    batch_size = st.slider("Processing batch size", 100, 2000, 1000)
                    timeout_seconds = st.slider("Request timeout", 10, 60, 30)
                
                # Performance monitoring
                st.markdown("#### 📊 Performance Metrics")
                if not filtered_df.empty:
                    perf_metrics_col1, perf_metrics_col2, perf_metrics_col3 = st.columns(3)
                    
                    with perf_metrics_col1:
                        processing_time = st.session_state.get('last_processing_time', 0)
                        st.metric("Last Processing Time", f"{processing_time:.2f}s")
                    
                    with perf_metrics_col2:
                        cache_hit_rate = st.session_state.get('cache_hit_rate', 0)
                        st.metric("Cache Hit Rate", f"{cache_hit_rate:.1f}%")
                    
                    with perf_metrics_col3:
                        memory_usage = st.session_state.get('memory_usage', 0)
                        st.metric("Memory Usage", f"{memory_usage:.1f} MB")
            
            with settings_tab3:
                st.markdown("#### 📋 System Information")
                
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.markdown("**Wave Detection Ultimate 3.0**")
                    st.markdown("- **Version**: 3.0.0-PROFESSIONAL")
                    st.markdown("- **Architecture**: Enhanced from proven OLD VERSION")
                    st.markdown("- **Pattern Engine**: 36 sophisticated patterns")
                    st.markdown("- **Market Intelligence**: Advanced regime detection")
                    st.markdown("- **Visualization**: Interactive Plotly charts")
                    st.markdown("- **Performance**: Production-grade optimization")
                
                with info_col2:
                    st.markdown("**🔧 Technical Features**")
                    st.markdown("- ✅ Master Score 3.0 with 6 weighted components")
                    st.markdown("- ✅ Advanced metrics calculator")
                    st.markdown("- ✅ Market intelligence engine")
                    st.markdown("- ✅ Pattern confidence scoring")
                    st.markdown("- ✅ Sector rotation analysis")
                    st.markdown("- ✅ Risk assessment framework")
                    st.markdown("- ✅ Intelligent search engine")
                    st.markdown("- ✅ Smart export capabilities")
                
                st.divider()
                
                # System Status
                st.markdown("#### 🟢 System Status")
                
                status_col1, status_col2, status_col3, status_col4 = st.columns(4)
                
                with status_col1:
                    st.metric("Data Source", "✅ Connected")
                
                with status_col2:
                    st.metric("Pattern Engine", "✅ Active")
                
                with status_col3:
                    st.metric("Market Intelligence", "✅ Enabled")
                
                with status_col4:
                    st.metric("Export System", "✅ Ready")
                
                # Configuration Summary
                st.markdown("#### ⚙️ Current Configuration")
                
                config_data = {
                    'Master Score Weights': {
                        'Position (from 52W low)': f"{CONFIG.POSITION_WEIGHT:.0%}",
                        'Volume (relative volume)': f"{CONFIG.VOLUME_WEIGHT:.0%}",
                        'Momentum (7D returns)': f"{CONFIG.MOMENTUM_WEIGHT:.0%}",
                        'Acceleration (price velocity)': f"{CONFIG.ACCELERATION_WEIGHT:.0%}",
                        'Breakout (from 52W high)': f"{CONFIG.BREAKOUT_WEIGHT:.0%}",
                        'RVOL Confirmation': f"{CONFIG.RVOL_WEIGHT:.0%}"
                    },
                    'Pattern Detection': f"36 patterns with confidence scoring",
                    'Cache TTL': f"{CONFIG.CACHE_TTL} seconds",
                    'Default Display': f"{CONFIG.DEFAULT_TOP_N} stocks"
                }
                
                st.json(config_data)
        
        with tab5:
            st.markdown("### 📁 Export Options")
            UIComponents.render_export_options(filtered_df)
            
            st.markdown("#### Export Formats")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export Full Report"):
                    # Create comprehensive report
                    report_data = {
                        'summary': {
                            'total_stocks': len(filtered_df),
                            'avg_score': filtered_df['master_score'].mean(),
                            'high_quality_count': len(filtered_df[filtered_df['master_score'] >= 80])
                        },
                        'top_10': filtered_df.head(10).to_dict('records'),
                        'timestamp': timestamp.isoformat()
                    }
                    
                    st.download_button(
                        "Download Report (JSON)",
                        json.dumps(report_data, indent=2),
                        f"wave_detection_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                        "application/json",
                        key="export_json_tab5"
                    )
            
            with col2:
                if st.button("🎯 Export Watchlist"):
                    top_symbols = filtered_df.head(20)['symbol'].tolist()
                    watchlist = "\n".join(top_symbols)
                    
                    st.download_button(
                        "Download Watchlist",
                        watchlist,
                        f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        "text/plain",
                        key="export_watchlist_tab5"
                    )
    
    else:
        st.info("👆 Load data using Google Sheets or CSV upload to start analysis")
    
    # Auto refresh
    if auto_refresh and df is not None:
        time.sleep(300)  # 5 minutes
        st.rerun()

if __name__ == "__main__":
    main()

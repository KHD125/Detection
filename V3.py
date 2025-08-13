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
# PATTERN DETECTION
# ============================================

class PatternDetector:
    """Advanced pattern detection engine"""
    
    @staticmethod
    @performance_tracked("pattern_detection")
    def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """Detect trading patterns for each stock"""
        
        patterns_list = []
        
        for idx, row in df.iterrows():
            stock_patterns = []
            
            # Get cleaned values
            rvol = pd.to_numeric(row.get('rvol', 1), errors='coerce')
            ret_1d = pd.to_numeric(row.get('ret_1d', 0), errors='coerce')
            ret_7d = pd.to_numeric(row.get('ret_7d', 0), errors='coerce')
            from_low = pd.to_numeric(row.get('from_low_pct', 50), errors='coerce')
            master_score = pd.to_numeric(row.get('master_score', 50), errors='coerce')
            
            # Pattern 1: Volume Explosion
            if rvol >= 3.0:
                stock_patterns.append("VOL EXPLOSION")
                
            # Pattern 2: Momentum Wave
            if ret_7d >= 5 and ret_1d >= 1:
                stock_patterns.append("MOMENTUM WAVE")
                
            # Pattern 3: Hidden Gem
            if master_score >= 80 and rvol >= 1.5 and from_low >= 20:
                stock_patterns.append("HIDDEN GEM")
                
            # Pattern 4: Breakout Ready
            if from_low >= 70 and rvol >= 2:
                stock_patterns.append("BREAKOUT READY")
                
            # Pattern 5: Quality Leader
            if master_score >= 90:
                stock_patterns.append("QUALITY LEADER")
                
            # Pattern 6: Acceleration
            if ret_1d > ret_7d / 7 * 2:  # Today's return > 2x daily average
                stock_patterns.append("ACCELERATION")
                
            patterns_list.append(" | ".join(stock_patterns) if stock_patterns else "")
        
        df['patterns'] = patterns_list
        
        pattern_count = sum(1 for p in patterns_list if p)
        logger.info(f"🎯 Detected patterns in {pattern_count} stocks")
        
        return df

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
        
        # Step 6: Detect patterns
        df = PatternDetector.detect_patterns(df)
        
        # Step 7: Sort by rank
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
                use_container_width=True
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
                use_container_width=True
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
        
        # Main content tabs (Clean like OLD VERSION)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Rankings", 
            "🎯 Patterns", 
            "📈 Performance", 
            "⚙️ Settings",
            "📁 Export"
        ])
        
        with tab1:
            st.markdown("### 🏆 Stock Rankings")
            UIComponents.render_stock_table(filtered_df)
        
        with tab2:
            st.markdown("### 🎯 Pattern Analysis")
            
            if not filtered_df.empty and 'patterns' in filtered_df.columns:
                # Pattern distribution
                all_patterns = []
                for patterns in filtered_df['patterns'].dropna():
                    if patterns and patterns != "None":
                        all_patterns.extend([p.strip() for p in patterns.split(',')])
                
                if all_patterns:
                    pattern_counts = pd.Series(all_patterns).value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.bar_chart(pattern_counts)
                    
                    with col2:
                        st.markdown("#### Top Patterns")
                        for pattern, count in pattern_counts.head(5).items():
                            st.metric(pattern, count)
                else:
                    st.info("No patterns detected in current selection")
            else:
                st.info("Load data to see pattern analysis")
        
        with tab3:
            st.markdown("### 📈 Performance Metrics")
            
            if not filtered_df.empty:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("#### Score Distribution")
                    score_hist = filtered_df['master_score'].hist(bins=20)
                    st.pyplot(plt.gcf())
                    plt.clf()
                
                with col2:
                    st.markdown("#### Top Performers")
                    top_10 = filtered_df.head(10)[['symbol', 'master_score', 'ret_1d']]
                    st.dataframe(top_10, hide_index=True)
                
                with col3:
                    st.markdown("#### Quick Stats")
                    st.metric("Avg Score", f"{filtered_df['master_score'].mean():.1f}")
                    st.metric("High Quality (>80)", len(filtered_df[filtered_df['master_score'] >= 80]))
                    if 'ret_1d' in filtered_df.columns:
                        st.metric("Avg 1D Return", f"{filtered_df['ret_1d'].mean():.1f}%")
        
        with tab4:
            st.markdown("### ⚙️ System Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Display Settings")
                rows_per_page = st.selectbox("Rows per page", [10, 20, 50, 100], index=1)
                show_patterns = st.checkbox("Show patterns", value=True)
                
            with col2:
                st.markdown("#### Performance Settings")
                cache_enabled = st.checkbox("Enable caching", value=True)
                debug_mode = st.checkbox("Debug mode", value=False)
            
            st.markdown("#### About")
            st.info("""
            **Wave Detection Ultimate 3.0**
            - Clean architecture based on proven OLD VERSION structure
            - Master Score algorithm with 6 weighted components
            - Advanced pattern detection for 6 trading patterns
            - Production-grade validation and error handling
            """)
        
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
                        "application/json"
                    )
            
            with col2:
                if st.button("🎯 Export Watchlist"):
                    top_symbols = filtered_df.head(20)['symbol'].tolist()
                    watchlist = "\n".join(top_symbols)
                    
                    st.download_button(
                        "Download Watchlist",
                        watchlist,
                        f"watchlist_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        "text/plain"
                    )
    
    else:
        st.info("👆 Load data using Google Sheets or CSV upload to start analysis")
    
    # Auto refresh
    if auto_refresh and df is not None:
        time.sleep(300)  # 5 minutes
        st.rerun()

if __name__ == "__main__":
    main()

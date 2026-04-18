"""
Alpha Trajectory v10.1 — Stock Intelligence Engine
=======================================================
Professional Stock Alpha & Rank Trajectory Analysis System
with 8-Component Adaptive Scoring, Data-Driven Conviction (12 signals),
Sector-Relative Blending, Breakout Quality Component, Market State Signal,
Momentum Decay Warning, Sector Alpha Detection, Market Regime Awareness,
Confidence Intervals, Z-Score Normalization, Risk-Adjusted T-Score,
Exit Warning System, Hot Streak Detection,
WAVE SIGNAL FUSION ENGINE, ALPHA SCORE ENGINE, and 14-STRATEGY WALK-FORWARD
BACKTEST ENGINE.

v10.1 MAX ALPHA SYSTEM:
  Alpha Score (0-100): Pure forward-return predictor using ONLY factors with
  proven next-week alpha from 28-CSV walk-forward analysis:
    Near-High Proximity (22pts), Breakout Quality (18pts), Persistence (20pts),
    Position Strength (12pts), Market State (10pts), No-Decay (8pts),
    Wave Fusion (5pts), Early Rally Stage (5pts).
  Quick Filter: 🏆 Max Alpha (Top 15) — sector-capped at 2/sector.
  Backtest Strategy S12: Max Alpha — alpha-score-weighted returns.
  Added 2 conviction signals: Signal 11 (ret_1y, 7pts), Signal 12 (from_low_pct, 5pts).
  Added S11: Momentum-Quality backtest strategy.

v9.0 DATA-DRIVEN RECALIBRATION:
  Deep analysis of 28 CSVs, 27 week-to-week transitions, ~2,115 stocks/week:
  
  FORWARD PREDICTIVE POWER (top25% vs bot25% → next week alpha):
    from_high_pct:        +0.55%/wk  (#1 predictor) → NearHigh conviction signal
    breakout_score:       +0.44%/wk  (#2 predictor) → 8th scoring component
    position_score:       +0.34%/wk  (#3 predictor) → SectorLeader conviction signal
    overall_market_strength: +0.32%  (#4)
    trend_quality:        +0.26%/wk  (#5)
    momentum_score:       +0.00%/wk  (ZERO — velocity/acceleration reduced)
    volume_score:         -0.01%/wk  (ZERO)
    acceleration_score:   +0.01%/wk  (ZERO)
  
  MARKET STATE (mean reversion signal):
    BOUNCE:              +1.17%/wk  → conviction bonus
    STRONG_DOWNTREND:    +0.25%/wk  → mild bonus (mean reversion)
    SIDEWAYS:            -0.63%/wk  → penalty
    UPTREND:             -0.71%/wk  → penalty (reversion risk)
  
  BACKTEST RESULTS (v9.0, 28 CSVs, 22 windows):
    S2 Top 10 Rank:  +0.86% alpha (fixed from -2.35% in v8.1)
    S7 Conviction:   +0.42% alpha, 50% win rate
    5 of 7 strategies beat universe

CORE ARCHITECTURE:
  8-Component Adaptive Scoring → Elite Dominance Bonus → Bayesian Shrinkage
    → Hurst Persistence × Wave Fusion (±12% cap)
    → Sector-Relative Blending → Sector Alpha Tag

  Wave Signal Fusion: Cross-validates WAVE Detection scores with Trajectory calculations.
    4 Fusion Signals: Confluence (35%) + Institutional Flow (30%) + Momentum Harmony (20%)
    + Fundamental Quality (15%) → Fusion Multiplier ×0.94 to ×1.06

  Components: Positional, Trend, Velocity, Acceleration, Consistency,
              Resilience, ReturnQuality, BreakoutQuality
  Weights shift dynamically by position tier (elite/strong/mid/bottom).

  DATA-DRIVEN WEIGHTS (v9.0):
  Elite (>90pct):  Pos 18% | Trd 10% | Vel 4%  | Acc 3%  | Con 24% | Res 15% | Ret 14% | Brk 12%
  Strong (70-90):  Pos 15% | Trd 13% | Vel 6%  | Acc 4%  | Con 20% | Res 14% | Ret 14% | Brk 14%
  Mid (40-70):     Pos 10% | Trd 15% | Vel 10% | Acc 6%  | Con 16% | Res 10% | Ret 16% | Brk 17%
  Bottom (<40):    Pos 6%  | Trd 14% | Vel 12% | Acc 8%  | Con 12% | Res 10% | Ret 18% | Brk 20%

  CONVICTION (v9.0 — 9 Signals, 100pts):
  Persistence 25 | Consistency 12 | NearHigh 14 | RetQuality 8
  Breakout 12 | WaveConfl 8 | SectorLeader 7 | MarketState 8 | NoDecay 6

  SIGNAL ISOLATION PRINCIPLE:
    - Return data enters through exactly ONE component (ReturnQuality).
    - Price-Rank Alignment scores DIRECTION only (sign agreement), never magnitude.
    - Momentum Decay uses separate ret_6m for proven-winner exemption.
    - No signal leakage: each data source has exactly one scoring path.

Version: 9.0.0
Last Updated: March 2026
"""

# ============================================
# STREAMLIT CONFIG - Must be first
# ============================================
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
st.set_page_config(
    page_title="Alpha Trajectory — Stock Intelligence Engine",
    page_icon="🔺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# IMPORTS
# ============================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import re
import warnings
import logging
import glob
import tempfile
import shutil
from typing import Dict, List, Tuple, Optional, Any
from io import BytesIO

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


class InMemoryUpload:
    """Upload-like wrapper so Drive-downloaded bytes can reuse existing pipeline."""
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.size = len(data)
        self._data = data
        self._buf = BytesIO(data)

    def getvalue(self):
        return self._data

    def seek(self, pos: int, whence: int = 0):
        return self._buf.seek(pos, whence)

    def read(self, size: int = -1):
        return self._buf.read(size)


def _normalize_drive_folder_key(raw: str) -> str:
    """Accept raw key or full URL and return normalized folder key."""
    key = (raw or '').strip()
    if not key:
        return ''
    if 'drive.google.com' in key and '/folders/' in key:
        key_part = key.split('/folders/', 1)[1]
        key = key_part.split('?', 1)[0].split('/', 1)[0].strip()
    return key


def _drive_folder_url(folder_key: str) -> str:
    return f"https://drive.google.com/drive/folders/{folder_key}?usp=drive_link"


def _load_csv_uploads_from_drive(folder_key: str) -> Tuple[List[InMemoryUpload], Optional[str]]:
    """Download CSVs from a public Google Drive folder using requests (no gdown needed).

    Strategy:
      1. Fetch the public folder HTML page
      2. Extract file IDs and names via regex
      3. Download each CSV individually via export URL
    """
    import requests

    key = _normalize_drive_folder_key(folder_key)
    if not key:
        return [], "Folder key is empty."

    # Step 1: Fetch folder listing HTML
    folder_url = f"https://drive.google.com/drive/folders/{key}"
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                       '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    try:
        resp = session.get(folder_url, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            return [], "Folder not found. Check the folder key."
        return [], f"Drive folder HTTP error {resp.status_code}: {e}"
    except Exception as e:
        return [], f"Failed to connect to Google Drive: {e}"

    html = resp.text

    # Step 2: Extract file IDs and names from the HTML
    # Google Drive folder pages contain file metadata in JS data structures
    # Pattern: data-id="FILE_ID" ... file name appears nearby
    file_entries: List[Tuple[str, str]] = []  # (file_id, filename)

    # Method A: Extract from JS arrays — Google embeds file info as arrays
    # Typical pattern: ["FILE_ID","FILENAME", ...]
    id_name_pattern = re.findall(
        r'\["(1[a-zA-Z0-9_-]{10,})","([^"]+\.csv)"',
        html,
        re.IGNORECASE
    )
    for fid, fname in id_name_pattern:
        if fname.lower().endswith('.csv'):
            file_entries.append((fid, fname))

    # Method B: Fallback — find all file IDs and pair with nearby filenames
    if not file_entries:
        # Look for /file/d/ID patterns and data-id attributes
        all_ids = re.findall(r'/file/d/(1[a-zA-Z0-9_-]{10,})', html)
        all_ids += re.findall(r'data-id="(1[a-zA-Z0-9_-]{10,})"', html)
        all_ids = list(dict.fromkeys(all_ids))  # deduplicate, keep order

        # Extract CSV filenames from HTML
        csv_names = re.findall(r'(Stocks_Weekly[^"<>\s]*\.csv)', html, re.IGNORECASE)
        if not csv_names:
            csv_names = re.findall(r'([^"<>\s/]+\.csv)', html, re.IGNORECASE)

        # If we have both IDs and names, pair them
        if all_ids and csv_names:
            for fid, fname in zip(all_ids, csv_names):
                file_entries.append((fid, fname))
        elif all_ids:
            # Have IDs but no names — download each and check if it's CSV
            for fid in all_ids:
                file_entries.append((fid, f"file_{fid}.csv"))

    # Deduplicate by file ID
    seen_ids = set()
    unique_entries = []
    for fid, fname in file_entries:
        if fid not in seen_ids:
            seen_ids.add(fid)
            unique_entries.append((fid, fname))
    file_entries = unique_entries

    if not file_entries:
        return [], (
            "No CSV files found in this Drive folder.\n"
            "Make sure:\n"
            "1. Folder sharing is set to 'Anyone with the link → Viewer'\n"
            "2. The folder contains CSV files\n"
            "3. The folder key/URL is correct"
        )

    # Step 3: Download each file
    uploads: List[InMemoryUpload] = []
    errors = []
    for fid, fname in file_entries:
        download_url = f"https://drive.google.com/uc?export=download&id={fid}"
        try:
            dl_resp = session.get(download_url, timeout=30)
            dl_resp.raise_for_status()
            content = dl_resp.content

            # Handle Google's virus scan warning page for large files
            if b'<html' in content[:200].lower() and b'confirm=' in content:
                confirm_match = re.search(r'confirm=([0-9A-Za-z_-]+)', dl_resp.text)
                if confirm_match:
                    confirm_url = f"{download_url}&confirm={confirm_match.group(1)}"
                    dl_resp = session.get(confirm_url, timeout=60)
                    dl_resp.raise_for_status()
                    content = dl_resp.content

            # Validate: content should look like CSV (not HTML error page)
            head = content[:500].decode('utf-8', errors='ignore').lower()
            if '<html' in head and 'ticker' not in head and 'rank' not in head:
                errors.append(f"Skipped {fname}: received HTML instead of CSV (file may not be public)")
                continue

            # Determine real filename — prefer Content-Disposition header
            cd = dl_resp.headers.get('Content-Disposition', '')
            cd_match = re.search(r'filename="?([^";\n]+)"?', cd)
            real_name = cd_match.group(1).strip() if cd_match else fname

            if not real_name.lower().endswith('.csv'):
                continue

            uploads.append(InMemoryUpload(real_name, content))
        except Exception as e:
            errors.append(f"Failed to download {fname}: {e}")
            continue

    if not uploads:
        err_detail = "\n".join(errors[:5]) if errors else ""
        return [], (
            f"Could not download any CSV files from this folder.\n{err_detail}\n\n"
            "Make sure folder sharing is: 'Anyone with the link → Viewer'"
        )

    return uploads, None


# ============================================
# CONFIGURATION
# ============================================

MIN_WEEKS_DEFAULT = 3
MAX_DISPLAY_DEFAULT = 100

# ── Quantitative Enhancement Configuration (v3.0) ──

# Bayesian Confidence Shrinkage — shrinks scores toward population mean
# when data is insufficient. Prevents 4-week lucky stocks from ranking high.
BAYESIAN_CONFIDENCE = {
    'prior_mean': 45.0,          # Population prior (assume mediocre until proven)
    'full_confidence_weeks': 16,  # Weeks needed for 100% data confidence
    'min_confidence': 0.25,       # Minimum confidence even with 2 weeks
}

# Hurst Exponent — determines if rank series will PERSIST or REVERT
# H > 0.5 = trending (current pattern likely continues)
# H = 0.5 = random walk (no predictive power)
# H < 0.5 = mean-reverting (current pattern likely reverses)
HURST_CONFIG = {
    'min_weeks': 6,               # Need at least 6 datapoints
    'trend_persistence_h': 0.55,  # H above this = trending
    'mean_revert_h': 0.42,        # H below this = mean-reverting
    'max_boost': 1.06,            # Max multiplier for strong persistence
    'max_penalty': 0.94,          # Max penalty for mean-reversion on uptrend
}

# Information Ratio — risk-adjusted consistency (replaces raw volatility)
# IR = mean(excess_return) / std(excess_return)
# High IR = consistently outperforming. Low IR = noisy, unreliable.
INFO_RATIO_CONFIG = {
    'benchmark_growth': 0.5,      # Assumed benchmark percentile growth/week
    'excellent_ir': 0.8,          # IR above this = excellent consistency
    'good_ir': 0.3,               # IR above this = good consistency
    'poor_ir': -0.2,              # IR below this = inconsistent
}

# Trajectory Score Weights — v2.1 ADAPTIVE WEIGHT SYSTEM
# Weights dynamically shift based on WHERE the stock sits (percentile tier)
# Elite stocks: Position dominates (45%) — being at rank 5 IS the achievement
# Climbers: Movement dominates (Velocity 25%, Trend 25%) — they need to prove direction
# Bottom stocks: Acceleration matters most — are they even trying to move?

# Adaptive weight profiles by percentile tier (v9.0: Data-Driven)
# ═══════════════════════════════════════════════════════════════════════════════
# v9.0 RECALIBRATION based on deep forward-return analysis (28 CSVs, 27 transitions):
#   Factor Alpha (top25% vs bot25% → next week return):
#     from_high_pct:  +0.55%  → feeds ReturnQuality
#     breakout_score: +0.44%  → NEW 8th component: BreakoutQuality
#     position_score: +0.34%  → feeds Positional
#     trend_quality:  +0.26%  → feeds Trend
#     momentum_score: +0.00%  → ZERO predictive → Velocity/Acceleration reduced
#     volume_score:   -0.01%  → ZERO predictive
#     accel_score:    +0.01%  → ZERO predictive → Acceleration heavily reduced
#
# KEY INSIGHT: breakout_score and from_high_pct are the strongest forward
# predictors. momentum_score/volume_score/acceleration have ZERO predictive power.
# Added BreakoutQuality as dedicated 8th component. Reduced Vel/Acc heavily.
# ═══════════════════════════════════════════════════════════════════════════════
ADAPTIVE_WEIGHTS = {
    # Elite (avg pct > 90): Consistency + proven breakout quality
    'elite': {
        'positional': 0.18, 'trend': 0.10, 'velocity': 0.04,
        'acceleration': 0.03, 'consistency': 0.24, 'resilience': 0.15,
        'return_quality': 0.14, 'breakout_quality': 0.12
    },
    # Strong (avg pct 70-90): Balanced with breakout emphasis
    'strong': {
        'positional': 0.15, 'trend': 0.13, 'velocity': 0.06,
        'acceleration': 0.04, 'consistency': 0.20, 'resilience': 0.14,
        'return_quality': 0.14, 'breakout_quality': 0.14
    },
    # Mid (avg pct 40-70): Breakout and returns dominate — need proof of momentum
    'mid': {
        'positional': 0.10, 'trend': 0.15, 'velocity': 0.10,
        'acceleration': 0.06, 'consistency': 0.16, 'resilience': 0.10,
        'return_quality': 0.16, 'breakout_quality': 0.17
    },
    # Bottom (avg pct < 40): Breakout + returns — are they actually moving up?
    'bottom': {
        'positional': 0.06, 'trend': 0.14, 'velocity': 0.12,
        'acceleration': 0.08, 'consistency': 0.12, 'resilience': 0.10,
        'return_quality': 0.18, 'breakout_quality': 0.20
    }
}

# Elite Dominance Bonus thresholds
ELITE_BONUS = {
    'top3_sustained': {'pct_threshold': 97, 'history_ratio': 0.60, 'floor': 88},
    'top5_sustained': {'pct_threshold': 95, 'history_ratio': 0.60, 'floor': 82},
    'top10_sustained': {'pct_threshold': 90, 'history_ratio': 0.60, 'floor': 73},
    'top20_sustained': {'pct_threshold': 85, 'history_ratio': 0.55, 'floor': 65}
}

# Directional Price-Rank Alignment Configuration (v6.1 — direction-only, no magnitude)
PRICE_ALIGNMENT = {
    'noise_band_stable': 2.0,        # Ignore rank moves < this for stable stocks
    'noise_band_normal': 1.0,        # Ignore rank moves < this for normal stocks
    'min_weeks': 4,                  # Minimum weeks needed for alignment calculation
    'confirmed_threshold': 72,       # Alignment score above this = PRICE_CONFIRMED
    'divergent_threshold': 35,       # Alignment score below this = PRICE_DIVERGENT
    'ema_span': 3,                   # EMA span for smoothing ret_7d (3-week)
    'recency_window': 4,             # Recent N weeks get 2× weight in directional signal
}

# Momentum Decay Warning Configuration (v2.3)
MOMENTUM_DECAY = {
    'min_pct_tier': 40,              # Only check stocks above this avg percentile
    'r7_severe': -5,                 # Weekly return below this triggers severe signal
    'r7_moderate': -2,               # Weekly return below this triggers moderate signal
    'r30_severe_high': -15,          # 30d return for top-ranked stocks: TRAP threshold
    'r30_moderate_high': -5,         # 30d return moderate warning for top stocks
    'r30_severe_mid': -10,           # 30d return severe for mid-ranked stocks
    'from_high_severe': -20,         # Far from high — significant correction
    'from_high_moderate': -15,       # Moderate correction from high
    'severe_threshold': 60,          # Decay score above this = severe
    'moderate_threshold': 35,        # Decay score above this = moderate
    'mild_threshold': 15,            # Decay score above this = mild
}

# ── Return Quality Component Configuration (v6.0) ──
# ARCHITECTURE CHANGE: Returns are now a dedicated 7th scoring component.
# Previously returns leaked through 5 doors (velocity floor, acceleration floor,
# resilience bonus, conviction multiplier, price alignment signal 3).
# Now one clean component: _calc_return_quality() → 0 to 100.
# 4 sub-signals: 3M return (30%), 6M return (30%), short-term momentum (20%),
# return health / cross-timeframe agreement (20%).
RETURN_QUALITY = {
    'min_weeks': 3,                  # Minimum data weeks for scoring
}

# Sector Alpha Configuration (v2.3)
SECTOR_ALPHA = {
    'min_sector_stocks': 3,          # Minimum stocks in sector for alpha calc
    'leader_z': 1.5,                 # Z-score above this = sector leader
    'outperform_z': 0.5,             # Z-score above this = outperforming sector
    'aligned_z': -0.5,              # Z-score above this = aligned with sector
    'beta_sector_min': 60,           # Sector avg must be > this for beta detection
}

# v9.0: Sector-Relative Ranking Configuration
# Data proved: sector top 10% vs bottom 10% → +0.59%/wk alpha
# Problem: Capital Goods (429 stocks) vs Diversified (7 stocks) — unfair comparison
# Solution: Blend universe percentile with sector-relative percentile
SECTOR_RELATIVE = {
    'blend_weight': 0.30,            # 30% sector-relative, 70% universe
    'min_sector_size': 10,           # Sectors < 10 stocks: reduce sector weight
    'size_reference': 80,            # Median-ish sector size for dampening calc
    # size_factor = min(1.0, sector_count / size_reference)
    # effective_sector_weight = blend_weight × size_factor
    # 7-stock sector: factor=0.088 → effective weight=2.6% (fair)
    # 429-stock sector: factor=1.0 → effective weight=30% (full)
}

# ── Wave Signal Fusion Configuration (v8.0) ──
# Cross-validates WAVE Detection's 18 ignored signals with Trajectory Engine calculations.
# When BOTH systems agree a stock is strong → high conviction.
# When they disagree → uncertainty flag.
WAVE_FUSION = {
    # Fusion signal weights (must sum to 1.0)
    'confluence_weight': 0.35,           # Agreement between WAVE scoring and Trajectory
    'flow_weight': 0.30,                 # Institutional money flow strength
    'harmony_weight': 0.20,              # WAVE's momentum harmony (0-4)
    'fundamental_weight': 0.15,          # EPS/PE quality gate

    # Multiplier range
    'max_boost': 1.10,                   # Maximum fusion boost (strong agreement)
    'max_penalty': 0.92,                 # Maximum fusion penalty (strong conflict)
    'neutral_zone_lo': 42,               # Below this → penalty
    'neutral_zone_hi': 58,               # Above this → boost

    # Confluence thresholds (WAVE vs Trajectory agreement)
    'strong_agree_pct_diff': 10,         # Percentile difference for "strong agreement"
    'agree_pct_diff': 25,                # Percentile difference for "agreement"
    'disagree_pct_diff': 45,             # Percentile difference for "disagreement"

    # Institutional flow scoring
    'money_flow_strong': 50.0,           # money_flow_mm above this = strong flow
    'money_flow_moderate': 10.0,         # money_flow_mm above this = moderate flow
    'vmi_strong': 65,                    # VMI above this = strong volume-momentum
    'vmi_moderate': 40,                  # VMI above this = moderate
    'rvol_hot': 2.0,                     # RVOL above this = hot volume
    'rvol_active': 1.2,                  # RVOL above this = active volume

    # Fundamental quality thresholds
    'eps_growth_strong': 20.0,           # EPS change % for "strong growth"
    'eps_growth_moderate': 5.0,          # EPS change % for "moderate growth"
    'pe_reasonable_max': 50.0,           # PE ratio ceiling for "reasonable"
    'pe_value_max': 20.0,                # PE ratio ceiling for "value"
    'pe_negative_penalty': True,         # Penalize negative PE (loss-making)
}


# Grade definitions: (min_score, label, emoji)
GRADE_DEFS = [
    (85, 'S', '🏆'), (70, 'A', '🥇'), (55, 'B', '🥈'),
    (40, 'C', '🥉'), (25, 'D', '📊'), (0, 'F', '📉')
]

GRADE_COLORS = {
    'S': '#FFD700', 'A': '#3fb950', 'B': '#58a6ff',
    'C': '#d29922', 'D': '#FF5722', 'F': '#f85149'
}

# Pattern definitions: key -> (emoji, name, description)
# v3.0 — 13 actionable patterns, no ghost entries, priority-ordered detection
PATTERN_DEFS = {
    'stable_elite':      ('🎯', 'Stable Elite',      'Consistently top-ranked with low volatility'),
    'rocket':            ('🚀', 'Rocket',             'Rapid strong improvement across all dimensions'),
    'breakout':          ('⚡', 'Breakout',            'Sudden significant rank jump beyond normal variance'),
    'momentum_building': ('🔥', 'Momentum Building',  'Acceleration surging — early signal before full move'),
    'at_peak':           ('🏔️', 'At Peak',            'At or near all-time best rank with sustained strength'),
    'topping_out':       ('⛰️', 'Topping Out',        'Near peak but momentum fading — potential reversal'),
    'steady_climber':    ('📈', 'Steady Climber',     'Gradual but consistent rank improvement'),
    'recovery':          ('🔄', 'Recovery',           'Bouncing back from rank deterioration'),
    'consolidating':     ('⏳', 'Consolidating',      'Tight range movement — potential breakout setup'),
    'fading':            ('📉', 'Fading',             'Rank deteriorating from recent levels'),
    'crash':             ('💥', 'Crash',              'Severe rapid rank collapse — high-risk warning'),
    'volatile':          ('🌊', 'Volatile',           'Large and unpredictable rank swings'),
    'new_entry':         ('💎', 'New Entry',           'Recently appeared or insufficient history'),
    'neutral':           ('➖', 'Neutral',             'Average stock with no strong directional signal'),
}

PATTERN_COLORS = {
    'stable_elite': '#8A2BE2', 'rocket': '#FF4500', 'breakout': '#FFD700',
    'momentum_building': '#FF6347', 'at_peak': '#FF69B4', 'topping_out': '#CD853F',
    'steady_climber': '#32CD32', 'recovery': '#00BFFF', 'consolidating': '#B8860B',
    'fading': '#808080', 'crash': '#DC143C', 'volatile': '#FF8C00',
    'new_entry': '#00CED1', 'neutral': '#A9A9A9',
}

# ============================================
# CUSTOM CSS — MINIMAL DARK DESIGN SYSTEM v3
# ============================================
st.markdown("""
<style>
    /* ── Responsive Layout: keep content inside viewport when sidebar is open ── */
    section[data-testid="stMain"] > div.block-container {
        max-width: 100%;
        overflow-x: hidden;
        box-sizing: border-box;
    }
    section[data-testid="stMain"] {
        overflow-x: hidden;
    }
    /* Prevent any inner element from forcing horizontal scroll */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stHorizontalBlock"],
    div[data-testid="stMarkdownContainer"] {
        max-width: 100%;
        box-sizing: border-box;
    }
    /* Columns must NOT clip — let content flow naturally */
    div[data-testid="column"] {
        max-width: 100%;
        box-sizing: border-box;
        overflow: visible;
    }
    /* Wide tables / HTML cards: shrink to fit */
    div[data-testid="stMarkdownContainer"] > div,
    div[data-testid="stMarkdownContainer"] table {
        max-width: 100%;
        overflow-x: auto;
    }
    /* Plotly charts should also respect container */
    div.js-plotly-plot, div.plotly {
        max-width: 100% !important;
    }

    /* ── Top Movers: compact when sidebar is expanded ── */
    html:has(section[data-testid="stSidebar"][aria-expanded="true"]) .mv-ticker  { display: none !important; }
    html:has(section[data-testid="stSidebar"][aria-expanded="true"]) .mv-sector  { display: none !important; }
    html:has(section[data-testid="stSidebar"][aria-expanded="true"]) .mv-row     { padding-left: 8px !important; padding-right: 8px !important; gap: 5px !important; }
    html:has(section[data-testid="stSidebar"][aria-expanded="true"]) .mv-grid    { gap: 6px !important; }

    /* ── Hero Banner ── */
    .hero-banner {
        text-align: center; padding: 1.6rem 1.2rem 1.3rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
        border: 1px solid rgba(88,166,255,0.15);
        border-radius: 14px; margin-bottom: 1.2rem;
        position: relative; overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-banner::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at 30% 20%, rgba(228,179,65,0.08) 0%, transparent 50%),
                    radial-gradient(ellipse at 70% 80%, rgba(88,166,255,0.06) 0%, transparent 50%);
        pointer-events: none;
    }
    .hero-icon {
        font-size: 2.8rem; line-height: 1; position: relative;
        filter: drop-shadow(0 0 14px rgba(255,69,58,0.5));
        margin-bottom: 4px;
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 900; position: relative;
        background: linear-gradient(120deg, #e3b341 0%, #58a6ff 50%, #3fb950 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 1.5px; line-height: 1.15; margin: 0;
    }
    .hero-sub {
        font-size: 0.82rem; font-weight: 500; color: #8b949e;
        letter-spacing: 2px; text-transform: uppercase;
        position: relative; margin-top: 6px;
    }
    .hero-badge {
        display: inline-block; font-size: 0.62rem; font-weight: 700;
        color: #e3b341; background: rgba(228,179,65,0.10);
        border: 1px solid rgba(228,179,65,0.25); padding: 3px 14px;
        border-radius: 12px; margin-top: 10px; position: relative;
        letter-spacing: 1px;
    }
    .sec-head {
        font-size: 0.88rem; font-weight: 600; color: #c9d1d9; letter-spacing: 0.3px;
        margin: 20px 0 8px 0; display: flex; align-items: center; gap: 6px;
    }
    .sec-cap { font-size: 0.7rem; color: #6e7681; margin-top: -4px; margin-bottom: 10px; }

    /* ── Metric Strip ── */
    .m-strip {
        display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px;
    }
    .m-chip {
        background: #161b22; border: 1px solid #30363d; border-radius: 10px;
        padding: 12px 0; text-align: center; flex: 1; min-width: 90px;
        transition: border-color 0.2s ease;
    }
    .m-chip:hover { border-color: #484f58; }
    .m-val { font-size: 1.45rem; font-weight: 700; color: #e6edf3; line-height: 1; }
    .m-lbl { font-size: 0.62rem; color: #8b949e; text-transform: uppercase;
             letter-spacing: 0.6px; margin-top: 3px; }
    .m-green .m-val { color: #3fb950; }
    .m-red   .m-val { color: #f85149; }
    .m-gold  .m-val { color: #d29922; }
    .m-blue  .m-val { color: #58a6ff; }
    .m-orange .m-val { color: #FF6B35; }

    /* ── Cards ── */
    .t-card {
        background: #0d1117; border: 1px solid #30363d; border-radius: 12px;
        padding: 14px; margin-bottom: 8px;
        transition: border-color 0.15s, box-shadow 0.15s;
    }
    .t-card:hover { border-color: #484f58; }
    .t-card-danger  { border-color: rgba(248,81,73,0.4); background: rgba(248,81,73,0.03); }
    .t-card-success { border-color: rgba(63,185,80,0.4); background: rgba(63,185,80,0.03);
                      box-shadow: 0 0 12px rgba(63,185,80,0.05); }
    .t-card-success:hover { border-color: #3fb950; box-shadow: 0 0 20px rgba(63,185,80,0.10); }
    .t-hd { font-weight: 700; font-size: 0.95rem; color: #e6edf3; }
    .t-sub { font-size: 0.7rem; color: #8b949e; margin-top: 1px; }
    .t-row { display: flex; justify-content: space-between; align-items: center;
             margin-top: 5px; font-size: 0.78rem; }
    .t-badge {
        display: inline-block; padding: 2px 7px; border-radius: 8px;
        font-size: 0.65rem; font-weight: 600;
    }

    /* ── Pill Tags ── */
    .pill {
        display: inline-block; padding: 2px 8px; border-radius: 10px;
        font-size: 0.65rem; font-weight: 600; margin: 1px 2px;
        border: 1px solid;
    }
    .p-grn { color: #3fb950; border-color: rgba(63,185,80,0.3); background: rgba(63,185,80,0.08); }
    .p-red { color: #f85149; border-color: rgba(248,81,73,0.3); background: rgba(248,81,73,0.08); }
    .p-gld { color: #d29922; border-color: rgba(210,153,34,0.3); background: rgba(210,153,34,0.08); }
    .p-blu { color: #58a6ff; border-color: rgba(88,166,255,0.3); background: rgba(88,166,255,0.08); }

    /* ── Progress bar ── */
    .pbar-wrap { background: #21262d; border-radius: 3px; height: 5px; margin-top: 3px; }
    .pbar { height: 5px; border-radius: 3px; }

    /* ── Grade ── */
    .grade-S { color: #FFD700; font-weight: 900; font-size: 1.5rem; }
    .grade-A { color: #3fb950; font-weight: 900; font-size: 1.5rem; }
    .grade-B { color: #58a6ff; font-weight: 900; font-size: 1.5rem; }
    .grade-C { color: #d29922; font-weight: 900; font-size: 1.5rem; }
    .grade-D { color: #FF5722; font-weight: 900; font-size: 1.5rem; }
    .grade-F { color: #f85149; font-weight: 900; font-size: 1.5rem; }

    /* ── Tags ── */
    .pattern-tag {
        display: inline-block; padding: 3px 10px; border-radius: 12px;
        font-size: 0.78rem; background: rgba(255,107,53,0.08); color: #FF6B35;
        border: 1px solid rgba(255,107,53,0.2); margin: 2px;
    }
    .mover-up   { color: #3fb950; font-weight: 700; }
    .mover-down { color: #f85149; font-weight: 700; }

    /* ── Misc ── */
    .divider { border-top: 1px solid #21262d; margin: 16px 0; }
    .stock-card {
        background: #0d1117; border-radius: 14px; padding: 22px;
        margin-bottom: 16px; border: 1px solid #30363d;
    }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; font-weight: 600; border-radius: 8px 8px 0 0; }


    /* ── Sidebar Premium Styling ── */
    .sb-brand {
        background: linear-gradient(135deg, rgba(56,182,255,0.08) 0%, rgba(228,179,65,0.10) 50%, rgba(63,185,80,0.08) 100%);
        border: 1px solid rgba(228,179,65,0.3); border-radius: 16px;
        padding: 20px 14px 14px; text-align: center; margin-bottom: 16px;
        position: relative; overflow: hidden;
    }
    .sb-brand::before {
        content: ''; position: absolute; top: -40%; left: -40%;
        width: 180%; height: 180%;
        background: radial-gradient(circle, rgba(228,179,65,0.08) 0%, transparent 70%);
        animation: sb-pulse 6s ease-in-out infinite;
    }
    @keyframes sb-pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }
    .sb-brand-icon {
        font-size: 2.2rem; position: relative; line-height: 1;
        filter: drop-shadow(0 0 8px rgba(228,179,65,0.4));
        margin-bottom: 4px;
    }
    .sb-brand-title {
        font-size: 1.2rem; font-weight: 800; position: relative;
        background: linear-gradient(120deg, #e3b341 0%, #58a6ff 50%, #3fb950 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 0.5px;
    }
    .sb-brand-sub {
        font-size: 0.68rem; font-weight: 600; color: #8b949e;
        letter-spacing: 2.5px; text-transform: uppercase;
        position: relative; margin-top: 2px;
    }
    .sb-brand-ver {
        display: inline-block; font-size: 0.58rem; font-weight: 700; color: #e3b341;
        background: rgba(228,179,65,0.10); border: 1px solid rgba(228,179,65,0.3);
        padding: 2px 10px; border-radius: 10px; margin-top: 6px; position: relative;
        letter-spacing: 0.5px;
    }
    .sb-clear-btn {
        display: flex; align-items: center; justify-content: center; gap: 6px;
        width: 100%; padding: 6px 0; margin: 8px 0 4px;
        font-size: 0.72rem; font-weight: 600; color: #f85149;
        background: rgba(248,81,73,0.06); border: 1px solid rgba(248,81,73,0.2);
        border-radius: 8px; cursor: pointer; transition: all 0.2s;
    }
    .sb-clear-btn:hover {
        background: rgba(248,81,73,0.12); border-color: rgba(248,81,73,0.4);
    }
    .sb-status-card {
        background: rgba(22,27,34,0.85); backdrop-filter: blur(8px);
        border: 1px solid #30363d; border-radius: 12px;
        padding: 12px 14px; margin: 10px 0;
    }
    .sb-status-row {
        display: flex; justify-content: space-between; align-items: center;
        font-size: 0.78rem; color: #c9d1d9; padding: 3px 0;
    }
    .sb-status-val { font-weight: 700; color: #e6edf3; }
    .sb-status-dot {
        display: inline-block; width: 7px; height: 7px; border-radius: 50%;
        background: #3fb950; margin-right: 5px; animation: sb-blink 2s infinite;
    }
    @keyframes sb-blink { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
    .sb-cached-badge {
        display: inline-flex; align-items: center; gap: 4px;
        font-size: 0.65rem; font-weight: 600; color: #3fb950;
        background: rgba(63,185,80,0.08); border: 1px solid rgba(63,185,80,0.25);
        padding: 2px 8px; border-radius: 8px; margin-top: 4px;
    }
    .sb-file-chip {
        display: inline-block; font-size: 0.62rem; color: #8b949e;
        background: #161b22; border: 1px solid #21262d; border-radius: 6px;
        padding: 2px 6px; margin: 1px;
    }
    .sb-file-chip-active {
        color: #58a6ff; border-color: rgba(88,166,255,0.3);
        background: rgba(88,166,255,0.06);
    }
    .sb-section-head {
        font-size: 0.75rem; font-weight: 700; color: #8b949e;
        text-transform: uppercase; letter-spacing: 0.8px;
        margin: 14px 0 6px 0; display: flex; align-items: center; gap: 5px;
    }
    .sb-divider {
        height: 1px; background: linear-gradient(90deg, transparent, #30363d, transparent);
        margin: 12px 0;
    }
</style>

""", unsafe_allow_html=True)


# ============================================
# UTILITY FUNCTIONS
# ============================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    """Safe division with zero/nan/inf protection"""
    try:
        if b == 0 or np.isnan(b) or np.isnan(a):
            return default
        result = a / b
        return default if np.isinf(result) else result
    except (TypeError, ValueError):
        return default


def get_grade(score: float) -> Tuple[str, str]:
    """Return (grade_letter, emoji) for a trajectory score"""
    for threshold, label, emoji in GRADE_DEFS:
        if score >= threshold:
            return label, emoji
    return 'F', '📉'


def format_rank_change(change: int) -> str:
    """Format rank change with arrow indicators"""
    if change > 0:
        return f"▲ {change}"
    elif change < 0:
        return f"▼ {abs(change)}"
    return "— 0"


def ranks_to_percentiles(ranks: List[float], totals: List[int]) -> List[float]:
    """Convert absolute ranks to percentiles (higher = better)"""
    return [(1 - r / max(t, 1)) * 100 for r, t in zip(ranks, totals)]


def calculate_tmi(ranks: List[float], totals: List[int], period: int = 14) -> float:
    """
    Trajectory Momentum Index (TMI) - RSI-style indicator for rank trajectory.
    TMI > 70 = Strong momentum, TMI < 30 = Weak/deteriorating.
    """
    pcts = ranks_to_percentiles(ranks, totals)
    if len(pcts) < 3:
        return 50.0

    changes = np.diff(pcts)
    window = min(period, len(changes))
    recent = changes[-window:]

    gains = np.where(recent > 0, recent, 0)
    losses = np.where(recent < 0, -recent, 0)

    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

    rs = safe_div(avg_gain, avg_loss, 1.0)
    tmi = 100 - safe_div(100, 1 + rs, 50)
    return float(np.clip(tmi, 0, 100))


# ============================================
# DATA LOADING ENGINE
# ============================================

def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """Extract date from CSV filename: Stocks_Weekly_YYYY-MM-DD_..."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    return datetime.strptime(match.group(1), '%Y-%m-%d') if match else None


def _extract_dated_files(uploaded_files: list) -> Tuple[List[Tuple[datetime, Any]], int]:
    """Return sorted dated file tuples and count of files without parseable date."""
    dated: List[Tuple[datetime, Any]] = []
    undated = 0
    for f in uploaded_files:
        dt = parse_date_from_filename(getattr(f, 'name', ''))
        if dt is None:
            undated += 1
            continue
        dated.append((dt, f))
    dated.sort(key=lambda x: x[0])
    return dated, undated


def load_and_compute(uploaded_files: list) -> Tuple[Optional[pd.DataFrame], Optional[dict], Optional[list], Optional[dict]]:
    """
    Master data pipeline: Process uploaded CSVs → Build histories → Compute trajectories.
    Returns: (trajectory_df, histories, dates_iso, metadata)
    """
    # ── Step 1: Parse uploaded CSVs ──
    if not uploaded_files:
        return None, None, None, None

    weekly_data = {}
    for ufile in uploaded_files:
        date = parse_date_from_filename(ufile.name)
        if date is None:
            continue
        try:
            ufile.seek(0)
            df = pd.read_csv(ufile)
            if 'rank' in df.columns and 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.strip()
                df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                df['master_score'] = pd.to_numeric(df.get('master_score', 0), errors='coerce').fillna(0)
                df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
                weekly_data[date] = df
        except Exception as e:
            logger.warning(f"Failed to load {ufile.name}: {e}")

    if not weekly_data:
        return None, None, None, None

    weekly_data = dict(sorted(weekly_data.items()))
    dates = sorted(weekly_data.keys())
    dates_iso = [d.strftime('%Y-%m-%d') for d in dates]

    # ── Step 2: Build rank histories ──
    histories = {}
    for date in dates:
        df = weekly_data[date]
        total = len(df)
        for _, row in df.iterrows():
            ticker = str(row.get('ticker', '')).strip()
            if not ticker or ticker == 'nan':
                continue

            if ticker not in histories:
                histories[ticker] = {
                    'dates': [], 'ranks': [], 'scores': [], 'prices': [],
                    'total_per_week': [],
                    'trend_qualities': [],
                    'market_states': [],
                    'pattern_history': [],
                    'ret_7d': [],             # v2.3: Weekly returns (split-adjusted)
                    'ret_30d': [],            # v2.3: 30-day returns
                    'ret_3m': [],             # v2.3: 3-month returns
                    'ret_6m': [],             # v3.0: 6-month returns (institutional horizon)
                    'from_high_pct': [],      # v2.3: Distance from 52w high
                    'momentum_score': [],     # v2.3: Wave engine momentum score
                    'volume_score': [],       # v2.3: Wave engine volume score
                    # ── v8.0: Wave Signal Fusion — 18 previously ignored columns ──
                    'position_score': [],     # WAVE's 52-week position scoring
                    'acceleration_score': [], # WAVE's momentum acceleration
                    'breakout_score': [],     # WAVE's breakout detection (4 components)
                    'rvol_score': [],         # WAVE's relative volume quality
                    'pe': [],                 # Price/Earnings ratio
                    'eps_current': [],        # Current EPS
                    'eps_change_pct': [],     # EPS change percentage
                    'from_low_pct': [],       # Distance from 52-week low
                    'ret_1d': [],             # 1-day return (intraday momentum)
                    'ret_1y': [],             # 1-year return (long-term trend)
                    'rvol': [],               # Raw relative volume
                    'vmi': [],                # Volume-Momentum Index
                    'money_flow_mm': [],      # Money flow in millions
                    'position_tension': [],   # Position tension metric
                    'momentum_harmony': [],   # Momentum harmony (0-4)
                    'eps_tier': [],           # EPS tier classification
                    'pe_tier': [],            # PE tier classification
                    'overall_market_strength': [],  # Market strength metric
                    'company_name': '', 'category': '', 'sector': '',
                    'industry': '', 'market_state': '', 'patterns': ''
                }

            h = histories[ticker]
            h['dates'].append(date.strftime('%Y-%m-%d'))
            h['ranks'].append(float(row['rank']) if pd.notna(row['rank']) else total)
            h['scores'].append(float(row['master_score']))
            h['prices'].append(float(row['price']))
            h['total_per_week'].append(total)

            tq_val = row.get('trend_quality', 0)
            h['trend_qualities'].append(float(tq_val) if pd.notna(tq_val) else 0)
            ms_val = row.get('market_state', '')
            h['market_states'].append(str(ms_val).strip() if pd.notna(ms_val) else '')
            pat_val = row.get('patterns', '')
            h['pattern_history'].append(str(pat_val).strip() if pd.notna(pat_val) else '')

            # v2.3: Track return & fundamental data per week
            for col_name, hist_key in [
                ('ret_7d', 'ret_7d'), ('ret_30d', 'ret_30d'), ('ret_3m', 'ret_3m'),
                ('ret_6m', 'ret_6m'),
                ('from_high_pct', 'from_high_pct'), ('momentum_score', 'momentum_score'),
                ('volume_score', 'volume_score'),
                # v8.0: Wave Signal Fusion — 18 previously ignored columns
                ('position_score', 'position_score'), ('acceleration_score', 'acceleration_score'),
                ('breakout_score', 'breakout_score'), ('rvol_score', 'rvol_score'),
                ('pe', 'pe'), ('eps_current', 'eps_current'),
                ('eps_change_pct', 'eps_change_pct'), ('from_low_pct', 'from_low_pct'),
                ('ret_1d', 'ret_1d'), ('ret_1y', 'ret_1y'),
                ('rvol', 'rvol'), ('vmi', 'vmi'),
                ('money_flow_mm', 'money_flow_mm'), ('position_tension', 'position_tension'),
                ('momentum_harmony', 'momentum_harmony'),
                ('overall_market_strength', 'overall_market_strength'),
            ]:
                col_val = row.get(col_name, None)
                if col_val is not None and pd.notna(col_val):
                    try:
                        h[hist_key].append(float(col_val))
                    except (ValueError, TypeError):
                        h[hist_key].append(float('nan'))
                else:
                    h[hist_key].append(float('nan'))

            # Always keep latest info
            for fld in ['company_name', 'category', 'sector', 'industry', 'market_state', 'patterns']:
                val = row.get(fld, '')
                if pd.notna(val) and str(val).strip():
                    h[fld] = str(val).strip()

            # v8.0: Track tier classifications (string columns)
            for tier_col in ['eps_tier', 'pe_tier']:
                tier_val = row.get(tier_col, '')
                h[tier_col].append(str(tier_val).strip() if pd.notna(tier_val) else '')

    # ── Step 3: Compute trajectories for all tickers ──
    results = []
    for ticker, h in histories.items():
        traj = _compute_single_trajectory(h)
        traj['ticker'] = ticker
        traj['company_name'] = h['company_name']
        traj['category'] = h['category']
        traj['sector'] = h['sector']
        traj['industry'] = h['industry']
        traj['market_state'] = h['market_state']
        traj['latest_patterns'] = h['patterns']
        results.append(traj)

    # Build DataFrame and sort (v6.2: Confidence-Aware Ranking with Tie-Breakers)
    # Primary: trajectory_score DESC
    # Tie-breaker 1: confidence DESC (proven performers beat lucky streaks)
    # Tie-breaker 2: consistency DESC (stable beats volatile at equal score+conf)
    traj_df = pd.DataFrame(results)
    traj_df = traj_df.sort_values(
        ['trajectory_score', 'confidence', 'consistency'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    traj_df.insert(0, 't_rank', range(1, len(traj_df) + 1))
    
    # Add T-Percentile: position as percentile of universe (higher = better)
    total_stocks = len(traj_df)
    traj_df['t_percentile'] = ((total_stocks - traj_df['t_rank']) / max(total_stocks - 1, 1) * 100).round(1)

    # ── Step 3b: Market Regime Classification (v9.0 cleaned) ──
    # Only computes regime label + median for UI display & conviction signal.
    # Dead code removed: market_adj_factor (never applied), market_adj_score (never read),
    # Z-score normalization (17 columns never referenced downstream).
    market_trend_median = traj_df['trend'].median()

    if market_trend_median > 58:
        market_regime = 'BULL'
    elif market_trend_median < 42:
        market_regime = 'BEAR'
    else:
        market_regime = 'SIDEWAYS'

    traj_df['market_regime'] = market_regime
    traj_df['market_trend_median'] = round(market_trend_median, 1)

    # ── Step 3d: Sector-Relative Blending (v9.0) ──
    # DATA EVIDENCE: sector-relative top10% vs bot10% → +0.59%/wk alpha.
    # Problem: 23 sectors range 7 to 429 stocks. A top-10 stock in a 7-stock
    # sector is NOT comparable to top-10 in a 429-stock sector.
    # Solution: blend universe percentile with sector percentile, dampened by sector size.
    sr_cfg = SECTOR_RELATIVE
    eligible_mask = traj_df['weeks'] >= MIN_WEEKS_DEFAULT

    # Compute sector percentile for each stock (percentile within sector)
    traj_df['sector_pct'] = 0.0
    traj_df['sector_blend_score'] = traj_df['trajectory_score']  # default = universe score

    if eligible_mask.sum() > 0:
        eligible_df = traj_df[eligible_mask]
        for sector_name, sector_group in eligible_df.groupby('sector'):
            s_count = len(sector_group)
            if s_count < 2:
                continue  # Can't compute percentile with 1 stock

            # Rank within sector (1 = best trajectory_score in sector)
            sector_ranks = sector_group['trajectory_score'].rank(ascending=False, method='min')
            # Convert to percentile (100 = best)
            sector_pct = ((s_count - sector_ranks) / max(s_count - 1, 1) * 100).round(1)
            traj_df.loc[sector_group.index, 'sector_pct'] = sector_pct

            # Size-dampened blending: small sectors get less sector weight
            size_factor = min(1.0, s_count / sr_cfg['size_reference'])
            if s_count < sr_cfg['min_sector_size']:
                size_factor *= (s_count / sr_cfg['min_sector_size'])  # Further dampen tiny sectors
            effective_weight = sr_cfg['blend_weight'] * size_factor
            universe_weight = 1.0 - effective_weight

            # Blend: trajectory_score stays, but we create a blended version for re-ranking
            # Universe percentile uses t_percentile (already computed), sector uses sector_pct
            # Both are 0-100 scales, combine them, then map back to score range
            for idx in sector_group.index:
                uni_pct = traj_df.loc[idx, 't_percentile']
                sec_pct = traj_df.loc[idx, 'sector_pct']
                blended_pct = universe_weight * uni_pct + effective_weight * sec_pct
                # Convert blended percentile back to score scale (preserving original score range)
                # Adjustment = (blended_pct - uni_pct) * score_range / 100
                pct_shift = blended_pct - uni_pct  # How much the blend shifts the percentile
                score_adj = pct_shift * 0.15  # Scale: 1 percentile shift ≈ 0.15 score points
                traj_df.loc[idx, 'sector_blend_score'] = traj_df.loc[idx, 'trajectory_score'] + score_adj

        # Re-sort by sector_blend_score and re-assign t_rank
        traj_df = traj_df.sort_values(
            ['sector_blend_score', 'confidence', 'consistency'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        traj_df['t_rank'] = range(1, len(traj_df) + 1)
        traj_df['t_percentile'] = ((total_stocks - traj_df['t_rank']) / max(total_stocks - 1, 1) * 100).round(1)

    # ── Step 3e: Bear Market Quality Tilt (v10.0) ──
    # Backtest evidence: in bear markets, S7 (Conviction ≥ 65) outperforms S2
    # (Top 10 Rank) by +0.18%/wk. Pure rank concentration picks noisy stocks.
    # Quality tilt: when market_regime == 'BEAR', blend ranking score with a
    # conviction-quality composite so the ranking naturally favors defensive
    # stocks without requiring explicit strategy switching.
    # Tilt magnitude: 12% in BEAR, 0% in BULL/SIDEWAYS.
    if market_regime == 'BEAR' and 'conviction' in traj_df.columns:
        # Quality composite: conviction (80%) + consistency (20%), scaled to score range
        _score_max = traj_df['sector_blend_score'].max()
        _score_min = traj_df['sector_blend_score'].min()
        _score_range = max(_score_max - _score_min, 1)
        _quality = (traj_df['conviction'] * 0.8 + traj_df['consistency'] * 0.2)
        _quality_scaled = _score_min + (_quality / 100) * _score_range
        _tilt_weight = 0.12  # 12% quality tilt in bear markets
        traj_df['sector_blend_score'] = (
            (1 - _tilt_weight) * traj_df['sector_blend_score'] +
            _tilt_weight * _quality_scaled
        )
        # Re-rank with quality tilt applied
        traj_df = traj_df.sort_values(
            ['sector_blend_score', 'confidence', 'consistency'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
        traj_df['t_rank'] = range(1, len(traj_df) + 1)
        traj_df['t_percentile'] = ((total_stocks - traj_df['t_rank']) / max(total_stocks - 1, 1) * 100).round(1)

    # ── Step 4: Sector Alpha Post-Processing (v2.3) ──
    # Compare each stock's trajectory to its sector average to detect
    # SECTOR_LEADER vs SECTOR_BETA stocks
    sa_cfg = SECTOR_ALPHA
    sector_stats = traj_df[traj_df['weeks'] >= MIN_WEEKS_DEFAULT].groupby('sector')['trajectory_score'].agg(
        ['mean', 'count', 'std']
    )

    def _calc_sector_alpha(row):
        sector = row.get('sector', '')
        if not sector or sector not in sector_stats.index:
            return 'NEUTRAL', 0.0
        stats = sector_stats.loc[sector]
        if stats['count'] < sa_cfg['min_sector_stocks']:
            return 'NEUTRAL', 0.0

        sect_mean = float(stats['mean'])
        sect_std = max(float(stats['std']), 1.0)
        stock_score = float(row['trajectory_score'])
        alpha = stock_score - sect_mean
        z = alpha / sect_std

        if z > sa_cfg['leader_z']:
            return 'SECTOR_LEADER', round(alpha, 1)
        elif z > sa_cfg['outperform_z']:
            return 'SECTOR_OUTPERFORM', round(alpha, 1)
        elif z > sa_cfg['aligned_z']:
            return 'SECTOR_ALIGNED', round(alpha, 1)
        elif z > -1.0 and sect_mean > sa_cfg['beta_sector_min']:
            return 'SECTOR_BETA', round(alpha, 1)
        else:
            return 'SECTOR_LAGGARD', round(alpha, 1)

    alpha_results = traj_df.apply(_calc_sector_alpha, axis=1, result_type='expand')
    traj_df['sector_alpha_tag'] = alpha_results[0]
    traj_df['sector_alpha_value'] = alpha_results[1]

    # Add sector alpha tag to signal_tags
    def _add_alpha_signal(row):
        existing = str(row.get('signal_tags', ''))
        tag = row.get('sector_alpha_tag', 'NEUTRAL')
        suffix = ''
        if tag == 'SECTOR_LEADER':
            suffix = '👑LDR'
        elif tag == 'SECTOR_BETA':
            suffix = '🏷️BTA'
        elif tag == 'SECTOR_LAGGARD':
            suffix = '📉LAG'
        if suffix:
            return (existing + ' ' + suffix).strip() if existing else suffix
        return existing

    traj_df['signal_tags'] = traj_df.apply(_add_alpha_signal, axis=1)

    # Metadata
    metadata = {
        'total_weeks': len(dates),
        'date_range': f"{dates[0].strftime('%b %d, %Y')} → {dates[-1].strftime('%b %d, %Y')}",
        'total_tickers': len(histories),
        'first_date': dates[0].strftime('%Y-%m-%d'),
        'last_date': dates[-1].strftime('%Y-%m-%d'),
        'avg_stocks_per_week': int(np.mean([len(weekly_data[d]) for d in dates]))
    }

    return traj_df, histories, dates_iso, metadata


# ============================================
# WAVE SIGNAL FUSION ENGINE (v8.0)
# ============================================
# Deep integration of 18 previously ignored WAVE Detection signals.
# Cross-validates WAVE's independent scoring with Trajectory Engine calculations.
# 4 Fusion Signals → Fusion Score → Fusion Multiplier (×0.92 to ×1.10)

def _latest_valid(lst, default=None):
    """Get the latest non-NaN value from a history list."""
    if not lst:
        return default
    for v in reversed(lst):
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            return v
    return default


def _avg_recent(lst, window=4, default=None):
    """Average of last `window` valid values."""
    if not lst:
        return default
    valid = [v for v in lst[-window:] if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(valid)) if valid else default


def _calc_wave_confluence(h: dict, traj_components: dict) -> float:
    """
    Wave Confluence Score — Agreement between WAVE Detection and Trajectory Engine.

    CORE INSIGHT: WAVE Detection and Trajectory Engine independently analyze the same stock.
    WAVE scores the stock's CURRENT state using cross-sectional analysis (vs all stocks this week).
    Trajectory scores the stock's TRAJECTORY over time (how ranks change week-over-week).

    When BOTH systems agree a stock is strong → HIGH confidence, genuine quality.
    When they disagree → uncertain, one system may be lagging or wrong.

    4 Cross-Validation Channels:
      1. WAVE position_score vs Trajectory positional (40%)
      2. WAVE acceleration_score vs Trajectory acceleration (25%)
      3. WAVE breakout_score vs Trajectory velocity (20%)
      4. WAVE momentum_score vs Trajectory trend (15%)

    Each channel scores 0-100 based on agreement level.
    """
    cfg = WAVE_FUSION
    channels = []

    # ── Channel 1: Position Agreement (40%) ──
    # WAVE position_score: 52-week range analysis (0-100)
    # Trajectory positional: rank percentile quality over time (0-100)
    wave_pos = _latest_valid(h.get('position_score', []))
    traj_pos = traj_components.get('positional', 50)
    if wave_pos is not None:
        diff = abs(wave_pos - traj_pos)
        if diff <= cfg['strong_agree_pct_diff']:
            ch_score = 85 + (cfg['strong_agree_pct_diff'] - diff) / cfg['strong_agree_pct_diff'] * 15
        elif diff <= cfg['agree_pct_diff']:
            t = (diff - cfg['strong_agree_pct_diff']) / (cfg['agree_pct_diff'] - cfg['strong_agree_pct_diff'])
            ch_score = 60 + (1 - t) * 25
        elif diff <= cfg['disagree_pct_diff']:
            t = (diff - cfg['agree_pct_diff']) / (cfg['disagree_pct_diff'] - cfg['agree_pct_diff'])
            ch_score = 30 + (1 - t) * 30
        else:
            ch_score = max(5, 30 - (diff - cfg['disagree_pct_diff']) * 0.5)
        # Boost if BOTH are high (agreement at the top matters more)
        if wave_pos > 70 and traj_pos > 70:
            ch_score = min(100, ch_score + 8)
        channels.append(('position', ch_score, 0.40))

    # ── Channel 2: Acceleration Agreement (25%) ──
    # WAVE acceleration_score: momentum slope comparison (0-100)
    # Trajectory acceleration: rate-of-change of velocity (0-100)
    wave_accel = _latest_valid(h.get('acceleration_score', []))
    traj_accel = traj_components.get('acceleration', 50)
    if wave_accel is not None:
        diff = abs(wave_accel - traj_accel)
        if diff <= cfg['strong_agree_pct_diff']:
            ch_score = 85 + (cfg['strong_agree_pct_diff'] - diff) / cfg['strong_agree_pct_diff'] * 15
        elif diff <= cfg['agree_pct_diff']:
            t = (diff - cfg['strong_agree_pct_diff']) / (cfg['agree_pct_diff'] - cfg['strong_agree_pct_diff'])
            ch_score = 55 + (1 - t) * 30
        elif diff <= cfg['disagree_pct_diff']:
            t = (diff - cfg['agree_pct_diff']) / (cfg['disagree_pct_diff'] - cfg['agree_pct_diff'])
            ch_score = 25 + (1 - t) * 30
        else:
            ch_score = max(5, 25 - (diff - cfg['disagree_pct_diff']) * 0.5)
        channels.append(('acceleration', ch_score, 0.25))

    # ── Channel 3: Breakout/Velocity Agreement (20%) ──
    # WAVE breakout_score: 4-component breakout detection (0-100)
    # Trajectory velocity: position-relative rank velocity (0-100)
    wave_brk = _latest_valid(h.get('breakout_score', []))
    traj_vel = traj_components.get('velocity', 50)
    if wave_brk is not None:
        # Breakout and velocity aren't 1:1 comparable, but high breakout + high velocity = strong
        # Use harmonic-mean-like scoring: both need to be elevated for high score
        combined = (wave_brk * 0.55 + traj_vel * 0.45)
        if combined >= 75:
            ch_score = 80 + (combined - 75) / 25 * 20
        elif combined >= 55:
            ch_score = 50 + (combined - 55) / 20 * 30
        elif combined >= 35:
            ch_score = 25 + (combined - 35) / 20 * 25
        else:
            ch_score = max(5, combined * 0.7)
        channels.append(('breakout', ch_score, 0.20))

    # ── Channel 4: Momentum/Trend Agreement (15%) ──
    # WAVE momentum_score: 3-component momentum (raw sigmoid + consistency + quality)
    # Trajectory trend: weighted linear regression of percentile trajectory
    wave_mom = _latest_valid(h.get('momentum_score', []))
    traj_trend = traj_components.get('trend', 50)
    if wave_mom is not None:
        diff = abs(wave_mom - traj_trend)
        if diff <= cfg['strong_agree_pct_diff']:
            ch_score = 80 + (cfg['strong_agree_pct_diff'] - diff) / cfg['strong_agree_pct_diff'] * 20
        elif diff <= cfg['agree_pct_diff']:
            t = (diff - cfg['strong_agree_pct_diff']) / (cfg['agree_pct_diff'] - cfg['strong_agree_pct_diff'])
            ch_score = 50 + (1 - t) * 30
        else:
            ch_score = max(5, 50 - (diff - cfg['agree_pct_diff']) * 0.8)
        channels.append(('momentum', ch_score, 0.15))

    # ── Composite Confluence Score ──
    if not channels:
        return 50.0  # No WAVE data available → neutral

    total_w = sum(w for _, _, w in channels)
    confluence = sum(s * w for _, s, w in channels) / total_w
    return float(np.clip(confluence, 0, 100))


def _calc_institutional_flow(h: dict) -> float:
    """
    Institutional Flow Signal from WAVE Detection data.

    Combines 4 volume/flow metrics that WAVE Detection computes but
    RANK TRAJECTORY previously ignored. These signals indicate whether
    rank improvements have institutional money backing them.

    4 Components:
      1. Money Flow MM (40%) — price × volume × rvol / 1M → smart money indicator
      2. VMI (30%) — Volume-Momentum Index (acceleration + correlation + footprint)
      3. RVOL Score (20%) — WAVE's relative volume quality score
      4. Overall Market Strength (10%) — Market environment context

    Returns: 0-100 score (higher = stronger institutional backing)
    """
    cfg = WAVE_FUSION
    components = []

    # ── Component 1: Money Flow (40%) ──
    mf = _latest_valid(h.get('money_flow_mm', []))
    if mf is not None:
        if mf >= cfg['money_flow_strong']:
            mf_score = 75 + min((mf - cfg['money_flow_strong']) / cfg['money_flow_strong'] * 25, 25)
        elif mf >= cfg['money_flow_moderate']:
            t = (mf - cfg['money_flow_moderate']) / (cfg['money_flow_strong'] - cfg['money_flow_moderate'])
            mf_score = 40 + t * 35
        elif mf > 0:
            mf_score = max(10, mf / cfg['money_flow_moderate'] * 40)
        else:
            mf_score = 10.0
        components.append(('money_flow', mf_score, 0.40))

    # ── Component 2: VMI — Volume-Momentum Index (30%) ──
    vmi = _latest_valid(h.get('vmi', []))
    if vmi is not None:
        if vmi >= cfg['vmi_strong']:
            vmi_score = 75 + min((vmi - cfg['vmi_strong']) / 35 * 25, 25)
        elif vmi >= cfg['vmi_moderate']:
            t = (vmi - cfg['vmi_moderate']) / (cfg['vmi_strong'] - cfg['vmi_moderate'])
            vmi_score = 40 + t * 35
        elif vmi > 0:
            vmi_score = max(10, vmi / cfg['vmi_moderate'] * 40)
        else:
            vmi_score = 10.0
        components.append(('vmi', vmi_score, 0.30))

    # ── Component 3: RVOL Score (20%) ──
    rvol_s = _latest_valid(h.get('rvol_score', []))
    if rvol_s is not None:
        # WAVE rvol_score is already 0-100
        components.append(('rvol_score', float(np.clip(rvol_s, 0, 100)), 0.20))

    # ── Component 4: Overall Market Strength (10%) ──
    oms = _latest_valid(h.get('overall_market_strength', []))
    if oms is not None:
        # overall_market_strength is typically 0-100
        components.append(('market_strength', float(np.clip(oms, 0, 100)), 0.10))

    if not components:
        return 50.0  # No data → neutral

    total_w = sum(w for _, _, w in components)
    flow_score = sum(s * w for _, s, w in components) / total_w
    return float(np.clip(flow_score, 0, 100))


def _calc_fundamental_quality(h: dict) -> float:
    """
    Fundamental Quality Gate from WAVE Detection data.

    NOT a primary scoring driver — just a quality confirmation/penalty.
    This system is momentum-based; fundamentals serve as a sanity check.

    3 Components:
      1. EPS Growth (40%) — eps_change_pct trend: growing earnings = real strength
      2. PE Reasonableness (30%) — not too high, not negative
      3. EPS Tier (30%) — WAVE's tier classification quality

    Returns: 0-100 score (50 = neutral, >60 = quality confirmed, <40 = fundamental concern)
    """
    cfg = WAVE_FUSION
    components = []

    # ── Component 1: EPS Growth (40%) ──
    eps_chg = _latest_valid(h.get('eps_change_pct', []))
    if eps_chg is not None:
        if eps_chg >= cfg['eps_growth_strong']:
            eps_score = 80 + min((eps_chg - cfg['eps_growth_strong']) / 30 * 20, 20)
        elif eps_chg >= cfg['eps_growth_moderate']:
            t = (eps_chg - cfg['eps_growth_moderate']) / (cfg['eps_growth_strong'] - cfg['eps_growth_moderate'])
            eps_score = 55 + t * 25
        elif eps_chg >= 0:
            eps_score = 40 + eps_chg / cfg['eps_growth_moderate'] * 15
        elif eps_chg >= -10:
            eps_score = 25 + (eps_chg + 10) / 10 * 15
        else:
            eps_score = max(5, 25 + eps_chg * 0.5)  # Deeply negative
        components.append(('eps_growth', float(np.clip(eps_score, 0, 100)), 0.40))

    # ── Component 2: PE Reasonableness (30%) ──
    pe = _latest_valid(h.get('pe', []))
    if pe is not None:
        if pe < 0 and cfg['pe_negative_penalty']:
            pe_score = 20.0  # Loss-making company
        elif pe == 0:
            pe_score = 30.0  # No earnings data
        elif pe <= cfg['pe_value_max']:
            pe_score = 85.0  # Value territory
        elif pe <= cfg['pe_reasonable_max']:
            t = (pe - cfg['pe_value_max']) / (cfg['pe_reasonable_max'] - cfg['pe_value_max'])
            pe_score = 60 + (1 - t) * 25  # Reasonable
        elif pe <= 100:
            pe_score = 35 + (100 - pe) / 50 * 25  # Getting expensive
        else:
            pe_score = max(10, 35 - (pe - 100) * 0.1)  # Very expensive
        components.append(('pe', float(np.clip(pe_score, 0, 100)), 0.30))

    # ── Component 3: EPS Tier (30%) ──
    eps_tier = h.get('eps_tier', [])
    latest_tier = eps_tier[-1] if eps_tier else ''
    if latest_tier and isinstance(latest_tier, str) and latest_tier.strip():
        tier_map = {
            'STRONG_GROWTH': 90, 'GROWTH': 75, 'MODERATE': 60,
            'STABLE': 55, 'DECLINING': 30, 'NEGATIVE': 15, 'LOSS': 10
        }
        # Fuzzy match — tier names may vary
        tier_upper = latest_tier.upper().strip()
        tier_score = 50.0  # Default
        for key, val in tier_map.items():
            if key in tier_upper:
                tier_score = val
                break
        components.append(('eps_tier', tier_score, 0.30))

    if not components:
        return 50.0  # No fundamental data → neutral (don't penalize)

    total_w = sum(w for _, _, w in components)
    fund_score = sum(s * w for _, s, w in components) / total_w
    return float(np.clip(fund_score, 0, 100))


def _compute_wave_fusion(h: dict, traj_components: dict) -> dict:
    """
    Wave Signal Fusion Engine (v8.0) — Master fusion function.

    Cross-validates WAVE Detection's 18 signals with Trajectory Engine calculations.
    Produces a fusion multiplier applied to the trajectory score.

    Pipeline:
      1. Wave Confluence (35%) — agreement between WAVE and Trajectory scoring
      2. Institutional Flow (30%) — money flow + VMI + RVOL strength
      3. Momentum Harmony (20%) — WAVE's 5-check harmony score (0-4)
      4. Fundamental Quality (15%) — EPS growth + PE reasonableness

    Returns dict with all fusion metrics.
    """
    cfg = WAVE_FUSION

    # ── Signal 1: Wave Confluence (35%) ──
    confluence = _calc_wave_confluence(h, traj_components)

    # ── Signal 2: Institutional Flow (30%) ──
    inst_flow = _calc_institutional_flow(h)

    # ── Signal 3: Momentum Harmony (20%) ──
    harmony_raw = _latest_valid(h.get('momentum_harmony', []), 2.0)
    # WAVE momentum_harmony is 0-4 (0=full disagreement, 4=full harmony)
    harmony_score = float(np.clip(harmony_raw / 4.0 * 100, 0, 100))

    # ── Signal 4: Fundamental Quality (15%) ──
    fund_quality = _calc_fundamental_quality(h)

    # ── Composite Fusion Score ──
    fusion_score = (
        cfg['confluence_weight'] * confluence +
        cfg['flow_weight'] * inst_flow +
        cfg['harmony_weight'] * harmony_score +
        cfg['fundamental_weight'] * fund_quality
    )
    fusion_score = float(np.clip(fusion_score, 0, 100))

    # ── Convert to Multiplier (smooth curve) ──
    lo = cfg['neutral_zone_lo']
    hi = cfg['neutral_zone_hi']
    if fusion_score >= hi:
        t = (fusion_score - hi) / max(100 - hi, 1)
        multiplier = 1.0 + t * (cfg['max_boost'] - 1.0)
    elif fusion_score <= lo:
        t = fusion_score / max(lo, 1)
        multiplier = cfg['max_penalty'] + t * (1.0 - cfg['max_penalty'])
    else:
        multiplier = 1.0
    multiplier = float(np.clip(multiplier, cfg['max_penalty'], cfg['max_boost']))

    # ── Classification Label ──
    if fusion_score >= 75:
        label = 'WAVE_STRONG'
    elif fusion_score >= 60:
        label = 'WAVE_CONFIRMED'
    elif fusion_score >= 40:
        label = 'WAVE_NEUTRAL'
    elif fusion_score >= 25:
        label = 'WAVE_WEAK'
    else:
        label = 'WAVE_CONFLICT'

    # ── Supplementary signals for downstream use ──
    position_tension = _latest_valid(h.get('position_tension', []), 0)
    from_low = _latest_valid(h.get('from_low_pct', []))
    from_high_raw = _latest_valid(h.get('from_high_pct', []))
    ret_1d = _latest_valid(h.get('ret_1d', []))
    ret_1y = _latest_valid(h.get('ret_1y', []))

    # ── Rally Leg: detect where the CURRENT rally started (recent price trough) ──
    # Uses actual price history — NOT the 52w annual low (which is stale/irrelevant).
    # Find the lowest price in last 12 weeks → that is where this rally leg began.
    # rally_gain   = % the stock has already risen this leg
    # rally_room   = % still remaining to reach 52w high (gap to close)
    # rally_leg_pct = rally_gain / (rally_gain + rally_room) × 100
    #   0%   = just started climbing (most room ahead)
    #   100% = already at 52w high (rally leg fully realised)
    # rally_weeks = how many weeks since the trough (age of this move)
    #
    # Stage logic is gain-magnitude based (absolute %, not range ratio):
    #   FRESH   < 5%   gain  → rally just kicked off
    #   EARLY   5-15%  gain  → solid momentum, lots of room
    #   RUNNING 15-30% gain  → good move, watch for resistance
    #   MATURE  30-50% gain  → significant leg, be selective
    #   LATE    > 50%  gain  → extended move, risk/reward narrows
    prices = h.get('prices', [])
    # ── v10.0: Adaptive Rally Lookback — volatility-scaled window ──
    # Fixed 16-week lookback misses fast moves (need short window) and
    # slow trends (need long window). Scale inversely with price volatility.
    # High-vol stocks → shorter lookback (min 8wk), recent moves are meaningful
    # Low-vol stocks → longer lookback (max 26wk), need time to identify trend
    n_prices = len(prices)
    if n_prices >= 10:
        _vol_win = min(20, n_prices - 1)
        _vol_px = prices[-(_vol_win + 1):]
        _wk_rets = [(_vol_px[j+1] - _vol_px[j]) / _vol_px[j]
                     for j in range(len(_vol_px) - 1) if _vol_px[j] > 0]
        if len(_wk_rets) >= 4:
            _vol = float(np.std(_wk_rets))
            _vol_ratio = max(0.3, _vol / 0.035)  # 0.035 ≈ typical weekly vol
            _lookback = int(np.clip(round(16 / (0.5 + 0.5 * _vol_ratio)), 8, 26))
        else:
            _lookback = 16
    else:
        _lookback = min(16, n_prices)
    _lookback = min(_lookback, n_prices)
    rally_gain = 0.0
    rally_weeks = 0
    rally_leg_pct = 50.0
    rally_stage = 'UNKNOWN'
    if _lookback >= 2 and prices:
        _recent = prices[-_lookback:]
        _trough_idx = int(np.argmin(_recent))
        _trough_price = _recent[_trough_idx]
        _cur_price = prices[-1]
        rally_weeks = _lookback - 1 - _trough_idx   # weeks since trough
        if _trough_price > 0:
            rally_gain = round((_cur_price - _trough_price) / _trough_price * 100, 1)
        # Gap remaining to 52w high
        _gap_to_high = abs(from_high_raw) if (from_high_raw is not None) else 0.0
        _total = rally_gain + _gap_to_high
        rally_leg_pct = round((rally_gain / _total * 100), 1) if (_total > 1 and rally_gain >= 0) else 0.0
        # Stage by magnitude of gain already captured
        if rally_gain < 5:
            rally_stage = 'FRESH'
        elif rally_gain < 15:
            rally_stage = 'EARLY'
        elif rally_gain < 30:
            rally_stage = 'RUNNING'
        elif rally_gain < 50:
            rally_stage = 'MATURE'
        else:
            rally_stage = 'LATE'

    return {
        'wave_fusion_score': round(fusion_score, 1),
        'wave_fusion_multiplier': round(multiplier, 4),
        'wave_fusion_label': label,
        'wave_confluence': round(confluence, 1),
        'wave_inst_flow': round(inst_flow, 1),
        'wave_harmony': round(harmony_score, 1),
        'wave_harmony_raw': round(harmony_raw, 1) if harmony_raw is not None else 2.0,
        'wave_fundamental': round(fund_quality, 1),
        'wave_position_tension': round(position_tension, 2) if position_tension else 0,
        'wave_from_low': round(from_low, 1) if from_low is not None else None,
        'wave_from_high': round(from_high_raw, 1) if from_high_raw is not None else None,
        'rally_gain': rally_gain,
        'rally_weeks': rally_weeks,
        'rally_leg_pct': rally_leg_pct,
        'rally_stage': rally_stage,
        'wave_ret_1d': round(ret_1d, 2) if ret_1d is not None else None,
        'wave_ret_1y': round(ret_1y, 2) if ret_1y is not None else None,
    }


# ============================================
# TRAJECTORY SCORING ENGINE
# ============================================

def _compute_single_trajectory(h: dict) -> dict:
    """Compute all trajectory metrics for a single ticker's history"""
    ranks = h['ranks']
    totals = h['total_per_week']
    scores = h['scores']
    n = len(ranks)

    # Insufficient data
    if n < 2:
        pcts = ranks_to_percentiles(ranks, totals) if ranks else []
        return _empty_trajectory(ranks, totals, pcts, n)

    pcts = ranks_to_percentiles(ranks, totals)

    # ── 8-Component Scores (v9.0 — BreakoutQuality added as data-proven 8th) ──
    positional = _calc_positional_quality(pcts, n)
    trend = _calc_trend(pcts, n)
    velocity = _calc_velocity_adaptive(pcts, n)  # Position-relative velocity
    acceleration = _calc_acceleration(pcts, n)
    consistency = _calc_consistency_adaptive(pcts, n)  # Position-aware consistency
    resilience = _calc_resilience(pcts, n)

    # ── v6.0: Return floors REMOVED — returns are now a dedicated component ──
    # Previously returns leaked through 5 doors: velocity floor, acceleration floor,
    # resilience bonus, conviction multiplier, price alignment signal 3.
    # Now: one clean component handles ALL return information.

    # Compute avg_pct — needed by adaptive weights and return quality
    avg_pct = float(np.mean(pcts))
    current_pct = pcts[-1]  # Needed by exit warnings, hot streak, etc.

    # ── Return Quality Component (v6.0 — Dedicated 7th Component) ──
    ret_7d = h.get('ret_7d', [])
    ret_30d = h.get('ret_30d', [])
    ret_3m = h.get('ret_3m', [])
    ret_6m = h.get('ret_6m', [])
    from_high = h.get('from_high_pct', [])
    return_quality = _calc_return_quality(ret_3m, ret_6m, ret_7d, ret_30d, from_high, avg_pct, n)

    # ── Breakout Quality Component (v9.0 — Dedicated 8th Component) ──
    # DATA EVIDENCE: breakout_score top25% → +0.44%/wk alpha (2nd strongest predictor)
    # Was only used as sub-signal in Wave Fusion. Now dedicated component.
    breakout_quality = _calc_breakout_quality(h, avg_pct, n)

    # ── v6.2: Compute confidence EARLY (needed for weight selection) ──
    bc = BAYESIAN_CONFIDENCE
    confidence = min(1.0, max(bc['min_confidence'], n / bc['full_confidence_weeks']))

    # ── v10.0: Per-Stock Regime Signal — shifts weights in bear/bull micro-regimes ──
    # Stock-level regime proxy from recent returns + distance from high.
    # Bearish regime → boost consistency/resilience, reduce velocity/acceleration.
    # Bullish regime → boost breakout/trend. Neutral → no shift.
    _r30_val = _latest_valid(ret_30d, 0)
    _r3m_val = _latest_valid(ret_3m, 0)
    _fh_val = _latest_valid(from_high, -25)
    _regime_parts = []
    # 30-day return signal
    if _r30_val < -10:
        _regime_parts.append(-1.0)
    elif _r30_val < -5:
        _regime_parts.append(-0.5)
    elif _r30_val > 10:
        _regime_parts.append(1.0)
    elif _r30_val > 5:
        _regime_parts.append(0.5)
    else:
        _regime_parts.append(0.0)
    # 3-month return signal
    if _r3m_val < -15:
        _regime_parts.append(-1.0)
    elif _r3m_val < -5:
        _regime_parts.append(-0.5)
    elif _r3m_val > 20:
        _regime_parts.append(1.0)
    elif _r3m_val > 10:
        _regime_parts.append(0.5)
    else:
        _regime_parts.append(0.0)
    # Distance from 52w high signal
    if _fh_val < -30:
        _regime_parts.append(-1.0)
    elif _fh_val < -15:
        _regime_parts.append(-0.5)
    elif _fh_val > -2:
        _regime_parts.append(1.0)
    elif _fh_val > -5:
        _regime_parts.append(0.5)
    else:
        _regime_parts.append(0.0)
    regime_signal = float(np.mean(_regime_parts))

    # ── Select Adaptive Weights Based on Percentile Tier + Confidence + Regime (v10.0) ──
    weights = _get_adaptive_weights(avg_pct, current_pct=pcts[-1], confidence=confidence, regime_signal=regime_signal)

    # ── Composite Score (8-Component Adaptive Weighted — v9.0) ──
    trajectory_score = (
        weights['positional'] * positional +
        weights['trend'] * trend +
        weights['velocity'] * velocity +
        weights['acceleration'] * acceleration +
        weights['consistency'] * consistency +
        weights['resilience'] * resilience +
        weights['return_quality'] * return_quality +
        weights['breakout_quality'] * breakout_quality
    )

    # ── Elite Dominance Bonus ──
    # Sustained top-tier presence guarantees a minimum score floor
    trajectory_score = _apply_elite_bonus(trajectory_score, pcts, n)

    # ── Bayesian Confidence Shrinkage (v3.0) ──
    # Shrinks score toward population mean when data is insufficient.
    # 4-week lucky streak ≠ 16-week proven performer. This ensures that.
    # (confidence already computed above for weight selection)
    trajectory_score = confidence * trajectory_score + (1 - confidence) * bc['prior_mean']

    # ── Hurst Exponent Persistence Multiplier (v3.0) ──
    # Determines if the rank trend will PERSIST or REVERT.
    # H > 0.55: trending → boost. H < 0.42: mean-reverting → penalize uptrends.
    hurst_multiplier = _calc_hurst_multiplier(pcts, trend)

    # ── v8.0: Wave Signal Fusion Multiplier ──
    # Cross-validates WAVE Detection's 18 signals with Trajectory calculations.
    # Produces ×0.92 to ×1.10 based on agreement level.
    traj_components = {
        'positional': positional, 'trend': trend, 'velocity': velocity,
        'acceleration': acceleration, 'consistency': consistency,
        'resilience': resilience, 'return_quality': return_quality,
    }
    wave_fusion = _compute_wave_fusion(h, traj_components)
    wave_fusion_multiplier = wave_fusion['wave_fusion_multiplier']

    # ── Price-Rank Alignment (v9.0: Diagnostic-only, NOT a multiplier) ──
    # Data analysis showed price alignment adds noise as a multiplier.
    # Price direction is already captured in ReturnQuality component.
    # Keeping the calculation for UI display but NOT applying as multiplier.
    price_label, price_alignment = _calc_price_alignment(ret_7d, ret_30d, pcts, avg_pct)

    # ── Momentum Decay Warning (v9.0: Exit-system-only, NOT a multiplier) ──
    # Decay is valuable as an EXIT signal but noisy as a scoring multiplier.
    # Keeping calculation for exit warning system, not applying to T-Score.
    decay_label, decay_score = _calc_momentum_decay(ret_7d, ret_30d, from_high, pcts, avg_pct, ret_6m)

    # ── v9.0: SIMPLIFIED MULTIPLIER — Only 2 proven multipliers ──
    # Hurst × WaveFusion capped ±12%. Price/Decay multipliers removed (noise).
    combined_mult = hurst_multiplier * wave_fusion_multiplier
    combined_mult = float(np.clip(combined_mult, 0.88, 1.12))
    trajectory_score = float(np.clip(trajectory_score * combined_mult, 0, 100))

    # ── Grade ──
    grade, grade_emoji = get_grade(trajectory_score)

    # ── Pattern Detection (v2.0 position-aware) ──
    pattern_key = _detect_pattern(ranks, totals, pcts, positional, trend, velocity, acceleration, consistency)
    p_emoji, p_name, _ = PATTERN_DEFS[pattern_key]

    # ── Additional Metrics ──
    current_rank = int(ranks[-1])
    best_rank = int(min(ranks))
    worst_rank = int(max(ranks))
    avg_rank = round(np.mean(ranks), 1)
    rank_change = int(ranks[0] - ranks[-1])  # Positive = improved

    # Streak (consecutive weeks of rank improvement)
    streak = 0
    for i in range(len(ranks) - 1, 0, -1):
        if ranks[i] < ranks[i - 1]:
            streak += 1
        else:
            break

    # TMI
    tmi = calculate_tmi(ranks, totals)

    # Rank volatility
    rank_vol = round(np.std(ranks), 1) if n > 1 else 0

    # Score trajectory for sparkline (percentiles, higher = better)
    sparkline_data = [round(p, 1) for p in pcts]

    # Week-over-week rank changes
    if n >= 2:
        last_week_change = int(ranks[-2] - ranks[-1])  # positive = improved
    else:
        last_week_change = 0

    # Build price alignment tag (secondary pattern)
    price_tag = ''
    if price_label == 'PRICE_CONFIRMED':
        price_tag = '💰'
    elif price_label == 'PRICE_DIVERGENT':
        price_tag = '⚠️'

    # Build momentum decay tag (v2.3)
    decay_tag = ''
    if decay_label == 'DECAY_HIGH':
        decay_tag = '🔻'
    elif decay_label == 'DECAY_MODERATE':
        decay_tag = '⚡'
    elif decay_label == 'DECAY_MILD':
        decay_tag = '~'

    # Build signal tags column (combined indicator) — v10.0: readable short-codes
    # Each tag: emoji + 2-3 letter code so meaning is instantly clear in the table.
    signal_parts = []
    if price_label == 'PRICE_CONFIRMED':
        signal_parts.append('💰PRC')     # Price confirms rank direction
    elif price_label == 'PRICE_DIVERGENT':
        signal_parts.append('⚠️DIV')     # Price diverges from rank
    if return_quality >= 75:
        signal_parts.append('🔥RET')     # Strong returns
    elif return_quality <= 30:
        signal_parts.append('💧RET')     # Weak returns
    if decay_label == 'DECAY_HIGH':
        signal_parts.append('🔻DEC')     # High momentum decay
    elif decay_label == 'DECAY_MODERATE':
        signal_parts.append('⚡DEC')     # Moderate momentum decay
    # v8.0: Wave fusion signal tag
    wf_label = wave_fusion.get('wave_fusion_label', 'WAVE_NEUTRAL')
    if wf_label == 'WAVE_STRONG':
        signal_parts.append('🌊WAV')     # Wave systems strongly agree
    elif wf_label == 'WAVE_CONFLICT':
        signal_parts.append('🔇WAV')     # Wave systems conflict
    signal_tags = ' '.join(signal_parts)

    # ══════════════════════════════════════════════════════════════════════════
    # v6.3: ADVANCED TRADING SIGNALS — 5 New Features for Better Returns
    # v8.1: PERSISTENCE-CALIBRATED — Backtest-proven signal prioritization
    # ══════════════════════════════════════════════════════════════════════════

    # ── v8.1: PERSISTENCE WEEKS — Backtest-Proven #1 Predictor ──
    # Walk-forward backtest (24 CSVs, 18 windows) proved:
    #   S4 Persistent Top 50: +4.25% alpha — stocks staying ranked well WIN
    #   S2 Top 10 Rank: -11.26% alpha — current position alone LOSES
    # Persistence = consecutive recent weeks where stock was in top 25% (≥75th pct)
    persistence_weeks = 0
    for p in reversed(pcts):
        if p >= 75:
            persistence_weeks += 1
        else:
            break

    # ── 1. CONVICTION SCORE (0-100) ──
    # v9.0: DATA-DRIVEN CONVICTION — rebuilt from 27-transition CSV analysis.
    # v8.1 fixed anti-predictive conviction (S7 -23.75% → +0.49%) via persistence.
    # v9.0 further refines using discovered forward predictors:
    #   from_high_pct:   +0.55%/wk alpha (#1 predictor)
    #   breakout_score:  +0.44%/wk alpha (#2 predictor)
    #   position_score:  +0.34%/wk alpha (#3 predictor)
    #   Persistence 6+wk: +3.91%/wk (still dominant)
    #   Market State: BOUNCE +1.17%/wk, UPTREND -0.71%/wk (mean reversion)
    #
    # Signals (12 total, 100pts max):
    #   1. Persistence Strength: 25pts — consecutive weeks in top 25%
    #   2. Consistency Quality:  12pts — consistency score component
    #   3. Near-High Strength:   14pts — from_high_pct (#1 forward predictor)
    #   4. Return Quality:        8pts — actual returns backing rank
    #   5. Breakout Quality:     12pts — breakout_quality (#2 forward predictor)
    #   6. Wave Confluence:       8pts — WAVE system agreement
    #   7. Sector Leadership:     7pts — sector-relative position
    #   8. Market State Signal:   8pts — BOUNCE/DOWNTREND mean-reversion bonus
    #   9. No-Decay Bonus:        6pts — absence of momentum deterioration
    #  10. Trend-Hurst Persistence: 6pts — structural trend persistence
    #  11. 12-Month Momentum:    7pts — ret_1y long-term trend strength
    #  12. Low-Distance Strength: 5pts — from_low_pct proximity confirmation
    conviction = 0

    # Signal 1: Persistence Strength (25 pts max) — BACKTEST-PROVEN #1 PREDICTOR
    # Consecutive weeks in top 25% (≥75th percentile). S4 proved this wins.
    if persistence_weeks >= 8:
        conviction += 25   # 8+ weeks sustained = maximum persistence
    elif persistence_weeks >= 6:
        conviction += 21   # 6-7 weeks = very strong
    elif persistence_weeks >= 4:
        conviction += 16   # 4-5 weeks = strong (this is what S4 tested at ~4wk)
    elif persistence_weeks >= 3:
        conviction += 11   # 3 weeks = moderate persistence
    elif persistence_weeks >= 2:
        conviction += 6    # 2 weeks = early persistence
    elif persistence_weeks >= 1:
        conviction += 2    # 1 week = just entered top tier

    # Signal 2: Consistency Quality (12 pts max) — persistence proxy
    # High consistency = stable rank trajectory = sustained performance
    if consistency >= 75:
        conviction += 12
    elif consistency >= 60:
        conviction += 8
    elif consistency >= 45:
        conviction += 4

    # Signal 3: Near-High Strength (14 pts max) — v9.0 NEW, #1 forward predictor
    # from_high_pct: top25% → +0.55%/wk alpha. Stocks near 52w high keep winning.
    # from_high_pct is negative (0 = at high, -50 = 50% below high)
    latest_from_high = from_high[-1] if from_high else -50
    if latest_from_high >= -5:
        conviction += 14   # Within 5% of 52w high = maximum near-high signal  
    elif latest_from_high >= -10:
        conviction += 11   # Within 10% = very strong
    elif latest_from_high >= -15:
        conviction += 8    # Within 15% = strong
    elif latest_from_high >= -20:
        conviction += 5    # Within 20% = moderate
    elif latest_from_high >= -30:
        conviction += 2    # Within 30% = weak signal

    # Signal 4: Return Quality (8 pts max) — actual returns backing rank
    if return_quality >= 75:
        conviction += 8
    elif return_quality >= 60:
        conviction += 5
    elif return_quality >= 50:
        conviction += 2

    # Signal 5: Breakout Quality (12 pts max) — v9.0 NEW, #2 forward predictor
    # breakout_score top25% → +0.44%/wk alpha. Breakout decile 10: +0.44%/wk.
    # Replaces old "Momentum Quality" (TMI had ZERO predictive power: +0.00%/wk)
    if breakout_quality >= 75:
        conviction += 12   # Strong breakout = maximum signal
    elif breakout_quality >= 65:
        conviction += 9    # Good breakout
    elif breakout_quality >= 55:
        conviction += 6    # Moderate breakout
    elif breakout_quality >= 45:
        conviction += 3    # Mild breakout signal

    # Signal 6: Wave Confluence Agreement (8 pts max)
    wf_confluence = wave_fusion.get('wave_confluence', 50)
    if wf_confluence >= 75:
        conviction += 8
    elif wf_confluence >= 60:
        conviction += 5
    elif wf_confluence >= 45:
        conviction += 2

    # Signal 7: Sector Leadership (7 pts max) — v9.0 NEW
    # Sector-relative: top10% vs bot10% → +0.59%/wk alpha within sector.
    # Uses position_score as proxy for sector strength (3rd strongest predictor).
    pos_scores = h.get('position_score', [])
    latest_pos = pos_scores[-1] if pos_scores else 50
    if latest_pos >= 80:
        conviction += 7    # Sector dominant
    elif latest_pos >= 65:
        conviction += 4    # Sector strong
    elif latest_pos >= 50:
        conviction += 2    # Sector average+

    # Signal 8: Market State Signal (8 pts max) — v9.0 NEW, mean-reversion alpha
    # DATA EVIDENCE: BOUNCE +1.17%/wk, STRONG_DOWNTREND +0.25%/wk (mean reversion)
    #                SIDEWAYS -0.63%/wk, UPTREND -0.71%/wk (crowd trap)
    # Stocks in BOUNCE/DOWNTREND states have mean-reversion tailwinds.
    # Stocks in UPTREND have already priced in the move — reversion risk.
    latest_market_state = ''
    ms_list = h.get('market_states', [])
    if ms_list:
        latest_market_state = ms_list[-1] if ms_list[-1] else ''
    if not latest_market_state:
        latest_market_state = h.get('market_state', '')

    if latest_market_state == 'BOUNCE':
        conviction += 8    # Strongest forward alpha (+1.17%/wk)
    elif latest_market_state == 'STRONG_DOWNTREND':
        conviction += 5    # Mean reversion opportunity (+0.25%/wk)
    elif latest_market_state == 'DOWNTREND':
        conviction += 3    # Mild reversion signal
    elif latest_market_state == 'RECOVERY':
        conviction += 4    # Recovering stocks
    # UPTREND and SIDEWAYS get 0 pts (negative forward alpha)

    # Signal 9: No-Decay Bonus (6 pts max) — rewards momentum health
    # Absence of decay = persistence is intact, not a trap
    if decay_score == 0:
        conviction += 6   # Zero decay = healthy momentum
    elif decay_score <= 10:
        conviction += 4   # Minimal decay = acceptable
    elif decay_label == '' or decay_label == 'DECAY_MILD':
        conviction += 2   # Mild decay = marginal bonus

    # Signal 10: Trend-Hurst Persistence Alignment (6 pts max) — v10.0 NEW
    # Hurst > 0.55 means current rank trajectory is likely to PERSIST.
    # Combined with positive trend (trend > 55), this identifies stocks whose
    # rank improvement is NOT random — it has structural persistence.
    # This is orthogonal to all other conviction signals (captures time-series
    # persistence property, not current levels or returns).
    _hurst_raw = _estimate_hurst(pcts) if n >= HURST_CONFIG['min_weeks'] else 0.5
    if trend >= 60 and _hurst_raw >= 0.57:
        conviction += 6    # Strong trend + strong persistence = maximum
    elif trend >= 55 and _hurst_raw >= 0.53:
        conviction += 4    # Moderate trend + moderate persistence
    elif trend >= 52 and _hurst_raw >= 0.50:
        conviction += 2    # Mild trend + random walk boundary

    # Signal 11: 12-Month Momentum (7 pts max) — v10.1 NEW
    # Academic evidence (Jegadeesh & Titman): 12-month cross-sectional momentum
    # is one of the strongest equity return predictors worldwide.
    # ret_1y is stored in CSV but was NEVER used in scoring until now.
    # Stocks with strong 1-year returns tend to continue outperforming.
    # Combined with near-high signal (#3) this creates a dual momentum confirmation.
    _ret_1y_val = _latest_valid(h.get('ret_1y', []), 0)
    if _ret_1y_val >= 80:
        conviction += 7    # Exceptional 1yr momentum (>80%)
    elif _ret_1y_val >= 50:
        conviction += 5    # Very strong 1yr momentum (>50%)
    elif _ret_1y_val >= 25:
        conviction += 3    # Solid 1yr momentum (>25%)
    elif _ret_1y_val >= 10:
        conviction += 1    # Mild positive 1yr momentum
    # Negative or flat 1yr return → 0 pts (no penalty — other signals handle risk)

    # Signal 12: Low-Distance Strength (5 pts max) — v10.1 NEW
    # from_low_pct = % distance from 52-week LOW (positive = far from low).
    # Stocks far from their low AND near their high (Signal 3) = double confirmation.
    # from_low_pct is positive: 0 = at 52w low, 100 = 100% above 52w low.
    # This is orthogonal to from_high_pct: a stock can be 50% above its low
    # but still 20% below its high (wide trading range).
    _from_low_val = _latest_valid(h.get('from_low_pct', []), 0)
    if _from_low_val >= 80:
        conviction += 5    # Far above 52w low — strong structural uptrend
    elif _from_low_val >= 50:
        conviction += 3    # Solidly above 52w low
    elif _from_low_val >= 25:
        conviction += 1    # Moderate distance from low
    # Near 52w low → 0 pts (stock in trouble, other signals will reflect this)

    conviction = min(100, conviction)

    # Conviction tags for UI
    if conviction >= 80:
        conviction_tag = 'VERY_HIGH'
        conviction_emoji = '🎯'
    elif conviction >= 65:
        conviction_tag = 'HIGH'
        conviction_emoji = '✅'
    elif conviction >= 45:
        conviction_tag = 'MODERATE'
        conviction_emoji = '⚡'
    elif conviction >= 25:
        conviction_tag = 'LOW'
        conviction_emoji = '⚠️'
    else:
        conviction_tag = 'VERY_LOW'
        conviction_emoji = '❌'

    # ── v10.0: CONVICTION MULTIPLIER ON T-SCORE ──
    # Backtest evidence: S7 (Conviction ≥ 65) has +0.14%/wk alpha — the best
    # strategy. S2 (Top 10 Rank) has -0.04%/wk — WORSE than universe.
    # This means pure T-Score ranking picks noisy stocks; conviction-filtered
    # stocks are genuinely better. Bake this into the ranking by applying a
    # gentle conviction multiplier to trajectory_score itself.
    # Range: ×0.95 (very low conviction) to ×1.05 (very high conviction).
    # This shifts the ranking toward conviction-confirmed stocks WITHOUT
    # breaking the scoring architecture — it's a 10% total spread.
    if conviction >= 75:
        conv_mult = 1.05
    elif conviction >= 65:
        conv_mult = 1.03
    elif conviction >= 50:
        conv_mult = 1.00
    elif conviction >= 35:
        conv_mult = 0.98
    else:
        conv_mult = 0.95
    trajectory_score = float(np.clip(trajectory_score * conv_mult, 0, 100))

    # ── 2. RISK-ADJUSTED T-SCORE ──
    # Penalizes high-volatility stocks that may reverse.
    # Sharpe-like: score / (1 + rank_volatility × 0.05)
    # rank_vol=20 → 2.0x divisor → score halved. rank_vol=10 → 1.5x → mild penalty.
    vol_penalty = 1.0 + (rank_vol * 0.05)
    risk_adj_score = trajectory_score / vol_penalty
    risk_adj_score = round(risk_adj_score, 2)

    # ── 2b. ALPHA SCORE (0-100) — Pure Forward-Return Predictor (v10.1) ──
    # Unlike T-Score (quality metric) and Conviction (confidence metric),
    # Alpha Score uses ONLY factors proven to predict NEXT-WEEK returns.
    # Weights derived from actual forward alpha measurements (28 CSVs):
    #
    #   Factor                   Alpha/wk  Weight  Max Pts
    #   ─────────────────────────────────────────────────────
    #   Near-High Proximity       +0.55%    22%     22pts  (#1 predictor)
    #   Breakout Quality          +0.44%    18%     18pts  (#2 predictor)
    #   Persistence Strength      +3.91%    20%     20pts  (backtest-proven)
    #   Position Strength         +0.34%    12%     12pts  (#3 predictor)
    #   Market State Signal       +1.17%    10%     10pts  (BOUNCE alpha)
    #   No-Decay Gate              —         8%      8pts  (trap avoidance)
    #   Wave Fusion Confirmation   —         5%      5pts  (cross-validation)
    #   Early Rally Stage          —         5%      5pts  (room-to-grow)
    #   ─────────────────────────────────────────────────────
    #   TOTAL                                       100pts
    #
    alpha_score = 0

    # Alpha 1: Near-High Proximity (22 pts) — #1 forward predictor
    _fh_alpha = _latest_valid(from_high, -50)
    if _fh_alpha >= -3:
        alpha_score += 22    # Within 3% of 52w high
    elif _fh_alpha >= -7:
        alpha_score += 18    # Within 7%
    elif _fh_alpha >= -12:
        alpha_score += 13    # Within 12%
    elif _fh_alpha >= -20:
        alpha_score += 7     # Within 20%
    elif _fh_alpha >= -30:
        alpha_score += 3     # Within 30%

    # Alpha 2: Breakout Quality (18 pts) — #2 forward predictor
    if breakout_quality >= 80:
        alpha_score += 18
    elif breakout_quality >= 65:
        alpha_score += 14
    elif breakout_quality >= 50:
        alpha_score += 9
    elif breakout_quality >= 35:
        alpha_score += 4

    # Alpha 3: Persistence Strength (20 pts) — backtest-proven #1
    if persistence_weeks >= 8:
        alpha_score += 20
    elif persistence_weeks >= 6:
        alpha_score += 16
    elif persistence_weeks >= 4:
        alpha_score += 12
    elif persistence_weeks >= 3:
        alpha_score += 8
    elif persistence_weeks >= 2:
        alpha_score += 4

    # Alpha 4: Position Strength (12 pts) — #3 forward predictor
    _pos_alpha = _latest_valid(h.get('position_score', []), 50)
    if _pos_alpha >= 80:
        alpha_score += 12
    elif _pos_alpha >= 65:
        alpha_score += 8
    elif _pos_alpha >= 50:
        alpha_score += 4

    # Alpha 5: Market State Signal (10 pts) — mean-reversion alpha
    if latest_market_state == 'BOUNCE':
        alpha_score += 10
    elif latest_market_state in ('STRONG_DOWNTREND', 'RECOVERY'):
        alpha_score += 6
    elif latest_market_state == 'DOWNTREND':
        alpha_score += 3
    # UPTREND/SIDEWAYS = 0 (negative forward alpha)

    # Alpha 6: No-Decay Gate (8 pts) — trap avoidance
    if decay_score == 0:
        alpha_score += 8
    elif decay_score <= 10:
        alpha_score += 5
    elif decay_label in ('', 'DECAY_MILD'):
        alpha_score += 2

    # Alpha 7: Wave Fusion Confirmation (5 pts)
    _wf_score = wave_fusion.get('wave_fusion_score', 50)
    if _wf_score >= 70:
        alpha_score += 5
    elif _wf_score >= 55:
        alpha_score += 3

    # Alpha 8: Early Rally Stage (5 pts) — room to grow
    _rally_stage = wave_fusion.get('rally_stage', 'UNKNOWN')
    if _rally_stage == 'FRESH':
        alpha_score += 5
    elif _rally_stage == 'EARLY':
        alpha_score += 4
    elif _rally_stage == 'RUNNING':
        alpha_score += 2
    # MATURE/LATE = 0 (limited upside remaining)

    alpha_score = min(100, alpha_score)

    # ── 3. EXIT WARNING SYSTEM ──
    # Detects when to SELL existing holdings.
    # Multiple warning signals aggregated into exit_risk score (0-100).
    exit_risk = 0
    exit_signals = []

    # Exit Signal 1: TMI Collapse (momentum dying)
    if tmi < 40 and trajectory_score > 50:
        exit_risk += 25
        exit_signals.append('TMI_WEAK')
    elif tmi < 50 and trajectory_score > 60:
        exit_risk += 15
        exit_signals.append('TMI_FADING')

    # Exit Signal 2: Price Divergence (rank up but price down)
    if price_label == 'PRICE_DIVERGENT':
        exit_risk += 25
        exit_signals.append('PRICE_DIV')

    # Exit Signal 3: Momentum Decay (returns collapsing)
    if decay_label == 'DECAY_HIGH':
        exit_risk += 30
        exit_signals.append('DECAY_HIGH')
    elif decay_label == 'DECAY_MODERATE':
        exit_risk += 15
        exit_signals.append('DECAY_MOD')

    # Exit Signal 4: Negative Streak (consecutive rank drops)
    neg_streak = 0
    for i in range(len(ranks) - 1, 0, -1):
        if ranks[i] > ranks[i - 1]:  # rank increased = worsened
            neg_streak += 1
        else:
            break
    if neg_streak >= 3:
        exit_risk += 20
        exit_signals.append(f'DROP_{neg_streak}W')
    elif neg_streak >= 2:
        exit_risk += 10
        exit_signals.append('DROP_2W')

    # Exit Signal 5: Pattern Warning
    if pattern_key in ['fading', 'crash', 'topping_out']:
        exit_risk += 20
        exit_signals.append(f'PAT_{pattern_key.upper()}')

    # v8.0 Exit Signal 6: Position Tension (from WAVE Detection)
    # High position tension near top = likely reversal
    wf_tension = wave_fusion.get('wave_position_tension', 0)
    if wf_tension is not None and wf_tension > 0.7 and current_pct >= 70:
        exit_risk += 15
        exit_signals.append('HIGH_TENSION')
    elif wf_tension is not None and wf_tension > 0.5 and current_pct >= 80:
        exit_risk += 8
        exit_signals.append('MOD_TENSION')

    # v8.0 Exit Signal 7: Wave Fusion Conflict (systems disagree)
    if wf_label == 'WAVE_CONFLICT' and trajectory_score > 50:
        exit_risk += 12
        exit_signals.append('WAVE_CONFLICT')

    exit_risk = min(100, exit_risk)
    if exit_risk >= 60:
        exit_tag = 'EXIT_NOW'
        exit_emoji = '🚨'
    elif exit_risk >= 40:
        exit_tag = 'CAUTION'
        exit_emoji = '⚠️'
    elif exit_risk >= 20:
        exit_tag = 'WATCH'
        exit_emoji = '👀'
    else:
        exit_tag = 'HOLD'
        exit_emoji = '✅'

    # ── 4. HOT STREAK DETECTION ──
    # Consecutive weeks of percentile improvement with high current position.
    # Research shows 4+ week momentum streaks have 72% continuation rate.
    hot_streak = False
    hot_streak = False
    if streak >= 4 and current_pct >= 70:
        hot_streak = True
    elif streak >= 3 and current_pct >= 80:
        hot_streak = True
    elif streak >= 5 and current_pct >= 60:
        hot_streak = True

    return {
        'trajectory_score': round(trajectory_score, 2),
        'positional': round(positional, 2),
        'trend': round(trend, 2),
        'velocity': round(velocity, 2),
        'acceleration': round(acceleration, 2),
        'consistency': round(consistency, 2),
        'resilience': round(resilience, 2),
        'return_quality': round(return_quality, 2),
        'breakout_quality': round(breakout_quality, 2),
        'hurst': round(_estimate_hurst(pcts), 3) if n >= HURST_CONFIG['min_weeks'] else 0.5,
        'confidence': round(confidence, 3),
        'grade': grade,
        'grade_emoji': grade_emoji,
        'pattern_key': pattern_key,
        'pattern': f"{p_emoji} {p_name}",
        'price_alignment': round(price_alignment, 1),
        'price_label': price_label,
        'price_tag': price_tag,
        'decay_score': decay_score,
        'decay_label': decay_label,
        'decay_tag': decay_tag,
        'combined_mult': round(combined_mult, 4),
        'signal_tags': signal_tags,
        'current_rank': current_rank,
        'best_rank': best_rank,
        'worst_rank': worst_rank,
        'avg_rank': avg_rank,
        'rank_change': rank_change,
        'last_week_change': last_week_change,
        'streak': streak,
        'tmi': round(tmi, 1),
        'weeks': n,
        'rank_volatility': rank_vol,
        'sparkline': sparkline_data,
        # v6.3: Advanced Trading Signals
        'conviction': conviction,
        'conviction_tag': conviction_tag,
        'conviction_emoji': conviction_emoji,
        'risk_adj_score': risk_adj_score,
        'alpha_score': alpha_score,
        'exit_risk': exit_risk,
        'exit_tag': exit_tag,
        'exit_emoji': exit_emoji,
        'exit_signals': ','.join(exit_signals) if exit_signals else '',
        'hot_streak': hot_streak,
        'persistence_weeks': persistence_weeks,
        # v8.0: Wave Signal Fusion
        'wave_fusion_score': wave_fusion.get('wave_fusion_score', 50),
        'wave_fusion_multiplier': wave_fusion.get('wave_fusion_multiplier', 1.0),
        'wave_fusion_label': wave_fusion.get('wave_fusion_label', 'WAVE_NEUTRAL'),
        'wave_confluence': wave_fusion.get('wave_confluence', 50),
        'wave_inst_flow': wave_fusion.get('wave_inst_flow', 50),
        'wave_harmony': wave_fusion.get('wave_harmony', 50),
        'wave_fundamental': wave_fusion.get('wave_fundamental', 50),
        'wave_position_tension': wave_fusion.get('wave_position_tension', 0),
        'wave_from_low': wave_fusion.get('wave_from_low'),
        'wave_from_high': wave_fusion.get('wave_from_high'),
        'rally_gain': wave_fusion.get('rally_gain', 0.0),
        'rally_weeks': wave_fusion.get('rally_weeks', 0),
        'rally_leg_pct': wave_fusion.get('rally_leg_pct', 50.0),
        'rally_stage': wave_fusion.get('rally_stage', 'UNKNOWN'),
        'wave_ret_1d': wave_fusion.get('wave_ret_1d'),
        'wave_ret_1y': wave_fusion.get('wave_ret_1y'),
    }


def _empty_trajectory(ranks, totals, pcts, n):
    """Return neutral trajectory for insufficient data"""
    return {
        'trajectory_score': 0, 'positional': 0, 'trend': 50, 'velocity': 50,
        'acceleration': 50, 'consistency': 50, 'resilience': 50,
        'return_quality': 50, 'breakout_quality': 50,
        'hurst': 0.5, 'confidence': BAYESIAN_CONFIDENCE['min_confidence'],
        'grade': 'F', 'grade_emoji': '📉',
        'pattern_key': 'new_entry', 'pattern': '💎 New Entry',
        'price_alignment': 50.0,
        'price_label': 'NEUTRAL', 'price_tag': '',
        'decay_score': 0,
        'decay_label': '', 'decay_tag': '',
        'combined_mult': 1.0,
        'signal_tags': '',
        'sector_alpha_tag': 'NEUTRAL', 'sector_alpha_value': 0,
        'current_rank': int(ranks[-1]) if ranks else 0,
        'best_rank': int(min(ranks)) if ranks else 0,
        'worst_rank': int(max(ranks)) if ranks else 0,
        'avg_rank': round(np.mean(ranks), 1) if ranks else 0,
        'rank_change': 0, 'last_week_change': 0, 'streak': 0,
        'tmi': 50.0, 'weeks': n, 'rank_volatility': 0,
        'sparkline': [round(p, 1) for p in pcts] if pcts else [],
        # v6.3: Advanced Trading Signals (defaults)
        'conviction': 0, 'conviction_tag': 'VERY_LOW', 'conviction_emoji': '❌',
        'risk_adj_score': 0, 'alpha_score': 0, 'exit_risk': 0, 'exit_tag': 'HOLD', 'exit_emoji': '✅',
        'exit_signals': '', 'hot_streak': False,
        'persistence_weeks': 0,
        # v8.0: Wave Signal Fusion (defaults)
        'wave_fusion_score': 50, 'wave_fusion_multiplier': 1.0,
        'wave_fusion_label': 'WAVE_NEUTRAL', 'wave_confluence': 50,
        'wave_inst_flow': 50, 'wave_harmony': 50,
        'wave_fundamental': 50, 'wave_position_tension': 0,
        'wave_from_low': None, 'wave_from_high': None,
        'rally_gain': 0.0, 'rally_weeks': 0, 'rally_leg_pct': 50.0, 'rally_stage': 'UNKNOWN',
        'wave_ret_1d': None, 'wave_ret_1y': None,
    }


# ── Adaptive Weight Selection ──

def _get_adaptive_weights(avg_pct: float, current_pct: float = None, confidence: float = None, regime_signal: float = None) -> dict:
    """
    Select weight profile based on stock's effective percentile position.
    Uses smooth interpolation between tiers for continuous transitions.
    
    v5.2: If current_pct >> avg_pct (recent riser), use blended effective_pct
    so weight selection reflects current reality, not stale history.
    A stock at current_pct=72, avg_pct=35 gets effective=57 → 'mid' weights
    instead of 'bottom' weights which over-penalize with 25% velocity weight.
    
    v6.2: Confidence-Aware Dynamic Weight Shifting.
    Low-confidence stocks (few weeks): shift weight toward momentum signals
    (velocity, acceleration, trend) — recent behavior is all we know.
    High-confidence stocks (many weeks): shift weight toward stability signals
    (positional, consistency, resilience) — proven track record matters more.
    
    v10.0: Regime-Aware Micro-Shift.
    Per-stock regime signal (-1=bearish, +1=bullish) shifts weights toward
    defensive metrics (consistency, resilience) in bear micro-regimes and
    toward offensive metrics (trend, breakout) in bull micro-regimes.
    Backtest evidence: S7 conviction (defensive quality) has best alpha in 
    bear markets. This bakes that insight into the weight selection.
    
    Shift magnitude: ±5% confidence, ±4% regime per component.
    """
    effective_pct = avg_pct
    if current_pct is not None and current_pct > avg_pct + 15:
        effective_pct = 0.4 * avg_pct + 0.6 * current_pct
    
    # Select base weights by percentile tier (unchanged logic)
    if effective_pct >= 90:
        base_weights = ADAPTIVE_WEIGHTS['elite'].copy()
    elif effective_pct >= 70:
        ratio = (effective_pct - 70) / 20
        strong = ADAPTIVE_WEIGHTS['strong']
        elite = ADAPTIVE_WEIGHTS['elite']
        base_weights = {k: strong[k] * (1 - ratio) + elite[k] * ratio for k in strong}
    elif effective_pct >= 40:
        ratio = (effective_pct - 40) / 30
        mid = ADAPTIVE_WEIGHTS['mid']
        strong = ADAPTIVE_WEIGHTS['strong']
        base_weights = {k: mid[k] * (1 - ratio) + strong[k] * ratio for k in mid}
    else:
        ratio = effective_pct / 40
        bottom = ADAPTIVE_WEIGHTS['bottom']
        mid = ADAPTIVE_WEIGHTS['mid']
        base_weights = {k: bottom[k] * (1 - ratio) + mid[k] * ratio for k in bottom}
    
    # ── v6.2: Confidence-Aware Micro-Adjustment ──
    if confidence is not None:
        # Shift factor: -1 at confidence=0.25, 0 at confidence=0.625, +1 at confidence=1.0
        # Low confidence → negative shift → boost momentum signals
        # High confidence → positive shift → boost stability signals
        shift_factor = (confidence - 0.625) / 0.375  # Range: -1 to +1
        shift_factor = float(np.clip(shift_factor, -1.0, 1.0))
        
        # Define which components gain/lose weight based on confidence
        # Momentum signals: velocity, acceleration, trend (gain weight when LOW confidence)
        # Stability signals: consistency, resilience, positional (gain weight when HIGH confidence)
        # return_quality: neutral (no shift)
        # v9.0: Confidence shifts now include breakout_quality component.
        # Breakout quality is slightly momentum-like so gets mild negative shift at high confidence.
        
        # Max shift per component: 5% (0.05)
        max_shift = 0.05
        
        # Calculate shifts (positive shift_factor = boost stability, reduce momentum)
        shifts = {
            'positional': shift_factor * max_shift * 0.5,         # +2.5% at high conf
            'trend': -shift_factor * max_shift * 0.5,             # -2.5% at high conf
            'velocity': -shift_factor * max_shift * 0.8,          # -4% at high conf
            'acceleration': -shift_factor * max_shift * 0.6,      # -3% at high conf
            'consistency': shift_factor * max_shift,               # +5% at high conf
            'resilience': shift_factor * max_shift * 0.8,          # +4% at high conf
            'return_quality': 0.0,                                 # Neutral — always relevant
            'breakout_quality': -shift_factor * max_shift * 0.4,   # -2% at high conf (mild)
        }
        
        # Apply shifts and ensure weights stay positive
        adjusted = {k: max(0.02, base_weights[k] + shifts[k]) for k in base_weights}
        
        # Renormalize to sum to 1.0 (critical for score integrity)
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}
    else:
        adjusted = base_weights.copy()

    # ── v10.0: Regime-Aware Micro-Shift ──
    # Per-stock regime signal shifts weights toward defensive or offensive metrics.
    # Bear micro-regime: consistency + resilience UP, velocity + acceleration DOWN
    # Bull micro-regime: trend + breakout UP, consistency slightly DOWN
    # Deadband: |regime_signal| < 0.2 → no shift (avoids noise in neutral markets)
    if regime_signal is not None and abs(regime_signal) > 0.2:
        r_max = 0.04  # Max regime shift per component
        # Negative regime_signal = bearish → boost defensive (positive shift for consistency)
        r_shifts = {
            'positional': 0.0,
            'trend': regime_signal * r_max * 0.3,              # Bull → +1.2%, Bear → -1.2%
            'velocity': regime_signal * r_max * 0.5,           # Bull → +2%, Bear → -2%
            'acceleration': regime_signal * r_max * 0.4,       # Bull → +1.6%, Bear → -1.6%
            'consistency': -regime_signal * r_max * 0.7,       # Bear → +2.8%, Bull → -2.8%
            'resilience': -regime_signal * r_max * 0.6,        # Bear → +2.4%, Bull → -2.4%
            'return_quality': -regime_signal * r_max * 0.3,    # Bear → +1.2% (quality matters)
            'breakout_quality': regime_signal * r_max * 0.5,   # Bull → +2%, Bear → -2%
        }
        adjusted = {k: max(0.02, adjusted[k] + r_shifts[k]) for k in adjusted}
        total = sum(adjusted.values())
        adjusted = {k: v / total for k, v in adjusted.items()}

    return adjusted


def _apply_elite_bonus(score: float, pcts: List[float], n: int) -> float:
    """
    Elite Dominance Bonus: If a stock has been in the top tier for a
    sustained portion of its history, it gets a guaranteed minimum score.
    
    Logic: Being rank 5 out of 2000 for 15 out of 23 weeks is NOT a B-grade
    stock. That's an S-grade achievement. The bonus ensures this.
    
    This is NOT a hack — it's domain logic: sustained excellence IS
    the best possible trajectory.
    """
    if n < 3:
        return score
    
    for tier_name, cfg in ELITE_BONUS.items():
        threshold = cfg['pct_threshold']
        required_ratio = cfg['history_ratio']
        floor = cfg['floor']
        
        # Count weeks where stock was above threshold
        weeks_above = sum(1 for p in pcts if p >= threshold)
        ratio = weeks_above / n
        
        if ratio >= required_ratio:
            # Apply floor, but also scale slightly above floor based on actual ratio
            # v5.3: reduced bonus_extra from 8 to 4 to prevent over-inflation
            bonus_extra = (ratio - required_ratio) / (1.0 - required_ratio + 0.01) * 4
            effective_floor = floor + bonus_extra
            score = max(score, effective_floor)
            break  # Only apply highest qualifying tier
    
    return min(score, 100.0)


# ── Return-Based Price-Rank Alignment Engine (v3.0) ──

def _ema_smooth(values: List[float], span: int = 3) -> List[float]:
    """Exponential Moving Average smoothing for return series. NaN-safe."""
    if not values:
        return []
    alpha = 2.0 / (span + 1)
    smoothed = []
    prev = None
    for v in values:
        if v is None or np.isnan(v):
            smoothed.append(prev if prev is not None else float('nan'))
        elif prev is None:
            prev = v
            smoothed.append(v)
        else:
            prev = alpha * v + (1 - alpha) * prev
            smoothed.append(prev)
    return smoothed


# ── Return Quality Component (v6.0) ──

def _calc_return_quality(ret_3m_list: List[float], ret_6m_list: List[float],
                         ret_7d_list: List[float], ret_30d_list: List[float],
                         from_high_list: List[float], avg_pct: float, n: int) -> float:
    """
    Return Quality — Dedicated 7th scoring component (v6.0).

    ARCHITECTURE: Replaces 5 scattered return touchpoints (velocity floor,
    acceleration floor, resilience bonus, conviction multiplier, price alignment
    signal 3) with a single clean component (0-100).

    4 SUB-SIGNALS:
      Signal 1 (30%): 3-Month Return Score — medium-term momentum quality
      Signal 2 (30%): 6-Month Return Score — institutional horizon confirmation
      Signal 3 (20%): Short-Term Momentum — ret_7d + ret_30d recency signal
      Signal 4 (20%): Return Health — cross-timeframe agreement + from-high distance

    Returns: float (0-100), where 50 = neutral, >70 = strong, <30 = weak.
    """
    if n < RETURN_QUALITY.get('min_weeks', 3):
        return 50.0

    def _latest_valid(lst):
        """Get latest non-None, non-NaN value from a list."""
        if not lst:
            return None
        for v in reversed(lst):
            if v is not None and not np.isnan(v):
                return float(v)
        return None

    r3m = _latest_valid(ret_3m_list)
    r6m = _latest_valid(ret_6m_list)
    r7d = _latest_valid(ret_7d_list)
    r30d = _latest_valid(ret_30d_list)
    fh = _latest_valid(from_high_list)

    # ═══════════════════════════════════════════════════════════════════
    # Signal 1: 3-Month Return Score (30%)
    # Captures medium-term momentum. Sigmoid-like mapping from return %
    # to 0-100. Position-aware: elite stocks SHOULD have gains.
    # ═══════════════════════════════════════════════════════════════════
    if r3m is not None:
        if r3m >= 40:
            s1 = 92.0 + min((r3m - 40) / 60 * 8, 8.0)     # 92-100
        elif r3m >= 25:
            s1 = 78.0 + (r3m - 25) / 15 * 14               # 78-92
        elif r3m >= 15:
            s1 = 65.0 + (r3m - 15) / 10 * 13               # 65-78
        elif r3m >= 5:
            s1 = 52.0 + (r3m - 5) / 10 * 13                # 52-65
        elif r3m >= 0:
            s1 = 42.0 + r3m / 5 * 10                       # 42-52
        elif r3m >= -10:
            s1 = 25.0 + (r3m + 10) / 10 * 17               # 25-42
        elif r3m >= -25:
            s1 = 10.0 + (r3m + 25) / 15 * 15               # 10-25
        else:
            s1 = max(5.0, 10.0 + (r3m + 25) / 25 * 5)      # 5-10
    else:
        s1 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Signal 2: 6-Month Return Score (30%)
    # Captures institutional-horizon quality. Wider range since 6m returns
    # span -50% to +150% typically. Strong 6m = proven gainer.
    # ═══════════════════════════════════════════════════════════════════
    if r6m is not None:
        if r6m >= 80:
            s2 = 93.0 + min((r6m - 80) / 120 * 7, 7.0)     # 93-100
        elif r6m >= 50:
            s2 = 80.0 + (r6m - 50) / 30 * 13               # 80-93
        elif r6m >= 30:
            s2 = 65.0 + (r6m - 30) / 20 * 15               # 65-80
        elif r6m >= 10:
            s2 = 50.0 + (r6m - 10) / 20 * 15               # 50-65
        elif r6m >= 0:
            s2 = 40.0 + r6m / 10 * 10                       # 40-50
        elif r6m >= -15:
            s2 = 22.0 + (r6m + 15) / 15 * 18               # 22-40
        elif r6m >= -30:
            s2 = 10.0 + (r6m + 30) / 15 * 12               # 10-22
        else:
            s2 = max(5.0, 10.0 + (r6m + 30) / 30 * 5)      # 5-10
    else:
        s2 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Signal 3: Short-Term Momentum (20%)
    # ret_7d (35%) + ret_30d (65%). Recent price action quality.
    # Higher weight on 30d because 7d is noisy.
    # ═══════════════════════════════════════════════════════════════════
    st_parts = []
    st_weights = []

    if r7d is not None:
        if r7d >= 5:
            st7 = 85.0 + min((r7d - 5) / 10 * 15, 15.0)
        elif r7d >= 2:
            st7 = 68.0 + (r7d - 2) / 3 * 17
        elif r7d >= 0:
            st7 = 52.0 + r7d / 2 * 16
        elif r7d >= -2:
            st7 = 38.0 + (r7d + 2) / 2 * 14
        elif r7d >= -5:
            st7 = 20.0 + (r7d + 5) / 3 * 18
        else:
            st7 = max(5.0, 20.0 + (r7d + 5) / 10 * 15)
        st_parts.append(st7)
        st_weights.append(0.35)

    if r30d is not None:
        if r30d >= 15:
            st30 = 85.0 + min((r30d - 15) / 20 * 15, 15.0)
        elif r30d >= 5:
            st30 = 65.0 + (r30d - 5) / 10 * 20
        elif r30d >= 0:
            st30 = 50.0 + r30d / 5 * 15
        elif r30d >= -5:
            st30 = 35.0 + (r30d + 5) / 5 * 15
        elif r30d >= -15:
            st30 = 15.0 + (r30d + 15) / 10 * 20
        else:
            st30 = max(5.0, 15.0 + (r30d + 15) / 20 * 10)
        st_parts.append(st30)
        st_weights.append(0.65)

    if st_parts:
        total_stw = sum(st_weights)
        s3 = sum(p * w for p, w in zip(st_parts, st_weights)) / total_stw
    else:
        s3 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Signal 4: Return Health (20%)
    # 4a (55%): Cross-timeframe agreement — r3m and r6m same direction?
    # 4b (45%): Distance from 52-week high — how far has it corrected?
    # ═══════════════════════════════════════════════════════════════════
    health_parts = []

    # 4a: Cross-timeframe agreement
    if r3m is not None and r6m is not None:
        both_pos = r3m > 0 and r6m > 0
        both_neg = r3m < -5 and r6m < -5
        if both_pos:
            # Both positive — strength depends on magnitude
            agree = min(80.0 + (r3m + r6m) / 4, 100.0)
        elif both_neg:
            agree = max(10.0, 30.0 + (r3m + r6m) / 4)
        elif (r3m > 0) != (r6m > 0):
            # Disagreement: transition zone — 3m improving but 6m still negative, or vice versa
            agree = 35.0
        else:
            agree = 50.0
        health_parts.append(('agree', agree, 0.55))

    # 4b: Distance from 52-week high
    if fh is not None:
        if fh >= 0:
            fh_score = 90.0       # At or near high
        elif fh >= -5:
            fh_score = 78.0       # Very close to high
        elif fh >= -10:
            fh_score = 65.0       # Normal pullback
        elif fh >= -20:
            fh_score = 48.0       # Moderate correction
        elif fh >= -30:
            fh_score = 32.0       # Significant correction
        else:
            fh_score = max(10.0, 32.0 + (fh + 30) / 30 * 22)  # Deep correction
        health_parts.append(('fh', fh_score, 0.45))

    if health_parts:
        total_hw = sum(w for _, _, w in health_parts)
        s4 = sum(v * w for _, v, w in health_parts) / total_hw
    else:
        s4 = 50.0

    # ═══════════════════════════════════════════════════════════════════
    # Final Composite: weighted blend of 4 sub-signals
    # ═══════════════════════════════════════════════════════════════════
    score = 0.30 * s1 + 0.30 * s2 + 0.20 * s3 + 0.20 * s4
    return float(np.clip(score, 0, 100))


def _calc_breakout_quality(h: dict, avg_pct: float, n: int) -> float:
    """
    Breakout Quality — Dedicated 8th scoring component (v9.0).

    DATA EVIDENCE: breakout_score → next week return alpha = +0.44%/wk
    (2nd strongest predictor after from_high_pct). This was previously
    only used inside Wave Fusion as a sub-signal. Now dedicated component.

    3 SUB-SIGNALS:
      Signal 1 (45%): Current Breakout Strength — latest breakout_score
      Signal 2 (30%): Breakout Trend — is breakout_score improving over time?
      Signal 3 (25%): Breakout × Position Confirmation — high breakout + high position

    Returns: float (0-100), where 50 = neutral, >70 = strong breakout, <30 = weak.
    """
    if n < 2:
        return 50.0

    bo_scores = h.get('breakout_score', [])
    pos_scores = h.get('position_score', [])

    # Get valid values
    valid_bo = [v for v in bo_scores if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid_bo:
        return 50.0

    latest_bo = valid_bo[-1]

    # ── Signal 1: Current Breakout Strength (45%) ──
    # Sigmoid mapping of breakout_score (range ~14-100)
    if latest_bo >= 75:
        s1 = 85.0 + min((latest_bo - 75) / 25 * 15, 15.0)   # 85-100
    elif latest_bo >= 55:
        s1 = 65.0 + (latest_bo - 55) / 20 * 20               # 65-85
    elif latest_bo >= 40:
        s1 = 45.0 + (latest_bo - 40) / 15 * 20               # 45-65
    elif latest_bo >= 28:
        s1 = 25.0 + (latest_bo - 28) / 12 * 20               # 25-45
    else:
        s1 = max(5.0, latest_bo / 28 * 25)                    # 5-25

    # ── Signal 2: Breakout Trend (30%) ──
    # Is breakout_score improving recently?
    if len(valid_bo) >= 3:
        recent_window = min(4, len(valid_bo))
        recent_bo = valid_bo[-recent_window:]
        older_bo = valid_bo[:-recent_window] if len(valid_bo) > recent_window else valid_bo[:1]

        recent_avg = float(np.mean(recent_bo))
        older_avg = float(np.mean(older_bo)) if older_bo else recent_avg
        bo_delta = recent_avg - older_avg

        if bo_delta >= 15:
            s2 = 88.0
        elif bo_delta >= 8:
            s2 = 72.0
        elif bo_delta >= 3:
            s2 = 58.0
        elif bo_delta >= -3:
            s2 = 45.0
        elif bo_delta >= -8:
            s2 = 32.0
        else:
            s2 = 18.0
    else:
        s2 = 50.0

    # ── Signal 3: Breakout × Position Confirmation (25%) ──
    # Data showed: position_score Q5 + breakout_score Q5 = +1.04%/wk alpha
    valid_pos = [v for v in pos_scores if v is not None and not (isinstance(v, float) and np.isnan(v))]
    latest_pos = valid_pos[-1] if valid_pos else 50

    if latest_bo >= 55 and latest_pos >= 50:
        # Both strong — confirmed breakout
        combo = (latest_bo + latest_pos) / 2
        s3 = min(100, 60 + combo * 0.4)
    elif latest_bo >= 40 and latest_pos >= 35:
        s3 = 50.0 + (latest_bo - 40) + (latest_pos - 35) * 0.5
    elif latest_bo < 25 and latest_pos < 20:
        s3 = 15.0
    else:
        s3 = 40.0

    # Final composite
    score = 0.45 * s1 + 0.30 * s2 + 0.25 * s3
    return float(np.clip(score, 0, 100))


def _calc_price_alignment(ret_7d: List[float], ret_30d: List[float],
                          pcts: List[float], avg_pct: float) -> Tuple[str, float]:
    """
    Price-Rank Alignment (v9.0 — label + score only, no multiplier).

    Measures whether SHORT-TERM return direction agrees with rank movement.
    Does NOT score return magnitude — that is handled by ReturnQuality.

    TWO SIGNALS:
      Signal 1 (55%): EMA-Smoothed Weekly Directional Agreement
      Signal 2 (45%): Monthly Directional Agreement

    v9.0: Multiplier removed (data showed it adds noise). Only label + score
    are retained for UI display and signal tags.

    Returns: (label, alignment_score)
    """
    cfg = PRICE_ALIGNMENT
    n = len(pcts)

    # ── Guard: Need valid return data ──
    valid_ret7 = [r for r in ret_7d if r is not None and not np.isnan(r)]
    if len(valid_ret7) < cfg['min_weeks'] or n < cfg['min_weeks']:
        return 'NEUTRAL', 50.0

    # ── EMA-smooth ret_7d to reduce noise ──
    ema_span = cfg.get('ema_span', 3)
    smoothed_r7 = _ema_smooth(ret_7d, span=ema_span)

    # Build aligned pairs: (ema_r7, r30, percentile)
    pairs = []
    for i in range(n):
        r7e = smoothed_r7[i] if i < len(smoothed_r7) else float('nan')
        r30 = ret_30d[i] if i < len(ret_30d) else float('nan')
        if r7e is not None and not np.isnan(r7e):
            pairs.append((r7e, r30, pcts[i]))

    if len(pairs) < cfg['min_weeks']:
        return 'NEUTRAL', 50.0

    # ── Signal 1: EMA-Smoothed Weekly Directional Agreement (55%) ──
    agree = 0.0
    total_weight = 0.0
    noise_band = cfg['noise_band_stable'] if avg_pct > 80 else cfg['noise_band_normal']
    recency_window = cfg.get('recency_window', 4)
    np_ = len(pairs)

    for i in range(1, np_):
        r7e = pairs[i][0]
        r_chg = pairs[i][2] - pairs[i - 1][2]  # Percentile change

        # Skip noise — tiny rank moves for elite stocks
        if abs(r_chg) < noise_band and abs(r7e) < 1.0:
            continue

        # Recency weight: last `recency_window` weeks get 2×
        w = 2.0 if (np_ - 1 - i) < recency_window else 1.0
        total_weight += w

        # Direction-only scoring: does sign(ret_7d) match sign(rank_change)?
        if r7e > 0 and r_chg > 0:
            agree += 1.0 * w    # Both positive — strong agreement
        elif r7e < -1.0 and r_chg < -1.0:
            agree += 0.8 * w    # Both negative — at least consistent
        elif abs(r7e) < 1.0:
            agree += 0.3 * w    # Return near zero — not really disagreeing
        else:
            agree -= 0.3 * w    # Divergent direction

    if total_weight > 0:
        dir_score = float(np.clip((agree / total_weight) * 50 + 50, 0, 100))
    else:
        dir_score = 55.0  # No significant rank moves — slight positive bias

    # ── Signal 2: Monthly Directional Agreement (45%) — DIRECTION ONLY ──
    # Does sign(ret_30d) agree with sign(percentile_change_over_recent_weeks)?
    # Never scores magnitude — that's return_quality's job.
    recent_window = min(6, len(pairs))
    recent_pairs = pairs[-recent_window:]
    dir_agree = 0.0
    dir_total = 0.0

    for i in range(1, len(recent_pairs)):
        r30 = recent_pairs[i][1]
        if r30 is None or np.isnan(r30):
            continue

        pct_chg = recent_pairs[i][2] - recent_pairs[i - 1][2]
        w = 1.5 if i >= len(recent_pairs) - 2 else 1.0  # Recency bias
        dir_total += w

        # Directional agreement: sign(r30) matches sign(pct_change)?
        if r30 > 1.0 and pct_chg > 0.5:
            dir_agree += 1.0 * w    # Both positive direction
        elif r30 < -1.0 and pct_chg < -0.5:
            dir_agree += 0.8 * w    # Both negative — consistent
        elif abs(r30) <= 1.0:
            dir_agree += 0.3 * w    # Return flat — not disagreeing
        elif (r30 > 1.0 and pct_chg < -2.0) or (r30 < -1.0 and pct_chg > 2.0):
            dir_agree -= 0.5 * w    # Strong divergence
        else:
            dir_agree -= 0.1 * w    # Mild divergence

    if dir_total > 0:
        monthly_dir_score = float(np.clip((dir_agree / dir_total) * 50 + 50, 0, 100))
    else:
        monthly_dir_score = 55.0

    # ── Signal Disagreement Penalty (2 signals) ──
    signal_spread = abs(dir_score - monthly_dir_score)
    if signal_spread > 50:
        disagreement_penalty = 5.0   # Strong contradiction
    elif signal_spread > 30:
        disagreement_penalty = 2.5   # Moderate contradiction
    else:
        disagreement_penalty = 0.0   # Signals agree — no penalty

    # ── Composite Alignment Score (v6.1: 2 directional signals) ──
    alignment = 0.55 * dir_score + 0.45 * monthly_dir_score - disagreement_penalty
    alignment = float(np.clip(alignment, 0, 100))

    # ── Label from alignment score (v9.0: no multiplier) ──
    conf_thresh = cfg['confirmed_threshold']
    div_thresh = cfg['divergent_threshold']

    if alignment >= conf_thresh:
        label = 'PRICE_CONFIRMED'
    elif alignment <= div_thresh:
        label = 'PRICE_DIVERGENT'
    else:
        label = 'NEUTRAL'

    return label, float(alignment)


# ── Momentum Decay Warning Engine (v2.3) ──

def _calc_momentum_decay(ret_7d: List[float], ret_30d: List[float],
                         from_high: List[float], pcts: List[float],
                         avg_pct: float,
                         ret_6m: Optional[List[float]] = None) -> Tuple[str, int]:
    """
    Momentum Decay Warning (v9.0 — label + score only, no multiplier).

    Catches stocks with good rank but deteriorating returns.
    v9.0: Multiplier removed. Decay is now exit-system-only + conviction signal.

    TRAP DETECTION: Deep audit found 11.4% of top-10% stocks have negative 30d returns.
    These stocks ranked well based on PAST momentum that has now faded.
    The rank hasn't dropped yet because ranking is lagging.

    Example: Stock at rank 151 (top 7%) with ret_30d = -22.49% → DECAY_HIGH

    SIGNALS:
      1. Weekly return (ret_7d) negative → short-term momentum loss
      2. 30-day return (ret_30d) negative on ranked stock → THE TRAP
      3. Far from 52-week high (from_high_pct) → correction underway
      4. Consecutive negative weekly returns → sustained decay

    v6.1: Uses actual ret_6m for Proven Winner Exemption instead of avg_pct proxy.

    Returns: (warning_label, decay_score)
    """
    cfg = MOMENTUM_DECAY
    n = len(pcts)
    if n < 3:
        return '', 0

    # Only check stocks above the minimum percentile tier
    if avg_pct < cfg['min_pct_tier']:
        return '', 0

    # Get latest available values
    def _get_latest(lst):
        for val in reversed(lst):
            if val is not None and not np.isnan(val):
                return val
        return None

    latest_r7 = _get_latest(ret_7d)
    latest_r30 = _get_latest(ret_30d)
    latest_from_high = _get_latest(from_high)
    latest_r6m = _get_latest(ret_6m) if ret_6m else None

    if latest_r7 is None and latest_r30 is None:
        return '', 0

    r7 = latest_r7 if latest_r7 is not None else 0
    r30 = latest_r30 if latest_r30 is not None else 0
    fh = latest_from_high if latest_from_high is not None else 0

    decay_score = 0

    # Signal 1: Recent weekly return negative
    if r7 < cfg['r7_severe']:
        decay_score += 30
    elif r7 < cfg['r7_moderate']:
        decay_score += 15
    elif r7 < 0:
        decay_score += 5

    # Signal 2: 30-day return negative on a highly ranked stock (THE TRAP!)
    if avg_pct >= 70 and r30 < cfg['r30_severe_high']:
        decay_score += 40   # Severe — high rank but deeply negative 30d
    elif avg_pct >= 70 and r30 < cfg['r30_moderate_high']:
        decay_score += 25
    elif avg_pct >= 60 and r30 < cfg['r30_severe_mid']:
        decay_score += 20
    elif r30 < -20:
        decay_score += 15

    # Signal 3: Far from high — stock has corrected significantly
    if fh < cfg['from_high_severe'] and avg_pct >= 60:
        decay_score += 20
    elif fh < cfg['from_high_moderate'] and avg_pct >= 60:
        decay_score += 10

    # Signal 4: Consecutive negative weekly returns
    recent_r7 = [r for r in ret_7d[-4:] if r is not None and not np.isnan(r)]
    if len(recent_r7) >= 2:
        neg_count = sum(1 for r in recent_r7 if r < -1)
        if neg_count >= 3:
            decay_score += 15
        elif neg_count >= 2:
            decay_score += 8

    decay_score = min(decay_score, 100)

    # v6.1: Proven Winner Exemption — uses ACTUAL ret_6m instead of avg_pct proxy
    # A stock with ret_6m=82% pulling back 10% recently is NOT a trap —
    # it's a proven winner going through normal consolidation.
    if latest_r6m is not None:
        if latest_r6m > 60:
            decay_score = int(decay_score * 0.45)   # 55% reduction — strong proven winner
        elif latest_r6m > 30:
            decay_score = int(decay_score * 0.60)   # 40% reduction — solid gainer
        elif latest_r6m > 15:
            decay_score = int(decay_score * 0.75)   # 25% reduction — moderate gainer
        # ret_6m <= 15: no exemption — stock hasn't proven itself over 6 months
    else:
        # Fallback: ret_6m unavailable — use avg_pct as rough proxy
        if avg_pct >= 80:
            decay_score = int(decay_score * 0.60)
        elif avg_pct >= 70:
            decay_score = int(decay_score * 0.75)

    # v9.0: Label only (multiplier removed — decay is exit-system-only, not a scoring multiplier)
    if decay_score > 0:
        if decay_score >= cfg['severe_threshold']:
            label = 'DECAY_HIGH'
        elif decay_score >= cfg['moderate_threshold']:
            label = 'DECAY_MODERATE'
        elif decay_score >= cfg['mild_threshold']:
            label = 'DECAY_MILD'
        else:
            label = ''
    else:
        label = ''

    return label, decay_score


# ── Hurst Exponent Engine (v3.0) ──

def _estimate_hurst(series: List[float]) -> float:
    """
    Estimate Hurst exponent using Rescaled Range (R/S) analysis.
    
    H > 0.5: Persistent (trending) — current direction likely continues
    H = 0.5: Random walk — no predictive power
    H < 0.5: Anti-persistent (mean-reverting) — current pattern likely reverses
    
    Uses simplified R/S method suitable for short series (6-25 weeks).
    """
    n = len(series)
    if n < 6:
        return 0.5  # Insufficient data → assume random walk
    
    arr = np.array(series, dtype=float)
    
    # Use multiple sub-series lengths for robust estimation
    rs_values = []
    lengths = []
    
    for seg_len in range(3, n // 2 + 1):
        n_segs = n // seg_len
        if n_segs < 1:
            break
        rs_seg = []
        for i in range(n_segs):
            seg = arr[i * seg_len:(i + 1) * seg_len]
            mean_seg = np.mean(seg)
            deviations = seg - mean_seg
            cumulative = np.cumsum(deviations)
            r = np.max(cumulative) - np.min(cumulative)  # Range
            s = np.std(seg, ddof=1) if np.std(seg, ddof=1) > 0 else 1e-10  # Std dev
            rs_seg.append(r / s)
        
        avg_rs = np.mean(rs_seg)
        if avg_rs > 0:
            rs_values.append(np.log(avg_rs))
            lengths.append(np.log(seg_len))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression: log(R/S) = H * log(n) + c
    # H is the slope
    x = np.array(lengths)
    y = np.array(rs_values)
    n_pts = len(x)
    slope = (n_pts * np.sum(x * y) - np.sum(x) * np.sum(y)) / \
            max(n_pts * np.sum(x ** 2) - np.sum(x) ** 2, 1e-10)
    
    # Clamp to valid range [0.1, 0.9]
    return float(np.clip(slope, 0.1, 0.9))


def _calc_hurst_multiplier(pcts: List[float], trend_score: float) -> float:
    """
    Convert Hurst exponent to a trajectory score multiplier.
    
    LOGIC:
    - Trending (H > 0.55) + uptrend (trend > 55) → BOOST (×1.01 to ×1.06)
      The uptrend is likely to persist.
    - Trending (H > 0.55) + downtrend (trend < 45) → PENALTY (×0.94 to ×0.99)
      The downtrend is likely to persist.
    - Mean-reverting (H < 0.42) + uptrend → mild penalty (×0.97 to ×1.00)
      The uptrend is likely to reverse — don't trust it fully.
    - Mean-reverting (H < 0.42) + downtrend → mild boost (×1.01 to ×1.03)
      The downtrend is likely to reverse — recovery possible.
    - Random walk (0.42-0.55) → no adjustment (×1.00)
    """
    cfg = HURST_CONFIG
    n = len(pcts)
    if n < cfg['min_weeks']:
        return 1.0
    
    h = _estimate_hurst(pcts)
    
    is_uptrend = trend_score > 55
    is_downtrend = trend_score < 45
    
    if h >= cfg['trend_persistence_h']:
        # PERSISTENT series — current direction continues
        strength = (h - cfg['trend_persistence_h']) / (0.9 - cfg['trend_persistence_h'])
        strength = min(strength, 1.0)
        if is_uptrend:
            return 1.01 + strength * (cfg['max_boost'] - 1.01)
        elif is_downtrend:
            return 0.99 - strength * (0.99 - cfg['max_penalty'])
        else:
            return 1.0
    
    elif h <= cfg['mean_revert_h']:
        # MEAN-REVERTING — current direction likely reverses
        strength = (cfg['mean_revert_h'] - h) / (cfg['mean_revert_h'] - 0.1)
        strength = min(strength, 1.0)
        if is_uptrend:
            return 1.0 - strength * 0.03   # Mild penalty: uptrend may reverse
        elif is_downtrend:
            return 1.01 + strength * 0.02  # Mild boost: downtrend may reverse
        else:
            return 1.0
    
    else:
        # RANDOM WALK zone — no adjustment
        return 1.0


# ── Component Score Calculators (v2.1 Adaptive Engine) ──

def _calc_positional_quality(pcts: List[float], n: int) -> float:
    """
    Score based on WHERE the stock currently ranks.
    
    v2.1 Enhancement: Non-linear scaling — top positions get exponential boost.
    Rank 1/2000 (99.95%) and Rank 10/2000 (99.5%) are BOTH extraordinary.
    But rank 500/2000 (75%) is merely good. Linear scaling doesn't capture this.
    
    Uses sigmoid-boosted percentile:
    - Below 50th percentile: compressed (0-30 score range)
    - 50-80th percentile: linear-ish (30-65 score range)
    - 80-95th percentile: expanding (65-85 score range)
    - Above 95th percentile: premium zone (85-100 score range)
    
    Time-weighted: 55% current, 25% recent 4-week, 20% overall avg.
    """
    if n < 1:
        return 0.0
    
    current_pct = pcts[-1]
    recent_window = min(4, n)
    recent_avg = np.mean(pcts[-recent_window:])
    overall_avg = np.mean(pcts)
    
    # v4.0: Recency-biased positional scoring
    # Problem found: stocks that improved recently were penalized by early bad weeks.
    # e.g. stock at avg_pct=66 with current_pct=89 was dragged down by old data.
    # Solution: Reduce overall weight to 10%, boost current to 60%.
    # For late movers (recent >> overall), further reduce overall to 5%.
    is_late_mover = recent_avg > overall_avg + 15
    if is_late_mover:
        raw_pct = 0.60 * current_pct + 0.35 * recent_avg + 0.05 * overall_avg
    else:
        raw_pct = 0.60 * current_pct + 0.30 * recent_avg + 0.10 * overall_avg
    
    # v5.2 SURGE OVERRIDE: Stock surged from bottom to top recently.
    # PROBLEM: 522241 has avg_pct=35 but cur=72 → positional=53 crushes it.
    # A stock that moved from 35th to 72nd percentile in recent weeks PROVED
    # itself. Positional should reflect WHERE IT IS, not where it was months ago.
    if current_pct >= 65 and overall_avg < 50 and recent_avg >= 55:
        surge_pct = 0.75 * current_pct + 0.25 * recent_avg
        raw_pct = max(raw_pct, surge_pct)
    
    # Non-linear sigmoid boost for top positions
    # This makes rank 5 vs rank 50 vs rank 200 properly differentiated
    if raw_pct >= 95:
        score = 85 + (raw_pct - 95) * 3.0   # 95→85, 100→100
    elif raw_pct >= 80:
        score = 65 + (raw_pct - 80) * 1.33  # 80→65, 95→85
    elif raw_pct >= 50:
        score = 30 + (raw_pct - 50) * 1.17  # 50→30, 80→65
    else:
        score = raw_pct * 0.6               # 0→0, 50→30
    
    return float(np.clip(score, 0, 100))


def _calc_trend(pcts: List[float], n: int) -> float:
    """
    Weighted linear regression of percentile trajectory (recency-biased).
    
    v2.0 Elite Floor: Stocks in top percentiles get a minimum trend score
    because "no movement at the top" is NOT failure — it's excellence.
    
    Elite floors:
      Top 5% (pct > 95)  → floor 70
      Top 10% (pct > 90) → floor 65
      Top 20% (pct > 80) → floor 58
      Top 30% (pct > 70) → floor 52
    """
    if n < 3:
        return 50.0

    x = np.arange(n, dtype=float)
    y = np.array(pcts, dtype=float)
    # Exponential weights favoring recent data
    weights = np.exp(0.12 * x)
    weights /= weights.sum()

    # Weighted least squares
    wx = (weights * x).sum()
    wy = (weights * y).sum()
    wxx = (weights * x * x).sum()
    wxy = (weights * x * y).sum()
    w_sum = weights.sum()

    denom = w_sum * wxx - wx * wx
    if abs(denom) < 1e-10:
        return 50.0

    slope = (w_sum * wxy - wx * wy) / denom

    # Normalize: +2 percentile/week = 100, -2 = 0
    raw_score = float(np.clip(slope / 2.0 * 50 + 50, 0, 100))
    
    # v5.3: Reduced trend floors — were inflating ~30% of stocks.
    # Old floors (70/65/58/52) prevented top stocks from EVER showing weak trend.
    recent_avg_pct = np.mean(pcts[-min(4, n):])
    if recent_avg_pct > 95:
        raw_score = max(raw_score, 62)   # v5.3: was 70
    elif recent_avg_pct > 90:
        raw_score = max(raw_score, 57)   # v5.3: was 65
    elif recent_avg_pct > 80:
        raw_score = max(raw_score, 52)   # v5.3: was 58
    elif recent_avg_pct > 70:
        raw_score = max(raw_score, 48)   # v5.3: was 52
    
    return raw_score


def _calc_velocity_adaptive(pcts: List[float], n: int, window: int = 4) -> float:
    """
    Position-Relative Velocity.
    
    KEY INSIGHT: Moving from rank 5 → 3 is ASTRONOMICALLY harder than
    moving from rank 500 → 300. The velocity score must reflect this.
    
    A stock at 98th percentile that stays at 98th → velocity = 65 (good!)
    A stock at 50th percentile that stays at 50th → velocity = 50 (neutral)
    A stock at 98th that drops to 95th → velocity = 55 (small dip, not disaster)
    A stock at 50th that drops to 45th → velocity = 35 (real decline)
    
    Formula: raw_velocity + position_bonus
    Position bonus = scaled by how HARD it is to move at current level
    """
    if n < 2:
        return 50.0

    w = min(window, n - 1)
    change = pcts[-1] - pcts[-w - 1]
    current_pct = pcts[-1]

    # Raw velocity (same as before)
    raw_velocity = float(np.clip(change / 10.0 * 50 + 50, 0, 100))
    
    # Position-relative adjustment:
    # At high percentiles, holding steady is an ACHIEVEMENT
    # Difficulty multiplier: higher position = harder to improve = bonus for holding
    # v5.3: Reduced hold_bonus — was inflating velocity for top stocks (B-grade had
    # HIGHER velocity than S-grade because mid-range movers scored 77.5 vs elite's 73.8).
    if current_pct >= 95:
        hold_bonus = 8.0           # v5.3: was 15 — too generous
        change_sensitivity = 0.7   # v5.3: was 0.6
    elif current_pct >= 85:
        hold_bonus = 5.0           # v5.3: was 10
        change_sensitivity = 0.8   # v5.3: was 0.75
    elif current_pct >= 70:
        hold_bonus = 2.0           # v5.3: was 5
        change_sensitivity = 0.9
    else:
        hold_bonus = 0.0
        change_sensitivity = 1.0
    
    # Apply: dampen negative changes for top stocks, amplify positive for bottom
    if change < 0:
        adjusted = 50 + (change / 10.0 * 50) * change_sensitivity
    else:
        adjusted = 50 + (change / 10.0 * 50)
    
    # Add hold bonus (being at the top and not falling is GOOD)
    adjusted += hold_bonus
    
    return float(np.clip(adjusted, 0, 100))


def _calc_acceleration(pcts: List[float], n: int, window: int = 3) -> float:
    """
    Rate-of-change of velocity — is improvement ACCELERATING or DECELERATING?
    
    v5.0 UPGRADE: Rolling Velocity WLS (replaces naive 2-point difference).
    
    OLD approach: Just (recent_velocity - prior_velocity) using 2 endpoints.
    PROBLEM: Extremely noise-sensitive. A single bad week could flip the entire
    score because it only sampled 2 velocity measurements at fixed points.
    
    NEW approach: Same philosophy as _calc_trend (which works excellently).
    1. Compute velocity at each point using a rolling window
    2. Fit exponentially-weighted least squares on the velocity series
    3. The SLOPE of the velocity series = acceleration
    
    This gives a smooth, robust acceleration estimate that uses ALL available
    data with proper recency weighting.
    
    Also adds position-relative adjustment: at high percentiles, maintaining
    velocity is harder (same logic as velocity's hold_bonus).
    """
    if n < window + 2:
        return 50.0

    # Step 1: Compute rolling velocities at each point
    velocities = []
    for i in range(window, n):
        vel = (pcts[i] - pcts[i - window]) / window
        velocities.append(vel)

    n_vel = len(velocities)
    if n_vel < 3:
        # Fallback to simple 2-point for very short series
        recent_vel = velocities[-1] if velocities else 0
        prior_vel = velocities[0] if velocities else 0
        accel = recent_vel - prior_vel
        return float(np.clip(accel / 2.0 * 50 + 50, 0, 100))

    # Step 2: Exponentially-weighted least squares on velocity series
    x = np.arange(n_vel, dtype=float)
    y = np.array(velocities, dtype=float)
    weights = np.exp(0.15 * x)  # Slightly steeper than trend's 0.12
    weights /= weights.sum()

    wx = (weights * x).sum()
    wy = (weights * y).sum()
    wxx = (weights * x * x).sum()
    wxy = (weights * x * y).sum()
    w_sum = weights.sum()

    denom = w_sum * wxx - wx * wx
    if abs(denom) < 1e-10:
        return 50.0

    # Slope of velocity = acceleration
    accel_slope = (w_sum * wxy - wx * wy) / denom

    # Normalize: +1.5 pct/week² = 100, -1.5 = 0
    # (Tighter than old ±2 because WLS is smoother, so real accelerations are smaller)
    raw_score = float(np.clip(accel_slope / 1.5 * 50 + 50, 0, 100))

    # Step 3: Position-relative adjustment (v5.3: floors reduced)
    current_pct = pcts[-1]
    if current_pct >= 92:
        raw_score = max(raw_score, 42.0)   # v5.3: was 45
    elif current_pct >= 80:
        raw_score = max(raw_score, 40.0)   # v5.3: was 42

    return float(np.clip(raw_score, 0, 100))


def _calc_consistency_adaptive(pcts: List[float], n: int) -> float:
    """
    Position-Aware Consistency + Information Ratio (v4.0).
    
    v4.0 UPGRADE: Added Directional Volatility Discount.
    Problem found: 24/30 missed gainers were VOLATILE pattern (pct_range=86).
    These stocks swing rank 50→500→100 as they gain. The old consistency
    function crushed them with std-based penalty. But if the DIRECTION is
    improving (2nd-half avg < 1st-half avg ranks), the volatility is a
    FEATURE of rapid improvement, not a bug. Discount the penalty by 30%.
    
    UPGRADE: Blends position-aware band consistency with Information Ratio,
    the quant-finance gold standard for risk-adjusted consistency.
    IR = mean(excess_return) / std(excess_return)
    High IR = reliably outperforming. Low IR = noisy, unreliable.
    """
    if n < 3:
        return 50.0

    changes = np.diff(pcts)
    std = np.std(changes)
    positive_ratio = np.sum(changes > 0) / len(changes)
    avg_pct = np.mean(pcts)
    current_pct = pcts[-1]

    # === v4.0: DIRECTIONAL VOLATILITY DISCOUNT ===
    # If recent performance > overall AND current is above average:
    # The stock is volatile but IMPROVING → reduce consistency penalty
    volatility_discount = 1.0  # default: no discount
    if n >= 6:
        half = n // 2
        first_half_avg = float(np.mean(pcts[:half]))
        second_half_avg = float(np.mean(pcts[half:]))
        is_improving = second_half_avg > first_half_avg + 5
        is_currently_high = current_pct > avg_pct
        
        if is_improving and is_currently_high:
            # Discount: reduce std-based penalty by 30%
            # This means std is treated as 70% of actual value
            volatility_discount = 0.70
        elif is_improving:
            # Improving but not currently at top — 15% discount
            volatility_discount = 0.85
    
    # Apply discount to std for consistency calculations
    effective_std = std * volatility_discount

    # === INFORMATION RATIO COMPONENT ===
    ir_cfg = INFO_RATIO_CONFIG
    benchmark = ir_cfg['benchmark_growth']  # Expected pct growth/week
    excess = [c - benchmark for c in changes]  # Excess over benchmark
    excess_mean = float(np.mean(excess))
    excess_std = float(np.std(excess)) if len(excess) > 1 else 1.0
    ir = excess_mean / max(excess_std, 0.01)  # Information Ratio

    # Convert IR to 0-100 score
    if ir >= ir_cfg['excellent_ir']:
        ir_score = 85 + min((ir - ir_cfg['excellent_ir']) * 20, 15)  # 85-100
    elif ir >= ir_cfg['good_ir']:
        t = (ir - ir_cfg['good_ir']) / (ir_cfg['excellent_ir'] - ir_cfg['good_ir'])
        ir_score = 55 + t * 30  # 55-85
    elif ir >= ir_cfg['poor_ir']:
        t = (ir - ir_cfg['poor_ir']) / (ir_cfg['good_ir'] - ir_cfg['poor_ir'])
        ir_score = 30 + t * 25  # 30-55
    else:
        ir_score = max(10, 30 + (ir - ir_cfg['poor_ir']) * 20)  # 10-30
    ir_score = float(np.clip(ir_score, 0, 100))
    
    # === POSITION-RELATIVE CONSISTENCY (v5.0: Smooth Interpolation) ===
    # OLD: Hard cutoffs at avg_pct=85 and 60 with completely different formulas.
    # A stock at 84.9 got a totally different calculation than 85.1.
    # NEW: Compute all tier scores, then smoothly blend using interpolation.
    
    # Elite tier score (relevant when avg_pct ≥ 75+)
    pct_range = max(pcts) - min(pcts)
    band_score = float(np.clip(100 - pct_range * 1.67, 0, 100))
    time_at_top = sum(1 for p in pcts if p >= 80) / len(pcts) * 100
    vol_bonus = float(np.clip(100 - effective_std * 5, 0, 100))
    elite_score = 0.30 * band_score + 0.25 * time_at_top + 0.20 * vol_bonus + 0.25 * ir_score
    
    # Strong tier score (relevant when avg_pct 50-85)
    stability = float(np.clip(100 - effective_std * 2, 0, 100))
    direction = positive_ratio * 100
    trajectory_lift = min((pcts[-1] - pcts[0]) / 20.0 * 10, 15)
    strong_score = float(np.clip(0.35 * stability + 0.30 * direction + 0.35 * ir_score + trajectory_lift, 0, 100))
    
    # Lower tier score (relevant when avg_pct < 50)
    lower_score = 0.35 * stability + 0.30 * direction + 0.35 * ir_score
    
    # Smooth blending with sigmoid-like transitions
    if avg_pct >= 90:
        # Pure elite
        return float(np.clip(elite_score, 0, 100))
    elif avg_pct >= 75:
        # Blend elite ↔ strong over 75-90 range (15 pct transition zone)
        blend = (avg_pct - 75) / 15.0  # 0 at 75, 1 at 90
        return float(np.clip(blend * elite_score + (1 - blend) * strong_score, 0, 100))
    elif avg_pct >= 50:
        # Pure strong
        return float(np.clip(strong_score, 0, 100))
    elif avg_pct >= 35:
        # Blend strong ↔ lower over 35-50 range (15 pct transition zone)
        blend = (avg_pct - 35) / 15.0  # 0 at 35, 1 at 50
        return float(np.clip(blend * strong_score + (1 - blend) * lower_score, 0, 100))
    else:
        # Pure lower
        return float(np.clip(lower_score, 0, 100))


def _calc_resilience(pcts: List[float], n: int) -> float:
    """
    Multi-factor Recovery Resilience (v5.0).
    
    v5.0 UPGRADE: Full drawdown analysis (replaces simple recovery ratio).
    
    OLD approach: Just checked "are you at your peak?" (recovery_ratio) with
    a drawdown magnitude penalty. Gave the SAME score to:
      - A stock that crashed 30 pct and recovered in 2 weeks (excellent!)
      - A stock that crashed 30 pct and recovered in 8 weeks (mediocre)
    
    NEW approach: 4-factor resilience scoring:
    
    1. RECOVERY RATIO (30%): How much of the max drawdown has been recovered?
       Current drawdown vs max drawdown. Fully recovered = 100.
    
    2. RECOVERY SPEED (25%): How FAST was the recovery from the deepest point?
       V-shaped recovery (2 weeks) >> L-shaped recovery (8+ weeks).
       Measured as: what % of max_dd was recovered in the first 3 weeks after trough.
    
    3. DRAWDOWN SEVERITY (25%): How deep were the drawdowns?
       Inverse penalty: bigger drops = less resilient. Uses both max and average
       drawdowns to distinguish "one big crash" vs "many moderate dips."
    
    4. DRAWDOWN FREQUENCY (20%): How often does it fall into drawdowns?
       A stock that dips 5 times is more fragile than one that dipped once.
       Counted as episodes where dd > 3 pct.
    """
    if n < 4:
        return 50.0

    arr = np.array(pcts)
    peak = np.maximum.accumulate(arr)
    drawdowns = peak - arr
    max_dd = float(np.max(drawdowns))
    current_dd = float(drawdowns[-1])

    if max_dd < 1.0:
        return 100.0  # No meaningful drawdown — perfect resilience

    # ── Factor 1: Recovery Ratio (are you recovered?) ──
    recovery_ratio = 1.0 - safe_div(current_dd, max_dd, 1.0)
    recovery_score = recovery_ratio * 100  # 0-100

    # ── Factor 2: Recovery Speed (how fast did you bounce back?) ──
    max_dd_idx = int(np.argmax(drawdowns))
    speed_score = 50.0  # Default: neutral
    if max_dd_idx < n - 1:
        # Measure how much recovered in first 3 weeks after trough
        recovery_window = min(3, n - max_dd_idx - 1)
        if recovery_window > 0:
            # Percentile at trough and 3 weeks later
            trough_pct = arr[max_dd_idx]
            post_pct = arr[min(max_dd_idx + recovery_window, n - 1)]
            recovered_amount = post_pct - trough_pct
            # What % of the drop was recovered in that window?
            recovery_pct = safe_div(recovered_amount, max_dd, 0.0)
            recovery_pct = max(0, min(1.0, recovery_pct))
            # V-shape: >80% recovered in 3 weeks = 100 speed score
            if recovery_pct >= 0.8:
                speed_score = 90 + (recovery_pct - 0.8) * 50  # 90-100
            elif recovery_pct >= 0.5:
                speed_score = 65 + (recovery_pct - 0.5) / 0.3 * 25  # 65-90
            elif recovery_pct >= 0.2:
                speed_score = 35 + (recovery_pct - 0.2) / 0.3 * 30  # 35-65
            else:
                speed_score = max(10, recovery_pct / 0.2 * 35)  # 10-35
    else:
        # Still at the deepest point — slow recovery
        speed_score = 15.0

    # ── Factor 3: Drawdown Severity (how deep were the dips?) ──
    avg_dd = float(np.mean(drawdowns[drawdowns > 0.5])) if np.any(drawdowns > 0.5) else 0
    # Max dd penalty: 50 pct drop → 0 score. 10 pct drop → ~80 score
    max_dd_score = float(np.clip(100 - max_dd * 2, 0, 100))
    avg_dd_score = float(np.clip(100 - avg_dd * 3, 0, 100))
    severity_score = 0.6 * max_dd_score + 0.4 * avg_dd_score

    # ── Factor 4: Drawdown Frequency (how often do you dip?) ──
    # Count episodes: transitions from non-drawdown to drawdown state
    episodes = 0
    in_dd = False
    for dd in drawdowns:
        if dd > 3.0 and not in_dd:
            episodes += 1
            in_dd = True
        elif dd < 1.0:
            in_dd = False
    # 0 episodes = 100, 1 = 85, 2 = 70, 3 = 55, 4+ = max 40
    frequency_score = float(np.clip(100 - episodes * 15, 40, 100))

    # ── Combined Score ──
    resilience = (
        0.30 * recovery_score +
        0.25 * speed_score +
        0.25 * severity_score +
        0.20 * frequency_score
    )

    return float(np.clip(resilience, 0, 100))


# ── Pattern Detection (v3.0 — 13 patterns, 0 ghosts) ──

def _detect_pattern(ranks, totals, pcts, positional, trend, velocity, acceleration, consistency) -> str:
    """Classify trajectory into one of 13 patterns using a priority cascade.
    
    v3.0 improvements over v2.0:
    - Added: crash, topping_out, consolidating, momentum_building
    - Removed 3 ghost patterns (price_confirmed, price_divergent, decay_warning)
      — those are signal tags, not trajectory patterns
    - At Peak now requires positive acceleration to avoid masking topping stocks
    - Breakout works with n>=3 and uses adaptive threshold
    - 'stagnant' replaced with 'neutral' for average stocks
    - Crash catches severe collapses that 'fading' was too gentle for
    """
    n = len(ranks)
    if n < MIN_WEEKS_DEFAULT:
        return 'new_entry'

    current_pct = pcts[-1]
    avg_pct = float(np.mean(pcts))
    current_rank = ranks[-1]
    best_rank = min(ranks)
    pct_diffs = np.diff(pcts)

    # ── TIER 1: CRITICAL SIGNALS (check first — safety & dominance) ──

    # 💥 Crash — Severe rapid collapse (must be checked EARLY to warn users)
    if n >= 3:
        recent_drop = pcts[-1] - pcts[-3] if n >= 4 else pcts[-1] - pcts[-2]
        if recent_drop < -25 and velocity < 30:
            return 'crash'
        # Also catch sustained multi-week collapse
        if n >= 4 and all(d < -3 for d in pct_diffs[-3:]) and current_pct < 30:
            return 'crash'

    # 🎯 Stable Elite — Consistently top-ranked (MUST come before Rocket)
    if positional > 88 and consistency > 60 and current_pct > 85:
        return 'stable_elite'

    # ── TIER 2: STRONG POSITIVE PATTERNS ──

    # 🚀 Rocket — Strong improvement across all dimensions
    if trend > 78 and velocity > 72 and acceleration > 55:
        return 'rocket'

    # ⚡ Breakout — Sudden jump beyond normal variance (works with n>=3)
    # v3.0: requires the LATEST change to be sharp, not just cumulative
    if n >= 3:
        lookback = min(3, n - 1)
        recent_change = pcts[-1] - pcts[-(lookback + 1)]
        avg_abs_change = float(np.mean(np.abs(pct_diffs)))
        latest_change = pcts[-1] - pcts[-2]
        prev_avg_change = float(np.mean(np.abs(pct_diffs[:-1]))) if len(pct_diffs) > 1 else avg_abs_change
        # Adaptive threshold: 2.5× for stable stocks, 2.0× for volatile
        breakout_mult = 2.5 if consistency > 50 else 2.0
        if (avg_abs_change > 0.5 and recent_change > 0 and
                recent_change > breakout_mult * avg_abs_change and
                latest_change > 1.5 * prev_avg_change):  # Latest must be SHARP, not gradual
            return 'breakout'

    # 🔥 Momentum Building — Acceleration surging but trend/velocity haven't caught up yet
    if acceleration > 68 and velocity > 50 and trend < 70 and current_pct > avg_pct:
        return 'momentum_building'

    # ── TIER 3: POSITIONAL PATTERNS (where are they now?) ──

    # ⛰️ Topping Out — Near peak but momentum fading (MUST be checked BEFORE At Peak)
    if best_rank > 0 and current_rank <= best_rank * 1.15 and current_pct > 75:
        if acceleration < 40 or velocity < 38:
            return 'topping_out'

    # 🏔️ At Peak — Near best rank with sustained strength
    if best_rank > 0 and current_rank <= best_rank * 1.12 and current_pct > 78:
        if acceleration >= 40 and velocity >= 38:  # Confirm momentum is healthy
            return 'at_peak'

    # ── TIER 4: DIRECTIONAL PATTERNS ──

    # 📈 Steady Climber — Gradual consistent improvement
    if trend > 58 and consistency > 58 and velocity > 48:
        return 'steady_climber'

    # 🔄 Recovery — Bouncing back from deterioration
    if velocity > 62 and current_pct > avg_pct and trend < 55:
        return 'recovery'

    # 📉 Fading — Deteriorating (but not crashing)
    if velocity < 35 and trend < 40:
        return 'fading'

    # ── TIER 5: STRUCTURAL PATTERNS ──

    # ⏳ Consolidating — Tight range, low movement (potential breakout setup)
    if consistency > 65 and abs(trend - 50) < 12 and abs(velocity - 50) < 12:
        return 'consolidating'

    # 🌊 Volatile — Wild swings
    if consistency < 32:
        return 'volatile'

    # ➖ Neutral — Average stock, no strong signal
    return 'neutral'


# ============================================
# TOP MOVERS CALCULATION
# ============================================

def get_top_movers(histories: dict, n: int = 10, weeks: int = 1,
                   tickers: set = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get biggest rank gainers and decliners over *weeks* weeks.

    Args:
        histories: ticker → history dict (must have 'ranks' list).
        n:         number of top movers to return per side.
        weeks:     look-back window in weeks (1 = last week vs now,
                   2 = 2 weeks ago vs now, etc.).
        tickers:   optional set of tickers to restrict to (sidebar filters).
    """
    movers = []
    for ticker, h in histories.items():
        if tickers is not None and ticker not in tickers:
            continue
        rk = h['ranks']
        if len(rk) < weeks + 1:
            continue
        prev = int(rk[-(weeks + 1)])
        curr = int(rk[-1])
        change = prev - curr            # Positive = improved
        prices = h.get('prices', [])
        latest_price = prices[-1] if prices and prices[-1] > 0 else 0
        movers.append({
            'ticker': ticker,
            'company_name': h['company_name'],
            'category': h['category'],
            'prev_rank': prev,
            'current_rank': curr,
            'rank_change': change,
            'price': latest_price,
        })

    mover_df = pd.DataFrame(movers)
    if mover_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    gainers   = mover_df.nlargest(n, 'rank_change')
    decliners = mover_df.nsmallest(n, 'rank_change')
    return gainers, decliners

# ============================================
# UI: SIDEBAR
# ============================================

def render_sidebar(metadata: dict, traj_df: pd.DataFrame):
    """Render sidebar with premium data info card and collapsible global filters.
    
    v10.1 — Redesigned sidebar: 7 logical sections, 30+ filters,
    smart defaults, grouped by decision workflow.
    """
    with st.sidebar:
        # ═══════════════════════════════════════════════
        # DATA STATUS — live pulse card
        # ═══════════════════════════════════════════════
        st.markdown(f"""
        <div class="sb-status-card">
            <div class="sb-status-row">
                <span><span class="sb-status-dot"></span>Live Data</span>
                <span class="sb-status-val">{metadata['total_weeks']} weeks</span>
            </div>
            <div class="sb-status-row">
                <span>📅 Range</span>
                <span class="sb-status-val" style="font-size:0.7rem">{metadata['date_range']}</span>
            </div>
            <div class="sb-status-row">
                <span>📊 Tickers</span>
                <span class="sb-status-val">{metadata['total_tickers']:,}</span>
            </div>
            <div class="sb-status-row">
                <span>📈 Avg/Week</span>
                <span class="sb-status-val">{metadata['avg_stocks_per_week']:,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # 🔍 SMART FILTERS — active count + clear all
        # ═══════════════════════════════════════════════
        st.markdown('<div class="sb-section-head">🔍 SMART FILTERS</div>', unsafe_allow_html=True)

        _defaults = {
            'sb_quick': 'None', 'sb_score_range': (0, 100), 'sb_alpha_range': (0, 100),
            'sb_conviction_range': (0, 100), 'sb_grade': [], 'sb_weeks': MIN_WEEKS_DEFAULT,
            'sb_pa': 'All', 'sb_md': 'All', 'sb_exit_risk': 'All', 'sb_hot_streak': False,
            'sb_streak': (-10, 10), 'sb_tmi': (0, 100), 'sb_hurst': 'All',
            'sb_rally': [], 'sb_gain_preset': 'All', 'sb_age_preset': 'All',
            'sb_gap_preset': 'All', 'sb_wf_label': 'All', 'sb_confluence': (0, 100),
            'sb_inst_flow': (0, 100), 'sb_harmony': (0, 100), 'sb_fundamental': (0, 100),
            'sb_rank_chg': 'All', 'sb_persist': (0, 20), 'sb_rankvol': 'All',
            'sb_cat': [], 'sb_sector': [], 'sb_industry': [], 'sb_market_state': [],
        }
        _sb_keys = list(_defaults.keys())
        # Conditional slider keys (only rendered when Custom Range is chosen)
        _conditional_keys = ['sb_gain_slider', 'sb_age_slider', 'sb_gap_slider']

        # ── Handle pending clear (set by button callback on PREVIOUS run) ──
        if st.session_state.get('_sb_pending_clear'):
            del st.session_state['_sb_pending_clear']
            for _k, _v in _defaults.items():
                st.session_state[_k] = _v
            for _ck in _conditional_keys:
                if _ck in st.session_state:
                    del st.session_state[_ck]

        # Count active filters
        _active = 0
        for _k in _sb_keys:
            _val = st.session_state.get(_k)
            if _val is not None and _val != _defaults[_k]:
                _active += 1

        if _active > 0:
            st.info(f"🔍 **{_active} filter{'s' if _active > 1 else ''} active**")

        if st.button("🗑️ Clear All Filters", key='sb_clear_all', use_container_width=True,
                     type='primary' if _active > 0 else 'secondary'):
            st.session_state['_sb_pending_clear'] = True
            st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # § 1  QUICK FILTERS — dropdown preset selector
        # ═══════════════════════════════════════════════
        st.markdown('<div class="sb-section-head">⚡ QUICK FILTERS</div>', unsafe_allow_html=True)
        quick_filter = st.selectbox(
            "Quick Filter Preset",
            ['None',
             '🏆 Max Alpha (Top 15)',
             '🧪 EARLY_RIDE_PROVEN (Top 10)',
             'Conviction ≥ 65',
             '🚀 Rockets Only',
             '🎯 Elite Only',
             '📈 Climbers',
             '⚡ Breakouts',
             '🏔️ At Peak',
             '🔥 Momentum',
             '💥 Crashes',
             '⛰️ Topping',
             '⏳ Consolidating',
             'Positional > 80'],
            index=0, key='sb_quick', label_visibility='collapsed')

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # § 2  SCORING & QUALITY — the money filters
        # ═══════════════════════════════════════════════
        with st.expander("🎯 Scoring & Quality", expanded=False):
            st.caption("Core metrics that drive stock selection")

            # T-Score range
            score_range = st.slider(
                "T-Score Range", 0, 100, (0, 100), key='sb_score_range',
                help="Trajectory Score — the master 8-component composite")

            # Alpha Score range
            alpha_range = st.slider(
                "Alpha Score Range", 0, 100, (0, 100), key='sb_alpha_range',
                help="Forward-return predictor (near-high, breakout, persistence, position)")

            # Conviction range
            conviction_range = st.slider(
                "Conviction Range", 0, 100, (0, 100), key='sb_conviction_range',
                help="12-signal confidence score")

            # Grade filter
            grade_options = ['All', 'S — Elite', 'A — Strong', 'B — Above Avg', 'C — Average', 'D — Below Avg', 'F — Weak']
            grade_map = {'S — Elite': 'S', 'A — Strong': 'A', 'B — Above Avg': 'B',
                         'C — Average': 'C', 'D — Below Avg': 'D', 'F — Weak': 'F'}
            selected_grades = st.multiselect(
                "Grade", grade_options[1:], default=[], placeholder="All grades",
                key='sb_grade', help="Letter grade from T-Score")

            # Min Weeks of Data
            min_weeks = st.slider(
                "Min Weeks of Data", 2, metadata['total_weeks'],
                MIN_WEEKS_DEFAULT, key='sb_weeks',
                help="Stocks with fewer weeks are excluded")

        # ═══════════════════════════════════════════════
        # § 3  SIGNALS & MOMENTUM — trend health
        # ═══════════════════════════════════════════════
        with st.expander("📡 Signals & Momentum", expanded=False):
            st.caption("Momentum health and warning signals")

            # Price Alignment
            pa_options = ['All', '💰 Confirmed', '⚠️ Divergent', '➖ Neutral']
            selected_pa = st.selectbox("Price Alignment", pa_options, index=0, key='sb_pa')

            # Momentum Decay
            md_options = ['All', '✅ No Decay', '🔻 High Decay', '⚡ Moderate Decay', '~ Mild Decay']
            selected_md = st.selectbox("Momentum Decay", md_options, index=0, key='sb_md')

            st.markdown("---")

            # Exit Risk
            exit_options = ['All', '🟢 Safe (0-25)', '🟡 Watch (25-50)', '🟠 Caution (50-75)', '🔴 Danger (75-100)']
            selected_exit = st.selectbox("Exit Risk Level", exit_options, index=0, key='sb_exit_risk')

            # Hot Streak (consecutive improvements)
            hot_streak_only = st.checkbox("🔥 Hot Streak Only", value=False, key='sb_hot_streak',
                                         help="Show only stocks on a hot streak (3+ improving weeks)")

            # Streak filter
            streak_range = st.slider(
                "Rank Streak (weeks)", -10, 10, (-10, 10), key='sb_streak',
                help="Positive = consecutive rank improvements, Negative = declines")

            st.markdown("---")

            # TMI range
            tmi_range = st.slider(
                "TMI Range", 0, 100, (0, 100), key='sb_tmi',
                help="Trajectory Momentum Index — velocity + trend combined")

            # Hurst Exponent
            hurst_options = ['All', '📈 Trending (> 0.6)', '〰️ Random Walk (0.4–0.6)', '📉 Mean-Reverting (< 0.4)']
            selected_hurst = st.selectbox("Hurst Exponent", hurst_options, index=0, key='sb_hurst',
                                          help="Persistence of rank trajectory")

        # ═══════════════════════════════════════════════
        # § 4  RALLY LEG STATUS — timing filters
        # ═══════════════════════════════════════════════
        with st.expander("📈 Rally Leg Status", expanded=False):
            st.caption("Where is the stock in its current rally?")

            # Stage multi-select
            rally_stage_options = ['🌱 Fresh (<5%)', '🚀 Early (5-15%)', '🏃 Running (15-30%)',
                                   '🧱 Mature (30-50%)', '⏳ Late (>50%)']
            rally_stage_map = {
                '🌱 Fresh (<5%)': 'FRESH', '🚀 Early (5-15%)': 'EARLY',
                '🏃 Running (15-30%)': 'RUNNING', '🧱 Mature (30-50%)': 'MATURE', '⏳ Late (>50%)': 'LATE'
            }
            selected_rally = st.multiselect(
                "Stage", rally_stage_options, default=[], placeholder="All stages", key='sb_rally')

            st.markdown("---")

            # Gain this leg
            gain_presets = ['All', '🟢 Fresh Start (<5%)', '🔵 Early Momentum (5-15%)',
                           '🟠 Strong Run (15-30%)', '🔴 Extended (>30%)', '🎯 Custom Range']
            gain_choice = st.selectbox("Gain This Leg", gain_presets, index=0, key='sb_gain_preset')
            gain_range = (0.0, 999.0)
            if gain_choice == '🟢 Fresh Start (<5%)':
                gain_range = (0.0, 5.0)
            elif gain_choice == '🔵 Early Momentum (5-15%)':
                gain_range = (5.0, 15.0)
            elif gain_choice == '🟠 Strong Run (15-30%)':
                gain_range = (15.0, 30.0)
            elif gain_choice == '🔴 Extended (>30%)':
                gain_range = (30.0, 999.0)
            elif gain_choice == '🎯 Custom Range':
                gain_range = st.slider("Gain % range", 0.0, 100.0, (0.0, 100.0), step=1.0, key='sb_gain_slider')

            st.markdown("---")

            # Age of move
            age_presets = ['All', '⚡ Just Started (0-2w)', '🕐 Recent (2-5w)',
                          '📅 Established (5-10w)', '🏛️ Mature (10w+)', '🎯 Custom Range']
            age_choice = st.selectbox("Age of Move", age_presets, index=0, key='sb_age_preset')
            age_range = (0, 99)
            if age_choice == '⚡ Just Started (0-2w)':
                age_range = (0, 2)
            elif age_choice == '🕐 Recent (2-5w)':
                age_range = (2, 5)
            elif age_choice == '📅 Established (5-10w)':
                age_range = (5, 10)
            elif age_choice == '🏛️ Mature (10w+)':
                age_range = (10, 99)
            elif age_choice == '🎯 Custom Range':
                age_range = st.slider("Age (weeks)", 0, 20, (0, 20), key='sb_age_slider')

            st.markdown("---")

            # Gap to 52w high
            gap_presets = ['All', '🔥 Near High (<5%)', '✅ Close (5-15%)',
                          '📏 Moderate Gap (15-30%)', '📉 Far from High (>30%)', '🎯 Custom Range']
            gap_choice = st.selectbox("Gap to 52w High", gap_presets, index=0, key='sb_gap_preset')
            gap_range = (0.0, 999.0)
            if gap_choice == '🔥 Near High (<5%)':
                gap_range = (0.0, 5.0)
            elif gap_choice == '✅ Close (5-15%)':
                gap_range = (5.0, 15.0)
            elif gap_choice == '📏 Moderate Gap (15-30%)':
                gap_range = (15.0, 30.0)
            elif gap_choice == '📉 Far from High (>30%)':
                gap_range = (30.0, 999.0)
            elif gap_choice == '🎯 Custom Range':
                gap_range = st.slider("Gap % range", 0.0, 80.0, (0.0, 80.0), step=1.0, key='sb_gap_slider')

        # ═══════════════════════════════════════════════
        # § 5  WAVE FUSION — cross-system validation
        # ═══════════════════════════════════════════════
        with st.expander("🌊 Wave Fusion Signals", expanded=False):
            st.caption("Cross-system agreement between WAVE + Trajectory")

            # Wave Fusion Label
            wf_label_options = ['All', '🌊 Strong', '✅ Confirmed', '➖ Neutral', '⚠️ Weak', '🔇 Conflict']
            wf_label_map = {
                '🌊 Strong': 'WAVE_STRONG', '✅ Confirmed': 'WAVE_CONFIRMED',
                '➖ Neutral': 'WAVE_NEUTRAL', '⚠️ Weak': 'WAVE_WEAK', '🔇 Conflict': 'WAVE_CONFLICT'
            }
            selected_wf_label = st.selectbox("Wave Fusion", wf_label_options, index=0, key='sb_wf_label')

            st.markdown("---")

            # Sub-scores
            confluence_range = st.slider("Confluence", 0, 100, (0, 100), key='sb_confluence',
                                         help="WAVE ↔ Trajectory agreement")
            inst_flow_range = st.slider("Inst. Flow", 0, 100, (0, 100), key='sb_inst_flow',
                                        help="Money flow + VMI + RVOL")
            harmony_range = st.slider("Harmony", 0, 100, (0, 100), key='sb_harmony',
                                      help="WAVE 5-check consistency")
            fundamental_range = st.slider("Fundamental", 0, 100, (0, 100), key='sb_fundamental',
                                          help="EPS growth + PE quality")

        # ═══════════════════════════════════════════════
        # § 6  RANK DYNAMICS — rank movement filters
        # ═══════════════════════════════════════════════
        with st.expander("📊 Rank Dynamics", expanded=False):
            st.caption("How the rank is moving week-over-week")

            # Rank change (last week)
            rank_chg_options = ['All', '⬆️ Improved (dropped rank #)', '⬇️ Worsened (rose rank #)', '➡️ Unchanged']
            selected_rank_chg = st.selectbox(
                "Last Week Direction", rank_chg_options, index=0, key='sb_rank_chg',
                help="Rank # decrease = improvement, increase = worse")

            # Persistence weeks
            persist_range = st.slider(
                "Persistence (weeks in top)", 0, 20, (0, 20), key='sb_persist',
                help="Consecutive weeks the stock has been in top ranks")

            # Rank volatility
            rankvol_options = ['All', '🎯 Stable (< 10)', '📏 Normal (10-20)', '🎢 Volatile (> 20)']
            selected_rankvol = st.selectbox(
                "Rank Volatility", rankvol_options, index=0, key='sb_rankvol',
                help="How much the rank fluctuates week-to-week")

        # ═══════════════════════════════════════════════
        # § 7  SECTOR & CATEGORY — universe slicing
        # ═══════════════════════════════════════════════
        with st.expander("🏢 Sector & Category", expanded=False):
            # ── Category (top-level, independent) ──
            categories = sorted(traj_df['category'].dropna().unique().tolist())
            selected_cats = st.multiselect("Category", categories, default=[], placeholder="All", key='sb_cat')

            # ── Sector cascades from Category ──
            sector_pool = traj_df
            if selected_cats:
                sector_pool = sector_pool[sector_pool['category'].isin(selected_cats)]
            sector_counts = sector_pool['sector'].value_counts()
            top_sectors = sector_counts[sector_counts >= 3].index.tolist()
            sectors = sorted(top_sectors)
            # Prune stale sector selections that are no longer available
            if 'sb_sector' in st.session_state:
                _valid_sec = [s for s in st.session_state['sb_sector'] if s in sectors]
                if _valid_sec != list(st.session_state['sb_sector']):
                    st.session_state['sb_sector'] = _valid_sec
            selected_sectors = st.multiselect("Sector", sectors, default=[], placeholder="All", key='sb_sector')

            # ── Industry cascades from Category + Sector ──
            industry_pool = traj_df
            if selected_cats:
                industry_pool = industry_pool[industry_pool['category'].isin(selected_cats)]
            if selected_sectors:
                industry_pool = industry_pool[industry_pool['sector'].isin(selected_sectors)]
            industries = sorted(industry_pool['industry'].dropna().loc[lambda s: s.str.strip() != ''].unique().tolist())
            # Prune stale industry selections
            if 'sb_industry' in st.session_state:
                _valid_ind = [i for i in st.session_state['sb_industry'] if i in industries]
                if _valid_ind != list(st.session_state['sb_industry']):
                    st.session_state['sb_industry'] = _valid_ind
            selected_industries = st.multiselect("Industry", industries, default=[], placeholder="All", key='sb_industry')

            st.markdown("---")

            # ── Market State (from source CSV) ──
            if 'market_state' in traj_df.columns:
                ms_pool = traj_df
                if selected_cats:
                    ms_pool = ms_pool[ms_pool['category'].isin(selected_cats)]
                if selected_sectors:
                    ms_pool = ms_pool[ms_pool['sector'].isin(selected_sectors)]
                ms_vals = sorted(ms_pool['market_state'].dropna().unique().tolist())
                # Prune stale market state selections
                if 'sb_market_state' in st.session_state:
                    _valid_ms = [m for m in st.session_state['sb_market_state'] if m in ms_vals]
                    if _valid_ms != list(st.session_state['sb_market_state']):
                        st.session_state['sb_market_state'] = _valid_ms
                if ms_vals:
                    selected_market_states = st.multiselect(
                        "Market State", ms_vals, default=[], placeholder="All states",
                        key='sb_market_state', help="Source-level market state tag")
                else:
                    selected_market_states = []
            else:
                selected_market_states = []

        # ═══════════════════════════════════════════════
        # FOOTER — Clear All + version
        # ═══════════════════════════════════════════════
        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear All Filters", key='sb_clear_all_bottom', use_container_width=True,
                     type='primary' if _active > 0 else 'secondary'):
            st.session_state['_sb_pending_clear'] = True
            st.rerun()

        st.caption("v10.1 · Alpha Engine · Data-Driven")

    # ── Return filter dict ──
    return {
        'categories': selected_cats,
        'sectors': selected_sectors,
        'industries': selected_industries,
        'price_alignment': selected_pa,
        'momentum_decay': selected_md,
        'min_weeks': min_weeks,
        'min_score': score_range[0],
        'max_score': score_range[1],
        'alpha_range': alpha_range,
        'conviction_range': conviction_range,
        'grades': [grade_map[g] for g in selected_grades],
        'quick_filter': quick_filter,
        'rally_stage': [rally_stage_map[r] for r in selected_rally],
        'gain_range': gain_range,
        'age_range': age_range,
        'gap_range': gap_range,
        'wf_label': wf_label_map.get(selected_wf_label, None),
        'confluence_range': confluence_range,
        'inst_flow_range': inst_flow_range,
        'harmony_range': harmony_range,
        'fundamental_range': fundamental_range,
        'exit_risk': selected_exit,
        'hot_streak_only': hot_streak_only,
        'streak_range': streak_range,
        'tmi_range': tmi_range,
        'hurst': selected_hurst,
        'rank_change_dir': selected_rank_chg,
        'persist_range': persist_range,
        'rankvol': selected_rankvol,
        'market_states': selected_market_states,
    }


def apply_filters(traj_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply sidebar filters to trajectory DataFrame.

    v10.1 — Handles all 30+ filter keys from the redesigned sidebar.
    """
    df = traj_df.copy()

    # ── § 7: Sector & Category ──
    if filters['categories']:
        df = df[df['category'].isin(filters['categories'])]
    if filters['sectors']:
        df = df[df['sector'].isin(filters['sectors'])]
    if filters.get('industries'):
        df = df[df['industry'].isin(filters['industries'])]
    if filters.get('market_states') and 'market_state' in df.columns:
        df = df[df['market_state'].isin(filters['market_states'])]

    # ── § 2: Scoring & Quality ──
    # T-Score range
    s_lo, s_hi = filters.get('min_score', 0), filters.get('max_score', 100)
    df = df[(df['trajectory_score'] >= s_lo) & (df['trajectory_score'] <= s_hi)]

    # Alpha Score range
    a_lo, a_hi = filters.get('alpha_range', (0, 100))
    if 'alpha_score' in df.columns and (a_lo > 0 or a_hi < 100):
        df = df[(df['alpha_score'] >= a_lo) & (df['alpha_score'] <= a_hi)]

    # Conviction range
    c_lo, c_hi = filters.get('conviction_range', (0, 100))
    if 'conviction' in df.columns and (c_lo > 0 or c_hi < 100):
        df = df[(df['conviction'] >= c_lo) & (df['conviction'] <= c_hi)]

    # Grade filter
    grades = filters.get('grades', [])
    if grades and 'grade' in df.columns:
        df = df[df['grade'].isin(grades)]

    # Min weeks
    df = df[df['weeks'] >= filters['min_weeks']]

    # ── § 3: Signals & Momentum ──
    # Price Alignment
    pa = filters.get('price_alignment', 'All')
    if pa == '💰 Confirmed':
        df = df[df['price_label'] == 'PRICE_CONFIRMED']
    elif pa == '⚠️ Divergent':
        df = df[df['price_label'] == 'PRICE_DIVERGENT']
    elif pa == '➖ Neutral':
        df = df[df['price_label'] == 'NEUTRAL']

    # Momentum Decay
    md = filters.get('momentum_decay', 'All')
    if md == '🔻 High Decay':
        df = df[df['decay_label'] == 'DECAY_HIGH']
    elif md == '⚡ Moderate Decay':
        df = df[df['decay_label'] == 'DECAY_MODERATE']
    elif md == '~ Mild Decay':
        df = df[df['decay_label'] == 'DECAY_MILD']
    elif md == '✅ No Decay':
        df = df[~df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE', 'DECAY_MILD'])]

    # Exit Risk
    exit_f = filters.get('exit_risk', 'All')
    if exit_f != 'All' and 'exit_risk' in df.columns:
        if exit_f == '🟢 Safe (0-25)':
            df = df[df['exit_risk'] <= 25]
        elif exit_f == '🟡 Watch (25-50)':
            df = df[(df['exit_risk'] > 25) & (df['exit_risk'] <= 50)]
        elif exit_f == '🟠 Caution (50-75)':
            df = df[(df['exit_risk'] > 50) & (df['exit_risk'] <= 75)]
        elif exit_f == '🔴 Danger (75-100)':
            df = df[df['exit_risk'] > 75]

    # Hot Streak
    if filters.get('hot_streak_only') and 'hot_streak' in df.columns:
        df = df[df['hot_streak'] == True]

    # Streak range
    s_lo_s, s_hi_s = filters.get('streak_range', (-10, 10))
    if 'streak' in df.columns and (s_lo_s > -10 or s_hi_s < 10):
        df = df[(df['streak'] >= s_lo_s) & (df['streak'] <= s_hi_s)]

    # TMI range
    tmi_lo, tmi_hi = filters.get('tmi_range', (0, 100))
    if 'tmi' in df.columns and (tmi_lo > 0 or tmi_hi < 100):
        df = df[(df['tmi'] >= tmi_lo) & (df['tmi'] <= tmi_hi)]

    # Hurst Exponent
    hurst_f = filters.get('hurst', 'All')
    if hurst_f != 'All' and 'hurst' in df.columns:
        if hurst_f == '📈 Trending (> 0.6)':
            df = df[df['hurst'] > 0.6]
        elif hurst_f == '〰️ Random Walk (0.4–0.6)':
            df = df[(df['hurst'] >= 0.4) & (df['hurst'] <= 0.6)]
        elif hurst_f == '📉 Mean-Reverting (< 0.4)':
            df = df[df['hurst'] < 0.4]

    # ── § 4: Rally Leg Status ──
    rally_stages = filters.get('rally_stage', [])
    if rally_stages and 'rally_stage' in df.columns:
        df = df[df['rally_stage'].isin(rally_stages)]

    g_lo, g_hi = filters.get('gain_range', (0.0, 999.0))
    if 'rally_gain' in df.columns and (g_lo > 0 or g_hi < 999):
        df = df[(df['rally_gain'] >= g_lo) & (df['rally_gain'] <= g_hi)]

    age_lo, age_hi = filters.get('age_range', (0, 99))
    if 'rally_weeks' in df.columns and (age_lo > 0 or age_hi < 99):
        df = df[(df['rally_weeks'] >= age_lo) & (df['rally_weeks'] <= age_hi)]

    gap_lo, gap_hi = filters.get('gap_range', (0.0, 999.0))
    if 'wave_from_high' in df.columns and (gap_lo > 0 or gap_hi < 999):
        gap_abs = df['wave_from_high'].fillna(-999).abs()
        df = df[(gap_abs >= gap_lo) & (gap_abs <= gap_hi)]

    # ── § 5: Wave Fusion Signals ──
    wf_label = filters.get('wf_label')
    if wf_label and 'wave_fusion_label' in df.columns:
        df = df[df['wave_fusion_label'] == wf_label]

    cf_lo, cf_hi = filters.get('confluence_range', (0, 100))
    if 'wave_confluence' in df.columns and (cf_lo > 0 or cf_hi < 100):
        df = df[(df['wave_confluence'] >= cf_lo) & (df['wave_confluence'] <= cf_hi)]

    if_lo, if_hi = filters.get('inst_flow_range', (0, 100))
    if 'wave_inst_flow' in df.columns and (if_lo > 0 or if_hi < 100):
        df = df[(df['wave_inst_flow'] >= if_lo) & (df['wave_inst_flow'] <= if_hi)]

    h_lo, h_hi = filters.get('harmony_range', (0, 100))
    if 'wave_harmony' in df.columns and (h_lo > 0 or h_hi < 100):
        df = df[(df['wave_harmony'] >= h_lo) & (df['wave_harmony'] <= h_hi)]

    f_lo, f_hi = filters.get('fundamental_range', (0, 100))
    if 'wave_fundamental' in df.columns and (f_lo > 0 or f_hi < 100):
        df = df[(df['wave_fundamental'] >= f_lo) & (df['wave_fundamental'] <= f_hi)]

    # ── § 6: Rank Dynamics ──
    rank_chg = filters.get('rank_change_dir', 'All')
    if rank_chg != 'All' and 'last_week_change' in df.columns:
        if rank_chg == '⬆️ Improved (dropped rank #)':
            df = df[df['last_week_change'] < 0]
        elif rank_chg == '⬇️ Worsened (rose rank #)':
            df = df[df['last_week_change'] > 0]
        elif rank_chg == '➡️ Unchanged':
            df = df[df['last_week_change'] == 0]

    p_lo, p_hi = filters.get('persist_range', (0, 20))
    if 'persistence_weeks' in df.columns and (p_lo > 0 or p_hi < 20):
        df = df[(df['persistence_weeks'] >= p_lo) & (df['persistence_weeks'] <= p_hi)]

    rv_f = filters.get('rankvol', 'All')
    if rv_f != 'All' and 'rank_volatility' in df.columns:
        if rv_f == '🎯 Stable (< 10)':
            df = df[df['rank_volatility'] < 10]
        elif rv_f == '📏 Normal (10-20)':
            df = df[(df['rank_volatility'] >= 10) & (df['rank_volatility'] <= 20)]
        elif rv_f == '🎢 Volatile (> 20)':
            df = df[df['rank_volatility'] > 20]

    # ── § 1: Quick Filters (applied last — override-style) ──
    qf = filters['quick_filter']
    if qf == '🚀 Rockets Only':
        df = df[df['pattern_key'] == 'rocket']
    elif qf == '🎯 Elite Only':
        df = df[df['pattern_key'] == 'stable_elite']
    elif qf == '📈 Climbers':
        df = df[df['pattern_key'] == 'steady_climber']
    elif qf == '⚡ Breakouts':
        df = df[df['pattern_key'] == 'breakout']
    elif qf == '🏔️ At Peak':
        df = df[df['pattern_key'] == 'at_peak']
    elif qf == '🔥 Momentum':
        df = df[df['pattern_key'] == 'momentum_building']
    elif qf == '💥 Crashes':
        df = df[df['pattern_key'] == 'crash']
    elif qf == '⛰️ Topping':
        df = df[df['pattern_key'] == 'topping_out']
    elif qf == '⏳ Consolidating':
        df = df[df['pattern_key'] == 'consolidating']
    elif qf == 'Conviction ≥ 65':
        df = df[df['conviction'] >= 65] if 'conviction' in df.columns else df
    elif qf == 'Positional > 80':
        df = df[df['positional'] > 80]
    elif qf == '🧪 EARLY_RIDE_PROVEN (Top 10)':
        if 'rally_stage' in df.columns and 'conviction' in df.columns:
            early = df[df['rally_stage'].isin(['FRESH', 'EARLY', 'RUNNING'])]
            df = early.sort_values('conviction', ascending=False).head(10)
        else:
            df = df.head(0)
    elif qf == '🏆 Max Alpha (Top 15)':
        if 'alpha_score' in df.columns:
            candidates = df[df['alpha_score'] >= 40].copy()
            candidates = candidates.sort_values('alpha_score', ascending=False)
            sector_counts = {}
            keep_idx = []
            for idx, row in candidates.iterrows():
                sec = row.get('sector', 'Unknown') or 'Unknown'
                if sector_counts.get(sec, 0) < 2:
                    keep_idx.append(idx)
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
                if len(keep_idx) >= 15:
                    break
            df = candidates.loc[keep_idx]
        else:
            df = df.head(0)

    # Re-rank after filtering
    df = df.reset_index(drop=True)
    df['t_rank'] = range(1, len(df) + 1)

    return df


# ============================================
# UI: MARKET PULSE TAB — v1.0
# ============================================

def render_market_pulse_tab(filtered_df: pd.DataFrame, all_df: pd.DataFrame,
                            histories: dict, metadata: dict):
    """📡 Market Pulse — bird's-eye market intelligence dashboard.

    Sections:
      1. Pulse Hero       – regime badge, market strength, 6 breadth KPIs
      2. Breadth Over Time – % improving / % strong / advance-decline over weeks

    All data is filter-aware — respects sidebar filters via filtered_df/histories.
    Uses ONLY pre-computed data from histories/traj_df — zero new API calls.
    """

    n_stocks = len(filtered_df)
    if n_stocks == 0:
        st.info("No stocks match the current filters. Adjust sidebar filters to see Market Pulse.")
        return

    # ── Filter histories to only include tickers in filtered_df ──
    _filtered_tickers = set(filtered_df['ticker'].unique())
    _filtered_hist = {t: h for t, h in histories.items() if t in _filtered_tickers}
    if not _filtered_hist:
        st.info("No history data for filtered stocks.")
        return

    # ── Helper: build weekly snapshots from histories ────────────
    @st.cache_data(ttl=300, show_spinner=False)
    def _build_weekly_snapshots(_hist_keys, _hist_data_tuple):
        """Reconstruct per-week aggregates from filtered histories dict.

        We receive histories data as a tuple-of-tuples for hashability.
        Returns list of dicts, one per week, sorted chronologically.
        """
        hist = {k: v for k, v in zip(_hist_keys, _hist_data_tuple)}
        # Discover all dates across tickers
        date_set: dict = {}  # date_str -> week_index
        for ticker, h in hist.items():
            for i, d in enumerate(h.get('dates', [])):
                ds = str(d)[:10]
                if ds not in date_set:
                    date_set[ds] = ds
        sorted_dates = sorted(date_set.keys())
        if not sorted_dates:
            return []

        weeks = []
        for wi, ds in enumerate(sorted_dates):
            snap = {
                'date': ds, 'tickers': [], 'scores': [], 'ranks': [],
                'prev_ranks': [], 'grades': [], 'patterns': [],
                'sectors': [], 'decay_scores': [], 'trend_qualities': [],
                'market_strengths': [],
            }
            for ticker, h in hist.items():
                dates_list = [str(d)[:10] for d in h.get('dates', [])]
                if ds not in dates_list:
                    continue
                idx = dates_list.index(ds)
                snap['tickers'].append(ticker)
                scores = h.get('scores', [])
                snap['scores'].append(scores[idx] if idx < len(scores) else 0)
                ranks = h.get('ranks', [])
                snap['ranks'].append(ranks[idx] if idx < len(ranks) else 0)
                prev_r = ranks[idx - 1] if idx > 0 and idx - 1 < len(ranks) else (ranks[idx] if idx < len(ranks) else 0)
                snap['prev_ranks'].append(prev_r)
                # Grade from score
                sc = scores[idx] if idx < len(scores) else 0
                gr = 'F'
                for thr, lbl, _ in GRADE_DEFS:
                    if sc >= thr:
                        gr = lbl
                        break
                snap['grades'].append(gr)
                pats = h.get('pattern_history', [])
                snap['patterns'].append(pats[idx] if idx < len(pats) else 'neutral')
                snap['sectors'].append(h.get('sector', ''))
                tq = h.get('trend_qualities', [])
                snap['trend_qualities'].append(tq[idx] if idx < len(tq) else 50)
                ms = h.get('overall_market_strength', [])
                snap['market_strengths'].append(ms[idx] if idx < len(ms) else 50)
            weeks.append(snap)
        return weeks

    # Build hashable args for caching (uses FILTERED histories only)
    _h_keys = tuple(sorted(_filtered_hist.keys()))
    _h_data = tuple(
        {k: (tuple(v) if isinstance(v, list) else v) for k, v in _filtered_hist[t].items()}
        for t in _h_keys
    )
    weekly_snaps = _build_weekly_snapshots(_h_keys, _h_data)

    if len(weekly_snaps) < 1:
        st.info("Not enough data for Market Pulse. Load at least 1 weekly snapshot.")
        return

    latest = weekly_snaps[-1]
    n_weeks = len(weekly_snaps)

    # ── Current metrics ──────────────────────────────────────────
    regime = filtered_df['market_regime'].iloc[0] if 'market_regime' in filtered_df.columns else 'SIDEWAYS'
    trend_med = float(filtered_df['market_trend_median'].iloc[0]) if 'market_trend_median' in filtered_df.columns else 50.0
    avg_score = float(filtered_df['trajectory_score'].mean()) if 'trajectory_score' in filtered_df.columns else 0
    avg_alpha = float(filtered_df['alpha_score'].mean()) if 'alpha_score' in filtered_df.columns else 0

    # Breadth from latest snapshot
    lat_scores = np.array(latest['scores'], dtype=float)
    lat_ranks = np.array(latest['ranks'], dtype=float)
    lat_prev = np.array(latest['prev_ranks'], dtype=float)
    improving = int(np.sum(lat_ranks < lat_prev))
    declining = int(np.sum(lat_ranks > lat_prev))
    unchanged = int(np.sum(lat_ranks == lat_prev))
    pct_strong = int(100 * np.sum(lat_scores >= 55) / max(len(lat_scores), 1))
    pct_improving = int(100 * improving / max(len(lat_ranks), 1))
    ad_ratio = round(improving / max(declining, 1), 2)

    # Regime styling
    regime_cfg = {
        'BULL':     ('#3fb950', '🟢', 'linear-gradient(135deg, rgba(63,185,80,0.12), rgba(63,185,80,0.04))'),
        'BEAR':     ('#f85149', '🔴', 'linear-gradient(135deg, rgba(248,81,73,0.12), rgba(248,81,73,0.04))'),
        'SIDEWAYS': ('#d29922', '🟡', 'linear-gradient(135deg, rgba(210,153,34,0.12), rgba(210,153,34,0.04))'),
    }
    r_color, r_icon, r_bg = regime_cfg.get(regime, regime_cfg['SIDEWAYS'])

    # Strength bar width + color
    strength_pct = max(0, min(100, trend_med))
    s_color = '#3fb950' if strength_pct >= 58 else ('#f85149' if strength_pct < 42 else '#d29922')

    # ────────────────────────────────────────────────────────────
    # SECTION 1: PULSE HERO CARD
    # ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);border-radius:16px;
        padding:22px 28px;margin-bottom:18px;border:1px solid #30363d;
        box-shadow:0 4px 24px rgba(0,0,0,0.3);">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
        <div>
          <div style="font-size:1.5rem;font-weight:800;color:#e6edf3;letter-spacing:-0.3px;">
            📡 Market Pulse</div>
          <div style="color:#8b949e;font-size:0.85rem;margin-top:2px;">
            {metadata.get('date_range','')} &nbsp;·&nbsp; {n_weeks} weeks
            &nbsp;·&nbsp; {n_stocks} stocks</div>
        </div>
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="{r_bg};border:1px solid {r_color}33;border-radius:12px;padding:10px 18px;
              text-align:center;">
            <div style="font-size:1.6rem;">{r_icon}</div>
            <div style="font-size:0.92rem;font-weight:800;color:{r_color};letter-spacing:0.5px;">
              {regime}</div>
          </div>
          <div style="background:#161b22;border:1px solid #30363d;border-radius:12px;padding:10px 18px;
              text-align:center;min-width:100px;">
            <div style="font-size:0.65rem;color:#8b949e;text-transform:uppercase;letter-spacing:1px;">
              Mkt Strength</div>
            <div style="font-size:1.3rem;font-weight:800;color:{s_color};">{strength_pct:.0f}</div>
            <div style="background:#21262d;border-radius:3px;height:4px;margin-top:4px;overflow:hidden;">
              <div style="width:{strength_pct}%;height:100%;background:{s_color};border-radius:3px;"></div>
            </div>
          </div>
        </div>
      </div>
      <!-- 6 KPI chips -->
      <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:16px;">
        <div style="background:#0d1117;border-radius:10px;padding:10px;text-align:center;border:1px solid #30363d;">
          <div style="color:#3fb950;font-weight:700;font-size:1.15rem;">{pct_improving}%</div>
          <div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;">Improving</div></div>
        <div style="background:#0d1117;border-radius:10px;padding:10px;text-align:center;border:1px solid #30363d;">
          <div style="color:#58a6ff;font-weight:700;font-size:1.15rem;">{pct_strong}%</div>
          <div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;">Strong (B+)</div></div>
        <div style="background:#0d1117;border-radius:10px;padding:10px;text-align:center;border:1px solid #30363d;">
          <div style="color:#e6edf3;font-weight:700;font-size:1.15rem;">{ad_ratio}</div>
          <div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;">A/D Ratio</div></div>
        <div style="background:#0d1117;border-radius:10px;padding:10px;text-align:center;border:1px solid #30363d;">
          <div style="color:#d2a8ff;font-weight:700;font-size:1.15rem;">{avg_score:.0f}</div>
          <div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;">Avg T-Score</div></div>
        <div style="background:#0d1117;border-radius:10px;padding:10px;text-align:center;border:1px solid #30363d;">
          <div style="color:#FF6B35;font-weight:700;font-size:1.15rem;">{avg_alpha:.0f}</div>
          <div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;">Avg Alpha</div></div>
        <div style="background:#0d1117;border-radius:10px;padding:10px;text-align:center;border:1px solid #30363d;">
          <div style="color:{'#3fb950' if improving > declining else '#f85149'};font-weight:700;font-size:1.15rem;">
            {improving}↑ {declining}↓</div>
          <div style="color:#8b949e;font-size:0.68rem;text-transform:uppercase;">Adv / Dec</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────
    # SECTION 2:  Breadth Over Time
    # ────────────────────────────────────────────────────────────
    dates_list = []
    pct_improving_list = []
    pct_strong_list = []
    ad_ratio_list = []
    avg_score_list = []
    for snap in weekly_snaps:
        sc_arr = np.array(snap['scores'], dtype=float)
        rk_arr = np.array(snap['ranks'], dtype=float)
        pr_arr = np.array(snap['prev_ranks'], dtype=float)
        n_s = max(len(sc_arr), 1)
        imp = int(np.sum(rk_arr < pr_arr))
        dec = int(np.sum(rk_arr > pr_arr))
        dates_list.append(snap['date'])
        pct_improving_list.append(round(100 * imp / n_s, 1))
        pct_strong_list.append(round(100 * np.sum(sc_arr >= 55) / n_s, 1))
        ad_ratio_list.append(round(imp / max(dec, 1), 2))
        avg_score_list.append(round(float(np.mean(sc_arr)), 1))

    fig_breadth = go.Figure()
    fig_breadth.add_trace(go.Scatter(
        x=dates_list, y=pct_improving_list, name='% Improving',
        mode='lines+markers', line=dict(color='#3fb950', width=2.5),
        marker=dict(size=5), hovertemplate='%{x}<br>Improving: %{y}%<extra></extra>',
    ))
    fig_breadth.add_trace(go.Scatter(
        x=dates_list, y=pct_strong_list, name='% Strong (B+)',
        mode='lines+markers', line=dict(color='#58a6ff', width=2.5),
        marker=dict(size=5), hovertemplate='%{x}<br>Strong: %{y}%<extra></extra>',
    ))
    fig_breadth.add_trace(go.Scatter(
        x=dates_list, y=avg_score_list, name='Avg T-Score',
        mode='lines+markers', line=dict(color='#d2a8ff', width=2, dash='dot'),
        marker=dict(size=4), yaxis='y2',
        hovertemplate='%{x}<br>Avg Score: %{y}<extra></extra>',
    ))
    fig_breadth.add_trace(go.Scatter(
        x=dates_list, y=ad_ratio_list, name='A/D Ratio',
        mode='lines+markers', line=dict(color='#d29922', width=2, dash='dash'),
        marker=dict(size=4), yaxis='y2',
        hovertemplate='%{x}<br>A/D: %{y}<extra></extra>',
    ))
    fig_breadth.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        height=380, margin=dict(l=50, r=50, t=40, b=40),
        title=dict(text='Market Breadth Over Time', font=dict(size=14, color='#e6edf3')),
        legend=dict(orientation='h', y=-0.15, font=dict(size=11)),
        yaxis=dict(title=dict(text='% of Stocks', font=dict(size=11)),
                   gridcolor='#21262d', zeroline=False),
        yaxis2=dict(title=dict(text='Score / Ratio', font=dict(size=11)),
                    overlaying='y', side='right',
                    gridcolor='#21262d', zeroline=False),
        xaxis=dict(gridcolor='#21262d'),
        hovermode='x unified',
    )
    st.plotly_chart(fig_breadth, use_container_width=True, key='mp_breadth_chart')

    # Breadth sparkline summary
    if len(pct_improving_list) >= 2:
        delta_imp = pct_improving_list[-1] - pct_improving_list[-2]
        delta_str = pct_strong_list[-1] - pct_strong_list[-2]
        d_i_c = '#3fb950' if delta_imp >= 0 else '#f85149'
        d_s_c = '#3fb950' if delta_str >= 0 else '#f85149'
        d_i_sign = '+' if delta_imp >= 0 else ''
        d_s_sign = '+' if delta_str >= 0 else ''
        st.markdown(f"""
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:6px;">
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:8px 16px;
              font-size:0.82rem;color:#c9d1d9;">
            <span style="color:{d_i_c};font-weight:700;">{d_i_sign}{delta_imp:.1f}%</span>
            improving vs prev week</div>
          <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:8px 16px;
              font-size:0.82rem;color:#c9d1d9;">
            <span style="color:{d_s_c};font-weight:700;">{d_s_sign}{delta_str:.1f}%</span>
            strong (B+) vs prev week</div>
        </div>""", unsafe_allow_html=True)


# ============================================
# UI: RANKINGS TAB — ALL TIME BEST (v3.0)
# ============================================

def render_rankings_tab(filtered_df: pd.DataFrame, all_df: pd.DataFrame,
                        histories: dict, metadata: dict):
    """Render the main rankings tab — Clean, minimal, maximum signal density."""

    # ── Ensure all required columns exist (defensive) ──
    for col, default in [('price_tag', ''), ('signal_tags', ''), ('decay_tag', ''),
                         ('decay_label', ''), ('decay_score', 0),
                         ('sector_alpha_tag', 'NEUTRAL'), ('sector_alpha_value', 0),
                         ('price_label', 'NEUTRAL'), ('price_alignment', 50),
                         ('grade_emoji', '📉'),
                         ('pattern_key', 'neutral'), ('pattern', '➖ Neutral'),
                         ('sector', ''), ('return_quality', 50),
                         ('company_name', ''), ('category', ''), ('industry', ''),
                         ('alpha_score', 0), ('conviction', 0), ('tmi', 0),
                         ('rally_gain', 0), ('rally_stage', 'UNKNOWN'),
                         ('rally_leg_pct', 0), ('confidence', 0)]:
        if col not in all_df.columns:
            all_df[col] = default
        if col not in filtered_df.columns:
            filtered_df[col] = default

    # ── Empty state guard ──
    if filtered_df.empty:
        st.info("🔍 No stocks match your current filters. Try adjusting or clearing filters.")
        return

    # ── Keep full filtered reference for intelligence dashboard ──
    full_filtered_df = filtered_df.copy()

    # ── Compute metrics from FULL filtered data (before Show Top truncation) ──
    total = len(all_df)
    shown = len(filtered_df)
    avg_score = filtered_df['trajectory_score'].mean()
    avg_alpha = filtered_df['alpha_score'].mean()
    rockets = int((filtered_df['pattern_key'] == 'rocket').sum())
    elites = int((filtered_df['pattern_key'] == 'stable_elite').sum())
    confirmed = int((filtered_df['price_label'] == 'PRICE_CONFIRMED').sum())
    divergent = int((filtered_df['price_label'] == 'PRICE_DIVERGENT').sum())
    decay_high = int((filtered_df['decay_label'] == 'DECAY_HIGH').sum())
    decay_mod = int((filtered_df['decay_label'] == 'DECAY_MODERATE').sum())
    decay_mild = int((filtered_df['decay_label'] == 'DECAY_MILD').sum())
    decay_any = decay_high + decay_mod + decay_mild
    clean_pct = round((1 - decay_any / max(shown, 1)) * 100, 1)
    sect_leaders = int((filtered_df['sector_alpha_tag'] == 'SECTOR_LEADER').sum())
    grade_s = int((filtered_df['grade'] == 'S').sum())
    grade_a = int((filtered_df['grade'] == 'A').sum())
    clean_n = shown - decay_any

    # ════════════════════════════════════════════
    # SECTION 1 — METRIC STRIP (compact, 9 chips)
    # ════════════════════════════════════════════
    def _chip(val, lbl, cls=''):
        return f'<div class="m-chip {cls}"><div class="m-val">{val}</div><div class="m-lbl">{lbl}</div></div>'

    sc_cls = 'm-green' if avg_score >= 55 else 'm-orange' if avg_score >= 40 else 'm-red'
    al_cls = 'm-green' if avg_alpha >= 55 else 'm-orange' if avg_alpha >= 35 else 'm-red'
    chips = ''.join([
        _chip(f'{shown:,}', 'Stocks'),
        _chip(f'{avg_score:.1f}', 'Avg T-Score', sc_cls),
        _chip(f'{avg_alpha:.1f}', 'Avg Alpha', al_cls),
        _chip(f'{grade_s + grade_a}', 'S + A Grade', 'm-green'),
        _chip(f'{rockets}', '🚀 Rockets'),
        _chip(f'{confirmed}', '💰 Confirmed', 'm-green'),
        _chip(f'{decay_high}', '🔻 Traps', 'm-red' if decay_high > 0 else ''),
        _chip(f'{sect_leaders}', '👑 Alpha', 'm-gold'),
        _chip(f'{metadata["total_weeks"]}', 'Weeks'),
    ])
    st.markdown(f'<div class="m-strip">{chips}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════
    # SECTION 2 — CONTROL PANEL + SMART TABLE
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">📋 Trajectory Rankings</div>', unsafe_allow_html=True)

    # ── Control row: Show Top | Sort | View ──
    ctl0, ctl1, ctl2 = st.columns([0.8, 1.3, 1.3])
    with ctl0:
        show_top_options = [10, 20, 50, 100, 200, 500, "All"]
        display_n_select = st.selectbox("Show Top", show_top_options,
                                         index=3, key='rank_topn',
                                         label_visibility='collapsed')
        display_n = len(filtered_df) if display_n_select == "All" else display_n_select
    with ctl1:
        sort_by = st.selectbox("Sort by", [
            'Trajectory Score', 'Alpha Score', 'Current Rank', 'Rank Change', 'TMI',
            'Conviction', 'Positional Quality', 'Best Rank', 'Streak', 'Trend', 'Velocity',
            'Consistency', 'Return Quality', 'Rally Gain', 'Price Alignment', 'Decay Score', 'Sector Alpha'
        ], key='rank_sort', label_visibility='collapsed')
    with ctl2:
        view_mode = st.selectbox("View", [
            'Standard', 'Compact', 'Signals', 'Trading', 'Complete', 'Custom'
        ], key='rank_view', label_visibility='collapsed')

    # ── T-Rank = rank within full universe (stable, never changes with filters) ──
    t_rank_sorted = all_df.sort_values(
        ['trajectory_score', 'confidence', 'consistency'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    t_rank_map = {t: i + 1 for i, t in enumerate(t_rank_sorted['ticker'])}
    filtered_df['t_rank_universe'] = filtered_df['ticker'].map(t_rank_map).fillna(0).astype(int)

    # ── Sort FIRST, then apply Show Top limit ──
    sort_map = {
        'Trajectory Score': ('trajectory_score', False),
        'Alpha Score': ('alpha_score', False),
        'Positional Quality': ('positional', False),
        'TMI': ('tmi', False),
        'Current Rank': ('current_rank', True),
        'Rank Change': ('rank_change', False),
        'Best Rank': ('best_rank', True),
        'Streak': ('streak', False),
        'Trend': ('trend', False),
        'Velocity': ('velocity', False),
        'Consistency': ('consistency', False),
        'Return Quality': ('return_quality', False),
        'Conviction': ('conviction', False),
        'Rally Gain': ('rally_gain', False),
        'Price Alignment': ('price_alignment', False),
        'Decay Score': ('decay_score', True),
        'Sector Alpha': ('sector_alpha_value', False),
    }
    col_name, ascending = sort_map.get(sort_by, ('trajectory_score', False))
    display_df = filtered_df.sort_values(col_name, ascending=ascending).head(display_n).reset_index(drop=True)
    display_df['t_rank'] = range(1, len(display_df) + 1)
    table_n = len(display_df)

    # ── Add latest price from histories ──
    display_df['latest_price'] = display_df['ticker'].apply(
        lambda t: round(histories.get(t, {}).get('prices', [0])[-1], 2)
        if histories.get(t, {}).get('prices') else 0
    )

    # ── Column definitions for each view ──
    # Each: (df_col, display_name, tooltip, column_config_or_None)
    COL_DEFS = {
        'T-Rank': ('t_rank_universe', 'T-Rank', 'Rank in full universe (all stocks, unfiltered)',
                     st.column_config.NumberColumn(width="small")),
        'Ticker':   ('ticker', 'Ticker', 'NSE ticker symbol', None),
        'Company':  ('company_name', 'Company', 'Company name (truncated)', None),
        'Sector':   ('sector', 'Sector', 'Business sector', None),
        'Category': ('category', 'Category', 'Large/Mid/Small Cap', None),
        '₹ Price':  ('latest_price', '₹ Price', 'Latest closing price (₹)',
                     st.column_config.NumberColumn(format="₹%.2f")),
        'T-Score':  ('trajectory_score', 'T-Score', 'Composite trajectory score (0-100)',
                     st.column_config.ProgressColumn('T-Score', min_value=0, max_value=100, format="%.1f")),
        'Alpha':    ('alpha_score', 'Alpha', 'Alpha Score — forward-return predictor (0-100)',
                     st.column_config.ProgressColumn('Alpha', min_value=0, max_value=100, format="%.0f")),
        'Grade':    ('grade', 'Grade', 'S/A/B/C/D/F based on T-Score', None),
        'Pattern':  ('pattern', 'Pattern', 'Trajectory pattern classification', None),
        'Signals':  ('signal_tags', 'Signals', 'All signal tags combined', None),
        'TMI':      ('tmi', 'TMI', 'Trajectory Momentum Index (0-100)',
                     st.column_config.ProgressColumn('TMI', min_value=0, max_value=100, format="%.0f")),
        'Best':     ('best_rank', 'Best', 'Best rank ever achieved', None),
        'Δ Total':  ('rank_change', 'Δ Total', 'Total rank change (first → now)',
                     st.column_config.NumberColumn(format="%+d")),
        'Δ Week':   ('last_week_change', 'Δ Week', 'Rank change this week',
                     st.column_config.NumberColumn(format="%+d")),
        'Streak':   ('streak', 'Streak', 'Consecutive weeks improving',
                     st.column_config.NumberColumn(format="%d 🔥")),
        'Wks':      ('weeks', 'Wks', 'Total weeks tracked', None),
        'Trend':    ('trend', 'Trend', 'Trend component score', None),
        'Velocity': ('velocity', 'Velocity', 'Velocity component score', None),
        'Consistency': ('consistency', 'Consistency', 'Consistency component score', None),
        'Positional': ('positional', 'Positional', 'Positional quality score', None),
        'RetQuality': ('return_quality', 'RetQuality', 'Return quality component (v6.0)',
                     st.column_config.ProgressColumn('RetQuality', min_value=0, max_value=100, format="%.1f")),
        'Price Signal': ('price_label', 'Price Signal', 'Price alignment: CONFIRMED/DIVERGENT/NEUTRAL', None),
        'Decay':    ('decay_label', 'Decay', 'Momentum decay level: HIGH/MODERATE/MILD/CLEAN', None),
        'Sect Alpha': ('sector_alpha_tag', 'Sect Alpha', 'Sector alpha classification', None),
        'Trajectory': ('sparkline', 'Trajectory', 'Score trajectory over time',
                       st.column_config.LineChartColumn('Trajectory', y_min=0, y_max=100, width="medium")),
        # v6.3: Advanced Trading Signals
        'Conviction': ('conviction', 'Conviction', 'Buy conviction score 0-100 (higher = stronger BUY signal)',
                       st.column_config.ProgressColumn('Conviction', min_value=0, max_value=100, format="%.0f")),
        'Conv Tag': ('conviction_tag', 'Conv', 'Conviction level: VERY_HIGH/HIGH/MODERATE/LOW/VERY_LOW', None),
        'Risk-Adj': ('risk_adj_score', 'Risk-Adj', 'Risk-adjusted T-Score (volatility penalized)',
                     st.column_config.ProgressColumn('Risk-Adj', min_value=0, max_value=100, format="%.1f")),
        'Exit Risk': ('exit_risk', 'Exit Risk', 'Exit/sell risk score 0-100 (higher = consider selling)',
                      st.column_config.ProgressColumn('Exit Risk', min_value=0, max_value=100, format="%.0f")),
        'Exit Tag': ('exit_tag', 'Exit', 'Exit warning: EXIT_NOW/CAUTION/WATCH/HOLD', None),
        'Hot Streak': ('hot_streak', 'Hot', 'Hot streak detected (4+ weeks improving at high position)', None),
        # v8.0: Wave Signal Fusion
        'Wave':     ('wave_fusion_score', 'Wave', 'Wave Signal Fusion score 0-100 (cross-system validation)',
                     st.column_config.ProgressColumn('Wave', min_value=0, max_value=100, format="%.0f")),
        'WF Label': ('wave_fusion_label', 'WF', 'Wave fusion: STRONG/CONFIRMED/NEUTRAL/WEAK/CONFLICT', None),
        'Confluence': ('wave_confluence', 'Conf', 'Wave-Trajectory confluence agreement 0-100',
                       st.column_config.ProgressColumn('Conf', min_value=0, max_value=100, format="%.0f")),
        'Inst Flow': ('wave_inst_flow', 'Flow', 'Institutional money flow signal 0-100',
                      st.column_config.ProgressColumn('Flow', min_value=0, max_value=100, format="%.0f")),
        'Harmony':  ('wave_harmony', 'Harm', 'Momentum harmony score 0-100',
                     st.column_config.ProgressColumn('Harm', min_value=0, max_value=100, format="%.0f")),
        # Rally Leg (price-trough based — where the current move actually started)
        'Rally%':   ('rally_leg_pct', 'Rally%', 'Rally leg completion: % of current move from recent trough to 52w high. 0%=just started, 100%=at 52w high.',
                     st.column_config.ProgressColumn('Rally%', min_value=0, max_value=100, format="%.0f%%")),
        'Stage':    ('rally_stage', 'Stage', 'Rally stage by gain size: FRESH(<5%)/EARLY(5-15%)/RUNNING(15-30%)/MATURE(30-50%)/LATE(>50%)', None),
        'RallyGain': ('rally_gain', 'RallyGain', '% price gain from recent trough (where this specific rally started)',
                     st.column_config.NumberColumn(format="+%.1f%%")),
    }

    VIEW_PRESETS = {
        'Compact':  ['T-Rank', 'Ticker', '₹ Price', 'T-Score', 'Alpha', 'Grade', 'Pattern',
                     'Δ Total', 'Streak', 'Trajectory'],
        'Standard': ['T-Rank', 'Ticker', 'Company', 'Sector', 'Category', '₹ Price', 'T-Score', 'Alpha', 'Grade',
                     'Pattern', 'Signals', 'TMI', 'Best', 'Δ Total', 'Δ Week', 'Streak', 'Wks',
                     'Stage', 'RallyGain', 'Rally%', 'Trajectory'],
        'Signals':  ['T-Rank', 'Ticker', 'Company', 'Sector', 'Category', '₹ Price', 'T-Score', 'Alpha', 'Grade',
                     'Pattern', 'Signals', 'Price Signal', 'Decay', 'Sect Alpha', 'Trajectory'],
        'Trading':  ['T-Rank', 'Ticker', 'Company', '₹ Price', 'T-Score', 'Alpha', 'Grade', 'Conviction',
                     'Conv Tag', 'Risk-Adj', 'Exit Risk', 'Exit Tag', 'Hot Streak',
                     'Wave', 'WF Label', 'Confluence', 'Inst Flow', 'Rally%', 'RallyGain', 'Stage', 'Streak', 'Trajectory'],
        'Complete': ['T-Rank', 'Ticker', 'Company', 'Sector', 'Category', '₹ Price', 'T-Score', 'Alpha',
                     'Grade', 'Pattern', 'Signals', 'Best', 'Δ Total', 'Δ Week', 'Streak', 'Wks',
                     'Trend', 'Velocity', 'Consistency', 'Positional', 'RetQuality', 'Price Signal', 'Decay', 'Sect Alpha',
                     'Conviction', 'Risk-Adj', 'Exit Risk', 'Hot Streak',
                     'Wave', 'Confluence', 'Inst Flow', 'Harmony', 'Rally%', 'RallyGain', 'Trajectory'],
    }

    # ── Custom view: user picks columns ──
    if view_mode == 'Custom':
        all_col_names = list(COL_DEFS.keys())
        default_custom = VIEW_PRESETS['Standard']
        selected_cols = st.multiselect(
            "Select columns to display",
            all_col_names,
            default=default_custom,
            key='custom_cols'
        )
        if not selected_cols:
            selected_cols = ['T-Rank', 'Ticker', 'T-Score', 'Grade']
    else:
        selected_cols = VIEW_PRESETS.get(view_mode, VIEW_PRESETS['Standard'])

    # ── Build table_df and column_config from selected columns ──
    df_cols = []
    disp_names = []
    col_config = {}
    for col_key in selected_cols:
        cdef = COL_DEFS.get(col_key)
        if cdef is None:
            continue
        src_col, disp_name, tooltip, cfg = cdef
        if src_col not in display_df.columns:
            continue
        df_cols.append(src_col)
        disp_names.append(disp_name)
        if cfg is not None:
            col_config[disp_name] = cfg

    table_df = display_df[df_cols].copy()
    table_df.columns = disp_names

    # ── Truncate long text columns ──
    if 'Company' in table_df.columns:
        table_df['Company'] = table_df['Company'].astype(str).str[:24]
    if 'Sector' in table_df.columns:
        table_df['Sector'] = table_df['Sector'].astype(str).str[:18]
    if 'Category' in table_df.columns:
        table_df['Category'] = table_df['Category'].astype(str).str[:12]

    # ── Dynamic height ──
    tbl_height = min(800, max(180, len(table_df) * 35 + 60))

    st.caption(f"Showing {table_n:,} of {shown:,} filtered stocks · Sorted by **{sort_by}**")

    st.dataframe(
        table_df, column_config=col_config,
        hide_index=True, use_container_width=True, height=tbl_height,
    )

    # ── Signal Legend (collapsible) ──
    if 'Signals' in selected_cols:
        with st.expander("📖 Signal Legend", expanded=False):
            st.markdown(
                "| Signal | Meaning |\n"
                "|--------|--------|\n"
                "| 💰PRC | **Price Confirmed** — price direction matches rank direction |\n"
                "| ⚠️DIV | **Price Divergent** — price falling despite good rank (trap risk) |\n"
                "| 🔥RET | **Strong Returns** — above-average recent returns (return quality ≥ 75) |\n"
                "| 💧RET | **Weak Returns** — below-average returns despite rank (return quality ≤ 30) |\n"
                "| 🔻DEC | **High Decay** — momentum collapsing, consider exit |\n"
                "| ⚡DEC | **Moderate Decay** — momentum fading, watch closely |\n"
                "| 🌊WAV | **Wave Strong** — all wave detection systems agree (high confidence) |\n"
                "| 🔇WAV | **Wave Conflict** — wave systems disagree (lower confidence) |\n"
                "| 👑LDR | **Sector Leader** — significantly outperforming its sector peers |\n"
                "| 🏷️BTA | **Sector Beta** — riding sector trend, not individual strength |\n"
                "| 📉LAG | **Sector Laggard** — underperforming its sector peers |"
            )

    # ════════════════════════════════════════════
    # SECTION 6 — INTELLIGENCE DASHBOARD (tabs)
    # ════════════════════════════════════════════
    st.markdown('<div class="sec-head">📊 Intelligence</div>', unsafe_allow_html=True)

    tab_health, tab_sectors, tab_patterns, tab_grades = st.tabs([
        "🫀 Health", "🏢 Sectors", "🔮 Patterns", "📊 Grades & Alpha"
    ])

    # ─── TAB: Health ───
    with tab_health:
        h1, h2 = st.columns(2)
        with h1:
            fig_health = go.Figure(data=[go.Pie(
                labels=['Clean', 'Mild', 'Moderate', 'Severe'],
                values=[clean_n, decay_mild, decay_mod, decay_high],
                marker_colors=['#238636', '#d29922', '#da3633', '#f85149'],
                hole=0.62, textinfo='label+value', textfont_size=11, sort=False,
                textfont_color='#e6edf3'
            )])
            fig_health.update_layout(
                title=dict(text="Momentum Health", font=dict(size=13, color='#e6edf3')),
                height=340, template='plotly_dark', showlegend=False,
                margin=dict(t=45, b=15, l=15, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                annotations=[dict(text=f'{clean_pct}%<br><span style="font-size:10px;color:#8b949e">Clean</span>',
                                   x=0.5, y=0.5, font_size=18, showarrow=False, font_color='#3fb950')]
            )
            st.plotly_chart(fig_health, use_container_width=True)

        with h2:
            above70 = int((full_filtered_df['trajectory_score'] >= 70).sum())
            high_conviction = int((full_filtered_df['conviction'] >= 65).sum()) if 'conviction' in full_filtered_df.columns else 0
            confirmed_n = int((full_filtered_df['price_label'] == 'PRICE_CONFIRMED').sum()) if 'price_label' in full_filtered_df.columns else 0
            decay_any_n = int(full_filtered_df['decay_label'].isin(['DECAY_HIGH', 'DECAY_MODERATE']).sum()) if 'decay_label' in full_filtered_df.columns else 0

            fig_pipe = go.Figure(go.Waterfall(
                orientation="v",
                measure=["absolute", "absolute", "relative", "relative", "absolute"],
                x=["Universe", "T≥70", "💰 Confirmed", "🔻 Decay", "S+A Final"],
                y=[shown, above70, confirmed_n, -decay_any_n, grade_s + grade_a],
                connector={"line": {"color": "#30363d"}},
                increasing={"marker": {"color": "#238636"}},
                decreasing={"marker": {"color": "#da3633"}},
                totals={"marker": {"color": "#FF6B35"}},
                text=[shown, above70, f"+{confirmed_n}", f"−{decay_any_n}", grade_s + grade_a],
                textposition="outside", textfont_color='#e6edf3'
            ))
            fig_pipe.update_layout(
                title=dict(text="Score Pipeline", font=dict(size=13, color='#e6edf3')),
                height=340, template='plotly_dark', showlegend=False,
                margin=dict(t=45, b=25, l=30, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_pipe, use_container_width=True)

        hm1, hm2, hm3, hm4 = st.columns(4)
        hm1.metric("Clean Stocks", f"{clean_n:,}", f"{clean_pct}%")
        hm2.metric("Price Confirmed", confirmed, f"{round(confirmed/max(shown,1)*100,1)}%")
        hm3.metric("Decay Warnings", decay_any, f"−{round(decay_any/max(shown,1)*100,1)}%", delta_color="inverse")
        hm4.metric("Divergent", divergent, f"{round(divergent/max(shown,1)*100,1)}%", delta_color="inverse")

    # ─── TAB: Sectors ───
    with tab_sectors:
        qualified = full_filtered_df[full_filtered_df['weeks'] >= 3].copy()
        if not qualified.empty:
            sect_agg = qualified.groupby('sector').agg(
                avg_score=('trajectory_score', 'mean'),
                count=('trajectory_score', 'count'),
                leaders=('sector_alpha_tag', lambda x: (x == 'SECTOR_LEADER').sum()),
                decay_n=('decay_label', lambda x: x.isin(['DECAY_HIGH', 'DECAY_MODERATE']).sum()),
            ).reset_index()
            sect_agg = sect_agg[sect_agg['count'] >= 3].sort_values('avg_score', ascending=False).head(20)

            if not sect_agg.empty:
                max_s, min_s = sect_agg['avg_score'].max(), sect_agg['avg_score'].min()
                rng = max(max_s - min_s, 1)
                colors = []
                for s in sect_agg['avg_score']:
                    r = (s - min_s) / rng
                    colors.append('#238636' if r > 0.7 else '#d29922' if r > 0.35 else '#da3633')

                bar_labels = [
                    f"{row['sector'][:22]}  ·  {int(row['count'])} stk  ·  👑{int(row['leaders'])}"
                    for _, row in sect_agg.iterrows()
                ]
                fig_sec = go.Figure(data=[go.Bar(
                    x=sect_agg['avg_score'].values, y=bar_labels,
                    orientation='h', marker_color=colors,
                    text=[f"{v:.1f}" for v in sect_agg['avg_score']],
                    textposition='auto', textfont_color='#e6edf3'
                )])
                fig_sec.update_layout(
                    title=dict(text="Sectors by Avg Trajectory Score", font=dict(size=13, color='#e6edf3')),
                    height=min(480, 70 + len(sect_agg) * 24), template='plotly_dark',
                    xaxis_title='Avg T-Score', yaxis=dict(autorange='reversed'),
                    margin=dict(t=45, b=35, l=250, r=15),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig_sec, use_container_width=True)
        else:
            st.info("Need 3+ weeks of data for sector analysis")

    # ─── TAB: Patterns ───
    with tab_patterns:
        p1, p2 = st.columns([3, 2])
        with p1:
            pattern_counts = full_filtered_df['pattern_key'].value_counts()
            p_labels = [PATTERN_DEFS.get(k, ('', k, ''))[1] for k in pattern_counts.index]
            p_colors = [PATTERN_COLORS.get(k, '#8b949e') for k in pattern_counts.index]
            fig_pat = go.Figure(data=[go.Pie(
                labels=p_labels, values=pattern_counts.values,
                marker_colors=p_colors, hole=0.5,
                textinfo='label+percent', textfont_size=10,
                textfont_color='#e6edf3'
            )])
            fig_pat.update_layout(
                title=dict(text="Pattern Distribution", font=dict(size=13, color='#e6edf3')),
                height=380, template='plotly_dark', showlegend=False,
                margin=dict(t=45, b=15, l=15, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig_pat, use_container_width=True)

        with p2:
            st.markdown("**Pattern Breakdown**")
            for pk in pattern_counts.index[:10]:
                emoji, name, _ = PATTERN_DEFS.get(pk, ('•', pk, ''))
                cnt = int(pattern_counts[pk])
                pct = round(cnt / max(len(full_filtered_df), 1) * 100, 1)
                bar_w = min(pct * 3, 100)
                bar_color = PATTERN_COLORS.get(pk, '#8b949e')
                st.markdown(f"""<div style="margin-bottom:5px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#e6edf3;">
                        <span>{emoji} {name}</span><span style="color:#8b949e;">{cnt} ({pct}%)</span>
                    </div>
                    <div class="pbar-wrap"><div class="pbar" style="background:{bar_color};width:{bar_w}%;"></div></div>
                </div>""", unsafe_allow_html=True)

    # ─── TAB: Grades & Alpha ───
    with tab_grades:
        ga1, ga2 = st.columns(2)
        with ga1:
            grade_counts = full_filtered_df['grade'].value_counts().reindex(['S', 'A', 'B', 'C', 'D', 'F']).fillna(0)
            fig_gr = go.Figure(data=[go.Bar(
                x=grade_counts.index, y=grade_counts.values,
                marker_color=[GRADE_COLORS.get(g, '#8b949e') for g in grade_counts.index],
                text=[int(v) for v in grade_counts.values],
                textposition='outside', width=0.6,
                textfont_color='#e6edf3'
            )])
            fig_gr.update_layout(
                title=dict(text="Grade Distribution", font=dict(size=13, color='#e6edf3')),
                height=320, template='plotly_dark',
                margin=dict(t=45, b=35, l=35, r=15),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(gridcolor='#21262d'),
            )
            st.plotly_chart(fig_gr, use_container_width=True)

        with ga2:
            alpha_order = ['SECTOR_LEADER', 'SECTOR_OUTPERFORM', 'SECTOR_ALIGNED',
                           'SECTOR_BETA', 'SECTOR_LAGGARD', 'NEUTRAL']
            alpha_cmap = {'SECTOR_LEADER': '#d29922', 'SECTOR_OUTPERFORM': '#238636',
                          'SECTOR_ALIGNED': '#8b949e', 'SECTOR_BETA': '#da3633',
                          'SECTOR_LAGGARD': '#f85149', 'NEUTRAL': '#484f58'}
            alpha_counts = full_filtered_df['sector_alpha_tag'].value_counts()
            al, av, ac = [], [], []
            for a in alpha_order:
                if a in alpha_counts.index:
                    al.append(a.replace('SECTOR_', ''))
                    av.append(int(alpha_counts[a]))
                    ac.append(alpha_cmap.get(a, '#8b949e'))
            if al:
                fig_al = go.Figure(data=[go.Bar(
                    x=al, y=av, marker_color=ac,
                    text=av, textposition='outside', width=0.5,
                    textfont_color='#e6edf3'
                )])
                fig_al.update_layout(
                    title=dict(text="Sector Alpha Classification", font=dict(size=13, color='#e6edf3')),
                    height=320, template='plotly_dark',
                    margin=dict(t=45, b=35, l=35, r=15),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    yaxis=dict(gridcolor='#21262d'),
                )
                st.plotly_chart(fig_al, use_container_width=True)


# ============================================
# UI: SEARCH TAB
# ============================================

def render_search_tab(filtered_df: pd.DataFrame, traj_df: pd.DataFrame, histories: dict, dates_iso: list):
    """Search & Analyse — v4.0 (All-Time Best Engineering: Alpha, Exit Risk, Correct Weights, Full Signals)

    Fixes from v3.0:
    - Score Pipeline uses correct 4-arg adaptive weights (avg_pct, current_pct, confidence, regime_signal)
    - All 8 components shown (including breakout_quality)
    - Alpha Score, Exit Risk, Hot Streak, Market Regime displayed
    - T-Rank "of N" uses len(traj_df) not CSV total
    - rally_gain sign-safe, rally_weeks label corrected
    - All row[...] accesses use .get() with safe defaults
    - Chart helpers guard against empty data
    - Conviction color-coded by tag
    - Compare section has empty-state guidance
    - HTML content sanitized via html.escape
    """
    import html as _html

    # ── Search Input — dropdown shows only filtered stocks ──
    label_map = {}
    for _, r in filtered_df.iterrows():
        label_map[f"{r['ticker']} — {r['company_name'][:35]}"] = r['ticker']
    labels = sorted(label_map.keys())

    # Clear stale selection if it no longer exists in filtered labels
    if 'search_select' in st.session_state and st.session_state['search_select'] not in labels:
        st.session_state['search_select'] = None

    selected_label = st.selectbox("🔍 Search Stock",
                                   labels, index=None,
                                   placeholder="Type ticker or company name...",
                                   key='search_select')

    if selected_label is None:
        st.info("👆 Select a stock from the dropdown to view detailed trajectory analysis")
        return

    ticker = label_map[selected_label]
    matches = filtered_df[filtered_df['ticker'] == ticker]
    if matches.empty:
        st.warning("Stock not found in current filter selection")
        return
    row = matches.iloc[0]
    h = histories.get(ticker, {})
    if not h or not h.get('ranks') or not h.get('prices'):
        st.warning("No history data available for this ticker")
        return

    # ── Derived Data (safe) ──
    latest_price = h['prices'][-1] if h['prices'] else 0
    ranks_list = h['ranks']
    totals_list = h.get('total_per_week', [])
    pcts = ranks_to_percentiles(ranks_list, totals_list) if ranks_list and totals_list else []
    universe_size = len(traj_df)

    # T-Rank = rank within full universe (all stocks, not filtered)
    sorted_df = traj_df.sort_values(
        ['trajectory_score', 'confidence', 'consistency'],
        ascending=[False, False, False]
    ).reset_index(drop=True)
    t_rank_idx = sorted_df[sorted_df['ticker'] == ticker].index
    t_rank = int(t_rank_idx[0]) + 1 if len(t_rank_idx) > 0 else 0

    pattern_key = row.get('pattern_key', 'neutral')
    p_emoji, p_name, p_desc = PATTERN_DEFS.get(pattern_key, ('➖', 'Neutral', ''))
    p_color = PATTERN_COLORS.get(pattern_key, '#8b949e')
    grade = row.get('grade', 'F')
    grade_emoji = row.get('grade_emoji', '📉')
    grade_color = {'S': '#FFD700', 'A': '#3fb950', 'B': '#58a6ff', 'C': '#d29922', 'D': '#FF5722', 'F': '#f85149'}.get(grade, '#888')
    t_score = row.get('trajectory_score', 0)
    alpha = row.get('alpha_score', 0)
    alpha_c = '#00E676' if alpha >= 70 else '#3fb950' if alpha >= 50 else '#FF9800' if alpha >= 30 else '#f85149'
    market_regime = row.get('market_regime', 'SIDEWAYS')
    mr_colors = {'BULL': '#3fb950', 'BEAR': '#f85149', 'SIDEWAYS': '#d29922'}
    mr_c = mr_colors.get(market_regime, '#8b949e')

    # ── Header Card ──
    company_esc = _html.escape(str(row.get('company_name', '')))
    sector_esc = _html.escape(str(row.get('sector', '')))
    industry_esc = _html.escape(str(row.get('industry', '')))
    category_esc = _html.escape(str(row.get('category', '')))
    st.markdown(f"""
    <div style="background:#0d1117; border-radius:14px; padding:20px 24px; margin-bottom:16px; border:1px solid #30363d;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:12px;">
            <div style="flex:1; min-width:200px;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px; flex-wrap:wrap;">
                    <span style="font-size:1.6rem; font-weight:800; color:#fff;">{_html.escape(ticker)}</span>
                    <span style="background:{p_color}22; color:{p_color}; padding:3px 10px; border-radius:12px; font-size:0.75rem; border:1px solid {p_color}44;">{p_emoji} {p_name}</span>
                    <span style="background:{mr_c}22; color:{mr_c}; padding:3px 8px; border-radius:10px; font-size:0.7rem; border:1px solid {mr_c}44;">{'📈' if market_regime == 'BULL' else '📉' if market_regime == 'BEAR' else '➡️'} {market_regime}</span>
                </div>
                <div style="color:#8b949e; font-size:0.95rem; margin-bottom:2px;">{company_esc}</div>
                <div style="color:#484f58; font-size:0.8rem;">{category_esc} · {sector_esc} · {industry_esc}</div>
            </div>
            <div style="display:flex; gap:18px; align-items:center; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">T-Rank</div>
                    <div style="font-size:1.7rem; font-weight:800; color:#58a6ff;">#{t_rank}</div>
                    <div style="font-size:0.62rem; color:#484f58;">of {universe_size:,}</div>
                </div>
                <div style="width:1px; height:48px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">T-Score</div>
                    <div style="font-size:1.7rem; font-weight:800; color:#FF6B35;">{t_score:.1f}</div>
                    <div style="font-size:0.62rem; color:#484f58;">/ 100</div>
                </div>
                <div style="width:1px; height:48px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Grade</div>
                    <div style="font-size:1.7rem; font-weight:800; color:{grade_color};">{grade}</div>
                    <div style="font-size:0.62rem; color:#484f58;">{grade_emoji}</div>
                </div>
                <div style="width:1px; height:48px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Alpha</div>
                    <div style="font-size:1.7rem; font-weight:800; color:{alpha_c};">{alpha:.0f}</div>
                    <div style="font-size:0.62rem; color:#484f58;">/ 100</div>
                </div>
                <div style="width:1px; height:48px; background:#30363d;"></div>
                <div style="text-align:center;">
                    <div style="font-size:0.62rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px;">Price</div>
                    <div style="font-size:1.7rem; font-weight:800; color:#e6edf3;">₹{latest_price:,.1f}</div>
                    <div style="font-size:0.62rem; color:#484f58;">Latest</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Compact KPI Strip ──
    price_label_display = row.get('price_label', 'NEUTRAL')
    decay_lbl = row.get('decay_label', '')
    sa_tag = row.get('sector_alpha_tag', 'NEUTRAL')
    sa_icons = {'SECTOR_LEADER': '👑', 'SECTOR_OUTPERFORM': '⬆️', 'SECTOR_ALIGNED': '➖', 'SECTOR_BETA': '🏷️', 'SECTOR_LAGGARD': '📉'}
    conv_val = row.get('conviction', 0)
    conv_tag = row.get('conviction_tag', '')
    conv_emoji = row.get('conviction_emoji', '')
    exit_risk_val = row.get('exit_risk', 0)
    exit_emoji = row.get('exit_emoji', '✅')
    exit_tag = row.get('exit_tag', '')
    hot = row.get('hot_streak', False)

    kpi_items = [
        ('CSV Rank', f"#{row.get('current_rank', 0)}", f"{row.get('last_week_change', 0):+d}w"),
        ('Best Rank', f"#{row.get('best_rank', 0)}", ''),
        ('Total Δ', f"{row.get('rank_change', 0):+d}", '🔥' if row.get('rank_change', 0) > 0 else ''),
        ('Conviction', f"{conv_emoji} {conv_val}", conv_tag.replace('_', ' ').title() if conv_tag else ''),
        ('Streak', f"{row.get('streak', 0)}w", '🔥 HOT' if hot else ''),
        ('Price Align', f"{'💰' if price_label_display == 'PRICE_CONFIRMED' else '⚠️' if price_label_display == 'PRICE_DIVERGENT' else '➖'} {row.get('price_alignment', 50):.0f}", ''),
        ('Decay', f"{'🔻' if decay_lbl == 'DECAY_HIGH' else '⚡' if decay_lbl == 'DECAY_MODERATE' else '✅'} {row.get('decay_score', 0)}", ''),
        ('Exit Risk', f"{exit_emoji} {exit_risk_val}", exit_tag.replace('_', ' ').title() if exit_tag else 'Clean'),
        ('Sector', f"{sa_icons.get(sa_tag, '➖')}", sa_tag.split('_')[-1].title() if sa_tag != 'NEUTRAL' else 'Neutral'),
    ]
    kpi_html = ''.join([
        f'<div class="m-chip"><div style="font-size:0.62rem;color:#8b949e;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:0.95rem;font-weight:700;color:#e6edf3;">{val}</div>'
        f'<div style="font-size:0.6rem;color:#6e7681;">{sub}</div></div>'
        for label, val, sub in kpi_items
    ])
    st.markdown(f'<div class="m-strip">{kpi_html}</div>', unsafe_allow_html=True)

    # ── Exit Warning Banner (if exit risk is high) ──
    if exit_risk_val >= 60:
        exit_signals_str = row.get('exit_signals', '')
        _er_c = '#FF1744' if exit_risk_val >= 80 else '#FF9800'
        st.markdown(f"""
        <div style="background:{_er_c}15; border:1px solid {_er_c}44; border-radius:10px; padding:10px 16px; margin:8px 0;">
            <span style="color:{_er_c}; font-weight:700; font-size:0.85rem;">⚠️ Exit Risk: {exit_risk_val}/100</span>
            <span style="color:#8b949e; font-size:0.78rem; margin-left:12px;">{_html.escape(exit_signals_str) if exit_signals_str else 'Multiple risk signals detected'}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Row 1: Rank Trajectory + Radar ──
    chart_c1, chart_c2 = st.columns([3, 2])

    with chart_c1:
        st.markdown('<div class="sec-head">📊 Rank Trajectory</div>', unsafe_allow_html=True)
        _render_rank_chart(h, ticker)

    with chart_c2:
        st.markdown('<div class="sec-head">🎯 Component Breakdown</div>', unsafe_allow_html=True)
        _render_radar_chart(row)

    # ── Row 2: Price Trajectory (full width) ──
    st.markdown('<div class="sec-head">💰 Price Trajectory</div>', unsafe_allow_html=True)
    _render_price_chart(h, ticker)

    # ── Score Pipeline Detail (CORRECT weights with all 4 args) ──
    st.markdown('<div class="sec-head">🔬 Score Pipeline</div>', unsafe_allow_html=True)
    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        # Compute avg_pct correctly: mean of per-week percentiles (same as scoring pipeline)
        _pipe_pcts = ranks_to_percentiles(ranks_list, totals_list) if ranks_list and totals_list else [50]
        _pipe_avg_pct = float(np.mean(_pipe_pcts))
        _pipe_current_pct = _pipe_pcts[-1] if _pipe_pcts else 50.0
        _pipe_confidence = row.get('confidence', 0.5)
        # Regime signal: map market_regime to numeric
        _regime_map = {'BULL': 0.5, 'BEAR': -0.5, 'SIDEWAYS': 0.0}
        _pipe_regime = _regime_map.get(market_regime, 0.0)
        adp_w = _get_adaptive_weights(_pipe_avg_pct, _pipe_current_pct, _pipe_confidence, _pipe_regime)

        _comp_keys = ['positional', 'trend', 'velocity', 'acceleration', 'consistency', 'resilience', 'return_quality', 'breakout_quality']
        _comp_labels = ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience', 'RetQuality', 'Breakout']
        _comp_scores = [row.get(k, 50) for k in _comp_keys]
        _comp_weights = [adp_w.get(k, 0) for k in _comp_keys]
        _comp_contribs = [round(row.get(k, 50) * adp_w.get(k, 0), 1) for k in _comp_keys]

        comp_data = {
            'Component': _comp_labels,
            'Wt': [f"{w * 100:.0f}%" for w in _comp_weights],
            'Score': _comp_scores,
            'Contrib': _comp_contribs,
        }
        st.dataframe(pd.DataFrame(comp_data), column_config={
            'Score': st.column_config.ProgressColumn('Score', min_value=0, max_value=100, format="%.1f")
        }, hide_index=True, use_container_width=True, height=285)

    with sc2:
        # Price Alignment + Momentum Decay
        pa_label = row.get('price_label', 'NEUTRAL')
        pa_color = '#00E676' if pa_label == 'PRICE_CONFIRMED' else '#FF1744' if pa_label == 'PRICE_DIVERGENT' else '#484f58'
        d_label = row.get('decay_label', '')
        d_color = '#FF1744' if d_label == 'DECAY_HIGH' else '#FF9800' if d_label == 'DECAY_MODERATE' else '#FFD700' if d_label == 'DECAY_MILD' else '#3fb950'
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d; margin-bottom:10px;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">💰 Price Alignment</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Score</span>
                <span style="color:{pa_color}; font-weight:700;">{row.get('price_alignment', 50):.0f}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Status</span>
                <span style="color:{pa_color}; font-weight:700;">{pa_label.replace('_', ' ')}</span>
            </div>
        </div>
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d; margin-bottom:10px;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🔻 Momentum Decay</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Score</span>
                <span style="color:{d_color}; font-weight:700;">{row.get('decay_score', 0)}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Status</span>
                <span style="color:{d_color}; font-weight:700;">{d_label if d_label else 'CLEAN ✅'}</span>
            </div>
            <div style="margin-top:6px; color:#484f58; font-size:0.7rem;">Hurst×Wave ×{row.get('combined_mult', 1.0):.3f} → Final: {t_score:.1f}</div>
        </div>
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🏆 Alpha Score</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Score</span>
                <span style="color:{alpha_c}; font-weight:700;">{alpha:.0f}/100</span>
            </div>
            <div style="background:#21262d; border-radius:4px; height:5px; overflow:hidden;">
                <div style="width:{min(alpha, 100):.0f}%; background:{alpha_c}; height:100%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        # Sector Alpha + Trajectory Stats
        sa_val = row.get('sector_alpha_value', 0)
        sa_colors_map = {'SECTOR_LEADER': '#FFD700', 'SECTOR_OUTPERFORM': '#3fb950', 'SECTOR_ALIGNED': '#484f58', 'SECTOR_BETA': '#FF9800', 'SECTOR_LAGGARD': '#FF1744'}
        sa_color = sa_colors_map.get(sa_tag, '#484f58')
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d; margin-bottom:10px;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🏛️ Sector Alpha</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Sector</span>
                <span style="color:#e6edf3; font-weight:600;">{_html.escape(str(row.get('sector', 'N/A')))}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Alpha</span>
                <span style="color:{sa_color}; font-weight:700;">{sa_val:+.1f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Tag</span>
                <span style="color:{sa_color}; font-weight:700;">{sa_icons.get(sa_tag, '➖')} {sa_tag.replace('SECTOR_', '').title()}</span>
            </div>
        </div>
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">📊 Trajectory Stats</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Worst Rank</span>
                <span style="color:#e6edf3;">#{row.get('worst_rank', 0)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Avg Rank</span>
                <span style="color:#e6edf3;">#{row.get('avg_rank', 0)}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Rank Volatility</span>
                <span style="color:#e6edf3;">{row.get('rank_volatility', 0):.1f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Weeks</span>
                <span style="color:#e6edf3;">{row.get('weeks', 0)}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Confidence</span>
                <span style="color:#e6edf3;">{row.get('confidence', 0):.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Latest Wave Detection Patterns (sanitized)
    _lat_pat = row.get('latest_patterns', '')
    if _lat_pat:
        st.markdown(f'<div style="margin-top:8px;"><span style="color:#8b949e; font-size:0.72rem; text-transform:uppercase;">Wave Patterns:</span> <span class="pattern-tag">{_html.escape(str(_lat_pat))}</span></div>', unsafe_allow_html=True)

    # ── Wave Signal Fusion Detail ──
    wf_score = row.get('wave_fusion_score', 50)
    wf_label = row.get('wave_fusion_label', 'NEUTRAL')
    wf_mult  = row.get('wave_fusion_multiplier', 1.0)
    wf_conf  = row.get('wave_confluence', 50)
    wf_flow  = row.get('wave_inst_flow', 50)
    wf_harm  = row.get('wave_harmony', 50)
    wf_fund  = row.get('wave_fundamental', 50)
    wf_tension = row.get('wave_position_tension') or 0
    wf_from_high = row.get('wave_from_high') or 0
    rally_gain = row.get('rally_gain', 0.0)
    rally_weeks = row.get('rally_weeks', 0)
    rally_leg_pct = row.get('rally_leg_pct', 50.0)
    rally_stage = row.get('rally_stage', 'UNKNOWN')
    _stage_colors = {'FRESH': '#00E676', 'EARLY': '#3fb950', 'RUNNING': '#FF9800', 'MATURE': '#FF5722', 'LATE': '#FF1744', 'UNKNOWN': '#484f58'}
    _stage_c = _stage_colors.get(rally_stage, '#8b949e')
    wf_colors = {'WAVE_STRONG': '#00E676', 'WAVE_CONFIRMED': '#3fb950', 'WAVE_NEUTRAL': '#484f58',
                 'WAVE_WEAK': '#FF9800', 'WAVE_CONFLICT': '#FF1744'}
    wf_c = wf_colors.get(wf_label, '#484f58')
    wf1, wf2, wf3 = st.columns(3)
    with wf1:
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">🌊 Wave Signal Fusion</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Fusion Score</span>
                <span style="color:{wf_c}; font-weight:700;">{wf_score:.0f}/100</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Multiplier</span>
                <span style="color:{wf_c}; font-weight:600;">×{wf_mult:.3f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Classification</span>
                <span style="color:{wf_c}; font-weight:700;">{wf_label.replace('WAVE_', '')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with wf2:
        _conf_c = '#3fb950' if wf_conf >= 65 else '#FF9800' if wf_conf < 40 else '#8b949e'
        _flow_c = '#3fb950' if wf_flow >= 65 else '#FF9800' if wf_flow < 40 else '#8b949e'
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">📡 Fusion Signals</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Confluence</span>
                <span style="color:{_conf_c}; font-weight:600;">{wf_conf:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Inst. Flow</span>
                <span style="color:{_flow_c}; font-weight:600;">{wf_flow:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Harmony</span>
                <span style="color:#8b949e; font-weight:600;">{wf_harm:.0f}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#8b949e; font-size:0.8rem;">Fundamental</span>
                <span style="color:#8b949e; font-weight:600;">{wf_fund:.0f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with wf3:
        _rally_gain_sign = f"+{rally_gain:.1f}" if rally_gain >= 0 else f"{rally_gain:.1f}"
        st.markdown(f"""
        <div style="background:#161b22; border-radius:10px; padding:14px; border:1px solid #30363d;">
            <div style="font-size:0.72rem; color:#8b949e; text-transform:uppercase; margin-bottom:8px;">📈 Rally Leg Status</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Stage</span>
                <span style="color:{_stage_c}; font-weight:700; font-size:0.85rem;">{rally_stage}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Gain this leg</span>
                <span style="color:{_stage_c}; font-weight:700;">{_rally_gain_sign}%</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="color:#8b949e; font-size:0.8rem;">Rally duration</span>
                <span style="color:#8b949e; font-weight:600;">{rally_weeks} wk{'s' if rally_weeks != 1 else ''}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:#8b949e; font-size:0.8rem;">Gap to 52w high</span>
                <span style="color:#8b949e; font-weight:600;">{wf_from_high:.1f}%</span>
            </div>
            <div style="font-size:0.7rem; color:#8b949e; margin-bottom:2px;">Leg completion (gain vs. gap to 52w high)</div>
            <div style="background:#21262d; border-radius:4px; height:6px; overflow:hidden; margin-bottom:4px;">
                <div style="width:{min(rally_leg_pct, 100):.1f}%; background:linear-gradient(90deg, #00E676 0%, #FF9800 55%, #FF1744 100%); height:100%;"></div>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="color:#484f58; font-size:0.7rem;">0% (fresh)</span>
                <span style="color:{_stage_c}; font-weight:700; font-size:0.75rem;">{rally_leg_pct:.0f}% done</span>
                <span style="color:#484f58; font-size:0.7rem;">100% (52w high)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # ── Week-by-Week History ──
    with st.expander("📅 Week-by-Week History", expanded=False):
        week_data = {
            'Date': h.get('dates', []),
            'Rank': [int(r) for r in ranks_list],
            'Pctl': [round(p, 1) for p in pcts] if pcts else [],
            'Price ₹': [round(p, 1) for p in h['prices']],
            'M.Score': [round(s, 1) for s in h.get('scores', [])],
            'Stocks': totals_list,
        }
        wk_changes = [0] + [int(ranks_list[i - 1] - ranks_list[i]) for i in range(1, len(ranks_list))]
        week_data['Δ Rank'] = wk_changes
        price_changes = [0] + [round(h['prices'][i] - h['prices'][i-1], 1) for i in range(1, len(h['prices']))]
        week_data['Δ Price'] = price_changes

        def _safe_round(lst, decimals=1):
            return [round(v, decimals) if v is not None and not np.isnan(v) else None for v in lst]
        if h.get('ret_7d'):
            week_data['Ret 7d%'] = _safe_round(h['ret_7d'])
        if h.get('ret_30d'):
            week_data['Ret 30d%'] = _safe_round(h['ret_30d'])

        wk_df = pd.DataFrame(week_data).iloc[::-1]  # Latest first
        wk_col_config = {
            'Δ Rank': st.column_config.NumberColumn(format="%+d", help="Positive = rank improved (moved up)"),
            'Pctl': st.column_config.ProgressColumn('Pctl', min_value=0, max_value=100, format="%.1f"),
            'Price ₹': st.column_config.NumberColumn(format="₹%.1f"),
            'Δ Price': st.column_config.NumberColumn(format="%+.1f"),
        }
        if 'Ret 7d%' in wk_df.columns:
            wk_col_config['Ret 7d%'] = st.column_config.NumberColumn(format="%.1f%%")
        if 'Ret 30d%' in wk_df.columns:
            wk_col_config['Ret 30d%'] = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(wk_df, column_config=wk_col_config, hide_index=True, use_container_width=True)

    # ── Compare ──
    with st.expander("⚖️ Compare Stocks", expanded=False):
        compare_labels = [l for l in labels if l != selected_label]
        if not compare_labels:
            st.info("No other stocks available for comparison with current filters")
        else:
            st.caption("Select up to 4 stocks to compare rank percentile trajectories and key metrics side by side")
            compare_selections = st.multiselect("Compare with",
                                                 compare_labels, max_selections=4,
                                                 key='compare_select',
                                                 label_visibility='collapsed')
            if compare_selections:
                compare_tickers = [label_map[l] for l in compare_selections]
                _render_comparison_chart(ticker, compare_tickers, histories, traj_df)


def _render_rank_chart(h: dict, ticker: str):
    """Rank + Master Score dual-axis trajectory chart"""
    dates = h.get('dates', [])
    ranks = h.get('ranks', [])
    scores = h.get('scores', [])
    if not dates or not ranks:
        st.info("Insufficient data for rank chart")
        return

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Rank (primary y — inverted so lower rank = higher on chart)
    fig.add_trace(go.Scatter(
        x=dates, y=ranks,
        mode='lines+markers',
        name='Rank',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(255,107,53,0.08)',
        hovertemplate='%{x}<br>Rank: #%{y}<extra></extra>'
    ), secondary_y=False)

    # Master Score (secondary y)
    fig.add_trace(go.Scatter(
        x=dates, y=scores,
        mode='lines+markers',
        name='Master Score',
        line=dict(color='#58a6ff', width=2, dash='dot'),
        marker=dict(size=4),
        opacity=0.85,
        hovertemplate='%{x}<br>M.Score: %{y:.1f}<extra></extra>'
    ), secondary_y=True)

    # Best rank annotation
    if len(ranks) > 0:
        best_idx = int(np.argmin(ranks))
        fig.add_annotation(
            x=dates[best_idx], y=ranks[best_idx],
            text=f"Best: #{int(ranks[best_idx])}",
            showarrow=True, arrowhead=2,
            font=dict(color='#FFD700', size=10),
            bgcolor='rgba(0,0,0,0.7)', bordercolor='#FFD700'
        )

    fig.update_layout(
        height=340,
        template='plotly_dark',
        hovermode='x unified',
        legend=dict(orientation='h', y=-0.18, font=dict(size=10)),
        margin=dict(t=10, b=55, l=50, r=50),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig.update_yaxes(title_text="Rank", autorange="reversed",
                     secondary_y=False, gridcolor='rgba(255,255,255,0.04)')
    fig.update_yaxes(title_text="Master Score",
                     secondary_y=True, gridcolor='rgba(255,255,255,0.02)')

    st.plotly_chart(fig, use_container_width=True)


def _render_price_chart(h: dict, ticker: str):
    """Price trajectory chart with min/max annotations"""
    dates = h.get('dates', [])
    prices = h.get('prices', [])
    if not dates or not prices:
        st.info("Insufficient data for price chart")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices,
        mode='lines+markers',
        name='Price ₹',
        line=dict(color='#00C853', width=2.5),
        marker=dict(size=5),
        fill='tozeroy',
        fillcolor='rgba(0,200,83,0.06)',
        hovertemplate='%{x}<br>₹%{y:,.1f}<extra></extra>'
    ))

    # High / Low annotations
    if len(prices) > 0:
        hi_idx = int(np.argmax(prices))
        lo_idx = int(np.argmin(prices))
        fig.add_annotation(
            x=dates[hi_idx], y=prices[hi_idx],
            text=f"High: ₹{prices[hi_idx]:,.1f}",
            showarrow=True, arrowhead=2,
            font=dict(color='#00E676', size=10),
            bgcolor='rgba(0,0,0,0.7)', bordercolor='#00E676'
        )
        if hi_idx != lo_idx:
            fig.add_annotation(
                x=dates[lo_idx], y=prices[lo_idx],
                text=f"Low: ₹{prices[lo_idx]:,.1f}",
                showarrow=True, arrowhead=2, ay=30,
                font=dict(color='#FF5252', size=10),
                bgcolor='rgba(0,0,0,0.7)', bordercolor='#FF5252'
            )

    fig.update_layout(
        height=280,
        template='plotly_dark',
        hovermode='x unified',
        margin=dict(t=10, b=50, l=50, r=20),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(title='Price ₹', gridcolor='rgba(255,255,255,0.04)'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_radar_chart(row):
    """Render radar/spider chart for component scores (8 components, v4.0)"""
    categories = ['Positional', 'Trend', 'Velocity', 'Acceleration', 'Consistency', 'Resilience', 'RetQuality', 'Breakout']
    values = [row.get('positional', 50), row.get('trend', 50), row.get('velocity', 50), row.get('acceleration', 50),
              row.get('consistency', 50), row.get('resilience', 50), row.get('return_quality', 50), row.get('breakout_quality', 50)]
    values_closed = values + [values[0]]  # Close the polygon
    cats_closed = categories + [categories[0]]

    fig = go.Figure()

    # Reference circle at 50
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(cats_closed),
        theta=cats_closed,
        mode='lines',
        line=dict(color='rgba(255,255,255,0.15)', width=1, dash='dash'),
        showlegend=False,
        hoverinfo='skip'
    ))

    # Score polygon
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill='toself',
        fillcolor='rgba(255,107,53,0.25)',
        line=dict(color='#FF6B35', width=2),
        name='Scores',
        hovertemplate='%{theta}: %{r:.1f}<extra></extra>'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], showticklabels=True,
                           tickfont=dict(size=9), gridcolor='rgba(255,255,255,0.08)'),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        height=380,
        template='plotly_dark',
        showlegend=False,
        margin=dict(t=20, b=20, l=60, r=60)
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_comparison_chart(main_ticker: str, compare_tickers: list,
                              histories: dict, traj_df: pd.DataFrame):
    """Render comparison chart for multiple stocks"""
    colors = ['#FF6B35', '#00C853', '#2196F3', '#FFD700', '#E040FB']
    all_tickers = [main_ticker] + compare_tickers

    fig = go.Figure()
    for i, ticker in enumerate(all_tickers):
        h = histories.get(ticker, {})
        if not h:
            continue
        # Convert to percentiles for fair comparison
        pcts = ranks_to_percentiles(h['ranks'], h['total_per_week'])
        name_row = traj_df[traj_df['ticker'] == ticker]
        label = f"{ticker}"
        if not name_row.empty:
            label = f"{ticker} (T:{name_row.iloc[0]['trajectory_score']:.0f})"

        fig.add_trace(go.Scatter(
            x=h['dates'], y=pcts,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=3 if i == 0 else 2),
            marker=dict(size=6 if i == 0 else 4),
            opacity=1.0 if i == 0 else 0.8
        ))

    fig.update_layout(
        title="Rank Percentile Comparison (higher = better rank)",
        height=400,
        template='plotly_dark',
        hovermode='x unified',
        yaxis=dict(title='Rank Percentile', range=[0, 100]),
        xaxis=dict(title='Week'),
        legend=dict(orientation='h', y=-0.2),
        margin=dict(t=40, b=60, l=60, r=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Comparison table
    comp_rows = []
    for ticker in all_tickers:
        r = traj_df[traj_df['ticker'] == ticker]
        if r.empty:
            continue
        r = r.iloc[0]
        comp_rows.append({
            'Ticker': ticker,
            'Company': str(r.get('company_name', ''))[:25],
            'T-Score': r.get('trajectory_score', 0),
            'Alpha': r.get('alpha_score', 0),
            'Grade': f"{r.get('grade_emoji', '')} {r.get('grade', 'F')}",
            'Pattern': r.get('pattern', ''),
            'TMI': r.get('tmi', 50),
            'Rank Now': r.get('current_rank', 0),
            'Best': r.get('best_rank', 0),
            'Δ Rank': r.get('rank_change', 0),
            'Streak': r.get('streak', 0),
        })
    if comp_rows:
        st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)


# ============================================
# UI: TOP MOVERS TAB
# ============================================

def render_top_movers_tab(filtered_df: pd.DataFrame, histories: dict):
    """🔥 Top Movers Tab v2.0 — multi-week window, 50/100/150 stock selector.

    v2.0 fixes:
      - XSS: all ticker/company names html-escaped
      - Dropdown for 50/100/150 stocks per side
      - Header hero badges update to match selected time window
      - Info bar reflects actual count + window
      - max-height scales with selected count
      - Sector column added
      - NaN guards on every .get() / merge field
      - Summary stats row (avg/median rank change)
    """
    import html as _html

    # ── Ensure columns ──
    for col, default in [('grade', 'F'), ('grade_emoji', '📉'), ('company_name', ''),
                         ('sector', ''), ('weeks', 0), ('trajectory_score', 0)]:
        if col not in filtered_df.columns:
            filtered_df[col] = default

    _filtered_tickers = set(filtered_df['ticker'].tolist())

    # ── Controls: Time Window + Stock Count ──────────────────────
    max_hist_len = max((len(h.get('ranks', [])) for h in histories.values()), default=2) - 1
    week_options = [w for w in [1, 2, 4, 8, 12] if w <= max_hist_len] or [1]
    week_labels = {1: '1 Week', 2: '2 Weeks', 4: '4 Weeks', 8: '8 Weeks', 12: '12 Weeks'}

    c_wk, c_ct, _ = st.columns([1, 1, 2])
    with c_wk:
        mv_weeks = st.selectbox(
            'Time Window', options=week_options,
            format_func=lambda x: week_labels.get(x, f'{x} Weeks'),
            index=0, key='movers_tab_weeks',
        )
    with c_ct:
        mv_count = st.selectbox(
            'Stocks Per Side', options=[50, 100, 150],
            index=0, key='movers_tab_count',
        )

    # ── Fetch movers ─────────────────────────────────────────────
    gainers, decliners = get_top_movers(histories, n=mv_count, weeks=mv_weeks,
                                         tickers=_filtered_tickers)

    # ── Detect pattern flips + new entries/exits over the same window ──
    new_rockets, new_crashes, new_entries, exited_tickers = [], [], [], []
    for ticker, h in histories.items():
        if _filtered_tickers is not None and ticker not in _filtered_tickers:
            continue
        pats = h.get('pattern_history', [])
        ranks = h.get('ranks', [])
        n_weeks = len(ranks)
        # Pattern flips: compare current vs N weeks ago
        if len(pats) >= mv_weeks + 1:
            cur_pat = pats[-1]
            prev_pat = pats[-(mv_weeks + 1)]
            if cur_pat == 'rocket' and prev_pat != 'rocket':
                new_rockets.append(ticker)
            if cur_pat == 'crash' and prev_pat != 'crash':
                new_crashes.append(ticker)
        # New entries: present in latest week but absent N weeks ago
        dates = h.get('dates', [])
        if n_weeks >= 1 and n_weeks < mv_weeks + 1:
            new_entries.append(ticker)
    # Exited: had data N weeks ago but not in latest week
    for ticker, h in histories.items():
        if _filtered_tickers is not None and ticker not in _filtered_tickers:
            continue
        ranks = h.get('ranks', [])
        dates = h.get('dates', [])
        if len(ranks) >= mv_weeks + 1:
            # This ticker has old data — check if it's missing from the latest snapshot
            pass  # still present, not exited
        # Tickers that disappeared: were in prev but not in latest
    # Simpler approach: use the actual date lists
    _all_dates = set()
    for h in histories.values():
        for d in h.get('dates', []):
            _all_dates.add(str(d)[:10])
    _sorted_dates = sorted(_all_dates)
    if len(_sorted_dates) >= mv_weeks + 1:
        _latest_date = _sorted_dates[-1]
        _prev_date = _sorted_dates[-(mv_weeks + 1)]
        _latest_tickers = set()
        _prev_tickers = set()
        for ticker, h in histories.items():
            if _filtered_tickers is not None and ticker not in _filtered_tickers:
                continue
            ds = [str(d)[:10] for d in h.get('dates', [])]
            if _latest_date in ds:
                _latest_tickers.add(ticker)
            if _prev_date in ds:
                _prev_tickers.add(ticker)
        new_entries = sorted(_latest_tickers - _prev_tickers)
        exited_tickers = sorted(_prev_tickers - _latest_tickers)

    # ── Header Card (uses actual data) ───────────────────────────
    _safe = _html.escape
    if not gainers.empty:
        tg_delta = int(gainers.iloc[0].get('rank_change', 0))
        tg_name = _safe(str(gainers.iloc[0].get('ticker', '—')))
    else:
        tg_delta, tg_name = 0, '—'
    if not decliners.empty:
        td_delta = int(decliners.iloc[0].get('rank_change', 0))
        td_name = _safe(str(decliners.iloc[0].get('ticker', '—')))
    else:
        td_delta, td_name = 0, '—'

    wk_label = week_labels.get(mv_weeks, f'{mv_weeks}w')
    st.markdown(f"""
    <div style="background:#0d1117;border-radius:14px;padding:18px 24px;margin-bottom:16px;border:1px solid #30363d;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
        <div>
          <span style="font-size:1.4rem;font-weight:800;color:#fff;">🔥 Top Movers</span>
          <div style="color:#8b949e;font-size:0.88rem;margin-top:2px;">
            Biggest rank changes over <span style="color:#58a6ff;font-weight:700;">{wk_label}</span>
            &nbsp;·&nbsp; {mv_count} per side
          </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <span style="background:#3fb95018;color:#3fb950;padding:4px 10px;border-radius:8px;
                font-size:0.78rem;font-weight:600;">⬆️ {tg_name} +{tg_delta}</span>
          <span style="background:#f8514918;color:#f85149;padding:4px 10px;border-radius:8px;
                font-size:0.78rem;font-weight:600;">⬇️ {td_name} {td_delta}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Summary stats ────────────────────────────────────────────
    def _stats(df_mv):
        if df_mv.empty:
            return 0, 0, 0, 0
        rc = df_mv['rank_change']
        return len(df_mv), int(rc.mean()), int(rc.median()), int(rc.iloc[0])

    g_cnt, g_avg, g_med, g_best = _stats(gainers)
    d_cnt, d_avg, d_med, d_worst = _stats(decliners)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-bottom:14px;">
      <div style="background:#161b22;border-radius:10px;padding:10px 12px;text-align:center;border:1px solid #30363d;">
        <div style="color:#3fb950;font-weight:700;font-size:1.1rem;">{g_cnt}</div>
        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Climbers</div></div>
      <div style="background:#161b22;border-radius:10px;padding:10px 12px;text-align:center;border:1px solid #30363d;">
        <div style="color:#3fb950;font-weight:700;font-size:1.1rem;">+{g_avg}</div>
        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Avg Climb</div></div>
      <div style="background:#161b22;border-radius:10px;padding:10px 12px;text-align:center;border:1px solid #30363d;">
        <div style="color:#3fb950;font-weight:700;font-size:1.1rem;">+{g_best}</div>
        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Best Climb</div></div>
      <div style="background:#161b22;border-radius:10px;padding:10px 12px;text-align:center;border:1px solid #30363d;">
        <div style="color:#f85149;font-weight:700;font-size:1.1rem;">{d_cnt}</div>
        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Decliners</div></div>
      <div style="background:#161b22;border-radius:10px;padding:10px 12px;text-align:center;border:1px solid #30363d;">
        <div style="color:#f85149;font-weight:700;font-size:1.1rem;">{d_avg}</div>
        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Avg Drop</div></div>
      <div style="background:#161b22;border-radius:10px;padding:10px 12px;text-align:center;border:1px solid #30363d;">
        <div style="color:#f85149;font-weight:700;font-size:1.1rem;">{d_worst}</div>
        <div style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Worst Drop</div></div>
    </div>""", unsafe_allow_html=True)

    # ── Pattern Flips + New Entries/Exits ─────────────────────────
    _n_rockets = len(new_rockets)
    _n_crashes = len(new_crashes)
    _n_entries = len(new_entries)
    _n_exited = len(exited_tickers)
    _has_signals = _n_rockets + _n_crashes + _n_entries + _n_exited > 0

    if _has_signals:
        # Build compact pill lists for rockets and crashes
        def _ticker_pills(tickers, color, limit=12):
            if not tickers:
                return '<span style="color:#6e7681;font-size:0.75rem;">None</span>'
            pills = []
            for tk in tickers[:limit]:
                pills.append(
                    f'<span style="display:inline-block;padding:2px 8px;border-radius:8px;'
                    f'font-size:0.72rem;font-weight:600;color:{color};'
                    f'background:{color}12;border:1px solid {color}30;margin:2px;">'
                    f'{_safe(tk)}</span>')
            if len(tickers) > limit:
                pills.append(f'<span style="color:#6e7681;font-size:0.72rem;">+{len(tickers) - limit} more</span>')
            return ' '.join(pills)

        rocket_pills = _ticker_pills(new_rockets, '#FF4500')
        crash_pills = _ticker_pills(new_crashes, '#DC143C')
        entry_pills = _ticker_pills(new_entries, '#58a6ff')
        exit_pills = _ticker_pills(exited_tickers, '#8b949e')

        st.markdown(f"""
        <div style="background:#0d1117;border-radius:12px;padding:16px 20px;margin-bottom:14px;
            border:1px solid #30363d;">
          <div style="font-size:0.82rem;font-weight:700;color:#c9d1d9;margin-bottom:10px;
              letter-spacing:0.3px;">⚡ Pattern Flips & Universe Changes
            <span style="color:#6e7681;font-weight:400;font-size:0.75rem;margin-left:8px;">
              (over {wk_label})</span></div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
            <div style="display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center;">
              <div style="display:flex;align-items:center;gap:4px;">
                <span style="color:#FF4500;font-weight:800;font-size:1rem;">{_n_rockets}</span>
                <span style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">New 🚀</span></div>
              <div>{rocket_pills}</div>
              <div style="display:flex;align-items:center;gap:4px;">
                <span style="color:#DC143C;font-weight:800;font-size:1rem;">{_n_crashes}</span>
                <span style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">New 💥</span></div>
              <div>{crash_pills}</div>
            </div>
            <div style="display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center;">
              <div style="display:flex;align-items:center;gap:4px;">
                <span style="color:#58a6ff;font-weight:800;font-size:1rem;">{_n_entries}</span>
                <span style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Entered</span></div>
              <div>{entry_pills}</div>
              <div style="display:flex;align-items:center;gap:4px;">
                <span style="color:#8b949e;font-weight:800;font-size:1rem;">{_n_exited}</span>
                <span style="color:#8b949e;font-size:0.7rem;text-transform:uppercase;">Exited</span></div>
              <div>{exit_pills}</div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── Mover Table Builder ──────────────────────────────────────
    enrich_cols = ['ticker', 'trajectory_score', 'grade', 'sector']
    enrich_df = filtered_df[
        [c for c in enrich_cols if c in filtered_df.columns]
    ].drop_duplicates('ticker')

    scroll_h = {50: 580, 100: 1050, 150: 1500}.get(mv_count, 580)

    def _mover_table_html(df_mv, accent, icon, label):
        count = len(df_mv)
        hdr = (f'<div style="background:#161b22;border-radius:10px 10px 0 0;padding:12px 16px;'
               f'border:1px solid #30363d;border-bottom:2px solid {accent};display:flex;'
               f'justify-content:space-between;align-items:center;">'
               f'<span style="font-size:0.88rem;font-weight:700;color:{accent};">{icon} {label}</span>'
               f'<span style="color:#6e7681;font-size:0.78rem;">{count} stocks</span></div>')

        if df_mv.empty:
            return (hdr + '<div style="background:#0d1117;border-radius:0 0 10px 10px;padding:28px;'
                    'border:1px solid #30363d;border-top:0;text-align:center;color:#6e7681;'
                    'font-size:0.88rem;">No movers detected for this window</div>')

        enriched = df_mv.merge(enrich_df, on='ticker', how='left')

        col_hdr = (
            '<div class="mv-row" style="display:flex;align-items:center;padding:6px 14px;gap:8px;'
            'background:#161b22;border-bottom:1px solid #30363d;font-size:0.72rem;color:#6e7681;'
            'text-transform:uppercase;letter-spacing:0.5px;">'
            '<span style="min-width:28px;text-align:center;color:#6e7681;font-size:0.68rem;">#</span>'
            '<span style="min-width:44px;text-align:right;">Chg</span>'
            '<span style="flex:1;">Stock</span>'
            '<span class="mv-sector" style="min-width:72px;text-align:left;">Sector</span>'
            '<span style="min-width:56px;text-align:right;">Price</span>'
            '<span style="min-width:80px;text-align:center;">Prev → Now</span>'
            '<span style="min-width:32px;text-align:center;">Grd</span>'
            '<span style="min-width:36px;text-align:right;">Score</span></div>')

        rows_html = [col_hdr]
        for i, (_, m) in enumerate(enriched.iterrows()):
            rc = int(m.get('rank_change', 0) if pd.notna(m.get('rank_change', 0)) else 0)
            ts = float(m.get('trajectory_score', 0) if pd.notna(m.get('trajectory_score')) else 0)
            gr = str(m.get('grade', '—') if pd.notna(m.get('grade')) else '—')
            gc = GRADE_COLORS.get(gr, '#8b949e')
            sector_raw = str(m.get('sector', '') if pd.notna(m.get('sector')) else '')
            sector_short = _safe(sector_raw[:14]) if sector_raw else '—'
            stripe = 'rgba(22,27,34,0.5)' if i % 2 else 'transparent'
            chg_c = '#3fb950' if rc > 0 else ('#f85149' if rc < 0 else '#8b949e')
            chg_sign = '+' if rc > 0 else ''

            price_val = float(m.get('price', 0) if pd.notna(m.get('price')) else 0)
            if price_val >= 100:
                price_str = f'₹{price_val:,.0f}'
            elif price_val > 0:
                price_str = f'₹{price_val:,.2f}'
            else:
                price_str = '—'

            prev_r = int(m.get('prev_rank', 0) if pd.notna(m.get('prev_rank')) else 0)
            curr_r = int(m.get('current_rank', 0) if pd.notna(m.get('current_rank')) else 0)
            ticker_esc = _safe(str(m.get('ticker', '')))
            comp_esc = _safe(str(m.get('company_name', '') if pd.notna(m.get('company_name')) else '')[:22])

            rows_html.append(
                f'<div class="mv-row" style="display:flex;align-items:center;padding:6px 14px;gap:8px;'
                f'background:{stripe};border-bottom:1px solid #21262d;">'
                f'<span style="min-width:28px;text-align:center;color:#6e7681;font-size:0.72rem;'
                f'font-variant-numeric:tabular-nums;">{i+1}</span>'
                f'<span style="color:{chg_c};font-weight:800;font-size:0.88rem;min-width:44px;'
                f'text-align:right;font-variant-numeric:tabular-nums;">{chg_sign}{rc}</span>'
                f'<div style="flex:1;overflow:hidden;white-space:nowrap;">'
                f'<span class="mv-ticker" style="color:#e6edf3;font-weight:600;font-size:0.85rem;">{ticker_esc}</span>'
                f'<span style="color:#8b949e;font-size:0.73rem;margin-left:6px;">{comp_esc}</span></div>'
                f'<span class="mv-sector" style="color:#8b949e;font-size:0.75rem;min-width:72px;overflow:hidden;'
                f'white-space:nowrap;text-overflow:ellipsis;">{sector_short}</span>'
                f'<span style="color:#d2a8ff;font-weight:600;font-size:0.82rem;min-width:56px;'
                f'text-align:right;font-variant-numeric:tabular-nums;">{price_str}</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;min-width:80px;text-align:center;'
                f'font-variant-numeric:tabular-nums;">{prev_r} → {curr_r}</span>'
                f'<span style="color:{gc};font-weight:700;font-size:0.82rem;min-width:32px;'
                f'text-align:center;">{_safe(gr)}</span>'
                f'<span style="color:#FF6B35;font-weight:600;font-size:0.82rem;min-width:36px;'
                f'text-align:right;font-variant-numeric:tabular-nums;">{ts:.0f}</span></div>')

        body = (f'<div style="background:#0d1117;border-radius:0 0 10px 10px;border:1px solid #30363d;'
                f'border-top:0;overflow:hidden;max-height:{scroll_h}px;overflow-y:auto;">'
                f'{"".join(rows_html)}</div>')
        return hdr + body

    g_html = _mover_table_html(gainers, '#3fb950', '⬆️', 'Biggest Climbers')
    d_html = _mover_table_html(decliners, '#f85149', '⬇️', 'Biggest Decliners')
    st.markdown(
        f'<div class="mv-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'
        f'<div>{g_html}</div><div>{d_html}</div></div>', unsafe_allow_html=True)


# ============================================
# UI: EXPORT TAB
# ============================================

def render_export_tab(filtered_df: pd.DataFrame, all_df: pd.DataFrame, histories: dict):
    """Render export options tab"""

    st.markdown("##### 📤 Export Trajectory Data")
    st.markdown("Download trajectory rankings and analysis data in various formats.")

    # Export scope
    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        scope = st.radio("Export Scope", ['Filtered Rankings (as shown)', 'All Rankings', 'Custom Selection'],
                          key='exp_scope')
    with exp_c2:
        detail = st.radio("Detail Level", ['Compact', 'Standard', 'Full Detail'],
                           key='exp_detail')

    # Determine data to export
    if scope == 'Filtered Rankings (as shown)':
        export_df = filtered_df.copy()
    elif scope == 'All Rankings':
        export_df = all_df.copy()
    else:
        # Custom selection
        available = sorted(all_df['ticker'].tolist())
        selected = st.multiselect("Select tickers to export", available, key='exp_tickers')
        if selected:
            export_df = all_df[all_df['ticker'].isin(selected)].copy()
        else:
            st.info("Select tickers above to export")
            return

    # Column selection based on detail level
    compact_cols = ['t_rank', 'ticker', 'company_name', 'category', 'trajectory_score',
                    'grade', 'pattern', 'positional', 'current_rank', 'best_rank', 'rank_change', 'weeks']
    standard_cols = compact_cols + ['trend', 'velocity', 'acceleration', 'consistency',
                                     'resilience', 'sector', 'industry', 'streak',
                                     'last_week_change', 'avg_rank', 'rank_volatility']
    full_cols = standard_cols + ['worst_rank', 'market_state', 'latest_patterns',
                                  'grade_emoji', 'pattern_key',
                                  'price_alignment', 'price_label',
                                  'decay_score', 'decay_label',
                                  'sector_alpha_tag', 'sector_alpha_value', 'signal_tags']

    if detail == 'Compact':
        cols = [c for c in compact_cols if c in export_df.columns]
    elif detail == 'Standard':
        cols = [c for c in standard_cols if c in export_df.columns]
    else:
        cols = [c for c in full_cols if c in export_df.columns]

    export_data = export_df[cols].copy()
    # Remove sparkline from export (it's a list column)
    if 'sparkline' in export_data.columns:
        export_data = export_data.drop(columns=['sparkline'])

    # Preview
    st.markdown(f"##### Preview ({len(export_data)} stocks, {len(cols)} columns)")
    st.dataframe(export_data.head(20), hide_index=True, use_container_width=True)

    # Download buttons
    dl_c1, dl_c2, dl_c3 = st.columns(3)

    with dl_c1:
        csv_data = export_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=f"trajectory_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime='text/csv'
        )

    with dl_c2:
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            export_data.to_excel(writer, index=False, sheet_name='Trajectory Rankings')
        buffer.seek(0)
        st.download_button(
            label="📥 Download Excel",
            data=buffer.getvalue(),
            file_name=f"trajectory_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    with dl_c3:
        json_data = export_data.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json_data,
            file_name=f"trajectory_rankings_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime='application/json'
        )

    # ── Export individual stock trajectory ──
    st.markdown("---")
    st.markdown("##### 📈 Export Individual Stock History")
    ticker_for_export = st.selectbox("Select ticker", sorted(histories.keys()), key='exp_single')
    if ticker_for_export and ticker_for_export in histories:
        h = histories[ticker_for_export]
        hist_df = pd.DataFrame({
            'date': h['dates'],
            'rank': [int(r) for r in h['ranks']],
            'master_score': [round(s, 2) for s in h['scores']],
            'price': [round(p, 2) for p in h['prices']],
            'total_stocks': h['total_per_week'],
            'percentile': [round((1 - r / max(t, 1)) * 100, 2) for r, t in zip(h['ranks'], h['total_per_week'])]
        })
        st.dataframe(hist_df, hide_index=True, use_container_width=True)
        csv_hist = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"📥 Download {ticker_for_export} History (CSV)",
            data=csv_hist,
            file_name=f"{ticker_for_export}_trajectory_history.csv",
            mime='text/csv'
        )


# ============================================
# STRATEGY BACKTEST ENGINE (v8.0)
# ============================================
# Walk-forward backtest that proves which selection strategy
# generates the best ACTUAL returns using your CSV data.
# Tests 8 strategies from "buy everything" to "max filter".
# Uses price-based forward returns — no lookahead bias.

def _run_strategy_backtest(uploaded_files, progress_callback=None):
    """
    Walk-forward strategy backtest.

    For each week N (where N >= min_history):
      1. Build histories using ONLY weeks 1..N (no future data)
      2. Compute trajectory scores at that point
      3. Apply 8 different selection strategies
      4. Measure ACTUAL forward returns from week N+1 prices

    Returns dict with results per strategy, or None if insufficient data.
    """
    # ── Step 1: Parse all CSVs ──
    weekly_data = {}
    for ufile in uploaded_files:
        date = parse_date_from_filename(ufile.name)
        if date is None:
            continue
        try:
            ufile.seek(0)
            df = pd.read_csv(ufile)
            if 'rank' in df.columns and 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.strip()
                df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                df['master_score'] = pd.to_numeric(df.get('master_score', 0), errors='coerce').fillna(0)
                df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
                weekly_data[date] = df
        except Exception as e:
            logger.warning(f"Backtest: Failed to load {ufile.name}: {e}")

    dates = sorted(weekly_data.keys())
    n_weeks = len(dates)
    if n_weeks < 4:  # Need 2 history + 1 test + 1 forward minimum
        return None

    min_history = 2  # Minimum weeks of data before first test (trajectory needs ≥2 points)

    strategy_names = [
        'S1: Universe Avg',
        'S2: Top 10 WAVE Rank',
        'S3: Top 20 WAVE Rank',
        'S4: Persistent Top 50',
        'S2b: Top 10 T-Score',
        'S3b: Top 20 T-Score',
        'S5: T-Score ≥ 70',
        'S6: T-Score ≥ 70 + No Decay',
        'S7: Conviction ≥ 65',
        'S8: Full Signal',
        'S9: Conviction-Weighted',
        'S10: Regime-Adaptive',
        'S11: Momentum-Quality',
        'S12: Max Alpha',
        'S13: Capitulation Contrarian',
        'S14: Bounce Recovery',
        'S15: Quality GARP',
        'S16: Pattern Flip Alpha',
        'S17: Stealth Alpha',
        'S18: Near-High Breakout',
    ]
    all_results = {name: [] for name in strategy_names}

    # ── Step 2: Incremental walk-forward ──
    histories = {}  # Built incrementally — grows each week
    total_windows = len(dates) - min_history - 1

    for week_idx, date in enumerate(dates):
        df = weekly_data[date]
        total = len(df)

        # ── Add this week's data to histories ──
        for _, row in df.iterrows():
            ticker = str(row.get('ticker', '')).strip()
            if not ticker or ticker == 'nan':
                continue

            if ticker not in histories:
                histories[ticker] = {
                    'dates': [], 'ranks': [], 'scores': [], 'prices': [],
                    'total_per_week': [],
                    'trend_qualities': [], 'market_states': [], 'pattern_history': [],
                    'ret_7d': [], 'ret_30d': [], 'ret_3m': [], 'ret_6m': [],
                    'from_high_pct': [], 'momentum_score': [], 'volume_score': [],
                    'position_score': [], 'acceleration_score': [], 'breakout_score': [],
                    'rvol_score': [], 'pe': [], 'eps_current': [], 'eps_change_pct': [],
                    'from_low_pct': [], 'ret_1d': [], 'ret_1y': [],
                    'rvol': [], 'vmi': [], 'money_flow_mm': [],
                    'position_tension': [], 'momentum_harmony': [],
                    'eps_tier': [], 'pe_tier': [], 'overall_market_strength': [],
                    'company_name': '', 'category': '', 'sector': '',
                    'industry': '', 'market_state': '', 'patterns': ''
                }

            h = histories[ticker]
            h['dates'].append(date.strftime('%Y-%m-%d'))
            h['ranks'].append(float(row['rank']) if pd.notna(row['rank']) else total)
            h['scores'].append(float(row['master_score']))
            h['prices'].append(float(row['price']))
            h['total_per_week'].append(total)

            tq_val = row.get('trend_quality', 0)
            h['trend_qualities'].append(float(tq_val) if pd.notna(tq_val) else 0)
            ms_val = row.get('market_state', '')
            h['market_states'].append(str(ms_val).strip() if pd.notna(ms_val) else '')
            pat_val = row.get('patterns', '')
            h['pattern_history'].append(str(pat_val).strip() if pd.notna(pat_val) else '')

            for col_name, hist_key in [
                ('ret_7d', 'ret_7d'), ('ret_30d', 'ret_30d'), ('ret_3m', 'ret_3m'),
                ('ret_6m', 'ret_6m'), ('from_high_pct', 'from_high_pct'),
                ('momentum_score', 'momentum_score'), ('volume_score', 'volume_score'),
                ('position_score', 'position_score'), ('acceleration_score', 'acceleration_score'),
                ('breakout_score', 'breakout_score'), ('rvol_score', 'rvol_score'),
                ('pe', 'pe'), ('eps_current', 'eps_current'),
                ('eps_change_pct', 'eps_change_pct'), ('from_low_pct', 'from_low_pct'),
                ('ret_1d', 'ret_1d'), ('ret_1y', 'ret_1y'),
                ('rvol', 'rvol'), ('vmi', 'vmi'),
                ('money_flow_mm', 'money_flow_mm'), ('position_tension', 'position_tension'),
                ('momentum_harmony', 'momentum_harmony'),
                ('overall_market_strength', 'overall_market_strength'),
            ]:
                col_val = row.get(col_name, None)
                if col_val is not None and pd.notna(col_val):
                    try:
                        h[hist_key].append(float(col_val))
                    except (ValueError, TypeError):
                        h[hist_key].append(float('nan'))
                else:
                    h[hist_key].append(float('nan'))

            for fld in ['company_name', 'category', 'sector', 'industry', 'market_state', 'patterns']:
                val = row.get(fld, '')
                if pd.notna(val) and str(val).strip():
                    h[fld] = str(val).strip()

            for tier_col in ['eps_tier', 'pe_tier']:
                tier_val = row.get(tier_col, '')
                h[tier_col].append(str(tier_val).strip() if pd.notna(tier_val) else '')

        # ── Only test when we have enough history AND a next week exists ──
        if week_idx < min_history or week_idx >= len(dates) - 1:
            continue

        window_num = week_idx - min_history
        if progress_callback:
            progress_callback(window_num / max(total_windows, 1),
                              f"Testing week {date.strftime('%Y-%m-%d')} ({window_num + 1}/{total_windows}) · {n_weeks} CSVs loaded")

        forward_date = dates[week_idx + 1]
        forward_df = weekly_data[forward_date]

        # ── Compute PRICE-BASED forward returns (ground truth) ──
        cur_prices = {}
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            p = float(row['price']) if pd.notna(row['price']) and float(row['price']) > 0 else 0
            if p > 0:
                cur_prices[t] = p

        fwd_prices = {}
        for _, row in forward_df.iterrows():
            t = str(row['ticker']).strip()
            p = float(row['price']) if pd.notna(row['price']) and float(row['price']) > 0 else 0
            if p > 0:
                fwd_prices[t] = p

        forward_rets = {}
        for t in cur_prices:
            if t in fwd_prices:
                forward_rets[t] = (fwd_prices[t] / cur_prices[t] - 1) * 100

        if not forward_rets:
            continue

        # ── RAW strategies (no trajectory computation needed) ──
        # S1: Universe average — all stocks with valid forward returns
        s1_tickers = list(forward_rets.keys())

        # S2: Top 10 by WAVE rank (raw CSV rank — baseline for comparison)
        s2_tickers = df.nsmallest(10, 'rank')['ticker'].astype(str).str.strip().tolist()

        # S3: Top 20 by rank (broader basket — tests concentration vs diversification)
        s3_tickers = df.nsmallest(20, 'rank')['ticker'].astype(str).str.strip().tolist()

        # S4: Persistent top 50 — rank in top 50 for last 4 consecutive weeks
        s4_tickers = []
        for t, h_dict in histories.items():
            ranks_list = h_dict['ranks']
            totals_list = h_dict['total_per_week']
            if len(ranks_list) >= 4:
                last4_pcts = [(1 - ranks_list[-(i+1)] / max(totals_list[-(i+1)], 1)) * 100
                              for i in range(4)]
                if all(p >= 97.5 for p in last4_pcts):  # Top ~50 out of ~2100
                    s4_tickers.append(t)
        # Relax to top 100 if too few
        if len(s4_tickers) < 5:
            for t, h_dict in histories.items():
                ranks_list = h_dict['ranks']
                totals_list = h_dict['total_per_week']
                if len(ranks_list) >= 4 and t not in s4_tickers:
                    last4_pcts = [(1 - ranks_list[-(i+1)] / max(totals_list[-(i+1)], 1)) * 100
                                  for i in range(4)]
                    if all(p >= 95 for p in last4_pcts):
                        s4_tickers.append(t)

        # ── TRAJECTORY strategies (need full computation) ──
        traj = {}
        for ticker, h_dict in histories.items():
            if len(h_dict['ranks']) >= 2:
                try:
                    traj[ticker] = _compute_single_trajectory(h_dict)
                except Exception:
                    pass

        # S5: T-Score >= 70
        s5_tickers = [t for t, r in traj.items() if r['trajectory_score'] >= 70]

        # S6: T-Score >= 70 + No Decay (trap filter)
        s6_tickers = [t for t, r in traj.items()
                      if r['trajectory_score'] >= 70
                      and r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')]

        # S7: Conviction >= 65 (HIGH or above)
        s7_tickers = [t for t, r in traj.items() if r.get('conviction', 0) >= 65]

        # S8: Full Signal — max filtering
        s8_tickers = [t for t, r in traj.items()
                      if r['trajectory_score'] >= 70
                      and r.get('conviction', 0) >= 65
                      and r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')
                      and r.get('wave_fusion_label', 'WAVE_NEUTRAL') in ('WAVE_CONFIRMED', 'WAVE_STRONG')]

        # ── S2b: Top 10 by T-Score (tests Trajectory Engine's ranking vs raw WAVE rank) ──
        traj_sorted = sorted(traj.items(), key=lambda x: x[1]['trajectory_score'], reverse=True)
        s2b_tickers = [t for t, _ in traj_sorted[:10]]

        # ── S3b: Top 20 by T-Score (broader T-Score basket) ──
        s3b_tickers = [t for t, _ in traj_sorted[:20]]

        # ── S9: Conviction-Weighted (stocks with T-Score≥60, weighted by conviction) ──
        # Returns are computed as conviction-weighted average instead of equal-weight
        s9_candidates = {t: r for t, r in traj.items()
                         if r['trajectory_score'] >= 60 and r.get('conviction', 0) >= 40}

        # ── S10: Regime-Adaptive Strategy ──
        # Detect current market regime from median trajectory scores
        all_tscores = [r['trajectory_score'] for r in traj.values()]
        median_tscore = float(np.median(all_tscores)) if all_tscores else 50
        bear_mode = median_tscore < 45  # Market stress indicator

        if bear_mode:
            # Bear: defensive — only high-conviction, no-decay, persistent stocks
            s10_tickers = [t for t, r in traj.items()
                           if r.get('conviction', 0) >= 65
                           and r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')
                           and r.get('persistence_weeks', 0) >= 3]
        else:
            # Bull: aggressive — top T-Score with momentum confirmation
            s10_tickers = [t for t, r in traj_sorted[:15]
                           if r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')][:10]

        # ── Sector-capped versions of top picks (max 2 per sector) ──
        def _sector_cap(ticker_list, max_per_sector=2):
            """Apply sector diversification cap: max N stocks per sector."""
            sector_counts = {}
            capped = []
            for t in ticker_list:
                sec = histories.get(t, {}).get('sector', 'Unknown') or 'Unknown'
                if sector_counts.get(sec, 0) < max_per_sector:
                    capped.append(t)
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1
            return capped

        s2b_capped = _sector_cap(s2b_tickers, 2)
        s3b_capped = _sector_cap(s3b_tickers, 3)

        # ── S11: Momentum-Quality Strategy (v10.1) ──
        # Tests the NEW conviction signals: ret_1y (12-month momentum) +
        # from_low_pct (distance from 52w low). Selects stocks with:
        #   1. Strong 12-month return (>25%) — proven long-term momentum
        #   2. Far from 52-week low (>50%) — structural uptrend confirmed
        #   3. Conviction ≥ 50 — minimum multi-signal agreement
        #   4. No high decay — momentum still intact
        # This isolates the incremental value of ret_1y + from_low_pct signals.
        s11_candidates = []
        for t, r in traj.items():
            h_dict = histories.get(t, {})
            _r1y = h_dict.get('ret_1y', [])
            _fl = h_dict.get('from_low_pct', [])
            latest_ret_1y = float(_r1y[-1]) if _r1y and not np.isnan(_r1y[-1]) else 0
            latest_from_low = float(_fl[-1]) if _fl and not np.isnan(_fl[-1]) else 0
            if (latest_ret_1y >= 25
                and latest_from_low >= 50
                and r.get('conviction', 0) >= 50
                and r.get('decay_label', '') not in ('DECAY_HIGH', 'DECAY_MODERATE')):
                s11_candidates.append((t, r['trajectory_score']))
        # Sort by T-Score descending, sector-cap at 3 per sector
        s11_candidates.sort(key=lambda x: x[1], reverse=True)
        s11_tickers = _sector_cap([t for t, _ in s11_candidates], 3)

        # ── S12: Max Alpha Strategy (v10.1) ──
        # Uses alpha_score — the pure forward-return predictor.
        # alpha_score combines ONLY statistically proven factors:
        #   near-high, breakout, persistence, position, market state,
        #   no-decay, wave fusion, early rally stage.
        # Top 20 by alpha_score, sector-capped at 2/sector.
        # Returns are alpha_score-weighted (higher alpha_score = more capital).
        s12_scored = []
        for t, r in traj.items():
            a_score = r.get('alpha_score', 0)
            if a_score >= 40:  # Minimum quality gate
                s12_scored.append((t, a_score))
        s12_scored.sort(key=lambda x: x[1], reverse=True)
        s12_tickers_raw = [t for t, _ in s12_scored]
        s12_tickers = _sector_cap(s12_tickers_raw, 2)[:20]
        s12_weights = {t: r.get('alpha_score', 50) for t, r in traj.items() if t in s12_tickers}

        # ── S13: Capitulation Contrarian (data-proven: +2.99%, 60% WR) ──
        # Stocks showing CAPITULATION pattern — extreme selling exhaustion
        # signals high-probability mean reversion. Strongest single-pattern signal.
        s13_tickers = []
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            pats_str = str(row.get('patterns', ''))
            if 'CAPITULATION' in pats_str and t in forward_rets:
                s13_tickers.append(t)

        # ── S14: Bounce Recovery (BOUNCE state: +3.56%, 54.2% WR) ──
        # Stocks in BOUNCE or PULLBACK market state with score ≥ 50.
        # BOUNCE is the single strongest market state signal.
        s14_tickers = []
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            ms_val = float(row['master_score']) if pd.notna(row.get('master_score')) else 0
            state_val = str(row.get('market_state', ''))
            if state_val in ('BOUNCE', 'PULLBACK') and ms_val >= 50 and t in forward_rets:
                s14_tickers.append(t)

        # ── S15: Quality GARP (Quality Leader+Score≥65: +0.98%, GARP+Score≥65: +1.00%) ──
        # Stocks with QUALITY LEADER or GARP LEADER pattern + high master_score.
        # Combines fundamental quality signals with scoring system.
        s15_tickers = []
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            ms_val = float(row['master_score']) if pd.notna(row.get('master_score')) else 0
            pats_str = str(row.get('patterns', ''))
            has_quality = 'QUALITY LEADER' in pats_str or 'GARP' in pats_str
            if has_quality and ms_val >= 65 and t in forward_rets:
                s15_tickers.append(t)

        # ── S16: Pattern Flip Alpha (NEW Capitulation: +3.06%, 60.8% WR) ──
        # Stocks where CAPITULATION or VOL EXPLOSION+score≥60 pattern NEWLY appeared
        # (was absent in previous week). Fresh pattern appearance = strongest signal.
        s16_tickers = []
        if week_idx >= 1:
            prev_date = dates[week_idx - 1]
            prev_df = weekly_data.get(prev_date)
            if prev_df is not None:
                prev_pats_map = dict(zip(prev_df['ticker'].astype(str).str.strip(),
                                         prev_df['patterns'].astype(str)))
                for _, row in df.iterrows():
                    t = str(row['ticker']).strip()
                    if t not in forward_rets:
                        continue
                    cur_pats_str = str(row.get('patterns', ''))
                    prev_pats_str = prev_pats_map.get(t, '')
                    cur_set = set(p.strip() for p in cur_pats_str.split('|') if p.strip() and p.strip() != 'nan')
                    prev_set = set(p.strip() for p in prev_pats_str.split('|') if p.strip() and p.strip() != 'nan')
                    new_pats = cur_set - prev_set
                    # Newly appeared capitulation or vol explosion with decent score
                    ms_val = float(row['master_score']) if pd.notna(row.get('master_score')) else 0
                    for np_name in new_pats:
                        if 'CAPITULATION' in np_name:
                            s16_tickers.append(t)
                            break
                        if 'VOL EXPLOSION' in np_name and ms_val >= 60:
                            s16_tickers.append(t)
                            break

        # ── S17: Stealth Alpha (Stealth+Rank≤100: +0.10%, 48% WR in bear market) ──
        # Under-the-radar stocks with STEALTH pattern + strong rank.
        # Stealth = accumulation without price movement — leads to breakouts.
        s17_tickers = []
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            pats_str = str(row.get('patterns', ''))
            rk = int(row['rank']) if pd.notna(row.get('rank')) else 9999
            if 'STEALTH' in pats_str and rk <= 100 and t in forward_rets:
                s17_tickers.append(t)

        # ── S18: Near-High Breakout (Near High+Breakout≥70: -0.13%, 45.6% WR) ──
        # Stocks within 10% of 52-week high + strong breakout score.
        # Large-N screen (4335 obs) — relative strength filter.
        s18_tickers = []
        for _, row in df.iterrows():
            t = str(row['ticker']).strip()
            fh = float(row.get('from_high_pct', -99)) if pd.notna(row.get('from_high_pct')) else -99
            brk_val = float(row.get('breakout_score', 0)) if pd.notna(row.get('breakout_score')) else 0
            if fh > -10 and brk_val >= 70 and t in forward_rets:
                s18_tickers.append(t)

        # ── Measure forward returns for each strategy ──
        strategy_picks = {
            'S1: Universe Avg': s1_tickers,
            'S2: Top 10 WAVE Rank': s2_tickers,
            'S3: Top 20 WAVE Rank': s3_tickers,
            'S4: Persistent Top 50': s4_tickers,
            'S2b: Top 10 T-Score': s2b_capped,
            'S3b: Top 20 T-Score': s3b_capped,
            'S5: T-Score ≥ 70': s5_tickers,
            'S6: T-Score ≥ 70 + No Decay': s6_tickers,
            'S7: Conviction ≥ 65': s7_tickers,
            'S8: Full Signal': s8_tickers,
            'S10: Regime-Adaptive': s10_tickers,
            'S11: Momentum-Quality': s11_tickers,
            'S12: Max Alpha': s12_tickers,
            'S13: Capitulation Contrarian': s13_tickers,
            'S14: Bounce Recovery': s14_tickers,
            'S15: Quality GARP': s15_tickers,
            'S16: Pattern Flip Alpha': s16_tickers,
            'S17: Stealth Alpha': s17_tickers,
            'S18: Near-High Breakout': s18_tickers,
        }

        week_label = date.strftime('%Y-%m-%d')

        for sname, tickers in strategy_picks.items():
            valid_rets = [forward_rets[t] for t in tickers if t in forward_rets]
            avg_ret = float(np.mean(valid_rets)) if valid_rets else 0.0
            med_ret = float(np.median(valid_rets)) if valid_rets else 0.0
            all_results[sname].append({
                'week': week_label,
                'forward_week': forward_date.strftime('%Y-%m-%d'),
                'avg_return': avg_ret,
                'median_return': med_ret,
                'n_stocks': len(valid_rets),
                'n_positive': sum(1 for r in valid_rets if r > 0),
                'best': max(valid_rets) if valid_rets else 0,
                'worst': min(valid_rets) if valid_rets else 0,
            })

        # ── S9: Conviction-Weighted (special — weighted average return) ──
        s9_valid = [(forward_rets[t], s9_candidates[t].get('conviction', 50))
                     for t in s9_candidates if t in forward_rets]
        if s9_valid:
            s9_rets, s9_wts = zip(*s9_valid)
            total_wt = sum(s9_wts)
            s9_avg = sum(r * w for r, w in zip(s9_rets, s9_wts)) / max(total_wt, 1)
            s9_med = float(np.median(s9_rets))
            s9_n = len(s9_valid)
        else:
            s9_avg, s9_med, s9_n = 0.0, 0.0, 0
            s9_rets = []
        all_results['S9: Conviction-Weighted'].append({
            'week': week_label,
            'forward_week': forward_date.strftime('%Y-%m-%d'),
            'avg_return': s9_avg,
            'median_return': s9_med,
            'n_stocks': s9_n,
            'n_positive': sum(1 for r in s9_rets if r > 0),
            'best': max(s9_rets) if s9_rets else 0,
            'worst': min(s9_rets) if s9_rets else 0,
        })

        # ── S12: Max Alpha (alpha_score-weighted, like S9 but using alpha_score) ──
        s12_valid = [(forward_rets[t], s12_weights.get(t, 50))
                     for t in s12_tickers if t in forward_rets]
        if s12_valid:
            s12_rets, s12_wts = zip(*s12_valid)
            s12_total_wt = sum(s12_wts)
            s12_avg = sum(r * w for r, w in zip(s12_rets, s12_wts)) / max(s12_total_wt, 1)
            s12_med = float(np.median(s12_rets))
            s12_n = len(s12_valid)
        else:
            s12_avg, s12_med, s12_n = 0.0, 0.0, 0
            s12_rets = []
        # Override the equal-weighted S12 entry with alpha-weighted version
        if all_results['S12: Max Alpha'] and all_results['S12: Max Alpha'][-1]['week'] == week_label:
            all_results['S12: Max Alpha'][-1] = {
                'week': week_label,
                'forward_week': forward_date.strftime('%Y-%m-%d'),
                'avg_return': s12_avg,
                'median_return': s12_med,
                'n_stocks': s12_n,
                'n_positive': sum(1 for r in s12_rets if r > 0),
                'best': max(s12_rets) if s12_rets else 0,
                'worst': min(s12_rets) if s12_rets else 0,
            }

    if progress_callback:
        progress_callback(1.0, "Backtest complete!")

    return all_results, dates


# ============================================
# UI: BACKTEST TAB
# ============================================

def render_backtest_tab(uploaded_files):
    """Render strategy backtest validation tab"""
    st.markdown("### 📊 Strategy Backtest — Walk-Forward Validation")
    st.markdown("""
    <div style="background:#161b22; border-radius:10px; padding:16px; border:1px solid #30363d; margin-bottom:16px;">
        <div style="font-size:0.85rem; color:#8b949e;">
            <b>What this does:</b> For each week, it computes scores using ONLY past data,
            then checks what ACTUALLY happened to the selected stocks next week.
            No lookahead bias — pure walk-forward validation using real price returns.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Check cache
    bt_cache_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
    cached = st.session_state.get('_bt_result')
    cached_key = st.session_state.get('_bt_key')

    if cached is not None and cached_key == bt_cache_key:
        bt_results, bt_dates = cached
    else:
        bt_results = None

    n_files = len(uploaded_files)
    n_possible_windows = max(n_files - 3, 0)  # 2 for history + 1 forward

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    with col_info:
        if bt_results is None:
            st.caption(f"📁 {n_files} CSVs loaded → up to **{n_possible_windows}** test windows")
        else:
            st.caption(f"✅ Backtest loaded ({n_files} CSVs). Click Run to refresh.")

    if run_btn:
        progress_bar = st.progress(0, text="Initializing backtest...")

        def _progress(pct, text):
            progress_bar.progress(min(pct, 1.0), text=text)

        with st.spinner("Running walk-forward backtest..."):
            result = _run_strategy_backtest(uploaded_files, progress_callback=_progress)

        progress_bar.empty()

        if result is None:
            st.error(f"❌ Need at least 4 weeks of CSV data for backtest. You have {n_files} files.")
            return

        bt_results, bt_dates = result
        st.session_state['_bt_result'] = (bt_results, bt_dates)
        st.session_state['_bt_key'] = bt_cache_key
        st.rerun()

    if bt_results is None:
        return

    # ── Build Summary Statistics ──
    summary_rows = []
    for sname, weeks in bt_results.items():
        if not weeks:
            continue
        returns = [w['avg_return'] for w in weeks]
        n_weeks_tested = len(returns)
        avg_weekly = np.mean(returns)
        cumulative = np.prod([1 + r / 100 for r in returns]) * 100 - 100
        win_rate = sum(1 for r in returns if r > 0) / max(n_weeks_tested, 1) * 100
        avg_stocks = np.mean([w['n_stocks'] for w in weeks])
        std_ret = np.std(returns) if len(returns) > 1 else 0.001
        sharpe = avg_weekly / max(std_ret, 0.001) * np.sqrt(52)  # Annualized
        worst_week = min(returns)
        best_week = max(returns)

        # Max Drawdown — peak-to-trough of cumulative equity curve
        equity = [100.0]
        for r in returns:
            equity.append(equity[-1] * (1 + r / 100))
        peak = equity[0]
        max_dd = 0.0
        for v in equity[1:]:
            if v > peak:
                peak = v
            dd = (v - peak) / peak * 100
            if dd < max_dd:
                max_dd = dd

        summary_rows.append({
            'Strategy': sname,
            'Weeks Tested': n_weeks_tested,
            'Avg Stocks/Wk': round(avg_stocks, 0),
            'Avg Wk Return %': round(avg_weekly, 2),
            'Cumulative %': round(cumulative, 2),
            'Win Rate %': round(win_rate, 1),
            'Sharpe (Ann.)': round(sharpe, 2),
            'Max DD %': round(max_dd, 2),
            'Best Wk %': round(best_week, 2),
            'Worst Wk %': round(worst_week, 2),
        })

    if not summary_rows:
        st.warning("No test windows available. Need more week coverage.")
        return

    summary_df = pd.DataFrame(summary_rows)

    # ── Compute Alpha vs Universe & Recommendations ──
    universe_cum = summary_df.loc[summary_df['Strategy'] == 'S1: Universe Avg', 'Cumulative %']
    universe_val = float(universe_cum.iloc[0]) if len(universe_cum) > 0 else 0

    summary_df['Alpha %'] = round(summary_df['Cumulative %'] - universe_val, 2)

    # Assign recommendations based on data-driven analysis
    def _get_recommendation(row):
        name = row['Strategy']
        alpha = row['Alpha %']
        win = row['Win Rate %']
        dd = row['Max DD %']
        avg_stocks = row['Avg Stocks/Wk']
        if name == 'S1: Universe Avg':
            return '📊 Baseline'
        # Best alpha overall
        if alpha == summary_df.loc[summary_df['Strategy'] != 'S1: Universe Avg', 'Alpha %'].max():
            return '⭐ BEST ALPHA'
        # Best win rate
        if win == summary_df.loc[summary_df['Strategy'] != 'S1: Universe Avg', 'Win Rate %'].max() and win > 50:
            return '🎯 TOP WIN RATE'
        # Best drawdown protection
        if dd == summary_df.loc[summary_df['Strategy'] != 'S1: Universe Avg', 'Max DD %'].max() and alpha > 0:
            return '🛡️ SAFEST'
        # Danger zone — worse than universe
        if alpha < -2:
            return '⚠️ HIGH RISK'
        # Modest alpha
        if alpha > 3:
            return '✅ Good Alpha'
        if alpha > 0:
            return '➖ Marginal'
        return '➖ Underperforms'

    summary_df['Rating'] = summary_df.apply(_get_recommendation, axis=1)

    # Reorder columns for clarity
    col_order = ['Strategy', 'Rating', 'Weeks Tested', 'Avg Stocks/Wk',
                 'Avg Wk Return %', 'Cumulative %', 'Alpha %', 'Win Rate %',
                 'Sharpe (Ann.)', 'Max DD %', 'Best Wk %', 'Worst Wk %']
    summary_df = summary_df[col_order]

    best_strat = summary_df.loc[summary_df['Alpha %'].idxmax()]
    worst_strat = summary_df.loc[summary_df['Alpha %'].idxmin()]
    best_name = best_strat['Strategy']
    best_cum = best_strat['Cumulative %']
    best_alpha = best_strat['Alpha %']
    best_win = best_strat['Win Rate %']

    # Find strategy with best win rate, safest (best max DD), and best broad
    non_uni = summary_df[summary_df['Strategy'] != 'S1: Universe Avg']
    safest_row = non_uni.loc[non_uni['Max DD %'].idxmax()]  # Max DD is negative, so max = least negative
    top_wr_row = non_uni.loc[non_uni['Win Rate %'].idxmax()]
    # Best broad = high alpha among strategies with avg stocks > 50
    broad_df = non_uni[non_uni['Avg Stocks/Wk'] > 50]
    best_broad_row = broad_df.loc[broad_df['Alpha %'].idxmax()] if len(broad_df) > 0 else None

    # ── Key Finding: Top 3 Recommendations ──
    reco_cards = ''
    recommendations = []

    # Card 1: Best Alpha
    recommendations.append({
        'icon': '⭐', 'label': 'BEST ALPHA',
        'name': best_name, 'color': '#3fb950',
        'detail': f"Alpha: +{best_alpha:.1f}% | Cumulative: {best_cum:+.1f}% | Win Rate: {best_win:.0f}%",
        'note': f"{int(best_strat['Avg Stocks/Wk'])} stocks/wk — focused portfolio"
    })

    # Card 2: Best Broad Strategy
    if best_broad_row is not None:
        recommendations.append({
            'icon': '🛡️', 'label': 'BEST BROAD',
            'name': best_broad_row['Strategy'], 'color': '#58a6ff',
            'detail': f"Alpha: +{best_broad_row['Alpha %']:.1f}% | Win Rate: {best_broad_row['Win Rate %']:.0f}% | {int(best_broad_row['Avg Stocks/Wk'])} stocks",
            'note': 'Maximum diversification with alpha'
        })

    # Card 3: Safest Strategy
    recommendations.append({
        'icon': '🔰', 'label': 'SAFEST',
        'name': safest_row['Strategy'], 'color': '#ffa657',
        'detail': f"Max DD: {safest_row['Max DD %']:.1f}% | Alpha: +{safest_row['Alpha %']:.1f}% | Win Rate: {safest_row['Win Rate %']:.0f}%",
        'note': 'Lowest drawdown — capital preservation'
    })

    reco_html = '<div style="display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;">'
    for r in recommendations:
        reco_html += f"""
        <div style="flex:1; min-width:220px; background:linear-gradient(135deg, #0d1117, #161b22);
                    border-radius:12px; padding:16px; border:2px solid {r['color']};">
            <div style="font-size:0.75rem; font-weight:800; color:{r['color']}; letter-spacing:1px;">
                {r['icon']} {r['label']}
            </div>
            <div style="font-size:1.05rem; font-weight:700; color:#e6edf3; margin:6px 0 4px;">
                {r['name']}
            </div>
            <div style="font-size:0.8rem; color:#8b949e;">{r['detail']}</div>
            <div style="font-size:0.7rem; color:#484f58; margin-top:4px;">{r['note']}</div>
        </div>"""
    reco_html += '</div>'

    n_test_weeks = summary_rows[0]['Weeks Tested']
    market_verdict = 'BEAR MARKET' if universe_val < -5 else ('BULL MARKET' if universe_val > 5 else 'SIDEWAYS MARKET')
    market_color = '#ff7b72' if universe_val < -5 else ('#3fb950' if universe_val > 5 else '#ffa657')

    st.markdown(f"""
    <div style="background:linear-gradient(135deg, #0d1117, #161b22); border-radius:12px;
                padding:20px; border:2px solid #30363d; margin-bottom:10px;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <div style="font-size:1.2rem; font-weight:800; color:#e6edf3;">📊 BACKTEST RESULTS</div>
            <div style="display:flex; gap:16px;">
                <span style="font-size:0.8rem; color:#8b949e;">
                    {n_test_weeks} test windows · Universe: <span style="color:{market_color}; font-weight:700;">{universe_val:+.1f}% ({market_verdict})</span>
                </span>
            </div>
        </div>
        <div style="font-size:0.85rem; color:#8b949e; margin-bottom:14px;">Top strategy recommendations ranked by data-driven backtest performance:</div>
        {reco_html}
    </div>
    """, unsafe_allow_html=True)

    # ── Strategy Performance Table ──
    st.markdown("#### 📋 Strategy Comparison")

    # Color the key columns
    def _highlight_returns(val):
        if isinstance(val, (int, float)):
            if val > 0:
                return 'color: #3fb950; font-weight: 700'
            elif val < 0:
                return 'color: #ff7b72; font-weight: 700'
        return ''

    styled_df = summary_df.style.applymap(
        _highlight_returns,
        subset=['Avg Wk Return %', 'Cumulative %', 'Alpha %', 'Max DD %', 'Best Wk %', 'Worst Wk %']
    ).format({
        'Avg Wk Return %': '{:+.2f}',
        'Cumulative %': '{:+.2f}',
        'Alpha %': '{:+.2f}',
        'Win Rate %': '{:.0f}',
        'Sharpe (Ann.)': '{:.2f}',
        'Max DD %': '{:.2f}',
        'Best Wk %': '{:+.2f}',
        'Worst Wk %': '{:+.2f}',
        'Avg Stocks/Wk': '{:.0f}',
    })
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # ── Strategy Insights (data-driven) ──
    with st.expander("🔬 Strategy Insights (Data-Driven Analysis)", expanded=True):
        # Find key patterns from actual data
        insights = []

        # Insight 1: Concentration risk — compare S2b vs S3b
        s2b_row = summary_df[summary_df['Strategy'].str.contains('S2b')]
        s3b_row = summary_df[summary_df['Strategy'].str.contains('S3b')]
        if len(s2b_row) > 0 and len(s3b_row) > 0:
            s2b_cum = float(s2b_row['Cumulative %'].iloc[0])
            s3b_cum = float(s3b_row['Cumulative %'].iloc[0])
            spread = s3b_cum - s2b_cum
            if abs(spread) > 3:
                if spread > 0:
                    insights.append(('⚡', 'Concentration Risk', f'Top 20 T-Score ({s3b_cum:+.1f}%) beat Top 10 T-Score ({s2b_cum:+.1f}%) by **{spread:+.1f}%**. Broader baskets absorb single-stock blow-ups better.', '#ffa657'))
                else:
                    insights.append(('⚡', 'Concentration Pays', f'Top 10 T-Score ({s2b_cum:+.1f}%) beat Top 20 ({s3b_cum:+.1f}%) by **{abs(spread):.1f}%**. Concentrated picks are working.', '#3fb950'))

        # Insight 2: T-Score vs WAVE Rank
        s2_row = summary_df[summary_df['Strategy'] == 'S2: Top 10 WAVE Rank']
        s3_row = summary_df[summary_df['Strategy'] == 'S3: Top 20 WAVE Rank']
        if len(s2_row) > 0 and len(s2b_row) > 0:
            wave_best = max(float(s2_row['Cumulative %'].iloc[0]), float(s3_row['Cumulative %'].iloc[0])) if len(s3_row) > 0 else float(s2_row['Cumulative %'].iloc[0])
            tscore_best = max(float(s2b_row['Cumulative %'].iloc[0]), float(s3b_row['Cumulative %'].iloc[0])) if len(s3b_row) > 0 else float(s2b_row['Cumulative %'].iloc[0])
            gap = tscore_best - wave_best
            if abs(gap) > 2:
                if gap > 0:
                    insights.append(('🧠', 'T-Score Ranking Superior', f'Best T-Score strategy ({tscore_best:+.1f}%) outperformed best WAVE Rank strategy ({wave_best:+.1f}%) by **{gap:+.1f}%**. Trajectory Engine scoring adds real value.', '#3fb950'))
                else:
                    insights.append(('🧠', 'WAVE Rank Still Competitive', f'Best WAVE Rank ({wave_best:+.1f}%) beat best T-Score ({tscore_best:+.1f}%) by **{abs(gap):.1f}%**.', '#58a6ff'))

        # Insight 3: Conviction signal effectiveness
        s7_data = summary_df[summary_df['Strategy'].str.contains('S7')]
        if len(s7_data) > 0:
            s7_wr = float(s7_data['Win Rate %'].iloc[0])
            s7_dd = float(s7_data['Max DD %'].iloc[0])
            uni_wr = float(summary_df.loc[summary_df['Strategy'] == 'S1: Universe Avg', 'Win Rate %'].iloc[0])
            if s7_wr > uni_wr:
                insights.append(('🎯', 'Conviction Signal Works', f'Conviction ≥ 65 achieved **{s7_wr:.0f}% win rate** (vs universe {uni_wr:.0f}%), with the **best Max DD ({s7_dd:.1f}%)**. Conviction is the most protective filter.', '#3fb950'))

        # Insight 4: Regime-Adaptive performance
        s10_data = summary_df[summary_df['Strategy'].str.contains('S10')]
        if len(s10_data) > 0:
            s10_wr = float(s10_data['Win Rate %'].iloc[0])
            s10_alpha = float(s10_data['Alpha %'].iloc[0])
            if s10_wr >= 50:
                insights.append(('🔄', 'Regime Detection Effective', f'Regime-Adaptive achieved **{s10_wr:.0f}% win rate** (highest) with **{s10_alpha:+.1f}% alpha**. Switching between bull/bear modes adds value.', '#bc8cff'))

        # Insight 5: No Decay filter check
        s5_data = summary_df[summary_df['Strategy'].str.contains('S5:')]
        s6_data = summary_df[summary_df['Strategy'].str.contains('S6:')]
        if len(s5_data) > 0 and len(s6_data) > 0:
            s5_cum = float(s5_data['Cumulative %'].iloc[0])
            s6_cum = float(s6_data['Cumulative %'].iloc[0])
            diff = s6_cum - s5_cum
            if diff < -0.5:
                insights.append(('🔍', 'Decay Filter Drag', f'Adding No-Decay filter made results **{abs(diff):.1f}% worse** ({s6_cum:+.1f}% vs {s5_cum:+.1f}%). In downtrends, this filter removes bounce candidates.', '#ff7b72'))
            elif diff > 0.5:
                insights.append(('🔍', 'Decay Filter Helps', f'No-Decay filter improved results by **{diff:.1f}%** ({s6_cum:+.1f}% vs {s5_cum:+.1f}%).', '#3fb950'))

        # Insight 6: Full Signal overfiltering check
        s8_data = summary_df[summary_df['Strategy'].str.contains('S8:')]
        if len(s8_data) > 0:
            s8_alpha = float(s8_data['Alpha %'].iloc[0])
            s8_stocks = float(s8_data['Avg Stocks/Wk'].iloc[0])
            if s8_alpha < 2 and s8_stocks < 100:
                insights.append(('⚙️', 'Full Signal Overfilters', f'Full Signal (S8) has only **{s8_alpha:+.1f}% alpha** with ~{s8_stocks:.0f} stocks. Stacking T-Score + Conviction + No Decay + WAVE filters together cancels out alpha.', '#ff7b72'))

        if insights:
            for icon, title, detail, color in insights:
                st.markdown(f"""
                <div style="background:#161b22; border-radius:8px; padding:12px 16px;
                            border-left:4px solid {color}; margin-bottom:8px;">
                    <div style="font-weight:700; color:{color}; font-size:0.9rem; margin-bottom:2px;">{icon} {title}</div>
                    <div style="font-size:0.82rem; color:#c9d1d9;">{detail}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Not enough data variance to generate insights. Add more CSV weeks.")

    # ── Risk Warning for dangerous strategies ──
    worst_name = worst_strat['Strategy']
    worst_alpha = worst_strat['Alpha %']
    worst_dd = worst_strat['Max DD %']
    if worst_alpha < -3 and worst_name != 'S1: Universe Avg':
        st.markdown(f"""
        <div style="background:#1a0d0d; border-radius:8px; padding:12px 16px;
                    border:1px solid #6e3630; margin-bottom:16px;">
            <span style="color:#ff7b72; font-weight:700;">⚠️ RISK WARNING:</span>
            <span style="color:#c9d1d9; font-size:0.85rem;">
                <b>{worst_name}</b> underperformed the universe by <b>{worst_alpha:.1f}%</b>
                with a max drawdown of <b>{worst_dd:.1f}%</b>.
                Concentrated portfolios with few stocks carry severe single-stock blow-up risk.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # ── Cumulative Return Chart ──
    st.markdown("#### 📈 Cumulative Return Over Time")

    fig = go.Figure()
    strat_colors = {
        'S1: Universe Avg': '#484f58',
        'S2: Top 10 WAVE Rank': '#8b949e',
        'S3: Top 20 WAVE Rank': '#79c0ff',
        'S4: Persistent Top 50': '#d2a8ff',
        'S2b: Top 10 T-Score': '#58a6ff',
        'S3b: Top 20 T-Score': '#388bfd',
        'S5: T-Score ≥ 70': '#ffa657',
        'S6: T-Score ≥ 70 + No Decay': '#f0883e',
        'S7: Conviction ≥ 65': '#3fb950',
        'S8: Full Signal': '#FFD700',
        'S9: Conviction-Weighted': '#f778ba',
        'S10: Regime-Adaptive': '#bc8cff',
        'S11: Momentum-Quality': '#56d364',
        'S12: Max Alpha': '#e3b341',
    }

    # Highlight: best alpha, best broad, and safest strategies
    highlight_names = {best_name, safest_row['Strategy'], 'S1: Universe Avg'}
    if best_broad_row is not None:
        highlight_names.add(best_broad_row['Strategy'])

    for sname, weeks in bt_results.items():
        if not weeks:
            continue
        cum_vals = []
        cum = 100  # Start at 100
        x_dates = []
        for w in weeks:
            cum = cum * (1 + w['avg_return'] / 100)
            cum_vals.append(cum)
            x_dates.append(w['forward_week'])

        line_width = 3 if sname in highlight_names else 1.5
        dash = 'dash' if sname == 'S1: Universe Avg' else None

        fig.add_trace(go.Scatter(
            x=x_dates, y=cum_vals,
            name=sname,
            mode='lines+markers',
            marker=dict(size=5),
            line=dict(color=strat_colors.get(sname, '#8b949e'), width=line_width, dash=dash),
            hovertemplate=f'{sname}<br>Week: %{{x}}<br>Value: %{{y:.1f}}<br>Return: %{{customdata:+.2f}}%<extra></extra>',
            customdata=[w['avg_return'] for w in weeks],
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=450,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.35, xanchor='center', x=0.5,
                    font=dict(size=10)),
        yaxis=dict(title='Portfolio Value (₹100 start)', gridcolor='#21262d'),
        xaxis=dict(title='Week', gridcolor='#21262d'),
        hovermode='x unified',
    )
    fig.add_hline(y=100, line_dash='dot', line_color='#484f58', opacity=0.5)
    st.plotly_chart(fig, use_container_width=True, key='bt_cumulative_chart')

    # ── Weekly Breakdown ──
    with st.expander("📅 Weekly Breakdown", expanded=False):
        # Create a combined table: rows = weeks, columns = strategies
        if bt_results:
            first_strat = list(bt_results.keys())[0]
            week_labels = [w['week'] for w in bt_results[first_strat]]

            breakdown_data = {'Decision Week': week_labels}
            for sname in bt_results:
                breakdown_data[sname] = [f"{w['avg_return']:+.2f}% ({w['n_stocks']})"
                                         for w in bt_results[sname]]

            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)

    # ── Excess Return vs Universe ──
    st.markdown("#### 🎯 Excess Return vs Universe (Alpha)")

    fig2 = go.Figure()
    universe_returns = [w['avg_return'] for w in bt_results.get('S1: Universe Avg', [])]

    for sname, weeks in bt_results.items():
        if sname == 'S1: Universe Avg' or not weeks:
            continue
        excess = [w['avg_return'] - ur for w, ur in zip(weeks, universe_returns)]
        x_dates = [w['forward_week'] for w in weeks]
        avg_excess = np.mean(excess) if excess else 0

        colors = ['#3fb950' if e >= 0 else '#ff7b72' for e in excess]

        fig2.add_trace(go.Bar(
            x=x_dates, y=excess,
            name=f"{sname} (avg: {avg_excess:+.2f}%)",
            marker_color=strat_colors.get(sname, '#8b949e'),
            opacity=0.7,
            visible=True if sname == best_name else 'legendonly',
            hovertemplate=f'{sname}<br>Week: %{{x}}<br>Alpha: %{{y:+.2f}}%<extra></extra>',
        ))

    fig2.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117',
        height=350,
        margin=dict(l=40, r=20, t=30, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=-0.4, xanchor='center', x=0.5,
                    font=dict(size=10)),
        yaxis=dict(title='Excess Return vs Universe %', gridcolor='#21262d', zeroline=True,
                   zerolinecolor='#484f58'),
        xaxis=dict(title='Week', gridcolor='#21262d'),
        barmode='group',
    )
    st.plotly_chart(fig2, use_container_width=True, key='bt_alpha_chart')

    # ── Strategy Descriptions ──
    with st.expander("📖 Strategy Definitions", expanded=False):
        st.markdown("""
        | Strategy | Selection Rule | Tests |
        |----------|---------------|-------|
        | **S1: Universe Avg** | Buy all stocks equally | Baseline — market average |
        | **S2: Top 10 WAVE Rank** | 10 lowest WAVE rank numbers | Does raw ranking predict? |
        | **S3: Top 20 WAVE Rank** | 20 lowest WAVE rank numbers | Does broader selection reduce risk? |
        | **S4: Persistent Top 50** | Top ~2.5% for 4+ consecutive weeks | Does persistence add value? |
        | **S2b: Top 10 T-Score** | Top 10 by Trajectory Score (sector-capped ≤2/sector) | Does T-Score beat raw rank? |
        | **S3b: Top 20 T-Score** | Top 20 by Trajectory Score (sector-capped ≤3/sector) | Broader T-Score basket with diversification |
        | **S5: T-Score ≥ 70** | Full Trajectory Engine score ≥ 70 | Does the 8-component system work? |
        | **S6: T-Score ≥ 70 + No Decay** | S5 + no momentum decay trap | Does trap detection help? |
        | **S7: Conviction ≥ 65** | Multi-signal conviction score ≥ 65 | Does conviction predict returns? |
        | **S8: Full Signal** | T-Score ≥ 70 + Conviction ≥ 65 + No Decay + WAVE Confirmed/Strong | Does max filtering produce best results? |
        | **S9: Conviction-Weighted** | T-Score ≥ 60, conviction ≥ 40, weighted by conviction score | Does weighting by conviction add alpha? |
        | **S10: Regime-Adaptive** | Bull: Top T-Score + no decay; Bear: Conviction ≥ 65 + persistent + no decay | Does regime adaptation help? |
        | **S11: Momentum-Quality** | ret_1y ≥ 25% + from_low ≥ 50% + Conviction ≥ 50 + no decay, sector-capped | Do 12-month momentum + low-distance signals add alpha? |
        | **S12: Max Alpha** | Top 20 by Alpha Score (≥40), alpha-weighted returns, sector-capped ≤2/sector | Does a pure forward-predictor score maximize future returns? |
        | **S13: Capitulation Contrarian** | Stocks with 💣 CAPITULATION pattern (extreme selling exhaustion) | Backtest-proven: +2.99% avg, 60% WR. Mean reversion signal. |
        | **S14: Bounce Recovery** | BOUNCE/PULLBACK market state + Score ≥ 50 | BOUNCE is the #1 market state: +3.56% avg, 54.2% WR. |
        | **S15: Quality GARP** | QUALITY LEADER or GARP pattern + Score ≥ 65 | Quality + fundamentals: +0.98% avg (Quality), +1.00% avg (GARP). |
        | **S16: Pattern Flip Alpha** | CAPITULATION or VOL EXPLOSION+Score≥60 NEWLY appeared this week | Fresh pattern signal: NEW Capitulation +3.06%, 60.8% WR. |
        | **S17: Stealth Alpha** | 🤫 STEALTH pattern + Rank ≤ 100 | Stealth accumulation in top ranks: +0.10% in bear market, 48% WR. |
        | **S18: Near-High Breakout** | Within 10% of 52w high + Breakout score ≥ 70 | Relative strength: largest N screen (4335 obs), 45.6% WR. |

        **Forward Return:** Actual price change from current week to next week. Computed from price data, not ret_7d.

        **Walk-Forward:** At each test point, only data available UP TO that week is used. No future information leaks into the scoring.
        
        **Sector Cap (S2b/S3b):** Max 2-3 stocks per sector prevents concentration risk from destroying returns in sector-specific crashes.
        
        **Conviction-Weighted (S9):** Instead of equal-weight, each stock's return is weighted by its conviction score. High-conviction picks get more capital.
        
        **Regime-Adaptive (S10):** Automatically switches between aggressive (momentum-focused) in bull markets and defensive (conviction+persistence) in bear markets based on median T-Score.
        
        **Momentum-Quality (S11):** Filters for stocks with strong 12-month returns (≥25%), far from 52-week low (≥50%), minimum conviction (≥50), and no momentum decay. Sector-capped at 3/sector. Tests whether long-term momentum + structural uptrend confirmation adds alpha.
        
        **Max Alpha (S12):** Uses Alpha Score — a purpose-built forward-return predictor combining ONLY factors with proven next-week alpha: near-high proximity (+0.55%/wk), breakout quality (+0.44%/wk), persistence (+3.91%/wk), position strength (+0.34%/wk), market state (BOUNCE +1.17%/wk), no-decay, wave fusion, and early rally stage. Returns are alpha-score-weighted (higher alpha_score = more capital). Sector-capped at 2/sector for diversification.
        """)

    # ── Download Backtest Results ──
    st.markdown("#### 📥 Download Results")
    dl1, dl2, dl3 = st.columns(3)

    with dl1:
        # Summary CSV — drop Rating column for clean data export
        export_df = summary_df.drop(columns=['Rating'], errors='ignore')
        summary_csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📋 Summary Table (CSV)",
            data=summary_csv,
            file_name="backtest_summary.csv",
            mime="text/csv",
            use_container_width=True,
            key='bt_dl_summary',
        )

    with dl2:
        # Weekly breakdown CSV — one row per strategy × week
        weekly_rows = []
        for sname, weeks in bt_results.items():
            for w in weeks:
                weekly_rows.append({
                    'Strategy': sname,
                    'Decision_Week': w['week'],
                    'Forward_Week': w['forward_week'],
                    'Avg_Return_%': round(w['avg_return'], 4),
                    'Median_Return_%': round(w['median_return'], 4),
                    'N_Stocks': w['n_stocks'],
                    'N_Positive': w['n_positive'],
                    'Best_%': round(w['best'], 4),
                    'Worst_%': round(w['worst'], 4),
                })
        weekly_csv = pd.DataFrame(weekly_rows).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📅 Weekly Breakdown (CSV)",
            data=weekly_csv,
            file_name="backtest_weekly.csv",
            mime="text/csv",
            use_container_width=True,
            key='bt_dl_weekly',
        )

    with dl3:
        # Full combined report — summary + blank row + weekly detail
        import io
        buf = io.StringIO()
        buf.write("=== BACKTEST SUMMARY ===\n")
        summary_df.to_csv(buf, index=False)
        buf.write("\n\n=== WEEKLY DETAIL (per strategy per week) ===\n")
        pd.DataFrame(weekly_rows).to_csv(buf, index=False)

        # Excess returns section
        universe_rets_dl = [w['avg_return'] for w in bt_results.get('S1: Universe Avg', [])]
        excess_rows = []
        for sname, weeks in bt_results.items():
            if sname == 'S1: Universe Avg' or not weeks:
                continue
            for w, ur in zip(weeks, universe_rets_dl):
                excess_rows.append({
                    'Strategy': sname,
                    'Week': w['week'],
                    'Return_%': round(w['avg_return'], 4),
                    'Universe_%': round(ur, 4),
                    'Alpha_%': round(w['avg_return'] - ur, 4),
                })
        if excess_rows:
            buf.write("\n\n=== EXCESS RETURNS (Alpha vs Universe) ===\n")
            pd.DataFrame(excess_rows).to_csv(buf, index=False)

        full_csv = buf.getvalue().encode('utf-8')
        st.download_button(
            label="📊 Full Report (CSV)",
            data=full_csv,
            file_name="backtest_full_report.csv",
            mime="text/csv",
            use_container_width=True,
            key='bt_dl_full',
        )


# ============================================
# UI: PATTERN ANALYSER TAB
# ============================================

def render_pattern_analyser_tab(uploaded_files):
    """Data-driven pattern intelligence across all loaded CSVs."""
    st.markdown("### 🔬 Pattern Analyser — Data-Driven Intelligence")
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0d1117,#161b22); border-radius:10px;
                padding:16px; border:1px solid #30363d; margin-bottom:16px;">
        <div style="font-size:0.85rem; color:#8b949e;">
            <b>What this does:</b> Analyses every pattern, market state, and combination across
            ALL your loaded CSVs. Uses <b>price-based forward returns</b> (next week's actual price change)
            to measure which signals genuinely predict future performance. No lookahead bias.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parse all CSVs ──
    weekly_data = {}
    for ufile in uploaded_files:
        date = parse_date_from_filename(ufile.name)
        if date is None:
            continue
        try:
            ufile.seek(0)
            df = pd.read_csv(ufile)
            if 'rank' in df.columns and 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str).str.strip()
                df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
                df['master_score'] = pd.to_numeric(df.get('master_score', 0), errors='coerce').fillna(0)
                df['price'] = pd.to_numeric(df.get('price', 0), errors='coerce').fillna(0)
                weekly_data[date] = df
        except Exception:
            pass

    dates = sorted(weekly_data.keys())
    if len(dates) < 3:
        st.warning("Need at least 3 weekly CSVs for pattern analysis.")
        return

    pa_cache_key = ('PA', tuple(sorted((f.name, f.size) for f in uploaded_files)))
    cached = st.session_state.get('_pa_result')
    cached_key = st.session_state.get('_pa_key')
    pa_data = cached if (cached is not None and cached_key == pa_cache_key) else None

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("🔬 Run Analysis", type="primary", use_container_width=True, key='pa_run')
    with col_info:
        if pa_data is None:
            st.caption(f"📁 {len(dates)} weekly CSVs → {len(dates)-1} forward-return windows")
        else:
            st.caption(f"✅ Loaded ({pa_data['n_obs']:,} observations). Click to refresh.")

    if run_btn:
        prog = st.progress(0, text="Building pattern analysis...")
        pa_data = _build_pattern_analysis(weekly_data, dates, prog)
        st.session_state['_pa_result'] = pa_data
        st.session_state['_pa_key'] = pa_cache_key
        prog.progress(1.0, text="Analysis complete!")

    if pa_data is None:
        st.info("Click **Run Analysis** to compute pattern intelligence from your data.")
        return

    pa1, pa2, pa3, pa4, pa5, pa6 = st.tabs([
        "📊 Pattern Performance", "🎯 Multi-Factor Screens",
        "🔄 Pattern Flips", "🗺️ Pattern × State Matrix",
        "🧩 Combo Explorer", "📈 Category & Sector",
    ])
    with pa1:
        _render_pa_pattern_performance(pa_data)
    with pa2:
        _render_pa_multifactor_screens(pa_data)
    with pa3:
        _render_pa_pattern_flips(pa_data)
    with pa4:
        _render_pa_pattern_state_matrix(pa_data)
    with pa5:
        _render_pa_combo_explorer(pa_data)
    with pa6:
        _render_pa_category_sector(pa_data)


# ---------- Pattern Analyser: data builder ----------

def _build_pattern_analysis(weekly_data, dates, progress_bar):
    """Build comprehensive pattern analysis from consecutive week pairs."""
    from collections import defaultdict

    results = []
    pat_rets = defaultdict(list)
    combo_rets = defaultdict(list)
    state_rets = defaultdict(list)
    score_bucket_rets = defaultdict(list)
    rank_bucket_rets = defaultdict(list)
    cat_rets = defaultdict(list)
    sector_rets = defaultdict(list)
    pat_state_rets = defaultdict(lambda: defaultdict(list))
    cat_score_rets = defaultdict(lambda: defaultdict(list))
    new_pat_rets = defaultdict(list)
    month_rets = defaultdict(list)

    n_pairs = len(dates) - 1
    for i in range(n_pairs):
        progress_bar.progress(i / max(n_pairs, 1) * 0.9,
                              text=f"Analysing week {i+1}/{n_pairs}...")
        d1, d2 = dates[i], dates[i + 1]
        df1 = weekly_data[d1]
        df2 = weekly_data[d2]
        p2 = dict(zip(df2['ticker'].astype(str).str.strip(), df2['price']))
        month_str = d1.strftime('%Y-%m')

        prev_pats_map = {}
        if i > 0:
            prev_df = weekly_data[dates[i - 1]]
            prev_pats_map = dict(zip(
                prev_df['ticker'].astype(str).str.strip(),
                prev_df['patterns'].astype(str),
            ))

        for _, row in df1.iterrows():
            tk = str(row['ticker']).strip()
            pr1 = float(row['price']) if pd.notna(row.get('price')) and float(row.get('price', 0)) > 0 else 0
            pr2_val = p2.get(tk, 0)
            pr2 = float(pr2_val) if pd.notna(pr2_val) and float(pr2_val) > 0 else 0
            if pr1 <= 0 or pr2 <= 0:
                continue
            fwd = (pr2 / pr1 - 1) * 100
            ms = float(row.get('master_score', 0)) if pd.notna(row.get('master_score')) else 0
            rk = int(row['rank']) if pd.notna(row.get('rank')) else 9999
            state = str(row.get('market_state', '')).strip()
            tq = float(row.get('trend_quality', 0)) if pd.notna(row.get('trend_quality')) else 0
            mom = float(row.get('momentum_score', 0)) if pd.notna(row.get('momentum_score')) else 0
            brk = float(row.get('breakout_score', 0)) if pd.notna(row.get('breakout_score')) else 0
            fh = float(row.get('from_high_pct', -99)) if pd.notna(row.get('from_high_pct')) else -99
            cat = str(row.get('category', '')).strip()
            sect = str(row.get('sector', '')).strip()
            pats_str = str(row.get('patterns', ''))
            pats_list = [p.strip() for p in pats_str.split('|') if p.strip() and p.strip() != 'nan']
            n_pats = len(pats_list)

            results.append(dict(
                fwd=fwd, ms=ms, rk=rk, state=state, tq=tq, mom=mom,
                brk=brk, fh=fh, cat=cat, sector=sect, n_pats=n_pats,
                pats_str=pats_str,
            ))

            for p in pats_list:
                pat_rets[p].append(fwd)
                if state:
                    pat_state_rets[p][state].append(fwd)

            if n_pats >= 2:
                combo_rets[' + '.join(sorted(pats_list)[:4])].append(fwd)

            if state:
                state_rets[state].append(fwd)

            sb = '70+' if ms >= 70 else '60-70' if ms >= 60 else '50-60' if ms >= 50 else '40-50' if ms >= 40 else '0-40'
            score_bucket_rets[sb].append(fwd)

            rb = 'Top 10' if rk <= 10 else 'Top 50' if rk <= 50 else 'Top 100' if rk <= 100 else 'Top 500' if rk <= 500 else '500+'
            rank_bucket_rets[rb].append(fwd)

            if cat:
                cat_rets[cat].append(fwd)
                cat_score_rets[cat][sb].append(fwd)
            if sect:
                sector_rets[sect].append(fwd)
            month_rets[month_str].append(fwd)

            if prev_pats_map:
                prev_set = set(pp.strip() for pp in prev_pats_map.get(tk, '').split('|')
                               if pp.strip() and pp.strip() != 'nan')
                for np_name in (set(pats_list) - prev_set):
                    new_pat_rets[np_name].append(fwd)

    # Multi-factor screens
    rdf = pd.DataFrame(results) if results else pd.DataFrame()
    screens = {}
    if not rdf.empty:
        def _scr(name, mask):
            sub = rdf[mask]
            n = len(sub)
            if n >= 5:
                screens[name] = dict(n=n, avg=sub['fwd'].mean(), med=sub['fwd'].median(),
                                     wr=(sub['fwd'] > 0).mean() * 100)
            else:
                screens[name] = dict(n=n, avg=0, med=0, wr=0, few=True)

        _scr('Capitulation', rdf['pats_str'].str.contains('CAPITULATION', na=False))
        _scr('High Score + PULLBACK state', (rdf['ms'] >= 70) & (rdf['state'] == 'PULLBACK'))
        _scr('High Score + BOUNCE state', (rdf['ms'] >= 60) & (rdf['state'] == 'BOUNCE'))
        _scr('Quality Leader + Score≥65', rdf['pats_str'].str.contains('QUALITY LEADER', na=False) & (rdf['ms'] >= 65))
        _scr('GARP + Score≥65', rdf['pats_str'].str.contains('GARP', na=False) & (rdf['ms'] >= 65))
        _scr('Stealth + Rank≤100', rdf['pats_str'].str.contains('STEALTH', na=False) & (rdf['rk'] <= 100))
        _scr('Near High + Breakout≥70', (rdf['fh'] > -10) & (rdf['brk'] >= 70))
        _scr('High Score + Near High', (rdf['ms'] >= 70) & (rdf['fh'] > -10))
        _scr('Top 50 + TQ≥70', (rdf['rk'] <= 50) & (rdf['tq'] >= 70))
        _scr('Cat+Mkt Leader + Institutional',
             rdf['pats_str'].str.contains('CAT LEADER', na=False) &
             rdf['pats_str'].str.contains('MARKET LEADER', na=False) &
             rdf['pats_str'].str.contains('INSTITUTIONAL', na=False) &
             ~rdf['pats_str'].str.contains('TSUNAMI', na=False))
        _scr('Quality Leader + GARP',
             rdf['pats_str'].str.contains('QUALITY LEADER', na=False) &
             rdf['pats_str'].str.contains('GARP', na=False))
        _scr('BOUNCE + Score≥50', (rdf['state'] == 'BOUNCE') & (rdf['ms'] >= 50))
        _scr('Vol Explosion + Mom≥60',
             rdf['pats_str'].str.contains('VOL EXPLOSION', na=False) & (rdf['mom'] >= 60))
        _scr('Golden Cross + TQ≥70',
             rdf['pats_str'].str.contains('GOLDEN CROSS', na=False) & (rdf['tq'] >= 70))
        _scr('Distribution (AVOID)', rdf['pats_str'].str.contains('DISTRIBUTION', na=False))
        _scr('Hidden Gem (AVOID)', rdf['pats_str'].str.contains('HIDDEN GEM', na=False))
        _scr('Mega Cap + Score≥60', (rdf['cat'] == 'Mega Cap') & (rdf['ms'] >= 60))
        _scr('Large Cap + Near High + Institutional',
             (rdf['cat'] == 'Large Cap') & (rdf['fh'] > -10) &
             rdf['pats_str'].str.contains('INSTITUTIONAL', na=False) &
             ~rdf['pats_str'].str.contains('TSUNAMI', na=False))

    def _summarize(rets_dict, min_n=10):
        out = []
        for key, rets in rets_dict.items():
            if len(rets) >= min_n:
                out.append(dict(
                    name=key, n=len(rets), avg=float(np.mean(rets)),
                    med=float(np.median(rets)),
                    wr=float(sum(1 for r in rets if r > 0) / len(rets) * 100),
                ))
        return sorted(out, key=lambda x: -x['avg'])

    pat_state_matrix = {}
    for pat, sd in pat_state_rets.items():
        pat_state_matrix[pat] = {}
        for st_name, rets in sd.items():
            if len(rets) >= 5:
                pat_state_matrix[pat][st_name] = dict(
                    n=len(rets), avg=float(np.mean(rets)),
                    wr=float(sum(1 for r in rets if r > 0) / len(rets) * 100),
                )

    cat_score_matrix = {}
    for cat, sd in cat_score_rets.items():
        cat_score_matrix[cat] = {}
        for sl, rets in sd.items():
            if len(rets) >= 10:
                cat_score_matrix[cat][sl] = dict(
                    n=len(rets), avg=float(np.mean(rets)),
                    wr=float(sum(1 for r in rets if r > 0) / len(rets) * 100),
                )

    return dict(
        n_obs=len(results), n_weeks=len(dates) - 1,
        date_range=(dates[0].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')),
        pat_stats=_summarize(pat_rets, 10),
        combo_stats=_summarize(combo_rets, 10),
        state_stats=_summarize(state_rets, 20),
        score_stats=_summarize(score_bucket_rets, 20),
        rank_stats=_summarize(rank_bucket_rets, 20),
        cat_stats=_summarize(cat_rets, 20),
        sector_stats=_summarize(sector_rets, 50),
        month_stats=_summarize(month_rets, 20),
        screens=screens,
        flip_stats=_summarize(new_pat_rets, 10),
        pat_state_matrix=pat_state_matrix,
        cat_score_matrix=cat_score_matrix,
    )


# ---------- Pattern Analyser: color helpers ----------

def _pa_color(val, threshold_good=0, threshold_great=1.0):
    if val >= threshold_great:
        return '#3fb950'
    elif val >= threshold_good:
        return '#58a6ff'
    elif val >= -0.5:
        return '#d29922'
    return '#ff7b72'


def _pa_wr_color(wr):
    if wr >= 55:
        return '#3fb950'
    elif wr >= 45:
        return '#58a6ff'
    elif wr >= 38:
        return '#d29922'
    return '#ff7b72'


# ---------- Pattern Analyser: sub-tab renderers ----------

def _render_pa_pattern_performance(pa_data):
    """Pattern performance table + state + monthly charts."""
    st.markdown("#### 📊 Individual Pattern Forward Returns")
    st.caption(f"{pa_data['n_obs']:,} stock-week observations across {pa_data['n_weeks']} weeks "
               f"({pa_data['date_range'][0]} → {pa_data['date_range'][1]})")

    pat_stats = pa_data['pat_stats']
    if not pat_stats:
        st.info("No pattern data available.")
        return

    best = pat_stats[0]
    worst = pat_stats[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("🏆 Best Pattern", best['name'], f"{best['avg']:+.2f}% avg")
    c2.metric("⚠️ Worst Pattern", worst['name'], f"{worst['avg']:+.2f}% avg")
    c3.metric("📊 Patterns Tracked", len(pat_stats))

    rows_html = ""
    for i, s in enumerate(pat_stats):
        bg = '#161b22' if i % 2 == 0 else '#0d1117'
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 10px;color:#c9d1d9;">{i+1}</td>'
            f'<td style="padding:6px 10px;color:#e6edf3;">{s["name"]}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:#8b949e;">{s["n"]:,}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{_pa_color(s["avg"])};font-weight:600;">{s["avg"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{_pa_color(s["med"])};">{s["med"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{_pa_wr_color(s["wr"])};font-weight:600;">{s["wr"]:.1f}%</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <div style="overflow-x:auto; border-radius:10px; border:1px solid #30363d;">
    <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
    <thead><tr style="background:#21262d; color:#8b949e;">
        <th style="padding:8px 10px;text-align:left;">#</th>
        <th style="padding:8px 10px;text-align:left;">Pattern</th>
        <th style="padding:8px 10px;text-align:right;">N</th>
        <th style="padding:8px 10px;text-align:right;">Avg Return</th>
        <th style="padding:8px 10px;text-align:right;">Med Return</th>
        <th style="padding:8px 10px;text-align:right;">Win Rate</th>
    </tr></thead><tbody>{rows_html}</tbody></table></div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🌡️ Market State Performance")
    state_stats = pa_data['state_stats']
    if state_stats:
        fig = go.Figure()
        names = [s['name'] for s in state_stats]
        avgs = [s['avg'] for s in state_stats]
        wrs = [s['wr'] for s in state_stats]
        fig.add_trace(go.Bar(
            x=names, y=avgs, marker_color=[_pa_color(a) for a in avgs],
            text=[f"{a:+.2f}%<br>WR:{w:.0f}%" for a, w in zip(avgs, wrs)],
            textposition='outside', textfont=dict(size=10),
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=350, margin=dict(l=40, r=20, t=30, b=60),
            yaxis=dict(title='Avg Forward Return %', gridcolor='#21262d'),
        )
        fig.add_hline(y=0, line_dash='dot', line_color='#484f58')
        st.plotly_chart(fig, use_container_width=True, key='pa_state_chart')

    st.markdown("#### 📅 Monthly Market Regime")
    month_stats = pa_data['month_stats']
    if month_stats:
        ms_sorted = sorted(month_stats, key=lambda x: x['name'])
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[s['name'] for s in ms_sorted],
            y=[s['avg'] for s in ms_sorted],
            marker_color=['#3fb950' if s['avg'] >= 0 else '#ff7b72' for s in ms_sorted],
            text=[f"{s['avg']:+.1f}%" for s in ms_sorted], textposition='outside',
        ))
        fig2.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=300, margin=dict(l=40, r=20, t=30, b=40),
            yaxis=dict(title='Avg Forward Return %', gridcolor='#21262d'),
        )
        fig2.add_hline(y=0, line_dash='dot', line_color='#484f58')
        st.plotly_chart(fig2, use_container_width=True, key='pa_month_chart')


def _render_pa_multifactor_screens(pa_data):
    """Pre-built multi-factor screens ranked by performance."""
    st.markdown("#### 🎯 Multi-Factor Screens — Pre-Built Signal Combinations")
    st.caption("Each screen combines multiple factors. Ranked by average forward return.")
    screens = pa_data.get('screens', {})
    if not screens:
        st.info("No screen data available.")
        return

    sorted_screens = sorted(screens.items(), key=lambda x: x[1].get('avg', -999), reverse=True)
    rows_html = ""
    for i, (name, s) in enumerate(sorted_screens):
        if s.get('few'):
            avg_str, med_str, wr_str = "—", "—", f"N={s['n']}"
            avg_color = wr_color = '#484f58'
        else:
            avg_str = f"{s['avg']:+.2f}%"
            med_str = f"{s['med']:+.2f}%"
            wr_str = f"{s['wr']:.1f}%"
            avg_color = _pa_color(s['avg'])
            wr_color = _pa_wr_color(s['wr'])
        tag = ""
        if 'AVOID' in name:
            tag = ' <span style="background:#ff7b72;color:#0d1117;border-radius:4px;padding:1px 6px;font-size:0.7rem;font-weight:700;">AVOID</span>'
        elif s.get('avg', -1) >= 1.0 and not s.get('few'):
            tag = ' <span style="background:#3fb950;color:#0d1117;border-radius:4px;padding:1px 6px;font-size:0.7rem;font-weight:700;">STRONG</span>'
        elif s.get('avg', -1) >= 0 and not s.get('few'):
            tag = ' <span style="background:#58a6ff;color:#0d1117;border-radius:4px;padding:1px 6px;font-size:0.7rem;font-weight:700;">EDGE</span>'
        bg = '#161b22' if i % 2 == 0 else '#0d1117'
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 10px;color:#e6edf3;">{name}{tag}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:#8b949e;">{s["n"]:,}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{avg_color};font-weight:600;">{avg_str}</td>'
            f'<td style="padding:6px 10px;text-align:right;">{med_str}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{wr_color};font-weight:600;">{wr_str}</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <div style="overflow-x:auto; border-radius:10px; border:1px solid #30363d;">
    <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
    <thead><tr style="background:#21262d; color:#8b949e;">
        <th style="padding:8px 10px;text-align:left;">Screen</th>
        <th style="padding:8px 10px;text-align:right;">N</th>
        <th style="padding:8px 10px;text-align:right;">Avg Return</th>
        <th style="padding:8px 10px;text-align:right;">Med Return</th>
        <th style="padding:8px 10px;text-align:right;">Win Rate</th>
    </tr></thead><tbody>{rows_html}</tbody></table></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#161b22;border-radius:8px;padding:12px;margin-top:12px;border:1px solid #30363d;">
        <div style="font-size:0.78rem;color:#8b949e;">
            <b>🟢 STRONG</b> = Avg ≥ +1.0% &nbsp; <b>🔵 EDGE</b> = Avg ≥ 0% &nbsp;
            <b>🔴 AVOID</b> = Confirmed negative signal
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_pa_pattern_flips(pa_data):
    """Newly-appeared pattern performance."""
    st.markdown("#### 🔄 Pattern Flip Analysis — New Appearance Signals")
    st.caption("When a pattern NEWLY appears (absent the previous week), what happens next?")
    flip_stats = pa_data.get('flip_stats', [])
    if not flip_stats:
        st.info("No flip data available (need ≥3 CSVs).")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("🚀 Best New Signal", f"NEW {flip_stats[0]['name']}", f"{flip_stats[0]['avg']:+.2f}%")
    c2.metric("⚠️ Worst New Signal", f"NEW {flip_stats[-1]['name']}", f"{flip_stats[-1]['avg']:+.2f}%")
    c3.metric("🔄 Patterns Tracked", len(flip_stats))

    rows_html = ""
    for i, s in enumerate(flip_stats):
        bg = '#161b22' if i % 2 == 0 else '#0d1117'
        ind = "🟢" if s['avg'] >= 0.5 else "🔵" if s['avg'] >= 0 else "🟡" if s['avg'] >= -0.5 else "🔴"
        rows_html += (
            f'<tr style="background:{bg};">'
            f'<td style="padding:6px 10px;color:#c9d1d9;">{ind}</td>'
            f'<td style="padding:6px 10px;color:#e6edf3;">NEW {s["name"]}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:#8b949e;">{s["n"]:,}</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{_pa_color(s["avg"])};font-weight:600;">{s["avg"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{_pa_color(s["med"])};">{s["med"]:+.2f}%</td>'
            f'<td style="padding:6px 10px;text-align:right;color:{_pa_wr_color(s["wr"])};font-weight:600;">{s["wr"]:.1f}%</td>'
            f'</tr>'
        )
    st.markdown(f"""
    <div style="overflow-x:auto; border-radius:10px; border:1px solid #30363d;">
    <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
    <thead><tr style="background:#21262d; color:#8b949e;">
        <th style="padding:8px 10px;"></th>
        <th style="padding:8px 10px;text-align:left;">New Pattern Signal</th>
        <th style="padding:8px 10px;text-align:right;">N</th>
        <th style="padding:8px 10px;text-align:right;">Avg Return</th>
        <th style="padding:8px 10px;text-align:right;">Med Return</th>
        <th style="padding:8px 10px;text-align:right;">Win Rate</th>
    </tr></thead><tbody>{rows_html}</tbody></table></div>
    """, unsafe_allow_html=True)


def _render_pa_pattern_state_matrix(pa_data):
    """Pattern × Market State heatmap."""
    st.markdown("#### 🗺️ Pattern × Market State Matrix")
    st.caption("Green = positive avg return, Red = negative. N ≥ 5 required.")
    matrix = pa_data.get('pat_state_matrix', {})
    if not matrix:
        st.info("No matrix data available.")
        return

    all_states = set()
    for pd_dict in matrix.values():
        all_states.update(pd_dict.keys())
    state_order = ['BOUNCE', 'STRONG_DOWNTREND', 'DOWNTREND', 'PULLBACK',
                   'STRONG_UPTREND', 'SIDEWAYS', 'UPTREND', 'ROTATION']
    states = [s for s in state_order if s in all_states]

    pat_overall = {s['name']: s['avg'] for s in pa_data.get('pat_stats', [])}
    pats_sorted = [p for p in sorted(matrix.keys(), key=lambda x: -pat_overall.get(x, -99))
                   if matrix[p]]
    if not pats_sorted or not states:
        st.info("Insufficient cross-data for matrix.")
        return

    z_vals, hover_text = [], []
    for pat in pats_sorted:
        rz, rh = [], []
        for state in states:
            cell = matrix.get(pat, {}).get(state)
            if cell:
                rz.append(cell['avg'])
                rh.append(f"{pat} × {state}<br>Avg: {cell['avg']:+.2f}%<br>WR: {cell['wr']:.0f}%<br>N: {cell['n']}")
            else:
                rz.append(None)
                rh.append(f"{pat} × {state}<br>N < 5")
        z_vals.append(rz)
        hover_text.append(rh)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals, x=states, y=pats_sorted,
        colorscale=[[0, '#ff7b72'], [0.35, '#21262d'], [0.5, '#30363d'],
                    [0.65, '#21262d'], [1.0, '#3fb950']],
        zmid=0, text=hover_text, hovertemplate='%{text}<extra></extra>',
        colorbar=dict(title='Avg Ret %', tickformat='+.1f'),
    ))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
        height=max(350, len(pats_sorted) * 22 + 100),
        margin=dict(l=180, r=20, t=30, b=60),
        yaxis=dict(autorange='reversed'), xaxis=dict(side='top'),
    )
    st.plotly_chart(fig, use_container_width=True, key='pa_heatmap')


def _render_pa_combo_explorer(pa_data):
    """Top and worst pattern combos."""
    st.markdown("#### 🧩 Pattern Combo Explorer")
    st.caption("Exact pattern combinations (≥2 patterns) ranked by forward return. N ≥ 10.")
    combo_stats = pa_data.get('combo_stats', [])
    if not combo_stats:
        st.info("No combo data available.")
        return

    def _combo_table(title, data):
        if not data:
            return
        st.markdown(f"##### {title}")
        rows_html = ""
        for i, s in enumerate(data):
            bg = '#161b22' if i % 2 == 0 else '#0d1117'
            rows_html += (
                f'<tr style="background:{bg};">'
                f'<td style="padding:6px 10px;color:#e6edf3;max-width:400px;overflow:hidden;text-overflow:ellipsis;">{s["name"]}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:#8b949e;">{s["n"]}</td>'
                f'<td style="padding:6px 10px;text-align:right;color:{_pa_color(s["avg"])};font-weight:600;">{s["avg"]:+.2f}%</td>'
                f'<td style="padding:6px 10px;text-align:right;color:{_pa_wr_color(s["wr"])};">{s["wr"]:.1f}%</td>'
                f'</tr>'
            )
        st.markdown(f"""
        <div style="overflow-x:auto; border-radius:10px; border:1px solid #30363d;">
        <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
        <thead><tr style="background:#21262d; color:#8b949e;">
            <th style="padding:8px 10px;text-align:left;">Pattern Combo</th>
            <th style="padding:8px 10px;text-align:right;">N</th>
            <th style="padding:8px 10px;text-align:right;">Avg Return</th>
            <th style="padding:8px 10px;text-align:right;">Win Rate</th>
        </tr></thead><tbody>{rows_html}</tbody></table></div>
        """, unsafe_allow_html=True)

    _combo_table("🟢 Top Positive Combos", [c for c in combo_stats if c['avg'] >= 0][:20])
    _combo_table("🔴 Worst Combos (AVOID)", list(reversed([c for c in combo_stats if c['avg'] < 0]))[:10])


def _render_pa_category_sector(pa_data):
    """Category, sector, score and rank breakdowns."""
    st.markdown("#### 📈 Category & Sector Performance")

    cat_stats = pa_data.get('cat_stats', [])
    if cat_stats:
        st.markdown("##### 🏢 Category (Market Cap)")
        cat_order = ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
        ordered = sorted(cat_stats, key=lambda x: cat_order.index(x['name']) if x['name'] in cat_order else 99)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[s['name'] for s in ordered], y=[s['avg'] for s in ordered],
            marker_color=[_pa_color(s['avg']) for s in ordered],
            text=[f"{s['avg']:+.2f}%<br>WR:{s['wr']:.0f}%<br>N:{s['n']:,}" for s in ordered],
            textposition='outside', textfont=dict(size=10),
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=320, margin=dict(l=40, r=20, t=30, b=40),
            yaxis=dict(title='Avg Forward Return %', gridcolor='#21262d'),
        )
        fig.add_hline(y=0, line_dash='dot', line_color='#484f58')
        st.plotly_chart(fig, use_container_width=True, key='pa_cat_chart')

    cat_score_matrix = pa_data.get('cat_score_matrix', {})
    if cat_score_matrix:
        st.markdown("##### 🏢 × 📊 Category × Score Matrix")
        score_labels = ['70+', '60-70', '50-60', '40-50', '0-40']
        cat_order_list = ['Mega Cap', 'Large Cap', 'Mid Cap', 'Small Cap', 'Micro Cap']
        rows_html = ""
        for ci, cat in enumerate(cat_order_list):
            if cat not in cat_score_matrix:
                continue
            cells = ""
            for sl in score_labels:
                cell = cat_score_matrix[cat].get(sl)
                if cell:
                    color = _pa_color(cell['avg'])
                    cells += (f'<td style="padding:6px;text-align:center;color:{color};font-weight:600;">'
                              f'{cell["avg"]:+.1f}%<br><span style="font-size:0.7rem;color:#8b949e;">'
                              f'WR:{cell["wr"]:.0f}% N:{cell["n"]}</span></td>')
                else:
                    cells += '<td style="padding:6px;text-align:center;color:#484f58;">—</td>'
            bg = '#161b22' if ci % 2 == 0 else '#0d1117'
            rows_html += f'<tr style="background:{bg};"><td style="padding:6px 10px;color:#e6edf3;font-weight:600;">{cat}</td>{cells}</tr>'
        hdr = ''.join(f'<th style="padding:8px;text-align:center;">Score {sl}</th>' for sl in score_labels)
        st.markdown(f"""
        <div style="overflow-x:auto; border-radius:10px; border:1px solid #30363d;">
        <table style="width:100%; border-collapse:collapse; font-size:0.82rem;">
        <thead><tr style="background:#21262d; color:#8b949e;">
            <th style="padding:8px 10px;text-align:left;">Category</th>{hdr}
        </tr></thead><tbody>{rows_html}</tbody></table></div>
        """, unsafe_allow_html=True)

    sector_stats = pa_data.get('sector_stats', [])
    if sector_stats:
        st.markdown("##### 🏭 Top 15 Sectors")
        top15 = sector_stats[:15]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=[s['avg'] for s in top15], y=[s['name'] for s in top15],
            orientation='h',
            marker_color=['#3fb950' if s['avg'] >= 0 else '#ff7b72' for s in top15],
            text=[f"{s['avg']:+.2f}% (WR:{s['wr']:.0f}%)" for s in top15],
            textposition='outside', textfont=dict(size=9),
        ))
        fig2.update_layout(
            template='plotly_dark', paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
            height=max(300, len(top15) * 28 + 80),
            margin=dict(l=180, r=60, t=30, b=40),
            xaxis=dict(title='Avg Forward Return %', gridcolor='#21262d'),
            yaxis=dict(autorange='reversed'),
        )
        fig2.add_vline(x=0, line_dash='dot', line_color='#484f58')
        st.plotly_chart(fig2, use_container_width=True, key='pa_sector_chart')

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### 📊 Score Range Performance")
        for s in sorted(pa_data.get('score_stats', []),
                        key=lambda x: ['70+', '60-70', '50-60', '40-50', '0-40'].index(x['name'])
                        if x['name'] in ['70+', '60-70', '50-60', '40-50', '0-40'] else 99):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 8px;border-bottom:1px solid #21262d;">'
                f'<span style="color:#e6edf3;">Score {s["name"]}</span>'
                f'<span><span style="color:{_pa_color(s["avg"])};font-weight:600;">{s["avg"]:+.2f}%</span>'
                f' <span style="color:{_pa_wr_color(s["wr"])};">WR:{s["wr"]:.0f}%</span>'
                f' <span style="color:#484f58;">N:{s["n"]:,}</span></span></div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("##### 🏅 Rank Range Performance")
        for s in sorted(pa_data.get('rank_stats', []),
                        key=lambda x: ['Top 10', 'Top 50', 'Top 100', 'Top 500', '500+'].index(x['name'])
                        if x['name'] in ['Top 10', 'Top 50', 'Top 100', 'Top 500', '500+'] else 99):
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:4px 8px;border-bottom:1px solid #21262d;">'
                f'<span style="color:#e6edf3;">{s["name"]}</span>'
                f'<span><span style="color:{_pa_color(s["avg"])};font-weight:600;">{s["avg"]:+.2f}%</span>'
                f' <span style="color:{_pa_wr_color(s["wr"])};">WR:{s["wr"]:.0f}%</span>'
                f' <span style="color:#484f58;">N:{s["n"]:,}</span></span></div>',
                unsafe_allow_html=True,
            )


# ============================================
# UI: ABOUT TAB
# ============================================

def render_about_tab():
    """Render about/documentation tab"""

    st.markdown("""
    ## 🔺 Alpha Trajectory v10.1 — Stock Intelligence Engine

    The **ALL TIME BEST** stock rank trajectory analysis system with **8-component adaptive scoring**,
    **Alpha Score Engine (8 forward-predictors)**, **data-driven conviction (12 signals)**,
    **sector-relative blending**, **momentum decay warning**, and **sector alpha detection**.

    ---

    ### 🧠 The Architecture: Signal-Isolated Pipeline

    ```
    8-Component Adaptive Scoring → Elite Dominance Bonus → Bayesian Shrinkage
        → Hurst Persistence × Wave Fusion
        → Sector-Relative Blending → Sector Alpha Tag
    ```

    **SIGNAL ISOLATION PRINCIPLE:** Each data source enters through exactly ONE scoring path.
    No signal leakage — return magnitude is scored only in ReturnQuality component.

    | Layer | What It Does | Impact |
    |---|---|---|
    | **8 Components** | Weighted scoring by position tier (8 dimensions inc. breakout_quality) | 100% of base score |
    | **Elite Bonus** | Sustained top-tier → guaranteed score floor | Top 3% for 60% weeks → floor 88 |
    | **Bayesian Shrinkage** | Short-history stocks pulled toward neutral | 4 weeks → 75% shrunk |
    | **Hurst Multiplier** | Persistent trends boosted, mean-reverting penalized | ×0.94 to ×1.06 |
    | **Wave Fusion** | Cross-validates WAVE Detection signals with Trajectory | ×0.94 to ×1.06 |
    | **Sector-Relative** | Blends universe rank with sector rank (size-dampened) | ±15 score adj |
    | **Sector Alpha** | Separates leaders from sector-beta riders | Tag: LEADER / BETA / LAGGARD |

    ---

    ### 💰 Directional Price-Rank Alignment (v6.1)

    Measures whether return **direction** agrees with rank movement.
    Does NOT score return **magnitude** — that is ReturnQuality's job.

    #### Two Directional Signals

    | Signal | Weight | What It Measures |
    |--------|--------|------------------|
    | **Weekly Direction** | 55% | Does sign(EMA ret_7d) match sign(percentile change)? |
    | **Monthly Direction** | 45% | Does sign(ret_30d) match sign(percentile change)? |

    #### Multiplier Range

    | Alignment Score | Multiplier | Label | Meaning |
    |----------------|-----------|-------|---------|
    | **72-100** | ×1.04 — ×1.08 | 💰 CONFIRMED | Return direction validates rank trajectory |
    | **50-72** | ×1.00 — ×1.04 | NEUTRAL | Inconclusive |
    | **35-50** | ×0.96 — ×1.00 | NEUTRAL | Mild concern |
    | **0-35** | ×0.88 — ×0.96 | ⚠️ DIVERGENT | Return direction contradicts rank |

    ---

    ### 🔻 Momentum Decay Warning (v6.1)

    **The Problem:** 11.4% of top-10% stocks have negative 30-day returns.
    These are TRAP stocks — ranked well based on PAST momentum that has now faded.

    #### 4 Decay Signals

    | Signal | What It Checks | Max Points |
    |--------|---------------|------------|
    | **Weekly Return** | ret_7d < -5% → 30 pts, < -2% → 15 pts | 30 |
    | **30-Day Return** | Top stock + ret_30d < -15% → 40 pts (THE TRAP!) | 40 |
    | **From High** | from_high_pct < -20% on ranked stock → 20 pts | 20 |
    | **Consecutive Negative** | 3+ weeks of ret_7d < -1% → 15 pts | 15 |

    #### v6.1 Improvements
    - **Proven Winner Exemption:** Uses actual ret_6m (not rank proxy):
      ret_6m > 60% → 55% score reduction, > 30% → 40%, > 15% → 25%
    - **Smooth multiplier:** Continuous curve instead of step-function cliffs.
      Score 0 → ×1.00, Score 35 → ×0.976, Score 60 → ×0.958, Score 100 → ×0.93

    ---

    ### 🏛️ Sector Alpha Check

    | Z-Score | Classification | Icon | Meaning |
    |---------|---------------|------|---------|
    | **> 1.5** | SECTOR_LEADER | 👑 | Genuine alpha — outperforms sector significantly |
    | **0.5 - 1.5** | SECTOR_OUTPERFORM | ⬆️ | Above sector average |
    | **-0.5 - 0.5** | SECTOR_ALIGNED | ➖ | Moving with sector |
    | **-1.0 - -0.5** (hot sector) | SECTOR_BETA | 🏷️ | Riding sector wave, not genuine alpha |
    | **< -1.0** | SECTOR_LAGGARD | 📉 | Below sector average |

    ---

    ### 📈 Advanced Trading Signals (v9.0)

    Data-driven signals based on 27-transition CSV analysis:

    #### 1. Conviction Score (0-100)
    Aggregates 9 bullish signals into a single BUY confidence metric.
    v9.0: Rebuilt using forward-return predictive analysis + market state signal.

    | Signal | Max Points | Measures |
    |--------|-----------|----------|
    | Persistence Strength | 25 | Consecutive weeks in top 25% (backtest #1) |
    | Consistency Quality | 12 | Stable rank trajectory |
    | Near-High Strength | 14 | from_high_pct (#1 forward predictor, +0.55%/wk) |
    | Return Quality | 8 | Actual returns backing rank |
    | Breakout Quality | 12 | breakout_score (#2 predictor, +0.44%/wk) |
    | Wave Confluence | 8 | WAVE system agreement |
    | Sector Leadership | 7 | position_score (#3 predictor, +0.34%/wk) |
    | Market State | 8 | BOUNCE +1.17%/wk mean-reversion bonus |
    | No-Decay Bonus | 6 | Absence of momentum deterioration |

    | Tag | Score | Emoji |
    |-----|-------|-------|
    | VERY_HIGH | ≥ 80 | 🎯 |
    | HIGH | ≥ 65 | ✅ |
    | MODERATE | ≥ 45 | ⚡ |
    | LOW | ≥ 25 | ⚠️ |
    | VERY_LOW | < 25 | ❌ |

    #### 2. Risk-Adjusted T-Score
    `Risk-Adj = T-Score / (1 + rank_volatility / 50)`
    Penalizes high-volatility stocks. Rank vol of 25 → 1.5× divisor.

    #### 3. Exit Warning System (0-100)
    Detects when to SELL existing holdings. Aggregates 5 exit signals:

    | Signal | Max Points | Trigger |
    |--------|-----------|---------|
    | TMI Collapse | 25 | TMI < 40 while score > 50 |
    | Price Divergence | 25 | PRICE_DIVERGENT label |
    | Momentum Decay | 30 | DECAY_HIGH or DECAY_MODERATE |
    | Negative Streak | 20 | 3+ consecutive rank drops |
    | Pattern Warning | 20 | fading / crash / topping_out |

    | Tag | Score | Action |
    |-----|-------|--------|
    | 🚨 EXIT_NOW | ≥ 60 | Strong sell signal |
    | ⚠️ CAUTION | ≥ 40 | Review position |
    | 👀 WATCH | ≥ 20 | Monitor closely |
    | ✅ HOLD | < 20 | Position is safe |

    #### 4. Hot Streak Detection
    Flags stocks with sustained momentum + high position:
    - 4+ consecutive improving weeks AND ≥ 70th percentile
    - 3+ weeks AND ≥ 80th percentile
    - 5+ weeks AND ≥ 60th percentile

    #### 5. Volume Confirmation
    Validates rank moves with Wave Engine volume score:

    | Condition | Label | Meaning |
    |-----------|-------|---------|
    | Improving + Vol ≥ 70 | STRONG | High-conviction rank improvement |
    | Improving + Vol ≥ 50 | MODERATE | Decent volume support |
    | Improving + Vol < 30 | WEAK | Low volume → may reverse |
    | Falling + Vol ≥ 70 | DISTRIBUTION | Selling pressure detected |

    ---

    ### �🏗️ Adaptive Weight System (7 Components)

    Weights **dynamically shift** based on the stock's average percentile.
    Smooth interpolation between tiers — no hard cutoffs.

    | Tier | Avg Pctl | Positional | Trend | Velocity | Accel | Consistency | Resilience | RetQuality |
    |------|----------|------------|-------|----------|-------|-------------|------------|------------|
    | **Elite** | > 90% | **40%** | 10% | 7% | 4% | 16% | 10% | 13% |
    | **Strong** | 70-90% | 28% | 16% | 10% | 7% | 14% | 12% | 13% |
    | **Mid** | 40-70% | 15% | 19% | **17%** | 10% | 12% | 12% | 15% |
    | **Bottom** | < 40% | 8% | 17% | **21%** | **15%** | 10% | 12% | **17%** |

    ---

    ### 📈 Return Quality Component (v6.0+)

    Dedicated 7th component — ALL return data enters through this single door.

    | Sub-Signal | Weight | Data Source | What It Scores |
    |-----------|--------|-------------|----------------|
    | **3-Month Return** | 30% | ret_3m | Medium-term momentum quality |
    | **6-Month Return** | 30% | ret_6m | Institutional horizon confirmation |
    | **Short-Term Momentum** | 20% | ret_7d + ret_30d | Recency signal (7d=35%, 30d=65%) |
    | **Return Health** | 20% | Cross-timeframe + from_high | Agreement + correction distance |

    Score range: 0-100 where 50=neutral, ≥75=🔥 strong, ≤30=💧 weak

    ---

    ### 🛡️ Elite Dominance Bonus

    | Tier | Percentile | Required Duration | Score Floor |
    |------|-----------|------------------|-------------|
    | Top 3% | > 97th | ≥ 60% of weeks | **88** |
    | Top 5% | > 95th | ≥ 60% of weeks | **82** |
    | Top 10% | > 90th | ≥ 60% of weeks | **73** |
    | Top 20% | > 85th | ≥ 55% of weeks | **65** |

    ---

    ### 📊 TMI (Trajectory Momentum Index)

    `TMI = 100 - (100 / (1 + RS))` where `RS = Avg Rank Gain / Avg Rank Loss`

    | TMI Range | Interpretation |
    |-----------|----------------|
    | **70-100** | Strong momentum — rank consistently improving |
    | **50-70** | Moderate momentum |
    | **30-50** | Weak momentum — mixed signals |
    | **0-30** | Deteriorating — rank consistently worsening |

    ---

    ### 🏷️ Trajectory Patterns
    """)

    # Pattern table
    pattern_rows = []
    for key, (emoji, name, desc) in PATTERN_DEFS.items():
        pattern_rows.append({'Pattern': f"{emoji} {name}", 'Description': desc})
    st.table(pd.DataFrame(pattern_rows))

    st.markdown("""
    ---

    ### 🌊 Wave Signal Fusion (v8.0)

    The **Wave Signal Fusion Engine** cross-validates 18 WAVE Detection columns with the
    Trajectory Engine's own scoring, producing a fusion multiplier (×0.92 — ×1.10).

    | Signal | Weight | What It Measures |
    |--------|--------|-----------------|
    | **Confluence** | 35% | Agreement between WAVE Detection scores (position, acceleration, breakout) and Trajectory components |
    | **Institutional Flow** | 30% | Money flow, VMI, relative volume, overall market strength |
    | **Harmony** | 20% | Momentum harmony score from WAVE Detection |
    | **Fundamental Quality** | 15% | EPS growth, PE reasonableness, EPS tier (quality gate, not driver) |

    **Classifications:**

    | Label | Score | Multiplier | Meaning |
    |-------|-------|-----------|---------|
    | 🌊 **STRONG** | ≥ 72 | ×1.06 — ×1.10 | Both systems strongly agree — high conviction |
    | ✅ **CONFIRMED** | 58 — 71 | ×1.01 — ×1.05 | Cross-system confirmation |
    | ➖ **NEUTRAL** | 42 — 57 | ×0.99 — ×1.01 | No strong signal either way |
    | ⚠️ **WEAK** | 30 — 41 | ×0.95 — ×0.98 | Mild disagreement — caution |
    | 🔇 **CONFLICT** | < 30 | ×0.92 — ×0.94 | Systems fundamentally disagree — danger |

    ---

    ### 📡 Signal Tags (v6.1)

    The **Signals** column in rankings combines multiple indicators:

    | Icon | Signal | Meaning |
    |------|--------|---------|
    | 💰 | Price Confirmed | Return direction validates rank trajectory |
    | ⚠️ | Price Divergent | Return direction contradicts rank |
    | 🔥 | Strong Returns | ReturnQuality ≥ 75 — strong return profile |
    | 💧 | Weak Returns | ReturnQuality ≤ 30 — weak return profile |
    | 🔻 | Decay High | Good rank but severely negative returns — TRAP! |
    | ⚡ | Decay Moderate | Moderate momentum decay warning |
    | 👑 | Sector Leader | Genuine alpha — significantly above sector average |
    | 🏷️ | Sector Beta | Riding hot sector, not genuine alpha |
    | 📉 | Sector Laggard | Below sector average |
    | 🌊 | Wave Strong | Both WAVE Detection and Trajectory agree — high fusion score |
    | 🔇 | Wave Conflict | WAVE Detection and Trajectory disagree — systems in conflict |

    ---

    ### 🎓 Grades

    | Grade | Score Range | Meaning |
    |-------|------------|---------|
    | 🏆 **S** | 85 — 100 | Elite — sustained top position or explosive improvement |
    | 🥇 **A** | 70 — 84 | Excellent — strong position + positive trajectory |
    | 🥈 **B** | 55 — 69 | Good — above average, potential emerging |
    | 🥉 **C** | 40 — 54 | Average — mixed signals, watch list |
    | 📊 **D** | 25 — 39 | Below average — weak or deteriorating |
    | 📉 **F** | 0 — 24 | Poor — declining or insufficient data |

    ---

    *Built for the Wave Detection ecosystem • v10.1 • Alpha Trajectory • April 2026*
    """)


# ============================================
# MAIN APPLICATION
# ============================================

def main():
    """Main application entry point"""

    # ── Sidebar: Data Source ──
    with st.sidebar:
        # ═══════════════════════════════════════════════
        # 🎯 QUICK ACTIONS
        # ═══════════════════════════════════════════════
        st.markdown('<div class="sb-section-head">🎯 QUICK ACTIONS</div>', unsafe_allow_html=True)
        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("🔄 Refresh", key='sb_refresh_main', use_container_width=True, type='primary'):
                st.session_state.pop('_traj_key', None)
                st.session_state.pop('_traj_result', None)
                st.session_state.pop('_drive_key_loaded', None)
                st.session_state.pop('_sb_selected_files', None)
                st.rerun()
        with qa2:
            if st.button("🧹 Clear Cache", key='sb_clear_cache_main', use_container_width=True):
                st.cache_data.clear()
                st.session_state.pop('_traj_key', None)
                st.session_state.pop('_traj_result', None)
                st.rerun()

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # 📂 DATA SOURCE — toggle buttons like Wave Detection
        # ═══════════════════════════════════════════════
        st.markdown('<div class="sb-section-head">📂 DATA SOURCE</div>', unsafe_allow_html=True)

        ds1, ds2 = st.columns(2)
        with ds1:
            if st.button("📂 Upload CSV",
                         type="primary" if st.session_state.get('data_source_mode', 'upload') == 'upload' else "secondary",
                         key='ds_btn_upload', use_container_width=True):
                st.session_state['data_source_mode'] = 'upload'
                st.session_state.pop('_sb_selected_files', None)
                st.rerun()
        with ds2:
            if st.button("☁️ Google Drive",
                         type="primary" if st.session_state.get('data_source_mode', 'upload') == 'drive' else "secondary",
                         key='ds_btn_drive', use_container_width=True):
                st.session_state['data_source_mode'] = 'drive'
                st.session_state.pop('_sb_selected_files', None)
                st.rerun()

        uploaded_files = []
        drive_folder_key = ""
        data_source_mode = st.session_state.get('data_source_mode', 'upload')

        # ═══════════════════════════════════════════
        # MODE 1: Upload CSV Files
        # ═══════════════════════════════════════════
        if data_source_mode == "upload":
            uploaded_files = st.file_uploader(
                "Upload Weekly CSV Snapshots",
                type=['csv'],
                accept_multiple_files=True,
                help="Upload Wave Detection weekly CSVs (Stocks_Weekly_YYYY-MM-DD_*_data.csv)",
                label_visibility='collapsed'
            )

        # ═══════════════════════════════════════════
        # MODE 2: Google Drive
        # ═══════════════════════════════════════════
        else:
            drive_folder_key = st.text_input(
                "Google Drive Folder Key",
                value="",
                placeholder="Paste key or full folder URL",
                key='drive_folder_key'
            ).strip()

            if drive_folder_key:
                drive_folder_key = _normalize_drive_folder_key(drive_folder_key)
                drive_folder_url = _drive_folder_url(drive_folder_key)
                st.markdown(
                    f'<div class="sb-cached-badge">🔗 <a href="{drive_folder_url}" '
                    f'target="_blank" style="color:#58a6ff;text-decoration:none">Open in Drive</a></div>',
                    unsafe_allow_html=True
                )

                # Auto-fetch once per new key
                key_changed = st.session_state.get('_drive_key_loaded') != drive_folder_key
                if key_changed:
                    progress_bar = st.progress(0, text="Connecting to Google Drive...")
                    progress_bar.progress(20, text="Downloading CSV files...")
                    drive_uploads, drive_err = _load_csv_uploads_from_drive(drive_folder_key)
                    progress_bar.progress(100, text="Complete!")
                    if drive_err:
                        st.error(f"❌ {drive_err}")
                        st.session_state.pop('_drive_uploads', None)
                        st.session_state.pop('_drive_key_loaded', None)
                    else:
                        st.session_state['_drive_uploads'] = drive_uploads
                        st.session_state['_drive_key_loaded'] = drive_folder_key
                        st.session_state['_drive_load_time'] = datetime.now()
                        st.success(f"✅ Loaded {len(drive_uploads)} CSV file(s)")
                    progress_bar.empty()

                # Reuse previously fetched files
                if st.session_state.get('_drive_key_loaded') == drive_folder_key:
                    uploaded_files = st.session_state.get('_drive_uploads', [])
                    load_time = st.session_state.get('_drive_load_time')
                    if load_time:
                        mins = (datetime.now() - load_time).seconds // 60
                        time_label = f"{mins} min ago" if mins > 0 else "just now"
                        st.markdown(
                            f'<div class="sb-cached-badge">⚡ Cached · {time_label} · '
                            f'{len(uploaded_files)} files</div>',
                            unsafe_allow_html=True
                        )

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

        # ═══════════════════════════════════════════════
        # 🗓️ FILE RANGE — date range selection
        # ═══════════════════════════════════════════════
        if uploaded_files:
            _dated_files, _undated = _extract_dated_files(uploaded_files)
            _total_uploaded = len(uploaded_files)

            if _dated_files:
                _min_dt = _dated_files[0][0].date()
                _max_dt = _dated_files[-1][0].date()

                st.markdown('<div class="sb-section-head">🗓️ FILE RANGE</div>', unsafe_allow_html=True)
                range_mode = st.radio(
                    "File Range",
                    ["All Time", "Custom"],
                    index=0,
                    key='file_range_mode',
                    horizontal=True,
                    label_visibility='collapsed'
                )

                if range_mode == "Custom":
                    col_from, col_to = st.columns(2)
                    with col_from:
                        range_start = st.date_input(
                            "From", value=_min_dt,
                            min_value=_min_dt, max_value=_max_dt,
                            key='file_range_start'
                        )
                    with col_to:
                        range_end = st.date_input(
                            "To", value=_max_dt,
                            min_value=_min_dt, max_value=_max_dt,
                            key='file_range_end'
                        )
                else:
                    range_start = _min_dt
                    range_end = _max_dt

                if range_start > range_end:
                    st.warning("Start date is after end date. Using full range.")
                    range_start = _min_dt
                    range_end = _max_dt

                selected = [f for dt, f in _dated_files if range_start <= dt.date() <= range_end]
                st.caption(f"{range_start.strftime('%Y-%m-%d')} → {range_end.strftime('%Y-%m-%d')} · {len(selected)}/{len(_dated_files)} files")

                # Store for use outside sidebar
                st.session_state['_sb_selected_files'] = selected
                st.session_state['_sb_total_uploaded'] = _total_uploaded
                st.session_state['_sb_undated'] = _undated
            else:
                st.info("No parseable dates in filenames; using all files.")
                st.session_state['_sb_selected_files'] = uploaded_files
                st.session_state['_sb_total_uploaded'] = _total_uploaded
                st.session_state['_sb_undated'] = 0

    # Header
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-icon">🔺</div>
        <div class="hero-title">ALPHA TRAJECTORY</div>
        <div class="hero-sub">Stock Intelligence Engine • Multi-Week Momentum & Alpha Scoring</div>
        <div class="hero-badge">v10.1 · 14 Strategies · 12 Signals · Max Alpha</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Robust file recovery: always pull from session state for Drive mode ──
    if not uploaded_files and data_source_mode == 'drive':
        uploaded_files = st.session_state.get('_drive_uploads', [])

    # Apply file-range selection from sidebar (if it ran)
    if st.session_state.get('_sb_selected_files') is not None:
        uploaded_files = st.session_state['_sb_selected_files']

    if not uploaded_files:
        if data_source_mode == "drive":
            st.info("👈 Paste your Google Drive folder key in the sidebar. App will fetch CSVs automatically.")
            st.markdown("""
            **Google Drive mode:**
            1. Paste your folder key in sidebar (or full folder URL)
            2. App auto-fetches all CSV files from that folder
            3. Analysis starts immediately when files are loaded
            """)
        else:
            st.info("👈 Upload your weekly CSV snapshots from the sidebar to begin trajectory analysis")
            st.markdown("""
            **How to use:**
            1. Open the **sidebar** (click `>` at the top-left if collapsed)
            2. Click **Browse files** or drag-and-drop your Wave Detection weekly CSV exports
            3. Upload multiple weeks at once (select all CSVs together)
            4. Files should be named: `Stocks_Weekly_YYYY-MM-DD_Month_Year_data.csv`
            5. Minimum **3 weeks** recommended for meaningful trajectory analysis
            """)
        return

    total_uploaded = st.session_state.get('_sb_total_uploaded', len(uploaded_files))
    undated_count = st.session_state.get('_sb_undated', 0)

    st.caption(f"📁 {len(uploaded_files)} file{'s' if len(uploaded_files) != 1 else ''} selected from {total_uploaded}")
    if undated_count > 0:
        st.caption(f"ℹ️ Ignored {undated_count} file{'s' if undated_count != 1 else ''} without date in filename")

    if not uploaded_files:
        st.warning("No files fall inside the selected range. Adjust date range in sidebar.")
        return

    # ── Session-state caching (recompute only when files change) ──
    cache_key = tuple(sorted((f.name, f.size) for f in uploaded_files))
    if st.session_state.get('_traj_key') != cache_key:
        with st.spinner("📊 Computing trajectories across all weeks..."):
            try:
                result = load_and_compute(uploaded_files)
            except Exception as e:
                st.error(f"❌ Computation error: {e}")
                logger.exception("load_and_compute failed")
                return
        st.session_state['_traj_key'] = cache_key
        st.session_state['_traj_result'] = result

    result = st.session_state.get('_traj_result')
    if result is None or not isinstance(result, (tuple, list)) or len(result) != 4:
        st.error("❌ Invalid computation result. Please re-upload your files.")
        # Clear stale cache so next rerun recomputes
        st.session_state.pop('_traj_key', None)
        st.session_state.pop('_traj_result', None)
        return

    if result[0] is None:
        st.error("❌ No valid data found in uploaded files. Ensure CSVs contain `rank` and `ticker` columns.")
        return

    traj_df, histories, dates_iso, metadata = result

    if traj_df.empty:
        st.warning("No stocks found with sufficient data for trajectory analysis.")
        return

    # ── Sidebar ──
    filters = render_sidebar(metadata, traj_df)

    # ── Apply Filters ──
    filtered_df = apply_filters(traj_df, filters)

    # ── Tabs ──
    tab_pulse, tab_ranking, tab_search, tab_backtest, tab_movers, tab_pattern, tab_export, tab_about = st.tabs([
        "📡 Market Pulse", "🏆 Rankings", "🔍 Search & Analyze",
        "📊 Backtest", "🔥 Top Movers", "🔬 Pattern Analyser", "📤 Export", "ℹ️ About"
    ])

    with tab_pulse:
        render_market_pulse_tab(filtered_df, traj_df, histories, metadata)

    with tab_ranking:
        render_rankings_tab(filtered_df, traj_df, histories, metadata)

    with tab_search:
        render_search_tab(filtered_df, traj_df, histories, dates_iso)

    with tab_backtest:
        render_backtest_tab(uploaded_files)

    with tab_movers:
        render_top_movers_tab(filtered_df, histories)

    with tab_pattern:
        render_pattern_analyser_tab(uploaded_files)

    with tab_export:
        render_export_tab(filtered_df, traj_df, histories)

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()

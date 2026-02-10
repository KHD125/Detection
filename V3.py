"""
WAVE Analyzer — System Intelligence Engine
═══════════════════════════════════════════
Companion to WAVE DETECTION 3.0.
Takes weekly CSV snapshots → adds the TIME dimension your scoring system lacks.

Answers: "Does my scoring system predict future gains? Which signals work?
          What should I buy THIS week? How should I size and exit?"

pip install streamlit pandas numpy plotly scipy
streamlit run wave_analyzer.py
"""

# ═══════════════════════════════════════════
#  IMPORTS
# ═══════════════════════════════════════════
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from collections import defaultdict
import re
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# ═══════════════════════════════════════════
#  PAGE CONFIG & STYLING
# ═══════════════════════════════════════════
st.set_page_config(
    page_title="WAVE Analyzer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 0.8rem; max-width: 1400px; }

/* Header */
.wave-header {
    text-align: center; padding: 0.4rem 0 0.2rem 0;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 12px; margin-bottom: 0.8rem;
    border: 1px solid #334155;
}
.wave-header h1 {
    font-size: 1.8rem !important; font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0; padding: 0;
}
.wave-header p { color: #94a3b8; font-size: 0.78rem; margin: 0; }

/* Metrics */
.verdict-box {
    padding: 1rem 1.5rem; border-radius: 10px; margin: 0.8rem 0;
    font-size: 0.95rem; font-weight: 600;
}
.verdict-good { background: #052e16; border: 1px solid #16a34a; color: #4ade80; }
.verdict-mid  { background: #422006; border: 1px solid #d97706; color: #fbbf24; }
.verdict-bad  { background: #450a0a; border: 1px solid #dc2626; color: #f87171; }

.big { font-size: 2.2rem; font-weight: 800; line-height: 1; }
.green { color: #4ade80; }
.red { color: #f87171; }
.amber { color: #fbbf24; }
.blue { color: #38bdf8; }
.muted { color: #64748b; font-size: 0.78rem; }
.signal-stack { font-size: 0.82rem; line-height: 1.6; }

/* Cards */
.stock-card {
    background: #0f172a; border: 1px solid #1e293b; border-radius: 10px;
    padding: 0.8rem 1rem; margin-bottom: 0.5rem;
}
.stock-card:hover { border-color: #38bdf8; }
.fresh-badge {
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 700;
}
.fresh-new { background: #065f46; color: #6ee7b7; }
.fresh-est { background: #1e3a5f; color: #93c5fd; }
.fresh-aging { background: #713f12; color: #fcd34d; }

/* Nav */
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] { padding: 0 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
#  DATA ENGINE
# ═══════════════════════════════════════════════════════════

NUMERIC_COLS = [
    'rank','master_score','position_score','volume_score','momentum_score',
    'acceleration_score','breakout_score','rvol_score','trend_quality',
    'price','pe','eps_current','eps_change_pct','from_low_pct','from_high_pct',
    'ret_1d','ret_7d','ret_30d','ret_3m','ret_6m','ret_1y',
    'rvol','vmi','money_flow_mm','position_tension','momentum_harmony',
    'overall_market_strength'
]

SCORE_COLS = ['master_score','position_score','volume_score','momentum_score',
              'acceleration_score','breakout_score','rvol_score','trend_quality']

WAVE_WEIGHTS = {
    'position_score': 0.27, 'volume_score': 0.23, 'momentum_score': 0.22,
    'breakout_score': 0.18, 'rvol_score': 0.10, 'acceleration_score': 0.0
}

PATTERN_TAGS = [
    ('CAT LEADER','cat_leader'), ('VOL EXPLOSION','vol_explosion'),
    ('MARKET LEADER','market_leader'), ('MOMENTUM WAVE','momentum_wave'),
    ('PREMIUM MOMENTUM','premium_momentum'), ('VELOCITY BREAKOUT','velocity_breakout'),
    ('INSTITUTIONAL TSUNAMI','institutional_tsunami'), ('INSTITUTIONAL','institutional'),
    ('GOLDEN CROSS','golden_cross'), ('STEALTH','stealth'),
    ('DISTRIBUTION','distribution'), ('CAPITULATION','capitulation'),
    ('HIGH PE','high_pe'), ('PHOENIX RISING','phoenix_rising'),
    ('PULLBACK SUPPORT','pullback_support'), ('RANGE COMPRESS','range_compress'),
    ('ACCELERATION','acceleration'), ('ROTATION LEADER','rotation_leader'),
    ('GARP LEADER','garp_leader'), ('VALUE MOMENTUM','value_momentum'),
    ('EARNINGS','earnings'), ('LIQUID LEADER','liquid_leader'),
    ('VELOCITY SQUEEZE','velocity_squeeze'), ('RUNAWAY GAP','runaway_gap'),
    ('PYRAMID','pyramid'), ('MOMENTUM DIVERGE','momentum_diverge'),
    ('VACUUM','vacuum'), ('PERFECT STORM','perfect_storm'),
    ('EXHAUSTION','exhaustion'), ('ENTROPY','entropy'),
    ('ATOMIC DECAY','atomic_decay'), ('INFORMATION DECAY','info_decay'),
]

RANK_BUCKETS = [(1, 20, 'Top 20'), (21, 50, 'Top 21-50'), (51, 100, 'Top 51-100'),
                (101, 200, '101-200'), (201, 500, '201-500'),
                (501, 1000, '501-1000'), (1001, 9999, '1000+')]

MARKET_STATES = ['STRONG_UPTREND','UPTREND','PULLBACK','ROTATION',
                 'SIDEWAYS','BOUNCE','DOWNTREND','STRONG_DOWNTREND']


@st.cache_data(show_spinner=False)
def load_weekly_csvs(uploaded_files):
    """Parse all uploaded weekly CSVs into a unified panel."""
    frames = []
    file_info = []
    for f in uploaded_files:
        try:
            df = pd.read_csv(f)
            m = re.search(r'(\d{4}-\d{2}-\d{2})', f.name)
            if not m:
                continue
            week = pd.to_datetime(m.group(1))
            df['week'] = week
            for c in NUMERIC_COLS:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
            frames.append(df)
            file_info.append({'file': f.name, 'date': week, 'stocks': len(df)})
        except Exception as e:
            st.warning(f"Skipped {f.name}: {e}")

    if not frames:
        return None, None, []

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(['ticker', 'week']).reset_index(drop=True)
    weeks = sorted(panel['week'].unique())
    return panel, weeks, file_info


@st.cache_data(show_spinner=False)
def build_enriched_panel(_panel, _weeks):
    """Add forward returns, rank deltas, entry freshness, pattern flags."""
    panel = _panel.copy()
    weeks = list(_weeks)
    n_weeks = len(weeks)
    week_idx = {w: i for i, w in enumerate(weeks)}

    # ── Fast lookup tables ──
    price_map = {}
    rank_map = {}
    score_map = {}
    for _, r in panel[['ticker','week','price','rank','master_score']].iterrows():
        key = (r['ticker'], r['week'])
        price_map[key] = r['price']
        rank_map[key] = r['rank']
        score_map[key] = r['master_score']

    # ── Compute per-row enrichments ──
    fwd_cols = {1: [], 2: [], 4: [], 8: [], 12: []}
    fwd_rank_cols = {1: [], 4: []}
    rank_delta_1w = []
    score_delta_1w = []
    rank_delta_2w = []

    for _, row in panel.iterrows():
        t, w, p, rk, sc = row['ticker'], row['week'], row['price'], row['rank'], row['master_score']
        wi = week_idx.get(w, -1)

        # Forward returns
        for fwd_n, col_list in fwd_cols.items():
            fwd_wi = wi + fwd_n
            if fwd_wi < n_weeks and p and p > 0:
                fp = price_map.get((t, weeks[fwd_wi]))
                col_list.append(((fp - p) / p * 100) if fp and fp > 0 else np.nan)
            else:
                col_list.append(np.nan)

        # Forward rank change
        for fwd_n, col_list in fwd_rank_cols.items():
            fwd_wi = wi + fwd_n
            if fwd_wi < n_weeks:
                fr = rank_map.get((t, weeks[fwd_wi]))
                col_list.append(rk - fr if fr else np.nan)  # +ve = rank improved
            else:
                col_list.append(np.nan)

        # Backward deltas (vs previous weeks)
        prev_rank = rank_map.get((t, weeks[wi - 1])) if wi > 0 else None
        prev_score = score_map.get((t, weeks[wi - 1])) if wi > 0 else None
        prev2_rank = rank_map.get((t, weeks[wi - 2])) if wi > 1 else None

        rank_delta_1w.append((prev_rank - rk) if prev_rank is not None else 0)
        score_delta_1w.append((sc - prev_score) if prev_score is not None else 0)
        rank_delta_2w.append((prev2_rank - rk) if prev2_rank is not None else 0)

    panel['fwd_1w'] = fwd_cols[1]
    panel['fwd_2w'] = fwd_cols[2]
    panel['fwd_4w'] = fwd_cols[4]
    panel['fwd_8w'] = fwd_cols[8]
    panel['fwd_12w'] = fwd_cols[12]
    panel['fwd_rank_1w'] = fwd_rank_cols[1]
    panel['fwd_rank_4w'] = fwd_rank_cols[4]
    panel['rank_delta_1w'] = rank_delta_1w
    panel['score_delta_1w'] = score_delta_1w
    panel['rank_delta_2w'] = rank_delta_2w

    # ── Pattern flags ──
    for tag, key in PATTERN_TAGS:
        panel[f'p_{key}'] = panel['patterns'].fillna('').str.contains(tag, case=False).astype(int)
    panel['n_patterns'] = panel['patterns'].fillna('').apply(
        lambda x: x.count('|') + 1 if x.strip() else 0
    )

    # ── Market state flags ──
    ms = panel['market_state'].fillna('')
    panel['is_uptrend'] = ms.isin(['UPTREND', 'STRONG_UPTREND']).astype(int)
    panel['is_downtrend'] = ms.isin(['DOWNTREND', 'STRONG_DOWNTREND']).astype(int)

    # ── Entry freshness ──
    panel['rank_bucket'] = pd.cut(
        panel['rank'], bins=[0, 20, 50, 100, 200, 500, 1000, 99999],
        labels=['Top20','21-50','51-100','101-200','201-500','501-1000','1000+']
    )

    # Compute weeks_in_top100 per stock
    weeks_in_top = []
    prev_top100 = {}  # ticker -> consecutive count
    for w in weeks:
        week_tickers = set(panel[(panel['week'] == w) & (panel['rank'] <= 100)]['ticker'])
        for t in panel[panel['week'] == w]['ticker'].unique():
            if t in week_tickers:
                prev_top100[t] = prev_top100.get(t, 0) + 1
            else:
                prev_top100[t] = 0

    # Map back
    wit_map = {}
    for w in weeks:
        wk_data = panel[panel['week'] == w]
        temp_top100 = {}
        for _, r in wk_data.iterrows():
            t = r['ticker']
            if t not in temp_top100:
                # Count consecutive weeks in top 100 up to this week
                count = 0
                wi = week_idx[w]
                for back in range(wi, -1, -1):
                    rk = rank_map.get((t, weeks[back]))
                    if rk is not None and rk <= 100:
                        count += 1
                    else:
                        break
                temp_top100[t] = count
            wit_map[(t, w)] = temp_top100[t]

    panel['weeks_in_top100'] = panel.apply(lambda r: wit_map.get((r['ticker'], r['week']), 0), axis=1)
    panel['is_fresh_entry'] = ((panel['weeks_in_top100'] <= 2) & (panel['weeks_in_top100'] > 0)).astype(int)

    return panel


# ═══════════════════════════════════════════════════════════
#  ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════

def compute_weekly_ic(panel, feature, fwd_col='fwd_4w'):
    """Spearman rank IC between a feature and forward returns, per week."""
    results = []
    for w in sorted(panel['week'].unique()):
        wk = panel[panel['week'] == w].dropna(subset=[feature, fwd_col])
        if len(wk) < 50:
            continue
        ic, pval = stats.spearmanr(wk[feature], wk[fwd_col])
        results.append({'week': w, 'IC': ic, 'p_value': pval, 'n': len(wk)})
    return pd.DataFrame(results)


def compute_quintile_analysis(panel, feature, fwd_col='fwd_4w'):
    """Split into quintiles by feature per week, measure avg forward return."""
    valid = panel.dropna(subset=[feature, fwd_col]).copy()
    if len(valid) < 200:
        return None, 0

    # Ascending = True for rank (lower=better), False for scores (higher=better)
    ascending = (feature == 'rank')
    valid['quintile'] = valid.groupby('week')[feature].transform(
        lambda x: pd.qcut(x.rank(method='first'), 5,
                          labels=[5,4,3,2,1] if ascending else [1,2,3,4,5])
    )
    valid['quintile'] = pd.to_numeric(valid['quintile'])

    qtable = valid.groupby('quintile').agg(
        count=(fwd_col, 'count'),
        mean_return=(fwd_col, 'mean'),
        median_return=(fwd_col, 'median'),
        win_rate=(fwd_col, lambda x: (x > 0).mean() * 100),
        big_win=(fwd_col, lambda x: (x > 10).mean() * 100),
        big_loss=(fwd_col, lambda x: (x < -10).mean() * 100),
    ).reset_index()

    returns = qtable['mean_return'].values
    mono = np.corrcoef(range(len(returns)), returns)[0, 1] if len(returns) >= 3 else 0
    return qtable, mono


def compute_pattern_edges(panel, fwd_col='fwd_4w'):
    """For each pattern, compute edge vs baseline with t-test."""
    valid = panel.dropna(subset=[fwd_col])
    if len(valid) < 100:
        return pd.DataFrame()

    baseline_ret = valid[fwd_col].mean()
    baseline_wr = (valid[fwd_col] > 0).mean() * 100

    results = []
    for tag, key in PATTERN_TAGS:
        col = f'p_{key}'
        if col not in valid.columns:
            continue
        with_p = valid[valid[col] == 1]
        without_p = valid[valid[col] == 0]
        if len(with_p) < 10:
            continue

        avg = with_p[fwd_col].mean()
        wr = (with_p[fwd_col] > 0).mean() * 100
        edge = avg - baseline_ret

        if len(with_p) >= 10 and len(without_p) >= 10:
            t, pval = stats.ttest_ind(with_p[fwd_col].dropna(), without_p[fwd_col].dropna(), equal_var=False)
        else:
            pval = 1.0

        results.append({
            'Pattern': tag, 'key': key, 'Count': len(with_p),
            'Avg Return %': round(avg, 2), 'Edge %': round(edge, 2),
            'Win Rate %': round(wr, 1), 'WR Edge': round(wr - baseline_wr, 1),
            'Big Win >10%': round((with_p[fwd_col] > 10).mean() * 100, 1),
            'Big Loss <-10%': round((with_p[fwd_col] < -10).mean() * 100, 1),
            'p-value': round(pval, 4),
            'Sig': '✅' if pval < 0.10 else '❌',
        })

    return pd.DataFrame(results).sort_values('Edge %', ascending=False)


def compute_transition_matrix(panel, horizon_weeks=4):
    """P(stock in rank bucket A → rank bucket B after N weeks)."""
    fwd_rank_col = f'fwd_rank_{"1w" if horizon_weeks == 1 else "4w"}'

    bucket_labels = [b[2] for b in RANK_BUCKETS]

    def get_bucket(rank):
        for lo, hi, label in RANK_BUCKETS:
            if lo <= rank <= hi:
                return label
        return '1000+'

    valid = panel.dropna(subset=['rank']).copy()
    weeks = sorted(valid['week'].unique())

    transitions = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)

    for _, row in valid.iterrows():
        t, w, rk = row['ticker'], row['week'], row['rank']
        wi = list(sorted(valid['week'].unique())).index(w) if w in weeks else -1

        fwd_wi = wi + horizon_weeks
        if fwd_wi >= len(weeks):
            continue

        # Get future rank
        fwd_w = weeks[fwd_wi]
        fwd_rows = valid[(valid['ticker'] == t) & (valid['week'] == fwd_w)]
        if len(fwd_rows) == 0:
            continue
        fwd_rk = fwd_rows.iloc[0]['rank']

        src = get_bucket(rk)
        dst = get_bucket(fwd_rk)
        transitions[src][dst] += 1
        totals[src] += 1

    # Build matrix
    matrix = pd.DataFrame(0.0, index=bucket_labels, columns=bucket_labels)
    for src in bucket_labels:
        if totals[src] > 0:
            for dst in bucket_labels:
                matrix.loc[src, dst] = transitions[src][dst] / totals[src] * 100

    counts = pd.Series({src: totals[src] for src in bucket_labels})
    return matrix, counts


def compute_survival_curve(panel, entry_threshold=50):
    """Of stocks entering top N for the first time, what % remain after 1,2,...,K weeks?"""
    weeks = sorted(panel['week'].unique())
    week_idx = {w: i for i, w in enumerate(weeks)}

    # Find first entry week for each stock
    first_entry = {}
    for w in weeks:
        wk = panel[(panel['week'] == w) & (panel['rank'] <= entry_threshold)]
        for t in wk['ticker'].unique():
            if t not in first_entry:
                first_entry[t] = w

    # For each entry, check how many subsequent weeks it stays in top N
    survival_data = defaultdict(list)
    rank_map = panel.set_index(['ticker', 'week'])['rank'].to_dict()

    for t, entry_week in first_entry.items():
        wi = week_idx[entry_week]
        for offset in range(0, min(16, len(weeks) - wi)):
            future_week = weeks[wi + offset]
            rk = rank_map.get((t, future_week))
            survived = 1 if rk is not None and rk <= entry_threshold else 0
            survival_data[offset].append(survived)

    curve = {}
    for offset, vals in sorted(survival_data.items()):
        if len(vals) >= 5:
            curve[offset] = np.mean(vals) * 100

    return pd.DataFrame({'weeks_after_entry': list(curve.keys()), 'survival_%': list(curve.values())})


def compute_market_state_edge(panel, fwd_col='fwd_4w'):
    """Average forward return by market_state."""
    valid = panel.dropna(subset=[fwd_col, 'market_state'])
    if len(valid) < 100:
        return pd.DataFrame()

    results = valid.groupby('market_state').agg(
        count=(fwd_col, 'count'),
        avg_return=(fwd_col, 'mean'),
        median_return=(fwd_col, 'median'),
        win_rate=(fwd_col, lambda x: (x > 0).mean() * 100),
    ).reset_index().sort_values('avg_return', ascending=False)
    return results


def compute_harmony_edge(panel, fwd_col='fwd_4w'):
    """Forward return by momentum_harmony level (0-4)."""
    valid = panel.dropna(subset=[fwd_col, 'momentum_harmony'])
    valid['mh'] = valid['momentum_harmony'].astype(int)
    if len(valid) < 100:
        return pd.DataFrame()

    return valid.groupby('mh').agg(
        count=(fwd_col, 'count'),
        avg_return=(fwd_col, 'mean'),
        win_rate=(fwd_col, lambda x: (x > 0).mean() * 100),
    ).reset_index().rename(columns={'mh': 'Harmony Level'})


def compute_entry_freshness_edge(panel, fwd_col='fwd_4w'):
    """Do fresh entries outperform established stocks?"""
    valid = panel[(panel['rank'] <= 100)].dropna(subset=[fwd_col]).copy()
    if len(valid) < 50:
        return pd.DataFrame()

    valid['freshness'] = valid['weeks_in_top100'].apply(
        lambda x: '🆕 Fresh (1-2 wks)' if x <= 2 else
                  '📊 Established (3-6 wks)' if x <= 6 else
                  '⏳ Aging (7+ wks)')

    return valid.groupby('freshness').agg(
        count=(fwd_col, 'count'),
        avg_return=(fwd_col, 'mean'),
        win_rate=(fwd_col, lambda x: (x > 0).mean() * 100),
    ).reset_index().sort_values('avg_return', ascending=False)


def compute_composite_entry_score(panel, weeks, pattern_edge_df, ic_data, fwd_col='fwd_4w'):
    """
    Data-driven composite score for latest week.
    Weights derived from IC analysis, pattern bonuses from actual edges.
    """
    latest_week = max(weeks)
    latest = panel[panel['week'] == latest_week].copy()
    if len(latest) == 0:
        return latest

    # ── Factor weights from IC ──
    factor_ics = {}
    for feat in SCORE_COLS:
        ic_df = ic_data.get(feat)
        if ic_df is not None and len(ic_df) > 0:
            factor_ics[feat] = max(0, ic_df['IC'].median())  # Only positive IC
        else:
            factor_ics[feat] = 0

    total_ic = sum(factor_ics.values())
    if total_ic > 0:
        ic_weights = {k: v / total_ic for k, v in factor_ics.items()}
    else:
        # Fallback to WAVE DETECTION weights
        ic_weights = {k: WAVE_WEIGHTS.get(k, 0.05) for k in SCORE_COLS}
        tot = sum(ic_weights.values())
        ic_weights = {k: v / tot for k, v in ic_weights.items()}

    # Weighted score (0-100)
    latest['composite'] = sum(
        latest[feat].fillna(0) * w for feat, w in ic_weights.items()
    )

    # ── Pattern bonus (from actual edges) ──
    if len(pattern_edge_df) > 0:
        sig_patterns = pattern_edge_df[pattern_edge_df['p-value'] < 0.15]
        bonus = pd.Series(0.0, index=latest.index)
        for _, pr in sig_patterns.iterrows():
            col = f"p_{pr['key']}"
            if col in latest.columns:
                edge_bonus = np.clip(pr['Edge %'] * 0.5, -3, 3)
                bonus += latest[col] * edge_bonus
        latest['composite'] += np.clip(bonus, -10, 10)

    # ── Entry freshness bonus ──
    latest['composite'] += latest['is_fresh_entry'] * 3
    latest.loc[latest['weeks_in_top100'] > 8, 'composite'] -= 2

    # ── Rank velocity bonus ──
    rv = latest['rank_delta_1w'].fillna(0)
    latest['composite'] += np.clip(rv / 100, -3, 5)

    # ── Uptrend bonus ──
    latest['composite'] += latest['is_uptrend'] * 2
    latest['composite'] -= latest['is_downtrend'] * 3

    # Normalize to 0-100
    cmin, cmax = latest['composite'].min(), latest['composite'].max()
    if cmax > cmin:
        latest['composite'] = ((latest['composite'] - cmin) / (cmax - cmin) * 100).round(1)
    else:
        latest['composite'] = 50.0

    # Percentile rank
    latest['composite_pctile'] = latest['composite'].rank(pct=True) * 100

    # Confidence tier
    latest['confidence'] = latest.apply(lambda r: (
        '🟢 HIGH' if r['composite_pctile'] >= 90 and r.get('is_uptrend', 0) == 1 else
        '🟢 HIGH' if r['composite_pctile'] >= 95 else
        '🟡 MEDIUM' if r['composite_pctile'] >= 75 else
        '🟠 LOW' if r['composite_pctile'] >= 50 else
        '🔴 AVOID'
    ), axis=1)

    # Freshness label
    latest['freshness_label'] = latest['weeks_in_top100'].apply(
        lambda x: '🆕 NEW' if 0 < x <= 2 else '📊 ESTAB' if x <= 6 else '⏳ AGING' if x > 6 else '—'
    )

    return latest.sort_values('composite', ascending=False)


def walk_forward_backtest(panel, weeks, top_n=20, rebalance=1, min_score=0, fwd_col='fwd_4w'):
    """Walk-forward: each period picks top-N using ONLY past + current data."""
    results = []
    fwd_weeks = int(fwd_col.replace('fwd_', '').replace('w', ''))

    for i in range(0, len(weeks) - fwd_weeks):
        w = weeks[i]
        wk = panel[panel['week'] == w].copy()
        if min_score > 0:
            wk = wk[wk['master_score'] >= min_score]

        wk_valid = wk.dropna(subset=[fwd_col])
        if len(wk_valid) < top_n * 2:
            continue

        top = wk_valid.nsmallest(top_n, 'rank')
        bottom = wk_valid.nlargest(top_n, 'rank')
        universe = wk_valid

        results.append({
            'week': w,
            'top_avg': top[fwd_col].mean(),
            'top_median': top[fwd_col].median(),
            'top_wr': (top[fwd_col] > 0).mean() * 100,
            'bottom_avg': bottom[fwd_col].mean(),
            'bottom_wr': (bottom[fwd_col] > 0).mean() * 100,
            'universe_avg': universe[fwd_col].mean(),
            'spread': top[fwd_col].mean() - bottom[fwd_col].mean(),
            'top_tickers': ', '.join(top['ticker'].head(5).tolist()),
        })

    return pd.DataFrame(results)


def get_sector_rotation(panel, weeks, top_n=100):
    """% of top-N stocks from each sector, per week."""
    records = []
    for w in weeks:
        wk = panel[(panel['week'] == w) & (panel['rank'] <= top_n)]
        total = len(wk)
        if total == 0:
            continue
        for sector, count in wk['sector'].value_counts().items():
            records.append({'week': w, 'sector': sector, 'pct': count / total * 100, 'count': count})
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════
#  SIDEBAR & NAVIGATION
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("# 🌊 WAVE Analyzer")
    st.caption("System Intelligence Engine")
    st.markdown("---")

    uploaded_files = st.file_uploader(
        "📁 Upload Weekly CSVs",
        type=['csv'], accept_multiple_files=True,
        help="Upload your Stocks_Weekly_*.csv files (5+ recommended)"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files")

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        horizon = st.radio("⏱️ Forward Horizon", ['1 Week','2 Weeks','4 Weeks','8 Weeks','12 Weeks'],
                           index=2, horizontal=True)
        fwd_col = {'1 Week':'fwd_1w','2 Weeks':'fwd_2w','4 Weeks':'fwd_4w',
                    '8 Weeks':'fwd_8w','12 Weeks':'fwd_12w'}[horizon]

        top_n = st.slider("Top N for strategies", 5, 50, 20)

        st.markdown("---")
        st.markdown("### 🔍 Filters")
        min_score_filter = st.slider("Min Master Score", 0, 80, 0, 5)
        max_rank_filter = st.slider("Max Rank", 50, 2200, 2200, 50)

        cat_filter = st.multiselect("Category", ['Mega Cap','Large Cap','Mid Cap','Small Cap','Micro Cap'])
        sector_input = st.text_input("Sector contains", "")


# ═══════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="wave-header">
    <h1>🌊 WAVE Analyzer</h1>
    <p>System Intelligence Engine — Does your scoring system predict future gains?</p>
</div>
""", unsafe_allow_html=True)


if not uploaded_files:
    st.markdown("---")
    st.markdown("""
    ## How to Use

    **1.** Upload your `Stocks_Weekly_*.csv` files (sidebar)
    **2.** Explore 6 pages of temporal intelligence

    | Page | Purpose |
    |------|---------|
    | **📊 System Health** | Does your master_score ACTUALLY predict gains? IC analysis proves it. |
    | **⚡ Signal Lab** | Which patterns/states have real statistical edge? |
    | **🎯 This Week's Picks** | Data-driven stock picks with confidence tiers |
    | **🔄 Rank Dynamics** | Transition matrix — where do top stocks end up? |
    | **🔬 Stock X-Ray** | Any stock's full weekly journey |
    | **🧪 Backtest Lab** | Walk-forward proof — would this have worked? |

    ---

    **Why this exists:** Your WAVE DETECTION system scores 2100+ stocks brilliantly at each point in time.
    But it has **zero temporal memory** — it doesn't know if a stock just entered the top ranks (strong signal)
    or has been there for 12 weeks (potentially exhausted). This app adds the **time dimension**.

    **Minimum:** 3 weekly CSVs &nbsp;|&nbsp; **Recommended:** 10+ &nbsp;|&nbsp; **Best:** 23+ weeks

    ```
    pip install streamlit pandas numpy plotly scipy
    streamlit run wave_analyzer.py
    ```
    """)
    st.stop()

if len(uploaded_files) < 3:
    st.warning("Upload at least 3 weekly CSV files for meaningful analysis.")
    st.stop()


# ── Load & Process ──
with st.spinner("Loading weekly snapshots..."):
    panel, weeks, file_info = load_weekly_csvs(uploaded_files)

if panel is None:
    st.error("Could not parse any CSV files. Check file format.")
    st.stop()

with st.spinner("Computing forward returns & enrichments..."):
    panel = build_enriched_panel(panel, weeks)

latest_week = max(weeks)
n_weeks = len(weeks)

# ── Pre-compute analytics (cached internally) ──
with st.spinner("Running analytics..."):
    # IC for all score columns
    ic_data = {}
    for feat in SCORE_COLS:
        ic_data[feat] = compute_weekly_ic(panel, feat, fwd_col)

    # Pattern edges
    pattern_edges = compute_pattern_edges(panel, fwd_col)

    # Composite entry score for latest week
    latest_scored = compute_composite_entry_score(panel, weeks, pattern_edges, ic_data, fwd_col)


# ═══════════════════════════════════════════════════════════
#  NAVIGATION
# ═══════════════════════════════════════════════════════════

page = st.radio(
    "nav", ["📊 System Health", "⚡ Signal Lab", "🎯 This Week's Picks",
            "🔄 Rank Dynamics", "🔬 Stock X-Ray", "🧪 Backtest Lab"],
    horizontal=True, label_visibility="collapsed"
)

# Quick stats bar
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Weeks", n_weeks)
with c2:
    n_stocks = panel[panel['week'] == latest_week]['ticker'].nunique()
    st.metric("Stocks", f"{n_stocks:,}")
with c3:
    st.metric("Range", f"{weeks[0].strftime('%b %d')} → {weeks[-1].strftime('%b %d, %Y')}")
with c4:
    avg_mkt = panel[panel['week'] == latest_week]['overall_market_strength'].mean()
    st.metric("Mkt Strength", f"{avg_mkt:.1f}")
with c5:
    st.metric("Horizon", horizon)

st.markdown("---")


# ═══════════════════════════════════════════════════════════
#  PAGE 1: SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════
if page == "📊 System Health":
    st.markdown(f"## 📊 System Health — Does Your Scoring Predict {horizon} Gains?")

    # ── Master Score IC ──
    ms_ic = ic_data.get('master_score', pd.DataFrame())

    if len(ms_ic) > 0:
        median_ic = ms_ic['IC'].median()
        pct_positive = (ms_ic['IC'] > 0).mean() * 100
        t_stat, t_pval = stats.ttest_1samp(ms_ic['IC'].dropna(), 0)

        # Verdict
        if median_ic > 0.05 and pct_positive > 60:
            verdict_class = 'verdict-good'
            verdict_text = f"✅ SYSTEM WORKS — Median IC = {median_ic:.4f}, positive in {pct_positive:.0f}% of weeks (p={t_pval:.4f}). Your master_score reliably predicts {horizon} forward returns."
        elif median_ic > 0.02 and pct_positive > 50:
            verdict_class = 'verdict-mid'
            verdict_text = f"⚠️ MODERATE SIGNAL — Median IC = {median_ic:.4f}, positive in {pct_positive:.0f}% of weeks. Signal exists but is noisy. More data will clarify."
        else:
            verdict_class = 'verdict-bad'
            verdict_text = f"❌ WEAK/NO SIGNAL — Median IC = {median_ic:.4f}, positive in {pct_positive:.0f}% of weeks. master_score doesn't reliably predict {horizon} returns at this data volume."

        st.markdown(f'<div class="verdict-box {verdict_class}">{verdict_text}</div>', unsafe_allow_html=True)

        # IC chart
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = go.Figure()
            colors = ['#4ade80' if x > 0 else '#f87171' for x in ms_ic['IC']]
            fig.add_trace(go.Bar(x=ms_ic['week'], y=ms_ic['IC'], marker_color=colors, name='Weekly IC'))
            fig.add_hline(y=median_ic, line_dash="dash", line_color="#38bdf8",
                         annotation_text=f"Median: {median_ic:.4f}")
            fig.add_hline(y=0, line_color="#475569")
            fig.update_layout(title=f'master_score IC vs {horizon} Forward Return (per week)',
                             height=350, template='plotly_dark', yaxis_title='Information Coefficient',
                             margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### IC Summary")
            st.metric("Median IC", f"{median_ic:.4f}")
            st.metric("Mean IC", f"{ms_ic['IC'].mean():.4f}")
            st.metric("% Weeks Positive", f"{pct_positive:.0f}%")
            st.metric("t-statistic", f"{t_stat:.2f}")
            st.metric("p-value", f"{t_pval:.4f}")
            st.caption("IC > 0.03 = signal exists")
            st.caption("IC > 0.05 = strong signal")
            st.caption("p-value < 0.05 = statistically significant")

    st.markdown("---")

    # ── Factor Decomposition ──
    st.markdown("### 🔍 Factor IC Decomposition — Which Sub-Scores Carry the Edge?")
    st.caption("Compares each of your 7 component scores. WAVE DETECTION weights: Position 27%, Volume 23%, Momentum 22%, Breakout 18%, RVOL 10%")

    factor_summary = []
    for feat in SCORE_COLS:
        ic_df = ic_data.get(feat, pd.DataFrame())
        if len(ic_df) > 0:
            med_ic = ic_df['IC'].median()
            wave_w = WAVE_WEIGHTS.get(feat, 0)
            factor_summary.append({
                'Factor': feat.replace('_score','').replace('_',' ').title(),
                'Median IC': round(med_ic, 4),
                'Mean IC': round(ic_df['IC'].mean(), 4),
                '% Positive': round((ic_df['IC'] > 0).mean() * 100, 0),
                'Current Weight': f"{wave_w:.0%}",
                'IC Suggests': '⬆️ Increase' if med_ic > 0.04 and wave_w < 0.25 else
                               '⬇️ Decrease' if med_ic < 0.01 and wave_w > 0.10 else
                               '✅ OK',
            })

    if factor_summary:
        fsum = pd.DataFrame(factor_summary).sort_values('Median IC', ascending=False)

        fig = px.bar(fsum, x='Factor', y='Median IC', color='Median IC',
                     color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                     title='Median IC by Factor (higher = more predictive)',
                     hover_data=['% Positive','Current Weight'])
        fig.add_hline(y=0, line_color="#475569")
        fig.update_layout(height=350, template='plotly_dark', xaxis_tickangle=-30,
                         margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(fsum.reset_index(drop=True), use_container_width=True, hide_index=True)

        # Optimal weights suggestion
        positive_factors = {row['Factor']: max(0, row['Median IC']) for _, row in fsum.iterrows()}
        total_pos = sum(positive_factors.values())
        if total_pos > 0:
            suggested = {k: round(v / total_pos * 100) for k, v in positive_factors.items()}
            st.markdown("#### 💡 Suggested Optimal Weights (based on IC)")
            cols = st.columns(len(suggested))
            for i, (k, v) in enumerate(sorted(suggested.items(), key=lambda x: -x[1])):
                with cols[i]:
                    st.metric(k, f"{v}%")

    st.markdown("---")

    # ── Quintile Spread ──
    st.markdown("### 📊 Quintile Analysis — Top vs Bottom")
    st.caption("All stocks split into 5 equal groups by master_score each week. Q5 = highest scores, Q1 = lowest.")

    qt, mono = compute_quintile_analysis(panel, 'master_score', fwd_col)
    if qt is not None:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(qt, x='quintile', y='mean_return', color='mean_return',
                         color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                         title=f'Average {horizon} Forward Return by Score Quintile',
                         labels={'quintile': 'Quintile (1=worst, 5=best)', 'mean_return': 'Avg Return %'})
            fig.update_layout(height=350, template='plotly_dark', margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(qt, x='quintile', y='win_rate', color='win_rate',
                         color_continuous_scale='RdYlGn', color_continuous_midpoint=50,
                         title='Win Rate by Quintile',
                         labels={'quintile': 'Quintile', 'win_rate': 'Win Rate %'})
            fig.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
            fig.update_layout(height=350, template='plotly_dark', margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)

        st.metric("Monotonicity Score", f"{mono:.3f}",
                  help="1.0 = quintiles perfectly predict returns. >0.7 strong, >0.4 moderate")
        st.dataframe(qt.round(2), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 2: SIGNAL LAB
# ═══════════════════════════════════════════════════════════
elif page == "⚡ Signal Lab":
    st.markdown(f"## ⚡ Signal Lab — What Actually Works? ({horizon})")

    # ── Pattern Edges ──
    st.markdown("### 🏷️ Pattern Edge Analysis")
    st.caption("Each pattern tested: does it predict gains beyond baseline? p-value < 0.10 = statistically significant.")

    if len(pattern_edges) > 0:
        valid = panel.dropna(subset=[fwd_col])
        base_ret = valid[fwd_col].mean()
        base_wr = (valid[fwd_col] > 0).mean() * 100
        n_sig = (pattern_edges['p-value'] < 0.10).sum()

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Baseline Avg Return", f"{base_ret:.2f}%")
        with c2: st.metric("Baseline Win Rate", f"{base_wr:.1f}%")
        with c3: st.metric("Significant Patterns", f"{n_sig}/{len(pattern_edges)}")

        fig = px.bar(pattern_edges, x='Pattern', y='Edge %', color='Edge %',
                     color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                     title=f'Pattern Edge: Extra Return vs Baseline ({horizon})',
                     hover_data=['Count','Win Rate %','p-value','Sig'], text='Sig')
        fig.add_hline(y=0, line_color="#475569")
        fig.update_layout(height=450, template='plotly_dark', xaxis_tickangle=-50,
                         margin=dict(t=40, b=80))
        st.plotly_chart(fig, use_container_width=True)

        # Win rate chart
        fig2 = px.bar(pattern_edges, x='Pattern', y='Win Rate %', color='WR Edge',
                      color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                      title='Win Rate by Pattern', hover_data=['Count','Edge %'])
        fig2.add_hline(y=base_wr, line_dash="dash", annotation_text=f"Baseline: {base_wr:.0f}%")
        fig2.update_layout(height=400, template='plotly_dark', xaxis_tickangle=-50,
                          margin=dict(t=40, b=80))
        st.plotly_chart(fig2, use_container_width=True)

        with st.expander("📋 Full Pattern Statistics Table"):
            st.dataframe(pattern_edges.drop(columns=['key']).reset_index(drop=True),
                        use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Pattern Combinations ──
    st.markdown("### 🔗 Pattern Combinations")
    combos = [
        ('p_stealth','p_institutional','STEALTH + INSTITUTIONAL'),
        ('p_cat_leader','p_market_leader','CAT LEADER + MARKET LEADER'),
        ('p_velocity_breakout','p_premium_momentum','VELOCITY BREAK + PREMIUM MOM'),
        ('p_golden_cross','p_momentum_wave','GOLDEN CROSS + MOMENTUM WAVE'),
        ('p_cat_leader','p_vol_explosion','CAT LEADER + VOL EXPLOSION'),
        ('p_institutional','p_golden_cross','INSTITUTIONAL + GOLDEN CROSS'),
        ('p_stealth','p_acceleration','STEALTH + ACCELERATION'),
        ('p_garp_leader','p_value_momentum','GARP + VALUE MOMENTUM'),
        ('p_liquid_leader','p_market_leader','LIQUID LEADER + MARKET LEADER'),
        ('p_institutional_tsunami','p_liquid_leader','INST TSUNAMI + LIQUID LEADER'),
    ]

    valid = panel.dropna(subset=[fwd_col])
    combo_results = []
    for c1_col, c2_col, label in combos:
        if c1_col in valid.columns and c2_col in valid.columns:
            combo = valid[(valid[c1_col] == 1) & (valid[c2_col] == 1)]
            if len(combo) >= 5:
                base = valid[fwd_col].mean()
                combo_results.append({
                    'Combination': label, 'Count': len(combo),
                    'Avg Return %': round(combo[fwd_col].mean(), 2),
                    'Edge %': round(combo[fwd_col].mean() - base, 2),
                    'Win Rate %': round((combo[fwd_col] > 0).mean() * 100, 1),
                })

    if combo_results:
        combo_df = pd.DataFrame(combo_results).sort_values('Edge %', ascending=False)
        st.dataframe(combo_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Market State Edge ──
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌐 Market State Edge")
        ms_edge = compute_market_state_edge(panel, fwd_col)
        if len(ms_edge) > 0:
            fig = px.bar(ms_edge, x='market_state', y='avg_return', color='avg_return',
                         color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                         title=f'{horizon} Return by Market State',
                         hover_data=['count','win_rate'])
            fig.add_hline(y=0, line_color="#475569")
            fig.update_layout(height=350, template='plotly_dark', xaxis_tickangle=-30,
                             margin=dict(t=40, b=50))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(ms_edge.round(2).reset_index(drop=True), use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🎵 Momentum Harmony Edge")
        mh_edge = compute_harmony_edge(panel, fwd_col)
        if len(mh_edge) > 0:
            fig = px.bar(mh_edge, x='Harmony Level', y='avg_return', color='avg_return',
                         color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                         title=f'{horizon} Return by Harmony Level (0-4)',
                         hover_data=['count','win_rate'])
            fig.update_layout(height=350, template='plotly_dark', margin=dict(t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(mh_edge.round(2).reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Entry Freshness Edge ──
    st.markdown("### 🆕 Entry Freshness Edge")
    st.caption("Do stocks that JUST entered Top 100 outperform those sitting there for weeks?")

    fr_edge = compute_entry_freshness_edge(panel, fwd_col)
    if len(fr_edge) > 0:
        fig = px.bar(fr_edge, x='freshness', y='avg_return', color='avg_return',
                     color_continuous_scale='RdYlGn', color_continuous_midpoint=0,
                     title=f'Fresh vs Established vs Aging — {horizon} Avg Return',
                     hover_data=['count','win_rate'])
        fig.update_layout(height=300, template='plotly_dark', margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(fr_edge.round(2).reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Score Threshold Analysis ──
    st.markdown("---")
    st.markdown("### 📈 Score Threshold Analysis")
    st.caption("What's the minimum master_score that matters?")

    valid = panel.dropna(subset=[fwd_col])
    thresholds = [0, 20, 30, 40, 50, 60, 70, 80]
    thresh_results = []
    for th in thresholds:
        subset = valid[valid['master_score'] >= th]
        if len(subset) >= 20:
            thresh_results.append({
                'Min Score': th, 'Stocks': len(subset),
                'Avg Return %': round(subset[fwd_col].mean(), 2),
                'Win Rate %': round((subset[fwd_col] > 0).mean() * 100, 1),
            })

    if thresh_results:
        th_df = pd.DataFrame(thresh_results)
        fig = px.line(th_df, x='Min Score', y='Avg Return %', markers=True,
                      title=f'Average {horizon} Return at Different Score Cutoffs')
        fig.update_layout(height=300, template='plotly_dark', margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(th_df.reset_index(drop=True), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 3: THIS WEEK'S PICKS
# ═══════════════════════════════════════════════════════════
elif page == "🎯 This Week's Picks":
    st.markdown(f"## 🎯 This Week's Picks — {latest_week.strftime('%B %d, %Y')}")
    st.caption("Composite score = data-driven weights (from IC analysis) + pattern bonuses (from tested edges) + entry freshness + rank velocity")

    picks = latest_scored.copy()

    # Apply sidebar filters
    if min_score_filter > 0:
        picks = picks[picks['master_score'] >= min_score_filter]
    if max_rank_filter < 2200:
        picks = picks[picks['rank'] <= max_rank_filter]
    if cat_filter:
        picks = picks[picks['category'].isin(cat_filter)]
    if sector_input:
        picks = picks[picks['sector'].str.contains(sector_input, case=False, na=False)]

    # Market overview
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        up_pct = picks['is_uptrend'].mean() * 100
        st.metric("% Uptrend", f"{up_pct:.0f}%")
    with c2:
        n_fresh = picks['is_fresh_entry'].sum()
        st.metric("Fresh Top-100 Entries", int(n_fresh))
    with c3:
        high_conf = (picks['confidence'] == '🟢 HIGH').sum()
        st.metric("High Confidence Picks", int(high_conf))
    with c4:
        avg_comp = picks.head(top_n)['composite'].mean()
        st.metric(f"Avg Composite (Top {top_n})", f"{avg_comp:.1f}")

    st.markdown("---")

    # ── Rank Velocity Radar ──
    st.markdown("### 🚀 Rank Velocity — Fastest Movers This Week")
    st.caption("Stocks with biggest rank improvement — freshest buy signals")

    movers = picks.nlargest(10, 'rank_delta_1w')
    mover_cols = st.columns(5)
    for i, (_, r) in enumerate(movers.head(10).iterrows()):
        with mover_cols[i % 5]:
            delta = r['rank_delta_1w']
            st.markdown(f"""
            <div class="stock-card">
                <b>{r['ticker']}</b><br>
                <span class="big green">+{delta:.0f}</span><br>
                <span class="muted">Rank #{r['rank']:.0f} | ₹{r['price']:.0f}</span><br>
                <span class="muted">{r.get('sector','')[:20]}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Top Picks Table ──
    st.markdown(f"### 🏆 Top {top_n} Picks by Composite Score")

    top_picks = picks.head(top_n)

    for i, (_, row) in enumerate(top_picks.iterrows()):
        comp = row.get('composite', 0)
        conf = row.get('confidence', '🟠 LOW')
        fresh = row.get('freshness_label', '—')

        fresh_class = 'fresh-new' if '🆕' in fresh else 'fresh-aging' if '⏳' in fresh else 'fresh-est'

        cols = st.columns([0.4, 2.2, 1, 1, 1, 1, 1.5])

        with cols[0]:
            st.markdown(f"**#{i+1}**")
        with cols[1]:
            st.markdown(f"**{row['ticker']}**")
            st.caption(f"{str(row.get('company_name',''))[:35]} · {row.get('sector','')}")
        with cols[2]:
            color = 'green' if comp > 70 else 'amber' if comp > 50 else 'red'
            st.markdown(f"<span class='big {color}'>{comp:.0f}</span><br><span class='muted'>Composite</span>", unsafe_allow_html=True)
        with cols[3]:
            rdelta = row.get('rank_delta_1w', 0)
            st.metric("Rank", f"#{row['rank']:.0f}", f"{rdelta:+.0f}")
        with cols[4]:
            sdelta = row.get('score_delta_1w', 0)
            st.metric("Score", f"{row['master_score']:.0f}", f"{sdelta:+.1f}")
        with cols[5]:
            st.metric("Price", f"₹{row['price']:.0f}", f"{row.get('ret_7d',0):+.1f}% 7d")
        with cols[6]:
            st.markdown(f"**{conf}**")
            st.markdown(f"<span class='fresh-badge {fresh_class}'>{fresh}</span> · {row.get('market_state','')}", unsafe_allow_html=True)

        # Pattern stack
        patterns = str(row.get('patterns', ''))
        if patterns and patterns != 'nan' and len(patterns) > 3:
            # Annotate patterns with their edge values
            annotated = []
            for tag, key in PATTERN_TAGS:
                if tag in patterns and len(pattern_edges) > 0:
                    edge_row = pattern_edges[pattern_edges['key'] == key]
                    if len(edge_row) > 0:
                        e = edge_row.iloc[0]['Edge %']
                        sig = edge_row.iloc[0]['Sig']
                        color = 'green' if e > 0 else 'red'
                        annotated.append(f"<span class='{color}'>{tag} ({e:+.1f}% {sig})</span>")
                elif tag in patterns:
                    annotated.append(tag)

            if annotated:
                st.markdown(f"<div class='signal-stack'>{'  ·  '.join(annotated)}</div>", unsafe_allow_html=True)

        st.markdown("---")

    # ── Sector Allocation ──
    st.markdown("### 🗺️ Sector Allocation of Top Picks")
    sector_counts = top_picks['sector'].value_counts().reset_index()
    sector_counts.columns = ['sector', 'count']
    fig = px.treemap(sector_counts, path=['sector'], values='count',
                     title=f'Top {top_n} Picks by Sector',
                     color='count', color_continuous_scale='Blues')
    fig.update_layout(height=350, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # ── Risk Flags ──
    st.markdown("### 🚨 Risk Flags in Top Picks")
    risk_stocks = top_picks[
        (top_picks.get('p_capitulation', 0) == 1) |
        (top_picks.get('p_distribution', 0) == 1) |
        (top_picks['is_downtrend'] == 1) |
        (top_picks['rank_delta_1w'] < -200)
    ]
    if len(risk_stocks) > 0:
        st.warning(f"⚠️ {len(risk_stocks)} stocks in Top {top_n} have risk flags:")
        for _, r in risk_stocks.iterrows():
            flags = []
            if r.get('p_capitulation', 0) == 1: flags.append("💣 CAPITULATION")
            if r.get('p_distribution', 0) == 1: flags.append("📊 DISTRIBUTION")
            if r['is_downtrend'] == 1: flags.append("📉 DOWNTREND")
            if r['rank_delta_1w'] < -200: flags.append(f"💀 Rank crashed {r['rank_delta_1w']:.0f}")
            st.markdown(f"- **{r['ticker']}** — {' | '.join(flags)}")
    else:
        st.success("✅ No major risk flags in top picks")

    # ── Download ──
    st.markdown("---")
    dl_cols = ['ticker','company_name','composite','confidence','freshness_label',
               'rank','rank_delta_1w','master_score','score_delta_1w',
               'price','ret_7d','ret_30d','ret_3m','market_state','patterns',
               'sector','category','weeks_in_top100']
    dl_available = [c for c in dl_cols if c in picks.columns]
    csv_data = picks[dl_available].to_csv(index=False)
    st.download_button("📥 Download Full Picks CSV", csv_data, "wave_picks.csv", "text/csv")


# ═══════════════════════════════════════════════════════════
#  PAGE 4: RANK DYNAMICS
# ═══════════════════════════════════════════════════════════
elif page == "🔄 Rank Dynamics":
    st.markdown("## 🔄 Rank Dynamics — Where Do Top Stocks End Up?")

    # ── Transition Matrix ──
    st.markdown("### 🔢 Rank Transition Matrix")

    tm_horizon = st.radio("Transition horizon", ['1 week', '4 weeks'], horizontal=True)
    h = 1 if '1' in tm_horizon else 4

    with st.spinner("Computing transition matrix..."):
        matrix, counts = compute_transition_matrix(panel, h)

    st.caption(f"Each cell = P(stock in row bucket this week → column bucket after {tm_horizon}). Diagonal = stocks that STAYED in same bucket.")

    fig = px.imshow(
        matrix.values.round(1), x=matrix.columns, y=matrix.index,
        color_continuous_scale='YlOrRd', text_auto='.0f',
        labels={'x': f'After {tm_horizon}', 'y': 'Current Bucket', 'color': 'Probability %'},
        title=f'Rank Transition Probabilities ({tm_horizon})'
    )
    fig.update_layout(height=450, template='plotly_dark', margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # Key insights
    diag = [matrix.iloc[i, i] for i in range(min(len(matrix), len(matrix.columns)))]
    st.markdown("#### 💡 Key Insights")
    bucket_labels = [b[2] for b in RANK_BUCKETS]
    for i, label in enumerate(bucket_labels[:len(diag)]):
        stability = diag[i]
        emoji = '🟢' if stability > 50 else '🟡' if stability > 30 else '🔴'
        st.markdown(f"- {emoji} **{label}**: {stability:.0f}% stay in same bucket after {tm_horizon}")

    st.markdown("---")

    # ── Survival Curves ──
    st.markdown("### 📉 Survival Curve — How Long Do Top Stocks Stay?")
    st.caption("Of stocks entering top N for the FIRST time, what % remain after K weeks?")

    surv_threshold = st.slider("Top N threshold", 20, 200, 50, 10)
    survival = compute_survival_curve(panel, surv_threshold)

    if len(survival) > 0:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=survival['weeks_after_entry'], y=survival['survival_%'],
            mode='lines+markers', line=dict(color='#38bdf8', width=3),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.1)'
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="#64748b", annotation_text="50% threshold")
        fig.update_layout(
            title=f'Survival: % of stocks remaining in Top {surv_threshold} after entry',
            xaxis_title='Weeks After First Entry', yaxis_title='% Still in Top N',
            height=350, template='plotly_dark', yaxis=dict(range=[0, 105]),
            margin=dict(t=40, b=30)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Half-life
        half_life = survival[survival['survival_%'] < 50]
        if len(half_life) > 0:
            hl_week = half_life.iloc[0]['weeks_after_entry']
            st.info(f"📊 **Half-life: {hl_week:.0f} weeks** — 50% of stocks that enter Top {surv_threshold} drop out within {hl_week:.0f} weeks. {'This suggests you should rebalance more frequently.' if hl_week < 6 else 'Your system has good staying power.'}")
        else:
            st.success(f"💪 Most stocks that enter Top {surv_threshold} stay there throughout the data period. Strong system stickiness.")

    st.markdown("---")

    # ── Biggest Movers ──
    st.markdown("### 🔄 Biggest Rank Movers This Week")

    latest = panel[panel['week'] == latest_week].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🚀 Top 20 Rank Improvers")
        improvers = latest.nlargest(20, 'rank_delta_1w')
        for i, (_, r) in enumerate(improvers.iterrows()):
            st.markdown(
                f"**{i+1}. {r['ticker']}** | #{r['rank']:.0f} "
                f"(+{r['rank_delta_1w']:.0f}) | Score {r['master_score']:.0f} "
                f"| ₹{r['price']:.0f} | {r.get('market_state','')}"
            )

    with col2:
        st.markdown("#### 💀 Top 20 Rank Crashers")
        crashers = latest.nsmallest(20, 'rank_delta_1w')
        for i, (_, r) in enumerate(crashers.iterrows()):
            st.markdown(
                f"**{i+1}. {r['ticker']}** | #{r['rank']:.0f} "
                f"({r['rank_delta_1w']:.0f}) | Score {r['master_score']:.0f} "
                f"| ₹{r['price']:.0f} | {r.get('market_state','')}"
            )

    st.markdown("---")

    # ── Sector Rotation ──
    st.markdown("### 🔄 Sector Rotation Over Time")
    st.caption("Which sectors dominated Top 100 each week?")

    rotation = get_sector_rotation(panel, weeks, 100)
    if len(rotation) > 0:
        top_sectors = rotation.groupby('sector')['pct'].mean().nlargest(8).index.tolist()
        rot_filtered = rotation[rotation['sector'].isin(top_sectors)]

        fig = px.area(rot_filtered, x='week', y='pct', color='sector',
                      title='Sector Share of Top 100 Over Time',
                      labels={'pct': '% of Top 100', 'week': 'Week'})
        fig.update_layout(height=400, template='plotly_dark', margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 5: STOCK X-RAY
# ═══════════════════════════════════════════════════════════
elif page == "🔬 Stock X-Ray":
    st.markdown("## 🔬 Stock X-Ray — Full Weekly Journey")

    all_tickers = sorted(panel['ticker'].unique())

    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.selectbox("Select Stock", all_tickers)
    with col2:
        search = st.text_input("Search ticker", "")
        if search:
            matches = [t for t in all_tickers if search.upper() in str(t).upper()]
            if matches:
                ticker = st.selectbox("Matches", matches, key="search_results")

    stock = panel[panel['ticker'] == ticker].sort_values('week')

    if len(stock) > 0:
        last = stock.iloc[-1]
        first = stock.iloc[0]

        # Header
        st.markdown(f"### {ticker} — {last.get('company_name','')}")

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            total_chg = ((last['price'] - first['price']) / first['price'] * 100) if first['price'] > 0 else 0
            st.metric("Price", f"₹{last['price']:.0f}", f"{total_chg:+.1f}% total")
        with c2:
            st.metric("Rank", f"#{last['rank']:.0f}", f"{last.get('rank_delta_1w',0):+.0f}")
        with c3:
            st.metric("Score", f"{last['master_score']:.1f}", f"{last.get('score_delta_1w',0):+.1f}")
        with c4:
            st.metric("State", last.get('market_state',''))
        with c5:
            st.metric("Sector", str(last.get('sector',''))[:15])
        with c6:
            wit = last.get('weeks_in_top100', 0)
            st.metric("Weeks in Top100", int(wit))

        # 6-panel chart
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=['💰 Price','📊 Rank (↓ = better)','🎯 Master Score',
                           '⚡ Breakout Score','📈 Momentum Score','🔊 Volume Score'],
            vertical_spacing=0.08
        )

        w = stock['week']
        chart_data = [
            (stock['price'], '#38bdf8', 1, 1),
            (stock['rank'], '#f87171', 1, 2),
            (stock['master_score'], '#4ade80', 2, 1),
            (stock['breakout_score'], '#fbbf24', 2, 2),
            (stock['momentum_score'], '#a855f7', 3, 1),
            (stock['volume_score'], '#06b6d4', 3, 2),
        ]

        for data, color, r, c in chart_data:
            fig.add_trace(go.Scatter(x=w, y=data, mode='lines+markers',
                         line=dict(color=color, width=2), showlegend=False), row=r, col=c)

        fig.update_yaxes(autorange="reversed", row=1, col=2)
        fig.update_layout(height=700, template='plotly_dark', margin=dict(t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Pattern timeline
        st.markdown("### 🏷️ Pattern & State Timeline")
        for _, row in stock.iterrows():
            pat = str(row.get('patterns', ''))
            state = row.get('market_state', '')
            rk = row['rank']
            sc = row['master_score']
            pr = row['price']

            # Market state color
            state_colors = {
                'STRONG_UPTREND': '🟢', 'UPTREND': '🟢', 'PULLBACK': '🟡',
                'ROTATION': '🟡', 'SIDEWAYS': '⚪', 'BOUNCE': '🔵',
                'DOWNTREND': '🔴', 'STRONG_DOWNTREND': '🔴'
            }
            state_icon = state_colors.get(state, '⚪')

            line = f"**{row['week'].strftime('%b %d')}** | {state_icon} `{state}` | Rank #{rk:.0f} | Score {sc:.1f} | ₹{pr:.0f}"
            if pat and pat != 'nan' and len(pat) > 3:
                line += f" | {pat}"
            st.markdown(line)

        # Score delta table
        with st.expander("📋 Full Weekly Data"):
            display_cols = ['week','rank','rank_delta_1w','master_score','score_delta_1w',
                           'price','breakout_score','momentum_score','volume_score',
                           'acceleration_score','rvol_score','trend_quality',
                           'ret_7d','ret_30d','ret_3m','market_state','patterns',
                           'weeks_in_top100']
            available = [c for c in display_cols if c in stock.columns]
            disp = stock[available].copy()
            disp['week'] = disp['week'].dt.strftime('%Y-%m-%d')
            st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)

        # Peer comparison
        st.markdown("### 👥 Industry Peers")
        industry = last.get('industry', '')
        if industry:
            peers = panel[(panel['week'] == latest_week) & (panel['industry'] == industry)].nsmallest(10, 'rank')
            if len(peers) > 0:
                peer_cols = ['ticker','company_name','rank','master_score','price',
                            'ret_7d','ret_30d','market_state']
                available = [c for c in peer_cols if c in peers.columns]
                st.dataframe(peers[available].reset_index(drop=True), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
#  PAGE 6: BACKTEST LAB
# ═══════════════════════════════════════════════════════════
elif page == "🧪 Backtest Lab":
    st.markdown(f"## 🧪 Backtest Lab — Walk-Forward Proof ({horizon})")
    st.caption("Each entry week: pick top-N stocks by rank (using ONLY current data). Measure ACTUAL forward return. No look-ahead bias.")

    col1, col2, col3 = st.columns(3)
    with col1:
        bt_top_n = st.slider("Top N stocks per period", 5, 50, top_n, key="bt_n")
    with col2:
        bt_min_score = st.slider("Min score threshold", 0, 80, 0, 10, key="bt_score")
    with col3:
        st.markdown(f"**Horizon:** {horizon}")

    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running walk-forward backtest..."):
            bt = walk_forward_backtest(panel, weeks, bt_top_n, 1, bt_min_score, fwd_col)

        if len(bt) > 0:
            # Summary
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: st.metric("Top N Avg Return", f"{bt['top_avg'].mean():.2f}%")
            with c2: st.metric("Bottom N Avg Return", f"{bt['bottom_avg'].mean():.2f}%")
            with c3: st.metric("Spread (Top-Bottom)", f"{bt['spread'].mean():.2f}%")
            with c4: st.metric("Top N Win Rate", f"{bt['top_wr'].mean():.0f}%")
            with c5: st.metric("Periods Tested", len(bt))

            spread_avg = bt['spread'].mean()
            top_wr = bt['top_wr'].mean()

            if spread_avg > 3 and top_wr > 55:
                st.markdown(f'<div class="verdict-box verdict-good">✅ SYSTEM HAS EDGE — Top {bt_top_n} stocks average {spread_avg:.1f}% more return than Bottom {bt_top_n}, with {top_wr:.0f}% win rate over {len(bt)} periods. This is actionable.</div>', unsafe_allow_html=True)
            elif spread_avg > 1:
                st.markdown(f'<div class="verdict-box verdict-mid">⚠️ MODERATE EDGE — {spread_avg:.1f}% spread, {top_wr:.0f}% win rate. Signal exists but may improve with more data or tighter filters.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="verdict-box verdict-bad">❌ WEAK EDGE — {spread_avg:.1f}% spread, {top_wr:.0f}% win rate. Rankings don\'t reliably predict {horizon} returns at this filter setting.</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Return comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bt['week'], y=bt['top_avg'], name=f'Top {bt_top_n}',
                                marker_color='#4ade80', opacity=0.8))
            fig.add_trace(go.Bar(x=bt['week'], y=bt['bottom_avg'], name=f'Bottom {bt_top_n}',
                                marker_color='#f87171', opacity=0.8))
            fig.add_trace(go.Scatter(x=bt['week'], y=bt['universe_avg'], name='Universe Avg',
                                    line=dict(color='#fbbf24', width=2, dash='dash')))
            fig.update_layout(
                title=f'Top {bt_top_n} vs Bottom {bt_top_n} — {horizon} Forward Return per Entry Week',
                yaxis_title='Return %', height=400, barmode='group',
                template='plotly_dark', margin=dict(t=40, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cumulative equity
            bt['cum_top'] = (1 + bt['top_avg'] / 100).cumprod() * 100
            bt['cum_bottom'] = (1 + bt['bottom_avg'] / 100).cumprod() * 100
            bt['cum_universe'] = (1 + bt['universe_avg'] / 100).cumprod() * 100

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=bt['week'], y=bt['cum_top'], name=f'Top {bt_top_n}',
                                     line=dict(color='#4ade80', width=3)))
            fig2.add_trace(go.Scatter(x=bt['week'], y=bt['cum_bottom'], name=f'Bottom {bt_top_n}',
                                     line=dict(color='#f87171', width=3)))
            fig2.add_trace(go.Scatter(x=bt['week'], y=bt['cum_universe'], name='Universe',
                                     line=dict(color='#94a3b8', width=2, dash='dot')))
            fig2.add_hline(y=100, line_dash="dash", line_color="#475569")
            fig2.update_layout(
                title='Cumulative Equity Curve (₹100 start)',
                yaxis_title='Portfolio Value ₹', height=400,
                template='plotly_dark', margin=dict(t=40, b=30)
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Win rate
            fig3 = px.bar(bt, x='week', y='top_wr', color='top_wr',
                          color_continuous_scale='RdYlGn', color_continuous_midpoint=50,
                          title=f'Win Rate per Period (Top {bt_top_n})')
            fig3.add_hline(y=50, line_dash="dash", line_color="#94a3b8")
            fig3.update_layout(height=300, template='plotly_dark', margin=dict(t=40, b=30))
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("📋 Period Details"):
                st.dataframe(bt.round(2), use_container_width=True, hide_index=True)

            # Strategy comparison
            st.markdown("---")
            st.markdown("### 📊 Strategy Comparison")

            strategies = {}
            for n in [10, 20, 50]:
                r = walk_forward_backtest(panel, weeks, n, 1, 0, fwd_col)
                if len(r) > 0:
                    strategies[f'Top {n}'] = {
                        'Avg Return': round(r['top_avg'].mean(), 2),
                        'Win Rate': round(r['top_wr'].mean(), 1),
                        'Spread': round(r['spread'].mean(), 2),
                        'Best Period': round(r['top_avg'].max(), 2),
                        'Worst Period': round(r['top_avg'].min(), 2),
                    }

            if strategies:
                st.dataframe(pd.DataFrame(strategies).T, use_container_width=True)

        else:
            st.warning("Not enough data for backtest. Try uploading more weekly files.")

    else:
        st.info("Click **Run Backtest** to start the walk-forward simulation.")

    # Data confidence
    st.markdown("---")
    st.markdown("### 📊 Data Confidence Level")
    if n_weeks >= 40:
        st.success(f"🟢 **HIGH CONFIDENCE** — {n_weeks} weeks of data. All statistical tests are reliable. ML models viable as complement.")
    elif n_weeks >= 20:
        st.info(f"🟡 **MODERATE CONFIDENCE** — {n_weeks} weeks. Statistical tests meaningful but p-values may be wide. Add {40 - n_weeks} more weeks for high confidence.")
    else:
        st.warning(f"🟠 **LOW CONFIDENCE** — {n_weeks} weeks. Results are directional but not definitive. Keep adding weekly data.")

    growth_note = f"Every new week you add makes the analysis more reliable. At 50+ weeks, ML models become viable as a complement to the statistical approach."
    st.caption(growth_note)


# ── Footer ──
st.markdown("---")
st.caption("WAVE Analyzer — System Intelligence Engine. Companion to WAVE Detection 3.0. Not financial advice.")

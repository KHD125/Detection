1. VELOCITY SQUEEZE 🎯
python# When momentum is ACCELERATING but range is COMPRESSING
velocity_squeeze = (
    (df['ret_7d'] / 7 > df['ret_30d'] / 30) &  # Daily velocity increasing
    (abs(df['from_high_pct']) + df['from_low_pct'] < 30) &  # In middle 30% of range
    (df['high_52w'] - df['low_52w']) / df['low_52w'] < 0.5  # Tight 52W range
)
# WHY: Coiled spring about to EXPLODE - best risk/reward!
2. VOLUME DIVERGENCE TRAP 🔍
python# Price going up but SMART volume going down
volume_divergence = (
    (df['ret_30d'] > 20) &  # Price rising
    (df['vol_ratio_30d_180d'] < 0.7) &  # But recent volume DECLINING
    (df['vol_ratio_90d_180d'] < 0.9) &  # Long-term volume also declining
    (df['from_high_pct'] > -5)  # Near highs
)
# WHY: Retail buying tops, institutions SELLING - GET OUT!
3. GOLDEN CROSSOVER MOMENTUM ⚡
python# Not just SMA crossover, but with ACCELERATION
golden_cross_momentum = (
    (df['sma_20d'] > df['sma_50d']) &  # 20 crossed 50
    (df['sma_50d'] > df['sma_200d']) &  # 50 crossed 200
    ((df['sma_20d'] - df['sma_50d']) / df['sma_50d'] > 0.02) &  # Diverging fast
    (df['rvol'] > 1.5) &  # Volume confirmation
    (df['ret_7d'] > df['ret_30d'] / 4)  # Accelerating
)
# WHY: STRONGEST trend confirmation - institutions piling in!
4. SMART MONEY ACCUMULATION INDEX 💰
python# Combining multiple "smart money" signals
smart_money_score = (
    (df['vol_ratio_90d_180d'] > 1.1) * 25 +  # Long-term volume up
    ((df['ret_30d'] / 30) < (df['ret_7d'] / 7)) * 25 +  # Recent acceleration
    (df['money_flow_mm'] > df['money_flow_mm'].quantile(0.8)) * 25 +  # High money flow
    ((df['high_52w'] - df['price']) / df['price'] < 0.1) * 25  # Near highs
)
accumulation_pattern = (smart_money_score >= 75)
# WHY: Multiple confirmation = HIGHEST probability!
5. MOMENTUM EXHAUSTION REVERSAL 📉
python# Too far, too fast - mean reversion imminent
exhaustion = (
    (df['ret_7d'] > 25) &  # Huge 7-day move
    (df['ret_1d'] < 0) &  # But negative today
    (df['rvol'] < df['rvol'].shift(1)) &  # Volume declining
    (df['from_low_pct'] > 80) &  # Far from support
    ((df['price'] - df['sma_20d']) / df['sma_20d'] > 0.15)  # 15% above SMA
)
# WHY: Parabolic moves ALWAYS revert - SHORT opportunity!
6. EARNINGS MOMENTUM SURPRISE 📊
python# EPS accelerating MORE than price
earnings_surprise = (
    (df['eps_change_pct'] > 50) &  # Strong EPS growth
    (df['eps_change_pct'] > df['ret_30d']) &  # EPS growing FASTER than price
    (df['pe'] < df['pe'].quantile(0.5)) &  # Still cheap
    (df['vol_ratio_30d_90d'] > 1)  # Volume picking up
)
# WHY: Market hasn't priced in earnings growth yet!
7. VOLATILITY CONTRACTION BREAKOUT 🎪
python# Bollinger Band squeeze equivalent
vol_contraction = pd.DataFrame()
vol_contraction['daily_range'] = (df['high_52w'] - df['low_52w']) / df['price']
vol_contraction['recent_volatility'] = df[['ret_1d', 'ret_3d', 'ret_7d']].std(axis=1)

squeeze_pattern = (
    (vol_contraction['recent_volatility'] < vol_contraction['recent_volatility'].quantile(0.2)) &
    (df['volume_30d'] < df['volume_90d'] * 0.7) &  # Volume drying up
    (abs(df['from_high_pct'] + df['from_low_pct']) < 20)  # Middle of range
)
# WHY: Low volatility precedes HIGH volatility - BIG move coming!
8. RELATIVE ROTATION LEADER 🏆
python# Leading its sector AND category
df['sector_rank'] = df.groupby('sector')['master_score'].rank(pct=True)
df['category_rank'] = df.groupby('category')['master_score'].rank(pct=True)

rotation_leader = (
    (df['sector_rank'] > 0.9) &  # Top 10% in sector
    (df['category_rank'] > 0.9) &  # Top 10% in category
    (df['ret_30d'] > df.groupby('sector')['ret_30d'].transform('mean') + 10) &  # Beating sector
    (df['volume_30d'] > df['volume_90d'])  # Rising volume
)
# WHY: Money rotating INTO this stock specifically!
9. PYRAMID ACCUMULATION 🔺
python# Institutions building position gradually
pyramid = (
    (df['vol_ratio_7d_90d'] > 1.1) &
    (df['vol_ratio_30d_90d'] > 1.05) &
    (df['vol_ratio_90d_180d'] > 1.02) &
    (df['ret_30d'].between(5, 15)) &  # Steady rise, not parabolic
    (df['from_low_pct'] < 50)  # Still room to run
)
# WHY: Big players accumulate SLOWLY to not spike price!
10. MOMENTUM VACUUM 🌪️
python# When selling exhausts and buyers step in
vacuum = (
    (df['ret_30d'] < -20) &  # Big decline
    (df['ret_7d'] > 0) &  # But turning positive
    (df['ret_1d'] > 2) &  # Strong today
    (df['rvol'] > 3) &  # Huge volume
    (df['from_low_pct'] < 10)  # Near 52W low
)
# WHY: Sellers exhausted, ANY buying creates violent bounce!
🔬 SMART CALCULATIONS NOT IN YOUR SCRIPT:
1. VWAP DEVIATION
pythondf['vwap'] = (df['price'] * df['volume_1d']).cumsum() / df['volume_1d'].cumsum()
df['vwap_deviation'] = (df['price'] - df['vwap']) / df['vwap'] * 100
# Institutional traders use VWAP!
2. ACCUMULATION/DISTRIBUTION LINE
pythondf['money_flow_multiplier'] = ((df['price'] - df['low_52w']) - (df['high_52w'] - df['price'])) / (df['high_52w'] - df['low_52w'])
df['money_flow_volume'] = df['money_flow_multiplier'] * df['volume_1d']
df['ad_line'] = df['money_flow_volume'].cumsum()
3. RELATIVE STRENGTH INDEX (MULTI-PERIOD)
pythondef calculate_rsi(df, periods=[7, 14, 30]):
    for period in periods:
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    return df
4. MOMENTUM QUALITY SCORE
pythondf['momentum_quality'] = (
    (df['ret_30d'] > 0) * 20 +
    (df['ret_30d'] > df['ret_3m'] / 3) * 20 +  # Accelerating
    (df['volume_30d'] > df['volume_90d']) * 20 +  # Volume support
    (df['sma_20d'] > df['sma_50d']) * 20 +  # Trend alignment
    (df['from_low_pct'] < 70) * 20  # Not overextended
)

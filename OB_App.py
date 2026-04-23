import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog
import warnings
import base64
import io
import datetime
from pathlib import Path
try:
    from weasyprint import HTML as WeasyprintHTML
    _WEASYPRINT_OK = True
except ImportError:
    _WEASYPRINT_OK = False
warnings.filterwarnings('ignore')

# ── Logo ───────────────────────────────────────────────────────────────────────
_logo_path = Path(__file__).parent / "OIA_Logo.svg"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode() if _logo_path.exists() else None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Orange Investment Advisors | Transition Optimizer",
                   layout="wide", initial_sidebar_state="expanded")

# Force sidebar always visible and hide the collapse/expand toggle
st.markdown("""<style>
  section[data-testid="stSidebar"] {
    transform: none !important;
    min-width: 244px !important;
    width: 244px !important;
  }
  [data-testid="collapsedControl"]      { display: none !important; }
  button[kind="header"]                 { display: none !important; }
</style>""", unsafe_allow_html=True)

# ── Brand colors ───────────────────────────────────────────────────────────────
#must change all of these once we know their colors better
NAVY    = '#12284c'
COPPER  = '#976f33'
OLIVE   = '#5D6B49'
WHITE   = '#FFFFFF'
GOLD    = COPPER
GREEN   = '#5D6B49'
RED     = '#A93226'
GRAY    = '#6B6B6B'
MUTED   = '#C7C6CA'
LIGHT   = '#F4F4F5'
CARD_BG = '#FFFFFF'
BLUE    = NAVY

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(f"""<style>
  [data-testid="stAppViewContainer"] {{ background-color: {LIGHT}; }}
  [data-testid="stMain"]             {{ background-color: {LIGHT}; }}
  [data-testid="stSidebar"]          {{ background-color: {WHITE}; }}
  header[data-testid="stHeader"]     {{ display: none !important; }}
  .block-container                   {{ padding-top: 0rem !important; }}
  .bison-header {{
    background-color:{WHITE}; border-bottom:3px solid {GOLD};
    padding:14px 24px; display:flex; justify-content:space-between;
    align-items:center; margin-bottom:24px; border-radius:0 0 4px 4px;
  }}
  .bison-page-title {{ font-size:32px; font-weight:800; color:{NAVY}; letter-spacing:0.5px; line-height:1.2; }}
  .bison-page-sub   {{ font-size:15px; color:{GRAY}; margin-top:6px; font-weight:400; }}
  .bison-client-info {{ text-align:right; font-size:14px; color:{GRAY}; }}
  .bison-client-name {{ font-size:18px; font-weight:700; color:{NAVY}; }}
  h2, h3 {{ font-size:22px !important; font-weight:700 !important; }}
  .sc-header {{ padding:12px 8px; border-radius:5px 5px 0 0; text-align:center;
                font-weight:700; font-size:13px; color:{WHITE}; letter-spacing:1.5px; }}
  .sc-body {{ background:{CARD_BG}; border:1px solid #C8D0D8; border-top:none;
              border-radius:0 0 5px 5px; padding:16px 12px 8px 12px; }}
  .sc-label {{ font-size:11px; color:{GRAY}; font-weight:700; letter-spacing:0.5px;
               text-align:center; text-transform:uppercase; margin-bottom:2px; }}
  .sc-value {{ font-size:26px; font-weight:700; text-align:center; line-height:1.1; margin-bottom:4px; }}
  .sc-divider {{ border:none; border-top:1px solid #E2E8EE; margin:10px 4px; }}
  [data-testid="stMetric"] {{ background:{WHITE}; border:1px solid #C8D0D8;
                               border-left:4px solid {COPPER} !important;
                               border-radius:5px; padding:12px 16px; }}
  [data-testid="stTab"] {{ font-weight:600; }}
  .bison-footer {{ text-align:center; color:{GRAY}; font-size:10px;
                   font-style:italic; margin-top:32px; padding-bottom:16px; }}
  button[kind="primary"] {{
    background-color:{COPPER} !important; color:{WHITE} !important;
    font-weight:700 !important; border:none !important;
  }}
  button[kind="primary"]:hover {{ opacity:0.88 !important; }}
</style>""", unsafe_allow_html=True)


# ── Load preset model portfolios ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_preset_models():
    path = Path(__file__).parent / "OB_models.xlsx"
    if not path.exists():
        return {}
    raw = pd.read_excel(path, header=None)
    # Row 0 = model names in cols 1, 3, 5; Row 1 = headers; Rows 2+ = data
    models = {}
    col_idx = 1
    while col_idx < raw.shape[1]:
        name = str(raw.iloc[0, col_idx]).strip()
        if name and name != 'nan':
            slice_df = raw.iloc[2:, col_idx:col_idx+2].copy()
            slice_df.columns = ['Ticker', 'Weight']
            slice_df['Ticker'] = slice_df['Ticker'].astype(str).str.strip()
            slice_df['Weight'] = pd.to_numeric(slice_df['Weight'], errors='coerce')
            df = slice_df[
                slice_df['Ticker'].notna() &
                (slice_df['Ticker'] != '') &
                (slice_df['Ticker'] != 'nan') &
                slice_df['Weight'].notna()
            ].copy()
            df = df.reset_index(drop=True)
            if df['Weight'].sum() > 0:
                df['Weight'] = df['Weight'] / df['Weight'].sum()
            models[name] = df
        col_idx += 2
    return models


PRESET_MODELS = load_preset_models()

# ── Default dummy holdings ─────────────────────────────────────────────────────
_EMPTY_HOLDINGS = pd.DataFrame(columns=['Ticker','Shares','Price','Cost_Basis','Holding'])

DEFAULT_MODEL = pd.DataFrame({
    'Ticker': ['AAPL','MSFT','NVDA','GOOGL','AMZN'],
    'Weight': [0.20, 0.20, 0.20, 0.20, 0.20],
})


# ── Data helpers ───────────────────────────────────────────────────────────────
def _to_float(val, default=0.0):
    try:
        f = float(val)
        return default if (f != f) else f
    except (TypeError, ValueError):
        return default


def _clean_col(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(r'[\$,]', '', regex=True)
              .str.replace(r'\(([0-9.]+)\)', r'-\1', regex=True)
              .str.strip(),
        errors='coerce',
    )


def parse_holdings(raw_df, lt_rate, st_rate):
    """Parse any supported input into a clean holdings DataFrame."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rows = []

    if 'Units' in df.columns:
        df = df.rename(columns={
            'Units': 'Shares', 'Cost Per Share': 'Cost_Basis',
            'Market Value': 'MktVal',
            'Short Term Gain Loss': 'ST_GL', 'Long Term Gain Loss': 'LT_GL',
        })
        for col in ['Shares', 'Price', 'Cost_Basis', 'MktVal', 'ST_GL', 'LT_GL']:
            if col in df.columns:
                df[col] = _clean_col(df[col])
        df = df[df['Ticker'].notna() & (df['Ticker'].astype(str).str.strip() != '')]
        df = df[df['Shares'].fillna(0) != 0]
        df = df[df['Price'].notna()].reset_index(drop=True)

        for _, row in df.iterrows():
            ticker = str(row['Ticker']).strip()
            shares = _to_float(row['Shares'])
            price  = _to_float(row['Price'])
            mv     = _to_float(row.get('MktVal'), shares * price)
            st_gl  = _to_float(row.get('ST_GL', 0.0))
            lt_gl  = _to_float(row.get('LT_GL', 0.0))

            if st_gl != 0.0 and lt_gl != 0.0:
                total_abs = abs(st_gl) + abs(lt_gl)
                lt_mv     = mv * abs(lt_gl) / total_abs
                st_mv     = mv - lt_mv
                lt_sh     = shares * abs(lt_gl) / total_abs
                st_sh     = shares - lt_sh
                rows.append(dict(Ticker=ticker, Shares=lt_sh, Price=price,
                                 MarketValue=lt_mv, UnrealizedGL=lt_gl, Holding='LT'))
                rows.append(dict(Ticker=ticker, Shares=st_sh, Price=price,
                                 MarketValue=st_mv, UnrealizedGL=st_gl, Holding='ST'))
            elif st_gl != 0.0:
                rows.append(dict(Ticker=ticker, Shares=shares, Price=price,
                                 MarketValue=mv, UnrealizedGL=st_gl, Holding='ST'))
            else:
                rows.append(dict(Ticker=ticker, Shares=shares, Price=price,
                                 MarketValue=mv, UnrealizedGL=lt_gl, Holding='LT'))
    else:
        df.columns = [c.replace(' ', '_').title() for c in df.columns]
        df = df.rename(columns={'Costbasis': 'Cost_Basis'})
        if 'Holding' not in df.columns:
            df['Holding'] = 'LT'
        df = df[df['Ticker'].notna() & (df['Shares'].fillna(0) != 0)].reset_index(drop=True)
        for _, row in df.iterrows():
            shares = _to_float(row['Shares'])
            price  = _to_float(row['Price'])
            cb     = _to_float(row['Cost_Basis'])
            mv     = shares * price
            rows.append(dict(Ticker=str(row['Ticker']).strip(), Shares=shares, Price=price,
                             MarketValue=mv, UnrealizedGL=mv - shares * cb,
                             Holding=str(row.get('Holding', 'LT'))))

    h = pd.DataFrame(rows)
    if h.empty:
        return h
    h['TaxRate'] = h['Holding'].map({'LT': lt_rate, 'ST': st_rate}).fillna(lt_rate)
    h['MaxTax']  = h['UnrealizedGL'].clip(lower=0) * h['TaxRate']
    return h.reset_index(drop=True)


def parse_model(raw_df):
    """Parse a target model upload into a Ticker/Weight DataFrame."""
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Flexible column matching
    ticker_col = next((c for c in df.columns if 'tick' in c.lower() or 'symbol' in c.lower()
                       or 'hold' in c.lower()), df.columns[0])
    weight_col = next((c for c in df.columns if 'weight' in c.lower() or 'alloc' in c.lower()
                       or 'pct' in c.lower() or '%' in c.lower()), df.columns[1])
    df = df[[ticker_col, weight_col]].rename(columns={ticker_col: 'Ticker', weight_col: 'Weight'})
    df['Ticker'] = df['Ticker'].astype(str).str.strip()
    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df = df[df['Ticker'].notna() & (df['Ticker'] != 'nan') & df['Weight'].notna()]
    if df['Weight'].sum() > 0:
        df['Weight'] = df['Weight'] / df['Weight'].sum()
    return df.reset_index(drop=True)


# ── Optimizer ──────────────────────────────────────────────────────────────────
def _max_sell_bounds(holdings_df, target_model_df, total_value):
    """
    For each holding row, compute the maximum fraction that should be sold.
    - Positions NOT in the target model: sell up to 100%.
    - Positions IN the target model: only sell the portion above the target weight.
      If already at or below target weight, upper bound is 0 (don't sell).
    """
    if target_model_df is None or target_model_df.empty or total_value <= 0:
        return [(0.0, 1.0)] * len(holdings_df)

    # Target value per ticker
    target_values = {
        str(r['Ticker']).strip(): float(r['Weight']) * total_value
        for _, r in target_model_df.iterrows()
    }

    # Aggregate current market value per ticker (handles split LT/ST lots)
    ticker_mv = {}
    for _, row in holdings_df.iterrows():
        t = str(row['Ticker']).strip()
        ticker_mv[t] = ticker_mv.get(t, 0.0) + float(row['MarketValue'])

    bounds = []
    for _, row in holdings_df.iterrows():
        t        = str(row['Ticker']).strip()
        curr_mv  = ticker_mv[t]
        tgt_mv   = target_values.get(t, 0.0)
        # Sellable = amount above target; expressed as fraction of this row's market value
        sellable_mv   = max(0.0, curr_mv - tgt_mv)
        max_frac      = min(1.0, sellable_mv / curr_mv) if curr_mv > 0 else 0.0
        bounds.append((0.0, max_frac))
    return bounds


def solve_transition(holdings_df, tax_budget, target_model_df=None, total_value=None):
    """LP: maximise proceeds subject to net tax <= tax_budget."""
    values     = np.array([float(v) for v in holdings_df['MarketValue']])
    gains      = np.array([float(v) for v in holdings_df['UnrealizedGL']])
    rates      = np.array([float(v) for v in holdings_df['TaxRate']])
    tax_coeffs = gains * rates
    n          = len(values)
    if n == 0:
        return None

    tv     = total_value if total_value is not None else float(values.sum())
    bounds = _max_sell_bounds(holdings_df, target_model_df, tv)

    result = linprog(
        c=-values,
        A_ub=tax_coeffs.reshape(1, -1),
        b_ub=np.array([float(tax_budget)]),
        bounds=bounds,
        method='highs',
    )
    if not result.success:
        return None
    x = result.x
    proceeds = float(values @ x)
    return {
        'x':              x,
        'proceeds':       proceeds,
        'realized_gl':    float(gains @ x),
        'net_tax':        max(0.0, float(tax_coeffs @ x)),
        'transition_pct': proceeds / values.sum() * 100 if values.sum() > 0 else 0.0,
        'sell_fracs':     x,
    }


def compute_frontier(holdings_df, max_tax, target_model_df=None, total_value=None):
    """Sweep 300 tax budgets and return the efficient frontier DataFrame."""
    values     = np.array([float(v) for v in holdings_df['MarketValue']])
    gains      = np.array([float(v) for v in holdings_df['UnrealizedGL']])
    rates      = np.array([float(v) for v in holdings_df['TaxRate']])
    tax_coeffs = gains * rates
    n          = len(values)
    total_val  = values.sum()
    sweep_max  = max_tax * 1.02 if max_tax > 0 else 1.0
    tv         = total_value if total_value is not None else float(total_val)
    bounds     = _max_sell_bounds(holdings_df, target_model_df, tv)
    points = []
    for T in np.linspace(0, sweep_max, 300):
        res = linprog(c=-values, A_ub=tax_coeffs.reshape(1, -1),
                      b_ub=np.array([float(T)]), bounds=bounds, method='highs')
        if res.success:
            x = res.x
            points.append({
                'tax_incurred':   max(0.0, float(tax_coeffs @ x)),
                'transition_pct': float(values @ x) / total_val * 100 if total_val > 0 else 0.0,
                'realized_gl':    float(gains @ x),
                'proceeds':       float(values @ x),
            })
    if not points:
        return pd.DataFrame(columns=['tax_incurred', 'transition_pct', 'realized_gl', 'proceeds'])
    return pd.DataFrame(points).drop_duplicates('transition_pct').reset_index(drop=True)


def simulate_transition_timeline(holdings_df, annual_budget, target_model_df=None,
                                 total_value=None, max_years=25):
    """Simulate year-by-year transitions using a fixed annual tax budget.
    Returns a list of dicts: year, cumulative_pct, remaining_gains, tax_incurred, proceeds."""
    if holdings_df.empty or annual_budget <= 0:
        return []
    current = holdings_df.reset_index(drop=True).copy()
    tv = total_value if total_value is not None else float(current['MarketValue'].sum())
    if tv <= 0:
        return []
    # cum_remaining[i] = fraction of original holding i still held
    cum_remaining = np.ones(len(current))
    current['_orig_idx'] = np.arange(len(current))

    initial_gains = float(current['UnrealizedGL'].clip(lower=0).sum())
    zero_fracs    = np.zeros(len(current))
    init_almt     = (alignment_score(current.drop(columns=['_orig_idx']), zero_fracs, target_model_df, tv)
                     if target_model_df is not None and not target_model_df.empty else 0.0)

    results = [{'year': 0, 'alignment_pct': init_almt,
                'remaining_gains': initial_gains, 'tax_incurred': 0.0, 'proceeds': 0.0}]

    original = current.drop(columns=['_orig_idx']).copy()

    for year in range(1, max_years + 1):
        if current.empty or float(current['MarketValue'].sum()) < 100:
            break
        solve_df = current.drop(columns=['_orig_idx'])
        r = solve_transition(solve_df, annual_budget, target_model_df, tv)
        if r is None or float(np.sum(r['sell_fracs'])) < 1e-6:
            break

        sf       = np.array(r['sell_fracs'])
        orig_idx = current['_orig_idx'].values

        # Update cumulative remaining fractions mapped back to original rows
        for j, oi in enumerate(orig_idx):
            cum_remaining[oi] *= (1.0 - float(sf[j]))

        # Alignment: same math as the donut cards — alignment_score on original holdings
        # with cumulative sell fracs, so Year 1 always matches the Custom scenario card
        cum_sell = 1.0 - cum_remaining
        almt = (alignment_score(original, cum_sell, target_model_df, tv)
                if target_model_df is not None and not target_model_df.empty
                else float(np.dot(cum_sell, original['MarketValue'].values)
                           / float(original['MarketValue'].sum()) * 100))

        # Scale current rows down and drop fully-sold ones
        remaining_arr = 1.0 - sf
        for col in ['Shares', 'MarketValue', 'UnrealizedGL', 'MaxTax']:
            if col in current.columns:
                current[col] = current[col].values * remaining_arr
        current = current[remaining_arr > 0.001].reset_index(drop=True)

        remaining_gains = (float(current['UnrealizedGL'].clip(lower=0).sum())
                           if not current.empty else 0.0)

        results.append({
            'year':            year,
            'alignment_pct':   almt,
            'remaining_gains': remaining_gains,
            'tax_incurred':    r['net_tax'],
            'proceeds':        r['proceeds'],
        })

        if almt >= 99.9 or current.empty:
            break

    return results


def compute_buys(holdings_df, sell_fracs, target_model_df, total_portfolio_value):
    """
    Given sell fractions and a target model, compute buy trades.
    Proceeds from sells are allocated to target model positions proportionally,
    prioritising positions most underweight vs target.
    Returns a DataFrame of buy trades.
    """
    if target_model_df is None or target_model_df.empty:
        return pd.DataFrame()

    # Remaining value per current holding after sells
    remaining = {}
    for i, row in holdings_df.iterrows():
        ticker = row['Ticker']
        kept_value = float(row['MarketValue']) * (1.0 - float(sell_fracs[i]))
        remaining[ticker] = remaining.get(ticker, 0.0) + kept_value

    proceeds = sum(
        float(holdings_df.iloc[i]['MarketValue']) * float(sell_fracs[i])
        for i in range(len(holdings_df))
    )

    # New total portfolio value after transition (same total, cash redeployed)
    new_total = total_portfolio_value

    # For each target position: how much do we still need to buy?
    buy_rows = []
    for _, trow in target_model_df.iterrows():
        ticker        = str(trow['Ticker']).strip()
        target_weight = float(trow['Weight'])
        target_value  = target_weight * new_total
        current_value = remaining.get(ticker, 0.0)
        shortfall     = target_value - current_value
        if shortfall > 1.0:   # only buy if meaningfully underweight
            buy_rows.append({'Ticker': ticker, 'Target Weight': target_weight,
                             'Needed': shortfall})

    if not buy_rows:
        return pd.DataFrame()

    buy_df = pd.DataFrame(buy_rows)
    total_needed = buy_df['Needed'].sum()

    # Scale buys to available proceeds
    scale = min(1.0, proceeds / total_needed) if total_needed > 0 else 0.0
    buy_df['Buy Value']       = (buy_df['Needed'] * scale).round(0)
    buy_df['Target Weight %'] = (buy_df['Target Weight'] * 100).round(2)
    buy_df = buy_df[buy_df['Buy Value'] > 0].sort_values('Buy Value', ascending=False)
    # Actual weight = buy value / total portfolio value (reflects what is actually purchased)
    buy_df['Actual Weight %'] = (buy_df['Buy Value'] / total_portfolio_value * 100).round(2) if total_portfolio_value > 0 else 0.0
    return buy_df[['Ticker', 'Target Weight %', 'Actual Weight %', 'Buy Value']].reset_index(drop=True)


def alignment_score(holdings_df, sell_fracs, target_model_df, total_portfolio_value):
    """
    Returns % of target portfolio that is already covered after the transition.
    """
    if target_model_df is None or target_model_df.empty or total_portfolio_value == 0:
        return 0.0

    remaining = {}
    for i, row in holdings_df.iterrows():
        ticker = row['Ticker']
        kept_value = float(row['MarketValue']) * (1.0 - float(sell_fracs[i]))
        remaining[ticker] = remaining.get(ticker, 0.0) + kept_value

    proceeds = sum(
        float(holdings_df.iloc[i]['MarketValue']) * float(sell_fracs[i])
        for i in range(len(holdings_df))
    )

    covered = 0.0
    for _, trow in target_model_df.iterrows():
        ticker       = str(trow['Ticker']).strip()
        target_val   = float(trow['Weight']) * total_portfolio_value
        current_val  = remaining.get(ticker, 0.0)
        covered     += min(current_val, target_val)

    # Also count proceeds allocated to buy underweights
    buy_df = compute_buys(holdings_df, sell_fracs, target_model_df, total_portfolio_value)
    if not buy_df.empty:
        covered += buy_df['Buy Value'].sum()

    return min(covered / total_portfolio_value * 100, 100.0)


# ── Charts ─────────────────────────────────────────────────────────────────────
def make_donut(pct, color):
    fig = go.Figure(go.Pie(
        values=[max(pct, 0.5), max(100 - pct, 0)], hole=0.64,
        marker=dict(colors=[color, '#DDE3EA'], line=dict(color=WHITE, width=3)),
        showlegend=False, textinfo='none', direction='clockwise', rotation=90,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=150,
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[dict(text=f'<b>{pct:.0f}%</b>', x=0.5, y=0.5,
                          showarrow=False, font=dict(size=22, color='#1C2833'))],
    )
    return fig


def make_frontier(frontier_df, scenarios):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df['transition_pct'], y=frontier_df['tax_incurred'],
        fill='tozeroy', fillcolor='rgba(38,55,89,0.07)',
        line=dict(color=NAVY, width=2.5), name='Efficient Frontier',
        hovertemplate='Transition: %{x:.1f}%<br>Tax: $%{y:,.0f}<extra></extra>',
    ))
    for sc in scenarios:
        fig.add_trace(go.Scatter(
            x=[sc['transition_pct']], y=[sc['net_tax']], mode='markers',
            marker=dict(color=sc['color'], size=14, line=dict(color=WHITE, width=2.5)),
            name=sc['name'],
            hovertemplate=(f"<b>{sc['name']}</b><br>Transition: {sc['transition_pct']:.1f}%<br>"
                           f"Tax: ${sc['net_tax']:,.0f}<br>G/L: ${sc['realized_gl']:,.0f}"
                           "<extra></extra>"),
        ))
    fig.update_layout(
        xaxis=dict(title='Portfolio Transitioned (%)', ticksuffix='%',
                   gridcolor='#E8ECF0', linecolor='#D5DADE', zeroline=False),
        yaxis=dict(title='Tax Liability ($)', tickprefix='$', tickformat=',.0f',
                   gridcolor='#E8ECF0', linecolor='#D5DADE', zeroline=False),
        height=380, plot_bgcolor=WHITE, paper_bgcolor=LIGHT,
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1, font=dict(size=11)),
        margin=dict(l=70, r=20, t=10, b=50), hovermode='closest',
    )
    return fig


def make_timeline_chart(timeline_data, annual_budget):
    years      = [d['year']            for d in timeline_data]
    cum_pcts   = [d['alignment_pct']   for d in timeline_data]
    rem_gains  = [d['remaining_gains'] for d in timeline_data]

    fig = go.Figure()

    # Left axis: % transitioned (area)
    fig.add_trace(go.Scatter(
        x=years, y=cum_pcts,
        name='% Transitioned',
        fill='tozeroy',
        fillcolor='rgba(38,55,89,0.15)',
        line=dict(color=NAVY, width=2.5),
        mode='lines+markers',
        marker=dict(size=7, color=NAVY),
        yaxis='y1',
        hovertemplate='Year %{x}<br>Transitioned: %{y:.1f}%<extra></extra>',
    ))

    # Right axis: remaining cap gains (dollars)
    fig.add_trace(go.Scatter(
        x=years, y=rem_gains,
        name='Remaining Cap Gains',
        line=dict(color=COPPER, width=2.5),
        mode='lines+markers',
        marker=dict(size=7, color=COPPER),
        yaxis='y2',
        hovertemplate='Year %{x}<br>Remaining Gains: $%{y:,.0f}<extra></extra>',
    ))

    fig.update_layout(
        xaxis=dict(title='Year', tickmode='linear', tick0=0, dtick=1,
                   gridcolor='#E8ECF0', linecolor='#D5DADE', zeroline=False,
                   range=[years[0] - 0.3, years[-1] + 0.3]),
        yaxis=dict(title=dict(text='% Transitioned', font=dict(color=NAVY)),
                   ticksuffix='%', range=[-3, 108], gridcolor='#E8ECF0',
                   linecolor='#D5DADE', zeroline=False, tickfont=dict(color=NAVY)),
        yaxis2=dict(title=dict(text='Remaining Cap Gains ($)', font=dict(color=COPPER)),
                    tickprefix='$', tickformat=',.0f', overlaying='y', side='right',
                    zeroline=False, tickfont=dict(color=COPPER), showgrid=False,
                    rangemode='nonnegative'),
        height=340,
        plot_bgcolor=WHITE,
        paper_bgcolor=LIGHT,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                    font=dict(size=11)),
        margin=dict(l=60, r=80, t=10, b=0),
        hovermode='x unified',
    )
    return fig


# ── Export helpers ─────────────────────────────────────────────────────────────
def generate_summary_html(client_name, account_num, total_value,
                          lt_unreal, st_unreal, net_unreal, scenarios, logo_b64=None):
    """Return a self-contained HTML bytes object that looks like the dashboard.
    User opens in browser → Ctrl+P → Save as PDF (landscape) for a pixel-perfect export."""

    _NAVY    = '#263759'
    _COPPER  = '#C17A49'
    _GREEN   = '#5D6B49'
    _RED     = '#A93226'
    _GRAY    = '#6B6B6B'
    _LIGHT   = '#F4F4F5'
    _WHITE   = '#FFFFFF'
    _BORDER  = '#C8D0D8'
    _DIVIDER = '#E2E8EE'

    def _gl_color(v): return _GREEN if v >= 0 else _RED
    def _fmt_gl(v):   return ('+' if v >= 0 else '-') + f'${abs(v):,.0f}'

    def _svg_donut(pct, color, size=150):
        import math
        R  = 46
        cx = cy = size / 2
        C  = 2 * math.pi * R
        filled = max(pct, 0.5) / 100 * C
        empty  = C - filled
        offset = C * 0.25   # start arc from 12 o'clock
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
            f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none"'
            f' stroke="#DDE3EA" stroke-width="17"/>'
            f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none"'
            f' stroke="{color}" stroke-width="17"'
            f' stroke-dasharray="{filled:.2f} {empty:.2f}"'
            f' stroke-dashoffset="{offset:.2f}"'
            f' transform="rotate(-90 {cx} {cy})"/>'
            f'<text x="{cx}" y="{cy + 8}" text-anchor="middle"'
            f' font-family="sans-serif" font-size="24" font-weight="bold"'
            f' fill="#1C2833">{pct:.0f}%</text>'
            f'</svg>'
        )

    # ── Logo tag ────────────────────────────────────────────────────────────
    logo_tag = (f'<img src="data:image/svg+xml;base64,{logo_b64}"'
                f' style="height:64px;object-fit:contain;"/>'
                if logo_b64 else
                f'<span style="font-size:18px;font-weight:900;color:{_NAVY};'
                f'letter-spacing:2px;">ORANGE INVESTMENT ADVISORS</span>')

    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics = [
        ('Portfolio Value',      f'${total_value:,.0f}',   _NAVY),
        ('LT Unrealized G/L',    _fmt_gl(lt_unreal),       _gl_color(lt_unreal)),
        ('ST Unrealized G/L',    _fmt_gl(st_unreal),       _gl_color(st_unreal)),
        ('Total Unrealized G/L', _fmt_gl(net_unreal),      _gl_color(net_unreal)),
    ]
    metric_html = ''.join(f'''
      <div style="background:{_WHITE};border:1px solid {_BORDER};border-radius:5px;
                  padding:14px 16px;flex:1;min-width:0;">
        <div style="font-size:10px;color:{_GRAY};font-weight:700;
                    text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">{lbl}</div>
        <div style="font-size:22px;font-weight:700;color:{col};">{val}</div>
      </div>''' for lbl, val, col in metrics)

    # ── Scenario cards ───────────────────────────────────────────────────────
    def _stat_row(label, value, color, divider=True):
        hr = f'<hr style="border:none;border-top:1px solid {_DIVIDER};margin:8px 0;"/>' if divider else ''
        return f'''
        <div style="text-align:center;padding:4px 0 2px 0;">
          <div style="font-size:10px;color:{_GRAY};font-weight:700;
                      text-transform:uppercase;letter-spacing:0.5px;margin-bottom:3px;">{label}</div>
          <div style="font-size:22px;font-weight:700;color:{color};">{value}</div>
        </div>{hr}'''

    cards_html = ''
    for sc in scenarios:
        tx_color = _RED   if sc['net_tax']    > 0  else _GREEN
        gl_color = _GREEN if sc['realized_gl'] <= 0 else _RED
        gl_sign  = '+' if sc['realized_gl'] > 0 else ''
        cards_html += f'''
      <div style="flex:1;min-width:0;border-radius:5px;overflow:hidden;
                  border:1px solid {_BORDER};display:flex;flex-direction:column;">
        <div style="background:{sc['color']};color:{_WHITE};text-align:center;
                    padding:11px;font-weight:700;font-size:13px;letter-spacing:1.5px;">
          {sc['name'].upper()}
        </div>
        <div style="background:{_WHITE};padding:14px 16px;">
          {_stat_row('Estimated Tax Liability', f"${sc['net_tax']:,.0f}", tx_color)}
          {_stat_row('Realized Gain / Loss', f"{gl_sign}${sc['realized_gl']:,.0f}", gl_color)}
          {_stat_row('Transition Amount', f"${sc['proceeds']:,.0f}", _NAVY, divider=False)}
        </div>
        <div style="background:{_WHITE};border-top:1px solid {_DIVIDER};
                    text-align:center;padding:10px 0 18px 0;flex:1;">
          <div style="font-size:9px;color:{_GRAY};font-weight:700;
                      text-transform:uppercase;letter-spacing:0.5px;margin-bottom:6px;">
            Target Alignment After Transition
          </div>
          {_svg_donut(sc['alignment'], sc['color'])}
        </div>
      </div>'''

    today = datetime.date.today().strftime('%B %d, %Y')

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: {_LIGHT}; font-family: -apple-system, Arial, sans-serif;
          padding: 0; }}
  @media print {{
    body {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    @page {{ margin: 0.4in; size: landscape; }}
  }}
</style>
</head>
<body>

<!-- Header -->
<div style="background:{_WHITE};border-bottom:3px solid {_COPPER};
            padding:14px 24px;display:flex;justify-content:space-between;
            align-items:center;margin-bottom:20px;">
  <div>{logo_tag}</div>
  <div style="text-align:center;">
    <div style="font-size:24px;font-weight:800;color:{_NAVY};">Portfolio Transition Optimizer</div>
    <div style="font-size:13px;color:{_GRAY};margin-top:4px;">Tax-Efficient Rebalancing Analysis</div>
  </div>
  <div style="text-align:right;">
    <div style="font-size:15px;font-weight:700;color:{_NAVY};">{client_name}</div>
    <div style="font-size:12px;color:{_GRAY};margin-top:3px;">Acct {account_num} &nbsp;·&nbsp; {today}</div>
  </div>
</div>

<!-- Metrics -->
<div style="display:flex;gap:10px;padding:0 24px;margin-bottom:20px;">
  {metric_html}
</div>

<!-- Scenario cards -->
<div style="display:flex;gap:14px;padding:0 24px;margin-bottom:20px;">
  {cards_html}
</div>

<!-- Footer -->
<div style="text-align:center;color:{_GRAY};font-size:10px;font-style:italic;
            padding:0 24px 20px 24px;">
  Illustrative purposes only. Tax estimates are approximations and may differ from actual liability.
  Consult a qualified tax advisor before transacting.
</div>

</body>
</html>"""

    return html.encode('utf-8')

    # ── Header (white, gold bottom border) ──────────────────────────────────
    ax_hdr = fig.add_axes([0, 0.905, 1.0, 0.095])
    ax_hdr.set_facecolor(_WHITE)
    ax_hdr.axis('off')
    ax_hdr.axhline(y=0, color=_COPPER, linewidth=5, xmin=0, xmax=1)

    # Logo top-left
    logo_placed = False
    if logo_path and Path(logo_path).exists():
        try:
            logo_img = plt.imread(str(logo_path))
            imagebox = OffsetImage(logo_img, zoom=0.11)
            ab = AnnotationBbox(imagebox, (0.075, 0.52),
                                frameon=False, xycoords='axes fraction',
                                box_alignment=(0.5, 0.5))
            ax_hdr.add_artist(ab)
            logo_placed = True
        except Exception:
            pass
    if not logo_placed:
        ax_hdr.text(0.02, 0.52, 'BISON WEALTH', color=_NAVY,
                    fontsize=13, fontweight='black', transform=ax_hdr.transAxes, va='center')

    # Centre title
    ax_hdr.text(0.5, 0.70, 'Portfolio Transition Optimizer',
                color=_NAVY, fontsize=19, fontweight='bold',
                ha='center', va='center', transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.25, 'Tax-Efficient Rebalancing Analysis',
                color=_GRAY, fontsize=10,
                ha='center', va='center', transform=ax_hdr.transAxes)

    # Client info top-right
    ax_hdr.text(0.985, 0.70, client_name,
                color=_NAVY, fontsize=12, fontweight='bold',
                ha='right', va='center', transform=ax_hdr.transAxes)
    ax_hdr.text(0.985, 0.25,
                f'Acct {account_num}  ·  {datetime.date.today().strftime("%B %d, %Y")}',
                color=_GRAY, fontsize=9,
                ha='right', va='center', transform=ax_hdr.transAxes)

    # ── Metric boxes ────────────────────────────────────────────────────────
    metrics = [
        ('Portfolio Value',      f'${total_value:,.0f}',                                   _NAVY),
        ('LT Unrealized G/L',    ('+' if lt_unreal  >= 0 else '-') + f'${abs(lt_unreal):,.0f}',  _GREEN if lt_unreal  >= 0 else _RED),
        ('ST Unrealized G/L',    ('+' if st_unreal  >= 0 else '-') + f'${abs(st_unreal):,.0f}',  _GREEN if st_unreal  >= 0 else _RED),
        ('Total Unrealized G/L', ('+' if net_unreal >= 0 else '-') + f'${abs(net_unreal):,.0f}', _GREEN if net_unreal >= 0 else _RED),
    ]
    MBW = 0.236   # metric box width
    MBH = 0.082   # metric box height
    MBY = 0.808   # metric boxes y (bottom)
    MGP = 0.009   # gap between metric boxes
    for i, (label, value, vcolor) in enumerate(metrics):
        bx = 0.013 + i * (MBW + MGP)
        ax = fig.add_axes([bx, MBY, MBW, MBH])
        ax.set_facecolor(_WHITE)
        for sp in ax.spines.values():
            sp.set_edgecolor(_BORDER); sp.set_linewidth(0.8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.05, 0.72, label, color=_GRAY, fontsize=8.5, fontweight='bold',
                transform=ax.transAxes, va='center')
        ax.text(0.05, 0.25, value, color=vcolor, fontsize=16, fontweight='bold',
                transform=ax.transAxes, va='center')

    # ── Scenario cards ──────────────────────────────────────────────────────
    CW   = 0.320   # card width
    CGP  = 0.013   # gap
    CX   = [0.013 + i * (CW + CGP) for i in range(3)]

    # Vertical zones (figure coords, bottom=0)
    HDR_Y = 0.740; HDR_H = 0.054      # coloured title strip
    SBY   = 0.415; SBH   = 0.325      # white stats body  (sits below header)
    LBY   = 0.375; LBH   = 0.033      # "TARGET ALIGNMENT" label row
    DNY   = 0.065; DNH   = 0.305      # donut area (frameless)

    for i, sc in enumerate(scenarios):
        cx = CX[i]

        # Coloured header strip
        ax_sh = fig.add_axes([cx, HDR_Y, CW, HDR_H])
        ax_sh.set_facecolor(sc['color'])
        ax_sh.axis('off')
        ax_sh.text(0.5, 0.5, sc['name'].upper(),
                   color=_WHITE, fontsize=12, fontweight='bold',
                   ha='center', va='center', transform=ax_sh.transAxes)

        # White stats body
        ax_sb = fig.add_axes([cx, SBY, CW, SBH])
        ax_sb.set_facecolor(_WHITE)
        for sp in ax_sb.spines.values():
            sp.set_edgecolor(_BORDER); sp.set_linewidth(0.8)
        ax_sb.set_xticks([]); ax_sb.set_yticks([])

        tax_color = _RED   if sc['net_tax']    > 0  else _GREEN
        gl_color  = _GREEN if sc['realized_gl'] <= 0 else _RED
        gl_sign   = '+' if sc['realized_gl'] > 0 else ''
        stat_rows = [
            ('ESTIMATED TAX LIABILITY', f"${sc['net_tax']:,.0f}",              tax_color),
            ('REALIZED GAIN / LOSS',    f"{gl_sign}${sc['realized_gl']:,.0f}", gl_color),
            ('TRANSITION AMOUNT',       f"${sc['proceeds']:,.0f}",             _NAVY),
        ]
        for j, (lbl, val, col) in enumerate(stat_rows):
            y_lbl = 0.87 - j * 0.295
            y_val = y_lbl - 0.10
            ax_sb.text(0.5, y_lbl, lbl, color=_GRAY, fontsize=7.5, fontweight='bold',
                       ha='center', va='center', transform=ax_sb.transAxes)
            ax_sb.text(0.5, y_val, val, color=col, fontsize=18, fontweight='bold',
                       ha='center', va='center', transform=ax_sb.transAxes)
            if j < 2:
                ax_sb.axhline(y=y_val - 0.095, color=_DIVIDER,
                              linewidth=0.8, xmin=0.04, xmax=0.96)

        # "TARGET ALIGNMENT" label — frameless
        ax_lbl = fig.add_axes([cx, LBY, CW, LBH])
        ax_lbl.set_facecolor(_LIGHT)
        ax_lbl.axis('off')
        ax_lbl.text(0.5, 0.5, 'TARGET ALIGNMENT AFTER TRANSITION',
                    color=_GRAY, fontsize=7.5, fontweight='bold',
                    ha='center', va='center', transform=ax_lbl.transAxes)

        # Donut — no frame, no axes, just the pie
        pad = CW * 0.18
        ax_pie = fig.add_axes([cx + pad, DNY, CW - 2*pad, DNH])
        ax_pie.set_aspect('equal')
        ax_pie.axis('off')
        pct = float(sc['alignment'])
        ax_pie.pie(
            [max(pct, 0.5), max(100 - pct, 0)],
            colors=[sc['color'], '#DDE3EA'],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.40, edgecolor=_WHITE, linewidth=3),
        )
        ax_pie.text(0, 0, f'{pct:.0f}%',
                    ha='center', va='center',
                    fontsize=20, fontweight='bold', color='#1C2833')

    # ── Footer ──────────────────────────────────────────────────────────────
    ax_ft = fig.add_axes([0, 0, 1.0, 0.055])
    ax_ft.set_facecolor(_LIGHT)
    ax_ft.axis('off')
    ax_ft.text(0.5, 0.5,
               'Illustrative purposes only. Tax estimates are approximations and may differ from actual liability. '
               'Consult a qualified tax advisor before transacting.',
               color=_GRAY, fontsize=8, ha='center', va='center',
               transform=ax_ft.transAxes, style='italic')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor=_LIGHT, edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def generate_summary_pdf(client_name, account_num, total_value,
                         lt_unreal, st_unreal, net_unreal, scenarios, logo_b64=None):
    """Convert the summary HTML to a PDF using WeasyPrint and return bytes."""
    html_bytes = generate_summary_html(client_name, account_num, total_value,
                                       lt_unreal, st_unreal, net_unreal,
                                       scenarios, logo_b64=logo_b64)
    return WeasyprintHTML(string=html_bytes.decode('utf-8')).write_pdf()


def generate_trades_excel(scenario, account_num, holdings_df):
    """Build a trade order sheet: Account ID, Action, Ticker, Shares."""
    price_lookup = {str(r['Ticker']).strip(): float(r['Price'])
                    for _, r in holdings_df.iterrows()}
    rows = []
    # Sells — dollar amount = shares sold × price
    for i, row in holdings_df.iterrows():
        frac = float(scenario['sell_fracs'][i])
        if frac < 0.005:
            continue
        amount = round(frac * float(row['Shares']) * float(row['Price']), 2)
        rows.append({
            'Account ID': account_num,
            'Action':     'Sell',
            'Ticker':     str(row['Ticker']).strip(),
            'Amount ($)': amount,
        })
    # Buys — dollar amount comes directly from the optimizer output
    buy_df = scenario.get('buys')
    if buy_df is not None and not buy_df.empty:
        for _, brow in buy_df.iterrows():
            rows.append({
                'Account ID': account_num,
                'Action':     'Buy',
                'Ticker':     str(brow['Ticker']).strip(),
                'Amount ($)': round(float(brow['Buy Value']), 2),
            })
    df = pd.DataFrame(rows, columns=['Account ID', 'Action', 'Ticker', 'Amount ($)'])
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Trades')
    buf.seek(0)
    return buf.read()


# ── Template helpers ──────────────────────────────────────────────────────────
def _make_holdings_template() -> bytes:
    """Return an Excel file (bytes) users can fill in for the simple holdings format."""
    rows = [
        {'Ticker': 'AAPL',  'Shares': 100,  'Price': 195.50, 'Cost_Basis': 120.00, 'Holding': 'LT'},
        {'Ticker': 'MSFT',  'Shares': 50,   'Price': 415.00, 'Cost_Basis': 380.00, 'Holding': 'LT'},
        {'Ticker': 'TSLA',  'Shares': 30,   'Price': 175.00, 'Cost_Basis': 210.00, 'Holding': 'ST'},
    ]
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Holdings')
        ws = writer.sheets['Holdings']
        ws.column_dimensions['A'].width = 10
        for col in ['B', 'C', 'D']:
            ws.column_dimensions[col].width = 12
        ws.column_dimensions['E'].width = 10
    return buf.getvalue()


def _make_model_template() -> bytes:
    """Return an Excel file (bytes) for the target model format."""
    rows = [
        {'Ticker': 'AAPL', 'Weight': 0.05},
        {'Ticker': 'MSFT', 'Weight': 0.05},
        {'Ticker': 'GOOGL', 'Weight': 0.05},
    ]
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Model')
        ws = writer.sheets['Model']
        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 10
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if _logo_b64:
        st.markdown(
            f'<div style="margin-bottom:20px;">'
            f'<img src="data:image/svg+xml;base64,{_logo_b64}" '
            f'style="width:100%;max-width:200px;object-fit:contain;" /></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div style='color:{NAVY};font-size:18px;font-weight:900;"
                    f"letter-spacing:2px;margin-bottom:20px;'>ORANGE INVESTMENT ADVISORS</div>",
                    unsafe_allow_html=True)

    # ── Client ────────────────────────────────────────────────────────────────
    st.subheader("Client Information")
    client_name = st.text_input("Client Name", "Acme Family Trust")
    account_num = st.text_input("Account Number", "*7842")

    # ── Tax rates ─────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Tax Rates")
    lt_rate = st.slider("Long-Term Rate",  0, 40, 20, format="%d%%") / 100
    st_rate = st.slider("Short-Term Rate", 0, 55, 37, format="%d%%") / 100

    # ── Custom scenario ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Custom Scenario")
    custom_budget = st.number_input("Tax Budget ($)", value=10000, step=1)

    # ── Target model ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Target Model")

    model_options = list(PRESET_MODELS.keys()) + ["Upload custom", "Build custom"]
    selected_model_name = st.selectbox("Model", model_options,
                                       label_visibility="collapsed")

    _edit_model = False
    if selected_model_name == "Upload custom":
        st.download_button(
            "Download template",
            data=_make_model_template(),
            file_name="model_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        model_file = st.file_uploader("Model file", type=["csv", "xlsx", "xls"],
                                      label_visibility="collapsed")
        if model_file:
            if model_file.name.lower().endswith(('.xlsx', '.xls')):
                raw_model = pd.read_excel(model_file)
            else:
                raw_model = pd.read_csv(model_file)
            target_model = parse_model(raw_model)
        else:
            target_model = None

    elif selected_model_name == "Build custom":
        st.caption("Edit tickers and weights in the table below.")
        _edit_model = True
        target_model = None  # set in main area below

    else:
        target_model = PRESET_MODELS.get(selected_model_name)

    # Show model summary
    if target_model is not None and not target_model.empty:
        st.caption(f"{len(target_model)} positions · weights sum to "
                   f"{target_model['Weight'].sum()*100:.1f}%")

    # ── Holdings data ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Holdings Data")
    data_source = st.radio("Source", ["Upload file", "Edit table"],
                           label_visibility="collapsed")

    # Initialize flag to prevent NameError
    _edit_holdings = False

    if data_source == "Upload file":
        st.download_button(
            "Download template",
            data=_make_holdings_template(),
            file_name="holdings_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        uploaded = st.file_uploader("Holdings file", type=["csv", "xlsx", "xls"],
                                    label_visibility="collapsed")
        if uploaded:
            fname = uploaded.name.lower()
            if fname.endswith(('.xlsx', '.xls')):
                xl  = pd.ExcelFile(uploaded)
                raw = None
                for sheet in xl.sheet_names:
                    df_try = xl.parse(sheet)
                    df_try.columns = [str(c).strip() for c in df_try.columns]
                    if 'Units' in df_try.columns or 'Ticker' in df_try.columns:
                        raw = df_try
                        break
                if raw is None:
                    raw = xl.parse(xl.sheet_names[0])
            else:
                raw_text = uploaded.read().decode('utf-8', errors='replace')
                uploaded.seek(0)
                header_row = 0
                for i, line in enumerate(raw_text.splitlines()[:15]):
                    if 'Units' in line or 'Ticker' in line:
                        header_row = i
                        break
                raw = pd.read_csv(uploaded, skiprows=header_row)
        else:
            raw = _EMPTY_HOLDINGS.copy()

    else:  # Edit table
        _edit_holdings = True
        raw = _EMPTY_HOLDINGS.copy()


# ══════════════════════════════════════════════════════════════════════════════
# FULL-WIDTH EDITORS (rendered in main area so they have room to breathe)
# ══════════════════════════════════════════════════════════════════════════════
_BLANK_HOLDING = {'Ticker': '', 'Shares': 0.0, 'Price': 0.0, 'Cost_Basis': 0.0, 'Holding': 'LT'}
_BLANK_MODEL   = {'Ticker': '', 'Weight': 0.0}

if _edit_holdings:
    if 'holdings_df' not in st.session_state:
        st.session_state.holdings_df = _EMPTY_HOLDINGS.copy()
    # Ensure the select column exists
    if '_sel' not in st.session_state.holdings_df.columns:
        st.session_state.holdings_df.insert(0, '_sel', False)

    st.subheader("Edit Holdings")
    _HOLDINGS_CFG = {
        "_sel":       st.column_config.CheckboxColumn("✓",        width="small"),
        "Ticker":     st.column_config.TextColumn("Ticker",        width="medium"),
        "Shares":     st.column_config.NumberColumn("Shares",      min_value=0, step=1,    format="%.2f",  width="medium"),
        "Price":      st.column_config.NumberColumn("Price ($)",   min_value=0, step=0.01, format="$%.2f", width="medium"),
        "Cost_Basis": st.column_config.NumberColumn("Cost/Share",  min_value=0, step=0.01, format="$%.2f", width="medium"),
        "Holding":    st.column_config.SelectboxColumn("Holding",  options=["LT", "ST"],   width="small"),
    }
    edited_holdings = st.data_editor(
        st.session_state.holdings_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        height=max(420, 36 + len(st.session_state.holdings_df) * 36),
        column_config=_HOLDINGS_CFG,
        key="holdings_editor",
    )
    st.session_state.holdings_df = edited_holdings

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("+ Add Row", key="add_holding_row", type="primary", use_container_width=True):
            new_row = pd.DataFrame([{**{'_sel': False}, **_BLANK_HOLDING}])
            st.session_state.holdings_df = pd.concat(
                [st.session_state.holdings_df, new_row], ignore_index=True,
            )
            st.rerun()
    with btn_c2:
        if st.button("Delete Selected", key="del_holding_row", use_container_width=True):
            st.session_state.holdings_df = (
                st.session_state.holdings_df[~st.session_state.holdings_df['_sel']]
                .reset_index(drop=True)
            )
            st.rerun()

    # Drop the select column before passing to parser
    raw = st.session_state.holdings_df.drop(columns=['_sel'], errors='ignore')
    st.divider()

if _edit_model:
    if 'model_df' not in st.session_state:
        st.session_state.model_df = DEFAULT_MODEL.copy()
    if '_sel' not in st.session_state.model_df.columns:
        st.session_state.model_df.insert(0, '_sel', False)

    st.subheader("Build Custom Target Model")
    st.caption("Check rows to select them, then click **Delete Selected** to remove. Weights are automatically normalised to 100%.")
    _MODEL_CFG = {
        "_sel":   st.column_config.CheckboxColumn("✓",      width="small"),
        "Ticker": st.column_config.TextColumn("Ticker",      width="medium"),
        "Weight": st.column_config.NumberColumn("Weight",    min_value=0.0, max_value=1.0,
                                                 step=0.01, format="%.4f", width="medium"),
    }
    edited_model = st.data_editor(
        st.session_state.model_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        height=max(360, 36 + len(st.session_state.model_df) * 36),
        column_config=_MODEL_CFG,
        key="model_editor",
    )
    st.session_state.model_df = edited_model

    btn_c1, btn_c2 = st.columns(2)
    with btn_c1:
        if st.button("+ Add Row", key="add_model_row", type="primary", use_container_width=True):
            new_row = pd.DataFrame([{**{'_sel': False}, **_BLANK_MODEL}])
            st.session_state.model_df = pd.concat(
                [st.session_state.model_df, new_row], ignore_index=True,
            )
            st.rerun()
    with btn_c2:
        if st.button("Delete Selected", key="del_model_row", use_container_width=True):
            st.session_state.model_df = (
                st.session_state.model_df[~st.session_state.model_df['_sel']]
                .reset_index(drop=True)
            )
            st.rerun()

    custom_model_raw = st.session_state.model_df.drop(columns=['_sel'], errors='ignore')
    target_model = parse_model(custom_model_raw)
    if target_model is not None and not target_model.empty:
        st.caption(f"{len(target_model)} positions · weights sum to "
                   f"{target_model['Weight'].sum()*100:.1f}%")
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE
# ══════════════════════════════════════════════════════════════════════════════
holdings = parse_holdings(raw, lt_rate, st_rate)
if holdings.empty:
    st.info("Upload a holdings file or use **Edit table** in the sidebar to get started.")
    st.stop()

TOTAL_VALUE  = float(holdings['MarketValue'].sum())
MAX_TAX      = float(holdings['MaxTax'].sum())
total_gains  = float(holdings['UnrealizedGL'].clip(lower=0).sum())
total_losses = float(holdings['UnrealizedGL'].clip(upper=0).sum())
net_unrealized = total_gains + total_losses   # gains positive, losses negative

lt_holdings = holdings[holdings['Holding'] == 'LT']
st_holdings = holdings[holdings['Holding'] == 'ST']
lt_unrealized = float(lt_holdings['UnrealizedGL'].sum())
st_unrealized = float(st_holdings['UnrealizedGL'].sum())

# Overlap: % of current portfolio (by value) already in the target model
if target_model is not None and not target_model.empty and TOTAL_VALUE > 0:
    target_tickers = set(target_model['Ticker'].str.strip())
    overlap_val = holdings[holdings['Ticker'].isin(target_tickers)]['MarketValue'].sum()
    overlap_pct = overlap_val / TOTAL_VALUE * 100
else:
    overlap_pct = 0.0

scenario_defs = [
    dict(name='Minimum Tax',    budget=0.0,                    color=GREEN),
    dict(name='Custom',         budget=float(custom_budget),   color=NAVY),
    dict(name='Max Transition', budget=MAX_TAX,                color=RED),
]
scenarios = []
for d in scenario_defs:
    r = solve_transition(holdings, d['budget'], target_model, TOTAL_VALUE)
    if r:
        sc = {**d, **r}
        sc['buys']      = compute_buys(holdings, r['sell_fracs'], target_model, TOTAL_VALUE)
        sc['alignment'] = alignment_score(holdings, r['sell_fracs'], target_model, TOTAL_VALUE)
        scenarios.append(sc)


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bison-header">
  <div class="bison-left">
    <div class="bison-page-title">Portfolio Transition Optimizer</div>
    <div class="bison-page-sub">Tax-Efficient Rebalancing Analysis
      · Target: <b>{selected_model_name if selected_model_name not in ("Upload custom","Build custom") else "Custom Model"}</b>
    </div>
  </div>
  <div class="bison-client-info">
    <div class="bison-client-name">{client_name}</div>
    <div>Acct {account_num}</div>
    <div>${TOTAL_VALUE:,.0f} portfolio value</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Summary metrics ────────────────────────────────────────────────────────────
def _fmt_gl(v):
    if v >= 0:
        return f"+${v:,.0f}"
    return f"-${abs(v):,.0f}"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Portfolio Value",         f"${TOTAL_VALUE:,.0f}")
m2.metric("LT Unrealized Gain/Loss", _fmt_gl(lt_unrealized))
m3.metric("ST Unrealized Gain/Loss", _fmt_gl(st_unrealized))
m4.metric("Total Unrealized G/L",    _fmt_gl(net_unrealized))

st.markdown("<br>", unsafe_allow_html=True)

# ── Scenario cards ─────────────────────────────────────────────────────────────
cols = st.columns(3)
for col, sc in zip(cols, scenarios):
    tx_color = RED   if sc['net_tax']    > 0  else GREEN
    gl_color = GREEN if sc['realized_gl'] <= 0 else RED
    gl_sign  = '+' if sc['realized_gl'] > 0 else ''
    with col:
        st.markdown(f"""
        <div class="sc-header" style="background:{sc['color']};">{sc['name'].upper()}</div>
        <div class="sc-body">
          <div class="sc-label">Estimated Tax Liability</div>
          <div class="sc-value" style="color:{tx_color};">${sc['net_tax']:,.0f}</div>
          <hr class="sc-divider"/>
          <div class="sc-label">Realized Gain / Loss</div>
          <div class="sc-value" style="color:{gl_color};">{gl_sign}${sc['realized_gl']:,.0f}</div>
          <hr class="sc-divider"/>
          <div class="sc-label">Transition Amount</div>
          <div class="sc-value" style="color:{NAVY};">${sc['proceeds']:,.0f}</div>
          <hr class="sc-divider"/>
          <div class="sc-label" style="margin-top:8px;">Target Alignment After Transition</div>
        </div>
        <div style="margin-top:20px;"></div>
        """, unsafe_allow_html=True)
        st.plotly_chart(make_donut(sc['alignment'], sc['color']),
                        use_container_width=True, config={'staticPlot': True})

# ── Export summary ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
if scenarios:
    if _WEASYPRINT_OK:
        pdf_bytes = generate_summary_pdf(
            client_name, account_num, TOTAL_VALUE,
            lt_unrealized, st_unrealized, net_unrealized, scenarios,
            logo_b64=_logo_b64,
        )
        st.download_button(
            label="Export Summary (PDF)",
            data=pdf_bytes,
            file_name=f"transition_summary_{client_name.replace(' ','_')}.pdf",
            mime="application/pdf",
        )
    else:
        html_bytes = generate_summary_html(
            client_name, account_num, TOTAL_VALUE,
            lt_unrealized, st_unrealized, net_unrealized, scenarios,
            logo_b64=_logo_b64,
        )
        st.download_button(
            label="Export Summary (HTML — open in browser, Ctrl+P to save as PDF)",
            data=html_bytes,
            file_name=f"transition_summary_{client_name.replace(' ','_')}.html",
            mime="text/html",
        )
        st.caption("Install WeasyPrint for one-click PDF: `conda install -c conda-forge weasyprint`")

st.divider()

# ── Transition Timeline ────────────────────────────────────────────────────────
st.subheader("Transition Timeline")
st.caption(f"How long to fully transition the account at ${custom_budget:,.0f}/yr tax budget")
timeline_data = simulate_transition_timeline(
    holdings, float(custom_budget), target_model, TOTAL_VALUE
)
if timeline_data:
    total_years = len(timeline_data) - 1  # exclude Year 0 baseline
    fully_done  = timeline_data[-1]['alignment_pct'] >= 99.9
    total_tax   = sum(d['tax_incurred'] for d in timeline_data)
    tl_c1, tl_c2, tl_c3 = st.columns(3)
    tl_c1.metric("Years to Full Transition" if fully_done else "Years Simulated",
                 f"{total_years} yr{'s' if total_years != 1 else ''}")
    tl_c2.metric("Cumulative Tax Over Period", f"${total_tax:,.0f}")
    tl_c3.metric("Final Alignment", f"{timeline_data[-1]['alignment_pct']:.0f}%")
    st.plotly_chart(make_timeline_chart(timeline_data, float(custom_budget)),
                    use_container_width=True, config={'displayModeBar': False})
else:
    st.info("Enter a tax budget greater than $0 in the sidebar to see the transition timeline.")

st.divider()

# ── Trade-level detail ─────────────────────────────────────────────────────────
st.subheader("Trade-Level Detail")
tabs = st.tabs([sc['name'] for sc in scenarios])

for tab, sc in zip(tabs, scenarios):
    with tab:
        sell_tab, buy_tab = st.tabs(["Sells", "Buys"])

        # ── Sells ──────────────────────────────────────────────────────────
        with sell_tab:
            sell_rows = []
            for i, row in holdings.iterrows():
                frac = float(sc['sell_fracs'][i])
                if frac < 0.005:
                    continue
                shares_sold = frac * float(row['Shares'])
                proceeds    = shares_sold * float(row['Price'])
                realized_gl = float(row['UnrealizedGL']) * frac
                tax_impact  = realized_gl * float(row['TaxRate'])
                sell_rows.append({
                    'Ticker':       row['Ticker'],
                    'Lot':          row['Holding'],
                    'Sell %':       round(frac * 100, 1),
                    'Shares Sold':  round(shares_sold, 2),
                    'Proceeds':     round(proceeds, 0),
                    'Realized G/L': round(realized_gl, 0),
                    'Tax Impact':   round(tax_impact, 0),
                })
            if sell_rows:
                tbl = pd.DataFrame(sell_rows)
                st.dataframe(
                    tbl.style
                    .format({'Sell %': '{:.1f}%', 'Shares Sold': '{:.2f}',
                             'Proceeds': '${:,.0f}', 'Realized G/L': '${:,.0f}',
                             'Tax Impact': '${:,.0f}'})
                    .map(
                        lambda v: f'color:{GREEN}' if isinstance(v, (int,float)) and v < 0
                        else (f'color:{RED}' if isinstance(v, (int,float)) and v > 0 else ''),
                        subset=['Realized G/L', 'Tax Impact'],
                    ),
                    use_container_width=True, hide_index=True,
                )
                c1, c2, c3 = st.columns(3)
                c1.metric("Sell Trades",    len(tbl))
                c2.metric("Total Proceeds", f"${tbl['Proceeds'].sum():,.0f}")
                c3.metric("Total Tax",      f"${tbl['Tax Impact'].sum():,.0f}")
            else:
                st.info("No sells recommended for this scenario.")

        # ── Buys ───────────────────────────────────────────────────────────
        with buy_tab:
            buy_df = sc['buys']
            if buy_df is not None and not buy_df.empty:
                st.dataframe(
                    buy_df.style
                    .format({'Target Weight %': '{:.2f}%', 'Actual Weight %': '{:.2f}%',
                             'Buy Value': '${:,.0f}'}),
                    use_container_width=True, hide_index=True,
                )
                c1, c2 = st.columns(2)
                c1.metric("Buy Trades",       len(buy_df))
                c2.metric("Total Buy Value",  f"${buy_df['Buy Value'].sum():,.0f}")
            else:
                if target_model is None or target_model.empty:
                    st.info("Select a target model in the sidebar to see buy recommendations.")
                else:
                    st.info("No buys needed — portfolio already aligned with target model.")


# ── Export trades ──────────────────────────────────────────────────────────────
if scenarios:
    st.divider()
    exp_c1, exp_c2 = st.columns([2, 3])
    with exp_c1:
        sc_names  = [sc['name'] for sc in scenarios]
        sc_choice = st.selectbox("Scenario", sc_names, label_visibility="collapsed")
    with exp_c2:
        sc_sel      = next(sc for sc in scenarios if sc['name'] == sc_choice)
        excel_bytes = generate_trades_excel(sc_sel, account_num, holdings)
        st.download_button(
            label=f"Export Trades — {sc_choice} (Excel)",
            data=excel_bytes,
            file_name=f"trades_{sc_choice.replace(' ','_')}_{account_num}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bison-footer">
  Illustrative purposes only. Tax estimates assume LT rate {lt_rate*100:.0f}%,
  ST rate {st_rate*100:.0f}%. Actual liability may differ.
  Consult a qualified tax advisor before transacting.
</div>
""", unsafe_allow_html=True)

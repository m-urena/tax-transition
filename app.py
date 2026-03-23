"""
Bison Wealth Management -- Portfolio Transition Optimizer
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog
import warnings
import base64
from pathlib import Path
warnings.filterwarnings('ignore')

# ── Logo ───────────────────────────────────────────────────────────────────────
_logo_path = Path(__file__).parent / "Bison Wealth Logo.png"
_logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode() if _logo_path.exists() else None

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Bison | Transition Optimizer",
                   layout="wide", initial_sidebar_state="expanded")

# ── Brand colors ───────────────────────────────────────────────────────────────
NAVY    = '#263759'
COPPER  = '#C17A49'
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
  .add-row-btn button {{
    background-color:{COPPER} !important; color:{WHITE} !important;
    font-size:15px !important; font-weight:700 !important;
    padding:10px 28px !important; border-radius:6px !important;
    border:none !important; cursor:pointer !important;
    width:100% !important;
  }}
  .add-row-btn button:hover {{ opacity:0.88 !important; }}
  .del-row-btn button {{
    background-color:{RED} !important; color:{WHITE} !important;
    font-size:15px !important; font-weight:700 !important;
    padding:10px 28px !important; border-radius:6px !important;
    border:none !important; cursor:pointer !important;
    width:100% !important;
  }}
  .del-row-btn button:hover {{ opacity:0.88 !important; }}
</style>""", unsafe_allow_html=True)


# ── Load preset model portfolios ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_preset_models():
    path = Path(__file__).parent / "model portfolios.xlsx"
    if not path.exists():
        return {}
    raw = pd.read_excel(path, header=None)
    models = {}
    # Row 0 = model names, Row 1 = "Holding"/"Weight" headers, Rows 2+ = data
    # Columns are in pairs: (ticker_col, weight_col) starting at col 1
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
            # Normalise weights to sum to 1
            if df['Weight'].sum() > 0:
                df['Weight'] = df['Weight'] / df['Weight'].sum()
            models[name] = df
        col_idx += 2
    return models


PRESET_MODELS = load_preset_models()

# ── Default dummy holdings ─────────────────────────────────────────────────────
DEFAULT_HOLDINGS = pd.DataFrame({
    'Ticker':     ['AAPL','MSFT','GOOGL','AMZN','NVDA','META','GE','XOM',
                   'JNJ','PG','TSLA','BAC','INTC','DIS','PFE'],
    'Shares':     [200, 150, 50, 100, 60, 120, 500, 250,
                   200, 150, 80, 400, 300, 200, 400],
    'Price':      [182.0,378.0,177.0,187.0,812.0,485.0,168.0,108.0,
                   158.0,157.0,198.0, 34.0, 29.0, 92.0, 27.0],
    'Cost_Basis': [ 45.0,120.0, 85.0, 95.0,218.0,198.0, 85.0, 74.0,
                   145.0,128.0,285.0, 42.0, 56.0,118.0, 41.0],
    'Holding':    ['LT','LT','LT','LT','LT','LT','LT','LT',
                   'LT','LT','ST','LT','LT','LT','LT'],
})

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


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    if _logo_b64:
        st.markdown(
            f'<div style="margin-bottom:20px;">'
            f'<img src="data:image/png;base64,{_logo_b64}" '
            f'style="width:100%;max-width:200px;object-fit:contain;" /></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"<div style='color:{NAVY};font-size:22px;font-weight:900;"
                    f"letter-spacing:3px;margin-bottom:20px;'>BISON WEALTH</div>",
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
    selected_model_name = st.selectbox("Select model", model_options,
                                       label_visibility="collapsed")

    _edit_model = False
    if selected_model_name == "Upload custom":
        st.caption("Columns: Ticker, Weight (or Allocation)")
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
    data_source = st.radio("Source", ["Use dummy data", "Upload file", "Edit table"],
                           label_visibility="collapsed")

    if data_source == "Upload file":
        st.caption("Columns: Ticker, Units, Price, Cost Per Share, "
                   "Market Value, Short Term Gain Loss, Long Term Gain Loss")
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
            raw = DEFAULT_HOLDINGS.copy()
    elif data_source == "Edit table":
        _edit_holdings = True
        raw = DEFAULT_HOLDINGS.copy()  # placeholder, overridden in main area
    else:
        _edit_holdings = False
        raw = DEFAULT_HOLDINGS.copy()


# ══════════════════════════════════════════════════════════════════════════════
# FULL-WIDTH EDITORS (rendered in main area so they have room to breathe)
# ══════════════════════════════════════════════════════════════════════════════
_BLANK_HOLDING = {'Ticker': '', 'Shares': 0.0, 'Price': 0.0, 'Cost_Basis': 0.0, 'Holding': 'LT'}
_BLANK_MODEL   = {'Ticker': '', 'Weight': 0.0}

if _edit_holdings:
    if 'holdings_df' not in st.session_state:
        st.session_state.holdings_df = DEFAULT_HOLDINGS.copy()
    # Ensure the select column exists
    if '_sel' not in st.session_state.holdings_df.columns:
        st.session_state.holdings_df.insert(0, '_sel', False)

    st.subheader("Edit Holdings")
    st.caption("Check rows to select them, then click **Delete Selected** to remove. Click **+ Add Row** to add a position.")
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
        st.markdown('<div class="add-row-btn">', unsafe_allow_html=True)
        if st.button("+ Add Row", key="add_holding_row"):
            new_row = pd.DataFrame([{**{'_sel': False}, **_BLANK_HOLDING}])
            st.session_state.holdings_df = pd.concat(
                [st.session_state.holdings_df, new_row], ignore_index=True,
            )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_c2:
        st.markdown('<div class="del-row-btn">', unsafe_allow_html=True)
        if st.button("Delete Selected", key="del_holding_row"):
            st.session_state.holdings_df = (
                st.session_state.holdings_df[~st.session_state.holdings_df['_sel']]
                .reset_index(drop=True)
            )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="add-row-btn">', unsafe_allow_html=True)
        if st.button("+ Add Row", key="add_model_row"):
            new_row = pd.DataFrame([{**{'_sel': False}, **_BLANK_MODEL}])
            st.session_state.model_df = pd.concat(
                [st.session_state.model_df, new_row], ignore_index=True,
            )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with btn_c2:
        st.markdown('<div class="del-row-btn">', unsafe_allow_html=True)
        if st.button("Delete Selected", key="del_model_row"):
            st.session_state.model_df = (
                st.session_state.model_df[~st.session_state.model_df['_sel']]
                .reset_index(drop=True)
            )
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

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
  <div>
    <div class="bison-page-title">Portfolio Transition Optimizer</div>
    <div class="bison-page-sub">Tax-Efficient Rebalancing Analysis
      {"· Target: <b>" + selected_model_name + "</b>" if selected_model_name not in ("Upload custom","Build custom") else ""}
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
                tax_impact  = max(0.0, realized_gl) * float(row['TaxRate'])
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
                    .applymap(
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


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="bison-footer">
  Illustrative purposes only. Tax estimates assume LT rate {lt_rate*100:.0f}%,
  ST rate {st_rate*100:.0f}%. Actual liability may differ.
  Consult a qualified tax advisor before transacting.
</div>
""", unsafe_allow_html=True)

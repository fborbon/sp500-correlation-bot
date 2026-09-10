"""
SP500 Correlation Bot — Dashboard
Run with:  streamlit run dashboard.py --server.headless true
"""
import base64
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config import OUTPUTS_DIR, TOP_N_HIGHLIGHT
from reporting.html_builders import (
    _build_echarts_html, _build_table_html, _vol_norm,
    _SIGNALS_DESC, _FUND_DESC, _CACHE_SUBDIR, _CHART_START_DATE,
)

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title='SP500 Bot Dashboard',
    page_icon='📈',
    layout='wide',
)

# ── Translate button (injected into the parent document — st.markdown never
# executes <script> tags, but a components.html iframe can reach window.parent) ──
import streamlit.components.v1 as components
components.html(
    """
    <script>
    (function() {
      var d = window.parent.document;
      if (!d.getElementById('ff-translate-loader')) {
        var s = d.createElement('script');
        s.id = 'ff-translate-loader';
        s.defer = true;
        s.src = 'https://www.forwardforecasting.eu/assets/translate-widget.js';
        d.body.appendChild(s);
      }
    })();
    </script>
    """,
    height=0,
)

# ── Mobile detection via User-Agent (no JS, no iframes, works on all browsers) ──
# st.context.headers is available server-side — no client round-trip needed.
_ua = st.context.headers.get("User-Agent", "")
_ua_mobile = any(k in _ua for k in ("Mobile", "Android", "iPhone", "iPad", "iPod"))

# Manual override: ?m=1 forces mobile, ?m=0 forces desktop
_param = st.query_params.get("m", "")
if _param == "1":
    is_mobile = True
elif _param == "0":
    is_mobile = False
else:
    is_mobile = _ua_mobile

# ── Responsive CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@media (max-width: 768px) {
    /* Kill horizontal scroll at every level */
    *, *::before, *::after { box-sizing: border-box !important; }
    html, body {
        overflow-x: hidden !important;
        width: 100% !important;
        max-width: 100vw !important;
    }
    [data-testid="stApp"],
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    section.main, .main {
        overflow-x: hidden !important;
        width: 100% !important;
        max-width: 100vw !important;
        min-width: unset !important;
    }
    .main .block-container {
        padding-left: 0.6rem !important;
        padding-right: 0.6rem !important;
        padding-top: 0.75rem !important;
        width: 100% !important;
        max-width: 100vw !important;
        min-width: unset !important;
        overflow-x: hidden !important;
    }
    /* Clamp dataframes and any wide child */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrame"] > div,
    iframe { max-width: 100% !important; }
    h1 { font-size: 1.25rem !important; line-height: 1.35 !important; }
    h2 { font-size: 1rem !important; }
    p  { font-size: 0.85rem !important; }
    /* Tab bar: single scrollable row, no wrapping */
    [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        overflow-y: hidden !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
        scrollbar-width: none !important;
        gap: 0 !important;
    }
    [data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
    /* Each tab button: fixed compact size, no text wrap */
    [data-baseweb="tab"] {
        flex-shrink: 0 !important;
        white-space: nowrap !important;
        font-size: 1.1rem !important;
        padding: 0.4rem 0.6rem !important;
        min-width: 0 !important;
    }
    /* Force metric columns to stay in a single horizontal row, no gaps */
    [data-testid="stHorizontalBlock"] {
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 0 !important;
    }
    [data-testid="column"] {
        min-width: 0 !important;
        flex: 1 1 0 !important;
        width: auto !important;
        padding: 0 !important;
        border-right: 1px solid #ddd !important;
    }
    [data-testid="column"]:last-child { border-right: none !important; }
    [data-testid="stMetric"] { padding: 0.3rem 0.4rem !important; text-align: center !important; }
    [data-testid="stMetric"] label { font-size: 0.62rem !important; white-space: nowrap !important; }
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    hr { margin: 0.4rem 0 !important; }
}
</style>
""", unsafe_allow_html=True)


# ── Helpers (desktop — base64 iframe path) ───────────────────────────────────

def _html_iframe(html: str, height: int, scrolling: bool = False) -> None:
    b64 = base64.b64encode(html.encode()).decode()
    st.iframe(src=f"data:text/html;charset=utf-8;base64,{b64}", height=height)


def _render_cached(run_dir, name: str, height: int) -> bool:
    """Render pre-built HTML from cache. Returns True on cache hit."""
    cp = run_dir / _CACHE_SUBDIR / f'{name}.html'
    if cp.exists():
        _html_iframe(cp.read_text(encoding='utf-8'), height=height)
        return True
    return False


@st.cache_data
def _load_csv(path, **kwargs):
    return pd.read_csv(path, **kwargs)


def _dual_scroll_table(df, row_styles=None, height=520, link_cols=None,
                        cell_hrefs=None, col_descriptions=None):
    html = _build_table_html(df, row_styles, height, link_cols, cell_hrefs, col_descriptions)
    _html_iframe(html, height=height + 30)


def _chart_title(title: str, tooltip: str):
    tip_json = json.dumps(tooltip)
    html = f"""<!DOCTYPE html><html><head><style>
body{{margin:0;padding:0;font-family:"Source Sans Pro",sans-serif;}}
.ttl{{font-size:1.05rem;font-weight:600;color:#0f0f0f;cursor:default;
      display:inline-block;padding:2px 0 3px;border-bottom:2px dashed #bbb;line-height:1.3;}}
</style></head><body><div class="ttl" id="ttl">{title}</div>
<script>
var pdoc=window.parent.document,tip=pdoc.getElementById('__chart_tip__');
if(!tip){{tip=pdoc.createElement('div');tip.id='__chart_tip__';
  tip.style.cssText='position:fixed;background:#222;color:#f5f5f5;padding:9px 13px;'+
    'border-radius:6px;font-size:13px;max-width:360px;line-height:1.5;z-index:99999;'+
    'pointer-events:none;display:none;white-space:normal;box-shadow:0 2px 8px rgba(0,0,0,0.4);';
  pdoc.body.appendChild(tip);}}
var el=document.getElementById('ttl'),msg={tip_json},timer=null;
el.addEventListener('mouseenter',function(){{timer=setTimeout(function(){{tip.textContent=msg;tip.style.display='block';}},2000);}});
el.addEventListener('mousemove',function(e){{var fr=window.frameElement?window.frameElement.getBoundingClientRect():{{left:0,top:0}};
  tip.style.left=(fr.left+e.clientX+16)+'px';tip.style.top=(fr.top+e.clientY-10)+'px';}});
el.addEventListener('mouseleave',function(){{clearTimeout(timer);tip.style.display='none';}});
</script></body></html>"""
    _html_iframe(html, height=38)


def _kpi_row(items: list) -> None:
    """Compact mobile KPI strip: label + value on one line, divided by vertical lines."""
    cells = ''.join(
        f'<div style="flex:1;text-align:center;padding:0.3rem 0.2rem;'
        f'{"border-right:1px solid #666;" if i < len(items) - 1 else ""}">'
        f'<span style="font-size:0.82rem;color:#fff;font-weight:600;">{lbl}:&nbsp;&nbsp;{val}</span>'
        f'</div>'
        for i, (lbl, val) in enumerate(items)
    )
    st.markdown(
        f'<div style="display:flex;flex-wrap:nowrap;background:#444;'
        f'border-radius:6px;overflow:hidden;margin-bottom:0.4rem;">{cells}</div>',
        unsafe_allow_html=True,
    )


def _echarts_dual_chart(df_all, top_t, nm, norm_fn, abs_fn,
                        title, y1_label, y2_label, height=880,
                        start_date=_CHART_START_DATE, top_t_bottom=None,
                        fit_to_highlights=True):
    html = _build_echarts_html(df_all, top_t, nm, norm_fn, abs_fn,
                               title, y1_label, y2_label, height,
                               start_date, top_t_bottom, fit_to_highlights)
    _html_iframe(html, height=height + 55)


# ── Mobile helpers — Yahoo Finance strategy ───────────────────────────────────
# No base64 iframes. Tables → st.dataframe (native DOM, works on all browsers).
# Charts → st.image with pre-built PNGs from General/ (350–790 KB vs 8.9 MB).

def _mobile_table(df: pd.DataFrame, style_fn, height: int = 400,
                  fmt: dict = None) -> None:
    """Render a DataFrame with row-level colour using Streamlit's native table.

    fmt maps column names to format strings (e.g. '{:.1f}%').
    Values that cannot be converted to float are shown as '—'.
    """
    df = df.copy()
    # Pre-convert any column we intend to format to numeric, silently coercing
    # strings like 'N/A' or '' to NaN so the format string never sees a str.
    if fmt:
        for col in fmt:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    styled = df.style.apply(style_fn, axis=None)
    if fmt:
        safe = {col: (lambda v, f=fstr: f.format(v) if pd.notna(v) else '—')
                for col, fstr in fmt.items() if col in df.columns}
        styled = styled.format(safe, na_rep='—')
    st.dataframe(styled, height=height, use_container_width=True)


def _mobile_chart(gen_dir, filename: str, caption: str = '') -> None:
    """Embed a pre-built PNG as an inline base64 <img> tag.

    Uses st.markdown instead of st.image so the image travels in the
    WebSocket message and never needs a separate HTTP round-trip through
    nginx. Safari allows data: URIs in <img> (only blocks them in iframes).
    """
    p = gen_dir / filename
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode()
        cap = (f'<p style="font-size:0.72rem;color:#888;text-align:center;'
               f'margin:3px 0 0;">{caption}</p>') if caption else ''
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;display:block;border-radius:6px;"/>{cap}',
            unsafe_allow_html=True,
        )
    else:
        st.info('Chart not yet generated — run the bot with save_plots=True.')


# ── Run directory ─────────────────────────────────────────────────────────────
run_dirs = sorted(
    [d for d in OUTPUTS_DIR.iterdir() if d.is_dir() and not d.name.startswith('.')],
    reverse=True,
)
if not run_dirs:
    st.warning('No runs found in outputs/. Run the bot first.')
    st.stop()

run_dir  = run_dirs[0]
corr_dir = run_dir / 'Correlation_method'
gen_dir  = run_dir / 'General'

# ── Page header ───────────────────────────────────────────────────────────────
_tcol, _bcol = st.columns([6, 1])
_tcol.title('📈 SP500 Correlation Bot')
with _bcol:
    st.write('')  # vertical alignment
    if is_mobile:
        if st.button('🖥️', help='Switch to desktop view'):
            st.query_params['m'] = '0'
            st.rerun()
    else:
        if st.button('📱', help='Switch to mobile view'):
            st.query_params['m'] = '1'
            st.rerun()

if not is_mobile:
    st.markdown(
        """
        Correlation-based trading signal system for the full **S&P 500** universe.
        Daily returns are computed for all ~500 companies and Pearson correlations identified.
        A **Random Forest** regression model predicts each stock's 7-day return using its
        top correlated peers as features. Signals are classified as **BUY** (predicted return > 1 %),
        **SELL** (< −10 %) or **HOLD**, enriched with fundamental data, and displayed here
        alongside interactive price, volume and market-cap time series.
        """,
        unsafe_allow_html=False,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────
if is_mobile:
    _tab_labels = ['📋', '🏦', '📊', '📈', '📦', '💹', '🔗']
else:
    _tab_labels = ['📋 Signals', '🏦 Fundamentals', '📊 Mkt Cap',
                   '📈 Prices', '📦 Volume', '💹 Returns', '🔗 Correlation']

tab_signals, tab_fund, tab_mcap, tab_prices, tab_volume, tab_returns, tab_corr = st.tabs(_tab_labels)


# ── Tab 1: Signals ────────────────────────────────────────────────────────────
with tab_signals:
    signals_path = run_dir / 'signals.csv'
    if not signals_path.exists():
        st.info('signals.csv not found for this run.')
    else:
        df = _load_csv(str(signals_path))

        buys  = (df['signal'] == 'BUY').sum()
        sells = (df['signal'] == 'SELL').sum()
        holds = (df['signal'] == 'HOLD').sum()
        if is_mobile:
            _kpi_row([('Tickers', len(df)), ('BUY', buys), ('SELL', sells), ('HOLD', holds)])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Tickers', len(df))
            c2.metric('BUY',  buys)
            c3.metric('SELL', sells)
            c4.metric('HOLD', holds)
        st.divider()

        if is_mobile:
            # ── Mobile: st.dataframe, no iframes — all columns ────────────────
            st.subheader('📋 Signals')
            df_mob = df.copy()

            _SIG_BG = {'BUY': '#d4edda', 'SELL': '#f8d7da', 'HOLD': '#fff3cd'}
            _SIG_FG = {'BUY': '#155724', 'SELL': '#721c24', 'HOLD': '#856404'}

            def _sig_style(frame):
                styles = pd.DataFrame('', index=frame.index, columns=frame.columns)
                if 'signal' in frame.columns:
                    for idx, sig in frame['signal'].items():
                        bg = _SIG_BG.get(sig, '')
                        fg = _SIG_FG.get(sig, '')
                        if bg:
                            styles.loc[idx, :] = f'background-color:{bg};color:{fg}'
                return styles

            _fmt = {c: v for c, v in {
                'predicted_return': '{:+.1f}%',
                'current_price':    '${:.2f}',
                'target_price_7d':  '${:.2f}',
                'model_r2':         '{:.2f}',
                'market_cap_B':     '${:.1f}B',
            }.items() if c in df_mob.columns}
            _mobile_table(df_mob, _sig_style, height=460, fmt=_fmt)
            st.caption('🟢 BUY  🔴 SELL  🟡 HOLD — scroll right for more columns')
        else:
            # ── Desktop: base64 iframe with full feature set ──────────────────
            if not _render_cached(run_dir, 'signals_table', 530):
                top_return_tickers = set(
                    df.dropna(subset=['predicted_return'])
                      .nlargest(TOP_N_HIGHLIGHT, 'predicted_return')['ticker']
                )
                SIG_STYLE = {
                    'BUY':  'background-color:#d4edda;color:#155724',
                    'SELL': 'background-color:#f8d7da;color:#721c24',
                    'HOLD': 'background-color:#fff3cd;color:#856404',
                }
                TOP_RETURN_STYLE = 'background-color:#cce5ff;color:#004085;font-weight:bold'
                row_styles = {}
                for i, row in df.iterrows():
                    sig  = SIG_STYLE.get(row.get('signal', ''), '')
                    tick = row.get('ticker', '')
                    if tick in top_return_tickers and not sig:
                        row_styles[i] = TOP_RETURN_STYLE
                    elif sig:
                        row_styles[i] = sig
                df_reset   = df.reset_index(drop=True)
                cell_hrefs = {}
                if corr_dir.exists():
                    for i, row in df_reset.iterrows():
                        plot = corr_dir / f'analysis_{row["ticker"]}.png'
                        if plot.exists():
                            b64 = base64.b64encode(plot.read_bytes()).decode()
                            cell_hrefs[('signal', i)] = f'data:image/png;base64,{b64}'
                _dual_scroll_table(df_reset, row_styles, height=500,
                                   link_cols={'ticker': 'https://finance.yahoo.com/quote/{value}'},
                                   cell_hrefs=cell_hrefs, col_descriptions=_SIGNALS_DESC)
            st.caption(
                f'🔵 Top {TOP_N_HIGHLIGHT} highest predicted return  '
                '🟢 BUY  🔴 SELL  🟡 HOLD  '
                '(underlined signal = click to view prediction chart)'
            )


# ── Tab 2: Fundamentals ───────────────────────────────────────────────────────
with tab_fund:
    path = run_dir / 'fundamentals.csv'
    if not path.exists():
        st.info('fundamentals.csv not found for this run.')
    else:
        df = _load_csv(str(path), index_col=0)

        if is_mobile:
            items = [('Companies', len(df))]
            if 'likelihood_pct' in df.columns:
                items += [
                    ('Avg score', f"{df['likelihood_pct'].mean():.1f}%"),
                    ('Top score', f"{df['likelihood_pct'].max():.1f}%"),
                ]
            _kpi_row(items)
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric('Companies', len(df))
            if 'likelihood_pct' in df.columns:
                c2.metric('Avg score', f"{df['likelihood_pct'].mean():.1f}%")
                c3.metric('Top score', f"{df['likelihood_pct'].max():.1f}%")
        st.divider()

        if is_mobile:
            # ── Mobile: st.dataframe — all columns ────────────────────────────
            st.subheader('🏦 Fundamentals')
            df_fund = df.reset_index()

            def _fund_style(frame):
                styles = pd.DataFrame('', index=frame.index, columns=frame.columns)
                if 'likelihood_pct' in frame.columns:
                    for idx, val in frame['likelihood_pct'].items():
                        try:
                            v = float(val)
                            if v >= 70:   color = 'background-color:#d4edda;color:#155724'
                            elif v >= 50: color = 'background-color:#fff3cd;color:#856404'
                            else:         color = 'background-color:#f8d7da;color:#721c24'
                            styles.loc[idx, :] = color
                        except Exception:
                            pass
                return styles

            _ffmt = {c: v for c, v in {
                'likelihood_pct':   '{:.1f}%',
                'market_cap_B':     '${:.1f}B',
                'pe_ratio':         '{:.1f}',
                'peg_ratio':        '{:.2f}',
                'ps_ratio':         '{:.1f}',
                'pb_ratio':         '{:.1f}',
                'ev_ebitda':        '{:.1f}',
                'eps':              '${:.2f}',
                'revenue_growth':   '{:.1f}%',
                'gross_margin':     '{:.1f}%',
                'operating_margin': '{:.1f}%',
                'net_margin':       '{:.1f}%',
                'earnings_growth':  '{:.1f}%',
                'debt_to_equity':   '{:.2f}',
                'current_ratio':    '{:.2f}',
            }.items() if c in df_fund.columns}
            _mobile_table(df_fund, _fund_style, height=460, fmt=_ffmt)
            st.caption('🟢 ≥70%  🟡 ≥50%  🔴 <50%  12-month price increase likelihood — scroll right for more columns')
        else:
            # ── Desktop ───────────────────────────────────────────────────────
            if not _render_cached(run_dir, 'fund_table', 570):
                row_styles = {}
                if 'likelihood_pct' in df.columns:
                    for i, val in enumerate(df['likelihood_pct']):
                        try:
                            v = float(val)
                            if v >= 70:   row_styles[i] = 'background-color:#d4edda;color:#155724'
                            elif v >= 50: row_styles[i] = 'background-color:#fff3cd;color:#856404'
                            else:         row_styles[i] = 'background-color:#f8d7da;color:#721c24'
                        except Exception:
                            pass
                _dual_scroll_table(df.reset_index(), row_styles, height=540,
                                   link_cols={'ticker': 'https://finance.yahoo.com/quote/{value}'},
                                   col_descriptions=_FUND_DESC)
            st.caption('🟢 ≥70%  🟡 ≥50%  🔴 <50%  likelihood of price increase in 12 months')


# ── Tab 3: Market Cap ─────────────────────────────────────────────────────────
with tab_mcap:
    signals_path = run_dir / 'signals.csv'
    if not signals_path.exists():
        st.info('signals.csv not found for this run.')
    else:
        sdf = _load_csv(str(signals_path))
        if {'ticker', 'company_name', 'current_price', 'market_cap_B', 'sector'}.issubset(sdf.columns):
            sdf = sdf.dropna(subset=['market_cap_B']).sort_values('market_cap_B', ascending=False)

            if is_mobile:
                # ── Mobile: pre-built PNGs ────────────────────────────────────
                st.subheader('📊 Market Cap')
                _mobile_chart(gen_dir, 'market_cap_bars.png',
                              f'Top vs bottom {TOP_N_HIGHLIGHT} companies — close price & market cap')
                st.divider()
                _mobile_chart(gen_dir, 'market_cap_series_absolute.png',
                              f'Market cap time series — top {TOP_N_HIGHLIGHT} by current market cap')
                st.divider()
                _mobile_chart(gen_dir, 'market_cap_series_normalized.png',
                              f'Market cap time series — top {TOP_N_HIGHLIGHT} by cap growth')
            else:
                # ── Desktop: interactive Plotly + ECharts ─────────────────────
                top    = sdf.head(TOP_N_HIGHLIGHT)
                bottom = sdf.tail(TOP_N_HIGHLIGHT)
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                    subplot_titles=[
                        f'Last Close Price ($) — top {TOP_N_HIGHLIGHT} vs bottom {TOP_N_HIGHLIGHT}',
                        f'Market Capitalization ($B) — top {TOP_N_HIGHLIGHT} vs bottom {TOP_N_HIGHLIGHT}',
                    ],
                )
                hover_price = ('<b>%{customdata[0]}</b><br>Sector: %{customdata[1]}<br>'
                               'Price: $%{y:,.2f}<br>Cap: $%{customdata[2]:.1f}B<extra></extra>')
                hover_cap   = ('<b>%{customdata[0]}</b><br>Sector: %{customdata[1]}<br>'
                               'Cap: $%{y:.1f}B<br>Price: $%{customdata[2]:,.2f}<extra></extra>')
                for grp, colour, name in [(top, 'mediumseagreen', f'Top {TOP_N_HIGHLIGHT}'),
                                          (bottom, 'tomato', f'Bottom {TOP_N_HIGHLIGHT}')]:
                    fig.add_trace(go.Bar(
                        x=grp['ticker'], y=grp['current_price'], name=name,
                        marker_color=colour,
                        customdata=grp[['company_name', 'sector', 'market_cap_B']].values,
                        hovertemplate=hover_price,
                    ), row=1, col=1)
                    fig.add_trace(go.Bar(
                        x=grp['ticker'], y=grp['market_cap_B'], name=name,
                        marker_color=colour, showlegend=False,
                        customdata=grp[['company_name', 'sector', 'current_price']].values,
                        hovertemplate=hover_cap,
                    ), row=2, col=1)
                fig.update_layout(
                    height=750, title=f'Top vs Bottom {TOP_N_HIGHLIGHT} S&P 500 Companies',
                    barmode='group',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                )
                fig.update_yaxes(title_text='Price ($)', row=1, col=1)
                fig.update_yaxes(title_text='Market Cap ($B)', row=2, col=1)
                _chart_title('Market Cap Bars — hover bars for details',
                    f'Bar chart: closing price (top) and market cap in $B (bottom) '
                    f'for the top {TOP_N_HIGHLIGHT} and bottom {TOP_N_HIGHLIGHT} companies.')
                st.plotly_chart(fig, width='stretch')

                prices_path = run_dir / 'prices.csv'
                if prices_path.exists():
                    prices_df = _load_csv(str(prices_path), index_col=0, parse_dates=True)
                    nm = dict(zip(sdf['ticker'], sdf['company_name']))
                    mcap_series = {}
                    for _, r in sdf.iterrows():
                        t = r['ticker']
                        if t in prices_df.columns and r['current_price'] > 0:
                            mcap_series[t] = prices_df[t] * (r['market_cap_B'] / r['current_price'])
                    mcap_df = pd.DataFrame(mcap_series)
                    if not mcap_df.empty:
                        by_abs  = [t for t in sdf['ticker'] if t in mcap_df.columns][:TOP_N_HIGHLIGHT]
                        _mcap_f = mcap_df.loc[mcap_df.index >= pd.Timestamp(_CHART_START_DATE)]
                        by_norm = (_mcap_f.iloc[-1] / _mcap_f.iloc[0]).sort_values(ascending=False).index.tolist()[:TOP_N_HIGHLIGHT]
                        st.divider()
                        _chart_title(f'Market Cap Time Series — Top {TOP_N_HIGHLIGHT} by Current Market Cap',
                            f'Estimated historical market cap for the {TOP_N_HIGHLIGHT} largest companies.')
                        if not _render_cached(run_dir, 'mcap_series_abs', 935):
                            _echarts_dual_chart(mcap_df, by_abs, nm,
                                norm_fn=lambda s: s / s.iloc[0] * 100, abs_fn=lambda s: s,
                                title=f'Top {TOP_N_HIGHLIGHT} by Current Market Cap',
                                y1_label='Normalized (base=100)', y2_label='Market Cap ($B)')
                        st.divider()
                        _chart_title(f'Market Cap Time Series — Top {TOP_N_HIGHLIGHT} by Normalized Cap Growth',
                            f'Same data highlighting fastest-growing companies by market cap.')
                        if not _render_cached(run_dir, 'mcap_series_norm', 935):
                            _echarts_dual_chart(mcap_df, by_norm, nm,
                                norm_fn=lambda s: s / s.iloc[0] * 100, abs_fn=lambda s: s,
                                title=f'Top {TOP_N_HIGHLIGHT} by Normalized Cap Growth',
                                y1_label='Normalized (base=100)', y2_label='Market Cap ($B)')
        else:
            st.info('Required columns missing in signals.csv.')


# ── Tab 4: Price Series ───────────────────────────────────────────────────────
with tab_prices:
    signals_path = run_dir / 'signals.csv'
    prices_path  = run_dir / 'prices.csv'

    if is_mobile:
        # ── Mobile: pre-built PNGs — all 3 price charts ───────────────────────
        st.subheader('📈 Price Series')
        _mobile_chart(gen_dir, 'price_series_market-cap.png',
                      f'Top {TOP_N_HIGHLIGHT} by market cap')
        st.divider()
        _mobile_chart(gen_dir, 'price_series_stock-price-absolute.png',
                      f'Top {TOP_N_HIGHLIGHT} by stock price')
        st.divider()
        _mobile_chart(gen_dir, 'price_series_normalized-return.png',
                      f'Top {TOP_N_HIGHLIGHT} by normalized return')
    else:
        # ── Desktop: interactive ECharts ──────────────────────────────────────
        if prices_path.exists() and signals_path.exists():
            prices_df = _load_csv(str(prices_path), index_col=0, parse_dates=True)
            sdf_full  = _load_csv(str(signals_path))
            nm        = dict(zip(sdf_full['ticker'], sdf_full.get('company_name', sdf_full['ticker'])))

            mcap_col = 'market_cap_B' if 'market_cap_B' in sdf_full.columns else None
            by_mcap  = (sdf_full.dropna(subset=[mcap_col])
                                 .sort_values(mcap_col, ascending=False)['ticker'].tolist()
                        if mcap_col else list(prices_df.columns))
            by_price = prices_df.iloc[-1].sort_values(ascending=False).index.tolist()
            _pf = prices_df.loc[prices_df.index >= pd.Timestamp(_CHART_START_DATE)]
            by_norm_price = ((_pf.iloc[-1] / _pf.iloc[0]).sort_values(ascending=False).index.tolist()
                             if not _pf.empty else by_price)
            by_norm  = (prices_df.iloc[-1] / prices_df.iloc[0]).sort_values(ascending=False).index.tolist()

            _kw = dict(norm_fn=lambda s: s / s.iloc[0] * 100, abs_fn=lambda s: s,
                       y1_label='Normalized (base=100)', y2_label='Close price ($)')

            for cname, ordering, title, tip in [
                ('price_by_mcap', by_mcap,
                 f'Price Series — Top {TOP_N_HIGHLIGHT} by Market Cap',
                 f'Top {TOP_N_HIGHLIGHT} largest S&P 500 companies by market cap. '
                 f'Normalised to base=100 at {_CHART_START_DATE} (top). Absolute price $ (bottom).'),
                ('price_by_price', by_price,
                 f'Price Series — Top {TOP_N_HIGHLIGHT} by Stock Price',
                 f'Top {TOP_N_HIGHLIGHT} companies by absolute closing price. '
                 'High price ≠ large company — depends on shares outstanding.'),
                ('price_by_norm_price', by_norm_price,
                 f'Price Series — Top {TOP_N_HIGHLIGHT} by Normalized Return',
                 f'Best-performing stocks since {_CHART_START_DATE} by price growth.'),
                ('price_by_norm', by_norm,
                 f'Price Series — Top {TOP_N_HIGHLIGHT} by Return (full history)',
                 f'Best-performing stocks from their first data point.'),
            ]:
                top_t = [t for t in ordering if t in prices_df.columns][:TOP_N_HIGHLIGHT]
                _chart_title(title, tip)
                if not _render_cached(run_dir, cname, 935):
                    _echarts_dual_chart(prices_df, top_t, nm, title=title, **_kw)
                st.divider()
        else:
            st.info('prices.csv not found — re-run the bot to enable interactive charts.')


# ── Tab 5: Volume ─────────────────────────────────────────────────────────────
with tab_volume:
    volume_path = run_dir / 'volume.csv'
    if not volume_path.exists():
        st.info('volume.csv not found — re-run the bot to enable this tab.')
    elif is_mobile:
        # ── Mobile: pre-built PNGs ────────────────────────────────────────────
        st.subheader('📦 Volume')
        _mobile_chart(gen_dir, 'volume_series_absolute.png',
                      f'Top {TOP_N_HIGHLIGHT} by average daily volume')
        st.divider()
        _mobile_chart(gen_dir, 'volume_series_normalized.png',
                      f'Top {TOP_N_HIGHLIGHT} by volume growth')
    else:
        # ── Desktop: interactive ECharts ──────────────────────────────────────
        vol_df  = _load_csv(str(volume_path), index_col=0, parse_dates=True)
        sdf_vol = _load_csv(str(run_dir / 'signals.csv')) if (run_dir / 'signals.csv').exists() else None
        nm_vol  = (dict(zip(sdf_vol['ticker'], sdf_vol['company_name']))
                   if sdf_vol is not None and 'company_name' in sdf_vol.columns else {})

        by_abs_v   = vol_df.mean().sort_values(ascending=False).index.tolist()[:TOP_N_HIGHLIGHT]
        _norm_last = {t: _vol_norm(vol_df[t]).iloc[-1] for t in vol_df.columns}
        by_norm_v  = sorted(_norm_last, key=lambda t: _norm_last[t] if pd.notna(_norm_last[t]) else 0,
                            reverse=True)[:TOP_N_HIGHLIGHT]

        _chart_title(f'Volume Time Series — Top {TOP_N_HIGHLIGHT} by Average Volume',
            f'Top {TOP_N_HIGHLIGHT} most actively traded S&P 500 stocks by average daily volume. '
            'Top: normalised to base=100. Bottom: absolute shares traded.')
        if not _render_cached(run_dir, 'volume_abs', 935):
            _echarts_dual_chart(vol_df, by_abs_v, nm_vol,
                norm_fn=_vol_norm, abs_fn=lambda s: s,
                title=f'Top {TOP_N_HIGHLIGHT} by Average Volume',
                y1_label='Normalized (base=100)', y2_label='Volume (shares)')
        st.divider()
        _chart_title(f'Volume Time Series — Top {TOP_N_HIGHLIGHT} by Normalized Volume Growth',
            f'Top {TOP_N_HIGHLIGHT} stocks with the most volume growth relative to their own history.')
        if not _render_cached(run_dir, 'volume_norm', 935):
            _echarts_dual_chart(vol_df, by_norm_v, nm_vol,
                norm_fn=_vol_norm, abs_fn=lambda s: s,
                title=f'Top {TOP_N_HIGHLIGHT} by Normalized Volume Growth',
                y1_label='Normalized (base=100)', y2_label='Volume (shares)')


# ── Tab 6: Cumulative Returns ─────────────────────────────────────────────────
with tab_returns:
    prices_path_r = run_dir / 'prices.csv'
    if not prices_path_r.exists():
        st.info('prices.csv not found — re-run the bot to enable this tab.')
    elif is_mobile:
        # ── Mobile: pre-built PNG ─────────────────────────────────────────────
        st.subheader('💹 Cumulative Returns')
        _mobile_chart(gen_dir, 'cumulative_returns.png',
                      f'Top {TOP_N_HIGHLIGHT} best performers since {_CHART_START_DATE}')
    else:
        # ── Desktop: interactive EChart ───────────────────────────────────────
        ret_prices = _load_csv(str(prices_path_r), index_col=0, parse_dates=True)
        sdf_r = _load_csv(str(run_dir / 'signals.csv')) if (run_dir / 'signals.csv').exists() else None
        nm_r  = (dict(zip(sdf_r['ticker'], sdf_r['company_name']))
                 if sdf_r is not None and 'company_name' in sdf_r.columns else {})
        _ret_f  = ret_prices.loc[ret_prices.index >= pd.Timestamp(_CHART_START_DATE)]
        top_t_r = ((_ret_f.iloc[-1] / _ret_f.iloc[0] - 1) * 100).sort_values(ascending=False).index.tolist()[:TOP_N_HIGHLIGHT]
        _chart_title(f'Cumulative Returns — Top {TOP_N_HIGHLIGHT} best performers',
            f'Top {TOP_N_HIGHLIGHT} stocks by cumulative return since {_CHART_START_DATE}. '
            'Top: cumulative % return. Bottom: dollar return per share.')
        if not _render_cached(run_dir, 'returns', 935):
            _echarts_dual_chart(ret_prices, top_t_r, nm_r,
                norm_fn=lambda s: (s / s.iloc[0] - 1) * 100,
                abs_fn=lambda s: s - s.iloc[0],
                title=f'Cumulative Returns — Top {TOP_N_HIGHLIGHT} best performers highlighted',
                y1_label='Cum. return (%)', y2_label='$ return/share')


# ── Tab 7: Correlation Analysis ───────────────────────────────────────────────
with tab_corr:
    if not corr_dir.exists():
        st.info('No Correlation_method/ plots found for this run.')
    else:
        matrix_path = corr_dir / 'correlation_matrix.png'
        if matrix_path.exists():
            if is_mobile:
                st.subheader('🔗 Correlation Matrix')
            else:
                _chart_title('Correlation Matrix',
                    f'Pearson correlation matrix for the top {TOP_N_HIGHLIGHT} S&P 500 companies. '
                    'Red = strong inverse. Green = strong direct. Values near 0 = no linear relationship.')
            if is_mobile:
                b64m = base64.b64encode(matrix_path.read_bytes()).decode()
                st.markdown(
                    f'<img src="data:image/png;base64,{b64m}" '
                    f'style="width:100%;display:block;border-radius:6px;"/>',
                    unsafe_allow_html=True,
                )
            else:
                st.image(str(matrix_path), width='stretch')
            st.divider()
        else:
            st.info('No plots found in Correlation_method/ for this run.')

        if is_mobile:
            st.caption('Open on desktop for the interactive correlation explorer.')
        else:
            prices_path_c  = run_dir / 'prices.csv'
            signals_path_c = run_dir / 'signals.csv'
            if prices_path_c.exists() and signals_path_c.exists():
                prices_c = _load_csv(str(prices_path_c), index_col=0, parse_dates=True)
                sdf_c    = _load_csv(str(signals_path_c))
                nm_c     = (dict(zip(sdf_c['ticker'], sdf_c['company_name']))
                            if 'company_name' in sdf_c.columns else {})
                mcap_col_c = 'market_cap_B' if 'market_cap_B' in sdf_c.columns else None
                corr_tickers = (sdf_c.dropna(subset=[mcap_col_c])
                                     .sort_values(mcap_col_c, ascending=False)['ticker']
                                     .head(TOP_N_HIGHLIGHT).tolist()
                                if mcap_col_c else list(prices_c.columns)[:TOP_N_HIGHLIGHT])
                corr_tickers = [t for t in corr_tickers if t in prices_c.columns]
                opt_map = {f"{nm_c.get(t, t)} ({t})": t for t in corr_tickers}
                selected_labels = st.multiselect(
                    'Select companies to plot',
                    options=list(opt_map.keys()),
                    default=list(opt_map.keys())[:5],
                )
                selected_t = [opt_map[lbl] for lbl in selected_labels]
                if selected_t:
                    _chart_title('Price Series — Selected Companies',
                        'Price chart for the selected companies. '
                        'Top: normalised to base=100. Bottom: absolute price $.')
                    _echarts_dual_chart(prices_c[selected_t], selected_t, nm_c,
                        norm_fn=lambda s: s / s.iloc[0] * 100, abs_fn=lambda s: s,
                        title='Price Series — Selected Companies',
                        y1_label='Normalized (base=100)', y2_label='Close price ($)')
                else:
                    st.info('Select at least one company above to display the chart.')

# ── Footer ────────────────────────────────────────────────────────────────────
try:
    from datetime import datetime as _dt
    _run_ts = _dt.strptime(run_dir.name, '%Y-%m-%d_%H-%M').strftime('%Y-%m-%d %H:%M UTC')
except Exception:
    _run_ts = run_dir.name

st.divider()
st.markdown(
    f"<div style='text-align:center;color:#666;font-size:0.8rem;padding:0.4rem 0 1rem;'>"
    f"🕐 Last bot run: <strong style='color:#aaa;'>{_run_ts}</strong>"
    f"</div>",
    unsafe_allow_html=True,
)

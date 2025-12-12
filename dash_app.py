import pandas as pd
import yfinance as yf
import numpy as np
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)
server = app.server

TICKERS = {"NASDAQ": "QQQ", "DJIA": "DIA"}  # use ETF proxies for cleaner aligned data
CACHE = {}  # simple in-memory cache: {(symbol, period, interval): (timestamp, series)}
CACHE_TTL = 300  # 5-minute TTL for cached data


def generate_insights(ratio, ratio_pct_change, threshold):
    """Generate short, auto-calculated insights based on latest ratio and pct change."""
    insights = []
    if ratio is None or ratio.empty or ratio_pct_change is None or ratio_pct_change.empty:
        return ["No data to generate insights."]

    try:
        latest_ratio = float(ratio.iloc[-1])
        latest_pct = float(ratio_pct_change.iloc[-1])
    except Exception:
        return ["Insufficient data for insights."]

    # Simple recent slope (last ~5 points)
    try:
        recent = ratio.dropna()
        if len(recent) >= 6:
            slope = recent.iloc[-1] - recent.iloc[-6]
        elif len(recent) > 1:
            slope = recent.iloc[-1] - recent.iloc[0]
        else:
            slope = 0.0
    except Exception:
        slope = 0.0

    if slope > 0:
        insights.append("Trend: NASDAQ is gaining vs DJIA (rotation into AI).")
    elif slope < 0:
        insights.append("Trend: NASDAQ is lagging vs DJIA (rotation out of AI).")
    else:
        insights.append("Trend: NASDAQ vs DJIA ratio is flat.")

    if abs(latest_pct) >= threshold:
        direction = "into AI" if latest_pct > 0 else "out of AI"
        insights.append(f"Reversal: {abs(latest_pct):.2f}% {direction} (threshold {threshold}%).")
    else:
        insights.append(f"No reversal: last change {latest_pct:.2f}% within threshold {threshold}%.")

    insights.append(f"Current ratio: {latest_ratio:.4f}.")
    return insights

def compute_backtest_stats(ratio, ratio_pct_change, threshold):
    """Compute simple backtest metrics: reversal count, win rate, avg forward return."""
    try:
        reversal_signal = ratio_pct_change.abs() >= threshold
        reversal_count = reversal_signal.sum()
        
        if reversal_count == 0:
            return {"reversals": 0, "win_rate": "N/A", "forward_1d": "N/A", "forward_5d": "N/A"}
        
        # compute forward returns for 1-day and 5-day ahead of each reversal
        forward_1d_returns = []
        forward_5d_returns = []
        
        reversal_indices = ratio_pct_change[reversal_signal].index
        for i, rev_date in enumerate(reversal_indices):
            try:
                idx_pos = ratio.index.get_loc(rev_date)
                # 1-day forward
                if idx_pos + 1 < len(ratio):
                    fwd_1d = ((ratio.iloc[idx_pos + 1] - ratio.iloc[idx_pos]) / ratio.iloc[idx_pos]) * 100
                    forward_1d_returns.append(fwd_1d)
                # 5-day forward
                if idx_pos + 5 < len(ratio):
                    fwd_5d = ((ratio.iloc[idx_pos + 5] - ratio.iloc[idx_pos]) / ratio.iloc[idx_pos]) * 100
                    forward_5d_returns.append(fwd_5d)
            except Exception:
                pass
        
        avg_fwd_1d = np.mean(forward_1d_returns) if forward_1d_returns else np.nan
        avg_fwd_5d = np.mean(forward_5d_returns) if forward_5d_returns else np.nan
        win_rate_1d = (len([x for x in forward_1d_returns if x > 0]) / len(forward_1d_returns) * 100) if forward_1d_returns else np.nan
        
        return {
            "reversals": int(reversal_count),
            "win_rate": f"{win_rate_1d:.1f}%" if not np.isnan(win_rate_1d) else "N/A",
            "forward_1d": f"{avg_fwd_1d:.3f}%" if not np.isnan(avg_fwd_1d) else "N/A",
            "forward_5d": f"{avg_fwd_5d:.3f}%" if not np.isnan(avg_fwd_5d) else "N/A"
        }
    except Exception:
        return {"reversals": 0, "win_rate": "N/A", "forward_1d": "N/A", "forward_5d": "N/A"}

def fetch_series(symbol, period, interval='1d'):
    # yfinance: for minute intervals the maximum period is limited; caller should choose period accordingly
    cache_key = (symbol, period, interval)
    now = time.time()
    
    # check cache first
    if cache_key in CACHE:
        cached_time, cached_series = CACHE[cache_key]
        if now - cached_time < CACHE_TTL:
            return cached_series
    
    # fetch fresh data
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    s = df["Close"].dropna()
    # normalize index to dates only to avoid timestamp misalignment
    try:
        s.index = pd.to_datetime(s.index).normalize()
    except Exception:
        pass
    
    # cache the result
    CACHE[cache_key] = (now, s)
    return s

def make_figures(nasdaq, djia, threshold):
    # align both series on dates, forward-fill missing values then drop remaining NaNs
    df = pd.concat([nasdaq, djia], axis=1, join='outer')
    df.columns = ['nasdaq', 'djia']
    # forward-fill to align intra-day gaps, then drop rows where either is missing
    df = df.ffill().dropna()
    if df.empty:
        # return empty plots if no aligned data
        empty_fig = make_subplots(rows=3, cols=1, subplot_titles=("NASDAQ vs DJIA Price", "NASDAQ/DJIA Ratio", "Reversal Signal"))
        return empty_fig, pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()

    ratio = df['nasdaq'] / df['djia']
    ratio_ma = ratio.rolling(window=30, min_periods=1).mean()
    ratio_pct_change = ratio.pct_change().fillna(0) * 100
    reversal_signal = ratio_pct_change.abs() > threshold

    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=("NASDAQ vs DJIA Price", "NASDAQ/DJIA Ratio", "Reversal Signal"))

    # plot using the aligned dataframe index
    fig.add_trace(go.Scatter(x=df.index, y=df['nasdaq'].values, name="NASDAQ (AI)", line=dict(color="blue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['djia'].values, name="DJIA (Traditional)", line=dict(color="green")), row=1, col=1)

    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name="Ratio (NASDAQ/DJIA)", line=dict(color="purple")), row=2, col=1)
    fig.add_trace(go.Scatter(x=ratio_ma.index, y=ratio_ma.values, name="30-day MA", line=dict(color="orange", dash="dash")), row=2, col=1)

    # Reversal signal plot
    fig.add_trace(go.Scatter(x=ratio_pct_change.index, y=ratio_pct_change.values, name="Ratio % Change", fill='tozeroy', line=dict(color="teal")), row=3, col=1)
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", annotation_text=f"+{threshold}%", row=3, col=1)
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", annotation_text=f"-{threshold}%", row=3, col=1)

    # Vertical lines for reversals
    # add vertical lines only for non-null reversal points
    rev_idx = ratio_pct_change[reversal_signal].dropna().index
    for d in rev_idx:
        fig.add_vline(x=d, line_dash="dot", line_color="purple")

    fig.update_layout(height=900, hovermode='x unified')
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Ratio", row=2, col=1)
    fig.update_yaxes(title_text="% Change", row=3, col=1)

    return fig, ratio, ratio_pct_change, df

app.layout = html.Div([
    html.Div([
        html.H2("AI Stock Reversal Detector (Dash)"),
        html.Div("Compare NASDAQ (AI-heavy) vs DJIA and detect reversals into/out of AI"),
            html.Label("Data period"),
            dcc.Dropdown(id='period', options=[{'label':p,'value':p} for p in ['3mo','6mo','1y','2y']], value='6mo'),
            html.Label("Data interval"),
            dcc.Dropdown(id='interval', options=[{'label':v,'value':v} for v in ['1d','1h','1m']], value='1d'),
            html.Label("Reversal threshold (%)"),
            dcc.Slider(id='threshold', min=0.5, max=5.0, step=0.1, value=2.0, marks={0.5:'0.5',1:'1',2:'2',3:'3',4:'4',5:'5'}),
            html.Br(),
            dcc.Checklist(id='use_vol_adj', options=[{'label':'Use volatility-adjusted threshold','value':'yes'}], value=[]),
            html.Label("Volatility factor (multiplier)"),
            dcc.Slider(id='vol_factor', min=0.5, max=5.0, step=0.1, value=2.0, marks={0.5:'0.5',1:'1',2:'2',3:'3',4:'4',5:'5'}),
                html.Div(id='insights', style={'marginTop':'10px','fontSize':'14px'}),
            html.Div(id='validation-warning', style={'color':'crimson','marginTop':'8px','fontSize':'13px'}),
                html.Div([
                    html.B("Chart Descriptions:"),
                    html.Ul([
                        html.Li("Price: Overlay of NASDAQ (QQQ) and DJIA (DIA) showing absolute performance."),
                        html.Li("Ratio: NASDAQ/DJIA ratio with 30-day MA showing relative performance (up = rotation into AI)."),
                        html.Li("Reversal Signal: % change of the ratio; vertical lines mark changes exceeding the threshold."),
                    ])
                ], style={'marginTop':'8px','fontSize':'13px','color':'#222'}),
                html.Div(id='backtest-stats', style={'marginTop':'10px','fontSize':'13px'}),
    ], style={'width':'20%','display':'inline-block','verticalAlign':'top','padding':'10px'}),

    html.Div([
        dcc.Graph(id='main-fig')
    ], style={'width':'75%','display':'inline-block'}),

    html.Div(id='metrics', style={'width':'95%','padding':'10px'})
])

@app.callback(
    Output('main-fig', 'figure'),
    Output('metrics', 'children'),
    Output('insights', 'children'),
    Output('validation-warning', 'children'),
    Output('backtest-stats', 'children'),
    Input('period', 'value'),
    Input('interval', 'value'),
    Input('threshold', 'value'),
    Input('use_vol_adj', 'value'),
    Input('vol_factor', 'value')
)
def update(period, interval, threshold, use_vol_adj, vol_factor):
    # validate period/interval combos (minute data is limited)
    def validate_combo(period, interval):
        # yfinance minute granularity is limited to ~7 days; warn user
        if interval == '1m' and period in ['3mo', '6mo', '1y', '2y']:
            return False, '1-minute data is limited (≈7 days). Choose a shorter period or switch to 1h/1d.'
        return True, ''

    valid, validation_msg = validate_combo(period, interval)
    if not valid:
        fig = go.Figure()
        fig.update_layout(title="Invalid period/interval selection")
        # return empty metrics and insights but show validation message in sidebar
        return fig, html.Div("No data due to invalid selection"), html.Div(""), html.Div(validation_msg), html.Div("")

    nasdaq = fetch_series(TICKERS['NASDAQ'], period, interval=interval)
    djia = fetch_series(TICKERS['DJIA'], period, interval=interval)

    if nasdaq.empty or djia.empty:
        fig = go.Figure()
        fig.update_layout(title="No data available for selected period")
        return fig, html.Div("No data"), html.Div(""), html.Div(""), html.Div("")

    # compute dynamic volatility-adjusted threshold if requested
    fig, ratio, ratio_pct_change, df = make_figures(nasdaq, djia, threshold)
    dynamic_threshold = None
    threshold_used = float(threshold)
    try:
        if not ratio.empty:
            vol = ratio.pct_change().rolling(window=20, min_periods=5).std()
            if not vol.empty and not np.isnan(vol.iloc[-1]):
                dynamic_threshold = float(vol.iloc[-1]) * float(vol_factor) * 100.0
                if use_vol_adj and len(use_vol_adj) > 0 and dynamic_threshold is not None:
                    # apply the more conservative (larger) threshold
                    threshold_used = max(float(threshold), float(dynamic_threshold))
    except Exception:
        dynamic_threshold = None

    # metrics
    latest = []
    try:
        latest.append(html.Span(f"NASDAQ: {nasdaq.iloc[-1]:.2f}", style={'marginRight':'20px'}))
        latest.append(html.Span(f"DJIA: {djia.iloc[-1]:.2f}", style={'marginRight':'20px'}))
        latest.append(html.Span(f"Ratio: {ratio.iloc[-1]:.6f}", style={'marginRight':'20px'}))
        latest.append(html.Span(f"Ratio % Change: {ratio_pct_change.iloc[-1]:.2f}%", style={'marginRight':'20px'}))
    except Exception:
        latest.append(html.Span("--"))

    # Debug info: show row counts and sample index range + small df sample
    try:
        debug_lines = []
        debug_lines.append(html.Div(f"NASDAQ rows: {len(nasdaq)} | index: {nasdaq.index[0]} -> {nasdaq.index[-1]}"))
        debug_lines.append(html.Div(f"DJIA rows: {len(djia)} | index: {djia.index[0]} -> {djia.index[-1]}"))
        debug_lines.append(html.Div(f"Ratio non-nulls: {ratio.dropna().shape[0]}"))
        if dynamic_threshold is not None:
            debug_lines.append(html.Div(f"Threshold (user): {threshold}% | Vol-adjusted: {dynamic_threshold:.3f}% | Applied: {threshold_used:.3f}%"))
        else:
            debug_lines.append(html.Div(f"Threshold (user): {threshold}% | Applied: {threshold_used}%"))
        # show small sample of aligned dataframe
        try:
            sample = df.head(6).to_string()
            debug_lines.append(html.Pre(sample, style={'fontSize':'11px'}))
        except Exception:
            debug_lines.append(html.Div("(could not render df sample)"))
    except Exception:
        debug_lines = [html.Div("Debug: could not extract index info")]

    # compute backtest stats
    backtest_stats = compute_backtest_stats(ratio, ratio_pct_change, threshold_used)
    backtest_html = html.Div([
        html.B("Backtest Stats:"),
        html.Ul([
            html.Li(f"Reversals triggered: {backtest_stats['reversals']}"),
            html.Li(f"Win rate (1d fwd): {backtest_stats['win_rate']}"),
            html.Li(f"Avg forward return (1d): {backtest_stats['forward_1d']}"),
            html.Li(f"Avg forward return (5d): {backtest_stats['forward_5d']}"),
        ])
    ], style={'fontSize':'12px','marginTop':'8px','color':'#333'})

    metrics_and_debug = html.Div([
        html.Div(latest),
        html.Hr(),
        html.Div(debug_lines, style={'fontSize':'12px','color':'gray'}),
        html.Hr(),
        # (Backtest stats moved to sidebar)
        html.Hr(),
        # (Insights moved to left sidebar)
        html.Hr(),
        # (Chart descriptions moved to sidebar)
    ])

    # build insights content to return to the left sidebar
    insights_content = html.Div([html.B("Auto Insights:"), html.Ul([html.Li(i) for i in generate_insights(ratio, ratio_pct_change, threshold_used)])], style={'marginTop':'6px'})

    return fig, metrics_and_debug, insights_content, html.Div(""), backtest_html

if __name__ == '__main__':
    app.run(debug=True, port=8050)

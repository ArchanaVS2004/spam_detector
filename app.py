"""Spam message + link detector dashboard (Plotly Dash). Run: python app.py"""
import os
import re
from datetime import datetime
from urllib.parse import urlparse

import joblib
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dash_table, dcc, html, Input, Output, State, ctx, no_update

from train_model import MODEL_PATH, train

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else train()
TFIDF = model.named_steps["tfidf"]
CLF = model.named_steps["clf"]
FEATURES = TFIDF.get_feature_names_out()

# ---------- Link analysis (rule-based) ----------
URL_RE = re.compile(
    r"(https?://[^\s]+|www\.[^\s]+|\b[a-z0-9-]+\.(?:com|net|org|xyz|top|click|info|ru|tk|ml|cf|gq|buzz|live)\b[^\s]*)",
    re.I)
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly",
              "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc"}
BAD_TLDS = {"xyz", "top", "click", "tk", "ml", "cf", "gq", "buzz", "live", "info", "ru"}
KEYWORDS = ["login", "verify", "secure", "account", "update", "bank", "password",
            "confirm", "free", "prize", "claim", "refund", "billing", "suspend", "wallet"]

def extract_urls(text):
    return [u.rstrip(".,;:!?)\"'") for u in URL_RE.findall(text)]

def analyze_link(url):
    score, reasons = 0, []
    full = url if re.match(r"https?://", url, re.I) else "http://" + url
    p = urlparse(full)
    host = (p.hostname or "").lower()
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        score += 3; reasons.append("uses a raw IP address")
    if host in SHORTENERS:
        score += 2; reasons.append("URL shortener hides real destination")
    if tld in BAD_TLDS:
        score += 2; reasons.append(f"suspicious domain ending .{tld}")
    if full.lower().startswith("http://"):
        score += 1; reasons.append("not HTTPS")
    if "@" in p.netloc:
        score += 3; reasons.append("contains '@' in address")
    if "xn--" in host:
        score += 3; reasons.append("punycode (look-alike characters)")
    if host.count("-") >= 2:
        score += 1; reasons.append("many hyphens in domain")
    if host.count(".") >= 3:
        score += 1; reasons.append("many subdomains")
    hits = [k for k in KEYWORDS if k in (host + p.path).lower()]
    if hits:
        score += min(2, len(hits)); reasons.append("keywords: " + ", ".join(hits[:3]))
    if len(full) > 75:
        score += 1; reasons.append("very long URL")
    label = "Dangerous" if score >= 4 else "Suspicious" if score >= 2 else "Looks OK"
    return score, label, reasons

def top_spam_words(text, n=6):
    """Words/phrases that pushed the model towards 'spam'."""
    row = TFIDF.transform([text]).tocoo()
    contrib = sorted(((FEATURES[j], v * CLF.coef_[0][j]) for j, v in zip(row.col, row.data)),
                     key=lambda x: x[1], reverse=True)
    return [w for w, c in contrib if c > 0][:n]

# ---------- Figures ----------
RED, GREEN, AMBER, BLUE = "#dc2626", "#16a34a", "#f59e0b", "#2563eb"

def gauge_fig(value=0):
    color = GREEN if value < 40 else AMBER if value < 60 else RED
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "Spam Risk Score", "font": {"size": 18}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color, "thickness": 0.3},
               "steps": [{"range": [0, 40], "color": "#dcfce7"},
                         {"range": [40, 60], "color": "#fef3c7"},
                         {"range": [60, 100], "color": "#fee2e2"}],
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.8, "value": 50}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def pie_fig(history):
    spam = sum(h["Verdict"] == "SPAM" for h in history)
    ham = len(history) - spam
    fig = go.Figure(go.Pie(labels=["Spam", "Not spam"], values=[spam, ham], hole=0.55,
                           marker=dict(colors=[RED, GREEN]),
                           textinfo="label+value" if history else "none"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10),
                      title="Spam vs Not Spam",
                      annotations=[] if history else [dict(text="No data yet", showarrow=False)])
    return fig

def timeline_fig(history):
    fig = go.Figure()
    if history:
        x = list(range(1, len(history) + 1))
        y = [h["Risk"] for h in history]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color=BLUE),
                                 marker=dict(size=10, color=[RED if v >= 50 else GREEN for v in y]),
                                 hovertext=[h["Message"] for h in history]))
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
    fig.update_layout(height=280, margin=dict(l=40, r=10, t=40, b=40),
                      title="Risk score per message", xaxis_title="Message #",
                      yaxis=dict(title="Risk %", range=[0, 100]))
    return fig

def kpi(title, value, color):
    return html.Div(style={"flex": "1", "background": "white", "borderLeft": f"6px solid {color}",
                           "padding": "12px 16px", "borderRadius": "10px",
                           "boxShadow": "0 1px 4px rgba(0,0,0,.15)"},
                    children=[html.Div(title, style={"color": "#6b7280", "fontSize": "13px"}),
                              html.Div(value, style={"fontSize": "28px", "fontWeight": "bold"})])

def kpi_row(history):
    total = len(history)
    spam = sum(h["Verdict"] == "SPAM" for h in history)
    links = sum(h["Links"] for h in history)
    rate = f"{spam / total * 100:.0f}%" if total else "0%"
    return [kpi("Messages scanned", total, BLUE), kpi("Spam detected", spam, RED),
            kpi("Spam rate", rate, AMBER), kpi("Links found", links, "#7c3aed")]

# ---------- Layout ----------
CARD = {"background": "white", "padding": "16px", "borderRadius": "10px",
        "boxShadow": "0 1px 4px rgba(0,0,0,.15)"}
BTN = {"padding": "8px 18px", "border": "none", "borderRadius": "6px",
       "cursor": "pointer", "color": "white", "marginRight": "8px"}
EXAMPLES = {
    "ex-spam": "WINNER!! Claim your free $1000 gift card now at http://bit.ly/claim-prize-now",
    "ex-phish": "URGENT: Your bank account is locked. Verify password at http://secure-login-bank.xyz/verify",
    "ex-ham": "Hey, are we still meeting for lunch tomorrow at 1? Let me know.",
}

app = Dash(__name__)
server = app.server  # exposed for gunicorn in production
app.title = "Spam Detector"

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "auto", "padding": "20px",
           "fontFamily": "Segoe UI, Arial", "background": "#f2f4f8"},
    children=[
        html.H2("🛡️ Spam Message & Link Detector"),
        html.Div(id="kpis", style={"display": "flex", "gap": "12px", "marginBottom": "16px"},
                 children=kpi_row([])),

        html.Div(style={**CARD, "marginBottom": "16px"}, children=[
            dcc.Textarea(id="msg", placeholder="Paste a message here...",
                         style={"width": "100%", "height": "110px", "padding": "8px"}),
            html.Div(style={"marginTop": "10px"}, children=[
                html.Button("Analyze", id="btn", n_clicks=0, style={**BTN, "background": BLUE}),
                html.Button("Clear history", id="clear", n_clicks=0, style={**BTN, "background": "#6b7280"}),
                html.Button("Download CSV", id="dl-btn", n_clicks=0, style={**BTN, "background": "#0f766e"}),
                html.Span("Try an example: ", style={"marginLeft": "12px", "color": "#6b7280"}),
                html.Button("Spam", id="ex-spam", n_clicks=0, style={**BTN, "background": RED, "padding": "4px 12px"}),
                html.Button("Phishing", id="ex-phish", n_clicks=0, style={**BTN, "background": AMBER, "padding": "4px 12px"}),
                html.Button("Normal", id="ex-ham", n_clicks=0, style={**BTN, "background": GREEN, "padding": "4px 12px"}),
            ]),
        ]),

        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px"}, children=[
            html.Div(style={**CARD, "flex": "1"}, children=[
                dcc.Graph(id="gauge", figure=gauge_fig(0), config={"displayModeBar": False})]),
            html.Div(style={**CARD, "flex": "1.2"}, id="result",
                     children=html.Div("Analyze a message to see details.", style={"color": "#6b7280"})),
        ]),

        html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "16px"}, children=[
            html.Div(style={**CARD, "flex": "1"}, children=[dcc.Graph(id="pie", figure=pie_fig([]))]),
            html.Div(style={**CARD, "flex": "1.5"}, children=[dcc.Graph(id="timeline", figure=timeline_fig([]))]),
        ]),

        html.Div(style=CARD, children=[
            html.H4("History"),
            dash_table.DataTable(
                id="table",
                columns=[{"name": c, "id": c} for c in ["Time", "Message", "Verdict", "Risk", "Links"]],
                data=[], page_size=8,
                style_cell={"textAlign": "left", "maxWidth": "400px",
                            "overflow": "hidden", "textOverflow": "ellipsis"},
                style_header={"fontWeight": "bold"},
                style_data_conditional=[
                    {"if": {"filter_query": '{Verdict} = "SPAM"'}, "backgroundColor": "#fee2e2"},
                    {"if": {"filter_query": '{Verdict} = "NOT SPAM"'}, "backgroundColor": "#dcfce7"}]),
        ]),
        dcc.Store(id="history", data=[]),
        dcc.Download(id="download"),
    ])

# ---------- Callbacks ----------
@app.callback(Output("msg", "value"),
              [Input(k, "n_clicks") for k in EXAMPLES], prevent_initial_call=True)
def fill_example(*_):
    return EXAMPLES[ctx.triggered_id]

@app.callback(
    Output("result", "children"), Output("gauge", "figure"), Output("kpis", "children"),
    Output("history", "data"), Output("table", "data"),
    Output("pie", "figure"), Output("timeline", "figure"),
    Input("btn", "n_clicks"), Input("clear", "n_clicks"),
    State("msg", "value"), State("history", "data"), prevent_initial_call=True)
def analyze(_, __, text, history):
    history = history or []

    if ctx.triggered_id == "clear":
        history = []
        return (html.Div("History cleared.", style={"color": "#6b7280"}), gauge_fig(0),
                kpi_row(history), history, [], pie_fig(history), timeline_fig(history))

    if not text or not text.strip():
        return (html.Div("Please enter a message.", style={"color": RED}), no_update, no_update,
                no_update, no_update, no_update, no_update)

    prob = float(model.predict_proba([text])[0][1]) * 100
    links = [(u, *analyze_link(u)) for u in extract_urls(text)]
    link_risk = min(100, max([s for _, s, _, _ in links], default=0) * 15)
    risk = round(max(prob, link_risk), 1)
    is_spam = risk >= 50

    words = top_spam_words(text)
    chips = [html.Span(w, style={"background": "#fee2e2", "color": RED, "padding": "2px 10px",
                                 "borderRadius": "12px", "marginRight": "6px", "fontSize": "13px"})
             for w in words] or [html.Span("none", style={"color": "#6b7280"})]

    link_blocks = [html.Div(style={"marginTop": "8px", "padding": "8px",
                                   "border": "1px solid #ddd", "borderRadius": "6px"},
                            children=[html.B(u, style={"wordBreak": "break-all"}),
                                      html.Span(f"  →  {lab} (score {sc})",
                                                style={"color": RED if lab == "Dangerous" else AMBER if lab == "Suspicious" else GREEN}),
                                      html.Ul([html.Li(r) for r in rs], style={"margin": "4px 0"})])
                   for u, sc, lab, rs in links] or [html.Div("No links found.", style={"color": "#6b7280"})]

    result = [
        html.H2("🚨 SPAM" if is_spam else "✅ NOT SPAM",
                style={"color": RED if is_spam else GREEN, "margin": "0 0 8px"}),
        html.Div([html.B("Text model probability: "), f"{prob:.1f}%"]),
        html.Div([html.B("Link risk: "), f"{link_risk:.0f}%"]),
        html.Div([html.B("Suspicious words: "), *chips], style={"marginTop": "6px"}),
        html.H4("Link analysis", style={"marginBottom": "0"}), *link_blocks]

    history.append({"Time": datetime.now().strftime("%H:%M:%S"), "Message": text[:120],
                    "Verdict": "SPAM" if is_spam else "NOT SPAM", "Risk": risk, "Links": len(links)})
    return (result, gauge_fig(risk), kpi_row(history), history, history[::-1],
            pie_fig(history), timeline_fig(history))

@app.callback(Output("download", "data"), Input("dl-btn", "n_clicks"),
              State("history", "data"), prevent_initial_call=True)
def download(_, history):
    if not history:
        return no_update
    return dcc.send_data_frame(pd.DataFrame(history).to_csv, "spam_history.csv", index=False)

if __name__ == "__main__":
    # Local: debug on. Hosted: set PORT env var and DEBUG=0
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)),
            debug=os.environ.get("DEBUG", "1") == "1")

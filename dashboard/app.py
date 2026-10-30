
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# SKYTRACK_ELECTRIC_BLUE_PLOTLY_TEMPLATE_V1
try:
    import plotly.io as pio
    import plotly.graph_objects as go

    pio.templates["skytrack_blue"] = go.layout.Template(
        layout=dict(
            paper_bgcolor="#020408",
            plot_bgcolor="#071321",
            font=dict(
                color="#D9F3FF",
                family="Arial, sans-serif"
            ),
            colorway=[
                "#168BFF",
                "#42D9FF",
                "#1264C5",
                "#A6E8FF",
                "#3478C5",
                "#D9F3FF"
            ],
            xaxis=dict(
                color="#A6DFFF",
                gridcolor="rgba(66,217,255,0.13)",
                zerolinecolor="rgba(22,139,255,0.35)",
                linecolor="rgba(22,139,255,0.35)"
            ),
            yaxis=dict(
                color="#A6DFFF",
                gridcolor="rgba(66,217,255,0.13)",
                zerolinecolor="rgba(22,139,255,0.35)",
                linecolor="rgba(22,139,255,0.35)"
            ),
            legend=dict(font=dict(color="#D9F3FF")),
            margin=dict(l=45, r=25, t=45, b=40)
        )
    )
    pio.templates.default = "skytrack_blue"
except Exception as exc:
    print("Plotly theme note:", exc)


PROJECT = Path("/content/drive/MyDrive/FSOC_SkyTrack_AI")
ENGINE = PROJECT / "ai_agent" / "copilot_engine.py"
VIDEO = PROJECT / "results/videos/validated_c18_airborne_demo_h264.mp4"
CINEMATIC_VIDEO = PROJECT / "results/videos/FSOC_SkyTrack_Cinematic_Demo_75sec.mp4"
MODEL = PROJECT / "simulation/models/beacon_detector_best.pt"
ARCH = PROJECT / "results/figures/original_system_architecture.png"
REPORT = PROJECT / "results/reports/evidence_diagnostics.json"

DATASETS = {
    "C18 Flight Run": PROJECT / "experiments/baselines/validated_c18_metrics.csv",
    "YOLO Centroid Benchmark": PROJECT / "experiments/baselines/yolo_centroid_benchmark.csv",
    "Trajectory Validation": PROJECT / "experiments/baselines/multi_trajectory_validation.csv",
}

st.set_page_config(
    page_title="SKYTRACK // FSOC",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DESIGN SYSTEM ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #06090e;
    --panel: #0b121b;
    --panel2: #101b28;
    --line: #203344;
    --cyan: #65e7f2;
    --cyan2: #27b8d1;
    --text: #eaf3fb;
    --muted: #8195a8;
    --green: #42D9FF;
    --amber: #ffc979;
}

.stApp {
    background:
        radial-gradient(ellipse at 78% 0%, rgba(20,78,103,.18), transparent 38%),
        var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(6,9,14,.92);
}
[data-testid="stSidebar"] {
    background: #080e16;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}
.block-container {
    max-width: 1600px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: var(--text) !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: .5px;
}
h1 { font-size: 2.8rem !important; font-weight: 800 !important; }
h2 { font-size: 1.8rem !important; }
h3 { font-size: 1.35rem !important; }

p, label, .stMarkdown {
    color: var(--text);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #101c29, #0a111a);
    border: 1px solid #263b4c;
    border-radius: 9px;
    padding: 15px 17px;
    min-height: 110px;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .75rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}
[data-testid="stMetricValue"] {
    color: var(--cyan) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 8px;
}
.stButton > button {
    background: linear-gradient(90deg, #123747, #125065);
    color: #e7fcff;
    border: 1px solid #27778b;
    border-radius: 5px;
    font-weight: 600;
    min-height: 42px;
}
.stButton > button:hover {
    background: #17677a;
    border-color: var(--cyan);
    color: white;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    background: #101923;
    border-color: #2a4052;
}
hr {
    border-color: var(--line);
}
code, pre {
    font-family: 'IBM Plex Mono', monospace !important;
}
section[data-testid="stSidebar"] .stRadio label {
    font-size: .92rem;
}
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}
.sky-eyebrow {
    color: var(--cyan);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .72rem;
    letter-spacing: 2.3px;
    text-transform: uppercase;
}
.sky-title {
    color: #f3f8ff;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: clamp(2.7rem, 5vw, 4.7rem);
    font-weight: 800;
    line-height: .96;
    letter-spacing: 1px;
    margin: .35rem 0 .7rem 0;
}
.sky-subtitle {
    color: #a4b8c9;
    font-size: .94rem;
    line-height: 1.65;
}
.sky-panel {
    background: linear-gradient(140deg, rgba(15,28,41,.96), rgba(8,14,22,.98));
    border: 1px solid #243a4c;
    border-radius: 9px;
    padding: 17px 19px;
}
.sky-panel-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    letter-spacing: 1.5px;
    color: #8da7b9;
    text-transform: uppercase;
    margin-bottom: 9px;
}
.sky-status {
    display: inline-block;
    color: var(--green);
    border: 1px solid #285849;
    background: #0b251f;
    border-radius: 3px;
    padding: 4px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .69rem;
    letter-spacing: .7px;
}
.sky-warn {
    display: inline-block;
    color: var(--amber);
    border: 1px solid #5a4627;
    background: #261d12;
    border-radius: 3px;
    padding: 4px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: .69rem;
}
.sky-value {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--cyan);
    font-size: 1.35rem;
    font-weight: 600;
}
.sky-label {
    font-size: .71rem;
    color: #8197a9;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.sky-divider {
    height: 1px;
    background: linear-gradient(90deg, #28556a, #172431, transparent);
    margin: 18px 0;
}
div[data-testid="stVideo"] {
    border: 1px solid #29495d;
    border-radius: 9px;
    overflow: hidden;
}

    /* SKYTRACK_CINEMATIC_UPGRADE_V1 */

    :root {
        --bg: #070a10;
        --panel: #101722;
        --panel2: #151f2d;
        --line: #26394d;
        --cyan: #38dff5;
        --cyan2: #119fbd;
        --text: #edf6ff;
        --muted: #91a5b8;
        --green: #48edac;
        --amber: #ffbd59;
        --red: #ff6175;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 75% 0%,
            rgba(13, 91, 130, .18), transparent 38%),
          linear-gradient(135deg, #070a10, #090f18 65%, #070b12);
        color: var(--text);
    }

    .block-container {
        max-width: 1750px !important;
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 10, 16, .92);
    }

    [data-testid="stSidebar"] {
        background: #090f18;
        border-right: 1px solid #23384b;
    }

    [data-testid="stSidebar"] > div {
        padding-top: 1.2rem;
    }

    h1, h2, h3 {
        color: #edf7ff !important;
        letter-spacing: .4px;
    }

    h1 {
        font-size: 2.5rem !important;
    }

    h2 {
        font-size: 1.75rem !important;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #151f2d, #0d141e);
        border: 1px solid #2a4055;
        border-radius: 11px;
        padding: 15px 17px;
        min-height: 105px;
        box-shadow: 0 8px 24px rgba(0,0,0,.18);
    }

    [data-testid="stMetricLabel"] {
        color: #94aabd !important;
        letter-spacing: 1.1px;
        font-size: .72rem !important;
    }

    [data-testid="stMetricValue"] {
        color: #48dff3 !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    [data-testid="stVideo"] {
        border: 1px solid #31566e;
        border-radius: 13px;
        overflow: hidden;
        background: #05090e;
        box-shadow: 0 0 28px rgba(25, 181, 220, .09);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #263d52;
        border-radius: 10px;
    }

    .stButton > button {
        background: linear-gradient(110deg, #103448, #12536b);
        color: #e9fbff;
        border: 1px solid #26758d;
        border-radius: 7px;
        min-height: 42px;
        font-weight: 650;
        transition: all .2s ease;
    }

    .stButton > button:hover {
        background: #176b84;
        border-color: #45e4f5;
        color: white;
        box-shadow: 0 0 16px rgba(56,223,245,.15);
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background: #111b28;
        border-color: #30465b;
    }

    [data-testid="stExpander"] {
        background: #0e1722;
        border: 1px solid #263b50;
        border-radius: 9px;
    }

    .skytrack-command-header {
        position: relative;
        overflow: hidden;
        padding: 22px 25px;
        margin: 5px 0 19px 0;
        border: 1px solid #29495e;
        border-radius: 13px;
        background:
          radial-gradient(ellipse at 88% 40%,
            rgba(26, 178, 214, .17), transparent 35%),
          linear-gradient(115deg, #101b29, #0b111b 75%);
        box-shadow: inset 0 0 35px rgba(36, 172, 215, .035);
    }

    .skytrack-command-header:after {
        content: "";
        position: absolute;
        width: 210px;
        height: 210px;
        right: 4%;
        top: -120px;
        border: 1px solid rgba(56,223,245,.14);
        border-radius: 50%;
        box-shadow:
          0 0 0 22px rgba(56,223,245,.025),
          0 0 0 48px rgba(56,223,245,.02);
        pointer-events: none;
    }

    .skytrack-command-kicker {
        color: #48dff3;
        font: 10px 'IBM Plex Mono', monospace;
        letter-spacing: 2.5px;
        text-transform: uppercase;
    }

    .skytrack-command-title {
        color: #f0f8ff;
        font: 800 clamp(27px, 3.4vw, 43px)
             'Barlow Condensed', sans-serif;
        letter-spacing: 1.5px;
        margin-top: 7px;
        line-height: 1.05;
    }

    .skytrack-command-subtitle {
        color: #91aabd;
        font-size: 12px;
        margin-top: 8px;
    }

    .skytrack-command-status {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 15px;
    }

    .skytrack-chip {
        border: 1px solid #2d5269;
        border-radius: 5px;
        background: rgba(5, 17, 27, .75);
        padding: 6px 9px;
        color: #b8d9e8;
        font: 9px 'IBM Plex Mono', monospace;
        letter-spacing: .8px;
    }

    .skytrack-chip-green {
        color: #5af0b0;
        border-color: #27664f;
    }

    .skytrack-hud {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 9px;
        margin: 8px 0 14px 0;
    }

    .skytrack-hud-card {
        padding: 12px 13px;
        border: 1px solid #294257;
        border-radius: 8px;
        background: linear-gradient(145deg, #131f2c, #0b121c);
    }

    .skytrack-hud-label {
        color: #8fa8bc;
        font: 9px 'IBM Plex Mono', monospace;
        letter-spacing: 1px;
    }

    .skytrack-hud-value {
        color: #4ce1f3;
        font: 600 19px 'IBM Plex Mono', monospace;
        margin-top: 7px;
    }

    .skytrack-hud-note {
        color: #7d93a7;
        font-size: 9px;
        margin-top: 4px;
    }

    @media(max-width: 750px) {
        .skytrack-hud {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    /* END SKYTRACK_CINEMATIC_UPGRADE_V1 */
    
</style>
""", unsafe_allow_html=True)


# SKYTRACK_GLOBAL_AEROSPACE_THEME_V1
# Shared visual language for all six website pages.

st.markdown("""
<style>
:root {
    --sky-bg: #030a10;
    --sky-panel: #091720;
    --sky-panel-2: #0c2028;
    --sky-border: rgba(67, 220, 153, .22);
    --sky-green: #38e99a;
    --sky-cyan: #35ddf2;
    --sky-text: #d8eee6;
    --sky-muted: #83a69a;
}

/* Main website */
.stApp,
[data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse at 70% 0%,
      rgba(0, 83, 67, .12), transparent 45%),
      linear-gradient(145deg, #030a10, #06121a 65%, #030a10);
    color: var(--sky-text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #050d14;
    border-right: 1px solid var(--sky-border);
}

[data-testid="stSidebar"] * {
    color: var(--sky-text);
}

/* Typography */
h1, h2, h3, h4 {
    color: #e1f5ed !important;
    letter-spacing: .3px;
}

p, label, .stMarkdown {
    color: var(--sky-text);
}

/* Containers and cards */
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stMetric"],
[data-testid="stExpander"] {
    background: linear-gradient(145deg,
      rgba(12, 30, 39, .92),
      rgba(5, 16, 23, .96));
    border: 1px solid var(--sky-border);
    border-radius: 12px;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--sky-green) !important;
    font-family: monospace;
}

[data-testid="stMetricLabel"] {
    color: var(--sky-muted) !important;
    letter-spacing: 1px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(110deg, #0b3329, #09221f);
    color: #8fffc5 !important;
    border: 1px solid rgba(56, 233, 154, .45);
    border-radius: 8px;
    transition: all .2s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #124d39;
    border-color: var(--sky-green);
    box-shadow: 0 0 18px rgba(56, 233, 154, .15);
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background: #07151c !important;
    color: #d8eee6 !important;
    border: 1px solid var(--sky-border) !important;
    border-radius: 8px;
}

[data-baseweb="select"] > div {
    background: #07151c !important;
    border-color: var(--sky-border) !important;
    color: var(--sky-text) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--sky-muted) !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--sky-green) !important;
    border-bottom-color: var(--sky-green) !important;
}

/* Tables and dataframes */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--sky-border);
    border-radius: 10px;
    overflow: hidden;
}

/* Alerts */
[data-testid="stAlert"] {
    background: #0a1c20;
    border: 1px solid var(--sky-border);
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: rgba(67, 220, 153, .2) !important;
}

/* Chat interface */
[data-testid="stChatMessage"] {
    background: rgba(8, 25, 31, .8);
    border: 1px solid rgba(67, 220, 153, .16);
    border-radius: 12px;
}

/* Video */
video {
    border-radius: 12px;
    border: 1px solid var(--sky-border);
}

/* Hide unnecessary decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)




def read_csv(path):
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def read_report():
    if not REPORT.is_file():
        return {}
    try:
        return json.loads(REPORT.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_card(label, value, note=""):
    st.markdown(
        f"""
        <div class="sky-panel">
          <div class="sky-panel-title">{label}</div>
          <div class="sky-value">{value}</div>
          <div style="color:#8195a8;font-size:.72rem;margin-top:7px">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_theme(fig, height=340):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(7,13,20,0.5)",
        font=dict(family="IBM Plex Mono, monospace", color="#a9bdcd", size=11),
        title_font=dict(family="Barlow Condensed, sans-serif", size=20, color="#edf7ff"),
        margin=dict(l=15, r=15, t=55, b=20),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=dict(gridcolor="#1b2b39", zerolinecolor="#294252"),
        yaxis=dict(gridcolor="#1b2b39", zerolinecolor="#294252"),
    )
    return fig


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding:4px 0 18px">
          <div style="font-family:'IBM Plex Mono';font-size:.67rem;color:#65e7f2;letter-spacing:2px">
          AEROSPACE SYSTEMS / 01
          </div>
          <div style="font-family:'Barlow Condensed';font-size:2rem;font-weight:800;color:#f2f8ff;line-height:1.05;margin-top:8px">
          SKYTRACK<span style="color:#65e7f2"> AI</span>
          </div>
          <div style="color:#8195a8;font-size:.76rem;margin-top:7px">
          Airborne optical tracking lab
          </div>
        </div>
        <div class="sky-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "MISSION CONSOLE",
        [
            "Mission Overview",
            "Optical Tracking Lab",
            "Flight Analytics",
            "AI Mission Engineer",
            "System Architecture",
            "Asset Registry",
        ],
        label_visibility="visible",
    )

    st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sky-panel-title">SYSTEM READINESS</div>
        <div style="margin:12px 0;color:#d5e2ee;font-size:.82rem">
          <span style="color:#42D9FF">●</span> Local dashboard operational
        </div>
        <div style="margin:12px 0;color:#d5e2ee;font-size:.82rem">
          <span style="color:#42D9FF">●</span> Evidence files imported
        </div>
        <div style="margin:12px 0;color:#d5e2ee;font-size:.82rem">
          <span style="color:#168BFF">●</span> Nemotron AI connected
        </div>
        <div style="margin-top:22px;color:#718699;font-size:.72rem;line-height:1.7">
          RECORDED TEST DATA<br>
          No live aircraft feed connected.<br>
          Beacon tracking is simulated.
        </div>
        """,
        unsafe_allow_html=True,
    )

report = read_report()
metrics = report.get("metrics", {})
centroid = metrics.get("centroid_error_px", {})
latency = metrics.get("inference_latency_ms", {})
confidence = metrics.get("confidence", {})

c18 = read_csv(DATASETS["C18 Flight Run"])
trajectory = read_csv(DATASETS["Trajectory Validation"])

# ---------------- MISSION OVERVIEW ----------------

# SKYTRACK_MULTIPAGE_AEROSPACE_UPGRADE_V1

def render_skytrack_page_hero(page_code, title, subtitle, status="RECORDED SYSTEM"):
    """Shared futuristic aerospace header for secondary workspaces."""
    st.markdown("""
    <style>
    .stApp {
        background:
          radial-gradient(ellipse at 80% 0%, rgba(0,105,110,.12), transparent 42%),
          linear-gradient(145deg,#03090f,#07131b 58%,#03090f) !important;
        color: #d9edf5;
    }

    [data-testid="stHeader"] {
        background: rgba(3,9,15,.88);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#07111a,#030a10);
        border-right: 1px solid rgba(54,210,210,.20);
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg,#0d1b27,#07121b);
        border: 1px solid rgba(65,190,210,.22);
        border-radius: 12px;
        padding: 15px 17px;
    }

    [data-testid="stPlotlyChart"] {
        background: linear-gradient(145deg,rgba(9,24,33,.9),rgba(4,13,20,.95));
        border: 1px solid rgba(54,190,205,.20);
        border-radius: 12px;
        padding: 7px;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid rgba(54,190,205,.20);
        border-radius: 10px;
        overflow: hidden;
    }

    [data-testid="stExpander"] {
        background: rgba(8,23,32,.78);
        border: 1px solid rgba(54,190,205,.18);
        border-radius: 10px;
    }

    .stButton > button,
    .stDownloadButton > button {
        border: 1px solid rgba(53,218,225,.55);
        background: linear-gradient(110deg,#0b2933,#10212d);
        color: #dffcff;
        border-radius: 7px;
        transition: .2s ease;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #48e6ed;
        color: #ffffff;
        box-shadow: 0 0 18px rgba(45,210,220,.18);
    }

    .sky-page-hero {
        position: relative;
        overflow: hidden;
        padding: 25px 28px 23px 28px;
        margin: 3px 0 25px 0;
        border: 1px solid rgba(60,190,210,.30);
        border-left: 3px solid #36dce5;
        border-radius: 13px;
        background:
          radial-gradient(circle at 90% 0%,rgba(20,160,170,.19),transparent 28%),
          linear-gradient(115deg,#0c1b28,#07121c 68%,#081922);
        box-shadow: 0 15px 45px rgba(0,0,0,.24);
    }

    .sky-page-hero:after {
        content: "";
        position: absolute;
        width: 190px;
        height: 190px;
        right: 7%;
        top: -130px;
        border: 1px solid rgba(57,215,224,.23);
        border-radius: 50%;
        box-shadow: 0 0 0 18px rgba(57,215,224,.025),
                    0 0 0 38px rgba(57,215,224,.02);
        pointer-events: none;
    }

    .sky-page-kicker {
        font: 10px monospace;
        letter-spacing: 3px;
        color: #4ce2e8;
        margin-bottom: 10px;
    }

    .sky-page-title {
        font-size: clamp(25px,3vw,37px);
        line-height: 1.1;
        font-weight: 850;
        letter-spacing: .5px;
        color: #e9f5ff;
        margin-bottom: 9px;
    }

    .sky-page-title span {
        color: #40dce7;
    }

    .sky-page-subtitle {
        color: #9ab2c2;
        font-size: 12px;
        line-height: 1.7;
        max-width: 760px;
    }

    .sky-page-status {
        display: inline-block;
        margin-top: 15px;
        margin-right: 7px;
        padding: 5px 9px;
        border: 1px solid rgba(65,210,190,.30);
        border-radius: 5px;
        color: #75edc0;
        font: 9px monospace;
        letter-spacing: 1px;
        background: rgba(15,90,73,.15);
    }

    .sky-page-divider {
        height: 1px;
        margin: 20px 0;
        background: linear-gradient(90deg,rgba(58,215,220,.45),rgba(58,215,220,0));
    }

    .sky-section-label {
        color: #43dce4;
        font: 10px monospace;
        letter-spacing: 2.5px;
        margin: 12px 0 8px 0;
    }

    h2, h3 {
        color: #e2f2fa !important;
    }

    p, label, [data-testid="stCaptionContainer"] {
        color: #9db4c3;
    }

    @media(max-width:750px) {
        .sky-page-hero {padding:19px;}
        .sky-page-title {font-size:25px;}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sky-page-hero">
      <div class="sky-page-kicker">
        FSOC / SKYTRACK AI / {page_code}
      </div>
      <div class="sky-page-title">
        {title}
      </div>
      <div class="sky-page-subtitle">
        {subtitle}
      </div>
      <span class="sky-page-status">● {status}</span>
      <span class="sky-page-status">VIRTUAL CAMERA / RESEARCH MODE</span>
    </div>
    """, unsafe_allow_html=True)



# SKYTRACK_GLOBAL_GREEN_AEROSPACE_THEME_V1
st.markdown("""
<style>
:root {
    --sky-green: #168BFF;
    --sky-green-bright: #42D9FF;
    --sky-green-dim: #1264C5;
    --sky-green-border: rgba(22,139,255,.25);
    --sky-bg: #030a09;
    --sky-panel: #0A1929;
    --sky-panel-2: #0a1c15;
}

/* GLOBAL BACKGROUND */
.stApp {
    background:
      radial-gradient(ellipse at 70% 0%,rgba(0,120,65,.13),transparent 42%),
      linear-gradient(145deg,#020706,#06120c 58%,#020706) !important;
    color: #e5f7ed !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#07130e,#020806) !important;
    border-right: 1px solid rgba(22,139,255,.22) !important;
}

[data-testid="stSidebar"] * {
    --primary-color: #168BFF;
}

[data-testid="stSidebar"] [aria-selected="true"] {
    color: #42D9FF !important;
}

/* PAGE HEADERS */
.sky-page-hero,
.skytrack-command-header {
    background:
      radial-gradient(circle at 90% 0%,rgba(30,180,100,.17),transparent 30%),
      linear-gradient(115deg,#0b1b12,#07130d 70%,#081a11) !important;
    border-color: rgba(22,139,255,.34) !important;
    border-left-color: #168BFF !important;
}

.sky-page-kicker,
.sky-page-title span,
.skytrack-command-kicker,
.skytrack-command-title span {
    color: #168BFF !important;
}

/* COMMAND CENTRE PANELS */
.sky-command,
.sky-glass,
.sky-radar-panel,
.sky-telemetry,
.sky-page-hero {
    border-color: rgba(22,139,255,.25) !important;
}

.sky-command {
    background:
      radial-gradient(ellipse at 50% 45%,rgba(0,120,65,.18),transparent 55%),
      linear-gradient(145deg,#020a06,#071810 65%,#020a06) !important;
}

.sky-glass,
.sky-telemetry {
    background: linear-gradient(145deg,#0a1c13,#04100a) !important;
}

/* RADAR / TRACKING */
.sky-radar-panel {
    background:
      radial-gradient(circle,rgba(0,120,65,.12),transparent 68%),
      #020b07 !important;
    border-color: rgba(22,139,255,.38) !important;
}

/* SECTION LABELS AND DIVIDERS */
.sky-section-label,
.sky-eyebrow,
.sky-page-status,
.skytrack-chip-green {
    color: #53f5a9 !important;
}

.sky-divider,
.sky-page-divider {
    background: linear-gradient(
      90deg,rgba(22,139,255,.55),rgba(22,139,255,0)
    ) !important;
}

/* METRICS */
[data-testid="stMetric"] {
    background: linear-gradient(145deg,#0b1c13,#061009) !important;
    border: 1px solid rgba(22,139,255,.27) !important;
    border-radius: 12px !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #e6fff0 !important;
}

[data-testid="stMetricDelta"] {
    color: #55efa5 !important;
}

/* INPUTS */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div {
    background: #08150e !important;
    color: #e5f7ed !important;
    border-color: rgba(22,139,255,.30) !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #168BFF !important;
    box-shadow: 0 0 0 1px #168BFF !important;
}

/* BUTTONS */
.stButton > button,
.stDownloadButton > button,
button[kind="primary"] {
    background: linear-gradient(110deg,#0c3522,#0b2115) !important;
    border: 1px solid rgba(22,139,255,.58) !important;
    color: #dffff0 !important;
    border-radius: 7px !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    border-color: #42D9FF !important;
    box-shadow: 0 0 20px rgba(22,139,255,.22) !important;
    color: #ffffff !important;
}

/* DATA TABLES */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid rgba(22,139,255,.23) !important;
    border-radius: 10px !important;
}

/* EXPANDERS */
[data-testid="stExpander"] {
    background: rgba(7,24,14,.85) !important;
    border: 1px solid rgba(22,139,255,.20) !important;
    border-radius: 10px !important;
}

/* LINKS AND ACCENTS */
a {
    color: #55efa5 !important;
}

a:hover {
    color: #9affc9 !important;
}

/* STATUS / SUCCESS */
[data-testid="stAlert"] {
    border-color: rgba(22,139,255,.32) !important;
}

/* PLOTLY CONTAINER */
[data-testid="stPlotlyChart"] {
    background: linear-gradient(145deg,#08170f,#030b07) !important;
    border: 1px solid rgba(22,139,255,.23) !important;
    border-radius: 12px !important;
    padding: 7px !important;
}

/* FOOTER / CAPTIONS */
[data-testid="stCaptionContainer"],
small {
    color: #91ad9b;
}
</style>
""", unsafe_allow_html=True)

if page == "Mission Overview":

    # SKYTRACK_CINEMATIC_UPGRADE_V1 — MISSION COMMAND HEADER
    st.markdown("""
    <div class="skytrack-command-header">
      <div class="skytrack-command-kicker">
        FSOC / AIRBORNE OPTICAL SYSTEMS / RESEARCH CONSOLE
      </div>
      <div class="skytrack-command-title">
        SKYTRACK <span style="color:#39d9f0">AI</span>
        // MISSION CONTROL
      </div>
      <div class="skytrack-command-subtitle">
        AI-assisted virtual camera tracking for mobile
        free-space optical communication terminals
      </div>
      <div class="skytrack-command-status">
        <span class="skytrack-chip skytrack-chip-green">
          ● SYSTEM READY
        </span>
        <span class="skytrack-chip">MODE / RECORDED SIMULATION</span>
        <span class="skytrack-chip">TRACK / VIRTUAL BEACON</span>
        <span class="skytrack-chip">ENVIRONMENT / AIRBORNE</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # SKYTRACK_FUTURISTIC_COMMAND_CENTER_V1
    # Premium aerospace surveillance interface.
    # Radar graphics are illustrative; telemetry is recorded evidence.

    def _sky_value(data, key, suffix="", decimals=3):
        value = data.get(key) if isinstance(data, dict) else None
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.{decimals}f}{suffix}"
        except (TypeError, ValueError):
            return "N/A"

    sky_confidence = _sky_value(confidence, "mean")
    sky_error = _sky_value(centroid, "mean", " px")
    sky_latency = _sky_value(latency, "mean", " ms", 1)

    st.components.v1.html(__import__("textwrap").dedent("""
    <style>
    html, body {
        margin: 0 !important;
        padding: 0 !important;
        background: transparent !important;
        color: #d9f7e9;
        overflow-x: hidden;
    }

    /* ===== FUTURISTIC SKYTRACK COMMAND CENTER ===== */

    .sky-command {
        background:
          radial-gradient(ellipse at 50% 45%,
          rgba(0,100,65,.15), transparent 55%),
          linear-gradient(145deg,#02090e,#06131a 65%,#02090e);
        border: 1px solid rgba(53,220,153,.24);
        border-radius: 18px;
        padding: 18px;
        margin: 12px 0 26px 0;
        color: #d9f7e9;
        box-shadow: 0 15px 55px rgba(0,0,0,.4);
        overflow: hidden;
    }

    .sky-command-top {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:12px;
        border-bottom:1px solid rgba(75,190,145,.2);
        padding-bottom:13px;
        margin-bottom:17px;
        font-family:monospace;
        letter-spacing:2px;
        font-size:11px;
        color:#75bda1;
    }

    .sky-command-status {
        color:#48e69b;
        letter-spacing:1px;
    }

    .sky-command-grid {
        display:grid;
        grid-template-columns: minmax(180px,.85fr)
                               minmax(320px,2fr)
                               minmax(180px,.85fr);
        gap:14px;
        align-items:stretch;
    }

    .sky-glass {
        background:linear-gradient(145deg,
          rgba(10,29,35,.94),rgba(4,15,22,.96));
        border:1px solid rgba(86,180,151,.22);
        border-radius:13px;
        padding:15px;
        min-width:0;
    }

    .sky-panel-title {
        color:#a5d7c2;
        font-size:11px;
        letter-spacing:2px;
        font-family:monospace;
        margin-bottom:15px;
    }

    .sky-source-item {
        border-bottom:1px solid rgba(110,180,155,.12);
        padding:12px 0;
    }

    .sky-source-label {
        color:#d3eee1;
        font-size:13px;
        font-weight:600;
    }

    .sky-source-note {
        color:#729b8c;
        font-size:10px;
        margin-top:5px;
    }

    .sky-status-dot {
        display:inline-block;
        width:7px;height:7px;
        border-radius:50%;
        background:#36df91;
        box-shadow:0 0 12px #36df91;
        margin-right:8px;
    }

    .sky-big-number {
        font-size:36px;
        color:#45e7a0;
        font-weight:500;
        font-family:monospace;
        margin-top:10px;
    }

    .sky-mini-number {
        font-size:21px;
        color:#52e8c0;
        font-family:monospace;
        margin-top:7px;
    }

    .sky-mini-label {
        font-size:10px;
        color:#83aa9c;
        letter-spacing:1px;
        margin-top:5px;
    }

    /* Radar */

    .sky-radar-panel {
        position:relative;
        min-height:440px;
        display:flex;
        justify-content:center;
        align-items:center;
        background:
          radial-gradient(circle,
          rgba(12,92,65,.16),rgba(2,13,18,.96) 72%);
        border:1px solid rgba(65,210,150,.22);
        border-radius:15px;
        overflow:hidden;
    }

    .sky-radar {
        position:relative;
        width:min(100%,410px);
        aspect-ratio:1;
        border-radius:50%;
        border:1px solid rgba(71,220,157,.65);
        background:
          repeating-radial-gradient(circle,
            transparent 0 19%,
            rgba(53,199,137,.18) 19.2% 19.5%),
          linear-gradient(90deg,
            transparent 49.8%,rgba(65,220,151,.2) 50%,
            transparent 50.3%),
          linear-gradient(0deg,
            transparent 49.8%,rgba(65,220,151,.2) 50%,
            transparent 50.3%),
          radial-gradient(circle,#09271f,#04120f 75%);
        box-shadow:
          0 0 35px rgba(20,220,130,.10),
          inset 0 0 35px rgba(20,220,130,.13);
    }

    .sky-radar:before {
        content:"";
        position:absolute;
        inset:0;
        border-radius:50%;
        background:conic-gradient(
          from 0deg,
          transparent 0deg,
          transparent 290deg,
          rgba(45,255,150,.02) 315deg,
          rgba(45,255,150,.30) 350deg,
          transparent 360deg
        );
        animation:sky-sweep 5s linear infinite;
    }

    .sky-radar:after {
        content:"";
        position:absolute;
        left:50%;top:0;bottom:0;
        width:1px;
        background:rgba(69,225,158,.25);
    }

    @keyframes sky-sweep {
        to { transform:rotate(360deg); }
    }

    .sky-radar-center {
        position:absolute;
        width:8px;height:8px;
        border-radius:50%;
        left:50%;top:50%;
        transform:translate(-50%,-50%);
        background:#67ffad;
        box-shadow:0 0 18px #4aff9d;
    }

    .sky-target {
        position:absolute;
        width:10px;height:10px;
        border:2px solid #4cffaa;
        border-radius:50%;
        background:rgba(60,255,160,.25);
        box-shadow:0 0 13px #35ff9c;
    }

    .sky-target.t1 {left:27%;top:30%;}
    .sky-target.t2 {left:70%;top:37%;border-color:#ffd166;
                    box-shadow:0 0 12px #ffd166;}
    .sky-target.t3 {left:61%;top:72%;}
    .sky-target.t4 {left:34%;top:67%;}

    .sky-radar-label {
        position:absolute;
        color:#8fd9b3;
        font-family:monospace;
        font-size:9px;
        white-space:nowrap;
        text-shadow:0 0 8px #1eaa75;
    }

    .sky-radar-label.l1 {left:30%;top:27%;}
    .sky-radar-label.l2 {left:73%;top:34%;color:#ffd166;}
    .sky-radar-label.l3 {left:64%;top:69%;}
    .sky-radar-label.l4 {left:17%;top:64%;}

    .sky-radar-heading {
        position:absolute;
        top:15px;left:17px;
        color:#8bbda8;
        font-family:monospace;
        font-size:10px;
        letter-spacing:2px;
        z-index:2;
    }

    .sky-radar-footer {
        position:absolute;
        bottom:15px;
        color:#6d9d89;
        font-size:9px;
        font-family:monospace;
        letter-spacing:1px;
        z-index:2;
    }

    .sky-meter {
        height:5px;
        background:#142a27;
        border-radius:8px;
        margin-top:8px;
        overflow:hidden;
    }

    .sky-meter-fill {
        height:100%;
        background:linear-gradient(90deg,#13a86d,#53ffc0);
        border-radius:8px;
    }

    .sky-command-foot {
        display:flex;
        justify-content:space-between;
        flex-wrap:wrap;
        gap:10px;
        margin-top:14px;
        border-top:1px solid rgba(75,190,145,.18);
        padding-top:12px;
        font-family:monospace;
        font-size:9px;
        color:#739b8b;
        letter-spacing:1px;
    }

    @media(max-width:900px) {
        .sky-command-grid {
            grid-template-columns:1fr;
        }
        .sky-radar-panel {min-height:370px;}
    }
    </style>

    <div class="sky-command">

      <div class="sky-command-top">
        <span>FSOC / SKYTRACK AI / COMMAND CENTER</span>
        <span class="sky-command-status">
          <span class="sky-status-dot"></span>RECORDED SYSTEM
        </span>
      </div>

      <div class="sky-command-grid">

        <!-- LEFT PANEL -->
        <div class="sky-glass">

          <div class="sky-panel-title">01 / SYSTEM SOURCE</div>

          <div class="sky-source-item">
            <div class="sky-source-label">
              <span class="sky-status-dot"></span>VISION DETECTOR
            </div>
            <div class="sky-source-note">
              YOLO BEACON MODEL / SAVED WEIGHTS
            </div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">DETECTOR CONFIDENCE</div>
            <div class="sky-big-number">""" + sky_confidence + r"""</div>
            <div class="sky-source-note">RECORDED MEAN CONFIDENCE</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">CENTROID ERROR</div>
            <div class="sky-mini-number">""" + sky_error + r"""</div>
            <div class="sky-source-note">C18 RECORDED EXPERIMENT</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">INFERENCE LATENCY</div>
            <div class="sky-mini-number">""" + sky_latency + r"""</div>
            <div class="sky-source-note">RECORDED DETECTOR LATENCY</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">
              <span class="sky-status-dot"></span>DATA ARCHIVE
            </div>
            <div class="sky-source-note">
              EXPERIMENT RECORDS AVAILABLE
            </div>
          </div>

        </div>

        <!-- CENTER RADAR -->
        <div class="sky-radar-panel">

          <div class="sky-radar-heading">
            VIRTUAL CAMERA / RADAR VISUALIZATION
          </div>

          <div class="sky-radar">

            <div class="sky-target t1"></div>
            <div class="sky-target t2"></div>
            <div class="sky-target t3"></div>
            <div class="sky-target t4"></div>

            <div class="sky-radar-label l1">TARGET A</div>
            <div class="sky-radar-label l2">TARGET B</div>
            <div class="sky-radar-label l3">TARGET C</div>
            <div class="sky-radar-label l4">TARGET D</div>

            <div class="sky-radar-center"></div>
          </div>

          <div class="sky-radar-footer">
            DECORATIVE RADAR / NOT LIVE AIRCRAFT DATA
          </div>

        </div>

        <!-- RIGHT PANEL -->
        <div class="sky-glass">

          <div class="sky-panel-title">02 / TRACKING TELEMETRY</div>

          <div class="sky-source-item">
            <div class="sky-source-label">VISION SYSTEM</div>
            <div class="sky-mini-number">READY</div>
            <div class="sky-source-note">MODEL ARTIFACT AVAILABLE</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">CENTROID WITHIN 2 PX</div>
            <div class="sky-mini-number">100.0%</div>
            <div class="sky-source-note">C18 EXPERIMENT / NOT F-15 VIDEO</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">TRAJECTORY CASES</div>
            <div class="sky-mini-number">05</div>
            <div class="sky-source-note">RECORDED VALIDATION PATHS</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">AI ENGINEER</div>
            <div class="sky-mini-number">NEMOTRON</div>
            <div class="sky-source-note">NEBIUS INTEGRATION</div>
          </div>

          <div class="sky-source-item">
            <div class="sky-source-label">SYSTEM MODE</div>
            <div class="sky-mini-number">OFFLINE</div>
            <div class="sky-source-note">RECORDED EXPERIMENT DATA</div>
          </div>

        </div>

      </div>

      <div class="sky-command-foot">
        <span>SKYTRACK AI / FSOC RESEARCH PROTOTYPE</span>
        <span>RADAR GRAPHICS ARE ILLUSTRATIVE</span>
        <span>NO LIVE AIRCRAFT CONNECTION</span>
      </div>

    </div>
    """), height=760, scrolling=False)



    # SKYTRACK-MISSION-STATUS-PANEL
    st.markdown("""
    <div style="
        display:grid;
        grid-template-columns:repeat(auto-fit,minmax(155px,1fr));
        gap:10px;
        margin:18px 0 22px 0;
        font-family:Inter,Arial,sans-serif;
    ">
      <div style="
          background:linear-gradient(135deg,#101e2b,#0b131d);
          border:1px solid #244154;
          border-left:3px solid #37d6e8;
          border-radius:8px;
          padding:14px 16px;
      ">
        <div style="font-size:10px;letter-spacing:1.8px;color:#8ba8ba">
          ENVIRONMENT
        </div>
        <div style="font-size:16px;font-weight:700;color:#e8f7ff;margin-top:7px">
          SIMULATION
        </div>
        <div style="font-size:11px;color:#7f9aac;margin-top:5px">
          Recorded research run
        </div>
      </div>
      <div style="
          background:linear-gradient(135deg,#101e2b,#0b131d);
          border:1px solid #244154;
          border-left:3px solid #37d6e8;
          border-radius:8px;
          padding:14px 16px;
      ">
        <div style="font-size:10px;letter-spacing:1.8px;color:#8ba8ba">
          TRACKING REFERENCE
        </div>
        <div style="font-size:16px;font-weight:700;color:#e8f7ff;margin-top:7px">
          OPTICAL BEACON
        </div>
        <div style="font-size:11px;color:#7f9aac;margin-top:5px">
          Virtual reference target
        </div>
      </div>
      <div style="
          background:linear-gradient(135deg,#101e2b,#0b131d);
          border:1px solid #244154;
          border-left:3px solid #54d6a0;
          border-radius:8px;
          padding:14px 16px;
      ">
        <div style="font-size:10px;letter-spacing:1.8px;color:#8ba8ba">
          EVIDENCE
        </div>
        <div style="font-size:16px;font-weight:700;color:#e8f7ff;margin-top:7px">
          DATA LOADED
        </div>
        <div style="font-size:11px;color:#7f9aac;margin-top:5px">
          Saved measurements
        </div>
      </div>
    </div>
    <div style="
        border:1px solid #493d25;
        background:#19170f;
        border-radius:6px;
        padding:10px 13px;
        color:#d8bd80;
        font-size:11px;
        letter-spacing:.3px;
        margin-bottom:14px;
    ">
      <strong>MISSION NOTE //</strong>
      The fighter-jet scene provides aviation context.
      The simulated controller tracks a virtual optical beacon;
      this dashboard is not a live aircraft tracking system.
    </div>
    """, unsafe_allow_html=True)

    # SKYTRACK-JET-TELEMETRY-LAYOUT
    st.markdown("""
    <style>
    .sky-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid #23475b;
        border-radius: 12px;
        min-height: 300px;
        padding: 24px;
        background:
          radial-gradient(ellipse at 73% 48%, rgba(20,135,177,.22), transparent 38%),
          linear-gradient(125deg, #0c1723 0%, #101e2d 52%, #07111a 100%);
        box-shadow: inset 0 0 50px rgba(36,170,210,.04);
    }
    .sky-hero:before {
        content: "";
        position: absolute;
        inset: 0;
        background-image:
          linear-gradient(rgba(54,153,186,.07) 1px, transparent 1px),
          linear-gradient(90deg, rgba(54,153,186,.07) 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none;
    }
    .sky-hero-content {position:relative;z-index:1}
    .sky-kicker {
        color:#52d8ed;font:10px monospace;letter-spacing:2px;
    }
    .sky-hero-title {
        color:#e8f6ff;font-size:clamp(24px,3vw,38px);
        font-weight:800;line-height:1.05;margin-top:12px;
    }
    .sky-hero-copy {
        color:#8ca8bb;font-size:12px;line-height:1.7;
        max-width:460px;margin-top:12px;
    }
    .sky-hud-row {
        display:flex;flex-wrap:wrap;gap:8px;margin-top:22px;
    }
    .sky-hud-tag {
        padding:7px 9px;border:1px solid #28495b;
        border-radius:4px;color:#b6d5e5;
        background:rgba(8,20,31,.72);
        font:9px monospace;letter-spacing:.7px;
    }
    .sky-jet-mark {
        position:absolute;right:5%;top:18%;
        width:42%;height:65%;opacity:.82;
    }
    .sky-jet-mark svg {width:100%;height:100%}
    .sky-telemetry {
        border:1px solid #203d50;border-radius:10px;
        background:linear-gradient(150deg,#101e2a,#0a121c);
        padding:16px 17px;margin-bottom:10px;
    }
    .sky-telemetry-label {
        font:10px monospace;letter-spacing:1.5px;color:#82a0b4;
    }
    .sky-telemetry-value {
        color:#e8f7ff;font-size:25px;font-weight:750;
        margin-top:7px;line-height:1.2;
    }
    .sky-telemetry-note {font-size:11px;color:#829bac;margin-top:5px}
    .sky-section-label {
        font:10px monospace;letter-spacing:1.8px;color:#4fd8e9;
        margin:18px 0 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    hero_left, hero_right = st.columns([1.65, 1], gap="medium")

    with hero_left:
        st.markdown("""
        <div class="sky-hero">
          <div class="sky-hero-content">
            <div class="sky-kicker">SKYTRACK // AIRBORNE SYSTEMS</div>
            <div class="sky-hero-title">OPTICAL<br>TRACKING CONSOLE</div>
            <div class="sky-hero-copy">
              Fighter-jet mission environment with a virtual optical beacon
              as the simulated tracking reference.
            </div>
            <div class="sky-hud-row">
              <span class="sky-hud-tag">MODE / SIMULATION</span>
              <span class="sky-hud-tag">TARGET / VIRTUAL BEACON</span>
              <span class="sky-hud-tag">SOURCE / RECORDED DATA</span>
            </div>
          </div>
          <div class="sky-jet-mark">
            <svg viewBox="0 0 420 300" xmlns="http://www.w3.org/2000/svg"
                 role="img" aria-label="Stylized fighter jet silhouette">
              <defs>
                <linearGradient id="jetMetal" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#9cecff"/>
                  <stop offset="45%" stop-color="#3986a8"/>
                  <stop offset="100%" stop-color="#18384d"/>
                </linearGradient>
              </defs>
              <g fill="none" stroke="#3ad9ed" stroke-width="1" opacity=".45">
                <circle cx="230" cy="148" r="105"/>
                <circle cx="230" cy="148" r="78" stroke-dasharray="3 8"/>
                <path d="M230 25V270 M105 148H355"/>
                <path d="M125 53L335 243 M335 53L125 243" opacity=".4"/>
              </g>
              <path d="M232 24
                       L246 92 L330 151 L332 164
                       L253 143 L244 211 L279 239
                       L278 250 L232 231 L186 250
                       L185 239 L220 211 L211 143
                       L132 164 L134 151 L218 92 Z"
                    fill="url(#jetMetal)" stroke="#a5efff" stroke-width="1.5"/>
              <path d="M232 43 L238 116 L232 193 L226 116 Z"
                    fill="#d2f8ff" opacity=".65"/>
              <path d="M217 145 L232 168 L247 145"
                    fill="none" stroke="#0a2637" stroke-width="3"/>
              <circle cx="232" cy="118" r="4" fill="#5af2ff"/>
              <g stroke="#59d9ee" stroke-width="1.5">
                <path d="M232 8V28 M232 268V289 M92 148H112 M352 148H372"/>
              </g>
            </svg>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # The right column uses conservative, evidence-grounded status labels.
    with hero_right:
        st.markdown('<div class="sky-section-label">SYSTEM TELEMETRY / STATUS</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="sky-telemetry">
          <div class="sky-telemetry-label">TRACKING MODE</div>
          <div class="sky-telemetry-value">VIRTUAL</div>
          <div class="sky-telemetry-note">Beacon reference tracking</div>
        </div>
        <div class="sky-telemetry">
          <div class="sky-telemetry-label">DATA SOURCE</div>
          <div class="sky-telemetry-value">RECORDED</div>
          <div class="sky-telemetry-note">Saved experiment evidence</div>
        </div>
        <div class="sky-telemetry">
          <div class="sky-telemetry-label">MISSION LINK</div>
          <div class="sky-telemetry-value" style="font-size:19px;color:#58d8a5">
            LOCAL DEMO
          </div>
          <div class="sky-telemetry-note">No live aircraft connection</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sky-section-label">PERFORMANCE EVIDENCE / VALIDATED RESULTS BELOW</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sky-subtitle">Virtual-camera alignment and optical-beacon tracking '
        'in an airborne jet scene.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:17px'></div>", unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("DETECTOR MODEL", "READY" if MODEL.exists() else "MISSING",
             "Saved weights available" if MODEL.exists() else "Weights not found")
    b.metric("FLIGHT DEMO", "READY" if VIDEO.exists() else "MISSING",
             "Recorded visualization")
    c.metric("MEAN CENTROID ERROR",
             f"{centroid['mean']:.3f} px" if "mean" in centroid else "N/A",
             "Saved coordinate comparison")
    d.metric("DETECTOR LATENCY",
             f"{latency['mean']:.1f} ms" if "mean" in latency else "N/A",
             "Mean recorded inference time")

    st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.75, 1], gap="large")

    with left:
        st.markdown('<div class="sky-eyebrow">01 / AIRBORNE VISUALIZATION</div>',
                    unsafe_allow_html=True)
        st.subheader("JET ENVIRONMENT // RECORDED RUN")

        # SKYTRACK_CINEMATIC_UPGRADE_V1 — RECORDED DATA HUD
        hud_confidence = confidence.get("mean")
        hud_centroid = centroid.get("mean")
        hud_latency = latency.get("mean")

        hud_confidence_text = (
            f"{hud_confidence:.3f}" if hud_confidence is not None else "N/A"
        )
        hud_centroid_text = (
            f"{hud_centroid:.3f} px" if hud_centroid is not None else "N/A"
        )
        hud_latency_text = (
            f"{hud_latency:.1f} ms" if hud_latency is not None else "N/A"
        )

        st.markdown(f"""
        <div class="skytrack-hud">
          <div class="skytrack-hud-card">
            <div class="skytrack-hud-label">DETECTOR CONFIDENCE</div>
            <div class="skytrack-hud-value">{hud_confidence_text}</div>
            <div class="skytrack-hud-note">Mean recorded confidence</div>
          </div>
          <div class="skytrack-hud-card">
            <div class="skytrack-hud-label">CENTROID ERROR</div>
            <div class="skytrack-hud-value">{hud_centroid_text}</div>
            <div class="skytrack-hud-note">Detector-to-reference displacement</div>
          </div>
          <div class="skytrack-hud-card">
            <div class="skytrack-hud-label">INFERENCE LATENCY</div>
            <div class="skytrack-hud-value">{hud_latency_text}</div>
            <div class="skytrack-hud-note">Mean recorded inference time</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="sky-eyebrow">01 / CINEMATIC AIRCRAFT DEMONSTRATION</div>',
            unsafe_allow_html=True,
        )
        st.subheader("FSOC SkyTrack AI // Cinematic Demo")
        st.caption(
            "75-second visual demonstration using archival F-15 footage. "
            "Aircraft detection and ByteTrack overlays are illustrative "
            "computer-vision results, not FSOC terminal performance evidence."
        )

        if CINEMATIC_VIDEO.exists():
            st.video(str(CINEMATIC_VIDEO))
        else:
            st.warning("The cinematic aircraft demonstration was not found.")

        st.markdown(
            "<span class='sky-warn'>ARCHIVAL AIRCRAFT FOOTAGE — NOT A LIVE FEED "
            "OR VALIDATED FSOC EXPERIMENT</span>",
            unsafe_allow_html=True,
        )


        st.markdown(
            "<span class='sky-warn'>RECORDED FOOTAGE — NOT A LIVE FEED</span>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="sky-eyebrow">02 / TRACKING TELEMETRY</div>',
                    unsafe_allow_html=True)
        st.subheader("OPTICAL SYSTEM")

        metric_card(
            "DETECTOR CONFIDENCE",
            f"{confidence['mean']:.3f}" if "mean" in confidence else "N/A",
            "Mean saved YOLO confidence",
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        metric_card(
            "CENTROID WITHIN 2 PX",
            f"{centroid['within_2px_percent']:.1f}%"
            if "within_2px_percent" in centroid else "N/A",
            "Against saved ground truth",
        )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        metric_card(
            "TRAJECTORY CASES",
            str(len(trajectory)) if trajectory is not None else "N/A",
            "Imported validation records",
        )

        st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
        st.markdown("**SYSTEM NOTES**")
        st.caption("• Virtual airborne environment")
        st.caption("• Optical beacon is the tracked reference")
        st.caption("• Saved data, not live inference")
        st.caption("• Nemotron copilot integration previously verified")

    st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
    st.markdown('<div class="sky-eyebrow">03 / MISSION SUMMARY</div>',
                unsafe_allow_html=True)
    st.subheader("Current system capabilities")

    caps = st.columns(3)
    with caps[0]:
        metric_card("VISION", "YOLO", "Beacon detection model artifact")
    with caps[1]:
        metric_card("EVALUATION", "5 PATHS", "Recorded trajectory validation")
    with caps[2]:
        metric_card("AI ENGINEER", "AI + LOCAL", "Evidence diagnostics and integrated copilot")

# ---------------- OPTICAL TRACKING LAB ----------------
elif page == "Optical Tracking Lab":

    render_skytrack_page_hero(
        "02 / OPTICAL LAB",
        "OPTICAL TRACKING <span>LAB</span>",
        "Detector measurements, virtual beacon alignment, recorded camera telemetry and experiment playback.",
        "EXPERIMENT DATA AVAILABLE"
    )

    st.markdown('<div class="sky-eyebrow">LAB / 02</div>', unsafe_allow_html=True)
    st.title("Optical Tracking Lab")
    st.caption("Inspect detector measurements and saved tracking data.")

    if c18 is None:
        st.error("C18 metrics CSV was not found.")
    else:
        st.markdown(f"**Loaded:** {len(c18)} frames · {len(c18.columns)} fields")
        numeric = list(c18.select_dtypes(include="number").columns)

        if numeric:
            selected = st.selectbox("Telemetry signal", numeric)
            x_col = "Time (s)" if "Time (s)" in c18.columns else None

            plot_data = c18.copy()
            plot_data[selected] = pd.to_numeric(plot_data[selected], errors="coerce")
            plot_data = plot_data.dropna(subset=[selected])

            if x_col:
                fig = px.line(
                    plot_data, x=x_col, y=selected,
                    title=f"RECORDED TELEMETRY / {selected}",
                )
            else:
                fig = px.line(
                    plot_data, y=selected,
                    title=f"RECORDED TELEMETRY / {selected}",
                )

            st.plotly_chart(chart_theme(fig), use_container_width=True)

        st.subheader("Frame records")
        st.dataframe(c18, use_container_width=True, hide_index=True, height=370)

        st.download_button(
            "Export current CSV",
            data=c18.to_csv(index=False).encode("utf-8"),
            file_name="skytrack_c18_metrics.csv",
            mime="text/csv",
        )


    st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sky-eyebrow'>RECORDED EVIDENCE / VIDEO PLAYBACK</div>",
        unsafe_allow_html=True,
    )
    st.subheader("Optical Tracking Demonstration")
    st.caption(
        "Recorded 10-second demonstration. This is not a live aircraft feed."
    )
    # OPTICAL LAB VIDEO PANEL
    if VIDEO.exists():
        st.video(str(VIDEO), format="video/mp4", start_time=0)
    else:
        st.error("The H.264 demonstration video is missing.")
    
# ---------------- FLIGHT ANALYTICS ----------------
elif page == "Flight Analytics":

    render_skytrack_page_hero(
        "03 / FLIGHT ANALYTICS",
        "FLIGHT <span>ANALYTICS</span>",
        "Trajectory validation, centroid error distribution and quantitative performance analysis.",
        "VALIDATION RECORDS LOADED"
    )

    st.markdown('<div class="sky-eyebrow">ANALYTICS / 03</div>',
                unsafe_allow_html=True)
    st.title("Flight & Tracking Analytics")
    st.caption("Comparisons are based on the saved validation dataset.")

    if trajectory is None or trajectory.empty:
        st.error("Trajectory validation CSV is unavailable.")
    else:
        st.subheader("Trajectory comparison")
        metric_options = [
            c for c in trajectory.columns
            if pd.api.types.is_numeric_dtype(trajectory[c])
        ]
        if metric_options:
            selected_metric = st.selectbox("Performance metric", metric_options)
            fig = px.bar(
                trajectory,
                x="Trajectory",
                y=selected_metric,
                title=f"RECORDED PERFORMANCE / {selected_metric}",
                text_auto=".3f",
            )
            st.plotly_chart(chart_theme(fig, 390), use_container_width=True)

        st.subheader("Validation records")
        st.dataframe(trajectory, use_container_width=True, hide_index=True)

        st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
        st.subheader("Detector error distribution")

        err_path = PROJECT / "results/reports/c18_derived_centroid_errors.csv"
        errors = read_csv(err_path)

        if errors is not None and "centroid_error_px" in errors.columns:
            fig = px.histogram(
                errors,
                x="centroid_error_px",
                nbins=25,
                title="RECORDED CENTROID ERROR / PIXELS",
            )
            st.plotly_chart(chart_theme(fig), use_container_width=True)
        else:
            st.info("Run the evidence diagnostics cell to generate the error distribution.")


# ---------------- AI MISSION ENGINEER ----------------
elif page == "AI Mission Engineer":

    render_skytrack_page_hero(
        "04 / AI ENGINEERING",
        "AI MISSION <span>ENGINEER</span>",
        "Evidence-aware engineering assistant with local diagnostics and NVIDIA Nemotron integration.",
        "AI WORKSTATION"
    )

    import importlib.util
    import json

    st.markdown(
        '<div class="sky-eyebrow">ENGINEERING ASSISTANT / 04</div>',
        unsafe_allow_html=True,
    )
    st.title("AI Mission Engineer")
    st.caption(
        "Evidence-aware diagnostics for the FSOC SkyTrack AI research prototype."
    )

    # Workstation header
    st.markdown("""
    <div style="
        background:linear-gradient(120deg,#101e2b,#0a121c);
        border:1px solid #244154;border-left:3px solid #35d6e8;
        border-radius:10px;padding:18px 20px;margin:12px 0 18px 0;
    ">
      <div style="font:10px monospace;letter-spacing:2px;color:#54d8eb">
        ENGINE STATUS / DUAL ANALYSIS
      </div>
      <div style="font-size:23px;font-weight:750;color:#e8f7ff;margin-top:8px">
        Mission Diagnostics Workstation
      </div>
      <div style="font-size:12px;line-height:1.7;color:#92aabc;margin-top:7px">
        Analyze saved project evidence using either the local evidence engine
        or NVIDIA Nemotron through Nebius Token Factory. External model
        requests require valid credentials and network access.
        Results remain subject to verification against source evidence.
      </div>
    </div>
    """, unsafe_allow_html=True)

    previous_ai_result = st.session_state.get("skytrack_ai_last_result", {})
    previous_ai_mode = str(
        previous_ai_result.get("mode", "mock")
    ).lower() if isinstance(previous_ai_result, dict) else "mock"

    if previous_ai_mode == "nebius":
        analysis_mode_label = "NEMOTRON"
        analysis_mode_note = "Nebius-hosted NVIDIA model"
    else:
        analysis_mode_label = "LOCAL"
        analysis_mode_note = "Evidence-grounded local engine"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="sky-telemetry">
          <div class="sky-telemetry-label">ANALYSIS MODE</div>
          <div class="sky-telemetry-value" style="font-size:19px">{analysis_mode_label}</div>
          <div class="sky-telemetry-note">{analysis_mode_note}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="sky-telemetry">
          <div class="sky-telemetry-label">EVIDENCE</div>
          <div class="sky-telemetry-value" style="font-size:19px">SAVED FILES</div>
          <div class="sky-telemetry-note">Reports and trajectory records</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        if previous_ai_mode == "nebius":
            nemotron_status = "CONNECTED"
            nemotron_color = "#42D9FF"
            nemotron_note = "Last Nebius request succeeded"
        else:
            nemotron_status = "READY TO TEST"
            nemotron_color = "#d8bd80"
            nemotron_note = "No successful Nebius response yet"

        st.markdown(f"""
        <div class="sky-telemetry">
          <div class="sky-telemetry-label">NEMOTRON LINK</div>
          <div class="sky-telemetry-value"
               style="font-size:19px;color:{nemotron_color}">
            {nemotron_status}
          </div>
          <div class="sky-telemetry-note">{nemotron_note}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        '<div class="sky-section-label">01 / ENGINEERING QUERY</div>',
        unsafe_allow_html=True,
    )

    quick_options = {
        "Choose a diagnostic": "",
        "Detector centroid accuracy": (
            "Summarize detector centroid accuracy and explain its limitations."
        ),
        "Trajectory RMSE comparison": (
            "Compare trajectory RMSE and flag unusual results."
        ),
        "Inference latency": (
            "Explain saved inference latency and the correct interpretation of FPS."
        ),
        "Engineering risks to verify": (
            "Flag unusual results and list engineering checks to perform."
        ),
    }

    with st.form("skytrack_engineering_query", clear_on_submit=False):
        ai_mode_label = st.radio(
            "Analysis engine",
            ["Local evidence engine", "NVIDIA Nemotron via Nebius"],
            horizontal=True,
            key="sky_ai_engine_mode",
            help=(
                "Local mode uses the saved evidence-grounded engine. "
                "Nemotron mode requires secure Nebius API configuration."
            ),
        )
        selected = st.selectbox(
            "Diagnostic preset",
            list(quick_options.keys()),
            key="sky_ai_preset",
        )
        question = st.text_area(
            "Ask an engineering question",
            placeholder=(
                "Example: Why should I verify the figure-eight RMSE "
                "before making a performance claim?"
            ),
            height=105,
            key="sky_ai_question",
        )
        submitted = st.form_submit_button(
            "Run engineering analysis",
            use_container_width=True,
        )

    if submitted:
        user_question = question.strip() or quick_options.get(selected, "")

        if not user_question:
            st.warning("Enter a question or select a diagnostic preset.")
        else:
            try:
                spec = importlib.util.spec_from_file_location(
                    "skytrack_copilot_engine", str(ENGINE)
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError("Could not load the local copilot engine.")

                engine = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(engine)

                selected_mode = (
                    "nebius"
                    if ai_mode_label == "NVIDIA Nemotron via Nebius"
                    else "mock"
                )
                result = engine.ask_copilot(
                    user_question,
                    mode=selected_mode,
                )

                st.session_state["skytrack_ai_last_result"] = result
                st.session_state["skytrack_ai_last_question"] = user_question

            except Exception as exc:
                st.error(
                    "The local analysis could not be completed. "
                    "Check that the evidence files exist and are readable."
                )
                st.code(f"{type(exc).__name__}: {exc}")

    result = st.session_state.get("skytrack_ai_last_result")

    if result:
        st.markdown(
            '<div class="sky-section-label">02 / ANALYSIS OUTPUT</div>',
            unsafe_allow_html=True,
        )

        response_mode = str(result.get("mode", "mock")).lower()
        response_label = (
            "NVIDIA NEMOTRON / NEBIUS RESPONSE"
            if response_mode == "nebius"
            else "LOCAL EVIDENCE ENGINE RESPONSE"
        )

        st.markdown(f"""
        <div style="
            border:1px solid #25465a;border-radius:9px;
            background:#0c1722;padding:16px 18px;
        ">
          <div style="font:10px monospace;letter-spacing:1.5px;color:#50d9eb">
            {response_label}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Render the answer using Streamlit's safe Markdown renderer.
        # Keep quality flags separate from the main response.
        answer_text = str(
            result.get("answer") or "No answer was returned."
        )

        # Normalize literal escaped newlines from model output.
        # Handle one or more backslashes before the letter n.
        import re
        answer_text = re.sub(r"\\+n", "\n", answer_text)
        answer_text = answer_text.replace("\\r", "")

        normal_lines = []
        warning_lines = []

        for line in answer_text.splitlines():
            if "QUALITY FLAG:" in line:
                before, _, warning_text = line.partition("QUALITY FLAG:")
                if before.strip():
                    normal_lines.append(before.strip())
                if warning_text.strip():
                    warning_lines.append(warning_text.strip())
            else:
                normal_lines.append(line)

        normal_text = "\n".join(normal_lines).strip()

        if normal_text:
            st.markdown(
                """
                <style>
                .skytrack-answer {
                    background: #0c1722;
                    border: 1px solid #25465a;
                    border-radius: 9px;
                    padding: 12px 18px;
                    color: #d8e5ee;
                    line-height: 1.7;
                }
                .skytrack-answer h1,
                .skytrack-answer h2,
                .skytrack-answer h3 {
                    color: #50d9eb;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Markdown is rendered normally; model text is not inserted
            # into an unsafe HTML string.
            with st.container(border=True):
                st.markdown(normal_text)

        import html

        warning = result.get("warning")
        if warning:
            st.info(warning)

        provider = result.get("provider", "Local engine")
        mode = result.get("mode", "mock")
        st.caption(f"Execution mode: {mode} · Provider: {provider}")

        # STEP 54C — DETERMINISTIC EVIDENCE AUDIT
        # STEP 56B — STRUCTURED AUDIT DASHBOARD
        with st.expander("Deterministic evidence audit", expanded=False):
            st.caption(
                "Read-only consistency checks against saved project evidence. "
                "REVIEW findings require investigation and are not proven failures."
            )

            try:
                audit_findings = result.get("audit_findings", [])
                audit_summary = result.get("audit_summary", "")

                # Fallback for older results that predate Step 55C.
                if not audit_findings:
                    import importlib.util

                    auditor_path = PROJECT / "ai_agent" / "evidence_auditor.py"
                    spec = importlib.util.spec_from_file_location(
                        "skytrack_dashboard_auditor", auditor_path
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError("Could not load the evidence auditor.")

                    auditor_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(auditor_module)
                    audit_result = auditor_module.run_evidence_audit(PROJECT)
                    audit_findings = audit_result.get("findings", [])
                    audit_summary = (
                        f"Checks: {len(audit_findings)} | "
                        f"Read-only: {audit_result.get('read_only', True)}"
                    )

                passed_count = sum(
                    1 for item in audit_findings
                    if item.get("severity") == "PASS"
                )
                review_count = sum(
                    1 for item in audit_findings
                    if item.get("severity") == "REVIEW"
                )

                metric_cols = st.columns(3)
                metric_cols[0].metric("Total checks", len(audit_findings))
                metric_cols[1].metric("PASS", passed_count)
                metric_cols[2].metric("REVIEW", review_count)

                if audit_summary:
                    st.caption(str(audit_summary).splitlines()[0])

                if not audit_findings:
                    st.warning("No deterministic audit findings were returned.")

                for index, finding in enumerate(audit_findings):
                    severity = str(finding.get("severity", "UNKNOWN")).upper()
                    code = str(finding.get("code", "UNNAMED_CHECK"))
                    message = str(finding.get("message", ""))

                    heading = f"{severity} · {code}"
                    if severity == "PASS":
                        st.success(f"**{heading}** — {message}")
                    elif severity == "REVIEW":
                        st.warning(f"**{heading}** — {message}")
                    else:
                        st.info(f"**{heading}** — {message}")

                    details = finding.get("details")
                    if details:
                        with st.expander(
                            f"Diagnostic details · {code} · {index + 1}",
                            expanded=False,
                        ):
                            st.json(details)

                st.caption(
                    "These checks verify specific data-consistency conditions. "
                    "They do not independently establish scientific validity."
                )

            except Exception as audit_error:
                st.warning(
                    "The deterministic evidence audit could not be displayed. "
                    "The Copilot response remains available."
                )
                st.caption(
                    f"Audit diagnostic: {type(audit_error).__name__}: "
                    f"{audit_error}"
                )

        with st.expander("Evidence sources and provenance", expanded=True):
            evidence_files = result.get("evidence_files", [])
            if evidence_files:
                for item in evidence_files:
                    path = Path(item)
                    exists = path.is_file()
                    st.write(
                        f"{'AVAILABLE' if exists else 'MISSING'} — `{item}`"
                    )
            else:
                st.write(
                    "The engine did not return an explicit evidence-file list."
                )

        with st.expander("Engineering checks to keep in view"):
            st.markdown("""
            - Verify the unusually low figure-eight RMSE against its source data.
            - Investigate the straight trajectory's high maximum error.
            - Keep detector centroid error separate from closed-loop tracking error.
            - Do not treat reciprocal detector latency as end-to-end control-loop FPS.
            - Compare experiments only when their conditions and metrics are compatible.
            """)

    else:
        st.markdown(
            '<div class="sky-section-label">02 / READY FOR QUERY</div>',
            unsafe_allow_html=True,
        )
        st.info(
            "Choose a diagnostic or enter an engineering question to inspect "
            "the saved project evidence."
        )

    st.markdown(
        '<div class="sky-section-label">03 / MODEL INTEGRATION STATUS</div>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div style="
        background:#0c1722;border:1px solid #25465a;
        border-radius:8px;padding:14px 16px;color:#b9cddd;
        font-size:12px;line-height:1.7;
    ">
      <strong style="color:#50d9eb">
        NVIDIA Nemotron / Nebius: CONFIGURABLE
      </strong><br>
      Select the local evidence engine or NVIDIA Nemotron via Nebius
      in the Mission Engineer form. Nebius requests require valid API
      credentials and model configuration in the Streamlit process.<br>
      This dashboard uses saved experimental evidence and does not
      connect to a live aircraft feed or claim live aircraft tracking.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "<div class='sky-divider'></div>",
        unsafe_allow_html=True,
    )



# ---------------- SYSTEM ARCHITECTURE ----------------
elif page == "System Architecture":

    render_skytrack_page_hero(
        "05 / SYSTEM DESIGN",
        "SYSTEM <span>ARCHITECTURE</span>",
        "Native architecture of the AI-assisted virtual camera tracking research system.",
        "ARCHITECTURE REFERENCE"
    )

    st.markdown("""
    <style>
    .arch-wrap {
        background:#000000;
        border:1px solid rgba(22,139,255,.30);
        border-radius:16px;
        padding:22px;
        margin:12px 0 20px;
        color:#E4FFF0;
    }

    .arch-kicker {
        color:#168BFF;
        font:11px monospace;
        letter-spacing:3px;
        margin-bottom:8px;
    }

    .arch-heading {
        font-size:23px;
        font-weight:800;
        color:#E4FFF0;
        margin-bottom:7px;
    }

    .arch-subtitle {
        font-size:12px;
        color:#8BAF9B;
        margin-bottom:22px;
    }

    .arch-node {
        background:#071321;
        border:1px solid rgba(22,139,255,.40);
        border-radius:11px;
        padding:15px 13px;
        min-height:105px;
        box-shadow:0 0 15px rgba(22,139,255,.035);
        height:100%;
    }

    .arch-node.primary {
        background:#0A1B30;
        border-color:#168BFF;
        box-shadow:0 0 18px rgba(22,139,255,.10);
    }

    .arch-node.output {
        background:#08182A;
        border-color:rgba(22,139,255,.55);
    }

    .arch-node .num {
        color:#168BFF;
        font:10px monospace;
        letter-spacing:2px;
        margin-bottom:9px;
    }

    .arch-node .name {
        color:#E4FFF0;
        font-size:14px;
        font-weight:750;
        margin-bottom:7px;
        line-height:1.35;
    }

    .arch-node .desc {
        color:#91B5A0;
        font-size:11px;
        line-height:1.55;
    }

    .arch-node .tag {
        display:inline-block;
        color:#42D9FF;
        border:1px solid rgba(22,139,255,.35);
        border-radius:4px;
        padding:3px 6px;
        font:9px monospace;
        margin-top:10px;
    }

    .arch-arrow {
        color:#168BFF;
        text-align:center;
        font-size:22px;
        line-height:35px;
        text-shadow:0 0 12px rgba(22,139,255,.45);
    }

    .arch-section {
        color:#168BFF;
        font:10px monospace;
        letter-spacing:2.5px;
        margin:25px 0 12px;
        border-bottom:1px solid rgba(22,139,255,.18);
        padding-bottom:8px;
    }

    .arch-branch {
        border:1px dashed rgba(22,139,255,.30);
        background:#020408;
        border-radius:10px;
        padding:13px;
        height:100%;
    }

    .arch-branch-title {
        color:#42D9FF;
        font:11px monospace;
        letter-spacing:1.5px;
        margin-bottom:7px;
    }

    .arch-branch-text {
        color:#A4C4B1;
        font-size:11px;
        line-height:1.6;
    }

    .arch-footer {
        color:#7FA991;
        font:9px monospace;
        letter-spacing:1.4px;
        text-align:center;
        border-top:1px solid rgba(22,139,255,.2);
        padding-top:15px;
        margin-top:22px;
    }

    .arch-flow-label {
        color:#8BAF9B;
        font:9px monospace;
        letter-spacing:2px;
        text-align:center;
        margin:8px 0 2px;
    }

    @media(max-width:700px) {
        .arch-wrap { padding:12px; }
        .arch-heading { font-size:18px; }
        .arch-node { min-height:90px; }
    }
    </style>

    <div class="arch-wrap">

      <div class="arch-kicker">
        SKYTRACK AI / FSOC / RESEARCH SYSTEM
      </div>

      <div class="arch-heading">
        AI-ASSISTED VIRTUAL CAMERA TRACKING
      </div>

      <div class="arch-subtitle">
        End-to-end software architecture for coarse alignment
        of mobile free-space optical communication terminals
      </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-section">01 / INPUT & SIMULATION ENVIRONMENT</div>',
                unsafe_allow_html=True)

    a, b = st.columns(2, gap="medium")

    with a:
        st.markdown("""
        <div class="arch-node">
          <div class="num">01 / INPUT</div>
          <div class="name">AIRBORNE PLATFORM / TARGET MOTION</div>
          <div class="desc">
            Simulated target movement, trajectory and
            optical beacon reference position.
          </div>
          <span class="tag">SIMULATED INPUT</span>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="arch-node">
          <div class="num">02 / ENVIRONMENT</div>
          <div class="name">DISTURBANCE ENGINE</div>
          <div class="desc">
            Environmental disturbance, measurement noise,
            haze and camera jitter scenarios.
          </div>
          <span class="tag">ENVIRONMENT MODEL</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="arch-flow-label">SIMULATION INPUTS / CAMERA SCENE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="arch-arrow">↓</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="arch-section">02 / VIRTUAL CAMERA & COMPUTER VISION</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-node primary">
      <div class="num">03 / CAMERA SYSTEM</div>
      <div class="name">VIRTUAL CAMERA MODEL</div>
      <div class="desc">
        Camera field of view, image formation, pan and tilt
        orientation, target projection and visual observation.
      </div>
      <span class="tag">CAMERA SIMULATION</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">↓</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-node primary">
      <div class="num">04 / AI DETECTION</div>
      <div class="name">YOLO BEACON DETECTOR</div>
      <div class="desc">
        Detects the optical beacon in the camera frame and
        produces bounding-box coordinates and confidence.
      </div>
      <span class="tag">SAVED DETECTOR WEIGHTS</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">↓</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown("""
        <div class="arch-node">
          <div class="num">05 / POSITION</div>
          <div class="name">CENTROID ESTIMATION</div>
          <div class="desc">
            Converts the detected bounding box into
            a target image-plane centre coordinate.
          </div>
          <span class="tag">X / Y POSITION</span>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="arch-node">
          <div class="num">06 / ESTIMATION</div>
          <div class="name">TARGET STATE ESTIMATION</div>
          <div class="desc">
            Uses measured target position and available
            motion information for tracking calculations.
          </div>
          <span class="tag">TRACKING STATE</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">↓</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="arch-section">03 / TRACKING CONTROL & FEEDBACK</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-node primary">
      <div class="num">07 / CONTROL ENGINE</div>
      <div class="name">TRACKING CONTROLLER</div>
      <div class="desc">
        Uses target position error and controller logic
        to calculate camera correction commands.
      </div>
      <span class="tag">PID / PREDICTIVE CONTROL — AS CONFIGURED</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-arrow">↓</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-node">
      <div class="num">08 / ACTUATION MODEL</div>
      <div class="name">PAN / TILT CAMERA UPDATE</div>
      <div class="desc">
        Applies the calculated camera movement to the virtual
        camera model, closing the simulation feedback loop.
      </div>
      <span class="tag">VIRTUAL CAMERA FEEDBACK</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-flow-label">FEEDBACK RETURNS TO VIRTUAL CAMERA / NEXT FRAME</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="arch-section">04 / ANALYSIS, STORAGE & AI ASSISTANCE</div>',
                unsafe_allow_html=True)

    o1, o2, o3 = st.columns(3, gap="medium")

    with o1:
        st.markdown("""
        <div class="arch-node output">
          <div class="num">09 / VALIDATION</div>
          <div class="name">PERFORMANCE MONITOR</div>
          <div class="desc">
            Tracking error, RMSE, lock percentage,
            confidence and latency analysis.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with o2:
        st.markdown("""
        <div class="arch-node output">
          <div class="num">10 / STORAGE</div>
          <div class="name">DATA LOGGER</div>
          <div class="desc">
            Recorded experiment results, CSV datasets,
            benchmark tables and diagnostic reports.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with o3:
        st.markdown("""
        <div class="arch-node output">
          <div class="num">11 / AI ASSISTANT</div>
          <div class="name">NVIDIA NEMOTRON</div>
          <div class="desc">
            Configurable Nebius integration for
            engineering questions and evidence analysis.
          </div>
          <span class="tag">OPTIONAL EXTERNAL MODEL</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-footer">
      SKYTRACK AI / FSOC RESEARCH PROTOTYPE /
      RECORDED EXPERIMENTAL EVIDENCE /
      NOT A LIVE AIRCRAFT TRACKING SYSTEM
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="arch-section">05 / SYSTEM BOUNDARIES</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="arch-branch">
      <div class="arch-branch-title">RESEARCH IMPLEMENTATION NOTES</div>
      <div class="arch-branch-text">
        • The airborne jet scene is visualization context.<br>
        • The virtual optical beacon is the simulated tracking reference.<br>
        • Detector and controller performance must be interpreted
          using their corresponding recorded experiments.<br>
        • Nemotron is a configurable engineering assistant,
          not part of the optical control feedback loop.<br>
        • The dashboard does not represent a live aircraft feed
          or a deployed FSOC terminal.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sky-divider'></div>",
                unsafe_allow_html=True)


# ---------------- ASSET REGISTRY ----------------
elif page == "Asset Registry":

    render_skytrack_page_hero(
        "06 / MISSION INVENTORY",
        "ASSET <span>REGISTRY</span>",
        "Technical inventory of detector weights, recorded demonstrations, datasets and experiment reports.",
        "ASSET AUDIT"
    )

    st.markdown('<div class="sky-eyebrow">INVENTORY / 06</div>',
                unsafe_allow_html=True)
    st.title("Asset Registry")

    inventory = [
        ("Beacon detector weights", MODEL),
        ("Recorded airborne demonstration", VIDEO),
        ("Cinematic F-15 visual demonstration", CINEMATIC_VIDEO),
        ("System architecture", ARCH),
        ("Diagnostic JSON report", REPORT),
    ] + [(name, path) for name, path in DATASETS.items()]

    rows = []
    for name, path in inventory:
        rows.append({
            "Asset": name,
            "Status": "AVAILABLE" if path.is_file() else "MISSING",
            "Size (MB)": round(path.stat().st_size / (1024 * 1024), 3)
                if path.is_file() else None,
            "File": str(path),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if VIDEO.exists():
        st.subheader("Demonstration playback")
        st.video(str(VIDEO))

st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
st.markdown(
    '<div style="font-family:IBM Plex Mono;color:#637d90;font-size:.68rem;'
    'letter-spacing:1px">SKYTRACK AI // FSOC RESEARCH PROTOTYPE // '
    'RECORDED EVIDENCE IS NOT A LIVE FEED</div>',
    unsafe_allow_html=True,
)



# SKYTRACK_FULL_BLACK_EMERALD_THEME_V2

import streamlit as st

# Global Plotly theme defaults
try:
    import plotly.io as pio
    import plotly.graph_objects as go

    pio.templates["skytrack_emerald"] = go.layout.Template(
        layout=dict(
            paper_bgcolor="#020408",
            plot_bgcolor="#071321",
            font=dict(
                color="#C9FCE3",
                family="Arial, sans-serif"
            ),
            colorway=[
                "#168BFF",
                "#42D9FF",
                "#1264C5",
                "#A6E8FF",
                "#104A91",
                "#D9F3FF"
            ],
            xaxis=dict(
                color="#9BECC2",
                gridcolor="rgba(22,139,255,0.13)",
                zerolinecolor="rgba(22,139,255,0.3)",
                linecolor="rgba(22,139,255,0.35)"
            ),
            yaxis=dict(
                color="#9BECC2",
                gridcolor="rgba(22,139,255,0.13)",
                zerolinecolor="rgba(22,139,255,0.3)",
                linecolor="rgba(22,139,255,0.35)"
            ),
            legend=dict(font=dict(color="#C9FCE3")),
            margin=dict(l=45, r=25, t=45, b=40)
        )
    )
    pio.templates.default = "skytrack_emerald"
except Exception as e:
    print("Plotly theme note:", e)

# Global website CSS
st.markdown("""
<style>

/* =========================================
   SKYTRACK GLOBAL COLOUR SYSTEM
========================================= */

:root {
    --sky-bg: #020408;
    --sky-bg-secondary: #071321;
    --sky-panel: #0A1929;
    --sky-panel-hover: #08150E;
    --sky-green: #168BFF;
    --sky-green-bright: #42D9FF;
    --sky-green-dark: #1264C5;
    --sky-text: #E4FFF0;
    --sky-muted: #8BAF9B;
    --sky-border: rgba(22,139,255,0.28);
    --sky-glow: 0 0 18px rgba(22,139,255,0.10);
}

/* Main application background */
html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main,
.block-container {
    background: var(--sky-bg) !important;
    color: var(--sky-text) !important;
}

/* Remove unwanted blue backgrounds */
[data-testid="stAppViewContainer"] *,
[data-testid="stMain"] * {
    scrollbar-color: var(--sky-green-dark) var(--sky-bg);
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebarContent"] {
    background: #010302 !important;
    color: var(--sky-text) !important;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--sky-border) !important;
}

/* All card and container surfaces */
[data-testid="stMetric"],
[data-testid="stExpander"],
[data-testid="stContainer"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stForm"],
[data-testid="stPopover"],
[data-testid="stDialog"],
[data-testid="stTable"],
[data-testid="stDataFrame"] {
    background-color: var(--sky-panel) !important;
    border-color: var(--sky-border) !important;
    color: var(--sky-text) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #0A1929 !important;
    border: 1px solid var(--sky-border) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    box-shadow: var(--sky-glow) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: var(--sky-muted) !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: var(--sky-green-bright) !important;
}

/* Headings and text */
h1, h2, h3, h4, h5, h6 {
    color: var(--sky-text) !important;
}

p, label, li, span {
    /* Preserve individual component colours where appropriate */
}

small, .stCaption, [data-testid="stCaptionContainer"] {
    color: var(--sky-muted) !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    background: #06150D !important;
    color: var(--sky-green-bright) !important;
    border: 1px solid var(--sky-green) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease-in-out !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: #102B48 !important;
    border-color: var(--sky-green-bright) !important;
    box-shadow: 0 0 16px rgba(22,139,255,0.25) !important;
    color: #FFFFFF !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div,
[data-baseweb="textarea"] {
    background: #071321 !important;
    color: var(--sky-text) !important;
    border-color: var(--sky-border) !important;
    border-radius: 7px !important;
}

input, textarea {
    caret-color: var(--sky-green) !important;
}

/* Select menus and dropdowns */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="select"] ul,
[role="listbox"],
[role="option"] {
    background: #071321 !important;
    color: var(--sky-text) !important;
}

[role="option"]:hover {
    background: #102B48 !important;
}

/* Navigation and radio controls */
[data-testid="stSidebar"] [role="radiogroup"] label {
    color: var(--sky-text) !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #0B2038 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #020408 !important;
    border-bottom: 1px solid var(--sky-border) !important;
}

.stTabs [data-baseweb="tab"] {
    color: var(--sky-muted) !important;
}

.stTabs [aria-selected="true"] {
    color: var(--sky-green-bright) !important;
    border-bottom-color: var(--sky-green) !important;
}

/* Progress bars and sliders */
.stProgress > div > div > div {
    background: var(--sky-green) !important;
}

.stSlider [role="slider"] {
    background: var(--sky-green) !important;
}

/* Tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #071321 !important;
    border: 1px solid var(--sky-border) !important;
}

[data-testid="stDataFrame"] *,
[data-testid="stTable"] * {
    color: var(--sky-text) !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: #0A1929 !important;
    border: 1px solid var(--sky-border) !important;
    border-radius: 10px !important;
}

/* Alerts and status messages */
[data-testid="stAlert"] {
    background: #06130B !important;
    border: 1px solid var(--sky-border) !important;
    color: var(--sky-text) !important;
}

/* Links */
a {
    color: var(--sky-green-bright) !important;
}

a:hover {
    color: #FFFFFF !important;
}

/* Chart and media containers */
[data-testid="stPlotlyChart"],
[data-testid="stVegaLiteChart"],
[data-testid="stDeckGlChart"],
[data-testid="stImage"],
[data-testid="stVideo"] {
    background: #020408 !important;
    border: 1px solid var(--sky-border) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Embedded iframe containers */
iframe {
    background: #020408 !important;
    border-color: var(--sky-border) !important;
}

/* Horizontal dividers */
hr {
    border-color: var(--sky-border) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #020408;
}

::-webkit-scrollbar-thumb {
    background: #126B47;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--sky-green);
}

/* Selection */
::selection {
    background: rgba(22,139,255,0.3);
    color: #FFFFFF;
}

/* Remove bright blue focus outlines */
*:focus {
    outline-color: var(--sky-green) !important;
}

</style>
""", unsafe_allow_html=True)

# END SKYTRACK FULL BLACK EMERALD THEME


# SKYTRACK_BLACK_GREEN_FINAL_V4
import streamlit as st

st.markdown('\n<style>\n:root {\n  --bg:#000000;\n  --panel:#071321;\n  --panel2:#0A1929;\n  --green:#168BFF;\n  --bright:#42D9FF;\n  --muted:#8BAF9B;\n}\n\n/* Main backgrounds */\nhtml, body,\n[data-testid="stApp"],\n[data-testid="stAppViewContainer"],\n[data-testid="stMain"],\n[data-testid="stMainBlockContainer"],\n.main, .block-container {\n  background:#000000 !important;\n  color:#E4FFF0 !important;\n}\n\n/* Sidebar */\n[data-testid="stSidebar"],\n[data-testid="stSidebar"] > div {\n  background:#000000 !important;\n  border-right:1px solid rgba(22,139,255,.3) !important;\n}\n\n/* Streamlit panels */\n[data-testid="stMetric"],\n[data-testid="stVerticalBlockBorderWrapper"],\n[data-testid="stContainer"],\n[data-testid="stExpander"],\n[data-testid="stForm"],\n[data-testid="stAlert"] {\n  background:#071321 !important;\n  background-image:none !important;\n  border-color:rgba(22,139,255,.3) !important;\n  color:#E4FFF0 !important;\n}\n\n/* Custom dashboard panels */\n.skytrack-card, .skytrack-panel, .skytrack-hero,\n.skytrack-radar, .hero-panel, .hero-card,\n.command-panel, .dashboard-card, .metric-card,\n.status-card, .panel, .card {\n  background-color:#020408 !important;\n  background-image:none !important;\n  border-color:rgba(22,139,255,.35) !important;\n  color:#E4FFF0 !important;\n}\n\n/* Emerald grid */\n.skytrack-hero, .skytrack-radar, .hero-panel,\n.command-panel {\n  background-color:#020408 !important;\n  background-image:\n    linear-gradient(rgba(22,139,255,.055) 1px, transparent 1px),\n    linear-gradient(90deg, rgba(22,139,255,.055) 1px, transparent 1px) !important;\n  background-size:30px 30px !important;\n}\n\n/* Typography */\nh1,h2,h3,h4,h5,h6 {\n  color:#E4FFF0 !important;\n}\n\n/* Metrics */\n[data-testid="stMetricValue"],\n[data-testid="stMetricValue"] div {\n  color:#42D9FF !important;\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button,\nbutton {\n  background:#08182A !important;\n  color:#42D9FF !important;\n  border-color:#168BFF !important;\n}\n\n/* Inputs */\ninput,textarea,\n[data-baseweb="select"] > div,\n[data-baseweb="input"] > div {\n  background:#020408 !important;\n  color:#E4FFF0 !important;\n  border-color:rgba(22,139,255,.35) !important;\n}\n\n/* Tabs and links */\na { color:#42D9FF !important; }\n[aria-selected="true"] {\n  color:#42D9FF !important;\n  border-color:#168BFF !important;\n}\n\n/* Chart surfaces */\n[data-testid="stPlotlyChart"],\n[data-testid="stVideo"],\n[data-testid="stImage"] {\n  background:#000000 !important;\n  border-color:rgba(22,139,255,.3) !important;\n}\n\n/* Scrollbar */\n::-webkit-scrollbar-track { background:#000000; }\n::-webkit-scrollbar-thumb {\n  background:#1264C5;\n  border-radius:10px;\n}\n</style>\n', unsafe_allow_html=True)


# SKYTRACK_LOGO_EMERALD_FIX_V5
import streamlit as st

st.markdown('\n<style>\n\n/* SKYTRACK LOGO — FORCE ALL TEXT TO EMERALD */\n[data-testid="stSidebar"] h1,\n[data-testid="stSidebar"] h1 span,\n[data-testid="stSidebar"] h1 strong,\n[data-testid="stSidebar"] h1 em,\n[data-testid="stSidebar"] h2,\n[data-testid="stSidebar"] .skytrack-logo,\n[data-testid="stSidebar"] .skytrack-logo span,\n[data-testid="stSidebar"] #skytrack-logo,\n[data-testid="stSidebar"] #skytrack-logo span {\n    color: #168BFF !important;\n    -webkit-text-fill-color: #168BFF !important;\n    background: none !important;\n    background-image: none !important;\n    text-shadow: 0 0 12px rgba(22,139,255,0.22) !important;\n}\n\n/* Specifically target the AI part of the brand */\n[data-testid="stSidebar"] h1 span,\n[data-testid="stSidebar"] .skytrack-logo span,\n[data-testid="stSidebar"] #skytrack-logo span {\n    color: #168BFF !important;\n    -webkit-text-fill-color: #168BFF !important;\n}\n\n/* Keep sidebar background pure black */\n[data-testid="stSidebar"],\n[data-testid="stSidebar"] > div {\n    background: #000000 !important;\n}\n\n</style>\n', unsafe_allow_html=True)


# SKYTRACK_PRECISE_EMERALD_COMPONENT_FIX_V6
import streamlit as st

st.markdown('\n<style>\n\n/* =========================================\n   HERO BADGES — BLACK + EMERALD\n========================================= */\n\n.skytrack-command-status .skytrack-chip {\n    background: #020408 !important;\n    background-image: none !important;\n    color: #42D9FF !important;\n    border: 1px solid rgba(22,139,255,.55) !important;\n    box-shadow: 0 0 10px rgba(22,139,255,.06) !important;\n}\n\n.skytrack-command-status .skytrack-chip-green {\n    background: rgba(22,139,255,.10) !important;\n    color: #42D9FF !important;\n    border-color: #168BFF !important;\n}\n\n/* =========================================\n   SIDEBAR — ALL STATUS DOTS GREEN\n========================================= */\n\n[data-testid="stSidebar"] span[style*="ffc979"],\n[data-testid="stSidebar"] span[style*="FFC979"] {\n    color: #168BFF !important;\n}\n\n/* =========================================\n   RADIO NAVIGATION — GREEN ACTIVE INDICATOR\n========================================= */\n\n[data-testid="stSidebar"] [role="radiogroup"] {\n    accent-color: #168BFF !important;\n}\n\n[data-testid="stSidebar"] [role="radio"] {\n    accent-color: #168BFF !important;\n}\n\n[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {\n    color: #42D9FF !important;\n}\n\n[data-testid="stSidebar"] [role="radio"][aria-checked="true"] svg {\n    color: #168BFF !important;\n    fill: #168BFF !important;\n    stroke: #168BFF !important;\n}\n\n[data-testid="stSidebar"] [role="radio"][aria-checked="true"] svg circle {\n    fill: #168BFF !important;\n    stroke: #168BFF !important;\n}\n\n/* Radio labels */\n[data-testid="stSidebar"] [role="radiogroup"] label {\n    accent-color: #168BFF !important;\n}\n\n/* General active controls */\ninput[type="radio"] {\n    accent-color: #168BFF !important;\n}\n\n/* =========================================\n   FORCE BLACK INTERIOR / NO BLUE\n========================================= */\n\n.skytrack-command-header,\n.skytrack-command-status {\n    background-color: transparent !important;\n}\n\n.skytrack-command-status .skytrack-chip {\n    -webkit-text-fill-color: #42D9FF !important;\n}\n\n</style>\n', unsafe_allow_html=True)


# SKYTRACK_UNIFIED_ELECTRIC_BLUE_DESIGN_V1
import streamlit as st

st.markdown('\n<style>\n\n/* ============================================\n   SKYTRACK — BLACK + ELECTRIC BLUE\n============================================ */\n\n:root {\n    --sky-bg: #020408;\n    --sky-panel: #071321;\n    --sky-panel-2: #0A1929;\n    --sky-panel-3: #0D2035;\n    --sky-blue: #168BFF;\n    --sky-cyan: #42D9FF;\n    --sky-blue-dark: #1264C5;\n    --sky-text: #EAF7FF;\n    --sky-muted: #91AFC8;\n    --sky-border: rgba(22,139,255,.35);\n    --sky-glow: 0 0 18px rgba(22,139,255,.12);\n}\n\n/* Main application */\nhtml, body,\n[data-testid="stApp"],\n[data-testid="stAppViewContainer"],\n[data-testid="stMain"],\n[data-testid="stMainBlockContainer"],\n.main,\n.block-container {\n    background: #020408 !important;\n    color: var(--sky-text) !important;\n}\n\n/* Sidebar */\n[data-testid="stSidebar"],\n[data-testid="stSidebar"] > div,\n[data-testid="stSidebarContent"] {\n    background: #020408 !important;\n    color: var(--sky-text) !important;\n    border-right: 1px solid rgba(22,139,255,.35) !important;\n}\n\n/* Main panels and cards */\n[data-testid="stMetric"],\n[data-testid="stVerticalBlockBorderWrapper"],\n[data-testid="stContainer"],\n[data-testid="stExpander"],\n[data-testid="stForm"],\n[data-testid="stPopover"],\n[data-testid="stAlert"] {\n    background-color: #071321 !important;\n    border-color: var(--sky-border) !important;\n    color: var(--sky-text) !important;\n}\n\n/* Custom SkyTrack components */\n.skytrack-command-header,\n.skytrack-command-panel,\n.skytrack-card,\n.skytrack-panel,\n.skytrack-radar,\n.skytrack-console,\n.skytrack-metric,\n.skytrack-status,\n.skytrack-widget,\n.skytrack-hero,\n.hero-panel,\n.hero-card,\n.command-panel,\n.dashboard-card,\n.metric-card,\n.status-card,\n.arch-wrap,\n.arch-node,\n.arch-branch {\n    color: var(--sky-text) !important;\n    border-color: var(--sky-border) !important;\n}\n\n/* Hero and command panels */\n.skytrack-command-header,\n.skytrack-hero,\n.hero-panel {\n    background-color: #071321 !important;\n    background-image:\n        radial-gradient(\n            circle at 85% 0%,\n            rgba(22,139,255,.16),\n            transparent 38%\n        ) !important;\n}\n\n/* Architecture diagram */\n.arch-wrap {\n    background: #020408 !important;\n    border: 1px solid var(--sky-border) !important;\n}\n\n.arch-node {\n    background: #071321 !important;\n    border: 1px solid rgba(22,139,255,.45) !important;\n    box-shadow: var(--sky-glow) !important;\n}\n\n.arch-node.primary {\n    background: #0A1B30 !important;\n    border-color: #168BFF !important;\n}\n\n.arch-node.output,\n.arch-branch {\n    background: #071321 !important;\n    border-color: rgba(22,139,255,.5) !important;\n}\n\n.arch-kicker,\n.arch-node .num,\n.arch-section,\n.arch-arrow,\n.arch-branch-title {\n    color: #42D9FF !important;\n}\n\n.arch-node .name,\n.arch-heading {\n    color: #EAF7FF !important;\n}\n\n.arch-node .desc,\n.arch-subtitle,\n.arch-branch-text {\n    color: #A6C2D8 !important;\n}\n\n/* Hero badges */\n.skytrack-command-status .skytrack-chip {\n    background: #071321 !important;\n    color: #42D9FF !important;\n    border: 1px solid rgba(22,139,255,.6) !important;\n}\n\n.skytrack-command-status .skytrack-chip-green {\n    background: rgba(22,139,255,.12) !important;\n    color: #42D9FF !important;\n    border-color: #168BFF !important;\n}\n\n/* Headings */\nh1, h2, h3, h4, h5, h6 {\n    color: #EAF7FF !important;\n}\n\n/* Metrics */\n[data-testid="stMetric"] {\n    background: #071321 !important;\n    border: 1px solid var(--sky-border) !important;\n    border-radius: 12px !important;\n    box-shadow: var(--sky-glow) !important;\n}\n\n[data-testid="stMetricLabel"],\n[data-testid="stMetricLabel"] p {\n    color: #91AFC8 !important;\n}\n\n[data-testid="stMetricValue"],\n[data-testid="stMetricValue"] div {\n    color: #42D9FF !important;\n}\n\n/* Buttons */\n.stButton > button,\n.stDownloadButton > button,\n[data-testid="stFormSubmitButton"] button {\n    background: #08182A !important;\n    color: #42D9FF !important;\n    border: 1px solid #168BFF !important;\n    border-radius: 8px !important;\n}\n\n.stButton > button:hover,\n.stDownloadButton > button:hover,\n[data-testid="stFormSubmitButton"] button:hover {\n    background: #102B48 !important;\n    color: #FFFFFF !important;\n    box-shadow: 0 0 16px rgba(22,139,255,.3) !important;\n}\n\n/* Inputs */\n.stTextInput input,\n.stTextArea textarea,\n.stNumberInput input,\n[data-baseweb="select"] > div,\n[data-baseweb="input"] > div,\n[data-baseweb="textarea"] {\n    background: #020812 !important;\n    color: #EAF7FF !important;\n    border-color: rgba(22,139,255,.45) !important;\n}\n\n/* Dropdown menus */\n[data-baseweb="popover"],\n[data-baseweb="menu"],\n[data-baseweb="select"] ul,\n[role="listbox"] {\n    background: #071321 !important;\n    color: #EAF7FF !important;\n}\n\n/* Navigation */\n[data-testid="stSidebar"] [role="radiogroup"] label {\n    color: #D9F3FF !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"] label:hover {\n    background: #0B2038 !important;\n}\n\n[data-testid="stSidebar"] [role="radiogroup"],\n[data-testid="stSidebar"] [role="radio"],\ninput[type="radio"] {\n    accent-color: #168BFF !important;\n}\n\n[data-testid="stSidebar"] [role="radio"][aria-checked="true"] {\n    color: #42D9FF !important;\n}\n\n/* Tabs */\n.stTabs [data-baseweb="tab-list"] {\n    background: #020408 !important;\n    border-bottom: 1px solid var(--sky-border) !important;\n}\n\n.stTabs [data-baseweb="tab"] {\n    color: #91AFC8 !important;\n}\n\n.stTabs [aria-selected="true"] {\n    color: #42D9FF !important;\n    border-bottom-color: #168BFF !important;\n}\n\n/* Tables */\n[data-testid="stDataFrame"],\n[data-testid="stTable"] {\n    background: #071321 !important;\n    border: 1px solid var(--sky-border) !important;\n}\n\n/* Charts and media */\n[data-testid="stPlotlyChart"],\n[data-testid="stVegaLiteChart"],\n[data-testid="stDeckGlChart"],\n[data-testid="stImage"],\n[data-testid="stVideo"] {\n    background: #020408 !important;\n    border: 1px solid rgba(22,139,255,.35) !important;\n    border-radius: 10px !important;\n}\n\n/* Links */\na {\n    color: #42D9FF !important;\n}\n\na:hover {\n    color: #FFFFFF !important;\n}\n\n/* Dividers */\nhr {\n    border-color: rgba(22,139,255,.3) !important;\n}\n\n/* Status indicators */\n.skytrack-command-status .skytrack-chip-green,\n.arch-node .tag {\n    color: #42D9FF !important;\n}\n\n/* Scrollbars */\n::-webkit-scrollbar-track {\n    background: #020408;\n}\n\n::-webkit-scrollbar-thumb {\n    background: #1264C5;\n    border-radius: 10px;\n}\n\n::-webkit-scrollbar-thumb:hover {\n    background: #168BFF;\n}\n\n/* Focus */\n*:focus {\n    outline-color: #42D9FF !important;\n}\n\n</style>\n', unsafe_allow_html=True)

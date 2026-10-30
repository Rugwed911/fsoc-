
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT = Path("/content/drive/MyDrive/FSOC_SkyTrack_AI")
ENGINE = PROJECT / "ai_agent" / "copilot_engine.py"
VIDEO = PROJECT / "results/videos/validated_c18_airborne_demo_h264.mp4"
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
    --green: #74edbd;
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
          <span style="color:#74edbd">●</span> Local dashboard operational
        </div>
        <div style="margin:12px 0;color:#d5e2ee;font-size:.82rem">
          <span style="color:#74edbd">●</span> Evidence files imported
        </div>
        <div style="margin:12px 0;color:#d5e2ee;font-size:.82rem">
          <span style="color:#ffc979">●</span> Nemotron link pending
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
if page == "Mission Overview":
    st.markdown('<div class="sky-eyebrow">AIRBORNE OPTICAL TRACKING / MISSION CONSOLE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sky-title">MISSION<br>CONTROL</div>', unsafe_allow_html=True)

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
        if VIDEO.exists():
            st.video(str(VIDEO))
        else:
            st.warning("The recorded demonstration video was not found.")

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
        st.caption("• Nemotron integration not connected")

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
        metric_card("AI ENGINEER", "LOCAL", "Evidence-based diagnostics; no LLM yet")

# ---------------- OPTICAL TRACKING LAB ----------------
elif page == "Optical Tracking Lab":
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
        ENGINE STATUS / LOCAL ANALYSIS
      </div>
      <div style="font-size:23px;font-weight:750;color:#e8f7ff;margin-top:8px">
        Mission Diagnostics Workstation
      </div>
      <div style="font-size:12px;line-height:1.7;color:#92aabc;margin-top:7px">
        Queries are analyzed against saved project evidence. This interface
        currently uses a deterministic local engine, not a connected LLM.
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
            nemotron_color = "#74edbd"
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

        # Render local-engine output with readable line breaks and a clear
        # warning panel. Escape text before inserting it into HTML.
        import html

        answer_text = str(result.get("answer", "No answer was returned."))
        answer_lines = answer_text.splitlines()

        normal_lines = []
        warning_lines = []

        for line in answer_lines:
            if "QUALITY FLAG:" in line:
                before, _, warning_text = line.partition("QUALITY FLAG:")
                if before.strip():
                    normal_lines.append(before.strip())
                warning_lines.append(warning_text.strip())
            else:
                normal_lines.append(line)

        if normal_lines:
            formatted_lines = []
            for line in normal_lines:
                escaped = html.escape(line)
                if line.startswith("- "):
                    formatted_lines.append(
                        f'<div style="padding:5px 0;border-bottom:1px solid #203341">'
                        f'<span style="color:#50d9eb">▸</span> {escaped[2:]}</div>'
                    )
                elif line.isupper() and len(line) < 60:
                    formatted_lines.append(
                        f'<div style="font:11px monospace;letter-spacing:1px;'
                        f'color:#50d9eb;margin:12px 0 6px">{escaped}</div>'
                    )
                else:
                    formatted_lines.append(
                        f'<div style="padding:3px 0;color:#d8e5ee">{escaped}</div>'
                    )

            st.markdown(
                '<div style="background:#0c1722;border:1px solid #25465a;'
                'border-radius:9px;padding:12px 16px;line-height:1.65">'
                + "".join(formatted_lines) +
                '</div>',
                unsafe_allow_html=True,
            )

        for warning_text in warning_lines:
            st.markdown(
                '<div style="margin-top:12px;padding:12px 14px;'
                'background:#211b10;border:1px solid #8b6b35;'
                'border-left:4px solid #e8bd67;border-radius:7px;'
                'color:#f1dba9;line-height:1.6">'
                '<strong>QUALITY FLAG / VERIFY BEFORE CLAIMING</strong><br>'
                + html.escape(warning_text) +
                '</div>',
                unsafe_allow_html=True,
            )

        warning = result.get("warning")
        if warning:
            st.info(warning)

        provider = result.get("provider", "Local engine")
        mode = result.get("mode", "mock")
        st.caption(f"Execution mode: {mode} · Provider: {provider}")

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
    st.markdown('<div class="sky-eyebrow">SYSTEM / 05</div>',
                unsafe_allow_html=True)
    st.title("System Architecture")
    if ARCH.exists():
        st.image(str(ARCH), use_container_width=True)
    else:
        st.warning("Architecture image not found.")

    st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    **Current project boundaries**
    - Airborne jet scene is a visualization.
    - The simulated optical beacon is the tracked reference.
    - Saved results are historical experimental evidence.
    - The dashboard does not claim a live aircraft feed.
    - Nemotron is available as a configurable Nebius integration.
    """)

# ---------------- ASSET REGISTRY ----------------
elif page == "Asset Registry":
    st.markdown('<div class="sky-eyebrow">INVENTORY / 06</div>',
                unsafe_allow_html=True)
    st.title("Asset Registry")

    inventory = [
        ("Beacon detector weights", MODEL),
        ("Recorded airborne demonstration", VIDEO),
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

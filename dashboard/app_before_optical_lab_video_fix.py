
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT = Path("/content/drive/MyDrive/FSOC_SkyTrack_AI")
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
    st.markdown('<div class="sky-eyebrow">ENGINEERING ASSISTANT / 04</div>',
                unsafe_allow_html=True)
    st.title("AI Mission Engineer")
    st.caption("Local evidence-aware prototype. A real Nemotron response is not connected.")

    st.markdown(
        """
        <div class="sky-panel">
          <div class="sky-panel-title">ENGINE STATUS</div>
          <span class="sky-warn">LOCAL RULE-BASED MODE</span>
          <p style="color:#9db2c4;margin-top:12px">
          Ask about detector error, trajectory RMSE, lock percentage, or inference latency.
          Responses use saved project evidence; they are not generated by an LLM.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    question = st.text_area(
        "Engineering query",
        placeholder="Example: Which trajectory has the highest RMSE, and what should I verify?",
        height=100,
    )

    quick = st.selectbox(
        "Quick analysis",
        [
            "Choose",
            "Summarize detector accuracy",
            "Compare trajectory RMSE",
            "Explain inference latency",
            "Flag unusual results",
        ],
    )

    if st.button("Run engineering analysis"):
        q = (question or "").strip().lower()
        if not q and quick == "Choose":
            st.warning("Enter a question or choose an analysis.")
        else:
            query = q if q else quick.lower()
            st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
            st.subheader("Analysis")

            if any(k in query for k in ["latency", "inference", "fps", "speed"]):
                if latency:
                    st.write(
                        f"Mean saved latency: **{latency['mean']:.2f} ms**; "
                        f"median: **{latency['median']:.2f} ms**."
                    )
                    st.write(
                        f"Reciprocal-of-mean latency: "
                        f"**{1000 / latency['mean']:.2f} inferences/s**."
                    )
                    st.warning(
                        "This is not measured end-to-end control-loop FPS."
                    )
                else:
                    st.warning("Latency metrics unavailable.")

            elif any(k in query for k in [
                "trajectory", "rmse", "lock", "path", "figure", "straight"
            ]):
                if trajectory is None or "RMSE (px)" not in trajectory.columns:
                    st.warning("Trajectory RMSE evidence is unavailable.")
                else:
                    temp = trajectory.copy()
                    temp["RMSE (px)"] = pd.to_numeric(
                        temp["RMSE (px)"], errors="coerce"
                    )
                    valid = temp.dropna(subset=["RMSE (px)"])

                    st.dataframe(
                        valid.sort_values("RMSE (px)", ascending=False),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if not valid.empty:
                        worst = valid.loc[valid["RMSE (px)"].idxmax()]
                        best = valid.loc[valid["RMSE (px)"].idxmin()]
                        st.write(
                            f"Highest recorded RMSE: **{worst['Trajectory']}** "
                            f"({worst['RMSE (px)']:.4f} px)."
                        )
                        st.write(
                            f"Lowest recorded RMSE: **{best['Trajectory']}** "
                            f"({best['RMSE (px)']:.6f} px)."
                        )

                        if best["RMSE (px)"] < 0.01:
                            st.warning(
                                "The smallest RMSE is exceptionally low. "
                                "Validate configuration and source data before making "
                                "a broad performance claim."
                            )

            elif any(k in query for k in [
                "centroid", "accuracy", "detector", "yolo", "confidence"
            ]):
                if centroid:
                    st.write(
                        f"Mean centroid error: **{centroid['mean']:.3f} px**; "
                        f"RMSE: **{centroid['rmse']:.3f} px**; "
                        f"within 2 px: **{centroid['within_2px_percent']:.1f}%**."
                    )
                    st.caption(
                        "Centroid error is not the same as closed-loop camera error."
                    )
                if confidence:
                    st.write(
                        f"Mean detector confidence: **{confidence['mean']:.4f}**."
                    )

            elif any(k in query for k in ["unusual", "flag", "validate", "risk", "check"]):
                st.markdown("""
                - Verify the exceptionally low figure-eight RMSE.
                - Investigate the straight trajectory's high maximum error.
                - Separate detector centroid error from camera tracking error.
                - Distinguish reciprocal latency from measured system FPS.
                - Compare only compatible experimental conditions.
                """)

            else:
                st.info(
                    "Try a question about centroid accuracy, trajectory RMSE, "
                    "lock percentage, or inference latency."
                )

    st.markdown("<div class='sky-divider'></div>", unsafe_allow_html=True)
    st.caption("Nebius / NVIDIA Nemotron connection: pending service access.")

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
    - Nemotron integration remains a future connection.
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

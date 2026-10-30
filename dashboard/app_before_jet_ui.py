
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT = Path("/content/drive/MyDrive/FSOC_SkyTrack_AI")

VIDEO = PROJECT / "results/videos/validated_c18_airborne_demo.mp4"
MODEL = PROJECT / "simulation/models/beacon_detector_best.pt"
ARCHITECTURE = PROJECT / "results/figures/original_system_architecture.png"

DATASETS = {
    "Validated C18 metrics": PROJECT / "experiments/baselines/validated_c18_metrics.csv",
    "YOLO centroid benchmark": PROJECT / "experiments/baselines/yolo_centroid_benchmark.csv",
    "Multi-trajectory validation": PROJECT / "experiments/baselines/multi_trajectory_validation.csv",
}

st.set_page_config(
    page_title="FSOC SkyTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp {
    background: #080b12;
    color: #e8edf5;
}
[data-testid="stSidebar"] {
    background: #0d111b;
    border-right: 1px solid #202a3a;
}
.block-container {
    padding-top: 1.8rem;
    max-width: 1500px;
}
[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #273449;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricLabel"] {
    color: #9caec4;
}
h1, h2, h3 {
    color: #f3f6fc;
}
div[data-testid="stDataFrame"] {
    border: 1px solid #273449;
    border-radius: 8px;
}
.small-note {
    color: #9caec4;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


def load_csv(path):
    if not path.exists():
        return None, f"File not found: {path.name}"
    try:
        df = pd.read_csv(path)
        if df.empty and len(df.columns) == 0:
            return df, "CSV exists but contains no readable columns."
        return df, None
    except Exception as exc:
        return None, str(exc)


def numeric_columns(df):
    if df is None:
        return []
    return list(df.select_dtypes(include="number").columns)


def choose_column(columns, keywords):
    for keyword in keywords:
        for col in columns:
            if keyword in str(col).lower():
                return col
    return None


def display_dataframe(df):
    st.dataframe(df, use_container_width=True, hide_index=True)


def format_value(value):
    try:
        value = float(value)
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        return f"{value:.3f}"
    except Exception:
        return str(value)


st.sidebar.title("FSOC SkyTrack AI")
st.sidebar.caption("Airborne optical tracking research")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Overview",
        "Tracking Evidence",
        "Trajectory Analysis",
        "System Architecture",
        "AI Engineering Copilot",
        "Project Assets",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**System status**")
st.sidebar.write("🟢 Local dashboard")
st.sidebar.write("🟢 Imported project assets")
st.sidebar.write("🟡 Nebius / Nemotron integration pending")
st.sidebar.caption(
    "Recorded demonstration and saved experiment data are shown as such. "
    "This dashboard does not claim live inference."
)

# Load evidence once per Streamlit cache cycle.
frames = {}
errors = {}
for label, path in DATASETS.items():
    frames[label], errors[label] = load_csv(path)

# ---------------- MISSION OVERVIEW ----------------
if page == "Mission Overview":
    st.title("FSOC SkyTrack AI")
    st.markdown(
        "### AI-assisted coarse alignment for airborne Free-Space Optical Communication"
    )
    st.markdown(
        "A virtual-camera tracking research platform for locating and following "
        "an optical beacon in an airborne scene."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Detector weights", "Available" if MODEL.exists() else "Missing")
    c2.metric("Recorded demo", "Available" if VIDEO.exists() else "Missing")
    c3.metric(
        "Experiment datasets",
        f"{sum(df is not None for df in frames.values())}/{len(DATASETS)}",
    )

    st.divider()
    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Airborne demonstration")
        if VIDEO.exists():
            st.video(str(VIDEO))
            st.caption(
                "Previously recorded project demonstration. Playback is not "
                "a live model-inference session."
            )
        else:
            st.warning("The imported demonstration video was not found.")

    with right:
        st.subheader("Project capabilities")
        st.markdown("""
        - Virtual camera and optical-beacon tracking
        - YOLO-based beacon detection model artifact
        - Tracking and trajectory evaluation evidence
        - Experiment comparison and diagnostic views
        - Planned engineering copilot using NVIDIA Nemotron
        """)
        st.info(
            "The fighter-plane scene is the airborne visualization. "
            "The tracked reference is the simulated optical beacon, not the aircraft."
        )

    st.divider()
    st.subheader("Available experiment files")
    for label, path in DATASETS.items():
        df = frames[label]
        if df is not None:
            st.write(f"✅ **{label}** — {len(df)} rows, {len(df.columns)} columns")
        else:
            st.write(f"⚠️ **{label}** — {errors[label]}")

# ---------------- TRACKING EVIDENCE ----------------
elif page == "Tracking Evidence":
    st.title("Tracking Evidence")
    st.caption("Inspect the imported CSV evidence without changing the source files.")

    label = st.selectbox("Dataset", list(DATASETS.keys()))
    df = frames[label]

    if df is None:
        st.error(errors[label])
    elif df.empty:
        st.warning("This CSV has no data rows.")
        st.write("Columns:", list(df.columns))
    else:
        st.write(f"**Rows:** {len(df)} · **Columns:** {len(df.columns)}")
        display_dataframe(df.head(100))

        nums = numeric_columns(df)
        if nums:
            st.subheader("Numeric column explorer")
            y_col = st.selectbox("Numeric field", nums)
            values = pd.to_numeric(df[y_col], errors="coerce").dropna()

            if not values.empty:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Count", f"{len(values):,}")
                m2.metric("Mean", format_value(values.mean()))
                m3.metric("Minimum", format_value(values.min()))
                m4.metric("Maximum", format_value(values.max()))

                chart_df = df[[y_col]].copy()
                chart_df[y_col] = pd.to_numeric(chart_df[y_col], errors="coerce")
                chart_df = chart_df.dropna()
                chart_df["Row"] = range(len(chart_df))

                fig = px.line(
                    chart_df,
                    x="Row",
                    y=y_col,
                    title=f"{y_col} by row",
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="#080b12",
                    plot_bgcolor="#0d111b",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns were detected in this dataset.")

        st.download_button(
            "Download selected CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=DATASETS[label].name,
            mime="text/csv",
        )

# ---------------- TRAJECTORY ANALYSIS ----------------
elif page == "Trajectory Analysis":
    st.title("Trajectory Analysis")
    st.caption("Explore the imported validation data; values are not regenerated here.")

    df = frames["Multi-trajectory validation"]

    if df is None:
        st.error(errors["Multi-trajectory validation"])
    elif df.empty:
        st.warning("The trajectory CSV is empty.")
    else:
        display_dataframe(df)

        cols = list(df.columns)
        category_col = choose_column(
            cols, ["trajectory", "scenario", "path", "motion", "test", "name"]
        )
        metric_col = choose_column(
            numeric_columns(df),
            ["lock", "success", "accuracy", "error", "rmse", "fps", "confidence"],
        )

        if category_col and metric_col:
            st.subheader("Compare a recorded metric")
            available_metrics = numeric_columns(df)
            selected_metric = st.selectbox(
                "Metric column",
                available_metrics,
                index=available_metrics.index(metric_col),
            )

            plot_df = df[[category_col, selected_metric]].copy()
            plot_df[selected_metric] = pd.to_numeric(
                plot_df[selected_metric], errors="coerce"
            )
            plot_df = plot_df.dropna()

            fig = px.bar(
                plot_df,
                x=category_col,
                y=selected_metric,
                title=f"{selected_metric} by {category_col}",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="#080b12",
                plot_bgcolor="#0d111b",
                xaxis_title="Recorded category",
                yaxis_title=selected_metric,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(
                "The app could not safely infer the category and metric columns. "
                "Use the table above to inspect the original column names."
            )

# ---------------- SYSTEM ARCHITECTURE ----------------
elif page == "System Architecture":
    st.title("System Architecture")
    if ARCHITECTURE.exists():
        st.image(str(ARCHITECTURE), use_container_width=True)
    else:
        st.warning("Original architecture image was not found.")

    st.markdown("""
    **Current scope**
    1. Simulated airborne scene and optical beacon
    2. Detector artifact and tracking experiment evidence
    3. Camera/control evaluation outputs
    4. Local dashboard for review and analysis
    5. Planned Nemotron engineering copilot, pending service access
    """)


# ---------------- AI ENGINEERING COPILOT ----------------
elif page == "AI Engineering Copilot":
    import json

    st.title("AI Engineering Copilot")
    st.caption("Evidence-aware local analysis — no external LLM call.")

    report_path = PROJECT / "results/reports/evidence_diagnostics.json"
    trajectory_path = PROJECT / "experiments/baselines/multi_trajectory_validation.csv"

    report = {}
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            st.error(f"Could not read diagnostic report: {exc}")

    metrics = report.get("metrics", {})
    centroid = metrics.get("centroid_error_px", {})
    latency = metrics.get("inference_latency_ms", {})
    confidence = metrics.get("confidence", {})

    st.info(
        "This prototype analyzes saved evidence using deterministic rules. "
        "It is not yet powered by NVIDIA Nemotron."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Mean centroid error",
        f"{centroid['mean']:.3f} px" if "mean" in centroid else "Unavailable"
    )
    c2.metric(
        "Within 2 px",
        f"{centroid['within_2px_percent']:.1f}%"
        if "within_2px_percent" in centroid else "Unavailable"
    )
    c3.metric(
        "Mean inference latency",
        f"{latency['mean']:.1f} ms" if "mean" in latency else "Unavailable"
    )
    c4.metric(
        "Mean confidence",
        f"{confidence['mean']:.3f}" if "mean" in confidence else "Unavailable"
    )

    st.subheader("Ask about the recorded experiment")
    question = st.text_input(
        "Engineering question",
        placeholder="e.g. Which trajectory has the highest RMSE?"
    )

    quick = st.selectbox(
        "Or choose an analysis",
        [
            "Choose an analysis",
            "Summarize detector accuracy",
            "Compare trajectory tracking",
            "Explain inference latency",
            "Flag results that need validation",
        ]
    )

    if st.button("Analyze saved evidence", type="primary"):
        q = question.strip().lower()
        query = q if q else quick.lower()

        if not q and quick == "Choose an analysis":
            st.warning("Enter a question or select an analysis.")
        elif any(k in query for k in [
            "latency", "inference", "fps", "speed"
        ]):
            if latency:
                st.write(
                    f"Mean recorded inference latency: "
                    f"**{latency['mean']:.2f} ms**; median: "
                    f"**{latency['median']:.2f} ms**; maximum: "
                    f"**{latency['maximum']:.2f} ms**."
                )
                if latency["mean"] > 0:
                    st.write(
                        f"Reciprocal-of-mean latency: "
                        f"**{1000 / latency['mean']:.2f} inferences/s**."
                    )
                st.warning(
                    "This is derived from detector latency, not measured "
                    "end-to-end camera/control-loop FPS."
                )
            else:
                st.warning("No latency metrics are present in the report.")

        elif any(k in query for k in [
            "trajectory", "rmse", "lock", "path", "figure8",
            "straight", "circular", "spiral", "random"
        ]):
            if not trajectory_path.exists():
                st.error("Trajectory validation CSV was not found.")
            else:
                tdf = pd.read_csv(trajectory_path)
                required = [
                    "Trajectory", "Average Error (px)", "RMSE (px)",
                    "Maximum Error (px)", "≤10 px Lock (%)"
                ]
                missing = [c for c in required if c not in tdf.columns]

                if missing:
                    st.error("Missing expected columns: " + ", ".join(missing))
                    st.dataframe(tdf, use_container_width=True)
                else:
                    for col in required[1:]:
                        tdf[col] = pd.to_numeric(tdf[col], errors="coerce")

                    st.dataframe(
                        tdf[required].sort_values("RMSE (px)", ascending=False),
                        use_container_width=True,
                        hide_index=True
                    )

                    valid = tdf.dropna(subset=["RMSE (px)"])
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

                    fig_rows = tdf[
                        tdf["Trajectory"].astype(str).str.lower() == "figure8"
                    ]
                    if not fig_rows.empty:
                        value = fig_rows.iloc[0]["RMSE (px)"]
                        if pd.notna(value) and value < 0.01:
                            st.warning(
                                "The figure-eight RMSE is exceptionally small "
                                "relative to the other recorded trajectories. "
                                "Verify its configuration, units, and source data "
                                "before making a general performance claim."
                            )

        elif any(k in query for k in [
            "accuracy", "centroid", "confidence", "detector", "yolo"
        ]):
            if centroid:
                st.write(
                    f"Across **{centroid['valid_frames']} valid frames**, "
                    f"mean centroid error was **{centroid['mean']:.3f} px**, "
                    f"median was **{centroid['median']:.3f} px**, and RMSE was "
                    f"**{centroid['rmse']:.3f} px**."
                )
                st.write(
                    f"Within 1 px: **{centroid['within_1px_percent']:.1f}%**. "
                    f"Within 2 px: **{centroid['within_2px_percent']:.1f}%**."
                )
                st.caption(
                    "Centroid error compares detector coordinates with saved "
                    "ground truth. It is not closed-loop camera tracking error."
                )
            else:
                st.warning("Centroid metrics are unavailable. Run Step 10 first.")

            if confidence:
                st.write(
                    f"Mean saved detector confidence: **{confidence['mean']:.4f}**."
                )

        elif any(k in query for k in [
            "validate", "unusual", "flag", "risk", "check", "reliable"
        ]):
            st.markdown("""
            **Validation checklist**
            1. Verify the exceptionally low figure-eight RMSE.
            2. Investigate the straight path's high maximum error despite high lock.
            3. Keep detector centroid error separate from closed-loop tracking error.
            4. Distinguish reciprocal latency from measured end-to-end FPS.
            5. Compare runs only when test conditions and units are compatible.
            """)

        else:
            st.write(
                "Try asking about detector centroid accuracy, trajectory RMSE, "
                "lock percentage, inference latency, or validation risks."
            )

    st.caption(
        "Local evidence-based prototype. Nemotron integration is pending; "
        "this panel does not run new experiments."
    )

# ---------------- ASSET INVENTORY ----------------
elif page == "Project Assets":
    st.title("Project Assets")
    paths = [
        ("Beacon detector weights", MODEL),
        ("Recorded airborne demo", VIDEO),
        ("Original system architecture", ARCHITECTURE),
        ("Asset import manifest", PROJECT / "asset_import_manifest.csv"),
    ] + [(label, path) for label, path in DATASETS.items()]

    inventory = []
    for label, path in paths:
        inventory.append({
            "Asset": label,
            "Status": "Available" if path.is_file() else "Missing",
            "Size (MB)": round(path.stat().st_size / (1024 * 1024), 3)
                if path.is_file() else None,
            "Path": str(path),
        })

    display_dataframe(pd.DataFrame(inventory))
    st.caption("File presence is verified at dashboard runtime; model quality is not retested.")

st.divider()
st.caption(
    "FSOC SkyTrack AI · Research prototype · "
    "Saved evidence is distinguished from live inference."
)

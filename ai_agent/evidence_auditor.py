from pathlib import Path
import json
import numpy as np
import pandas as pd

def run_evidence_audit(project_root, atol=1e-8, rtol=1e-6):
    root = Path(project_root)
    report_path = root / "results/reports/evidence_diagnostics.json"
    centroid_path = root / "experiments/baselines/validated_c18_metrics.csv"
    trajectory_path = root / "experiments/baselines/multi_trajectory_validation.csv"

    for path in (report_path, centroid_path, trajectory_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required evidence file missing: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(centroid_path)
    trajectory = pd.read_csv(trajectory_path)
    findings = []

    def add(severity, code, message, details=None):
        findings.append({
            "severity": severity,
            "code": code,
            "message": message,
            "details": details or {},
        })

    # Recompute the saved detector metric from its source coordinates.
    coord_cols = ["True FPA X", "True FPA Y", "YOLO X", "YOLO Y"]
    if all(col in frame.columns for col in coord_cols):
        coords = frame[coord_cols].apply(pd.to_numeric, errors="coerce")
        valid = coords.notna().all(axis=1)
        dx = coords.loc[valid, "YOLO X"] - coords.loc[valid, "True FPA X"]
        dy = coords.loc[valid, "YOLO Y"] - coords.loc[valid, "True FPA Y"]
        error = np.hypot(dx, dy)
        saved = report.get("metrics", {}).get("centroid_error_px", {})

        computed = {
            "valid_frames": int(len(error)),
            "mean": float(error.mean()) if len(error) else None,
            "median": float(error.median()) if len(error) else None,
            "rmse": float(np.sqrt(np.mean(error ** 2))) if len(error) else None,
            "maximum": float(error.max()) if len(error) else None,
            "within_1px_percent": float((error <= 1).mean() * 100) if len(error) else None,
            "within_2px_percent": float((error <= 2).mean() * 100) if len(error) else None,
            "within_5px_percent": float((error <= 5).mean() * 100) if len(error) else None,
        }

        mismatches = {}
        for key, value in computed.items():
            reference = saved.get(key)
            if value is None or reference is None:
                mismatches[key] = {"saved": reference, "computed": value}
            elif not np.isclose(float(reference), float(value), atol=atol, rtol=rtol):
                mismatches[key] = {"saved": reference, "computed": value}

        add(
            "PASS" if not mismatches else "REVIEW",
            "CENTROID_METRICS_MATCH",
            "Saved detector metrics match coordinate-derived values."
            if not mismatches else "Some saved detector metrics differ from coordinate-derived values.",
            {"saved": saved, "computed": computed, "mismatches": mismatches},
        )
    else:
        add("REVIEW", "CENTROID_SCHEMA", "Required coordinate columns are missing.",
            {"required": coord_cols, "available": list(frame.columns)})

    # Row-count and duplicate checks.
    if report.get("frame_rows") == len(frame):
        add("PASS", "FRAME_ROW_COUNT", "Diagnostic report frame count matches the source CSV.",
            {"report": report.get("frame_rows"), "csv": len(frame)})
    else:
        add("REVIEW", "FRAME_ROW_COUNT", "Frame row counts differ.",
            {"report": report.get("frame_rows"), "csv": len(frame)})

    if report.get("trajectory_rows") == len(trajectory):
        add("PASS", "TRAJECTORY_ROW_COUNT", "Trajectory row count matches the source CSV.",
            {"report": report.get("trajectory_rows"), "csv": len(trajectory)})
    else:
        add("REVIEW", "TRAJECTORY_ROW_COUNT", "Trajectory row counts differ.",
            {"report": report.get("trajectory_rows"), "csv": len(trajectory)})

    duplicate_count = int(trajectory.duplicated().sum())
    add(
        "REVIEW" if duplicate_count else "PASS",
        "EXACT_DUPLICATE_ROWS",
        f"Found {duplicate_count} exact duplicate trajectory rows.",
        {"count": duplicate_count},
    )

    # Compare report records with CSV records row by row.
    report_records = pd.DataFrame(report.get("trajectory_records", []))
    same = (
        list(report_records.columns) == list(trajectory.columns)
        and len(report_records) == len(trajectory)
    )
    if same:
        for col in trajectory.columns:
            a, b = trajectory[col], report_records[col]
            if pd.api.types.is_numeric_dtype(a):
                same = same and np.allclose(
                    pd.to_numeric(a, errors="coerce").to_numpy(dtype=float),
                    pd.to_numeric(b, errors="coerce").to_numpy(dtype=float),
                    atol=atol, rtol=rtol, equal_nan=True,
                )
            else:
                same = same and np.array_equal(
                    a.fillna("<NA>").astype(str).to_numpy(),
                    b.fillna("<NA>").astype(str).to_numpy(),
                )
    add(
        "PASS" if same else "REVIEW",
        "TRAJECTORY_REPORT_MATCH",
        "Trajectory report records match the source CSV."
        if same else "Trajectory report records differ from the source CSV.",
        {"csv_rows": len(trajectory), "report_rows": len(report_records)},
    )

    # Flag unusual trajectory metrics for review, not as proven errors.
    if "Trajectory" in trajectory.columns and "RMSE (px)" in trajectory.columns:
        rmse = pd.to_numeric(trajectory["RMSE (px)"], errors="coerce")
        valid_rmse = rmse.dropna()
        if len(valid_rmse) >= 2:
            positive = valid_rmse[valid_rmse > 0]
            if len(positive) >= 2:
                median_rmse = float(positive.median())
                for idx in trajectory.index:
                    value = pd.to_numeric(
                        pd.Series([trajectory.loc[idx, "RMSE (px)"]]),
                        errors="coerce",
                    ).iloc[0]
                    if pd.notna(value) and value > 0 and value < median_rmse * 1e-3:
                        add(
                            "REVIEW", "UNUSUALLY_LOW_TRAJECTORY_RMSE",
                            "Trajectory RMSE is much smaller than the median positive RMSE; verify the raw logs.",
                            {"trajectory": str(trajectory.loc[idx, "Trajectory"]),
                             "rmse_px": float(value), "median_positive_rmse_px": median_rmse},
                        )

    if "Trajectory" in trajectory.columns and "Maximum Error (px)" in trajectory.columns:
        maximum = pd.to_numeric(trajectory["Maximum Error (px)"], errors="coerce")
        for idx in trajectory.index:
            value = maximum.loc[idx]
            if pd.notna(value) and value >= 50:
                add(
                    "REVIEW", "HIGH_MAXIMUM_TRACKING_ERROR",
                    "Large maximum tracking error warrants review against raw trajectory logs.",
                    {"trajectory": str(trajectory.loc[idx, "Trajectory"]),
                     "maximum_error_px": float(value)},
                )

    return {
        "audit_name": "SkyTrack AI deterministic evidence audit",
        "read_only": True,
        "source_files": [str(report_path), str(centroid_path), str(trajectory_path)],
        "finding_count": len(findings),
        "findings": findings,
    }


import json
import os
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd


PROJECT = Path("/content/drive/MyDrive/FSOC_SkyTrack_AI")
REPORT_PATH = PROJECT / "results/reports/evidence_diagnostics.json"
TRAJECTORY_PATH = PROJECT / "experiments/baselines/multi_trajectory_validation.csv"


def load_evidence():
    if not REPORT_PATH.is_file():
        raise FileNotFoundError(f"Diagnostic report missing: {REPORT_PATH}")
    if not TRAJECTORY_PATH.is_file():
        raise FileNotFoundError(f"Trajectory CSV missing: {TRAJECTORY_PATH}")

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    trajectory = pd.read_csv(TRAJECTORY_PATH)

    return {
        "diagnostic_report": report,
        "trajectory_records": trajectory.where(
            pd.notna(trajectory), None
        ).to_dict(orient="records"),
    }


def build_messages(question):
    evidence = load_evidence()

    system_prompt = (
        "You are the engineering analysis assistant for FSOC SkyTrack AI, "
        "a virtual-camera optical-beacon tracking research prototype. "
        "Use only the supplied evidence for claims about this project. "
        "Distinguish recorded facts from hypotheses and recommendations. "
        "Do not claim to have run experiments, changed controllers, or accessed "
        "live sensors. Detector centroid error is different from closed-loop "
        "tracking error. Reciprocal detector latency is not end-to-end control "
        "FPS. Flag suspicious or unusually small values for validation. "
        "Do not invent performance results. Provide a concise answer with "
        "sections: Finding, Evidence, Caveat, Next action."
    )

    user_payload = {
        "question": question,
        "evidence": evidence,
    }

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, ensure_ascii=False),
        },
    ]


def mock_response(question):
    """Generate a local, evidence-grounded response without an LLM."""
    evidence = load_evidence()
    report = evidence.get("diagnostic_report", {})
    metrics = report.get("metrics", {})
    centroid = metrics.get("centroid_error_px", {})
    confidence = metrics.get("confidence", {})
    latency = metrics.get("inference_latency_ms", {})
    trajectories = report.get("trajectory_records", [])

    q = (question or "").lower()
    lines = []

    def fmt(value, digits=2):
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "N/A"

    broad = any(x in q for x in [
        "summar", "overview", "overall", "evidence",
        "performance", "report", "validated", "results"
    ])
    latency_query = any(x in q for x in [
        "latency", "inference time", "fps", "speed"
    ])
    centroid_query = any(x in q for x in [
        "centroid", "detection accuracy", "detector error",
        "confidence", "yolo"
    ])
    trajectory_query = any(x in q for x in [
        "trajectory", "trajectories", "lock", "closed-loop",
        "closed loop", "tracking error", "figure", "spiral",
        "circular", "straight", "random"
    ])

    if broad:
        lines.append("LOCAL EVIDENCE SUMMARY")
        lines.append(
            f"Detector centroid error: mean {fmt(centroid.get('mean'))} px; "
            f"RMSE {fmt(centroid.get('rmse'))} px; "
            f"maximum {fmt(centroid.get('maximum'))} px; "
            f"{centroid.get('valid_frames', 'N/A')} valid frames."
        )
        lines.append(
            f"Mean detector confidence: {fmt(confidence.get('mean'), 3)}; "
            f"median: {fmt(confidence.get('median'), 3)}."
        )
        lines.append(
            f"Detector inference latency: mean {fmt(latency.get('mean'))} ms; "
            f"median {fmt(latency.get('median'))} ms."
        )
        lines.append("Saved trajectory results:")
        for row in trajectories:
            lines.append(
                f"- {row.get('Trajectory', 'Unknown')}: "
                f"RMSE {fmt(row.get('RMSE (px)'))} px; "
                f"maximum error {fmt(row.get('Maximum Error (px)'))} px; "
                f"≤10 px lock {fmt(row.get('≤10 px Lock (%)'))}%."
            )
        lines.append(
            "Caution: detector centroid error is different from closed-loop "
            "camera tracking error."
        )
        lines.append(
            "Reciprocal mean detector latency is not measured end-to-end FPS."
        )

    elif latency_query:
        lines.append("DETECTOR LATENCY")
        lines.append(f"Mean: {fmt(latency.get('mean'))} ms.")
        lines.append(f"Median: {fmt(latency.get('median'))} ms.")
        lines.append(
            f"Range: {fmt(latency.get('minimum'))}–"
            f"{fmt(latency.get('maximum'))} ms."
        )
        try:
            lines.append(
                f"Reciprocal of mean latency: "
                f"{1000 / float(latency['mean']):.2f} inferences/s, "
                "a derived value, not measured end-to-end FPS."
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            pass

    elif centroid_query:
        lines.append("DETECTOR CENTROID EVIDENCE")
        lines.append(f"Valid frames: {centroid.get('valid_frames', 'N/A')}.")
        lines.append(f"Mean error: {fmt(centroid.get('mean'))} px.")
        lines.append(f"Median error: {fmt(centroid.get('median'))} px.")
        lines.append(f"RMSE: {fmt(centroid.get('rmse'))} px.")
        lines.append(f"Maximum error: {fmt(centroid.get('maximum'))} px.")
        lines.append(
            f"Within 1 px: {fmt(centroid.get('within_1px_percent'))}%; "
            f"within 2 px: {fmt(centroid.get('within_2px_percent'))}%; "
            f"within 5 px: {fmt(centroid.get('within_5px_percent'))}%."
        )
        lines.append(
            "These are saved detector measurements, not proof of live "
            "aircraft detection or closed-loop tracking accuracy."
        )

    elif trajectory_query:
        requested = [
            name for name in ["straight", "circular", "figure8", "spiral", "random"]
            if name in q.replace("figure-eight", "figure8").replace("figure 8", "figure8")
        ]
        selected = [
            row for row in trajectories
            if not requested or str(row.get("Trajectory", "")).lower() in requested
        ]
        lines.append("SAVED TRAJECTORY VALIDATION")
        for row in selected:
            lines.append(
                f"{row.get('Trajectory', 'Unknown')}: "
                f"average error {fmt(row.get('Average Error (px)'))} px; "
                f"RMSE {fmt(row.get('RMSE (px)'))} px; "
                f"maximum error {fmt(row.get('Maximum Error (px)'))} px; "
                f"≤10 px lock {fmt(row.get('≤10 px Lock (%)'))}%."
            )
            if (
                str(row.get("Trajectory", "")).lower() == "figure8"
                and fmt(row.get("RMSE (px)")) != "N/A"
            ):
                try:
                    if float(row.get("RMSE (px)")) < 0.01:
                        lines.append(
                            "QUALITY FLAG: figure-eight RMSE is exceptionally "
                            "small. Verify the source data and calculation "
                            "before making a broad performance claim."
                        )
                except (TypeError, ValueError):
                    pass
        lines.append(
            "These are saved simulation results, not a live aircraft-tracking test."
        )

    else:
        lines.append(
            "I can analyze saved evidence. Ask about an overall summary, "
            "centroid accuracy, detector latency, trajectory lock, or the "
            "figure-eight result."
        )

    return {
        "mode": "mock",
        "provider": "local evidence-grounded analysis",
        "question": question,
        "answer": "\\n".join(lines),
        "evidence_files": [
            str(PROJECT / "results/reports/evidence_diagnostics.json"),
            str(PROJECT / "experiments/baselines/validated_c18_metrics.csv"),
            str(PROJECT / "experiments/baselines/multi_trajectory_validation.csv"),
        ],
        "warning": (
            "Local deterministic analysis of saved evidence; not a generative "
            "model response and not a live tracking test."
        ),
    }


def nebius_response(question):
    api_key = os.environ.get("NEBIUS_API_KEY", "").strip()
    base_url = os.environ.get("NEBIUS_BASE_URL", "").strip().rstrip("/")
    model = os.environ.get("NEBIUS_MODEL", "").strip()

    missing = [
        name for name, value in [
            ("NEBIUS_API_KEY", api_key),
            ("NEBIUS_BASE_URL", base_url),
            ("NEBIUS_MODEL", model),
        ] if not value
    ]
    if missing:
        raise RuntimeError(
            "Nebius configuration is incomplete. Set: " + ", ".join(missing)
        )

    messages = build_messages(question)
    endpoint = base_url if base_url.endswith("/chat/completions") else (
        base_url + "/chat/completions"
    )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 800,
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1500]
        raise RuntimeError(
            f"Nebius endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach the configured endpoint: {exc}") from exc

    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError(
            "Endpoint response did not contain a choices array. "
            "Check the configured API endpoint and response format."
        )

    answer = choices[0].get("message", {}).get("content", "")
    return {
        "mode": "nebius",
        "provider": "configured OpenAI-compatible endpoint",
        "model": model,
        "question": question,
        "answer": answer,
        "warning": (
            "Model output is generated text; verify recommendations against "
            "the source experiment evidence before acting."
        ),
    }


def ask_copilot(question, mode="mock"):
    if not question or not question.strip():
        raise ValueError("Enter an engineering question.")
    if mode == "mock":
        return mock_response(question.strip())
    if mode == "nebius":
        return nebius_response(question.strip())
    raise ValueError("mode must be 'mock' or 'nebius'")

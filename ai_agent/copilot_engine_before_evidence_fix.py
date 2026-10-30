
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
    evidence = load_evidence()
    report = evidence["diagnostic_report"]
    metrics = report.get("metrics", {})
    centroid = metrics.get("centroid_error_px", {})
    latency = metrics.get("inference_latency_ms", {})
    confidence = metrics.get("confidence", {})
    rows = evidence["trajectory_records"]

    q = question.lower()
    findings = []

    if any(word in q for word in ["latency", "fps", "inference", "speed"]):
        if latency:
            mean_ms = latency.get("mean")
            findings.append(
                f"Mean saved detector latency is {mean_ms:.2f} ms, "
                f"with median {latency.get('median', float('nan')):.2f} ms."
            )
            if mean_ms and mean_ms > 0:
                findings.append(
                    f"The reciprocal of mean latency is {1000 / mean_ms:.2f} "
                    "inferences/s, derived from the saved latency values."
                )
            findings.append(
                "This is not a measured end-to-end camera/control-loop FPS."
            )
        else:
            findings.append("No latency summary is available.")

    elif any(word in q for word in [
        "trajectory", "rmse", "lock", "figure", "straight", "path"
    ]):
        rmse_rows = [
            row for row in rows
            if row.get("RMSE (px)") is not None
        ]
        rmse_rows.sort(key=lambda row: float(row["RMSE (px)"]), reverse=True)

        if rmse_rows:
            worst = rmse_rows[0]
            best = rmse_rows[-1]
            findings.append(
                f"Highest recorded RMSE: {worst['Trajectory']} "
                f"({float(worst['RMSE (px)']):.4f} px)."
            )
            findings.append(
                f"Lowest recorded RMSE: {best['Trajectory']} "
                f"({float(best['RMSE (px)']):.6f} px)."
            )

        for row in rows:
            if (
                str(row.get("Trajectory", "")).lower() == "figure8"
                and row.get("RMSE (px)") is not None
                and float(row["RMSE (px)"]) < 0.01
            ):
                findings.append(
                    "The figure-eight RMSE is unusually small compared with "
                    "other paths; verify the test configuration and source data."
                )

    elif any(word in q for word in [
        "centroid", "detector", "yolo", "confidence", "accuracy"
    ]):
        if centroid:
            findings.append(
                f"Mean saved detector centroid error: {centroid['mean']:.3f} px; "
                f"RMSE: {centroid['rmse']:.3f} px; "
                f"within 2 px: {centroid['within_2px_percent']:.1f}%."
            )
            findings.append(
                "These are detector centroid measurements, not closed-loop "
                "camera tracking error."
            )
        if confidence:
            findings.append(
                f"Mean saved detector confidence: {confidence['mean']:.4f}."
            )

    else:
        findings.append(
            "I can analyze saved detector centroid error, trajectory RMSE and "
            "lock, detector confidence, and inference latency. Ask about one "
            "of those topics for a focused result."
        )

    return {
        "mode": "mock",
        "provider": "local deterministic test adapter",
        "question": question,
        "answer": "\n".join(findings),
        "evidence_files": [
            str(REPORT_PATH),
            str(TRAJECTORY_PATH),
        ],
        "warning": (
            "This is a local rule-based response, not a generative model response."
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

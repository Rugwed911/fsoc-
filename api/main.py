import os
import sys
from pathlib import Path
from typing import Literal

import json
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse

PROJECT = Path(__file__).resolve().parents[1]

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from ai_agent.copilot_engine import ask_copilot, load_evidence
from ai_agent.evidence_auditor import run_evidence_audit

REPORT_PATH = PROJECT / "results/reports/evidence_diagnostics.json"
TRAJECTORY_PATH = PROJECT / "experiments/baselines/multi_trajectory_validation.csv"
ARCHITECTURE_PATH = PROJECT / "results/figures/original_system_architecture.png"

app = FastAPI(
    title="FSOC SkyTrack AI API",
    description="Evidence-aware API for the FSOC SkyTrack virtual-camera tracking research prototype.",
    version="1.0.0",
)

origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class CopilotRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    mode: Literal["mock", "nebius"] = "mock"


def clean_records(records):
    """Convert pandas/numpy values into JSON-safe Python values."""
    return json.loads(
        json.dumps(records, default=str, allow_nan=False)
    )


@app.get("/")
def root():
    return {
        "name": "FSOC SkyTrack AI API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "fsoc-skytrack-api",
        "project_files_available": {
            "diagnostic_report": REPORT_PATH.is_file(),
            "trajectory_data": TRAJECTORY_PATH.is_file(),
            "architecture_image": ARCHITECTURE_PATH.is_file(),
        },
        "nebius_configured": all(
            os.getenv(key, "").strip()
            for key in [
                "NEBIUS_API_KEY",
                "NEBIUS_BASE_URL",
                "NEBIUS_MODEL",
            ]
        ),
    }


@app.get("/api/overview")
def overview():
    try:
        if not REPORT_PATH.is_file():
            raise FileNotFoundError("Diagnostic report is missing.")

        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        trajectory = pd.read_csv(TRAJECTORY_PATH)

        return {
            "project": "FSOC SkyTrack AI",
            "description": (
                "Virtual-camera optical-beacon tracking research prototype"
            ),
            "data_status": "recorded_project_evidence",
            "metrics": report.get("metrics", {}),
            "frame_rows": report.get("frame_rows"),
            "trajectory_rows": report.get("trajectory_rows"),
            "trajectory_records": clean_records(
                trajectory.where(pd.notna(trajectory), None).to_dict(
                    orient="records"
                )
            ),
            "evidence_warning": (
                "These are saved experiment records, not live sensor readings."
            ),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/audit")
def evidence_audit():
    try:
        audit = run_evidence_audit(PROJECT)
        return clean_records(audit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/copilot")
def copilot(request: CopilotRequest):
    try:
        result = ask_copilot(
            question=request.question,
            mode=request.mode,
        )
        return clean_records(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/assets/architecture")
def architecture_image():
    if not ARCHITECTURE_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail="Architecture image is unavailable."
        )
    return FileResponse(
        ARCHITECTURE_PATH,
        media_type="image/png",
        filename="fsoc-system-architecture.png",
    )

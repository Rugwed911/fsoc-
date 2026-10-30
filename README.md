# SKYTRACK — AI-Based Virtual Camera Tracking for Mobile FSOC

**SKYTRACK** is an engineering research prototype for software-based coarse alignment and optical beacon tracking in mobile Free-Space Optical Communication (FSOC) terminals. It combines a virtual camera environment, AI-assisted beacon detection, tracking-control experiments, recorded flight-style demonstrations, and evidence-oriented performance analysis in a web dashboard.

## Project overview

Free-Space Optical Communication can provide high-bandwidth, license-free optical links, but mobile platforms introduce a pointing, acquisition, and tracking (PAT) challenge. SKYTRACK explores a camera-based software approach to coarse alignment: detect the remote optical beacon in a camera field of view, estimate its image-plane position, and evaluate tracking behaviour under simulated motion and disturbances.

The project is a research and simulation prototype. It does **not** control a real aircraft, optical terminal, or physical pan-tilt unit.

## Main capabilities

- **Virtual camera and tracking experiments** — evaluate beacon localisation and closed-loop tracking in a controlled software environment.
- **AI-assisted detection** — use a trained beacon detector and recorded detection evidence for image-plane localisation.
- **Tracking and reliability analysis** — review centroid error, trajectory benchmarks, inference latency, confidence, and deterministic evidence-audit findings.
- **Mission dashboard** — React + Vite frontend with mission overview, video demonstrations, flight analytics, evidence auditor, AI Mission Engineer, and system architecture views.
- **Python API** — FastAPI service for health, overview metrics, audit findings, and copilot requests.
- **Recorded demonstrations** — embedded MP4 demonstrations are served as static frontend assets by the deployed website.

## Demonstration videos

1. **Validated C18 airborne AI-in-the-loop demo** — `frontend/public/videos/validated_c18_airborne_demo_h264.mp4`
2. **SKYTRACK cinematic 75-second demo** — `frontend/public/videos/FSOC_SkyTrack_Cinematic_Demo_75sec.mp4`

Both are accessible from the Mission Overview and Mission Video Demo sections of the website. These are prerecorded demonstrations, not live sensor streams.

## Technology stack

- **Frontend:** React, Vite, Recharts, CSS
- **Backend:** Python, FastAPI, Uvicorn
- **AI / experimentation:** Python-based computer vision and tracking experiments
- **Deployment:** Vercel frontend; API deployment configured separately

## Repository structure

```text
ai_agent/                 AI Mission Engineer and evidence-audit modules
api/                      FastAPI backend
dashboard/                Streamlit research dashboard
experiments/              Recorded benchmark datasets
frontend/                 React + Vite website
  public/videos/          MP4 assets served by the website
  src/                    Frontend application and styles
results/                  Experiment outputs, reports, figures, and source videos
simulation/               Virtual camera and detector assets
reproducibility/           Experiment scripts, reports, and validation outputs
```

## Run the frontend locally

```bash
cd frontend
npm install
npm run dev
```

To create a production build:

```bash
npm run build
```

If the API is deployed separately, set `VITE_API_URL` to the backend base URL before building the frontend.

## Research integrity note

Dashboard metrics and video demonstrations are based on saved experimental or simulated evidence. Detector centroid error and closed-loop tracking error are distinct measurements and should not be treated as interchangeable. Any AI copilot mode must be interpreted according to the configured provider and its actual connection status.

## Project status

SKYTRACK is an ongoing engineering research prototype. Some integrations and deployment components may be configuration-dependent; the repository and interface should not be interpreted as evidence of operational, real-world FSOC flight hardware.

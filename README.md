# Fight Detection for School CCTV — YOLO26-pose + mLSTM

Real-time fight detection that reads **skeletons, not pixels**. YOLO26-pose turns every camera frame into 17 body keypoints per person, the service adds per-joint velocity, and an **mLSTM** (matrix-memory LSTM from the 2024 xLSTM paper) classifies each one-second window as fight or no fight. When it fires, the service saves the clip and alerts the school dashboard.

Built as my AI module for the *AI-Powered School Management System* senior project at Arab International University (2026).

## Results

| Evaluation set | Sequences | Accuracy | F1 fight | F1 no-fight |
|---|---:|---:|---:|---:|
| SCVD — real CCTV, held-out test split | 547 | **81%** | 0.78 | 0.84 |
| RWF-2000 — held-out test clips | 1,800 | 73% | 0.74 | 0.72 |
| RWF-2000 — validation | 3,600 | 71% | 0.70 | 0.71 |

Trained on 15,596 thirty-frame sequences. Full write-up: [`docs/Fight_Detection_Report.docx`](docs/Fight_Detection_Report.docx).

## How it works

```
camera frame ──► YOLO26-pose ──► 17 keypoints × 2 people ──► + velocity (dx, dy)
                                   102 position features      68 velocity features
                                              │
                         30-frame sliding window (≈1 s), 170 features per frame
                                              │
                                 3-layer mLSTM, hidden 384, 6 heads
                                              │
                         fight confidence ≥ 0.60 ──► save clip + webhook alert
```

- **Why skeletons:** keypoints are far less sensitive to camera angle, lighting and clothing than raw pixels, and they are cheap enough to run live on one GPU.
- **Why velocity:** fights are fast, erratic, high-magnitude movement. Giving the model `dx, dy` per joint makes that explicit.
- **Training:** focal loss (focuses on hard examples), keypoint augmentation (Gaussian noise + horizontal flip), early stopping.
- **Evidence:** a rolling in-memory buffer keeps the last 10 s of every stream, so each alert comes with 10 s before and 10 s after the trigger. A 15 s cooldown per camera stops repeated alerts.

## Project layout

| File | What it does |
|---|---|
| `detector.py` | YOLO26-pose keypoint extraction, velocity features, mLSTM model and inference |
| `camera_manager.py` | Concurrent camera streams, rolling pre-buffer, footage writer |
| `alert_sender.py` | Webhook to the dashboard with camera id, confidence and clip path |
| `main.py` | FastAPI service: start/stop streams, list streams, health check |
| `extract_dataset.py` | Turns RWF-2000 / SCVD videos into keypoint + velocity sequences |
| `train.py` | Trains the mLSTM classifier and prints per-set reports |
| `test_detector.py` | Quick smoke test for the detector |

## Run it

```bash
python -m venv .venv && .venv\Scripts\activate      # or: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                               # or: cp .env.example .env, then edit
uvicorn main:app --host 0.0.0.0 --port 8001
```

Put the trained weights `fight_mlstm_81pct_final.pt` next to `detector.py` (see this repo's Releases). YOLO26-pose weights are downloaded by Ultralytics on first run.

### API

All endpoints except `/health` need the header `X-API-Key: <AI_API_KEY>`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/stream/start` | Connect a camera by RTSP/HTTP URL or device index |
| `POST` | `/stream/stop` | Disconnect a camera |
| `GET` | `/stream/list` | Active streams with frame and alert counts |

## Train your own

```bash
# 1. Extract keypoint sequences (RWF-2000, optionally SCVD)
python extract_dataset.py --dataset data/RWF-2000 --scvd data/SCVD/SCVD_converted

# 2. Train
python train.py --epochs 80 --batch 64 --hidden 384 --layers 3 --heads 6
```

Datasets are not included: [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) and SCVD (Smart-City CCTV Violence Detection) are available from their authors.

## Stack

Python · PyTorch · Ultralytics YOLO26-pose · OpenCV · FastAPI · scikit-learn

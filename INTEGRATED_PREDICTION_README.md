# Integrated 4-Model Prediction (FedAvg)

This project now includes a unified predictor that combines:

- `Viral/saved_viral_models/viral_skin_model.pth`
- `Parastic_Skin_Disease/checkpoints/parasitic_local.pth`
- `fungal/trained_model.pth`
- `Bacterial_Skin_Disease/bacterial_model.pth`

## What Was Added

- `multi_model_predictor.py`
  - Runs all 4 models on one image.
  - Returns:
    - per-model predictions + confidence
    - global FedAvg prediction + confidence
    - labels by model
- `integrated_api.py`
  - FastAPI service with:
    - `GET /`
    - `GET /labels`
    - `POST /predict`
- `fedavg_utils.py`
  - `fedavg_label_probabilities(...)` for prediction-time aggregation.
  - `fedavg_compatible_state_dicts(...)` for checkpoint parameter averaging.
- `fedavg_checkpoints.py`
  - Builds one FedAvg checkpoint from your 4 trained checkpoints.
  - Skips incompatible classifier-head keys automatically.

## Run The Unified API

```bash
uvicorn integrated_api:app --reload --host 0.0.0.0 --port 8000
```

Prediction request:

```bash
curl -X POST "http://127.0.0.1:8000/predict?topk=5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/image.jpg"
```

## Run CLI Prediction

```bash
python3 multi_model_predictor.py --image /absolute/path/to/image.jpg --topk 5
```

## Create A FedAvg Global Checkpoint

```bash
python3 fedavg_checkpoints.py --output ./global/fedavg_global_checkpoint.pth
```

Optional model path overrides:

```bash
python3 fedavg_checkpoints.py \
  --output ./global/fedavg_global_checkpoint.pth \
  --viral-model "/Users/nikshanps/Desktop/RESEARCH/COMPS_TRAINED copy/Viral/saved_viral_models" \
  --parasitic-model "/Users/nikshanps/Desktop/RESEARCH/COMPS_TRAINED copy/Parastic_Skin_Disease/checkpoints" \
  --fungal-model "/Users/nikshanps/Desktop/RESEARCH/COMPS_TRAINED copy/fungal/trained_model.pth" \
  --bacterial-model "/Users/nikshanps/Desktop/RESEARCH/COMPS_TRAINED copy/Bacterial_Skin_Disease/bacterial_model.pth"
```

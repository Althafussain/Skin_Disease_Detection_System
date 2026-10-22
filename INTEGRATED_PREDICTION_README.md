# Unified Skin Disease Screening

The integrated API uses one classifier trained across the disease classes present in the
project's four datasets. This gives the model a shared label space, so scores from unrelated
specialist classifiers are not compared as though they were calibrated against one another.

The current training data contains eight classes: two bacterial, two viral, three fungal, and
one parasitic class. The parasitic checkpoint had only one class; the unified classifier can
still distinguish its parasitic examples from other included classes because it is trained
jointly on all eight classes.

## Train The Unified Classifier

- `train_unified_model.py` fine-tunes an ImageNet-pretrained MobileNetV2 using each disease
  group's existing `Train` folders. It evaluates against the existing `Test` folders and saves
  the checkpoint as `unified_skin_model.pth`.

```bash
python train_unified_model.py
```

The saved model's overall and per-class held-out test accuracy are printed at the end of training.

These scores are model estimates, not calibrated probabilities or a medical diagnosis. The
classifier can only identify the eight classes it was trained on; add labeled examples for other
diseases and healthy/normal skin before using it for those cases.

## API Contents

- `integrated_api.py`
  - FastAPI service with:
    - `GET /`
    - `GET /labels`
    - `POST /predict`
- `unified_skin_classifier.py` loads the shared classifier and returns top-k disease scores.

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

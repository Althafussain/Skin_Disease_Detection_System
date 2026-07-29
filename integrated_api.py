from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from multi_model_predictor import MultiModelFedAvgPredictor
from symptoms_engine import SymptomRefinementEngine


BASE_DIR = Path(__file__).resolve().parent

predictor = MultiModelFedAvgPredictor(
    model_paths={
        # Directory paths are accepted for viral/parasitic and resolved to checkpoint files.
        "viral": BASE_DIR / "Viral" / "saved_viral_models",
        "parasitic": BASE_DIR / "Parastic_Skin_Disease" / "checkpoints",
        "fungal": BASE_DIR / "fungal" / "trained_model.pth",
        "bacterial": BASE_DIR / "Bacterial_Skin_Disease" / "bacterial_model.pth",
    }
)
symptom_engine = SymptomRefinementEngine(BASE_DIR / "Symptoms")


class RankedPrediction(BaseModel):
    prediction: str
    confidence: float = 0.0


class SymptomSuggestRequest(BaseModel):
    topk: list[RankedPrediction] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class SymptomFinalPredictRequest(BaseModel):
    selected_symptoms: list[str] = Field(default_factory=list)
    custom_keywords: list[str] = Field(default_factory=list)
    image_topk: list[RankedPrediction] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


app = FastAPI(title="Integrated Skin Disease Prediction API (FedAvg)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Integrated 4-model prediction API running",
        "aggregation_method": "fedavg_label_probabilities",
    }


@app.get("/labels")
def labels():
    return {"labels_by_model": predictor.get_label_catalog()}


@app.get("/symptoms/catalog")
def symptom_catalog():
    return {"symptoms": symptom_engine.symptom_catalog()}


@app.post("/symptoms/suggest")
def suggest_symptoms(payload: SymptomSuggestRequest):
    topk_payload = [row.model_dump() for row in payload.topk]
    show_suggestions = symptom_engine.should_show_suggestions(topk_payload)
    suggestions = symptom_engine.suggest_symptoms(
        image_topk=topk_payload,
        limit=payload.limit,
    )
    return {
        "show_suggestions": show_suggestions,
        "suggestions": suggestions,
        "symptoms": symptom_engine.symptom_catalog(),
    }


@app.post("/symptoms/final-predict")
def final_predict(payload: SymptomFinalPredictRequest):
    if not payload.selected_symptoms and not payload.custom_keywords:
        raise HTTPException(status_code=400, detail="Select symptoms or enter symptom keywords")

    try:
        return symptom_engine.final_predict(
            selected_symptoms=payload.selected_symptoms,
            custom_keywords=payload.custom_keywords,
            image_topk=[row.model_dump() for row in payload.image_topk],
            limit=payload.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = Query(5, ge=1, le=20)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        return predictor.predict_from_bytes(image_bytes, topk=topk)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

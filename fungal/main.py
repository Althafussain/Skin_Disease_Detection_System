from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

try:
    from .PredictModel import predict_from_bytes
except ImportError:
    from PredictModel import predict_from_bytes

app = FastAPI()

# Allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = predict_from_bytes(img_bytes, topk=3)
    return result

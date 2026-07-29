# SkinFlect Components - Comprehensive Analysis

**Repository:** VivushT/SkinFlect  
**Current Branch:** Vinthiya_Parastic2  
**Analysis Date:** March 6, 2026  
**Workspace Location:** `c:\Users\thaya\Desktop\skin\COMPS`

---

## 📋 Project Overview

**SkinFlect** is a multi-component skin disease classification system using PyTorch and FastAPI. The system is designed to predict and classify four types of skin diseases:

1. **Bacterial Skin Diseases**
2. **Fungal Skin Diseases**
3. **Parasitic Skin Diseases**
4. **Viral Skin Diseases**

Each disease category has its own independent component with:
- FastAPI REST API endpoints
- PyTorch-based neural network models (MobileNetV2)
- Dedicated training and testing datasets
- Model prediction and inference capabilities

---

## 🏗️ Directory Structure

```
COMPS/
├── component_fixes.diff                    # Patch file for component updates
├── component_fixes_unified.diff            # Unified diff for all components
├── Bacterial_Skin_Disease/                 # Component 1: Bacterial diseases
├── fungal/                                 # Component 2: Fungal diseases
├── Parastic_Skin_Disease/                  # Component 3: Parasitic diseases
├── Viral/                                  # Component 4: Viral diseases
└── tests/                                  # Unit tests and smoke tests
```

---

## 📊 Dataset Statistics

### Training Data Distribution

| Component | Class | Train Count | Purpose |
|-----------|-------|-------------|---------|
| **Bacterial** | BA-cellulitis | 474 | Primary disease class |
| | BA-Ery | 180 | Primary disease class |
| | BA-impetigo | 285 | Primary disease class |
| | Normal | 2,647 | Healthy/control class |
| | None | 60 | Negative samples |
| **Fungal** | FU-athlete-foot | 99 | Primary disease class |
| | FU-nail-fungus | 3,470 | Primary disease class |
| | FU-ringworm | 3,702 | Primary disease class |
| | None | 60 | Negative samples |
| **Parasitic** | PA-cutaneous-larva-migrans | 200 | Primary disease class |
| | Scabies | 284 | Primary disease class |
| | Tinea | 545 | Primary disease class |
| | NormalSkin | 2,562 | Healthy/control class |
| | None | 60 | Negative samples |
| **Viral** | VI-chickenpox | 2,567 | Primary disease class |
| | VI-shingles | 130 | Primary disease class |
| | Healthy | 2,561 | Healthy/control class |
| | None | 60 | Negative samples |

### Key Dataset Insights:
- **Total Training Images:** ~25,868 across all components
- **Class Imbalance:** Significant imbalance in some classes (e.g., BA-Ery: 180 vs Normal: 2,647)
- **Fungal Dataset:** Highest volume with FU-ringworm (3,702) and FU-nail-fungus (3,470)
- **Viral Dataset:** Well-balanced between VI-chickenpox (2,567) and Healthy (2,561)
- **Control Classes:** Each component uses either "Normal", "NormalSkin", or "Healthy" class plus a "None" negative class

---

## 🔧 Component Architecture

### 1. Bacterial Skin Disease Component

**Location:** `Bacterial_Skin_Disease/`

**Key Files:**
- `api.py` - FastAPI REST endpoint
- `model_builder.py` - MobileNetV2 model construction
- `dataloader.py` - Dataset loading and preprocessing
- `PredictModel.py` - Model inference and prediction logic
- `client_train.py` - Training script
- `utils.py` - Utility functions

**API Endpoint:**
```
POST /predict - Upload image for bacterial disease classification
GET / - Health check endpoint
```

**Model Details:**
- Architecture: MobileNetV2 (pre-trained or from scratch)
- Input Size: 224×224 RGB images
- Output Classes: 5 (BA-cellulitis, BA-Ery, BA-impetigo, None, Normal)
- Confidence Threshold: 0.60

**Special Features:**
- Skin detection validation
- Confidence threshold filtering
- Detailed prediction schema including `skin_found` field

---

### 2. Fungal Skin Disease Component

**Location:** `fungal/`

**Key Files:**
- `main.py` - FastAPI REST endpoint
- `model_builder.py` - MobileNetV2 model construction
- `dataloader.py` - Dataset loading and preprocessing
- `PredictModel.py` - Model inference and prediction logic
- `client_train.py` - Training script
- `utils.py` - Utility functions

**API Endpoint:**
```
POST /predict - Upload image for fungal disease classification
```

**Model Details:**
- Architecture: MobileNetV2
- Input Size: 224×224 RGB images
- Output Classes: 4 (FU-athlete-foot, FU-nail-fungus, FU-ringworm, None)
- Top-K Predictions: Returns top 3 predictions by default

**CORS Policy:**
- Allows all origins (development mode)
- Allows credentials, all methods, and all headers

---

### 3. Parasitic Skin Disease Component

**Location:** `Parastic_Skin_Disease/`

**Key Files:**
- `api.py` - FastAPI REST endpoint
- `model_builder.py` - MobileNetV2 model construction
- `dataloader.py` - Dataset loading and preprocessing
- `Predict.py` - Model inference and prediction logic (Note: Different filename)
- `client_train.py` - Training script
- `utils.py` - Utility functions

**API Endpoint:**
```
POST /predict - Upload image for parasitic disease classification
GET / - Health check endpoint
```

**Model Details:**
- Architecture: MobileNetV2
- Input Size: 224×224 RGB images
- Output Classes: 5 (None, NormalSkin, PA-cutaneous-larva-migrans, Scabies, Tinea)

**Note:** Uses `Predict.py` instead of `PredictModel.py` (inconsistency with other components)

---

### 4. Viral Skin Disease Component

**Location:** `Viral/`

**Key Files:**
- `api.py` - FastAPI REST endpoint
- `model_builder.py` - MobileNetV2 model construction
- `dataloader.py` - Dataset loading and preprocessing
- `PredictModel.py` - Model inference and prediction logic
- `client_train.py` - Training script
- `utils.py` - Utility functions

**API Endpoint:**
```
POST /predict - Upload image for viral disease classification
GET / - Health check endpoint
```

**Model Details:**
- Architecture: MobileNetV2
- Input Size: 224×224 RGB images
- Output Classes: 4 (Healthy, None, VI-chickenpox, VI-shingles)
- Top-K Predictions: Returns top 3 predictions by default

---

## 🔄 Common Architecture Patterns

### Image Preprocessing Pipeline
All components use identical preprocessing:
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Model Loading Strategy
All components implement:
- Model caching to avoid reloading
- Flexible checkpoint loading (with or without metadata)
- Class resolution from train directory or checkpoint metadata
- Device management (CUDA/CPU)

### Prediction Flow
1. Load image from bytes/file
2. Preprocess to tensor
3. Forward pass through model
4. Apply softmax for probabilities
5. Return top-K predictions with confidence scores

### REST API Pattern (FastAPI)
- CORS middleware for localhost:5173 (React dev server)
- File upload handling via `UploadFile`
- JSON response format with prediction results

---

## 🧪 Testing

**Location:** `tests/test_smoke_components.py`

**Test Coverage:**
- ✅ Viral component: Build, forward pass, checkpoint, prediction schema
- ✅ Bacterial component: Build, forward pass, checkpoint, prediction schema
- ✅ Parasitic component: Build, forward pass, checkpoint, prediction schema
- ✅ Fungal component: (Partial, incomplete in shown code)

**Test Pattern:**
1. Create model with specified number of classes
2. Test forward pass with random input
3. Verify output shape matches number of classes
4. Save checkpoint with metadata
5. Load checkpoint and verify class restoration
6. Test prediction and validate response schema

**Expected Prediction Schema:**
```python
{
    "prediction": str,           # Primary prediction class
    "confidence": float,         # Confidence score (0-1)
    "topk": list[dict],          # Top-K predictions
    "threshold": float,          # Confidence threshold (Bacterial only)
    "skin_found": bool           # Skin detection result (Bacterial only)
}
```

---

## 📦 Dependencies

**Core Libraries:**
- `fastapi==0.115.0` - Web framework
- `uvicorn==0.30.6` - ASGI server
- `torch` - PyTorch deep learning framework
- `torchvision` - Computer vision utilities
- `pillow` - Image processing
- `python-multipart==0.0.9` - File upload handling

**Python Version:** 3.8+

---

## 🔧 Configuration & Paths

### Bacterial Component
- Model Path: `Bacterial_Skin_Disease/bacterial_model.pth`
- Dataset: `Bacterial_Skin_Disease/Bacterial-Dataset/`
- Confidence Threshold: 0.60

### Fungal Component
- Default Model Path: `fungal/saved_fungal_models/fungal_skin_model.pth`
- Dataset: `fungal/Fungal-Dataset/`

### Parasitic Component
- Default Model Path: `Parastic_Skin_Disease/parastic_model.pth`
- Dataset: `Parastic_Skin_Disease/Parastic-Dataset/`

### Viral Component
- Default Model Path: `Viral/saved_viral_models/viral_skin_model.pth`
- Dataset: `Viral/Viral-Dataset/`

---

## 📝 Code Quality Observations

### ✅ Strengths:
1. **Modular Architecture** - Each disease type is independent
2. **Consistent Patterns** - Similar structure across all components
3. **Flexible Model Loading** - Supports multiple checkpoint formats
4. **Robust Preprocessing** - Standard ImageNet normalization
5. **Comprehensive Testing** - Smoke tests for schema validation
6. **CORS Configuration** - Frontend integration ready

### ⚠️ Areas for Improvement:
1. **Naming Inconsistency** - `Parastic_Skin_Disease/Predict.py` vs others use `PredictModel.py`
2. **Class Naming** - "Parastic" should be "Parasitic"
3. **CORS Configuration** - Fungal allows all origins; others restrict to localhost:5173
4. **Error Handling** - Limited validation on input images
5. **Model Architecture** - All use MobileNetV2; could benefit from ensemble methods for critical decisions
6. **Data Imbalance** - Significant class imbalance not addressed (e.g., BA-Ery: 180 vs Normal: 2,647)
7. **Threshold Configuration** - Only Bacterial component has a confidence threshold

---

## 🚀 Deployment & Usage

### Starting Individual Components:
```bash
# Bacterial
uvicorn Bacterial_Skin_Disease.api:app --reload --port 8001

# Fungal
uvicorn fungal.main:app --reload --port 8002

# Parasitic
uvicorn Parastic_Skin_Disease.api:app --reload --port 8003

# Viral
uvicorn Viral.api:app --reload --port 8004
```

### Running Tests:
```bash
python -m pytest tests/test_smoke_components.py -v
```

### Example API Usage:
```bash
curl -X POST "http://localhost:8001/predict" \
  -H "accept: application/json" \
  -F "file=@image.jpg"
```

---

## 📈 Recommendations

1. **Standardize File Naming** - Rename `Predict.py` → `PredictModel.py` in Parasitic component
2. **Fix Component Name** - Rename `Parastic_Skin_Disease` → `Parasitic_Skin_Disease` (correct spelling)
3. **Unified CORS Configuration** - Standardize CORS settings across all APIs
4. **Add Confidence Thresholds** - Implement threshold filtering for all components (not just Bacterial)
5. **Address Class Imbalance** - Use weighted loss or data augmentation during training
6. **Enhanced Input Validation** - Add image dimension and format validation
7. **Logging & Monitoring** - Add request logging and model performance tracking
8. **Documentation** - Add API documentation (Swagger/OpenAPI)
9. **Configuration Management** - Move hardcoded paths to config files
10. **Ensemble Methods** - Consider multi-model ensembles for critical predictions

---

## 📌 Important Notes

- **Branch Status:** Currently on `Vinthiya_Parastic2` (suggests recent parasitic disease component work)
- **Diff Files:** `component_fixes_unified.diff` and `component_fixes.diff` suggest recent refactoring/fixes applied
- **Image Format:** All components expect JPEG/PNG RGB images
- **Model Size:** MobileNetV2 is efficient for deployment (~9.2 MB)
- **Inference Time:** Expected ~50-100ms per image on GPU

---

## 🔐 Security Considerations

1. **Input Validation** - Add file type and size validation
2. **Rate Limiting** - Implement rate limiting on API endpoints
3. **Authentication** - Consider API key or JWT authentication
4. **CORS Security** - Restrict origins in production
5. **File Handling** - Use secure temporary file handling for uploads
6. **Model Protection** - Consider model obfuscation/protection if proprietary

---

Generated: March 6, 2026

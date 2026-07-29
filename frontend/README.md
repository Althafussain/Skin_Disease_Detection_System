# Skin Disease Prediction Frontend (React)

This frontend uploads an image and shows the top 5 FedAvg predictions from the integrated API.

## 1) Start backend API

From repository root:

```bash
uvicorn integrated_api:app --reload --host 0.0.0.0 --port 8000
```

## 2) Start frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

## Optional API base override

Default API base URL:

```txt
http://127.0.0.1:8000
```

To override, create `frontend/.env`:

```bash
VITE_API_BASE_URL=http://your-api-host:8000
```

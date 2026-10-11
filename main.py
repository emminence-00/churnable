"""
FastAPI Application for Churnable.
Coordinates dataset management, analytical tools, API endpoints, and frontend dashboard.
"""

import os
import io
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

load_dotenv()

import tools
import agent
from data_generator import ensure_data_directory, generate_synthetic_customers

app = FastAPI(
    title="Churnable",
    description="Agentic Customer Churn Prediction & Investigation for Digital Banking and FinTech",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data/customers.csv"
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB maximum CSV upload

# Ensure initial synthetic data exists on startup
ensure_data_directory(DATA_PATH)

# Mount static folder
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class InvestigationRequest(BaseModel):
    task: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the single-page frontend application."""
    index_file = "static/index.html"
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h2>ChurnGuard AI loading...</h2>")


@app.get("/api/overview")
async def get_overview():
    """Returns dataset overview, data quality score, and API key configuration status."""
    try:
        inspection = tools.inspect_data(DATA_PATH)
        quality = tools.check_data_quality(DATA_PATH)
        risk = tools.calculate_churn_risk(DATA_PATH, limit=10)
        patterns = tools.summarize_churn_patterns(DATA_PATH)

        has_api_key = bool(os.getenv("GEMINI_API_KEY", "").strip())
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

        return {
            "status": "success",
            "has_gemini_key": has_api_key,
            "gemini_model": model_name,
            "prediction_engine": f"Agentic AI ({model_name})" if has_api_key else "Heuristic Rule Fallback",
            "dataset_info": {
                "total_rows": inspection.get("total_rows", 0),
                "total_columns": inspection.get("total_columns", 0),
                "columns": inspection.get("columns", []),
                "missing_count": sum(v["count"] for v in inspection.get("missing_values_summary", {}).values()),
                "duplicate_rows": inspection.get("duplicate_rows", 0),
                "sample_preview": inspection.get("sample_preview", [])
            },
            "quality_info": {
                "quality_score": quality.get("data_quality_score", 100),
                "is_healthy": quality.get("is_healthy", True),
                "warnings": quality.get("warnings", []),
                "issues": quality.get("issues_found", []),
                "supported": quality.get("supported_analyses", []),
                "unsupported": quality.get("unsupported_analyses", [])
            },
            "historical_churn": {
                "has_label": quality.get("has_historical_churn_label", False),
                "churn_rate_pct": patterns.get("historical_churn_rate_pct", None) if patterns.get("status") == "success" else None,
                "churned_count": patterns.get("churned_count", None) if patterns.get("status") == "success" else None
            },
            "risk_summary": {
                "average_risk_score": risk.get("average_risk_score", 0),
                "distribution": risk.get("risk_distribution", {})
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load overview: {str(e)}")


class PredictChurnRequest(BaseModel):
    cohort_type: Optional[str] = "candidates"
    limit: Optional[int] = 15


@app.post("/api/predict-churn")
async def predict_churn(req: PredictChurnRequest):
    """Directly triggers the Gemini AI Agent to evaluate customer profiles and determine churn risk."""
    try:
        result = agent.run_direct_agent_churn_prediction(
            cohort_type=req.cohort_type or "candidates",
            limit=req.limit or 15,
            dataset_path=DATA_PATH
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent churn prediction failed: {str(e)}")


@app.post("/api/investigate")
async def investigate_churn(req: InvestigationRequest):
    """Executes the agentic investigation loop via Gemini or fallback."""
    task = req.task.strip()
    if not task:
        raise HTTPException(status_code=400, detail="Investigation task prompt cannot be empty.")
    
    if len(task) > 1000:
        raise HTTPException(status_code=400, detail="Task query exceeds maximum length of 1000 characters.")

    try:
        result = agent.run_gemini_investigation(task, dataset_path=DATA_PATH)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Investigation failed: {str(e)}")


@app.post("/api/chat")
async def chat_with_agent_endpoint(req: ChatRequest):
    """
    Interactive conversational chatbot endpoint.
    Passes user prompt and conversation history to the LangGraph Gemini Agent,
    returning conversational reasoning, executed tool trace, and live churn predictions.
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        result = agent.chat_with_agent(
            message=message,
            history=req.history or [],
            dataset_path=DATA_PATH
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/regenerate-data")
async def regenerate_data():
    """Regenerates the synthetic banking dataset with a new random seed."""
    try:
        import numpy as np
        new_seed = int(np.random.randint(100, 99999))
        df = generate_synthetic_customers(n_customers=750, seed=new_seed)
        df.to_csv(DATA_PATH, index=False)
        return {
            "status": "success",
            "message": f"Dataset regenerated successfully ({len(df)} records, seed={new_seed})."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to regenerate dataset: {str(e)}")


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Validates and uploads a custom customer dataset CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")

    try:
        import pandas as pd
        df = pd.read_csv(io.BytesIO(contents))
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV file is empty.")

        # Save to active path
        df.to_csv(DATA_PATH, index=False)
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "message": f"Successfully uploaded {file.filename} with {len(df)} records and {len(df.columns)} columns."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV structure: {str(e)}")


@app.get("/api/data")
async def get_all_data(page: int = 1, page_size: int = 50, search: Optional[str] = None):
    """Returns paginated customer records with optional search filter."""
    if not os.path.exists(DATA_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found.")
    try:
        import pandas as pd
        df = pd.read_csv(DATA_PATH)
        if search:
            s = search.lower().strip()
            mask = df.astype(str).apply(lambda row: row.str.lower().str.contains(s, regex=False).any(), axis=1)
            df = df[mask]

        total_records = len(df)
        total_pages = max(1, (total_records + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        page_df = df.iloc[start_idx:end_idx]

        import math
        raw_records = page_df.to_dict(orient="records")
        records = [
            {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v)
             for k, v in row.items()}
            for row in raw_records
        ]
        return {
            "status": "success",
            "total_records": total_records,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "columns": list(df.columns),
            "records": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dataset: {str(e)}")


@app.get("/api/download-sample-csv")
async def download_current_csv():
    """Allows downloading the active CSV dataset."""
    if os.path.exists(DATA_PATH):
        return FileResponse(DATA_PATH, filename="customers_sample.csv", media_type="text/csv")
    raise HTTPException(status_code=404, detail="Dataset not found.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

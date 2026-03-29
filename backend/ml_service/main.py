import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx
import uuid
import numpy as np
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../../model"))

from inference.predictor import UIPredictor
from analysis.analyzer import UIAnalyzer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../model/checkpoints/train/weights/best.pt"
)

predictor = UIPredictor(model_path=MODEL_PATH)

@app.get("/")
def health_check():
    return { "status": "ML service is running" }

@app.post("/evaluate")
async def evaluate(payload: dict):
    image_url = payload.get("imageUrl")

    if not image_url:
        raise HTTPException(status_code=400, detail="imageUrl is required")

    temp_path = f"temp_{uuid.uuid4().hex}.jpg"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(image_url)
            with open(temp_path, "wb") as f:
                f.write(response.content)

        result = predictor.predict_and_analyze(
            image_path=temp_path,
            viewport_width=1920,
            viewport_height=1080,
            confidence_threshold=0.15,
            save_visualization=False
        )

        # use the same converter UIAnalyzer uses internally
        # this is the exact same function your model already uses for JSON export
        json_str = json.dumps(result, default=UIAnalyzer._convert_numpy)
        clean_result = json.loads(json_str)

        return JSONResponse(content=clean_result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
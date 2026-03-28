from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def homeroute_check():
    return { "status":"ML service is running"}
@app.post("/evaluate")
async def evaluate(payload: dict):
    image_url = payload.get("imageUrl")
    
    if not image_url:
        return { "error": "imageUrl is required" }
    
    # dummy score for now — real model comes later
    return {
        "scores": { "clutter": 72, "alignment": 88, "colorContrast": 65 },
        "components": [
    { "id": 1, "label": "Navbar", "x": 0.0, "y": 0.0, "width": 1.0, "height": 0.08 },
    { "id": 2, "label": "Hero Button", "x": 0.35, "y": 0.42, "width": 0.3, "height": 0.09 },
    { "id": 3, "label": "Input Field", "x": 0.1, "y": 0.25, "width": 0.5, "height": 0.07 },
    { "id": 4, "label": "Card", "x": 0.05, "y": 0.55, "width": 0.4, "height": 0.3 },
    { "id": 5, "label": "Sidebar", "x": 0.75, "y": 0.1, "width": 0.22, "height": 0.85 },
  ],
    }
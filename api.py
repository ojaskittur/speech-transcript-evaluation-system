from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from scorer import IntroductionScorer
import uvicorn
import os

app = FastAPI()

# Input Model
class ScoreRequest(BaseModel):
    transcript: str
    duration: int

# Mount static folder to serve index.html
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open(os.path.join("static", "index.html"), "r") as f:
        return f.read()

@app.post("/api/score")
async def get_score(request: ScoreRequest):
    scorer = IntroductionScorer(request.transcript, request.duration)
    results = scorer.calculate_overall_score()
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
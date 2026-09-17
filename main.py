from fastapi import FastAPI
from schemas import EssayRequest, GraderResponse
from grader import grade_essay

app = FastAPI(
    title="Hybrid Essay Grader API",
    description="API untuk koreksi esai otomatis menggunakan Rule-based keyword matching dan LSA.",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome to Hybrid Essay Grader API"}

@app.post("/grade", response_model=GraderResponse)
def grade_endpoint(request: EssayRequest):
    return grade_essay(request)

from fastapi import FastAPI
from schemas import (
    EssayRequest,
    GraderResponse,
    PosTagRequest,
    PosTagResponse,
    PosTagItem,
    LemmatizeRequest,
    LemmatizeResponse,
)
from grader import grade_essay
from nlp_pipeline.preprocessing import get_pos_tags, lemmatize_text

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

@app.post("/pos-tag", response_model=PosTagResponse)
def pos_tag_endpoint(request: PosTagRequest):
    tags = get_pos_tags(request.text, refine_foreign=request.refine_foreign)
    return PosTagResponse(tags=[PosTagItem(token=t[0], tag=t[1]) for t in tags])

@app.post("/lemmatize", response_model=LemmatizeResponse)
def lemmatize_endpoint(request: LemmatizeRequest):
    lemmatized = lemmatize_text(request.text)
    return LemmatizeResponse(original_text=request.text, lemmatized_text=lemmatized)


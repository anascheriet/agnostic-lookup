"""
FastAPI app exposing the RAG comparison endpoint.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from rag import compare
from subjects import DOMAINS

app = FastAPI(title="AgnosticLookup", version="0.1.0")


class CompareRequest(BaseModel):
    subject_a: str
    subject_b: str
    domain: str | None = None


class CompareResponse(BaseModel):
    subject_a: str
    subject_b: str
    domain: str | None
    comparison: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/domains")
def list_domains():
    return {"domains": DOMAINS}


@app.post("/compare", response_model=CompareResponse)
def compare_subjects(body: CompareRequest):
    if body.domain and body.domain not in DOMAINS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown domain '{body.domain}'. Valid domains: {DOMAINS}",
        )

    result = compare(body.subject_a, body.subject_b, domain=body.domain)

    return CompareResponse(
        subject_a=body.subject_a,
        subject_b=body.subject_b,
        domain=body.domain,
        comparison=result,
    )

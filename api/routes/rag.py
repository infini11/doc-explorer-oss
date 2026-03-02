"""
RAG endpoints using the MedicalKnowledgeGraph class.

Endpoints:
  POST /api/v1/ingest          — ingests a text into the Knowledge Graph
  POST /api/v1/ask             — asks a question (RAG)
  POST /api/v1/ingest-and-ask  — ingestion + question in a single pass
"""
import sys
import logging
from fastapi import APIRouter

from api.schemas.rag import (
    QuestionRequest,
    QuestionResponse,
    FullPipelineRequest,
    FullPipelineResponse,
)
sys.path.append(".")
from pipelines.knowledge_graph.graph_builder import MedicalKnowledgeGraph

rag_router = APIRouter(tags=["rag"])
logger = logging.getLogger(__name__)

kg = MedicalKnowledgeGraph()

@rag_router.post("/ask", response_model=QuestionResponse, status_code=200)
def ask(request: QuestionRequest):
    
    logger.info(f"question='{request.question}'")

    answer = kg.ask_question(request.question)

    return QuestionResponse(
        question=request.question,
        answer=answer,
    )


@rag_router.post("/ingest-and-ask", response_model=FullPipelineResponse, status_code=201)
def ingest_and_ask(request: FullPipelineRequest):

    logger.info(f"full_pipeline question='{request.question}'")

    result = kg.ingest_and_ask(request.text, request.question)

    return FullPipelineResponse(
        stored_disease=result["stored_disease"],
        question=request.question,
        answer=result["answer"],
        error=result.get("error"),
    )


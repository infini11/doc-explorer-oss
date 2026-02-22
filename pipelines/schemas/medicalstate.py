from typing import Optional
from typing_extensions import TypedDict

# State — data flowing between nodes
class MedicalState(TypedDict):
    """
    Shared state across all LangGraph nodes.
    Each node reads the state, enriches it, and passes it to the next.
    """
    text:      str            # Raw medical text (for ingestion)
    question:  str            # User question (for Q&A)
    entities:  Optional[dict] # Entities extracted by the LLM
    retrieved: Optional[list] # Entities found in the store
    answer:    str            # Final answer
    error:     Optional[str]  # Potential error
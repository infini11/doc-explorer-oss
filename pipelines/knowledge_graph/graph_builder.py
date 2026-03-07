"""
Medical Knowledge Graph orchestrated with LangGraph.

In our case, LangGraph orchestrates:
  1. Medical entity extraction (LLM)
  2. Storage in a local JSON store
  3. Search within that store
  4. Response generation

"""

import os
import sys
import json
import logging
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

sys.path.append(".")
from pipelines.preprocessing.entity_extractor import EntityExtractor
from pipelines.schemas.medicalstate import MedicalState

logger = logging.getLogger(__name__)

STORE_PATH = Path(os.getenv("KG_STORE_PATH"))
STORE_PATH.parent.mkdir(parents=True, exist_ok=True)

class MedicalKnowledgeGraph:

    def __init__(self, store_path: Path = STORE_PATH, llm_model: str = "llama3.1"):
        self.store_path = store_path
        self.llm_model = llm_model
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON Store — persistance locale
    def _load_store(self) -> list[dict]:
        """Loads all entities from the store."""
        if not self.store_path.exists():
            return []
        return json.loads(self.store_path.read_text())
    
    def _save_entity(self, entity: dict) -> None:
        """Adds or updates an entity (no duplicates on disease name)."""
        store = self._load_store()
        store = [e for e in store if e["disease"].lower() != entity["disease"].lower()]
        store.append(entity)
        self.store_path.write_text(json.dumps(store, indent=2))

    def _search_entities(self, question: str) -> list[dict]:
        """Returns entities whose disease or symptoms match the question."""
        q = question.lower()
        matches = []
        for entity in self._load_store():
            terms = (
                [entity["disease"].lower()] +
                [s.lower() for s in entity.get("symptoms", [])] +
                [c.lower() for c in entity.get("causes", [])]
            )
            if any(t in q or q in t for t in terms):
                matches.append(entity)
        return matches
    
    # LangGraph flows

    def node_extract(self, state: MedicalState) -> MedicalState:
        """
        Node 1 — Extracts medical entities from text via LLM.
        If no text is provided, passes without error (Q&A-only mode).
        """
        logger.info("[node] extract")

        text = state.get("text", "").strip()
        if not text:
            return {**state, "error": None}
        
        result = EntityExtractor(model=self.llm_model).extract(text)

        if result is None:
            return {**state, "error": "Entity extraction failed"}

        logger.info(f"  → extracted: {result.disease}")
        return {**state, "entities": result.model_dump(), "error": None}
    
    def node_store(self, state: MedicalState) -> MedicalState:
        """Node 2 — Saves extracted entities to the JSON store."""
        logger.info("[node] store")

        entities = state.get("entities")
        if entities:
            self._save_entity(entities)
            logger.info(f"  → stored: {entities['disease']}")

        return {**state}
    
    def node_query(self, state: MedicalState) -> MedicalState:
        """Node 3 — Searches the store for entities related to the question."""
        logger.info("[node] query")

        question = state.get("question", "")
        results  = self._search_entities(question) if question else []

        logger.info(f"  → {len(results)} entities found")
        return {**state, "retrieved": results}
    
    def node_generate(self, state: MedicalState) -> MedicalState:
        """Node 4 — Generates an answer via LLM from the retrieved entities."""
        logger.info("[node] generate")

        retrieved = state.get("retrieved", [])
        question  = state.get("question", "")

        if not retrieved:
            return {**state, "answer": "No relevant medical data found. Please ingest documents first."}

        context = "\n\n".join([
            f"Disease   : {e['disease']}\n"
            f"Severity  : {e.get('severity', 'unknown')}\n"
            f"Symptoms  : {', '.join(e.get('symptoms', []))}\n"
            f"Causes    : {', '.join(e.get('causes', []))}\n"
            f"Treatments: {', '.join(e.get('treatments', []))}"
            for e in retrieved
        ])
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a medical AI assistant.
            Answer ONLY using the structured knowledge below.
            Always include: symptoms, causes, treatments and severity.

            === Medical Knowledge ===
            {context}

            === Question ===
            {question}

            === Answer ==="""
        )

        answer = (prompt | OllamaLLM(model=self.llm_model, temperature=0)).invoke({
            "context": context,
            "question": question,
        })

        logger.info("  → answer generated")
        return {**state, "answer": answer}
    
    def node_error(self, state: MedicalState) -> MedicalState:
        """Terminal node — handles pipeline errors."""
        logger.error(f"[node] error: {state.get('error')}")
        return {**state, "answer": f"Pipeline error: {state.get('error')}"}

    # Conditional routing

    def _route_after_extract(self, state: MedicalState) -> str:
        """After extraction: error → node_error, otherwise → store."""
        return "error" if state.get("error") else "store"

    def _route_after_store(self, state: MedicalState) -> str:
        """After storage: question present → query, otherwise → end."""
        return "query" if state.get("question", "").strip() else END

   # Compiled graphs — 3 usage modes

    def build_ingestion_graph(self):
        """
        Ingestion-only mode.
        Flow: extract → store → END

        Usage: Airflow pipeline after a document upload.
        Input: { text: "..." }
        """
        g = StateGraph(MedicalState)
        g.add_node("extract", self.node_extract)
        g.add_node("store",   self.node_store)
        g.add_node("error",   self.node_error)

        g.set_entry_point("extract")
        g.add_conditional_edges("extract", self._route_after_extract, {
            "store": "store", "error": "error"
        })
        g.add_edge("store", END)
        g.add_edge("error", END)

        return g.compile()

    def build_qa_graph(self):
        """
        Q&A-only mode (documents already ingested).
        Flow: query → generate → END

        Usage: api/routes/rag.py
        Input: { question: "..." }
        """
        g = StateGraph(MedicalState)
        g.add_node("query",    self.node_query)
        g.add_node("generate", self.node_generate)

        g.set_entry_point("query")
        g.add_edge("query", "generate")
        g.add_edge("generate", END)

        return g.compile()

    def build_full_pipeline(self):
        """
        Full mode: ingestion + Q&A in a single pass.
        Flow: extract → store → query → generate → END

        Usage: real-time processing (upload + immediate question).
        Input: { text: "...", question: "..." }
        """
        g = StateGraph(MedicalState)
        g.add_node("extract", self.node_extract)
        g.add_node("store", self.node_store)
        g.add_node("query", self.node_query)
        g.add_node("generate", self.node_generate)
        g.add_node("error", self.node_error)

        g.set_entry_point("extract")
        g.add_conditional_edges("extract", self._route_after_extract, {
            "store": "store", "error": "error"
        })
        g.add_conditional_edges("store", self._route_after_store, {
            "query": "query", END: END
        })
        g.add_edge("query", "generate")
        g.add_edge("generate", END)
        g.add_edge("error", END)

        return g.compile()

    # Public helpers — used by api/routes/rag.py and Airflow

    def ingest_text(self, text: str) -> dict:
        """Ingests a medical text into the store."""
        graph = self.build_ingestion_graph()
        state = graph.invoke({
            "text": text, "question": "", "entities": None,
            "retrieved": None, "answer": "", "error": None,
        })
        return {"stored": state.get("entities"), "error": state.get("error")}

    def ask_question(self, question: str) -> str:
        """Asks a question and returns the generated answer."""
        graph = self.build_qa_graph()
        state = graph.invoke({
            "text": "", "question": question, "entities": None,
            "retrieved": None, "answer": "", "error": None,
        })
        return state["answer"]

    def ingest_and_ask(self, text: str, question: str) -> dict:
        """Ingestion + Q&A in a single pass."""
        graph = self.build_full_pipeline()
        state = graph.invoke({
            "text": text, "question": question, "entities": None,
            "retrieved": None, "answer": "", "error": None,
        })
        stored = state.get("entities")
        return {
            "stored_disease": stored["disease"] if stored else None,
            "answer": state["answer"],
            "error": state.get("error"),
        }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    kg = MedicalKnowledgeGraph()

    # Inject demo entity directly
    kg._save_entity({
        "disease": "type 2 diabetes",
        "symptoms": ["fatigue", "frequent urination", "blurred vision", "increased thirst"],
        "causes": ["insulin resistance", "obesity", "sedentary lifestyle"],
        "treatments": ["metformin", "lifestyle changes", "blood sugar monitoring"],
        "severity": "medium",
    })
    print("✅ Demo entity injected")

    answer = kg.ask_question("What are the symptoms and treatments of type 2 diabetes?")
    print(f"\n💬 Answer:\n{answer}")


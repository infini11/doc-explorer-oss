# Doc-Explorer Open Source Software

> Production-Grade Document Intelligence Platform — Cloud-Agnostic MLOps

---

## 📌 Project Overview

Doc-Explorer OSS is a cloud-agnostic, production-ready MLOps platform that industrializes end-to-end document intelligence pipelines from raw document ingestion to RAG-powered question answering.

Built entirely on open-source technologies and Kubernetes-based infrastructure, the platform combines classical ML and LLMs to turn unstructured documents into structured, queryable knowledge.

## What it does

A document goes in. Knowledge comes out.

1. **Ingest** — REST API validates and stores any document (PDF, DOCX, TXT)
2. **Process** — LLM extracts entities and builds a knowledge graph
3. **Train** — XGBoost classifies documents using LLM-generated embeddings
4. **Serve** — RAG pipeline answers natural language questions over documents
5. **Monitor** — Embedding drift and model performance tracked in real time

## Use case — Medical AI Assistant

Demonstrated on a medical document corpus where the system answers:
> "What are the causes, symptoms and treatments of type 2 diabetes?"

Not by retrieving similar snippets — but by traversing structured medical knowledge extracted from source documents.

It ensures:

- Reproducibility
- Scalability
- Observability
- Environment isolation (dev / staging / prod)

The goal of this project is to demonstrate full ownership of the ML lifecycle without relying on managed cloud ML services, following modern ML Platform Engineering best practices.

---

## 🎯 Objectives

- Build an end-to-end automated ML lifecycle
- Ensure training-serving consistency
- Implement reproducible infrastructure
- Enable CI/CD for ML systems
- Provide scalable real-time & batch inference
- Implement model monitoring & automated retraining

---

## 🏗 Architecture

```
Clients (Web / Mobile / API)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  API Gateway — FastAPI :8000                        │
│  POST /upload  POST /ask  POST /ingest-and-ask      │
└──────────────┬──────────────────────────────────────┘
               │
       ┌───────┴────────┐
       ▼                ▼
┌─────────────┐  ┌──────────────────────────────────┐
│  Ingestion  │  │  RAG Inference — LangGraph        │
│  upload.py  │  │  node_query → node_generate       │
│  chunker.py │  │  entities.json + Ollama llama3.1  │
│  FAISS      │  └──────────────────────────────────┘
└─────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│  Training Pipeline                                  │
│  XGBoost + HuggingFace embeddings + MLflow          │
└─────────────────────────────────────────────────────┘
```

---

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| **API** | FastAPI, Uvicorn, Pydantic v2 |
| **LLM / RAG** | LangChain, LangGraph, Ollama (llama3.1), FAISS |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **ML** | XGBoost, scikit-learn, HuggingFace datasets |
| **Experiment Tracking** | MLflow 2.21 |
| **Orchestration** | Apache Airflow |
| **Event Streaming** | Apache Kafka |
| **Monitoring** | Evidently AI, Prometheus, Grafana |
| **Infrastructure** | Docker, Kubernetes, Helm, Terraform |
| **CI/CD** | GitHub Actions |
| **Logging** | structlog (JSON) |

---

## 🚀 Getting Started

### Prerequisites

- Docker + Docker Compose
- Make
- Python 3.11+ (for local runs)

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/infini11/doc-explorer-oss.git
cd doc-explorer-oss

# 2. Create .env and storage directories
make setup

# 3. Build and start all services
make build
make up
```

### Services

| Service | URL |
|---|---|
| API + Swagger UI | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Ollama | http://localhost:11434 |

---

## 📖 Usage

### Upload a document
```bash
make upload FILE=./document.pdf
```

### Ask a question
```bash
make ask Q="What are the symptoms of type 2 diabetes?"
```

### Run the full RAG test suite
```bash
make test-rag
```

### Train the XGBoost classifier
```bash
make train          # runs in Docker
make train-local    # runs locally (MLflow must be up)
```

### All available commands
```bash
make help
```

---

## 🔄 Data Flow

```
POST /upload
  → validate (extension + size)
  → save to storage/uploads/
  → chunker.py — RecursiveCharacterTextSplitter (512 chars, overlap 64)
  → FAISS index — all-MiniLM-L6-v2 embeddings (384D)
  → entity_extractor.py — Ollama extracts disease/symptoms/causes/treatments
  → MedicalKnowledgeGraph — persists entities to entities.json

POST /ask
  → LangGraph: node_query → node_generate
  → Match entities in entities.json
  → Ollama llama3.1 generates answer from structured context
```

---

## 🔥 Advanced MLOps Features

- Versioned ML pipelines
- Experiment tracking & model registry
- Feature consistency between training & serving
- Automated drift detection
- Automatic retraining trigger
- Canary deployments (90% stable / 10% new model)
- Environment isolation (dev, staging, prod)
- Full system & model observability

---

## 🧪 Testing

```bash
make test-rag       # End-to-end curl tests
make test           # pytest unit + integration
```

---

## ⚙️ CI/CD

**CI** (`ci.yml`) — triggered on every push:
- Lint with ruff
- Unit tests with pytest
- Docker build

**CD** (`cd.yml`) — triggered on merge to main:
- Build + push Docker image
- Deploy to staging via Helm
- Manual approval → deploy to prod

---

## 📖 License

This project is open-source.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`feature/your-feature`)
3. Commit your changes
4. Push and open a Pull Request

---

## ⭐ Project Vision

Doc-Explorer OSS demonstrates how to build a fully industrialized ML platform using open-source tools only — cloud-agnostic, scalable, reproducible, and production-ready.
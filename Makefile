# =============================================================================
# Makefile — Doc-Explorer OSS
# =============================================================================
# Usage : make <target>
# List  : make help
# =============================================================================

COMPOSE = docker compose

# SETUP
setup: ## Create .env from .env.local and storage directories
	@if [ ! -f .env ] || [ ! -s .env ]; then \
		cp .env.local .env; \
		echo "✅ .env created from .env.local — edit it before running"; \
	else \
		echo "ℹ️  .env already exists"; \
	fi
	@mkdir -p storage/uploads storage/faiss_index storage/knowledge_graph storage/models
	@echo "✅ Storage directories created"

# DOCKER — Main services (api + mlflow + ollama)
build: ## Build all images (api + mlflow)
	$(COMPOSE) build

up: ## Start all services (api + mlflow + ollama)
	$(COMPOSE) up -d

down: ## Stop all services
	$(COMPOSE) down

restart: ## Restart all services
	$(COMPOSE) down && $(COMPOSE) up -d

status: ## Show running containers
	$(COMPOSE) ps

# LOGS
logs: ## Follow logs of all services
	$(COMPOSE) logs -f 

api-logs: ## Follow API logs only
	$(COMPOSE) logs -f api

mlflow-logs: ## Follow MLflow logs only
	$(COMPOSE) logs -f mlflow

ollama-logs: ## Follow Ollama logs only
	$(COMPOSE) logs -f ollama

# TRAINING
train: ## Train the model
	$(COMPOSE) --profile training build training
	$(COMPOSE) --profile training run --rm training

train-local:
	python pipelines/training/train.py

# API — Quick test commands
health: ## Check API health
	curl -s http://localhost:8000/healthz | python3 -m json.tool

ask: ## Ask a question to the API (set question with Q="your question here")
	curl -s -X POST http://localhost:8000/api/v1/ask \
		-H "Content-Type: application/json" \
		-d '{"question": "$(Q)"}' | python3 -m json.tool

upload: ## Upload a document to the API (set file path with FILE=./path/to/file)
	curl -s -X POST http://localhost:8000/api/v1/upload \
		-F "file=@$(FILE)" | python3 -m json.tool

test-rag: ## Run RAG test script (make sure API is running with make up)
	bash tests/test_rag.sh

# OPEN INTERFACES
open-api: ## Open FastAPI Swagger UI in browser
	xdg-open http://localhost:8000/docs 2>/dev/null || open http://localhost:8000/docs

open-mlflow: ## Open MLflow UI in browser
	xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000

# CLEAN
clean: ## Stop containers and remove images
	$(COMPOSE) down --rmi local

clean-all: ## Stop containers, remove images AND volumes (⚠ deletes all data)
	$(COMPOSE) down -v --rmi local

clean-storage: ## Delete all files in storage/ (keeps directory structure)
	rm -rf storage/uploads/* storage/faiss_index/* storage/knowledge_graph/* storage/models/*
	@echo "✅ Storage cleaned"

# HELP

help: ## Show this help
	@echo ""
	@echo "Doc-Explorer OSS — Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make up"
	@echo "  make ask Q=\"What are the symptoms of diabetes?\""
	@echo "  make upload FILE=./document.pdf"
	@echo "  make train"
	@echo ""

.PHONY: setup build up down restart status logs api-logs mlflow-logs ollama-logs \
        train train-local health ask upload test-rag open-api open-mlflow \
        clean clean-all clean-storage help
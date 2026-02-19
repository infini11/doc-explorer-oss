# Doc-Explorer Open Source Software

> Production-Grade Document Intelligence Platform (Cloud-Agnostic MLOps)

---

## 📌 Project Overview

Doc-Explorer OSS is a cloud-agnostic, production-ready MLOps platform
that industrializes end-to-end document intelligence pipelines from
raw document ingestion to RAG-powered question answering.

Built entirely on open-source technologies and Kubernetes-based
infrastructure, the platform combines classical ML and LLMs to turn
unstructured documents into structured, queryable knowledge.

## What it does

A document goes in. Knowledge comes out.

1. **Ingest** — REST API validates and stores any document (PDF, DOCX, TXT)
2. **Process** — LLM extracts entities and builds a knowledge graph
3. **Train**  — XGBoost classifies documents using LLM-generated embeddings
4. **Serve**  — RAG pipeline answers natural language questions over documents
5. **Monitor** — Embedding drift and model performance tracked in real time

## Use case — Medical AI Assistant

Demonstrated on a medical document corpus where the system answers:
> "What are the causes, symptoms and treatments of type 2 diabetes?"

Not by retrieving similar snippets — but by traversing structured
medical knowledge extracted from source documents.

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

## 🏗 High-Level Architecture

The platform is composed of modular services orchestrated on Kubernetes:

### 1️⃣ Document Ingestion
- REST API via FastAPI  
- Storage or event-driven ingestion  

### 2️⃣ Event Streaming
- Apache Kafka for asynchronous processing  

### 3️⃣ Orchestration
- Apache Airflow DAGs for training pipelines  

### 4️⃣ Feature Store
- Feast for consistent training/serving features  

### 5️⃣ Experiment Tracking & Model Registry
- MLflow  

### 6️⃣ Model Serving
- FastAPI inference service deployed on Kubernetes  
- Horizontal Pod Autoscaling (HPA)  

### 7️⃣ Monitoring & Observability
- Prometheus for metrics  
- Grafana dashboards  
- Evidently AI for drift detection  

### 8️⃣ Infrastructure Provisioning
- Terraform for infrastructure as code  
- Ansible + Helm for deployment automation  

---

## 🛠 Tech Stack

### 🔹 Core
- Python (FastAPI, Scikit-learn, TensorFlow)  
- Docker  
- Kubernetes  

### 🔹 ML Lifecycle
- MLflow (experiment tracking & registry)  
- Feast (feature store)  
- Apache Airflow (orchestration)  
- Great Expectations (data validation)  
- Evidently AI (drift detection)  

### 🔹 Infrastructure & DevOps
- Terraform  
- Ansible  
- Helm  
- GitHub Actions  
- Prometheus  
- Grafana  
- Apache Kafka  

---

## ⚙️ CI/CD Strategy

### 🔄 Continuous Integration

- Linting (black, flake8)  
- Unit tests (pytest)  
- Data validation tests  
- Docker image build  
- Container security scanning  
- Push to container registry  

### 🚀 Continuous Deployment

- Infrastructure provisioning via Terraform  
- Application deployment via Ansible + Helm  
- Blue/Green deployment strategy  
- Canary model rollout  
- Automated rollback  
- Model promotion (staging → production)  

## 🚀 Running Locally (Development Mode)
```
make up
```

Services available:
- PI → http://localhost:8000
- MLflow UI → http://localhost:5000
- Airflow UI → http://localhost:8080
- Grafana → http://localhost:3000

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

## 📖 License

This project is open-source.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create a feature branch (`feature/your-feature`)  
3. Commit your changes  
4. Push to your branch  
5. Open a Pull Request  

---

## ⭐ Project Vision

Doc-Explorer OSS demonstrates how to build a fully industrialized ML platform using open-source tools only — cloud-agnostic, scalable, reproducible, and production-ready.

#!/bin/bash
# =============================================================================
# Doc-Explorer OSS — Test complet des endpoints RAG via curl
# Compatible endpoint UploadFile (multipart/form-data)
# =============================================================================

BASE="http://127.0.0.1:8000/api/v1"
SEP="─────────────────────────────────────────────────────────"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
RESET='\033[0m'

header()  { echo -e "\n${BLUE}${SEP}${RESET}"; echo -e "${BLUE}$1${RESET}"; echo -e "${BLUE}${SEP}${RESET}"; }
ok()      { echo -e "${GREEN}✅  $1${RESET}"; }
info()    { echo -e "${YELLOW}▶  $1${RESET}"; }
err()     { echo -e "${RED}❌  $1${RESET}"; }

# =============================================================================
# 0. HEALTH CHECK
# =============================================================================
header "0. Health check — GET /healthz"

info "Vérification que l'API répond..."
HEALTH=$(curl -s http://127.0.0.1:8000/healthz)
echo "$HEALTH" | python3 -m json.tool 2>/dev/null || echo "$HEALTH"

if echo "$HEALTH" | grep -q '"ok"'; then
    ok "API opérationnelle"
else
    err "API non accessible — lance 'make up' d'abord"
    exit 1
fi

# =============================================================================
# 1. INGESTION — POST /api/v1/upload (multipart)
# =============================================================================
header "1. Ingestion — POST /api/v1/upload"
info "Création d'un fichier temporaire pour ingestion..."

DIABETES_FILE="diabetes_test.txt"

cat <<EOF > $DIABETES_FILE
Type 2 diabetes is a chronic metabolic disorder characterized by high blood sugar levels (hyperglycemia).
It occurs when the body becomes resistant to insulin or fails to produce enough insulin.
Common symptoms include frequent urination (polyuria), excessive thirst (polydipsia),
unexplained weight loss, fatigue, blurred vision, and slow wound healing.
Risk factors include obesity, sedentary lifestyle, family history, and age over 45.
Treatment involves lifestyle modifications such as diet and exercise,
oral medications like metformin, and sometimes insulin therapy.
Complications include cardiovascular disease, nephropathy, and neuropathy.
EOF

echo ""
info "Envoi du fichier via multipart/form-data..."
echo ""

INGEST_RESP=$(curl -s -X POST "$BASE/upload" \
  -F "file=@$DIABETES_FILE" \
  -F "document_id=test-diabetes-001")

echo "$INGEST_RESP" | python3 -m json.tool 2>/dev/null || echo "$INGEST_RESP"

if echo "$INGEST_RESP" | grep -qi "error"; then
    err "Ingestion échouée"
else
    ok "Ingestion terminée"
fi

sleep 2

# =============================================================================
# 2. QUESTION RAG — POST /api/v1/ask
# =============================================================================
header "2. Question RAG — POST /api/v1/ask"

ask_question () {
  QUESTION=$1
  echo ""
  echo "📤 Question : $QUESTION"
  echo ""
  curl -s -X POST "$BASE/ask" \
      -H "Content-Type: application/json" \
      -d "{\"question\": \"$QUESTION\"}" | python3 -m json.tool
}

ask_question "What are the main symptoms of type 2 diabetes?"
ask_question "What treatments are available for diabetes?"
ask_question "What causes type 2 diabetes?"

# =============================================================================
# 3. PIPELINE COMPLET — POST /api/v1/ingest-and-ask
# =============================================================================
header "3. Pipeline complet — POST /api/v1/ingest-and-ask"

HYPERTENSION_FILE="hypertension_test.txt"

cat <<EOF > $HYPERTENSION_FILE
Hypertension, also known as high blood pressure, is a condition where blood pressure is consistently too high.
It is often called the silent killer.
Causes include genetics, high salt diet, obesity, stress, and inactivity.
Complications include stroke, heart attack, kidney failure, and vision loss.
Treatments include lifestyle changes, ACE inhibitors, calcium channel blockers, and diuretics.
EOF

echo ""
info "Envoi fichier + question en une seule requête..."

curl -s -X POST "$BASE/ingest-and-ask" \
  -F "file=@$HYPERTENSION_FILE" \
  -F "question=What are the complications and treatments for hypertension?" \
  | python3 -m json.tool

ok "Pipeline complet exécuté"

# =============================================================================
# 4. VÉRIFICATION KNOWLEDGE GRAPH
# =============================================================================
header "4. Vérification du Knowledge Graph local"

if [ -f "storage/knowledge_graph/entities.json" ]; then
    cat storage/knowledge_graph/entities.json | python3 -m json.tool
    ok "Knowledge Graph persisté"
else
    info "Fichier non trouvé localement (normal si Docker)"
    info "Commande Docker :"
    echo "docker exec doc-explorer-api cat storage/knowledge_graph/entities.json"
fi

# =============================================================================
# 5. CAS SANS MATCH
# =============================================================================
header "5. Cas sans match"

ask_question "What are the symptoms of malaria?"

# =============================================================================
# CLEANUP
# =============================================================================
rm -f $DIABETES_FILE
rm -f $HYPERTENSION_FILE

header "Résumé des endpoints testés"
echo ""
echo "POST /api/v1/upload           → Ingestion fichier (multipart)"
echo "POST /api/v1/ask              → Recherche KG + génération LLM"
echo "POST /api/v1/ingest-and-ask   → Pipeline complet fichier + question"
echo ""
echo "Swagger UI : http://127.0.0.1:8000/docs"
echo ""

ok "Tests terminés ✨"

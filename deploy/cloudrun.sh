#!/usr/bin/env bash
# Deploy the public demo to Google Cloud Run (read-only, Gemini rate-limited). Run from the repo root, with the
# built scene (web/boiler_room_clean.spz + .ply) and data/builder_pack.xlsx present, after `gcloud auth login`.
#
#   bash deploy/cloudrun.sh <project-id> [region]
#
# The Gemini key goes into Secret Manager (asked once, never written to disk or the image).
set -euo pipefail

PROJECT="${1:?usage: bash deploy/cloudrun.sh <project-id> [region]}"
REGION="${2:-europe-north1}"  # Finland, next to Valmet; any Cloud Run region works
SERVICE="molfars-demo"
SECRET="gemini-api-key"

for f in data/builder_pack.xlsx web/boiler_room_clean.spz web/boiler_room_clean.ply web/scene.json; do
  [ -f "$f" ] || { echo "missing $f: build the scene first (see README Quick start)"; exit 1; }
done

gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

if ! gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  read -rsp "Gemini API key (from aistudio.google.com/apikey; input hidden): " KEY; echo
  printf '%s' "$KEY" | gcloud secrets create "$SECRET" --replication-policy=automatic --data-file=-
  unset KEY
fi
# the service runs as the default compute account; it may read this one secret and nothing else
SA="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
gcloud secrets add-iam-policy-binding "$SECRET" --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor >/dev/null

# --max-instances 1: the Gemini limits are counted in memory, and it caps the bill; --min-instances 0: free when idle
gcloud run deploy "$SERVICE" --source . --region "$REGION" --allow-unauthenticated \
  --memory 1Gi --cpu 1 --min-instances 0 --max-instances 1 --concurrency 40 --timeout 120 \
  --set-env-vars PUBLIC_DEMO=1,GEMINI_PER_IP_HOUR=20,GEMINI_PER_DAY=300 \
  --set-secrets GEMINI_API_KEY="$SECRET":latest

gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'

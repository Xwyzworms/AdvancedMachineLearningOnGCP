MODEL_NAME=babyweight
MODEL_VERSION=ML_gcp
MODEL_LOCATION= $(gsutils ls gs://${BUCKET}/babyweight/export/exporter | tail -l)
BUCKET = qwiklabs-gcp-00-c50e984ef845
REGION = us-central1

gcloud config set ai_platform/region global

echo "Deploying Model to GCP '$MODEL_NAME', version '$MODEL_VERSION' from '$MODEL_LOCATION' "
echo "... Hold on a sec, this may take a while ..."


gloud ai-platform models create ${MODEL_NAME} -- regions ${REGION}

gcloud ai-platform versions create ${MODEL_VERSION} -- model ${MODEL_NAME} -- origin ${MODEL_LOCATION} --runtime-version 2.25
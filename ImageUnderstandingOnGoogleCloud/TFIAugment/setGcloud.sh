
$PROJECT=$(wc -l < $1)
$REGION=$(wc -l < $2)

gcloud config set project $PROJECT
gcloud config set compute/region $REGION

echo "SucessFully Executed"
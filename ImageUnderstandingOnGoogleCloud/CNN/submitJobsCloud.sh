model_type="cnn"
job_dir="gs://qwiklabs-gcp-01-e9771587c0e2/mnist_cnn" # Change This Later
job_name="PrimGanteng"
bucket="qwiklabs-gcp-01-e9771587c0e2"
region="us-central1"
image_uri="gcr.io/qwiklabs-gcp-01-e9771587c0e2/mnist_models"
echo $model_type $job_dir $job_name

gcloud ai-platform jobs submit training $job_name --staging-bucket=gs://$bucket --region=$region --master-image-uri=$image_uri --scale-tier=BASIC_GPU --job-dir=$job_dir
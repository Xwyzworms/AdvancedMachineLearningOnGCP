job_dir="mnist_models\models\mnist_linear_model1"
python -m mnist_models.trainer.task --job_dir="$job_dir" --epochs=5 steps_per_epoch=50 --model_type="linear" \
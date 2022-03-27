
import argparse
import json
import os
import sys

from . import model
def _parse_arguments(args):
    
    parse = argparse.ArgumentParser()
    parse.add_argument(
		"--model_type",
		help="Which Model You want to use ?",
		type=str, default="linear"  )
    parse.add_argument(
		"--epochs",
		help="How many epochs you want to train ?",
		type=int, default=10)
  
    parse.add_argument(
		"--steps_per_epoch",
		help="How Many Gradient Per epoch to Train ? ",
		type=int, default=100	)
    parse.add_argument(
		"--job-dir",
		help="Directory For saving The model",
		type=str, default="mnist/models")
    
    return parse.parse_known_args(args)

def main():
    args = _parse_arguments(sys.argv[1:])[0]
    
    trial_id = json.loads(
		os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
    print("CONSOLO.log")
    model_layers = model.get_layers(args.model_type)
    image_model = model.build_model(model_layers, args.job_dir)
    model_history = model.train_and_evaluate(
		image_model, args.epochs, args.steps_per_epoch, args.job_dir)
    
if __name__ == "__main__":
	main()

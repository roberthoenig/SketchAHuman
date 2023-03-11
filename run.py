import argparse
import logging
import pprint
import torch

from models.ShapeModel import ShapeModel
from models.ShapeModelNoDiffusion import ShapeModelNoDiffusion
from utils.experiment_utils import init_logger, prep_experiment_dir, load_experiment_config, init_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=
                                     """
                                     Run an experiment.
                                     """)
    parser.add_argument("experiment", type=str,
                        help=f"Name of the experiment that you want to run.",
                        )
    args = parser.parse_args()

    config = load_experiment_config(args.experiment)
    experiment_dir = prep_experiment_dir(args.experiment)
    config["experiment_dir"] = experiment_dir
    if config["device"] == 'gpu':
        config["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_seeds(config["seed"])

    init_logger(str(experiment_dir / 'output.log'))

    logging.info(pprint.pformat(config))

    if config["model"] == "ShapeModel":
        model = ShapeModel(config)
    elif config["model"] == "ShapeModelNoDiffusion":
        model = ShapeModelNoDiffusion(config)
    else:
        raise Exception(f"Unknown model {config['model']}")
    logging.info("Training model...")
    if config["type"] == "train":
        model.train()
    elif config["type"] == "test":
        model.test()
    else:
        raise Exception(f"Unkown experiment type {config['type']}")


if __name__ == "__main__":
    main()
"""Main CLI command"""

import argparse
import os
import pathlib
import sys
from typing import Final, cast

import hydra

from tensorial import reaxkit as rkit

COMMAND: Final[str] = "command"
TRAIN: Final[str] = "train"
PREDICT: Final[str] = "predict"
TRAIN_SCRIPT_DEFAULT: Final[str] = "configs/train.yaml"
EVAL_SCRIPT_DEFAULT: Final[str] = "configs/eval.yaml"
REAX_COMMAND: Final[str] = "REAX_COMMAND"


def main_cli():
    os.environ[REAX_COMMAND] = " ".join(sys.argv)

    parser = argparse.ArgumentParser("tensorial")
    commands = parser.add_subparsers(dest=COMMAND, required=True)

    # The 'train' command
    train_parser = commands.add_parser(TRAIN, help="Train a model")
    train_parser.add_argument(
        "-i",
        "--input",
        nargs="?",
        type=pathlib.Path,
        help="Input file with training details",
        default=TRAIN_SCRIPT_DEFAULT,
    )
    # The 'predict' command
    train_parser = commands.add_parser(PREDICT, help="Make predictions using a trained model")
    train_parser.add_argument(
        "-i",
        "--input",
        nargs="?",
        type=pathlib.Path,
        help="Input file with evaluation details",
        default=EVAL_SCRIPT_DEFAULT,
    )

    # Parse the args
    args, _rest = parser.parse_known_args()

    if args.command == TRAIN:
        # Set the command line arguments to what remains so hydra can deal with it
        sys.argv = sys.argv[0:1] + _rest
        script_path: pathlib.Path = args.input
        hydra_fn = hydra.main(
            version_base="1.3",
            config_path=str(script_path.parent.absolute()),
            config_name=script_path.stem,
        )(rkit.train.main)
    elif args.command == PREDICT:
        # Set the command line arguments to what remains so hydra can deal with it
        sys.argv = sys.argv[0:1] + _rest

        script_path = cast(pathlib.Path, args.input)
        if script_path.is_dir():
            script_path = script_path / rkit.config.DEFAULT_CONFIG_FILE
            if not script_path.is_file():
                print(f"Could not find configuration file: {script_path}")
                sys.exit(1)

        hydra_fn = hydra.main(
            version_base="1.3",
            config_path=str(script_path.parent.absolute()),
            config_name=script_path.stem,
        )(rkit.evaluate.main)
    else:
        raise ValueError(f"Unrecognised command '{args.command}'")

    # Call Hydra to launch the actual command
    hydra_fn()


if __name__ == "__main__":
    main_cli()

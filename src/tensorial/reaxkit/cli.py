"""Main CLI command"""

import argparse
from importlib import resources
import os
import pathlib
import re
import shutil
import sys
from typing import Final, cast

import hydra

from tensorial import reaxkit as rkit

COMMAND: Final[str] = "command"
TRAIN: Final[str] = "train"
PREDICT: Final[str] = "predict"
EXAMPLES: Final[str] = "examples"
TRAIN_SCRIPT_DEFAULT: Final[str] = "configs/train.yaml"
EVAL_SCRIPT_DEFAULT: Final[str] = "configs/eval.yaml"
REAX_COMMAND: Final[str] = "REAX_COMMAND"


def _examples_root():
    """The directory holding all bundled example configurations."""
    return resources.files("tensorial").joinpath("examples")


def _dotted_example_names(root_path: pathlib.Path):
    """Return every example as a dotted name, e.g. ``mlip``.

    An example is any directory that directly contains a ``train.yaml``.
    """
    names = set()
    for train_yaml in root_path.rglob("train.yaml"):
        rel = train_yaml.parent.relative_to(root_path)
        if str(rel) != ".":
            names.add(rel.as_posix().replace("/", "."))
    return sorted(names)


def _variant_group(example_path: pathlib.Path):
    """Return the config group that gives an example its variants, if any.

    A variant group is the ``data`` group: an example differs from its
    siblings only in which dataset it trains on (e.g. the ``data`` group of
    ``mlip`` holding ``si.yaml`` and ``qm9.yaml``).  Other Hydra groups such
    as ``model`` or ``trainer`` are just alternative hyper-parameters and are
    not exposed as separate examples.
    """
    group_path = example_path / "data"
    if not group_path.is_dir():
        return None
    yaml_files = [p for p in group_path.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")]
    if len(yaml_files) >= 2:
        return "data"
    return None


#: Matches ``# requires: <path> [<path> ...]`` comments in example files.
REQUIRES_RE: Final[re.Pattern[str]] = re.compile(r"^\s*#\s*requires:\s*(.+?)\s*$", re.M)


def _required_files(yaml_file: pathlib.Path):
    """Return the paths declared in a variant file's ``# requires:`` comments.

    Each declared path is taken relative to the variant file's directory and
    copied next to it when the example is initialised, so that the dataset the
    configuration points at (e.g. ``./data/sitraj.xyz``) lands where
    ``${paths.data_dir}`` expects it.
    """
    files = []
    for line in REQUIRES_RE.findall(yaml_file.read_text(encoding="utf-8")):
        files.extend(part for part in line.split() if part)
    return files


def _list_examples():
    with resources.as_file(_examples_root()) as root_path:
        names = _dotted_example_names(root_path)
    print("Available examples:")
    for name in names:
        base = name.split(".")[0]
        with resources.as_file(_examples_root()) as root_path:
            example_path = root_path / base
            group = _variant_group(example_path)
        if group:
            for yaml_file in sorted((example_path / group).iterdir()):
                if yaml_file.is_file() and yaml_file.suffix in (".yaml", ".yml"):
                    print(f"  - {base}.{yaml_file.stem}")
        else:
            print(f"  - {name}")


def _split_name(name: str):
    """Split an example name into (base, group, choice).

    ``mlip`` -> ``("mlip", None, None)``;  ``mlip.si`` -> ``("mlip", "data", "si")``.
    """
    parts = name.split(".")
    if len(parts) == 1:
        return parts[0], None, None
    if len(parts) == 2:
        base, choice = parts
        with resources.as_file(_examples_root()) as root_path:
            group = _variant_group(root_path / base)
        if not group:
            raise ValueError(f"Example '{base}' has no variants.")
        return base, group, choice
    raise ValueError(f"Invalid example name '{name}' (expected 'name' or 'name.<choice>')")


def _init_example(name: str, dest: pathlib.Path):
    with resources.as_file(_examples_root()) as root_path:
        base, group, choice = _split_name(name)
        example_path = root_path / base
        if not example_path.is_dir():
            available = ", ".join(_dotted_example_names(root_path))
            print(f"Example '{name}' not found. Available examples: {available}")
            sys.exit(1)
        if group is not None:
            if not (example_path / group / f"{choice}.yaml").is_file():
                available = ", ".join(
                    f"{base}.{p.stem}"
                    for p in (example_path / group).iterdir()
                    if p.is_file() and p.suffix in (".yaml", ".yml")
                )
                print(f"Example '{name}' not found. Available examples: {available}")
                sys.exit(1)

        # Variants of the same example all share their files and differ only
        # in the ``data`` Hydra default, so a plain base name resolves to the
        # default ``data`` choice already present in ``train.yaml``.
        dest_folder = pathlib.Path(dest) / (f"{base}-{choice}" if group is not None else base)
        print(f"Initializing example '{name}' in {dest_folder}")
        dest_folder.mkdir(parents=True, exist_ok=True)
        for item in example_path.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_folder / item.name)
            elif group is not None and item.is_dir() and item.name == group:
                # Only copy the dataset file this variant uses, not the
                # sibling datasets the user didn't ask for.
                target = dest_folder / item.name
                target.mkdir(parents=True, exist_ok=True)
                source_yaml = item / f"{choice}.yaml"
                shutil.copy2(source_yaml, target / f"{choice}.yaml")
                # Copy the data files the variant declares it needs.
                for rel in _required_files(source_yaml):
                    src = item / rel
                    if not src.is_file():
                        print(f"error: '{name}' requires '{rel}' but {src} does not exist")
                        sys.exit(1)
                    dst = target / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            elif item.is_dir():
                shutil.copytree(item, dest_folder / item.name, dirs_exist_ok=True)

    if group is not None:
        _override_group(dest_folder, group, choice)


def _override_group(dest_folder: pathlib.Path, group: str, choice: str):
    """Rewrite the Hydra default for ``group`` in ``train.yaml`` to ``choice``."""
    train_yaml = dest_folder / "train.yaml"
    if not train_yaml.is_file():
        return
    text = train_yaml.read_text(encoding="utf-8")
    pattern = rf"^(\s*)[-]\s*{re.escape(group)}:\s*\S+.*$"
    replaced, count = re.subn(pattern, rf"\1- {group}: {choice}", text, count=1, flags=re.M)
    if count == 0:
        # The default may be written as ``- {group: default}``.
        pattern = rf"^(\s*)[-]\s*{{\s*{re.escape(group)}:\s*\S+\s*}}.*$"
        replaced, count = re.subn(pattern, rf"\1- {group}: {choice}", text, count=1, flags=re.M)
    if count:
        train_yaml.write_text(replaced, encoding="utf-8")


def _run_examples(tokens: list):
    """Dispatch the `examples` sub-arguments.

    Accepts ``list``, ``init <name> [dest]`` or the shorthand
    ``<name> [dest]``.  With no arguments, equivalent to ``list``.
    """
    if not tokens:
        _list_examples()
        return
    if tokens[0] == "list":
        _list_examples()
    elif tokens[0] == "init":
        if len(tokens) < 2:
            print("error: 'examples init' requires an example name (e.g. mlip.si)")
            sys.exit(2)
        name = tokens[1]
        dest = pathlib.Path(tokens[2]) if len(tokens) > 2 else pathlib.Path(".")
        _init_example(name, dest)
    else:
        name = tokens[0]
        dest = pathlib.Path(tokens[1]) if len(tokens) > 1 else pathlib.Path(".")
        _init_example(name, dest)


def main_cli():
    """Entry point for the ``tensorial`` command-line tool.

    Parses the command-line arguments and dispatches to the ``train`` or
    ``predict`` subcommand, which in turn run the appropriate Hydra config script.
    """
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

    # The 'examples' command.  Sub-arguments are handled manually from the
    # trailing (unparsed) arguments so that both
    # ``tensorial examples mlip.si`` and ``tensorial examples init mlip.si``
    # work.
    examples_parser = commands.add_parser(EXAMPLES, help="List or copy example configurations")
    examples_parser.add_argument(
        "examples_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="e.g. 'list', 'init <name> [dest]', or simply '<name> [dest]'",
    )

    # Parse the args
    args, _rest = parser.parse_known_args()

    if args.command == EXAMPLES:
        _run_examples(list(args.examples_args))
        return

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

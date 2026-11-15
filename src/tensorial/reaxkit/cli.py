"""Command-line entry point for the ``tensorial`` executable.

Installing the package exposes a ``tensorial`` command (see
``[project.scripts]`` in ``pyproject.toml``) with three subcommands:

``tensorial train -i <config>``
    Train a model from a Hydra config (see :func:`tensorial.reaxkit.train`).

``tensorial predict -i <config-or-dir>``
    Evaluate a trained checkpoint (see :func:`tensorial.reaxkit.evaluate`).

``tensorial examples [list [-l|--long] [--json] | <name> | init <name> [<dest>]]``
    List the bundled example configurations (with a short summary of each, or full
    descriptions with ``--long``), show the full description of one, or copy one.

``train`` and ``predict`` forward their remaining arguments to Hydra, so any
config value can be overridden on the command line.  See the
:doc:`command-line interface guide </starting/cli_guide>` for details.
"""

import argparse
from importlib import resources
import json
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


#: Matches a ``# <key>: ...`` directive comment for one of the known keys.
DIRECTIVE_RE: Final[re.Pattern[str]] = re.compile(r"^#\s*(summary|requires)\s*:\s*(.*?)\s*$")


def _example_info(yaml_file: pathlib.Path) -> tuple[str | None, str]:
    """Return the ``(summary, description)`` of an example from its header comments.

    The summary is the text of a ``# summary: <text>`` comment.  The description is the
    block of comment lines at the top of the file (up to the first line that is not a
    comment), with directive comments such as ``# summary:`` and ``# requires:`` removed.
    """
    summary = None
    lines = []
    for line in yaml_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        match = DIRECTIVE_RE.match(stripped)
        if match:
            if match.group(1) == "summary" and match.group(2):
                summary = match.group(2)
            continue
        text = stripped[1:]
        lines.append(text[1:] if text.startswith(" ") else text)

    # Drop leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return summary, "\n".join(lines)


def _examples() -> list[tuple[str, pathlib.Path]]:
    """Return ``(name, yaml_file)`` for every example, including each variant.

    ``yaml_file`` is the variant's file in the ``data`` group (e.g. ``mlip/data/si.yaml`` for
    ``mlip.si``), or the example's ``train.yaml`` if it has no variants.  It is the file whose
    header comments describe the example.

    The paths are only valid while the package resources are on disk, which is always the case
    for a regular (non-zipped) install.
    """
    examples = []
    with resources.as_file(_examples_root()) as root_path:
        for name in _dotted_example_names(root_path):
            base = name.split(".")[0]
            example_path = root_path / base
            group = _variant_group(example_path)
            if group:
                for yaml_file in sorted((example_path / group).iterdir()):
                    if yaml_file.is_file() and yaml_file.suffix in (".yaml", ".yml"):
                        examples.append((f"{base}.{yaml_file.stem}", yaml_file))
            else:
                examples.append((name, root_path / name.replace(".", "/") / "train.yaml"))
    return examples


def _example_names() -> list[str]:
    """Return the name of every example, including each variant (e.g. ``mlip.si``)."""
    return [name for name, _ in _examples()]


def _list_examples(as_json: bool = False, long: bool = False):
    """Print the available examples.

    By default, each example is shown with its one-line summary.  With ``long``, its full
    description is printed beneath it.  With ``as_json``, a JSON list of names is printed (or,
    with ``long`` too, a list of ``{"name", "summary", "description"}`` objects).
    """
    examples = [(name, *_example_info(yaml_file)) for name, yaml_file in _examples()]
    if as_json:
        if long:
            entries = [
                {"name": name, "summary": summary, "description": description}
                for name, summary, description in examples
            ]
            print(json.dumps(entries))
        else:
            print(json.dumps([name for name, *_ in examples]))
        return

    print("Available examples:")
    width = max((len(name) for name, *_ in examples), default=0)
    for i, (name, summary, description) in enumerate(examples):
        if long and i > 0:
            print()
        print(f"  - {name:<{width}}  {summary}" if summary else f"  - {name}")
        if long and description:
            for line in description.splitlines():
                print(f"      {line}".rstrip())
    if not long and any(description for *_, description in examples):
        print("\nUse 'tensorial examples list --long' for full descriptions.")


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
    """Rewrite the Hydra default for ``group`` to ``choice`` in the example's top-level configs.

    Both ``train.yaml`` and ``eval.yaml`` are updated, so evaluating a trained model uses the
    same dataset it was trained on.
    """
    for filename in ("train.yaml", "eval.yaml"):
        _override_group_in(dest_folder / filename, group, choice)


def _override_group_in(config_file: pathlib.Path, group: str, choice: str):
    """Rewrite the Hydra default for ``group`` in ``config_file`` to ``choice``."""
    if not config_file.is_file():
        return
    text = config_file.read_text(encoding="utf-8")
    pattern = rf"^(\s*)[-]\s*{re.escape(group)}:\s*\S+.*$"
    replaced, count = re.subn(pattern, rf"\1- {group}: {choice}", text, count=1, flags=re.M)
    if count == 0:
        # The default may be written as ``- {group: default}``.
        pattern = rf"^(\s*)[-]\s*{{\s*{re.escape(group)}:\s*\S+\s*}}.*$"
        replaced, count = re.subn(pattern, rf"\1- {group}: {choice}", text, count=1, flags=re.M)
    if count:
        config_file.write_text(replaced, encoding="utf-8")


def _default_choice(config_file: pathlib.Path, group: str) -> str | None:
    """Return the Hydra default chosen for ``group`` in ``config_file``, if any."""
    if not config_file.is_file():
        return None
    match = re.search(
        rf"^\s*-\s*{{?\s*{re.escape(group)}:\s*([^\s}}]+)",
        config_file.read_text(encoding="utf-8"),
        flags=re.M,
    )
    return match.group(1) if match else None


def _show_example(name: str):
    """Print the full description of an example.

    For a base name whose example has variants (e.g. ``mlip``), list the variants with their
    summaries and say which one ``init`` copies by default.
    """
    examples = _examples()
    for example_name, yaml_file in examples:
        if example_name == name:
            summary, description = _example_info(yaml_file)
            print(f"{name} — {summary}" if summary else name)
            if description:
                print()
                for line in description.splitlines():
                    print(f"  {line}".rstrip())
            print(f"\nTo copy it: tensorial examples init {name} [dest]")
            return

    variants = [(n, f) for n, f in examples if n.startswith(f"{name}.")]
    if variants:
        with resources.as_file(_examples_root()) as root_path:
            default = _default_choice(root_path / name / "train.yaml", "data")
        print(f"'{name}' has the following variants:")
        width = max(len(n) for n, _ in variants)
        for variant_name, yaml_file in variants:
            summary, _ = _example_info(yaml_file)
            print(f"  - {variant_name:<{width}}  {summary}" if summary else f"  - {variant_name}")
        print(f"\nShow one with: tensorial examples {variants[0][0]}")
        if default:
            print(f"'tensorial examples init {name}' copies the default, {name}.{default}")
        return

    available = ", ".join(n for n, _ in examples)
    print(f"Example '{name}' not found. Available examples: {available}")
    sys.exit(1)


def _run_examples(tokens: list):
    """Dispatch the `examples` sub-arguments.

    Accepts ``list [-l|--long] [--json]``, ``init <name> [dest]`` or ``<name>`` to show
    the full description of one example.  With no arguments, equivalent to ``list``.
    """
    if not tokens:
        _list_examples()
        return
    if tokens[0] in ("list", "-l", "--long"):
        # Allow ``tensorial examples -l`` as a shorthand for ``examples list -l``
        opts = tokens if tokens[0] != "list" else tokens[1:]
        _list_examples(
            as_json="--json" in opts,
            long="-l" in opts or "--long" in opts,
        )
    elif tokens[0] == "init":
        if len(tokens) < 2:
            print("error: 'examples init' requires an example name (e.g. mlip.si)")
            sys.exit(2)
        name = tokens[1]
        dest = pathlib.Path(tokens[2]) if len(tokens) > 2 else pathlib.Path(".")
        _init_example(name, dest)
    elif len(tokens) > 1:
        # ``examples <name> <dest>`` used to be a shorthand for ``init``; a destination only
        # makes sense when copying, so point the user at ``init`` rather than guess.
        print(
            "error: to copy an example use "
            f"'tensorial examples init {' '.join(tokens)}' "
            f"(or 'tensorial examples {tokens[0]}' to show its description)"
        )
        sys.exit(2)
    else:
        _show_example(tokens[0])


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
    # trailing (unparsed) arguments so that ``tensorial examples mlip.si`` (show)
    # and ``tensorial examples init mlip.si`` (copy) can share the command.
    examples_parser = commands.add_parser(
        EXAMPLES, help="List, describe or copy example configurations"
    )
    examples_parser.add_argument(
        "examples_args",
        nargs=argparse.REMAINDER,
        default=[],
        help="'list [--long] [--json]', '<name>' to describe one, or 'init <name> [dest]'",
    )

    # Parse the args
    args, _rest = parser.parse_known_args()

    if args.command == EXAMPLES:
        # Leading option-like tokens (e.g. ``tensorial examples -l``) are not captured by
        # ``REMAINDER`` and end up in ``_rest`` instead.
        _run_examples(list(args.examples_args) + _rest)
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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core import rank_zero
import omegaconf
import rich
from rich.prompt import Prompt
import rich.syntax
import rich.tree

from . import pylogger

_LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)
TREE_STYLE: Final[str] = "dim"


def print_tree(root: Mapping, name: str):
    tree = rich.tree.Tree(name, style=TREE_STYLE, guide_style=TREE_STYLE)

    queue = []

    # add all the other fields to queue (not specified in `print_order`)
    for field in root:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=TREE_STYLE, guide_style=TREE_STYLE)

        config_group = root[field]
        branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)


@rank_zero.rank_zero_only
def print_config_tree(
    cfg: omegaconf.DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "listeners",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    Args:
        cfg: A DictConfig composed by Hydra.
        print_order: Determines in what order config components are
            printed. Default is ``("data", "model", "listeners",
            "logger", "trainer", "paths", "extras")``.
        resolve: Whether to resolve reference fields of DictConfig.
            Default is ``False``.
        save_to_file: Whether to export config to the hydra output
            folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        if field in cfg:
            queue.append(field)
        else:
            _LOGGER.warning(
                "Field '%s' not found in config. Skipping '%s' config printing...", field, field
            )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w", encoding="utf-8") as file:
            rich.print(tree, file=file)


@rank_zero.rank_zero_only
def enforce_tags(cfg: omegaconf.DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config.

    Args:
        cfg: A DictConfig composed by Hydra.
        save_to_file: Whether to export tags to the hydra output folder.
            Default is ``False``.
    """
    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        _LOGGER.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with omegaconf.open_dict(cfg):
            cfg.tags = tags

        _LOGGER.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w", encoding="utf-8") as file:
            rich.print(cfg.tags, file=file)

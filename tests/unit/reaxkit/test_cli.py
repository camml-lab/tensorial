import json
import pathlib
import re

import pytest

from tensorial.reaxkit import cli


def _required_files_of(text: str, tmp_path: pathlib.Path):
    source = tmp_path / "example.yaml"
    source.write_text(text, encoding="utf-8")
    return cli._required_files(source)


def test_required_files_parse(tmp_path: pathlib.Path):
    files = _required_files_of(
        "# Top comment\n# requires: data/foo.xyz data/bar.json\nkey: value\n# requires: qux.xyz\n",
        tmp_path,
    )
    assert files == ["data/foo.xyz", "data/bar.json", "qux.xyz"]


def test_required_files_none(tmp_path: pathlib.Path):
    assert _required_files_of("key: value\n# a comment\n", tmp_path) == []


def test_init_example_copies_required_data(tmp_path: pathlib.Path):
    cli._init_example("mlip.si", tmp_path)

    out = tmp_path / "mlip-si"
    assert (out / "data" / "si.yaml").is_file()
    # The file declared by si.yaml's ``# requires:`` comment is copied.
    assert (out / "data" / "sitraj.xyz").is_file()
    # Sibling variants (and their data) are not copied.
    assert not (out / "data" / "qm9.yaml").exists()


def test_list_examples_json(capsys):
    cli._list_examples(as_json=True)
    names = json.loads(capsys.readouterr().out)
    assert "mlip.si" in names
    assert names == cli._example_names()


def test_init_example_sets_data_for_train_and_eval(tmp_path: pathlib.Path):
    cli._init_example("mlip.3bpa", tmp_path)

    out = tmp_path / "mlip-3bpa"
    for config in ("train.yaml", "eval.yaml"):
        assert re.search(r"^\s*- data: 3bpa\s*$", (out / config).read_text(), flags=re.M)


def _example_info_of(text: str, tmp_path: pathlib.Path):
    source = tmp_path / "example.yaml"
    source.write_text(text, encoding="utf-8")
    return cli._example_info(source)


def test_example_info_parse(tmp_path: pathlib.Path):
    summary, description = _example_info_of(
        "# summary: Tiny dataset; energy\n"
        "# requires: foo.xyz\n"
        "# First line of the description.\n"
        "#   an indented override=1\n"
        "#\n"
        "# Note: not a directive.\n"
        "key: value\n"
        "# Trailing comment, not part of the description\n",
        tmp_path,
    )
    assert summary == "Tiny dataset; energy"
    assert description == (
        "First line of the description.\n  an indented override=1\n\nNote: not a directive."
    )


def test_example_info_none(tmp_path: pathlib.Path):
    assert _example_info_of("key: value\n# a comment\n", tmp_path) == (None, "")


def test_bundled_examples_have_summaries():
    for name, yaml_file in cli._examples():
        summary, _ = cli._example_info(yaml_file)
        assert summary, f"Example '{name}' has no '# summary:' comment in {yaml_file.name}"


def test_list_examples_long(capsys):
    cli._list_examples(long=True)
    out = capsys.readouterr().out
    for name in cli._example_names():
        assert f"- {name}" in out
    assert "Kovács" in out  # part of the mlip.3bpa description


def test_list_examples_json_long(capsys):
    cli._list_examples(as_json=True, long=True)
    entries = json.loads(capsys.readouterr().out)
    assert [entry["name"] for entry in entries] == cli._example_names()
    assert all(entry["summary"] for entry in entries)


def test_show_example(capsys, tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli._run_examples(["mlip.3bpa"])
    out = capsys.readouterr().out
    assert out.startswith("mlip.3bpa — 3BPA")
    assert "Kovács" in out
    assert "tensorial examples init mlip.3bpa" in out
    # Showing an example must not copy anything
    assert not any(tmp_path.iterdir())


def test_show_example_variants(capsys):
    cli._run_examples(["mlip"])
    out = capsys.readouterr().out
    for name in cli._example_names():
        assert f"- {name}" in out
    assert "copies the default, mlip.si" in out


def test_show_example_unknown(capsys):
    with pytest.raises(SystemExit) as exc:
        cli._run_examples(["nonexistent"])
    assert exc.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_name_with_dest_points_to_init(capsys, tmp_path: pathlib.Path):
    with pytest.raises(SystemExit) as exc:
        cli._run_examples(["mlip.si", str(tmp_path)])
    assert exc.value.code == 2
    assert f"tensorial examples init mlip.si {tmp_path}" in capsys.readouterr().out
    assert not any(tmp_path.iterdir())


def test_default_choice(tmp_path: pathlib.Path):
    config = tmp_path / "train.yaml"
    config.write_text("defaults:\n  - _self_\n  - data: si\n  - model: nequip\n")
    assert cli._default_choice(config, "data") == "si"
    config.write_text("defaults:\n  - {data: qm9}\n")
    assert cli._default_choice(config, "data") == "qm9"
    assert cli._default_choice(config, "trainer") is None

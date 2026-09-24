import pathlib

from tensorial.reaxkit import cli


def _required_files_of(text: str, tmp_path: pathlib.Path):
    source = tmp_path / "example.yaml"
    source.write_text(text, encoding="utf-8")
    return cli._required_files(source)


def test_required_files_parse(tmp_path: pathlib.Path):
    files = _required_files_of(
        "# Top comment\n"
        "# requires: data/foo.xyz data/bar.json\n"
        "key: value\n"
        "# requires: qux.xyz\n",
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

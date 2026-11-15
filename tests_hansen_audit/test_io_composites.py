from hansen_audit.io import COMPOSITE, index_images


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")


def test_composites_are_skipped_by_default(tmp_path):
    _touch(tmp_path, "way_1_2020_2020-06-07.tiff", "way_2_2020.tiff", "notes.txt")
    out = index_images(tmp_path)
    assert set(out) == {"way_1"} and out["way_1"][0][0] == "2020-06-07"


def test_composites_on_request_are_marked(tmp_path):
    _touch(tmp_path, "way_1_2020_2020-06-07.tiff", "way_2_2020.tiff", "relation_3_2024.tif")
    out = index_images(tmp_path, composites=True)
    assert set(out) == {"way_1", "way_2", "relation_3"}
    assert out["way_2"] == [(COMPOSITE, tmp_path / "way_2_2020.tiff")]
    assert out["way_1"][0][0] == "2020-06-07"


def test_a_dated_scene_comes_before_a_composite_of_the_same_farm(tmp_path):
    _touch(tmp_path, "way_1_2020.tiff", "way_1_2020_2020-06-07.tiff")
    out = index_images(tmp_path, composites=True)
    assert [d for d, _ in out["way_1"]] == ["2020-06-07", COMPOSITE]

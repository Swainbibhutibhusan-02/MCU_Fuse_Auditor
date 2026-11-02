
import pandas as pd
import pytest

from src.fuse_decoder import decode_fuse_map, load_schema


def test_decode_basic_fields():
    # Fuse map: binary 1101
    schema = pd.DataFrame({
        "field": ["FIELD_A", "FIELD_B"],
        "lsb": [0, 2],
        "nbits": [2, 2],
    })

    result = decode_fuse_map("0xD", schema)

    assert result == {
        "FIELD_A": 1,  # bits 0-1 = 01
        "FIELD_B": 3,  # bits 2-3 = 11
    }


def test_decode_hex_with_whitespace_and_prefix():
    schema = pd.DataFrame({
        "field": ["FIELD_A"],
        "lsb": [0],
        "nbits": [4],
    })

    result = decode_fuse_map("  0xA  ", schema)

    assert result["FIELD_A"] == 10


def test_invalid_hexadecimal_raises_error():
    schema = pd.DataFrame({
        "field": ["FIELD_A"],
        "lsb": [0],
        "nbits": [4],
    })

    with pytest.raises(ValueError, match="Invalid hexadecimal"):
        decode_fuse_map("not_hex", schema)


def test_negative_fuse_map_raises_error():
    schema = pd.DataFrame({
        "field": ["FIELD_A"],
        "lsb": [0],
        "nbits": [4],
    })

    with pytest.raises(ValueError, match="cannot be negative"):
        decode_fuse_map("-1", schema)


def test_duplicate_schema_fields_raise_error(monkeypatch):
    schema = pd.DataFrame({
        "field": ["FIELD_A", "FIELD_A"],
        "lsb": [0, 4],
        "nbits": [2, 2],
    })

    monkeypatch.setattr(
        "src.fuse_decoder.pd.read_csv",
        lambda *args, **kwargs: schema,
    )

    with pytest.raises(ValueError, match="Duplicate field names"):
        load_schema()


def test_negative_lsb_raises_error(monkeypatch):
    schema = pd.DataFrame({
        "field": ["FIELD_A"],
        "lsb": [-1],
        "nbits": [2],
    })

    monkeypatch.setattr(
        "src.fuse_decoder.pd.read_csv",
        lambda *args, **kwargs: schema,
    )

    with pytest.raises(ValueError, match="LSB positions cannot be negative"):
        load_schema()


def test_zero_width_raises_error(monkeypatch):
    schema = pd.DataFrame({
        "field": ["FIELD_A"],
        "lsb": [0],
        "nbits": [0],
    })

    monkeypatch.setattr(
        "src.fuse_decoder.pd.read_csv",
        lambda *args, **kwargs: schema,
    )

    with pytest.raises(ValueError, match="Field widths must be greater than zero"):
        load_schema()


def test_overlapping_bits_raise_error(monkeypatch):
    schema = pd.DataFrame({
        "field": ["FIELD_A", "FIELD_B"],
        "lsb": [0, 1],
        "nbits": [2, 2],
    })

    monkeypatch.setattr(
        "src.fuse_decoder.pd.read_csv",
        lambda *args, **kwargs: schema,
    )

    with pytest.raises(ValueError, match="overlaps existing bits"):
        load_schema()

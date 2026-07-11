"""`MeshData.__post_init__` の shape/dtype 契約を直接検証するテスト。

`tests/test_io_mesh.py` は load_mesh/save_mesh 経由の間接検証のみで、
`MeshData` コンストラクタ自体の検証ロジック(異常系・境界値)を狙い撃ちする
テストが無かった(Codex 二次レビュー指摘)。ここでは正常系3件+異常系を
フィールドごとに直接カバーする。
"""

from __future__ import annotations

import numpy as np
import pytest

from atlasmith.types import MeshData

_N = 4
_M = 2


def _valid_kwargs() -> dict:
    """全フィールドが仕様どおりの kwargs(そのまま MeshData(**kwargs) できる)。"""
    return dict(
        vertices=np.zeros((_N, 3), dtype=np.float64),
        faces=np.zeros((_M, 3), dtype=np.int64),
        uv=np.zeros((_N, 2), dtype=np.float32),
        maps={
            "basecolor": np.zeros((8, 8, 3), dtype=np.float32),
            "gray": np.zeros((8, 8), dtype=np.float32),  # 2D も許容
        },
        source_vertex=np.arange(_N, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# 正常系
# ---------------------------------------------------------------------------


def test_valid_mesh_data_constructs_with_all_fields():
    mesh = MeshData(**_valid_kwargs())
    assert mesh.vertices.shape == (_N, 3)
    assert mesh.faces.shape == (_M, 3)
    assert mesh.uv is not None and mesh.uv.shape == (_N, 2)
    assert set(mesh.maps.keys()) == {"basecolor", "gray"}
    assert mesh.source_vertex is not None
    assert mesh.source_vertex.shape == (_N,)


def test_valid_mesh_data_constructs_with_uv_and_source_vertex_none():
    mesh = MeshData(
        vertices=np.zeros((_N, 3), dtype=np.float64),
        faces=np.zeros((_M, 3), dtype=np.int64),
        uv=None,
        maps={},
        source_vertex=None,
    )
    assert mesh.uv is None
    assert mesh.maps == {}
    assert mesh.source_vertex is None


def test_valid_mesh_data_allows_zero_faces():
    """faces は M>=0 を許容する(vertices の N>=1 制約とは非対称、docstring 明記済み)。

    行長制約のため要約を分割しているだけで、上記1行の説明が本テストの意図。
    """
    mesh = MeshData(
        vertices=np.zeros((_N, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int64),
        uv=None,
        maps={},
        source_vertex=None,
    )
    assert mesh.faces.shape == (0, 3)


# ---------------------------------------------------------------------------
# 異常系: vertices
# ---------------------------------------------------------------------------


def test_vertices_wrong_shape_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["vertices"] = np.zeros((_N, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="vertices must have shape"):
        MeshData(**kwargs)


@pytest.mark.parametrize("dtype", [np.float32, np.int64])
def test_vertices_wrong_dtype_raises_value_error(dtype):
    kwargs = _valid_kwargs()
    kwargs["vertices"] = np.zeros((_N, 3), dtype=dtype)
    with pytest.raises(ValueError, match="vertices must have dtype float64"):
        MeshData(**kwargs)


def test_vertices_empty_n_zero_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["vertices"] = np.zeros((0, 3), dtype=np.float64)
    kwargs["uv"] = None
    kwargs["source_vertex"] = None
    with pytest.raises(ValueError, match="N >= 1"):
        MeshData(**kwargs)


# ---------------------------------------------------------------------------
# 異常系: faces
# ---------------------------------------------------------------------------


def test_faces_wrong_shape_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["faces"] = np.zeros((_M, 4), dtype=np.int64)
    with pytest.raises(ValueError, match="faces must have shape"):
        MeshData(**kwargs)


@pytest.mark.parametrize("dtype", [np.float64, np.int32])
def test_faces_wrong_dtype_raises_value_error(dtype):
    kwargs = _valid_kwargs()
    kwargs["faces"] = np.zeros((_M, 3), dtype=dtype)
    with pytest.raises(ValueError, match="faces must have dtype int64"):
        MeshData(**kwargs)


# ---------------------------------------------------------------------------
# 異常系: uv
# ---------------------------------------------------------------------------


def test_uv_wrong_shape_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["uv"] = np.zeros((_N, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uv must have shape"):
        MeshData(**kwargs)


def test_uv_wrong_dtype_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["uv"] = np.zeros((_N, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="uv must have dtype float32"):
        MeshData(**kwargs)


def test_uv_n_mismatch_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["uv"] = np.zeros((_N + 1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="uv must have the same N"):
        MeshData(**kwargs)


# ---------------------------------------------------------------------------
# 異常系: maps
# ---------------------------------------------------------------------------


def test_maps_not_a_dict_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["maps"] = [np.zeros((8, 8, 3), dtype=np.float32)]
    with pytest.raises(ValueError, match="maps must be a dict"):
        MeshData(**kwargs)


@pytest.mark.parametrize("shape", [(8,), (8, 8, 3, 1)])
def test_maps_value_wrong_ndim_raises_value_error(shape):
    kwargs = _valid_kwargs()
    kwargs["maps"] = {"basecolor": np.zeros(shape, dtype=np.float32)}
    with pytest.raises(ValueError, match=r"maps\['basecolor'\] must be 2D"):
        MeshData(**kwargs)


def test_maps_value_wrong_dtype_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["maps"] = {"basecolor": np.zeros((8, 8, 3), dtype=np.uint8)}
    with pytest.raises(
        ValueError, match=r"maps\['basecolor'\] must have dtype float32"
    ):
        MeshData(**kwargs)


# ---------------------------------------------------------------------------
# 異常系: source_vertex
# ---------------------------------------------------------------------------


def test_source_vertex_wrong_shape_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["source_vertex"] = np.zeros((_N, 1), dtype=np.int64)
    with pytest.raises(ValueError, match="source_vertex must have shape"):
        MeshData(**kwargs)


def test_source_vertex_n_mismatch_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["source_vertex"] = np.arange(_N + 1, dtype=np.int64)
    with pytest.raises(ValueError, match="source_vertex must have the same N"):
        MeshData(**kwargs)


def test_source_vertex_wrong_dtype_raises_value_error():
    kwargs = _valid_kwargs()
    kwargs["source_vertex"] = np.arange(_N, dtype=np.int32)
    with pytest.raises(ValueError, match="source_vertex must have dtype int64"):
        MeshData(**kwargs)

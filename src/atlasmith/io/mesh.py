"""GLB/glTF/OBJ ⇄ MeshData の入出力。trimesh はこのモジュールに閉じ込める
(横断規約の依存方向: `io → types (+trimesh/PIL)`。逆方向の import は禁止)。

スコープ: 単一メッシュ・単一マテリアル・単一UVセット(判断6の正式制約)。
複数メッシュ GLB/glTF は明確なエラーメッセージで拒否する。

V方向規約: IR の maps は「row 0 = 画像上端 = V=0」(glTF 規約)。
WHY: trimesh 4.12.2 実測(2026-07-10)— 既知の非対称テクスチャ(行ごとに異なる色)
+ 既知 vt(四隅 0/1)を持つメッシュを GLB と OBJ の双方で書き出し→読み戻したところ、
vt 値・画像配列とも無変換のまま一致した(max|d|=0、GLB/OBJ 間でも画像・UV が
bit 一致)。つまり trimesh は OBJ の vt を GLB と同じ「row0=V=0」規約で扱っており、
io 層で追加の V 反転を行う必要はない。trimesh をバージョンアップした際は、同じ手順
(既知の非対称テクスチャ+既知 UV で GLB/OBJ を書き出し→読み戻し、画像配列と UV が
無変換で一致するか)で再実測し、この前提が崩れていないか確認すること。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from atlasmith.io.image import _array_to_pil, _pil_to_array
from atlasmith.types import MeshData

_SUPPORTED_EXTENSIONS = (".glb", ".gltf", ".obj")

_MAP_MATERIAL_ATTRS = {
    "normal": "normalTexture",
    "metallic_roughness": "metallicRoughnessTexture",
}


def _check_extension(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported mesh file extension: {ext!r} "
            f"(expected one of: {', '.join(_SUPPORTED_EXTENSIONS)})"
        )
    return ext


def _basecolor_image(visual: trimesh.visual.texture.TextureVisuals):
    """basecolor 画像を format 非依存に取り出す。

    WHY: `visual.image` は「コンストラクタに image= を直接渡した」場合しか
    埋まらないインスタンス属性で、`trimesh.load()` で読み戻した TextureVisuals
    では常に欠落する(probe_vdir.py/probe_chan.py で実測済み)。実データは
    material 側(GLB/glTF: PBRMaterial.baseColorTexture、OBJ: SimpleMaterial.image)
    にあるため、そちらを優先して見る。
    """
    material = visual.material
    image = getattr(material, "baseColorTexture", None)
    if image is not None:
        return image
    image = getattr(material, "image", None)
    if image is not None:
        return image
    return getattr(visual, "image", None)


def _single_geometry(
    loaded: trimesh.Trimesh | trimesh.Scene, path: Path
) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if len(geometries) != 1:
            raise ValueError(
                f"Expected a single mesh in {path}, found {len(geometries)} "
                "geometries (multi-mesh files are not supported)"
            )
        return geometries[0]
    return loaded


def load_mesh(path: str | Path) -> MeshData:
    """GLB/glTF/OBJ ファイルを読み、`MeshData` として返す。"""
    path = Path(path)
    _check_extension(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    # process=False: trimesh 側の頂点マージ/後処理を無効化する。
    # WHY: 面ごとに独立した UV を持つメッシュ(例: cube fixture の 24 頂点)は
    # 位置だけ見ると重複するため、process=True だと想定外に welding されうる。
    loaded = trimesh.load(path, process=False)
    geom = _single_geometry(loaded, path)

    vertices = np.asarray(geom.vertices, dtype=np.float64)
    faces = np.asarray(geom.faces, dtype=np.int64)

    uv: np.ndarray | None = None
    maps: dict[str, np.ndarray] = {}
    visual = geom.visual
    if (
        isinstance(visual, trimesh.visual.texture.TextureVisuals)
        and visual.uv is not None
    ):
        uv = np.asarray(visual.uv, dtype=np.float32)
        basecolor_image = _basecolor_image(visual)
        if basecolor_image is not None:
            maps["basecolor"] = _pil_to_array(basecolor_image)
        material = visual.material
        for map_name, attr in _MAP_MATERIAL_ATTRS.items():
            texture = getattr(material, attr, None)
            if texture is not None:
                maps[map_name] = _pil_to_array(texture)

    # io では weld 追跡をしない(恒等写像) — 横断規約が許容する2択の単純な方。
    source_vertex = np.arange(len(vertices), dtype=np.int64)

    return MeshData(
        vertices=vertices,
        faces=faces,
        uv=uv,
        maps=maps,
        source_vertex=source_vertex,
    )


def _build_visual(mesh: MeshData) -> trimesh.visual.texture.TextureVisuals | None:
    if mesh.uv is None:
        return None
    material = trimesh.visual.material.PBRMaterial()
    basecolor_image = None
    basecolor = mesh.maps.get("basecolor")
    if basecolor is not None:
        basecolor_image = _array_to_pil(basecolor)
        material.baseColorTexture = basecolor_image
    for map_name, attr in _MAP_MATERIAL_ATTRS.items():
        arr = mesh.maps.get(map_name)
        if arr is not None:
            setattr(material, attr, _array_to_pil(arr))
    return trimesh.visual.texture.TextureVisuals(
        uv=np.asarray(mesh.uv, dtype=np.float64),
        material=material,
        image=basecolor_image,
    )


def save_mesh(mesh: MeshData, path: str | Path) -> None:
    """`MeshData` を GLB/glTF/OBJ ファイルへ書き出す。"""
    path = Path(path)
    _check_extension(path)

    visual = _build_visual(mesh)
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tri_mesh.export(str(path))

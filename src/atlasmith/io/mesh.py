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

import hashlib
import json
import re
import urllib.parse
from pathlib import Path

import numpy as np
import trimesh
from trimesh.exchange.gltf import export_gltf
from trimesh.resolvers import FilePathResolver

from atlasmith.io.image import _array_to_pil, _pil_to_array
from atlasmith.types import MeshData

_SUPPORTED_EXTENSIONS = (".glb", ".gltf", ".obj")

_MAP_MATERIAL_ATTRS = {
    "normal": "normalTexture",
    "metallic_roughness": "metallicRoughnessTexture",
}


class _UnquotingFileResolver(FilePathResolver):
    """glTF の percent-encoded な `uri` を decode してからファイルを探す resolver。

    WHY(指摘3): save 側は glTF 2.0 仕様(URI 中の予約文字は percent-encode)に
    従い、サイドカー参照 `uri` を `urllib.parse.quote(name, safe="")` で符号化する。
    一方、ディスク上のサイドカーファイル名は仕様が指す「decode 後の名前」で書く。
    trimesh の `FilePathResolver` は `uri` を生のファイル名として扱い decode しない
    (resolvers.py `FilePathResolver.get` 実測)ため、この resolver を挟まないと
    `a%23b_...bin` という存在しない名前を探して読み込みが壊れる。ASCII のみの
    通常 stem では `quote`/`unquote` とも no-op なので既存挙動は不変。
    """

    def get(self, name: str):
        return super().get(urllib.parse.unquote(name))


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
    ext = _check_extension(path)
    if not path.exists():
        raise FileNotFoundError(f"Mesh file not found: {path}")

    # process=False: trimesh 側の頂点マージ/後処理を無効化する。
    # WHY: 面ごとに独立した UV を持つメッシュ(例: cube fixture の 24 頂点)は
    # 位置だけ見ると重複するため、process=True だと想定外に welding されうる。
    if ext == ".gltf":
        # サイドカー `uri` は save 側で percent-encode 済み。trimesh の既定
        # resolver は decode しないため、decode を挟む専用 resolver で読む
        # (`_UnquotingFileResolver` の WHY 参照)。
        loaded = trimesh.load(
            path, process=False, resolver=_UnquotingFileResolver(str(path))
        )
    else:
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


def _material_name_for_stem(stem: str) -> str:
    """OBJ の `newmtl`/画像ファイル名の基礎として安全な識別子を stem から作る。

    WHY: trimesh の `SimpleMaterial.to_obj()` は `material.name` をそのまま
    `newmtl <name>` トークンと画像ファイル名(`{name}.png`)の両方に使う。
    stem に空白等が含まれると OBJ の空白区切りトークンとして壊れるため、
    ファイル名としての「stem 由来」は保ちつつ英数字・`_`・`-` のみに正規化する。

    WHY(バグ修正 Step 0-4c): 単純な文字置換(非対応文字→`_`)だけでは非可逆
    (元の stem を復元できない)ため、異なる stem が同じ結果へ潰れて衝突しう
    る(実測: `"赤"`/`"青"` がともに `"_"` へ潰れ、同一ディレクトリ保存で
    サイドカーが衝突する)。置換結果が元の stem と一致しない場合(=非 ASCII
    文字や記号を含み情報が失われた場合)は、元 stem の UTF-8 バイト列の
    sha1 先頭8桁 hex を付与して一意化する。ASCII のみの stem(`left` 等)は
    そのまま返し、見た目を変えない。
    """
    sanitized = re.sub(r"[^0-9A-Za-z_-]", "_", stem)
    if sanitized == stem and sanitized.strip("_"):
        return sanitized
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    base = sanitized.strip("_")
    return f"{base}_{digest}" if base else f"_{digest}"


def _build_visual(
    mesh: MeshData, material_name: str
) -> trimesh.visual.texture.TextureVisuals | None:
    if mesh.uv is None:
        return None
    material = trimesh.visual.material.PBRMaterial(name=material_name)
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


def _export_gltf_with_stem_sidecars(tri_mesh: trimesh.Trimesh, path: Path) -> None:
    """glTF 分離形式を、サイドカー名を出力ファイルの stem から一意に導出して書き出す。

    WHY(バグ修正 Step 0-4b): `trimesh.exchange.gltf.export_gltf()` が返すサイド
    カー名は常に固定(`gltf_buffer_0.bin` 等、出力パスに非依存)。実測の結果、
    `export_gltf()` にはこれを制御する引数が無いため、返ってきた
    `{ファイル名: バイト列}` 辞書を post-process する: `model.gltf` 以外の
    キーへ stem プレフィックスを付けて改名し、JSON 内の `buffers`/`images` の
    `uri` 参照も同じ対応表で書き換えてから書き出す。これにより同一ディレクトリ
    へ異なるメッシュを複数 `.gltf` 保存しても、2件目が1件目のサイドカーを
    黙って上書きしなくなる(実測: 改名後のファイル名衝突なし、往復読み込みで
    元の色が保たれることを確認済み)。
    """
    files = export_gltf(tri_mesh)
    stem = path.stem
    # ディスク上のサイドカー名は生の(decode 後の)名前。JSON の `uri` はここから
    # percent-encode して書く(glTF 2.0 は URI 中の予約文字 `#`/`?`/`%`/空白等の
    # エンコードを要求する。例: stem に `#` を含むと fragment 扱いで参照が壊れる)。
    # 読み戻しは `_UnquotingFileResolver` が decode して生の名前へ突き合わせる。
    rename_map = {name: f"{stem}_{name}" for name in files if name != "model.gltf"}

    tree = json.loads(files["model.gltf"])
    for buffer in tree.get("buffers", []):
        uri = buffer.get("uri")
        if uri in rename_map:
            buffer["uri"] = urllib.parse.quote(rename_map[uri], safe="")
    for image in tree.get("images", []):
        uri = image.get("uri")
        if uri in rename_map:
            image["uri"] = urllib.parse.quote(rename_map[uri], safe="")

    output_files = {path.name: json.dumps(tree, separators=(",", ":")).encode("utf-8")}
    for name, data in files.items():
        if name != "model.gltf":
            output_files[rename_map[name]] = data

    for name, data in output_files.items():
        (path.parent / name).write_bytes(data)


def save_mesh(mesh: MeshData, path: str | Path) -> None:
    """`MeshData` を GLB/glTF/OBJ ファイルへ書き出す。"""
    path = Path(path)
    ext = _check_extension(path)

    material_name = _material_name_for_stem(path.stem)
    visual = _build_visual(mesh, material_name)
    tri_mesh = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=np.float64),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        visual=visual,
        process=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)

    if ext == ".gltf":
        # サイドカー名(gltf_buffer_N.bin 等)は stem 由来に改名する必要がある
        # ため、標準の export() を経由せず専用ヘルパで書き出す(上記 WHY 参照)。
        _export_gltf_with_stem_sidecars(tri_mesh, path)
    elif ext == ".obj":
        # mtl_name を明示し、mtl/画像ファイル名の衝突(同一ディレクトリへの
        # 複数保存で "material.mtl"/"material_0.png" に固定される問題)を防ぐ。
        tri_mesh.export(str(path), mtl_name=f"{material_name}.mtl")
    else:  # .glb — 単一ファイルでサイドカーが無いため衝突しない。
        tri_mesh.export(str(path))

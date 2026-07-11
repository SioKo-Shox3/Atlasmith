"""Step 0-4 io ラウンドトリップテスト(test-design-step0-4.md 第2部 ケース表準拠)。

ケースID(R1/V1/X1/M1/Q1/P1/N1/N2/N3/U1)は test-design-step0-4.md と対応させる。
C1/C2 は Step 0-4b/0-4c(verifier 発見バグの回帰テスト、test-design 外の追加ケース)。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from atlasmith.io.mesh import load_mesh, save_mesh
from atlasmith.types import MeshData

VERTEX_ATOL = 1e-5
UV_ATOL = 1e-6
PIXEL_ATOL = 1e-6

MESH_FIXTURES = ("cube_mesh", "sphere_mesh", "torus_mesh")
FORMATS = (".glb", ".gltf", ".obj")

# R1 の 9 combos のうち torus_mesh の3件だけ RGBA(4ch)にして、チャンネル数保存
# (RGB が往復で RGBA に化けないか)を回帰テストする(前任が実測しかけていた懸念)。
_CHANNELS_BY_MESH = {"cube_mesh": 3, "sphere_mesh": 3, "torus_mesh": 4}


def _bilinear_sample(img: np.ndarray, u: float, v: float) -> np.ndarray:
    """テスト側の独立バイリニアサンプラ(横断規約の UV↔テクセル変換式に準拠)。

    production 側に実装が無い Step 0-4 時点でも、V1/X1 の検証は「本実装のロー
    ダが返した配列」を独立ロジックでサンプルすることで担保する。
    """
    height, width = img.shape[:2]
    x = u * width - 0.5
    y = v * height - 0.5
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0
    x0c, x1c = int(np.clip(x0, 0, width - 1)), int(np.clip(x1, 0, width - 1))
    y0c, y1c = int(np.clip(y0, 0, height - 1)), int(np.clip(y1, 0, height - 1))
    top = img[y0c, x0c] * (1 - fx) + img[y0c, x1c] * fx
    bottom = img[y1c, x0c] * (1 - fx) + img[y1c, x1c] * fx
    return top * (1 - fy) + bottom * fy


# ---------------------------------------------------------------------------
# R1: 往復 [glb/gltf/obj] x [cube/sphere/torus]
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", FORMATS)
@pytest.mark.parametrize("mesh_name", MESH_FIXTURES)
def test_r1_round_trip_preserves_geometry_and_texture(
    request, mesh_name, ext, tmp_path, make_texture
):
    mesh = request.getfixturevalue(mesh_name)
    channels = _CHANNELS_BY_MESH[mesh_name]
    texture = make_texture(
        "gradient", size=(64, 64), channels=channels, seed=0, quantize8=True
    )
    mesh.maps = {"basecolor": texture}

    path = tmp_path / f"mesh{ext}"
    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert loaded.vertices.shape == mesh.vertices.shape
    assert loaded.faces.shape == mesh.faces.shape
    np.testing.assert_allclose(loaded.vertices, mesh.vertices, atol=VERTEX_ATOL)
    assert loaded.uv is not None
    np.testing.assert_allclose(loaded.uv, mesh.uv, atol=UV_ATOL)
    assert set(loaded.maps.keys()) == {"basecolor"}
    assert loaded.maps["basecolor"].shape == texture.shape
    max_diff = np.abs(loaded.maps["basecolor"] - texture).max()
    assert max_diff <= PIXEL_ATOL


# ---------------------------------------------------------------------------
# C1: 同一ディレクトリへの複数保存でサイドカーが衝突しないこと(Step 0-4b 回帰)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".obj", ".gltf"])
def test_c1_multiple_meshes_same_directory_do_not_clobber_sidecars(
    cube_mesh, tmp_path, ext
):
    """OBJ/glTF は mtl/画像やバッファをサイドカーファイルとして書き出す。

    サイドカー名が出力パスの stem から一意に導出されず固定名(例:
    `material.mtl`/`material_0.png`, `gltf_buffer_0.bin`)のままだと、同一
    ディレクトリへ異なるテクスチャの2メッシュを保存したとき2件目が1件目の
    サイドカーを黙って上書きし、1件目を読み戻すと2件目のテクスチャが返る
    (静かなデータ破損)。verifier が発見した実バグの回帰テスト。

    WHY(verifier 再検証で判明): `make_texture("gradient", seed=...)` は
    gradient が seed 非依存のため2枚が bit 同一になり、衝突しても検出できず
    テストが空虚化していた。ここでは異なるチャンネルに値を置いた明示的な
    配列を使い、冒頭で非同一性をガードして将来の差し替えでも空虚化を防ぐ。
    """
    texture_left = np.zeros((64, 64, 3), dtype=np.float32)
    texture_left[:, :, 0] = 0.9
    texture_left = (np.round(texture_left * 255.0) / 255.0).astype(np.float32)

    texture_right = np.zeros((64, 64, 3), dtype=np.float32)
    texture_right[:, :, 2] = 0.9
    texture_right = (np.round(texture_right * 255.0) / 255.0).astype(np.float32)

    assert not np.array_equal(texture_left, texture_right)

    # 幾何は使い回し、テクスチャだけ変えた独立メッシュを2つ用意する
    # (サイドカー名衝突はテクスチャ/マテリアルのサイドカーファイルで起きる)。
    mesh_left = MeshData(
        vertices=cube_mesh.vertices.copy(),
        faces=cube_mesh.faces.copy(),
        uv=cube_mesh.uv.copy(),
        maps={"basecolor": texture_left},
        source_vertex=cube_mesh.source_vertex.copy(),
    )
    mesh_right = MeshData(
        vertices=cube_mesh.vertices.copy(),
        faces=cube_mesh.faces.copy(),
        uv=cube_mesh.uv.copy(),
        maps={"basecolor": texture_right},
        source_vertex=cube_mesh.source_vertex.copy(),
    )

    path_left = tmp_path / f"left{ext}"
    path_right = tmp_path / f"right{ext}"
    save_mesh(mesh_left, path_left)
    save_mesh(mesh_right, path_right)

    loaded_left = load_mesh(path_left)
    loaded_right = load_mesh(path_right)

    max_diff_left = np.abs(loaded_left.maps["basecolor"] - texture_left).max()
    max_diff_right = np.abs(loaded_right.maps["basecolor"] - texture_right).max()
    assert max_diff_left <= PIXEL_ATOL
    assert max_diff_right <= PIXEL_ATOL


# ---------------------------------------------------------------------------
# C2: 非 ASCII(Unicode のみ)stem 同士のサイドカー衝突が残っていないこと
# ---------------------------------------------------------------------------


def test_c2_unicode_only_stems_do_not_clobber_sidecars_obj(cube_mesh, tmp_path):
    """`_material_name_for_stem` が非 ASCII stem を単純な文字置換だけで作ると、

    例えば `"赤"`/`"青"` がともに `"_"` へ潰れて `_.mtl`/`_.png` を共有し、
    同一ディレクトリ保存で1件目が破損する(verifier 実測)。ハッシュ付与で
    一意化した後もこの衝突が起きないことを確認する回帰テスト。glTF は
    サイドカー名の導出に stem を無加工のまま使うため対象外(コーディネーター
    裁定)。
    """
    texture_red = np.zeros((64, 64, 3), dtype=np.float32)
    texture_red[:, :, 0] = 0.9
    texture_red = (np.round(texture_red * 255.0) / 255.0).astype(np.float32)

    texture_blue = np.zeros((64, 64, 3), dtype=np.float32)
    texture_blue[:, :, 2] = 0.9
    texture_blue = (np.round(texture_blue * 255.0) / 255.0).astype(np.float32)

    assert not np.array_equal(texture_red, texture_blue)

    mesh_red = MeshData(
        vertices=cube_mesh.vertices.copy(),
        faces=cube_mesh.faces.copy(),
        uv=cube_mesh.uv.copy(),
        maps={"basecolor": texture_red},
        source_vertex=cube_mesh.source_vertex.copy(),
    )
    mesh_blue = MeshData(
        vertices=cube_mesh.vertices.copy(),
        faces=cube_mesh.faces.copy(),
        uv=cube_mesh.uv.copy(),
        maps={"basecolor": texture_blue},
        source_vertex=cube_mesh.source_vertex.copy(),
    )

    path_red = tmp_path / "赤.obj"
    path_blue = tmp_path / "青.obj"
    save_mesh(mesh_red, path_red)
    save_mesh(mesh_blue, path_blue)

    loaded_red = load_mesh(path_red)
    loaded_blue = load_mesh(path_blue)

    max_diff_red = np.abs(loaded_red.maps["basecolor"] - texture_red).max()
    max_diff_blue = np.abs(loaded_blue.maps["basecolor"] - texture_blue).max()
    assert max_diff_red <= PIXEL_ATOL
    assert max_diff_blue <= PIXEL_ATOL


# ---------------------------------------------------------------------------
# C3: glTF stem に URI 予約文字(`#`)を含んでも往復すること(指摘3 回帰)
# ---------------------------------------------------------------------------


def test_c3_gltf_stem_with_hash_round_trips(cube_mesh, tmp_path, make_texture):
    """glTF のサイドカー `uri` は glTF 2.0 仕様に従い percent-encode すべき。

    stem に `#` を含むと未エンコードでは URI fragment として解釈され参照が
    壊れる(spec 準拠ローダで破損)。save 側は `uri` を percent-encode し、
    ディスク上のサイドカー名は decode 後の生名で書く。load 側は decode して
    突き合わせる。ここでは `a#b.gltf` を保存→読み込みで basecolor が往復する
    ことに加え、(1) 生名サイドカーがディスクに存在し、(2) JSON の `uri` が
    percent-encode 済み(`%23` を含み生 `#` を含まない)ことを固定する。
    """
    texture = make_texture(
        "gradient", size=(64, 64), channels=3, seed=6, quantize8=True
    )
    cube_mesh.maps = {"basecolor": texture}

    path = tmp_path / "a#b.gltf"
    save_mesh(cube_mesh, path)
    loaded = load_mesh(path)

    assert loaded.uv is not None
    assert loaded.maps["basecolor"].shape == texture.shape
    max_diff = np.abs(loaded.maps["basecolor"] - texture).max()
    assert max_diff <= PIXEL_ATOL

    # (1) ディスク上のサイドカー名は decode 後の生名(`#` を含む)。
    sidecars = [p.name for p in tmp_path.iterdir() if p.name != path.name]
    assert sidecars, "expected at least one glTF sidecar file"
    assert all(name.startswith("a#b_") for name in sidecars), sidecars

    # (2) JSON の `uri` は percent-encode 済み(生 `#` を含まない)。
    tree = json.loads(path.read_text(encoding="utf-8"))
    uris = [b.get("uri") for b in tree.get("buffers", [])]
    uris += [i.get("uri") for i in tree.get("images", [])]
    encoded = [u for u in uris if u]
    assert encoded, "expected sidecar uri references in the glTF JSON"
    assert all("#" not in u for u in encoded), encoded
    assert any("%23" in u for u in encoded), encoded


# ---------------------------------------------------------------------------
# V1: GLB ローダの V 方向(独立ライター → 本実装ローダ)
# ---------------------------------------------------------------------------


def test_v1_glb_loader_v_direction(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
    texture = np.zeros((4, 4, 3), dtype=np.uint8)
    texture[0, :, :] = [255, 0, 0]  # row0 = 画像上端 = 赤
    texture[1, :, :] = [0, 255, 0]
    texture[2, :, :] = [0, 128, 255]
    texture[3, :, :] = [0, 0, 255]  # 最終行 = 青
    image = Image.fromarray(texture, mode="RGB")

    # trimesh を直接叩いて GLB を書く(本実装の save_mesh は使わない — ローダ単体
    # を独立ライターで固定検証するため、V1 は往復では見えない対称バグを検出する)。
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=image)
    visual = trimesh.visual.texture.TextureVisuals(
        uv=uv, material=material, image=image
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    path = tmp_path / "v1.glb"
    mesh.export(str(path))

    loaded = load_mesh(path)
    basecolor = loaded.maps["basecolor"]
    np.testing.assert_allclose(basecolor[0, 0], [1.0, 0.0, 0.0], atol=1e-6)

    # 頂点0・1は uv の v=0 側 → コーナーサンプルは赤であるべき。
    for vertex_index in (0, 1):
        u, v = loaded.uv[vertex_index]
        color = _bilinear_sample(basecolor, float(u), float(v))
        np.testing.assert_allclose(color, [1.0, 0.0, 0.0], atol=1e-2)


# ---------------------------------------------------------------------------
# X1: OBJ の V/面取り(クロス形式で GLB 経路と一致)
# ---------------------------------------------------------------------------


def test_x1_obj_face_id_matches_glb_path(cube_mesh, tmp_path, make_face_id_texture):
    texture = make_face_id_texture(cube_mesh, size=(64, 64))
    cube_mesh.maps = {"basecolor": texture}

    glb_path = tmp_path / "x1.glb"
    obj_path = tmp_path / "x1.obj"
    save_mesh(cube_mesh, glb_path)
    save_mesh(cube_mesh, obj_path)

    glb_loaded = load_mesh(glb_path)
    obj_loaded = load_mesh(obj_path)

    n_faces = len(cube_mesh.faces)
    for face_index, face in enumerate(cube_mesh.faces):
        centroid_uv = cube_mesh.uv[face].mean(axis=0)
        expected_ch0 = (face_index + 0.5) / n_faces
        expected = np.array([expected_ch0, 1.0 - expected_ch0, 0.5], dtype=np.float32)

        glb_color = _bilinear_sample(
            glb_loaded.maps["basecolor"], float(centroid_uv[0]), float(centroid_uv[1])
        )
        obj_color = _bilinear_sample(
            obj_loaded.maps["basecolor"], float(centroid_uv[0]), float(centroid_uv[1])
        )

        np.testing.assert_allclose(glb_color, expected, atol=0.05)
        np.testing.assert_allclose(obj_color, expected, atol=0.05)
        np.testing.assert_allclose(obj_color, glb_color, atol=1e-2)


# ---------------------------------------------------------------------------
# M1: 複数マップ GLB(basecolor + normal + metallic_roughness)
# ---------------------------------------------------------------------------


def test_m1_multiple_maps_round_trip_glb(cube_mesh, tmp_path, make_texture):
    basecolor = make_texture(
        "gradient", size=(64, 64), channels=3, seed=1, quantize8=True
    )
    normal = make_texture(
        "multisine", size=(64, 64), channels=3, seed=2, quantize8=True
    )
    metallic_roughness = make_texture(
        "aperiodic", size=(64, 64), channels=3, seed=3, quantize8=True
    )
    cube_mesh.maps = {
        "basecolor": basecolor,
        "normal": normal,
        "metallic_roughness": metallic_roughness,
    }

    path = tmp_path / "m1.glb"
    save_mesh(cube_mesh, path)
    loaded = load_mesh(path)

    assert set(loaded.maps.keys()) == {"basecolor", "normal", "metallic_roughness"}
    for name, expected in cube_mesh.maps.items():
        np.testing.assert_allclose(loaded.maps[name], expected, atol=PIXEL_ATOL)


# ---------------------------------------------------------------------------
# Q1: 非正方解像度([glb])
# ---------------------------------------------------------------------------


def test_q1_non_square_resolution_round_trip(cube_mesh, tmp_path, make_texture):
    texture = make_texture(
        "gradient", size=(32, 64), channels=3, seed=4, quantize8=True
    )
    cube_mesh.maps = {"basecolor": texture}

    path = tmp_path / "q1.glb"
    save_mesh(cube_mesh, path)
    loaded = load_mesh(path)

    assert loaded.maps["basecolor"].shape == (32, 64, 3)
    np.testing.assert_allclose(loaded.maps["basecolor"], texture, atol=PIXEL_ATOL)


# ---------------------------------------------------------------------------
# P1: Unicode+空白パス([glb, obj])
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".glb", ".obj"])
def test_p1_unicode_and_space_path_round_trip(cube_mesh, tmp_path, make_texture, ext):
    texture = make_texture(
        "gradient", size=(64, 64), channels=3, seed=5, quantize8=True
    )
    cube_mesh.maps = {"basecolor": texture}

    unicode_dir = tmp_path / "日本語 スペース"
    unicode_dir.mkdir()
    path = unicode_dir / f"mesh{ext}"

    save_mesh(cube_mesh, path)
    loaded = load_mesh(path)

    assert loaded.vertices.shape == cube_mesh.vertices.shape
    assert loaded.faces.shape == cube_mesh.faces.shape
    assert loaded.maps["basecolor"].shape == texture.shape


# ---------------------------------------------------------------------------
# N1: テクスチャ無し OBJ(負の対照 — maps のデフォルト捏造禁止)
# ---------------------------------------------------------------------------


def test_n1_obj_without_texture_yields_empty_maps(tmp_path):
    obj_text = "v 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 1.0 1.0 0.0\nf 1 2 3\n"
    path = tmp_path / "n1.obj"
    path.write_text(obj_text, encoding="utf-8", newline="\n")

    loaded = load_mesh(path)

    assert loaded.uv is None
    assert loaded.maps == {}


# ---------------------------------------------------------------------------
# N2: 複数メッシュ GLB(明確な英語 ValueError で拒否)
# ---------------------------------------------------------------------------


def test_n2_multi_mesh_glb_raises_value_error(cube_mesh, tmp_path):
    mesh_a = trimesh.Trimesh(
        vertices=cube_mesh.vertices, faces=cube_mesh.faces, process=False
    )
    mesh_b = mesh_a.copy()
    mesh_b.apply_translation([5.0, 0.0, 0.0])
    scene = trimesh.Scene()
    scene.add_geometry(mesh_a, node_name="a")
    scene.add_geometry(mesh_b, node_name="b")
    path = tmp_path / "n2.glb"
    scene.export(str(path))

    with pytest.raises(ValueError, match="single mesh"):
        load_mesh(path)


# ---------------------------------------------------------------------------
# N3: 不在パス/未対応拡張子
# ---------------------------------------------------------------------------


def test_n3_missing_file_raises_file_not_found(tmp_path):
    path = tmp_path / "missing.glb"
    with pytest.raises(FileNotFoundError, match=r"missing\.glb"):
        load_mesh(path)


def test_n3_unsupported_extension_raises_value_error(tmp_path):
    path = tmp_path / "mesh.stl"
    path.write_text("not a real mesh", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match=r"\.stl"):
        load_mesh(path)


# ---------------------------------------------------------------------------
# U1: uv=None の保存(None uv での save クラッシュ・空 UV の捏造を禁止)
# ---------------------------------------------------------------------------


def test_u1_uv_none_round_trip(tmp_path):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        uv=None,
        maps={},
        source_vertex=np.arange(4, dtype=np.int64),
    )
    path = tmp_path / "u1.glb"

    save_mesh(mesh, path)
    loaded = load_mesh(path)

    assert loaded.uv is None
    assert loaded.maps == {}
    np.testing.assert_allclose(loaded.vertices, vertices, atol=VERTEX_ATOL)
    assert loaded.faces.shape == faces.shape

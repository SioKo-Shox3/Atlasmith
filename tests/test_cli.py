"""Step 1-3 CLI 結線テスト(計画 Step 1-3 検証項目準拠)。

`main()` 直接呼び+console script の subprocess 実行+空白/Unicode パス+
normal map 警告を検証する。CLI は `atlasmith.rebake()` の薄いラッパなので、
再展開・焼き直し自体の数値的正しさは test_pack_naive.py/test_bake_oracle.py が
既に担保している — ここでは「結線が動くこと」と「入出力の疎通」に絞る。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from atlasmith.cli import main
from atlasmith.io import load_mesh, save_mesh

# CI/開発機での実行時間を抑えるため、既定の 1024 より小さいテクスチャで検証する
# (main() の結線・疎通が目的であり、既定値そのものの検証は計画の対象外)。
_TEXTURE_SIZE = 64
_PADDING = 4


def _find_atlasmith_executable() -> str:
    """venv の Scripts/bin ディレクトリから実 console script `atlasmith` を解決する。

    WHY(一次レビュー指摘): `sys.executable -c "from atlasmith.cli import main"` は
    import 経由の起動であり、`pyproject.toml` の
    `[project.scripts] atlasmith = "atlasmith.cli:main"` という entry point
    マッピング自体が壊れていても(誤記・欠落等)検出できない。`sys.executable` と
    同じ venv の Scripts/bin ディレクトリを第一候補にすることで、pytest を起動した
    venv が実際にインストールした実行ファイルを確実に掴む(Windows は
    `atlasmith.exe`、それ以外は `atlasmith`)。レイアウトが想定と異なる環境向けの
    フォールバックとして `shutil.which` も試す。どちらでも見つからない場合は
    テストを fail させる(venv 実行前提のため見つからないこと自体が異常 — skip で
    握り潰さない)。
    """
    bin_dir = Path(sys.executable).parent
    for candidate in (bin_dir / "atlasmith.exe", bin_dir / "atlasmith"):
        if candidate.exists():
            return str(candidate)
    found = shutil.which("atlasmith")
    if found is not None:
        return found
    raise AssertionError(
        f"`atlasmith` console script not found next to {sys.executable} nor on "
        "PATH — expected `uv sync` to have installed the [project.scripts] entry "
        "point into the active venv"
    )


def _build_input_mesh(cube_mesh, make_texture, *, with_normal: bool = False):
    basecolor = make_texture(
        "gradient", size=(32, 32), channels=3, seed=0, quantize8=True
    )
    cube_mesh.maps = {"basecolor": basecolor}
    if with_normal:
        cube_mesh.maps["normal"] = make_texture(
            "multisine", size=(32, 32), channels=3, seed=1, quantize8=True
        )
    return cube_mesh


# ---------------------------------------------------------------------------
# main() 直接呼び: 出力に UV/テクスチャが存在すること
# ---------------------------------------------------------------------------


def test_main_direct_call_rebakes_and_writes_output(tmp_path, cube_mesh, make_texture):
    mesh = _build_input_mesh(cube_mesh, make_texture)
    input_path = tmp_path / "in.glb"
    output_path = tmp_path / "out.glb"
    save_mesh(mesh, input_path)

    exit_code = main(
        [
            str(input_path),
            "-o",
            str(output_path),
            "--padding",
            str(_PADDING),
            "--texture-size",
            str(_TEXTURE_SIZE),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()
    result = load_mesh(output_path)
    assert result.uv is not None
    assert result.maps  # non-empty: basecolor が焼き直されて残っていること
    assert "basecolor" in result.maps
    assert result.maps["basecolor"].shape[:2] == (_TEXTURE_SIZE, _TEXTURE_SIZE)


# ---------------------------------------------------------------------------
# console script(相当)の subprocess 実行
# ---------------------------------------------------------------------------


def test_console_entry_point_runs_as_subprocess(tmp_path, cube_mesh, make_texture):
    """実 console script `atlasmith` 実行ファイルを subprocess で起動し検証する。

    `pyproject.toml` の `[project.scripts] atlasmith = "atlasmith.cli:main"` が
    生成した実行ファイルそのものを叩く(`_find_atlasmith_executable` の WHY 参照)
    — entry point マッピングが壊れると本テストが落ちる。
    """
    atlasmith_exe = _find_atlasmith_executable()

    mesh = _build_input_mesh(cube_mesh, make_texture)
    input_path = tmp_path / "in.glb"
    output_path = tmp_path / "out.glb"
    save_mesh(mesh, input_path)

    result = subprocess.run(
        [
            atlasmith_exe,
            str(input_path),
            "-o",
            str(output_path),
            "--texture-size",
            str(_TEXTURE_SIZE),
            "--padding",
            str(_PADDING),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 空白/Unicode を含むパス
# ---------------------------------------------------------------------------


def test_main_succeeds_with_whitespace_and_unicode_paths(
    tmp_path, cube_mesh, make_texture
):
    mesh = _build_input_mesh(cube_mesh, make_texture)
    work_dir = tmp_path / "with space ディレクトリ"
    work_dir.mkdir()
    input_path = work_dir / "入力 mesh.glb"
    output_path = work_dir / "出力 mesh.glb"
    save_mesh(mesh, input_path)

    exit_code = main(
        [
            str(input_path),
            "-o",
            str(output_path),
            "--texture-size",
            str(_TEXTURE_SIZE),
            "--padding",
            str(_PADDING),
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# normal map 警告
# ---------------------------------------------------------------------------


def test_main_warns_on_normal_map(tmp_path, cube_mesh, make_texture):
    mesh = _build_input_mesh(cube_mesh, make_texture, with_normal=True)
    input_path = tmp_path / "in.glb"
    output_path = tmp_path / "out.glb"
    save_mesh(mesh, input_path)

    with pytest.warns(UserWarning, match="normal map"):
        exit_code = main(
            [
                str(input_path),
                "-o",
                str(output_path),
                "--texture-size",
                str(_TEXTURE_SIZE),
                "--padding",
                str(_PADDING),
            ]
        )

    assert exit_code == 0
    assert output_path.exists()


def test_main_does_not_warn_without_normal_map(tmp_path, cube_mesh, make_texture):
    """負の対照: normal map が無ければ警告は出ない(誤検出の回帰防止)。"""
    mesh = _build_input_mesh(cube_mesh, make_texture, with_normal=False)
    input_path = tmp_path / "in.glb"
    output_path = tmp_path / "out.glb"
    save_mesh(mesh, input_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exit_code = main(
            [
                str(input_path),
                "-o",
                str(output_path),
                "--texture-size",
                str(_TEXTURE_SIZE),
                "--padding",
                str(_PADDING),
            ]
        )

    assert exit_code == 0

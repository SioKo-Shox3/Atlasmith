"""Step 1-3 CLI 結線テスト(計画 Step 1-3 検証項目準拠)。

`main()` 直接呼び+console script の subprocess 実行+空白/Unicode パス+
normal map 警告を検証する。CLI は `atlasmith.rebake()` の薄いラッパなので、
再展開・焼き直し自体の数値的正しさは test_pack_naive.py/test_bake_oracle.py が
既に担保している — ここでは「結線が動くこと」と「入出力の疎通」に絞る。

**Step 2-7 で既存 5 件に `--segmenter geometric` を明示した**(2026-08-06
ユーザー裁定 — §0-A 条件3「無変更で green」の文言には触れるが、条件3 が保護して
いた対象=Phase 1 の CLI 挙動〈実行成功・出力生成・normal map 警告・Unicode パス・
console entry point〉は全て維持される)。**WHY 明示するか**: Phase 2 から CLI の
`--segmenter` 既定は `sam2`(裁定 E)なので、フラグ無しだと `[ml]` を入れた開発機
では本ファイルの 5 件が実際に SAM2 を回す。**実測 2026-08-06: 665.44s / 5 件**
(subprocess 1 件だけで 297s)で、しかも SAM2 が依存側から出す
`DeprecationWarning`(`sam2/utils/transforms.py` の `torch.jit.script`)が
`test_main_does_not_warn_without_normal_map` の `simplefilter("error")` に当たって
落ちる。ここは「Phase 1 の結線が壊れていないこと」を測る場所なので、決定的で
依存ゼロの `geometric` に固定する。**既定経路(= SAM2)が実際に通ることは
`test_default_cli_path_uses_sam2_backend`(`@pytest.mark.gl @pytest.mark.ml`)が
担当する** — 既定を変えても誰も気づかない状態にはしない。
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
# Phase 1 挙動を測る 5 件が踏む分割バックエンド(上の module docstring の WHY 参照)。
_GEOMETRIC = ["--segmenter", "geometric"]

# --- subprocess のハング上限(秒)。`tests/test_rebake_part.py` も import する ---
#
# **WHY 必要か**: `timeout=` の無い `subprocess.run` は、子がハングすると失敗では
# なく**無限待ち**になる。全量実行が止まったのか遅いのかを外から区別できず、
# 実測 2026-08-06 では SAM2 経路で 11 分待たされた(結局これは正常な低速実行だった
# が、それを判断する材料がゼロだった)。上限を付ければハングは
# `subprocess.TimeoutExpired` で必ず落ちる。
#
# **WHY この値か(実測から決める — 発明しない)**: これは*ハング検出*の上限で
# あって性能回帰の検出器ではないので、正当な最悪実行時間の上に十分な余裕を積む。
#
#   - 幾何 / naive 経路の console script: **実測 0.92-1.23s**
#     (2026-08-06 `pytest -m "not ml and not gl" --durations=0`、4 件)。
#     → 300s は実測の約 240 倍。負荷・ウイルス対策・コールドな FS キャッシュで
#       桁が変わっても誤検出しない。
#   - SAM2 経路の console script: **実測 31.3s**(本ファイルの
#     `test_default_cli_path_uses_sam2_backend`)/ **31.9s**
#     (`test_rebake_part.py::test_explicit_sam2_cli_subprocess_succeeds`)。いずれも
#     重みキャッシュ済み・`--seg-views 6`(2026-08-06 `pytest -m ml --durations=0`)。
#     観測済みの最悪値は本ファイル冒頭に記録した **1 subprocess で 297s**(既定
#     24 視点)であり、初回実行にはさらに ~900MB の重み DL が上乗せされる。
#     → 1200s は最悪実測 297s の約 4 倍で、初回 DL 込みでも足りる。
_CLI_TIMEOUT_SEC = 300
_SAM2_CLI_TIMEOUT_SEC = 1200


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
    """`--segmenter geometric` 明示の理由は module docstring(既定は sam2)。"""
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
            *_GEOMETRIC,
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
    `--segmenter geometric` 明示の理由は module docstring(既定は sam2)。
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
            *_GEOMETRIC,
        ],
        capture_output=True,
        text=True,
        timeout=_CLI_TIMEOUT_SEC,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert output_path.exists()


# ---------------------------------------------------------------------------
# 空白/Unicode を含むパス
# ---------------------------------------------------------------------------


def test_main_succeeds_with_whitespace_and_unicode_paths(
    tmp_path, cube_mesh, make_texture
):
    """`--segmenter geometric` 明示の理由は module docstring(既定は sam2)。"""
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
            *_GEOMETRIC,
        ]
    )

    assert exit_code == 0
    assert output_path.exists()


# ---------------------------------------------------------------------------
# normal map 警告
# ---------------------------------------------------------------------------


def test_main_warns_on_normal_map(tmp_path, cube_mesh, make_texture):
    """`--segmenter geometric` 明示の理由は module docstring(既定は sam2)。"""
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
                *_GEOMETRIC,
            ]
        )

    assert exit_code == 0
    assert output_path.exists()


def test_main_does_not_warn_without_normal_map(tmp_path, cube_mesh, make_texture):
    """負の対照: normal map が無ければ警告は出ない(誤検出の回帰防止)。

    `--segmenter geometric` 明示の理由は module docstring(既定は sam2)。
    `simplefilter("error")` を使うので、この経路が出す**あらゆる**警告が例外化する
    — Phase 2 で `rebake` が持ったガター消失警告(`g == 0`)にも当たらないことを
    同時に測っている(`_TEXTURE_SIZE=64` / `_PADDING=4` は実測で `g=2`)。
    """
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
                *_GEOMETRIC,
            ]
        )

    assert exit_code == 0


# ---------------------------------------------------------------------------
# 既定 CLI 経路(= SAM2)が実際に通ること
#
# 上の 5 件が `--segmenter geometric` に固定された結果、**フラグ無しの既定経路を
# 踏むテストが 1 つも無くなる**。それでは「既定を変えても誰も気づかない」ので、
# ここで既定経路そのものを測る(2026-08-06 ユーザー裁定の条件2)。
# ---------------------------------------------------------------------------


@pytest.mark.gl
@pytest.mark.ml
def test_default_cli_path_uses_sam2_backend(tmp_path, cube_mesh, make_texture):
    """フラグ無しの CLI が SAM2 バックエンドを構築して完走する。

    **WHY subprocess か**: 既定経路は torch/sam2/moderngl をセッションへ常駐させる。
    同一プロセスで回すと以降のテストの `sys.modules` を汚すので、新規インタプリタで
    起動して戻り値だけを見る(前例: `_find_atlasmith_executable` の console script
    実行、および計画 §0-A 条件1 の import 隔離ゲート)。

    **WHY `simplefilter("error")` を使わないか**: SAM2 は依存側から警告を出す
    (`sam2/utils/transforms.py` の `torch.jit.script` が `DeprecationWarning`)。
    それは Atlasmith の挙動ではないので、ここで例外化しても自分のバグは捕まらない。
    測るのは「既定経路が SAM2 を構築して exit 0 で出力を書くこと」だけ。

    `--seg-views 6` で視点数を落として実行時間を抑える(品質は測らない —
    ML の数値ゲートは `tests/test_multiview_sam2.py` の領分)。
    """
    mesh = _build_input_mesh(cube_mesh, make_texture)
    input_path = tmp_path / "in.glb"
    output_path = tmp_path / "out.glb"
    save_mesh(mesh, input_path)

    # `--segmenter` は**渡さない** — 既定が sam2 であることがこのテストの主題。
    result = subprocess.run(
        [
            _find_atlasmith_executable(),
            str(input_path),
            "-o",
            str(output_path),
            "--texture-size",
            str(_TEXTURE_SIZE),
            "--padding",
            str(_PADDING),
            "--seg-views",
            "6",
        ],
        capture_output=True,
        text=True,
        timeout=_SAM2_CLI_TIMEOUT_SEC,
    )

    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert output_path.exists()
    # 既定が geometric へ落ちていない(= SAM2 が実際に構築された)ことの証拠。
    # `--seg-views 6` は検証済み既定(24)を下回るので、SAM2 経路なら必ずこの
    # 警告が出る(`multiview/__init__.py` の `_warn_if_below_validated_defaults`)。
    # geometric へ落ちていれば `--seg-views` は無視されて警告も出ない。
    assert "below its validated defaults" in result.stderr, (
        f"the default CLI path did not build the SAM2 backend; stderr={result.stderr!r}"
    )

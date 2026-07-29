"""import 隔離ゲート(計画v4 §2.1 の機械的ゲート (a)(b)・Step 2-1)。

Atlasmith の決定的層(`atlasmith` 本体と `atlasmith.segmentation` 配下)は
**torch / moderngl / sam2 を import しない**。重い ML/GL 依存を触ってよいのは
`segmentation/multiview/render.py` と `segmentation/multiview/sam2_masks.py` の
2 ファイルの、しかも **関数内 import だけ**である(§2.1 の隔離モジュール規約)。
この規約が壊れると `[ml]` 未導入環境と CI(extras 無し・GPU 無しランナー)が
import の時点で落ちるので、人手のレビューではなく機械で守る。

検査は 2 種類あり、どちらか片方では穴が開くので両方置く:

  (a) `sys.modules` 検査 — 実際に import してみて禁止パッケージが載らないこと。
      「今の実装が実際にどう振る舞うか」を見る。
  (b) AST 検査 — ソース上に禁止 import が書かれていないこと。実行されない経路
      (`if TYPE_CHECKING:` ブロックや、まだ誰も呼んでいない関数の中)に書かれた
      違反は (a) では踏まれず素通りするため、静的にも見る。

**この時点(Step 2-1)では違反は 1 件も無く、隔離 2 ファイルもまだ存在しない。**
ゲートが自明に通ることは承知のうえで先に置く — Step 2-3 以降で追加される
ファイルに自動的に効かせるためであり、検査対象はディレクトリ走査で集めるので
新しいファイルが検査から漏れることはない。
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src" / "atlasmith"


# ---------------------------------------------------------------------------
# (a) sys.modules 検査(新規インタプリタ 3 本)
# ---------------------------------------------------------------------------

# 実行時に「載っていてはいけない」重量パッケージ。trimesh/xatlas/PIL は必須依存で
# `atlasmith.io` / `atlasmith.pack` が eager import するため、ここには含めない
# (それらは (b) の静的検査で `segmentation` 配下に漏れていないことを担保する)。
_RUNTIME_FORBIDDEN = ("torch", "moderngl", "sam2")

_PROBE_PREFIX = "ATLASMITH_PROBE "

# 新規インタプリタで対象 module を import し、`sys.modules` の状態を 1 行の JSON で
# 報告するプローブ。対象 module 自体が存在しない場合だけ status="missing" を返し、
# それ以外の失敗(依存の欠落・import 時例外)は例外のまま落として非 0 終了させる
# — 「不在」と「壊れている」を取り違えないため。
_PROBE_SOURCE = """\
import importlib
import json
import sys

module_name, forbidden_csv = sys.argv[1], sys.argv[2]
forbidden = forbidden_csv.split(",")
try:
    importlib.import_module(module_name)
except ModuleNotFoundError as exc:
    if exc.name != module_name:
        raise
    print("ATLASMITH_PROBE " + json.dumps({"status": "missing"}))
    sys.exit(0)
loaded = sorted(
    name
    for name in list(sys.modules)
    if any(name == pkg or name.startswith(pkg + ".") for pkg in forbidden)
)
print("ATLASMITH_PROBE " + json.dumps({"status": "ok", "loaded": loaded}))
"""


def _run_import_probe(module_name: str) -> dict:
    """`module_name` を **新規インタプリタ**で import し、プローブの JSON を返す。

    WHY subprocess か(計画 §0-A 条件1・blocker): `tests/conftest.py` の能力検出
    (`pytest_collection_modifyitems`)は、`gl`/`ml` マーカーの item が収集された
    実行で moderngl / torch / sam2 を **実際に import する**。同一プロセスで
    `sys.modules` を見ると、Step 2-3 で `gl` マークのテストが入った時点から
    この 3 本が必ず赤くなり、Step 2-1 の合否条件と Phase 2 の Done 条件が同時に
    満たせなくなる。前例: `tests/test_cli.py:28-53`(console script の subprocess 実行)。

    `cwd` を repo root に固定する: `python -c` は cwd を `sys.path[0]` に載せるので、
    たとえば `src/` から pytest を起動された場合に `src/atlasmith` がインストール済み
    パッケージを影で置き換えてしまう。repo root には `atlasmith/` が無いので安全。
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE_SOURCE,
            module_name,
            ",".join(_RUNTIME_FORBIDDEN),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(_REPO_ROOT),
        check=False,
    )
    assert result.returncode == 0, (
        f"import probe for {module_name!r} exited with {result.returncode}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    reports = [
        line[len(_PROBE_PREFIX) :]
        for line in result.stdout.splitlines()
        if line.startswith(_PROBE_PREFIX)
    ]
    assert len(reports) == 1, (
        f"import probe for {module_name!r} did not emit exactly one report line\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return json.loads(reports[0])


def _assert_import_stays_light(
    module_name: str, *, allow_missing: bool = False
) -> None:
    """`module_name` の import 後に `_RUNTIME_FORBIDDEN` が 0 件であることを検査する。

    `allow_missing=True` の module が存在しないときは skip する — ただし
    **握り潰さない**: skip 理由に「まだ作られていない」と「作られた瞬間に効き始める」
    ことを書き、実行ログから不在が分かるようにする。
    """
    report = _run_import_probe(module_name)
    if report["status"] == "missing":
        if not allow_missing:
            pytest.fail(
                f"{module_name} does not exist — this module is required by the "
                "isolation gate and its absence is a real failure, not a skip"
            )
        pytest.skip(
            f"{module_name} does not exist yet (it is created in Step 2-3); this "
            "sys.modules isolation gate starts enforcing automatically as soon as "
            "the module lands"
        )
    assert report["loaded"] == [], (
        f"importing {module_name} pulled heavy packages into sys.modules: "
        f"{report['loaded']}. Only {_SRC_ROOT.name}/segmentation/multiview/render.py "
        "and sam2_masks.py may touch torch/moderngl/sam2, and only inside functions "
        "(plan v4 §2.1)."
    )


def test_import_atlasmith_does_not_load_heavy_packages() -> None:
    """`import atlasmith` で torch/moderngl/sam2 が載らない。

    `atlasmith/__init__.py` は io/pack を eager import するので trimesh/xatlas/PIL は
    載る(実測・想定内)。ここが検査するのは ML/GL 側だけ。
    """
    _assert_import_stays_light("atlasmith")


def test_import_segmentation_does_not_load_heavy_packages() -> None:
    """`import atlasmith.segmentation` で torch/moderngl/sam2 が載らない。"""
    _assert_import_stays_light("atlasmith.segmentation")


def test_import_multiview_does_not_load_heavy_packages() -> None:
    """`import atlasmith.segmentation.multiview` で torch/moderngl/sam2 が載らない。

    多視点パッケージは `render.py` / `sam2_masks.py` を **持っている**が、それらを
    module 直下で import してはならない(§2.1 規約2)。ここが本ゲートの主戦場。
    Step 2-3 でパッケージが作られるまでは skip する。
    """
    _assert_import_stays_light("atlasmith.segmentation.multiview", allow_missing=True)


# ---------------------------------------------------------------------------
# (b) AST 検査(位置別判定 — 計画 §0-A 条件5)
# ---------------------------------------------------------------------------

# 静的に禁止する第三者パッケージ。実行時 3 種に加えて、`segmentation` 配下から
# 触ってはならない trimesh/xatlas/PIL も含む(§2.1 規約5)。
_STATIC_FORBIDDEN = frozenset({"torch", "moderngl", "sam2", "trimesh", "xatlas", "PIL"})

# 隔離モジュール = 関数内 import に限り禁止パッケージを許される 2 ファイル。
# (Step 2-3 / 2-5 で作られる。存在しない間は分類に使われないだけで害は無い。)
_ISOLATED_MODULES = frozenset(
    {
        _SRC_ROOT / "segmentation" / "multiview" / "render.py",
        _SRC_ROOT / "segmentation" / "multiview" / "sam2_masks.py",
    }
)

# 動的 import の引数が文字列リテラルでなく、静的に解決できないことを表す番兵。
_UNRESOLVABLE = "<not a string literal>"

# 走査が空振り(glob の書き間違い等)していないことを担保する最小集合。
_SCAN_ANCHORS = (
    "src/atlasmith/__init__.py",
    "src/atlasmith/cli.py",
    "src/atlasmith/types.py",
    "src/atlasmith/segmentation/__init__.py",
)


def _scanned_files() -> list[Path]:
    """検査対象を **ディレクトリ走査で** 集める(将来増えるファイルも自動で入る)。

    対象 = `src/atlasmith/segmentation/**/*.py` +
    `src/atlasmith/{__init__,cli,types}.py`。
    `src/atlasmith/io/`(trimesh/PIL 可)と `src/atlasmith/pack/`(xatlas 可)は既存
    規約どおり対象外 — 走査に含めないことで除外する。
    """
    files = sorted((_SRC_ROOT / "segmentation").rglob("*.py"))
    files += [_SRC_ROOT / name for name in ("__init__.py", "cli.py", "types.py")]
    return files


def _link_parents(tree: ast.AST) -> None:
    """各ノードに親リンクを張る(`ast.walk` は祖先を教えてくれないため)。"""
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child._atlasmith_parent = parent  # type: ignore[attr-defined]


def _inside_function(node: ast.AST) -> bool:
    """祖先に `FunctionDef` / `AsyncFunctionDef` を持つか(= module 直下ではないか)。

    `if TYPE_CHECKING:` ブロックや `try/except` ブロックは関数ではないので、その中の
    import は **module 直下**と判定される(§2.1 規約3 — 「実行時に走らないから安全」を
    抜け道にしない)。
    """
    current = getattr(node, "_atlasmith_parent", None)
    while current is not None:
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return True
        current = getattr(current, "_atlasmith_parent", None)
    return False


def _dynamic_import_target(node: ast.Call) -> str | None:
    """`importlib.import_module(...)` / `__import__(...)` の対象ルートパッケージ名。

    import 呼び出しでなければ `None`、対象が文字列リテラルでなければ `_UNRESOLVABLE`。
    """
    func = node.func
    is_import_call = (
        isinstance(func, ast.Name) and func.id in ("__import__", "import_module")
    ) or (isinstance(func, ast.Attribute) and func.attr == "import_module")
    if not is_import_call:
        return None
    if not node.args:
        return _UNRESOLVABLE
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value.split(".")[0]
    return _UNRESOLVABLE


def _import_violations(path: Path, source: str, *, isolated: bool) -> list[str]:
    """1 ファイル分の隔離規約違反を、人が読めるメッセージのリストで返す。

    **位置別判定**(計画 §0-A 条件5。計画本文の「module 直下のみ禁止」を上書き):

    - **非隔離ファイル**: 深さを問わず違反。関数内 `import torch` も
      `__import__("moderngl")` も通さない。v4 本文のままだと `fusion.py` の関数内
      `import torch` が素通りし、規約(重い import は隔離 2 ファイルのみ)と
      検査が食い違う。
    - **隔離ファイル**(`render.py` / `sam2_masks.py`): module 直下のみ違反。
      関数内 import は規約 2 が明示的に認める唯一の逃げ道。

    `source` を `path` と別に受けるのは、この検査自体が「違反を仕込んだら本当に
    落ちるか」を実ファイル無しで自己検証できるようにするため
    (`test_ast_gate_catches_planted_violations`)。

    静的に解決できない動的 import(`import_module(name)` のように引数が変数)は
    **違反として報告する**: 中身が禁止パッケージかどうかを検査できない以上、
    素通しにすると `__import__` 禁止の趣旨(§2.1 規約4)がそのまま抜け道になる。
    現時点で該当箇所は 0 件。
    """
    tree = ast.parse(source, filename=str(path))
    _link_parents(tree)
    violations: list[str] = []

    def _record(node: ast.AST, detail: str) -> None:
        in_function = _inside_function(node)
        if isolated and in_function:
            return  # 隔離ファイルの関数内 import だけが許される。
        where = "inside a function" if in_function else "at module level"
        lineno = getattr(node, "lineno", 0)
        violations.append(f"{path.as_posix()}:{lineno}: {detail} ({where})")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _STATIC_FORBIDDEN:
                    _record(node, f"imports forbidden package {root!r}")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # 相対 import は atlasmith 内部にしか届かない。
            root = (node.module or "").split(".")[0]
            if root in _STATIC_FORBIDDEN:
                _record(node, f"from-imports forbidden package {root!r}")
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is None:
                continue
            if target == _UNRESOLVABLE:
                _record(
                    node,
                    "calls a dynamic import whose target is not a string literal, "
                    "so the isolation gate cannot verify it",
                )
            elif target in _STATIC_FORBIDDEN:
                _record(node, f"dynamically imports forbidden package {target!r}")

    return violations


def test_no_forbidden_imports_in_deterministic_layer() -> None:
    """決定的層のソースに、禁止パッケージの import が 1 件も書かれていない。

    どう壊れたら落ちるか: `segmentation/` 配下(隔離 2 ファイル以外)のどこかに
    `import torch` 等が **深さを問わず** 書かれた時点、または隔離 2 ファイルの
    module 直下に書かれた時点で、該当行が列挙されて落ちる。
    """
    files = _scanned_files()
    scanned = {path.relative_to(_REPO_ROOT).as_posix() for path in files}
    for anchor in _SCAN_ANCHORS:
        assert anchor in scanned, (
            f"{anchor} is not in the isolation scan set {sorted(scanned)} — the scan "
            "would pass vacuously"
        )
    missing = [path.as_posix() for path in files if not path.is_file()]
    assert missing == [], f"isolation scan points at non-existent files: {missing}"

    violations: list[str] = []
    for path in files:
        violations += _import_violations(
            path,
            path.read_text(encoding="utf-8"),
            isolated=path in _ISOLATED_MODULES,
        )
    assert violations == [], "import isolation violations:\n" + "\n".join(violations)


# 仕込み違反の一覧。`(id, source, isolated, expected_violation_count)`。
# 「現状のコードでは 0 件だから通っている」だけの空虚なゲートでないことを示す
# (= 検査器そのものが壊れたら落ちる)。
_PLANTED_CASES = (
    ("module-level import in a normal module", "import torch\n", False, 1),
    (
        "function-level import in a normal module (plan §0-A cond.5)",
        "def f():\n    import torch\n\n    return torch\n",
        False,
        1,
    ),
    (
        "function-level __import__ in a normal module (plan §0-A cond.5)",
        'def f():\n    return __import__("moderngl")\n',
        False,
        1,
    ),
    (
        "TYPE_CHECKING import is module level (§2.1 rule 3)",
        "from typing import TYPE_CHECKING\n\nif TYPE_CHECKING:\n    import torch\n",
        False,
        1,
    ),
    (
        "from-import of a forbidden package",
        "from PIL import Image\n",
        False,
        1,
    ),
    (
        "nested function does not hide the import",
        "def outer():\n    def inner():\n        import sam2\n\n        return sam2\n"
        "\n    return inner\n",
        False,
        1,
    ),
    (
        "dynamic import with a non-literal target is unverifiable",
        "import importlib\n\n\ndef f(name):\n"
        "    return importlib.import_module(name)\n",
        False,
        1,
    ),
    (
        "module-level import is forbidden even in an isolated module (§2.1 rule 2)",
        "import moderngl\n",
        True,
        1,
    ),
    (
        "function-level import is allowed in an isolated module",
        "def _import_moderngl():\n    import moderngl\n\n    return moderngl\n",
        True,
        0,
    ),
    (
        "function-level import_module is allowed in an isolated module",
        "import importlib\n\n\ndef _import_sam2():\n"
        '    return importlib.import_module("sam2")\n',
        True,
        0,
    ),
    (
        "allowed imports produce no false positives",
        "from __future__ import annotations\n\nimport numpy as np\n\n"
        "from atlasmith.types import MeshData\n",
        False,
        0,
    ),
)


def test_ast_gate_catches_planted_violations() -> None:
    """AST 検査器が「仕込んだ違反」を実際に検出し、正当な書き方を誤検出しない。

    どう壊れたら落ちるか: `_import_violations` が位置別判定を失って
    (例: 非隔離ファイルでも関数内なら許す実装に戻る)違反を見逃した瞬間、
    または numpy/相対 import を誤検出し始めた瞬間に落ちる。
    """
    fake_path = _SRC_ROOT / "segmentation" / "_planted_for_selftest.py"
    mismatches: list[str] = []
    for case_id, source, isolated, expected in _PLANTED_CASES:
        found = _import_violations(fake_path, source, isolated=isolated)
        if len(found) != expected:
            mismatches.append(
                f"{case_id}: expected {expected} violation(s), "
                f"got {len(found)}: {found}"
            )
    assert mismatches == [], "AST gate self-check failed:\n" + "\n".join(mismatches)

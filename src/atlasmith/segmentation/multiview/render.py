"""moderngl オフスクリーンレンダラ(**隔離モジュール1** — 計画v4 §2.1 / §2.4.2)。

1 回の描画で MRT(Multiple Render Targets)へ「色」と「面ID」を同時出力し、
`RenderedView(face_id, color, coverage)` を返す。

**隔離モジュール規約(計画v4 §2.1 規約1〜4 / §0-A 条件5)**:

  - moderngl の import は `_import_moderngl()` の**関数内だけ**。module 直下は
    禁止で、`if TYPE_CHECKING:` ブロックも「module 直下」として禁止される。
    型注釈は `from __future__ import annotations` の文字列化で書く。
  - `ImportError` は行動可能なメッセージへ変換して `from e` で再送出する。
    **WHY 関数内 import か**: 依存が導入済みの環境でも「未導入時のエラー経路」を
    monkeypatch でテストできる(module 直下の try/except は導入済み環境で
    不到達コードになり、ゲートが張れない)。
  - この規約は `tests/test_import_isolation.py` の AST 検査が機械的に守る。

**第三者ライセンス表記**(承認事項 G): moderngl は MIT。
一覧は `Docs/licenses/THIRD-PARTY-ML.md`。

**GL の状態と probe 実測(2026-07-29、当開発機 RTX 4080 / GL 3.3.0 / moderngl 5.12.0)**:

  - `GL_MAX_COLOR_ATTACHMENTS = 8`(>= 2 なので MRT 1 パスで足りる。2 パス方式は
    実装しない — 計画v4 §5 Step 2-3 probe (d) の分岐は不要と確定した)。
  - `fbo.read(...)` の row 0 は**画面下端**(左下原点)。読み戻し後に必ず行反転する。
  - `ctx.disable_direct(GL_DITHER)` が使える(`ctx.error == GL_NO_ERROR`)ので、
    **ディザは明示的に無効化する**(裁定4)。
"""

from __future__ import annotations

import logging
import warnings
from enum import Enum
from types import TracebackType
from typing import Any

import numpy as np

from atlasmith.segmentation.adjacency import face_normals
from atlasmith.segmentation.multiview import RenderedView
from atlasmith.segmentation.multiview.cameras import Camera
from atlasmith.segmentation.multiview.faceid import (
    decode_face_id,
    encode_face_codes,
    validate_coverage_consistency,
    validate_face_count,
)
from atlasmith.types import MeshData

LOG = logging.getLogger(__name__)

__all__ = ["MIN_IMAGE_SIZE", "SHADING_MODES"]

# `shading` の受理値(計画v4 §2.4.3)。値はシェーダの `u_shading` へ渡す整数で、
# GLSL 側の分岐と 1 対 1 に対応する。
_SHADING_TEXTURE = 0
_SHADING_NORMAL = 1
_SHADING_TEXTURE_NORMAL = 2
SHADING_MODES: dict[str, int] = {
    "texture": _SHADING_TEXTURE,
    "normal": _SHADING_NORMAL,
    "texture_normal": _SHADING_TEXTURE_NORMAL,
}

# レンダ画像の最小辺。これ未満は「独立オラクルと比較する内部画素が存在しない」
# ほど小さく、実用上も意味が無い(計画v4 §5 Step 2-3 ゲート10)。
MIN_IMAGE_SIZE = 8

# 対象が画面をこの割合未満しか覆っていなければ警告する(2周目レビュー B4)。
# **WHY 1%**: 外接球フィットが正常に働いていれば、球はおおむね画面の 50〜65% を
# 占める(実測: cube 0.44 / cylinder 0.31 / sphere 0.63 @ 384px)。1% はそこから
# 1.5 桁以上離れており、「たまたま小さい」と「フィットが破綻している」を余裕を
# もって分ける。実測の破綻例(1 x 1e-2 x 1e-2 の針)は 0.95% でここに掛かる。
_LOW_COVERAGE_WARN_RATIO = 0.01

# AABB 中心の絶対値が寸法のこの倍数を超えたら float32 精度の劣化を警告する
# (2周目レビュー N1)。**WHY 1e4**: 頂点は GPU へ f4 で送るので、座標の丸めは
# `|centre| * 1.2e-7`(float32 eps)。`|centre|/R = 1e4` でこれは `R * 1.2e-3`
# ≒ 1024px 画像の約 1 画素に達する。実測では `|centre|/R = 2e4` で全画素一致率が
# 0.9992(内部画素はまだ厳密)、`2e6` で内部画素が 6226 件不一致になる。
_FAR_FROM_ORIGIN_WARN_RATIO = 1e4

# 生の GL 列挙値。moderngl は `GL_DITHER` を定数として公開していないが、
# `Context.disable_direct` が生の列挙値を受け取る(probe (f) で確認)。
# **WHY 無効化するか**: 固定小数点カラーバッファを厳密な面IDに使っているので、
# ディザが 1 bit でも混ざれば面IDが別の面のものに化ける。既定で有効な機能を
# 「たぶん影響しない」で放置せず、明示的に切る(計画v4 §2.4.2 / 裁定4)。
_GL_DITHER = 0x0BD0

# 頂点属性のレイアウト(属性名, moderngl のフォーマット, float32 の個数)。
# probe (c) 実測: production 形状の FS では 4 属性すべてが生き残る。
_VERTEX_LAYOUT: tuple[tuple[str, str, int], ...] = (
    ("in_position", "3f", 3),
    ("in_uv", "2f", 2),
    ("in_code", "3f", 3),
    ("in_normal", "3f", 3),
)

# `in_code` / `in_normal` の `flat` は必須(計画v4 §2.4.2)。面IDは補間されては
# ならず、法線もオブジェクト空間の面法線(視点非依存)でなければ視点間融合が壊れる。
_VERTEX_SHADER = """#version 330 core
in vec3 in_position;
in vec2 in_uv;
in vec3 in_code;
in vec3 in_normal;
uniform mat4 u_mvp;
out vec2 v_uv;
flat out vec3 v_code;
flat out vec3 v_normal;
void main() {
    v_uv = in_uv;
    v_code = in_code;
    v_normal = in_normal;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_FRAGMENT_SHADER = """#version 330 core
in vec2 v_uv;
flat in vec3 v_code;
flat in vec3 v_normal;
uniform sampler2D u_basecolor;
uniform int u_shading;
uniform int u_has_texture;
layout(location = 0) out vec4 f_color;
layout(location = 1) out vec4 f_code;
vec3 shaded_rgb() {
    vec3 n_enc = clamp(v_normal * 0.5 + 0.5, 0.0, 1.0);
    if (u_has_texture == 0 || u_shading == 1) {
        return n_enc;
    }
    vec3 albedo = texture(u_basecolor, v_uv).rgb;
    if (u_shading == 0) {
        return albedo;
    }
    return 0.5 * albedo + 0.5 * n_enc;
}
void main() {
    f_color = vec4(shaded_rgb(), 1.0);
    f_code = vec4(v_code, 1.0);
}
"""


class _RendererState(Enum):
    """レンダラの寿命状態(裁定2 — 3 状態を明示的に管理する)。

    計画v4 §2.1 は「`__exit__` 後」と「`__enter__` 再入」しか決めていないが、
    `with` を書き忘れた経路(未入場のまま `render_view`)は現実に踏まれる
    (§0-A 条件10)。「まだ入っていない」と「もう出た」は原因も直し方も違うので、
    状態として分けてメッセージも分ける。
    """

    NOT_ENTERED = "not-entered"
    ACTIVE = "active"
    CLOSED = "closed"


def _import_moderngl() -> Any:
    """moderngl を**関数内で**import する(計画v4 §2.1 規約2)。

    Returns:
        moderngl モジュール。**戻り値が `Any` の WHY**: 型注釈に moderngl の型を
        書くには module 直下 import が要り、隔離規約に反する。GL オブジェクトを
        受け渡す内部境界にだけ `Any` を使う(公開契約には現れない)。

    Raises:
        ImportError: moderngl が未導入のとき。導入手順と代替経路を提示する。
    """
    try:
        import moderngl
    except ImportError as e:
        raise ImportError(
            "moderngl is required by the multi-view (SAM2) segmenter but is not "
            "installed. Install the optional extra with `uv sync --extra ml` "
            '(or `pip install "atlasmith[ml]"`), or use `--segmenter geometric`, '
            "which needs no GPU at all."
        ) from e
    return moderngl


def _build_code_attribute(face_codes: np.ndarray) -> np.ndarray:
    """面コード `(M,)` を頂点属性 `(3M, 3) float32` へ展開する。

    面ごとに頂点をアンロールしてあるので、1 面の 3 頂点には**同じ**コード色が
    入る(`np.repeat(..., 3, axis=0)`)。

    **この関数がテストの注入点でもある**(計画 §0-A 条件6): 通常経路では 3 頂点の
    コードが同値なので、`flat` 修飾子を外しても補間結果が同値になり
    「`flat` の不在」ゲートが空虚になる。テストはこの関数を monkeypatch して
    1 つの三角形の 3 頂点へ**異なる**コードを積み、覆う全画素のデコード値が
    ちょうど 1 種類であることを見る。
    """
    return np.repeat(encode_face_codes(face_codes), 3, axis=0)


class _ModernglRenderer:
    """メッシュ 1 個に束縛された moderngl オフスクリーンレンダラ(internal)。

    `MeshRenderer` Protocol(`atlasmith.segmentation.multiview`)の実装。
    **context manager として使うこと** — GL コンテキストと GPU リソースは
    `__enter__` で確保し、`__exit__` で生成の逆順に解放する。

        with _ModernglRenderer(mesh, image_size=1024, shading="texture_normal") as r:
            view = r.render_view(camera)

    描画方式(計画v4 §2.4.2): 面ごとに頂点をアンロールした 3M 頂点を 1 回描画し、
    MRT で `color0 = 色` / `color1 = 面ID` を同時に書く。マルチサンプルは無し
    (samples=0)。深度テスト有効・背面カリング無効・ブレンド無効・
    `FRAMEBUFFER_SRGB` は有効化しない・ディザは明示的に無効化。

    決定性: 同じメッシュ・同じカメラなら同じ画素を返す(RNG も時間依存も無い)。

    引数は非破壊: 渡された `MeshData` とその配列を書き換えない。
    """

    __slots__ = (
        "_basecolor",
        "_can_disable_dither",
        "_code_texture",
        "_color_texture",
        "_ctx",
        "_depth",
        "_fbo",
        "_has_texture",
        "_interleaved",
        "_program",
        "_state",
        "_texture",
        "_vao",
        "_vbo",
        "face_codes",
        "image_size",
        "shading",
    )

    def __init__(
        self,
        mesh: MeshData,
        *,
        image_size: int,
        shading: str,
        face_codes: np.ndarray | None = None,
    ) -> None:
        """CPU 側の頂点バッファを組み立て、引数を検証する(GL はまだ触らない)。

        **GL を `__enter__` まで遅らせる WHY**: 入口検証(`image_size` / `shading` /
        面数上限)は GL コンテキストが無い環境 — つまり CI — でも実行できるべき
        だから。構築時に落とせるものは構築時に落とす(fail-fast)。

        Args:
            mesh: 描画対象。`vertices` / `faces` / `uv` / `maps["basecolor"]` を
                読む。書き換えない。
            image_size: 正方形出力の一辺(画素)。`MIN_IMAGE_SIZE` 以上。
            shading: `"texture"` / `"normal"` / `"texture_normal"`
                (計画v4 §2.4.3)。テクスチャが無いメッシュでは `"normal"` へ
                自動的に落ち、`logging.info` で記録する(黙って変えない)。
            face_codes: **internal・テスト専用**。既定は
                `np.arange(M, dtype=np.int64)`。**WHY 引数にするか**: GPU 上の
                sentinel 面IDゲート(計画v4 §5 Step 2-3 ゲート6 / BL-10)が、
                バイト境界(255/256・65534/65535)を跨ぐ面IDを実際に GPU へ
                書かせて厳密一致を確かめるため。production の呼び出し側は
                指定しない。

        Raises:
            ValueError: `image_size` が小さすぎる・`shading` が未知・面数が
                24bit 上限超過・面が 0 枚・`face_codes` の shape/値域が不正なとき。
        """
        if isinstance(image_size, bool) or not isinstance(
            image_size, (int, np.integer)
        ):
            raise ValueError(
                f"image_size must be an int, got {type(image_size).__name__}"
            )
        if int(image_size) < MIN_IMAGE_SIZE:
            raise ValueError(
                f"image_size must be >= {MIN_IMAGE_SIZE}, got {int(image_size)}"
            )
        if shading not in SHADING_MODES:
            raise ValueError(
                f"unknown shading {shading!r}, expected one of {sorted(SHADING_MODES)}"
            )

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        n_faces = int(faces.shape[0])
        validate_face_count(n_faces)
        if n_faces == 0:
            # 面 0 枚は `MeshData` としては合法だが、空の頂点バッファは moderngl の
            # 内部エラーになる。原因の分かるメッセージで先に止める(黙って
            # 全面背景の画像を返すと、上流のバグが最後まで顕在化しない)。
            raise ValueError(
                "cannot render a mesh with 0 faces; the multi-view segmenter needs "
                "at least one face to produce a face-id buffer"
            )

        codes = (
            np.arange(n_faces, dtype=np.int64)
            if face_codes is None
            else np.asarray(face_codes)
        )
        if codes.shape != (n_faces,):
            raise ValueError(
                f"face_codes must have shape ({n_faces},), got {codes.shape}"
            )

        self.image_size: int = int(image_size)
        self.face_codes: np.ndarray = codes

        basecolor = mesh.maps.get("basecolor")
        self._has_texture = mesh.uv is not None and basecolor is not None
        if not self._has_texture and shading != "normal":
            LOG.info(
                "mesh has no basecolor texture (uv=%s, basecolor=%s); shading "
                "%r falls back to 'normal' (object-space face normals)",
                "present" if mesh.uv is not None else "missing",
                "present" if basecolor is not None else "missing",
                shading,
            )
            shading = "normal"
        self.shading: str = shading
        self._basecolor = basecolor if self._has_texture else None

        _warn_if_far_from_origin(vertices)
        self._interleaved = _build_vertex_buffer(
            vertices, faces, mesh.uv if self._has_texture else None, codes
        )
        self._state = _RendererState.NOT_ENTERED
        self._can_disable_dither = False
        self._ctx: Any = None
        self._program: Any = None
        self._vbo: Any = None
        self._vao: Any = None
        self._color_texture: Any = None
        self._code_texture: Any = None
        self._depth: Any = None
        self._texture: Any = None
        self._fbo: Any = None

    def __enter__(self) -> _ModernglRenderer:
        """GL コンテキストと GPU リソースを確保する。

        Returns:
            自分自身。

        Raises:
            RuntimeError: すでに入場中/退場済みのインスタンスを再入したとき、
                GL コンテキストを作れないとき、MRT に必要な
                `GL_MAX_COLOR_ATTACHMENTS >= 2` を満たさないとき。
            ImportError: moderngl 未導入のとき。
        """
        if self._state is _RendererState.ACTIVE:
            raise RuntimeError(
                "_ModernglRenderer is already entered; a renderer owns a GL context "
                "and must not be re-entered (create a second renderer instead)"
            )
        if self._state is _RendererState.CLOSED:
            raise RuntimeError(
                "_ModernglRenderer has already been exited and cannot be re-entered; "
                "its GL context was released (create a new renderer instead)"
            )

        moderngl = _import_moderngl()
        try:
            self._ctx = moderngl.create_context(standalone=True)
        except Exception as e:
            raise RuntimeError(
                "failed to create a standalone OpenGL context. A working GPU/GL "
                "driver is required by the multi-view segmenter; use "
                "`--segmenter geometric` if none is available."
            ) from e
        try:
            self._build_gl_resources(moderngl)
        except BaseException:
            # 例外経路でも解放を必ず通す(計画v4 §2.1)。ここで漏らすと、以降の
            # `create_context` がドライバ資源を食い潰したまま失敗しうる。
            self._release_gl()
            self._state = _RendererState.CLOSED
            raise
        self._state = _RendererState.ACTIVE
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """GPU リソースを生成の逆順に解放する(例外は握り潰さない)。"""
        self._release_gl()
        self._state = _RendererState.CLOSED

    def render_view(self, camera: Camera) -> RenderedView:
        """`camera` から 1 視点をレンダして `RenderedView` を返す。

        Args:
            camera: `cameras.build_camera` / `build_cameras` が作ったカメラ。
                視錐台の検証はカメラ生成時に済んでいる前提。

        Returns:
            `RenderedView`(`face_id (H,W) int32` / `color (H,W,3) uint8` /
            `coverage (H,W) bool`、いずれも **row 0 = 画面上端**)。

        Raises:
            RuntimeError: 入場中でないとき(未入場・退場後)。
            ValueError: `coverage <=> (face_id >= 0)` が破れたとき
                (ディザ/ブレンド事故の検出網 — 計画v4 §2.4.2)。

        Warns:
            UserWarning: 対象が画面のごく一部しか占めていないとき、および
                1 画素も描かれなかったとき(2周目レビュー B4)。
        """
        if self._state is _RendererState.NOT_ENTERED:
            raise RuntimeError(
                "_ModernglRenderer.render_view() was called before __enter__(); "
                "wrap the renderer in a `with` block (its GL resources are only "
                "alive between __enter__ and __exit__)"
            )
        if self._state is _RendererState.CLOSED:
            raise RuntimeError(
                "_ModernglRenderer.render_view() was called after __exit__(); "
                "its GL resources have been released"
            )

        moderngl = _import_moderngl()
        self._apply_gl_state(moderngl)
        if self._texture is not None:
            self._texture.use(location=0)
        # GL は列優先なので転置して渡す(`Camera.mvp` は行優先の数学記法)。
        mvp = np.ascontiguousarray(camera.mvp.T, dtype="f4")
        self._program["u_mvp"].write(mvp.tobytes())
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self._vao.render(moderngl.TRIANGLES)

        color_buf = self._read_rgba(0)
        code_buf = self._read_rgba(1)
        face_id, coverage = decode_face_id(code_buf)
        validate_coverage_consistency(face_id, coverage, context=f"view {camera.index}")
        _warn_if_screen_coverage_is_tiny(coverage, view_index=camera.index)
        return RenderedView(
            face_id=face_id,
            color=np.ascontiguousarray(color_buf[..., :3]),
            coverage=coverage,
        )

    def _build_gl_resources(self, moderngl: Any) -> None:
        """GPU 側リソースを**解放と逆順に**確保する(program → ... → fbo)。"""
        max_attachments = int(self._ctx.info.get("GL_MAX_COLOR_ATTACHMENTS", 0))
        if max_attachments < 2:
            raise RuntimeError(
                f"GL_MAX_COLOR_ATTACHMENTS is {max_attachments} (< 2), so the "
                "single-pass MRT face-id renderer cannot run on this driver; use "
                "`--segmenter geometric`."
            )
        # `image_size` の上限はドライバ依存なので、入場時に実測値と突き合わせる
        # (2周目レビュー N5)。ここで弾かないと `ctx.texture` / `ctx.framebuffer` が
        # 生の `moderngl.Error: the framebuffer is not complete` を出し、原因
        # (要求サイズが GL の上限を超えた)が呼び出し側に伝わらない。
        max_texture = int(self._ctx.info.get("GL_MAX_TEXTURE_SIZE", 0))
        max_renderbuffer = int(self._ctx.info.get("GL_MAX_RENDERBUFFER_SIZE", 0))
        max_size = min(v for v in (max_texture, max_renderbuffer) if v > 0)
        if self.image_size > max_size:
            raise ValueError(
                f"image_size={self.image_size} exceeds what this GL driver can "
                f"allocate (GL_MAX_TEXTURE_SIZE={max_texture}, "
                f"GL_MAX_RENDERBUFFER_SIZE={max_renderbuffer}); use "
                f"image_size <= {max_size}."
            )
        self._program = self._ctx.program(
            vertex_shader=_VERTEX_SHADER, fragment_shader=_FRAGMENT_SHADER
        )
        self._vbo = self._ctx.buffer(self._interleaved.tobytes())
        self._vao = self._ctx.vertex_array(
            self._program, _vao_content(self._program, self._vbo)
        )
        self._texture = self._upload_basecolor(moderngl)
        size = (self.image_size, self.image_size)
        # samples 既定 0(マルチサンプル無し)。解決パスが挟まると面IDが混ざる。
        self._color_texture = self._ctx.texture(size, 4, dtype="f1")
        self._code_texture = self._ctx.texture(size, 4, dtype="f1")
        self._depth = self._ctx.depth_renderbuffer(size)
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._color_texture, self._code_texture],
            depth_attachment=self._depth,
        )
        # uniform は素直に代入する(存在しなければ `KeyError` で落ちる)。
        # **WHY 防御しないか**: production のフラグメントシェーダは 3 つとも常に
        # 参照しており、リンカが落とせない(probe (c) 実測で全部残ることを確認)。
        # 「消えていたら黙って既定値で描く」方が、面IDと色が静かに壊れて危険。
        self._program["u_shading"].value = SHADING_MODES[self.shading]
        self._program["u_has_texture"].value = 1 if self._has_texture else 0
        self._program["u_basecolor"].value = 0
        # 裁定4(probe (f)): `disable_direct` が使えるなら GL_DITHER を明示的に
        # 無効化する。判定は入場時に 1 回だけ行い、毎フレームの警告連打を避ける。
        self._can_disable_dither = hasattr(self._ctx, "disable_direct")
        if not self._can_disable_dither:
            LOG.warning(
                "this moderngl build has no Context.disable_direct, so GL_DITHER "
                "cannot be turned off explicitly; face-id exactness is then only "
                "*not refuted* by the GPU sentinel gate, not proven (8-bit -> "
                "8-bit writes are not expected to be dithered)"
            )

    def _upload_basecolor(self, moderngl: Any) -> Any:
        """basecolor を RGB8 テクスチャとしてアップロードする(無ければ None)。

        **V 方向**: `maps` の規約は「row 0 = 画像上端 = V=0」
        (`architecture.md:73`)。GL のテクセル行 0 は `t = 0` なので、両者は
        一致し**アップロード時の行反転は不要**。読み戻し側の行反転(左下原点)
        とは独立した規約であり、ゲート4 が 2 本ある理由でもある。

        sRGB 変換はしない(`maps` は float32 [0,1] 線形扱い —
        `architecture.md:68`)。`uint8 = round(clip(v, 0, 1) * 255)`。
        """
        if self._basecolor is None:
            return None
        image = np.asarray(self._basecolor, dtype=np.float32)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.shape[2] >= 3:
            rgb = image[:, :, :3]
        else:
            # 単チャンネル(グレースケール)は 3 チャンネルへ複製する。
            rgb = np.repeat(image[:, :, :1], 3, axis=2)
        data = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        height, width = int(data.shape[0]), int(data.shape[1])
        texture = self._ctx.texture((width, height), 3, np.ascontiguousarray(data))
        texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        texture.repeat_x = True
        texture.repeat_y = True
        return texture

    def _apply_gl_state(self, moderngl: Any) -> None:
        """描画ごとに GL 状態を明示する(計画v4 §2.4.2)。

        **WHY 毎回設定するか**: GL の状態は大域で、同一プロセスの他コード
        (テスト・将来の別レンダラ)が触りうる。描画直前に固定すれば、状態の
        取り違えで面IDが壊れる経路が消える。
        """
        self._ctx.enable(moderngl.DEPTH_TEST)
        # AI 生成メッシュは巻き順が不整合なことがある。カリングすると面が消えるので
        # 無効化し、深度テストで最近接面を勝たせる(計画v4 §2.4.1)。
        self._ctx.disable(moderngl.CULL_FACE)
        # 固定小数点カラーバッファを厳密IDに使うのでブレンドは切る(§2.4.2)。
        self._ctx.disable(moderngl.BLEND)
        # ディザは既定で有効。moderngl の定数には無いので生の列挙値で切る
        # (probe (f) 実測: `disable_direct` は GL_NO_ERROR で受理される)。
        # **証拠の等級について正直に書く**(2周目レビュー N2): sentinel ゲートは
        # 「ディザが面IDを壊していないこと」を**反証していない**だけで、
        # ディザ不在を実証してはいない — 実測として、この無効化を注入で外しても
        # 全テストが green のままだった(NVIDIA の RGBA8 では 8bit -> 8bit の
        # 書き込みにディザが掛からないため)。したがってこの一行は
        # 「掛かるドライバがあったときの保険」であり、ゲートの根拠ではない。
        if self._can_disable_dither:
            self._ctx.disable_direct(_GL_DITHER)
        # `FRAMEBUFFER_SRGB` は有効化しない(有効だと色が非線形変換され、面IDの
        # バイトも書き換わる)。既定で無効なので明示的な操作はしない。

    def _read_rgba(self, attachment: int) -> np.ndarray:
        """アタッチメントを `(H, W, 4) uint8` で読み戻し、**行方向を反転**する。

        `glReadPixels` の原点は左下、`maps` / `RenderedView` の規約は
        row 0 = 画面上端。ここで揃えないと V 方向が反転したまま下流へ流れる
        (計画v4 §2.4.2)。
        """
        raw = self._fbo.read(
            attachment=attachment, components=4, dtype="f1", alignment=1
        )
        buf = np.frombuffer(raw, dtype=np.uint8).reshape(
            self.image_size, self.image_size, 4
        )
        return np.ascontiguousarray(buf[::-1])

    def _release_gl(self) -> None:
        """GPU リソースを生成の逆順に解放する(計画v4 §2.1)。

        順序は fbo → texture/renderbuffer → vao → vbo → program → ctx。
        **解放中の例外は握り潰さず記録して続行する**: `__exit__` が例外を投げると
        伝播中の元例外を覆い隠すし、1 個の解放失敗で残りを漏らすのはもっと悪い。
        """
        ordered = (
            ("fbo", self._fbo),
            ("color_texture", self._color_texture),
            ("code_texture", self._code_texture),
            ("depth", self._depth),
            ("basecolor_texture", self._texture),
            ("vao", self._vao),
            ("vbo", self._vbo),
            ("program", self._program),
            ("ctx", self._ctx),
        )
        for name, obj in ordered:
            if obj is None:
                continue
            try:
                obj.release()
            except Exception as e:
                LOG.warning("failed to release GL resource %s: %s", name, e)
        self._fbo = None
        self._color_texture = None
        self._code_texture = None
        self._depth = None
        self._texture = None
        self._vao = None
        self._vbo = None
        self._program = None
        self._ctx = None


def _build_vertex_buffer(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray | None,
    face_codes: np.ndarray,
) -> np.ndarray:
    """面ごとにアンロールした頂点属性 `(3M, 11) float32` を組む。

    列は `in_position(3) | in_uv(2) | in_code(3) | in_normal(3)`。

    **WHY アンロールするか**(計画v4 §2.4.2): `gl_PrimitiveID` の
    「ジオメトリシェーダ無しのときの値」というスペックの隅に依存せず、面IDを
    ただの頂点属性として渡せる。M <= 数十万なら 3M 頂点のコストは無視できる。

    法線は `adjacency.face_normals` を再利用する(零面積面の扱い —
    法線 `[0,0,0]` → 符号化して灰色 — を 2 箇所に書かないため)。
    """
    positions = vertices[faces].reshape(-1, 3).astype("f4")
    n_corners = positions.shape[0]
    if uv is None:
        uvs = np.zeros((n_corners, 2), dtype="f4")
    else:
        uvs = np.asarray(uv, dtype=np.float32)[faces].reshape(-1, 2).astype("f4")
    codes = np.asarray(_build_code_attribute(face_codes), dtype="f4")
    if codes.shape != (n_corners, 3):
        raise ValueError(
            f"face-id attribute must have shape {(n_corners, 3)}, got {codes.shape}"
        )
    normals, _zero_area = face_normals(vertices, faces)
    normal_attr = np.repeat(normals.astype("f4"), 3, axis=0)
    return np.ascontiguousarray(
        np.hstack([positions, uvs, codes, normal_attr]).astype("f4")
    )


def _vao_content(program: Any, vbo: Any) -> list[tuple[Any, ...]]:
    """プログラムに実在する属性だけを名指しし、残りはパディングで読み飛ばす。

    **WHY**(probe (c) で実測): GLSL のリンカは、フラグメントシェーダが読まない
    varying を経由する頂点属性を除去してよい。固定のフォーマット文字列で
    `ctx.vertex_array` を呼ぶと、除去が起きた環境でだけ `KeyError: 'in_uv'` で
    落ちる(当開発機でも縮小 FS で再現した)。実在する属性だけを名指しし、
    消えた属性は `Nx4` パディングで読み飛ばせば、頂点バッファの
    レイアウトを変えずに済む。
    """
    parts: list[str] = []
    names: list[str] = []
    for name, fmt, n_floats in _VERTEX_LAYOUT:
        if _has_member(program, name):
            parts.append(fmt)
            names.append(name)
        else:
            parts.append(f"{n_floats}x4")  # float32 n 個ぶんのパディング
    return [(vbo, " ".join(parts), *names)]


def _has_member(program: Any, name: str) -> bool:
    """`program` に `name` の属性/uniform が実在するか。"""
    try:
        program[name]
    except KeyError:
        return False
    return True


def _warn_if_screen_coverage_is_tiny(coverage: np.ndarray, *, view_index: int) -> None:
    """対象が画面をほとんど覆っていないとき警告する(2周目レビュー B4)。

    **WHY 必要か**(実測): カメラは AABB の**外接球**に合わせるので、細長い形状では
    投影面積が画面のごく一部になる。`1 x 1e-2 x 1e-2` の針では被覆 0.95%、
    `1 x 1e-6 x 1e-6` では **0 画素**で、以前は例外もログも無しに「全面背景」の
    view を返していた。破綻するのは短い辺が 2 本あるときで、扁平(1:1:1e-4)は
    画面の 29% を占めるので無害。

    **WHY `ValueError` ではなく警告か**: 「観測できなかった」ことは*品質*の条件で
    あって契約違反ではない。計画v4 §2.4.1 は未観測面を許容する設計(`visible_ratio`
    / `assigned_ratio` の警告で degrade を伝え、幾何プライアへ劣化する)であり、
    §2.6 が `ValueError` を割り当てているのは**データが壊れている**場合
    (`coverage <=> face_id` の破れ・面数上限超過)だけ。実例として、薄板を真横から
    見た視点は 0 画素になるが他の視点は健全 — ここで例外を投げると、パイプラインが
    扱えるはずの入力を 1 視点の都合で全部捨てることになる。**全視点で観測できて
    いない**という全体判断は Step 2-4 の `visible_ratio` ゲートの責務。
    """
    ratio = float(coverage.mean())
    if ratio == 0.0:
        warnings.warn(
            f"view {view_index}: the mesh drew 0 pixels, so this view observes "
            "nothing at all. The camera frames the bounding sphere, which is a "
            "poor fit for very thin or needle-like shapes; try a larger "
            "image_size, more views (--seg-views), or `--segmenter geometric`.",
            stacklevel=3,
        )
    elif ratio < _LOW_COVERAGE_WARN_RATIO:
        warnings.warn(
            f"view {view_index}: the mesh covers only {ratio:.4%} of the image. "
            "The camera frames the bounding sphere, which is not an efficient fit "
            "for elongated shapes, so masks may miss most faces; try a larger "
            "image_size, more views (--seg-views), or `--segmenter geometric`.",
            stacklevel=3,
        )


def _warn_if_far_from_origin(vertices: np.ndarray) -> None:
    """AABB 中心が寸法に比べて極端に遠いとき、float32 精度の劣化を警告する。

    **WHY 必要か**(2周目レビュー N1・実測): `validate_frustum` は float64 で包含を
    保証するが、頂点は GPU へ f4 で送るので、原点から遠いメッシュでは座標の
    有効桁が足りなくなる。`|centre|/R = 2e4` で全画素一致率 0.999179、`2e6` で
    内部画素が 6226 件不一致(= 主ゲートなら FAIL)になったのに、production には
    ガードが無かった。**黙って劣化させない**ためにここで知らせる(座標系の
    再原点化は呼び出し側の判断なので、こちらでは移動しない)。
    """
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    centre = (lo + hi) * 0.5
    radius = float(np.linalg.norm(vertices - centre, axis=1).max())
    if radius <= 0.0 or not np.isfinite(radius):
        return  # 縮退メッシュは `cameras` 側が `ValueError` で扱う。
    ratio = float(np.linalg.norm(centre)) / radius
    if ratio > _FAR_FROM_ORIGIN_WARN_RATIO:
        warnings.warn(
            f"the mesh sits {ratio:.3g} times its own size away from the origin "
            "(|AABB centre| / bounding radius). Vertices are uploaded as float32, "
            "so at this offset the coordinate rounding is comparable to a pixel "
            "and face ids near shared edges can be mis-assigned. Re-centre the "
            "mesh on the origin before segmenting.",
            stacklevel=3,
        )

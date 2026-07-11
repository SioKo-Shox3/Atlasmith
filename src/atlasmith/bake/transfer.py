"""UV 間テクスチャ転写のコア。

新 UV をテクセル空間にラスタライズ → 行・corner 整列済みの旧 UV から各テクセルの
旧 UV を重心補間 → 旧テクスチャをバイリニアサンプル → ガター(チャート外縁)を
最近傍値で膨張、の順で焼き直す。numpy と標準ライブラリのみ(横断規約の依存方向:
`bake → numpy のみ`)。

座標規約(横断規約の品質契約):
- テクセル (r, c) の中心 UV = ((c + 0.5) / W, (r + 0.5) / H)。
- サンプリング座標 x = u * W - 0.5, y = v * H - 0.5。
  → テクセル (r, c) の中心はテクセル空間で (x = c, y = r) に一致する。
- V 方向: row 0 = 画像上端 = V = 0(glTF 規約)。
- 画素は float32 [0, 1]・channels last。バイリニア境界は clamp。

入力契約(裁定5): `faces_new_uv` と `faces_old_uv` は行・corner 整列済み。すなわち
新面 i の corner k に対応する旧 UV は `uv_old[faces_old_uv[i, k]]`。bake は面対応表
(face_map)を持たない — 整列は呼び出し側が face_map で実施する。
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

# 縮退三角形(テクセル空間の符号付き面積がほぼ 0)は転写に寄与しないため除外する。
# しきい値はテクセル空間の面積(= |det|)基準。実運用解像度(〜512^2)では有効な
# 三角形の面積は 1 を大きく超えるため、この微小値で正常な三角形を誤って落とさない。
_DEGENERATE_AREA2 = 1e-12

# 重心座標での辺の内外許容。テクセル中心が辺上(w ≈ 0)に載ったときの取り込みを
# top-left 規則で一意に決めるための微小量。辺関数そのものは三角形サイズに比例して
# しまうため、面サイズに依存しない正規化済み重心 w = e / det に対して判定する。
_BARY_EPS = 1e-9

# ガター膨張の近傍走査順。直交(距離 1)を対角(距離 √2)より先に見て「最近傍値」の
# コピー元を優先する。順序を固定してコピー元選択を決定的にする(再現性のため)。
_GUTTER_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


class BakeResult(NamedTuple):
    """`bake_maps` の戻り値(裁定4)。

    フィールド:
        maps: 焼き上がりテクスチャ `{name: (H, W, C) float32}`。チャート外かつ
            ガター外は 0。入力 `maps_old` と同じキー・同じチャンネル数を保つ。
        chart_coverage: チャート(= 新 UV 三角形に覆われたテクセル)`(H, W) bool`。
            ガター膨張前のカバレッジ。
        valid_mask: ガター膨張後の有効テクセル `(H, W) bool`。chart_coverage を
            padding_px 回 8 近傍で非循環膨張したもの。chart_coverage とは別フィールド
            (膨張前後を区別できるようにする)。
    """

    maps: dict[str, np.ndarray]
    chart_coverage: np.ndarray
    valid_mask: np.ndarray


def bake_maps(
    faces_new_uv: np.ndarray,
    uv_new: np.ndarray,
    faces_old_uv: np.ndarray,
    uv_old: np.ndarray,
    maps_old: dict[str, np.ndarray],
    *,
    size: tuple[int, int],
    padding_px: int,
) -> BakeResult:
    """旧 UV/旧テクスチャ群を新 UV レイアウトへ焼き直す。

    引数:
        faces_new_uv: 新 UV の面 `(M, 3) int`(`uv_new` への index)。
        uv_new: 新 UV 頂点 `(N_new, 2) float`。
        faces_old_uv: 旧 UV の面 `(M, 3) int`(`uv_old` への index)。
            `faces_new_uv` と行・corner 整列済みであること(入力契約・裁定5)。
        uv_old: 旧 UV 頂点 `(N_old, 2) float`。
        maps_old: 旧テクスチャ群 `{name: (H, W, C) float32 [0, 1]}`。
        size: 出力テクスチャの `(height, width)`(= 出力配列の先頭2軸)。
        padding_px: ガター膨張の反復回数(テクセル単位)。0 で膨張なし。

    戻り値:
        BakeResult(maps / chart_coverage / valid_mask)。
    """
    height, width = size
    faces_new = np.asarray(faces_new_uv, dtype=np.int64)
    faces_old = np.asarray(faces_old_uv, dtype=np.int64)
    uv_new_arr = np.asarray(uv_new, dtype=np.float64)
    uv_old_arr = np.asarray(uv_old, dtype=np.float64)
    if faces_new.shape != faces_old.shape:
        raise ValueError(
            "bake_maps: faces_new_uv と faces_old_uv は同 shape(行・corner 整列済み)"
            f"でなければならない: {faces_new.shape} vs {faces_old.shape}"
        )

    # 対応付け(テクセル → 旧 UV)は全マップで共通なので1度だけ計算して使い回す。
    chart_coverage, old_uv_per_texel = _rasterize_correspondence(
        faces_new, uv_new_arr, faces_old, uv_old_arr, height, width
    )
    valid_mask = _dilate_mask(chart_coverage, padding_px)

    sample_u = old_uv_per_texel[..., 0][chart_coverage]
    sample_v = old_uv_per_texel[..., 1][chart_coverage]

    baked: dict[str, np.ndarray] = {}
    for name, map_old in maps_old.items():
        map_old64 = np.asarray(map_old, dtype=np.float64)
        channels = map_old64.shape[2] if map_old64.ndim == 3 else 1
        out = np.zeros((height, width, channels), dtype=np.float64)
        sampled = _bilinear_sample(map_old64, sample_u, sample_v)
        if sampled.ndim == 1:
            sampled = sampled[:, np.newaxis]
        out[chart_coverage] = sampled
        out = _grow_gutter(out, chart_coverage, padding_px)
        baked[name] = out.astype(np.float32)

    return BakeResult(maps=baked, chart_coverage=chart_coverage, valid_mask=valid_mask)


def _rasterize_correspondence(
    faces_new: np.ndarray,
    uv_new: np.ndarray,
    faces_old: np.ndarray,
    uv_old: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """新 UV をラスタライズし、各テクセルに対応する旧 UV を重心補間する。

    戻り値: (chart_coverage `(H, W) bool`, old_uv `(H, W, 2) float64`)。
    非被覆テクセルの old_uv は 0(サンプリングされないため値は不問)。
    """
    coverage = np.zeros((height, width), dtype=bool)
    old_uv = np.zeros((height, width, 2), dtype=np.float64)

    for tri_new, tri_old in zip(faces_new, faces_old):
        p = uv_new[tri_new]  # (3, 2) 新 UV corner
        o = uv_old[tri_old]  # (3, 2) 旧 UV corner(整列済み — 同じ corner 順)
        # UV → テクセル空間。
        x0, y0 = p[0, 0] * width - 0.5, p[0, 1] * height - 0.5
        x1, y1 = p[1, 0] * width - 0.5, p[1, 1] * height - 0.5
        x2, y2 = p[2, 0] * width - 0.5, p[2, 1] * height - 0.5
        o0, o1, o2 = o[0], o[1], o[2]

        # 符号付き面積で巻き順を正規化(det > 0 の CCW に揃える)。UV 三角形は両巻き順
        # あり得るため(S3)。置換は新 UV corner と旧 UV corner に同一に適用し、重心
        # 座標と旧 UV の対応を崩さない。
        det = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        if abs(det) < _DEGENERATE_AREA2:
            continue
        if det < 0.0:
            x1, y1, x2, y2 = x2, y2, x1, y1
            o1, o2 = o2, o1
            det = -det

        # BBox を [0, W-1] x [0, H-1] に clamp する。これが画像外・反対端への回り込み
        # (C9/T4)を遮断する唯一の要 — ここでは np.roll を使わない。
        cmin = max(0, int(np.floor(min(x0, x1, x2))))
        cmax = min(width - 1, int(np.ceil(max(x0, x1, x2))))
        rmin = max(0, int(np.floor(min(y0, y1, y2))))
        rmax = min(height - 1, int(np.ceil(max(y0, y1, y2))))
        if cmin > cmax or rmin > rmax:
            continue
        cc = np.arange(cmin, cmax + 1)
        rr = np.arange(rmin, rmax + 1)
        gx, gy = np.meshgrid(cc, rr)  # (nr, nc): gx = 列(x), gy = 行(y)

        # 辺関数(det > 0 の CCW 基準)。各 wi は頂点 i に対する重心座標。
        e0 = (x2 - x1) * (gy - y1) - (y2 - y1) * (gx - x1)  # 頂点0 対辺(v1→v2)
        e1 = (x0 - x2) * (gy - y2) - (y0 - y2) * (gx - x2)  # 頂点1 対辺(v2→v0)
        e2 = (x1 - x0) * (gy - y0) - (y1 - y0) * (gx - x0)  # 頂点2 対辺(v0→v1)
        w0, w1, w2 = e0 / det, e1 / det, e2 / det
        inside = (
            _edge_inside(w0, _is_top_left(x1, y1, x2, y2))
            & _edge_inside(w1, _is_top_left(x2, y2, x0, y0))
            & _edge_inside(w2, _is_top_left(x0, y0, x1, y1))
        )
        if not inside.any():
            continue

        # 新 UV 側の重心で旧 UV corner を混ぜる(整列済み契約なので同じ重心で良い)。
        interp_u = w0 * o0[0] + w1 * o1[0] + w2 * o2[0]
        interp_v = w0 * o0[1] + w1 * o1[1] + w2 * o2[1]
        cov_sub = coverage[rmin : rmax + 1, cmin : cmax + 1]
        old_sub = old_uv[rmin : rmax + 1, cmin : cmax + 1]
        cov_sub[inside] = True
        old_sub[inside, 0] = interp_u[inside]
        old_sub[inside, 1] = interp_v[inside]

    return coverage, old_uv


def _is_top_left(ax: float, ay: float, bx: float, by: float) -> bool:
    """有向辺 a→b(det > 0 の CCW・y 下向き)が top-left 辺か。

    共有辺を隣接三角形の一方だけが所有するための tie-break 規約。y が下向きの CCW では
    「左辺 = 下向き(ey > 0)」「上辺 = 水平かつ左向き(ey == 0 かつ ex < 0)」。隣接
    三角形は同じ辺を逆向きに走査するため、どちらか一方だけが top-left と判定される。
    """
    ex, ey = bx - ax, by - ay
    return (ey > 0.0) or (ey == 0.0 and ex < 0.0)


def _edge_inside(w: np.ndarray, top_left: bool) -> np.ndarray:
    """辺 i の内側判定。top-left 辺は辺上(w ≈ 0)を含め、それ以外は含めない。"""
    if top_left:
        return w >= -_BARY_EPS
    return w > _BARY_EPS


def _bilinear_sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """`img` を UV 座標 (u, v) でバイリニアサンプル(clamp 境界)。

    `img` は `(H, W, C)` または `(H, W)`。u, v は `(K,)`。戻り値は `(K, C)` または
    `(K,)`。座標規約は x = u*W - 0.5, y = v*H - 0.5(横断規約)。
    """
    height, width = img.shape[:2]
    x = u * width - 0.5
    y = v * height - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1, y1 = x0 + 1, y0 + 1
    fx, fy = x - x0, y - y0
    # clamp 境界: サンプル点が画像外でも端テクセルを引き伸ばす(wrap しない)。
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    if img.ndim == 3:
        fx, fy = fx[:, np.newaxis], fy[:, np.newaxis]
    top = img[y0c, x0c] * (1.0 - fx) + img[y0c, x1c] * fx
    bottom = img[y1c, x0c] * (1.0 - fx) + img[y1c, x1c] * fx
    return top * (1.0 - fy) + bottom * fy


def _shift(src: np.ndarray, dr: int, dc: int, fill: float | bool) -> np.ndarray:
    """近傍 `src[r+dr, c+dc]` を位置 (r, c) へ集める非循環シフト(境界は `fill`)。

    np.roll は反対端へ回り込むため使わない(C9)。境界外はゼロ/False 相当の `fill`
    で埋め、スライスで内側だけコピーする。
    """
    out = np.full_like(src, fill)
    height, width = src.shape[:2]
    r_src0, r_src1 = max(0, dr), min(height, height + dr)
    c_src0, c_src1 = max(0, dc), min(width, width + dc)
    r_dst0, r_dst1 = max(0, -dr), min(height, height - dr)
    c_dst0, c_dst1 = max(0, -dc), min(width, width - dc)
    if r_src0 < r_src1 and c_src0 < c_src1:
        out[r_dst0:r_dst1, c_dst0:c_dst1] = src[r_src0:r_src1, c_src0:c_src1]
    return out


def _dilate_mask(mask: np.ndarray, iters: int) -> np.ndarray:
    """`mask` を 8 近傍で `iters` 回、非循環に二値膨張する。"""
    out = mask.copy()
    for _ in range(iters):
        acc = out.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                acc |= _shift(out, dr, dc, False)
        out = acc
    return out


def _grow_gutter(values: np.ndarray, valid: np.ndarray, iters: int) -> np.ndarray:
    """`valid` の外縁へ最近傍値を `iters` 回伝播する(ガター膨張)。

    各反復で膨張リングを1つ追加し、新たに有効になったテクセルへ「その反復開始時点で
    有効だった近傍」の値をコピーする(平均しない = 最近傍値伝播)。コピー元は
    `_GUTTER_DIRECTIONS` の優先順で最初に見つかった有効近傍。
    """
    cur_vals = values.copy()
    cur_valid = valid.copy()
    for _ in range(iters):
        dilated = _dilate_mask(cur_valid, 1)
        newly = dilated & ~cur_valid
        if newly.any():
            filled = cur_vals.copy()
            remaining = newly.copy()
            for dr, dc in _GUTTER_DIRECTIONS:
                neigh_valid = _shift(cur_valid, dr, dc, False)
                take = remaining & neigh_valid
                if take.any():
                    neigh_vals = _shift(cur_vals, dr, dc, 0.0)
                    filled[take] = neigh_vals[take]
                    remaining &= ~take
            cur_vals = filled
        cur_valid = dilated
    return cur_vals

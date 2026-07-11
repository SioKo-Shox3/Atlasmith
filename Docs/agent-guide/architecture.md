# Atlasmith — アーキテクチャ

<!-- 書くのは「エージェントが編集を誤らないために必要な構造知識」であり、網羅的な設計書ではない。
     コードが正、文書は補助。 -->

> **状態: Phase 0 実装済み・Phase 1 実装中(2026-07-10)。** 本文書は
> `Docs/plans/2026-07-10-phase0-1-plan.md`(承認済み)の横断規約を正として転記したもの。
> Phase 2(①部位分割〜④パッキング独立化)は未着手 — 着手時にこの文書を実態で更新する。

## 全体構造

- `src/atlasmith/types.py` — 公開 IR(`MeshData`)。他の atlasmith サブパッケージに依存せず、
  外部ライブラリは **numpy のみ**(フィールドが `np.ndarray` のため必須。trimesh/PIL/xatlas は import しない)。
- `src/atlasmith/cli.py` — CLI エントリポイント(`atlasmith` コマンド)。他の全レイヤーを束ねる。
- `src/atlasmith/io/`(Phase 0 実装済み)— メッシュ/画像 I/O。`load_mesh` / `save_mesh` /
  `load_image` / `save_image`。trimesh・Pillow はこの層に閉じ込める。
- `src/atlasmith/pack/`(Phase 1 で追加)— UV 再展開+パッキング(xatlas 素朴統合、internal)。
- `src/atlasmith/bake/`(Phase 1 で追加)— テクスチャ転写(numpy のみ、依存追加なし)。
- `src/atlasmith/metrics/`(Phase 1 で追加)— 品質計測(PSNR 等、numpy のみ)。
- `src/atlasmith/segmentation/` `seams/` `unwrap/`(Phase 2 予定、未着手)— 部位分割・シーム決定・
  ML 由来の平面展開。**空パッケージの先行作成はしない**(死んだ骨組みを作らない、coding-style.md)。
- `tests/` — pytest テスト(src とミラー構成)。`Docs/` — 文書。`pyproject.toml` — プロジェクト定義。

## 所有権と寿命

長寿命リソース(プロセス・ファイルハンドル・外部接続・キャッシュ)は現状なし。全処理は
CLI 呼び出し1回で完結する短命プロセス。導入する際はここに生成/破棄の正規ルートを記録する。

## スレッド / 並行性

全段シングルスレッド前提。`MeshData` は値渡しとし、引数の破壊的変更は禁止
(呼び出し側は渡した `MeshData` が変更されないことを前提にできる)。並行処理を採用する場合は
technique-selection.md に記録して決定し、決定後にここへ前提を書く。

## 依存方向(越えてはいけない境界)

- `cli` → `io` / `pack` / `bake` / `metrics`
- `pack` → `types`(+ xatlas)
- `io` → `types`(+ trimesh / Pillow)
- `bake` / `metrics` → **numpy のみ**(trimesh・xatlas・Pillow の import 禁止)
- `types` → **numpy のみ**(atlasmith サブパッケージ・trimesh・xatlas・Pillow の import 禁止)

上記と逆向きの import は禁止(例: `io` が `cli` を import しない、`bake` が `pack` を import
しない)。`bake`/`metrics` を numpy 専用に保つのは、依存を増やさず境界を証明可能にするため
(Step 1-1 の設計方針)。

## 公開 API 契約

公開関数は次の**5関数**+ CLI エントリポイントに限定する:

| 関数 | 所在 | 契約 |
|---|---|---|
| `load_mesh(path) -> MeshData` | `io.mesh` | 単一メッシュ・単一マテリアル・単一UVセットのみ対応 |
| `save_mesh(mesh, path) -> None` | `io.mesh` | 形式は拡張子で判定(GLB/glTF/OBJ) |
| `bake_maps(faces_new_uv, uv_new, faces_old_uv, uv_old, maps_old, *, size, padding_px) -> BakeResult` | `bake.transfer` | 入力は行・corner 整列済み前提(対応表は呼び出し側が用意、bake は対応表を知らない) |
| `masked_psnr(a, b, mask) -> float` | `metrics.quality` | マスクが空なら例外(`ValueError` 系) |
| `rebake(input_path, output_path, *, texture_size=1024, padding_px=8) -> None` | `atlasmith` (`__init__.py`) | io+pack+bake の結線を行う高水準 API。CLI はこの薄いラッパ |

- `MeshData` は**公開・provisional**(pre-1.0 はフィールド追加があり得る旨を docstring に明記)。
- `_naive_unwrap_and_pack(mesh, *, resolution, padding_px) -> tuple[MeshData, ndarray]`
  (`pack.xatlas_naive`)は **internal**。展開③+パッキング④の一体化は Phase 1 の暫定実装であり、
  公開の安定境界(unwrap 済み UV を受けるパッキング)は Phase 2/3 で `pack/` に置く前提。
- `BakeResult`(NamedTuple): `maps: dict[str, ndarray]` / `chart_coverage: ndarray bool`
  (チャート内かどうか)/ `valid_mask: ndarray bool`(ガター膨張後の有効領域)。

## IR・サンプリング品質契約

- 画素は **float32 [0,1]・channels last・data_range=1.0**。PSNR の MSE は全チャンネル・全評価点
  一括で計算する。
- **sRGB は Phase 1 では変換しない**(線形配列として byte 保存的に扱う)。
- alpha は通常チャンネルとして補間する(premultiply しない)。
- サンプラの境界規約は **clamp が既定**(wrap はオプション)。
- maps の V 方向は「row 0 = 画像上端 = V=0」(glTF 規約)。OBJ の V 反転は `io` 層で吸収する。
- **UV↔テクセル変換式**: テクセル `(r, c)` の中心 UV は `((c+0.5)/W, (r+0.5)/H)`。
  サンプリング座標は `x = u*W - 0.5, y = v*H - 0.5`。

## 将来の抽象点(設計記録のみ、コード化しない)

- `SegmentationBackend` — 部位分割(①)の抽象。ML(PartField 予定)と幾何フォールバックを
  差し替え可能にする想定(Phase 2)。
- `SeamStrategy` — シーム決定(②)の抽象。決定的アルゴリズムの複数方式を想定(Phase 2)。
- 左右対称検出→UVミラー配置は将来フェーズ(Phase 2 以降)の拡張候補であり、現時点では未設計。

## 正式制約(承認済み — 判断6、2026-07-10 ユーザー承認)

- 対応範囲は**単一メッシュ・単一マテリアル・単一UVセット**に限定する。複数メッシュ GLB 等は
  明確なエラーメッセージで拒否する。
- **normal map は転写するが警告付き**(タンジェント空間法線マップは UV 変更で基底が変わるため、
  画素転写だけでは照明的に不正確 — 照明的正しさは保証外)。
- **OBJ は basecolor のみ保持**(trimesh の `SimpleMaterial` の制約により、normal・
  metallic_roughness は OBJ 書き出し時に落ちる — 2026-07-10 実測)。
- **OBJ/glTF のサイドカーファイル名は出力パスの stem から導出する**(例: `left.obj` →
  `left.mtl`/`left.png`、`left.gltf` → `left_gltf_buffer_0.bin` 等)。trimesh の既定
  エクスポータはサイドカー名を固定(`material.mtl`/`gltf_buffer_0.bin` 等、出力パスに
  非依存)で返すため、stem 導出をしないと同一ディレクトリへの複数メッシュ保存でサイド
  カーが衝突し、2件目が1件目を黙って上書きする(2026-07-11 verifier 発見・修正済み —
  Step 0-4b)。

## 危険地帯(変更時に必ず計画レビューを通す領域)

1. **技術選定・依存追加**(`pyproject.toml` への依存追加、フレームワーク採用)—
   technique-selection.md の記録+ユーザー承認が必須。
2. **公開 API・IR の変更**(上記5関数・`MeshData`・`BakeResult` のシグネチャ/契約) —
   後方互換に影響し、テスト全体が前提にしている。
3. **bake の数値規約**(品質契約の閾値・変換式・PSNR ゲート) — テストの合否閾値と直結する
   (恒等転写ゲート `max abs diff ≤ 1e-6`、一方向オラクルの PSNR 閾値等)。
4. **ワークフロー資産**(CLAUDE.md / AGENTS.md、`.claude/`、`.codex/`、`Docs/agent-guide/`)—
   ミラー同一性と Codex フックの trust(→ codex-delegation.md「運用の落とし穴」)を壊しやすい。

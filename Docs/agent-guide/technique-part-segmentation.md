# 技術選定記録: 幾何部位分割バックエンド(①部位分割・非ML経路)

日付: 2026-07-27 ／ 状態: **決定済み** ／
決定者: ユーザー承認(計画v2裁定Aでは「保留」だったが、2026-07-27付ユーザー裁定Aで「実装再開」—
詳細は `Docs/agent-guide/technique-ml-part-segmentation.md`「ユーザー裁定(2026-07-27)」節を参照)

## 課題定義

- **解こうとしている問題**: メッシュを部位へ分割する幾何的手段。Phase 2 の①部位分割段のうち、
  ML を使わない経路。
- **制約(最重要)**: 滑らかな有機的キャラクター(AI生成のマーチングキューブ由来メッシュ)では、
  二面角クラスタリングは **P=1 に退化する**【実測 2026-07-26、ユーザー自身による試行 — 出所と
  詳細は `Docs/agent-guide/technique-ml-part-segmentation.md` の課題定義節】。
  → **幾何単独では主用途(AI生成の有機的キャラクター)を満たさない**。本記録は
  (a) 「ML 非搭載時のフォールバック」、(b) 「2パス融合(`MultiViewSegmenter`)の幾何プライア供給元」
  という**2つの役割に限定した選定**であることを明確にする。
- Apache-2.0(絶対規則3: 非互換依存を持ち込まない)。Windows開発機での動作を前提とする。
- **成功基準**: (a) 追加依存なしで動く、(b) 決定的(ビット単位で再現可能)、(c) 鋭い折れ目を持つ
  ハードサーフェスや複数連結成分を持つメッシュで妥当な分割が得られる、(d) 2パス融合の幾何プライア
  として機能する(未観測領域の劣化先として妥当)。

## 候補
(「何もしない/今はやらない」を含む)

| 候補 | 概要 | 主な根拠 | 主な懸念 |
|---|---|---|---|
| **(a) 位置weld+二面角+union-find(採用)** | `np.lexsort` による厳密ビット一致weld → 面隣接 → 二面角しきい値クラスタリング → 自前union-find | 追加依存ゼロ・決定的・計画v2(2026-07-14)で設計済み | 滑らかな有機形状で P=1 に退化する(上記制約により主用途を単独では満たさない) |
| (b) scipy(csgraph等)を導入 | `scipy.sparse.csgraph.connected_components` 等で連結成分を求める | 実装が枯れている | 依存を1つ増やす対価に見合わない(union-find は自前で30行程度で足りる) |
| (c) libigl の幾何処理を導入 | SLIM/ARAP 等の高度な幾何アルゴリズムを利用 | 理論的に高品質な展開・部位分割が可能 | 当機で pip 導入不可の実測(下記引用) |
| (d) 何もしない | Phase 1 の naive 経路(素朴 xatlas 展開、部位概念なし)のまま | 実装コストゼロ | 「部位単位アイランド」という Phase 2 の価値そのものが得られない |

## 証拠等級付き比較

- **(a) 採用**: 追加依存ゼロ。位置weldは `np.lexsort` による辞書順ソート→厳密ビット一致グループ化、
  面隣接は辺を weld代表indexで正規化しちょうど2面共有のみを隣接とする、union-findは純numpy/Pythonの
  自前実装(30行程度)で足りる【計画v2/v4での設計に基づく — `Docs/plans/2026-07-14-phase2-plan.md:64-84`、
  `Docs/plans/2026-07-27-phase2-plan-v4.md:354-363`】。
- **(b) scipy 不採用**: csgraph の連結成分機能はunion-findの代替になるが、依存を1つ増やす対価に見合わない
  【計画v4 §3の判断、`Docs/plans/2026-07-27-phase2-plan-v4.md:590`】。同じ理由で Hungarian 法
  (scipy.optimize)も ML 品質ゲートの一対一マッチングには使わない(貪欲近似で代替 — 同ファイル §2.4.5(a)
  BL-5(a))。
- **(c) libigl 不採用(延期)**: libigl は当機で pip 導入不可の実測
  【実測2026-07-14、`Docs/plans/2026-07-14-phase2-plan.md:50` の記録の引用 — 本セッションでは再測していない】
  — PyPI 2.6.2 に win wheel 無し、sdist は nanobind の CMake ビルドが失敗。同件は
  `Docs/agent-guide/technique-core-stack.md` の libigl 節(38行の却下理由・41行の再評価トリガー(4))にも
  記録されており、本記録では詳細を重複させず参照のみとする。
- **(d) 何もしない不採用**: Phase 2 のスコープそのものを放棄することになり、成功基準(c)(d)を満たさない。

## プロトタイプコスト

- 追加のスパイクは不要。位置weld+二面角+union-findは計画v2(2026-07-14)で設計済みであり、
  計画v4(2026-07-27)で `DihedralSegmenter` として無改訂流用が確定している
  【`Docs/plans/2026-07-27-phase2-plan-v4.md:354-363`「v2 §2.1 を無改訂で流用」】。

## Advisor往復

- なし(本記録時点)。

## 決定記録

- **採用**: `DihedralSegmenter`(位置weld+二面角+連結成分)。既定 `angle_deg=60.0` /
  `min_faces=None`(自動 `max(2, M // 100)`)。
  - 位置weldは `trimesh.load(path, process=False)` で weld されない実装事実に依存する
    【実測・本セッションで実ファイル確認、`src/atlasmith/io/mesh.py:107-118`, `:141-142`】。
    `load_mesh` は `process=False` で trimesh 側のweld/後処理を無効化し、`source_vertex` は
    io では恒等写像(`np.arange`)として返す。
  - P 上限ガード(既定1024)は `_part_unwrap_and_pack` 入口に据え置き
    (`Docs/plans/2026-07-27-phase2-plan-v4.md:513`)。
  - アトラス寸法規約は裁定D = 案(b)(等方スケール+正方形出力、`D = max(width, height)`、
    計画v2で承認済み・再確認不要)。
- **却下**:
  - (b) scipy — 依存追加の対価に見合わない。
  - (d) 何もしない — Phase 2 のスコープ(部位単位アイランド)を満たさない。
- **延期**:
  - (c) libigl — Windows 導入不可の実測(引用)により Phase 3 以降へ延期
    (`Docs/agent-guide/technique-core-stack.md` のlibigl節を参照)。
- **再評価トリガー**:
  1. SAM2 経路(`MultiViewSegmenter`)が No-Go となり幾何のみで着地する判断になった場合
     (計画v4 Step 2-1.5 の No-Go 分岐(iii)「Phase 2 を幾何のみで着地させ ML を Phase 3 へ送る」)。
  2. libigl の Windows 導入性が変わった場合(wheel 提供・ビルド要件の変化)。
  3. 幾何プライアが2パス融合の品質を損なうと実測された場合
     (計画v4 §0-A 条件12「2パス化の副作用」— 部分遮蔽で滑らかな面が過分割されうる懸念)。

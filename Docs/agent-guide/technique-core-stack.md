# 技術選定記録: core-stack

日付: 2026-07-10 ／ 状態: 決定済み ／ 決定者: ユーザー承認(2026-07-10、AskUserQuestion で「atlasmith (Recommended)」「uv+3.12+hatchling (Recommended)」等を含む計画を承認。品質契約・正式制約の承認を含む)

## 課題定義
- 解こうとしている問題: テクスチャ付きメッシュの入出力・UV 再展開・ベイクを実装する Phase 0/1 のコア依存(メッシュ処理・数値計算・UV アトラス生成・静的解析・テスト)を選定する。
- 制約(技術・規約・予算): Apache-2.0 と非互換のライセンス(GPL 系等)は不可(絶対規則3)。依存追加はユーザー承認が必要(絶対規則1)。Windows 11 + Python 3.12(technique-package-manager.md で確定)上で wheel が入手できること。Phase 0/1 のスコープ(単一メッシュ・単一マテリアル・単一UVセット)を超える重量級依存(GPU 学習済みモデル等)は導入しない。
- 成功基準(何が満たされれば成功か): (1) glTF/OBJ の読み書き・UV アトラスパッキングに必要な機能が揃う。(2) 全依存が Apache-2.0 互換ライセンスで、Windows cp312 wheel が存在する。(3) Step 0-3 で全依存の import が実測で成功する。

## 候補

| 候補 | 概要 | 主な根拠 | 主な懸念 |
|---|---|---|---|
| trimesh + numpy + xatlas + ruff + pytest(採用) | trimesh でメッシュ I/O・幾何処理、numpy で数値計算、xatlas で UV アトラス生成、ruff/pytest は開発ツール | 全て MIT/BSD-3 で Apache-2.0 互換、trimesh は glTF/OBJ 読み書きを標準サポート、xatlas は業界標準の UV アトラスアルゴリズムの Python バインディング | xatlas は非公式バインディングでメンテナが小規模(ただし継続メンテ・週DL数万で実用十分と判断) |
| libigl を今回から採用 | trimesh に加えて libigl(MPL-2.0)の幾何処理関数群も同時導入 | 高度な幾何アルゴリズム(パラメータ化・リメッシュ等)が使える | MPL-2.0 は未改変利用なら Apache-2.0 から利用可能だが、wheel に copyleft サブモジュール(CGAL/tetgen 等)が含まれるか未確認。Phase 0/1 のスコープ(単純な入出力・恒等転写・一方向オラクル)には過剰で、ライセンス確認コストに見合わない。 |
| PartField を今回から採用 | NVIDIA の学習済みシーム生成モデルを導入 | 高品質な自動シーム生成が期待できる | NVIDIA 独自ライセンスで Apache-2.0 非互換の可能性大。torch-scatter/open3d/pymeshlab 等の重量級依存が必要でPhase 0/1 のスコープを大幅に超える。 |
| 何もしない(依存ゼロで自前実装) | glTF パーサ・UV アトラスパッキングを独自実装 | 依存追加の承認・ライセンス確認が不要 | glTF パーサとアトラスパッキングの車輪の再発明でスコープを大幅に超過し、Phase 0/1 の期限内に実装しきれない。不採用。 |

## 証拠等級付き比較

| 論点 | trimesh+numpy+xatlas(採用) | libigl 追加 | PartField 追加 | 何もしない |
|---|---|---|---|---|
| ライセンス | trimesh MIT【外部: PyPI trimesh 4.12.2、2026-05-01 参照】、numpy BSD-3【外部: PyPI numpy 参照】、xatlas MIT【外部: PyPI xatlas 0.0.11、2025-07-04 参照】、ruff/pytest MIT【外部: PyPI 参照】。全て Apache-2.0 互換。 | MPL-2.0【外部: PyPI libigl 2.6.2 参照】。未改変利用なら Apache-2.0 から利用可能だが、wheel 同梱の copyleft サブモジュール(CGAL/tetgen 等)の有無は未確認【推測: 要 Phase 2 時点での確認】。 | NVIDIA 独自ライセンス【外部: https://github.com/nv-tlabs/PartField、2026-07-10 参照】。Apache-2.0 非互換の可能性大【推測: ライセンス全文未読のため確度は推測】。 | 該当なし(依存を持たない) |
| Windows wheel 対応(cp312) | trimesh は pure Python(wheel 不問)、numpy/xatlas とも cp312 wheel あり【外部: PyPI 各ページ、2026-07-10 参照】 | 2.6.2 は cp38〜cp312 wheel を提供【外部: https://pypi.org/project/libigl/】→ 3.12 は圏内だが上限 | 学習済みモデルは Hugging Face 配布、依存(torch-scatter/open3d/pymeshlab 等)のインストールコストが大きい【外部: 同 GitHub リポジトリ README】 | 該当なし |
| Phase 0/1 スコープとの整合 | glTF/OBJ I/O・UV アトラス生成という Phase 0/1 の中核要件に直接対応【実測: 計画書(2026-07-10-phase0-1-plan.md)のステップ定義との突合】 | 高度な幾何処理は Phase 0/1 の要件(恒等転写・一方向オラクル検証)に不要【推測】 | シーム自動生成は Phase 2 相当の機能でPhase 0/1 の正式制約(単一メッシュ・単一マテリアル)を超える【推測】 | スコープを満たせない(自前実装コストが計画期間を超過)【推測】 |
| 導入・保守コスト | 週DL数万・継続メンテのバインディングで実用十分と判断【外部: PyPI xatlas ダウンロード統計、2026-07-10 参照】 | ライセンス確認(サブモジュール同梱有無)が Phase 2 まで未了の追加コスト | 依存グラフが重く、Apache-2.0 適合確認にユーザーによるライセンス全文確認が必須という追加ステップが発生 | 実装工数が最大(見積り不能なほど大きい)【推測】 |

## プロトタイプコスト
- 本命候補を安く検証する方法: Step 0-3 で `uv run python -c "import trimesh, numpy, xatlas, PIL"` を実行し、全パッケージの import が例外なく成功するかを実測する。これにより xatlas の非公式バインディングに起因する wheel 事故(依存する Visual C++ ランタイム不足等)を実装着手前に前倒しで検出できる。
- 判定基準: import が成功すれば合格。失敗した場合は該当パッケージのビルド要件(Visual C++ Redistributable 等)を再調査し、解決できなければ本記録の再評価トリガーに従い代替候補を再選定する。

## Advisor往復
- 相談日時・方向・要旨: 2026-07-10、オーケストレーターがコア依存構成(trimesh/numpy/xatlas/ruff/pytest)を含む計画をユーザーへ提示し、AskUserQuestion で承認を得た。パートナーAI(非メイン側)による計画二次レビューでは、libigl のライセンス(MPL-2.0)と PartField のライセンス(NVIDIA 独自)がリスクとして指摘され、両者とも「今回導入しない」判断に反映済み(2026-07-10-phase0-1-plan.md の依存整理の前提)。反例として「libigl を Phase 0 から導入し高度な幾何アルゴリズムを先取りできないか」を検討したが、ライセンス確認コストと Phase 0/1 のスコープ不整合から却下した。

## 決定記録
- 採用: trimesh 4.12.2(MIT)+ numpy(BSD-3)+ xatlas 0.0.11(MIT)+ ruff(MIT)+ pytest(MIT)。理由: 全て Apache-2.0 互換で、Phase 0/1 の中核要件(glTF/OBJ I/O・UV アトラス生成・恒等転写検証)を直接満たすため。
- 却下:
  - libigl 2.6.2(MPL-2.0): 今回導入せず Phase 2 で再評価する。理由: wheel に copyleft サブモジュール(CGAL/tetgen 等)が含まれるか Phase 2 記録時に要確認であり、Phase 0/1 のスコープには過剰なため。
  - PartField: 今回導入しない。理由: NVIDIA 独自ライセンスで Apache-2.0 非互換の可能性大であり、導入判断はユーザーがライセンス全文を確認してから行う必要があるため。将来的に optional 依存 `[ml]` としての設計余地のみ確保する。
  - 何もしない(自前実装): glTF パーサ・アトラスパッキングの再発明でスコープを大幅に超過するため不採用。
- **再評価トリガー**: (1) xatlas バインディングのメンテナンスが停止した場合、(2) trimesh に破壊的変更が入り移行コストが問題になる場合、(3) Phase 2 でシーム生成モデル(SeamGPT 系等)の導入を検討する時点、(4) libigl の wheel 内 copyleft サブモジュール混入の有無が確認できた時点(Phase 2 再評価の前提条件)。

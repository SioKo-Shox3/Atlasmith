# 技術選定記録: package-manager

日付: 2026-07-10 ／ 状態: 決定済み ／ 決定者: ユーザー承認(2026-07-10、AskUserQuestion で「uv+3.12+hatchling (Recommended)」を選択。品質契約・正式制約の承認を含む)

## 課題定義
- 解こうとしている問題: Windows 11 開発環境の Python バージョンを固定し、依存(trimesh/xatlas/libigl 等)の wheel 提供範囲と整合させた上で、パッケージ管理と build backend を決定する。
- 制約(技術・規約・予算): 言語は Python に決定済み(絶対規則1の対象外)。Apache-2.0 非互換の依存を持ち込まない(絶対規則3)。ライセンス・依存追加はユーザー承認が必要(絶対規則1)。追加コスト(ダウンロード・インストール手順)は最小限に抑える。
- 成功基準(何が満たされれば成功か): (1) 開発機で `xatlas`・`libigl` を含む全依存の wheel が解決できる Python バージョンに固定されている。(2) `.python-version` とロックファイルでバージョン・依存が再現可能。(3) build backend が `[build-system]` を持ち、`uv run` でインストール・entry point 生成が機能する。

## 候補

| 候補 | 概要 | 主な根拠 | 主な懸念 |
|---|---|---|---|
| uv + Python 3.12 pin(採用) | uv でツールチェーン管理(`uv python install 3.12`)+ `.python-version` + `uv.lock` | build-and-verify.md の第一候補指定、winget で即導入可、lockfile 標準搭載 | 新しいツールでエコシステムの慣熟度がまだ低い |
| pip + venv + pyenv-win 等 | 標準 venv + 手動 Python バージョン管理 | 枯れた手段、追加ツール不要に見える | Windows での複数バージョン管理が煩雑(pyenv-win 自体が追加依存)、lockfile は pip-tools 等の別ツールが必要で結局ツール数が増える |
| 何もしない(システム Python 3.14 のまま) | 追加ツール無しで既存の 3.14.3 を使う | 導入コストゼロ | xatlas 0.0.11 の Windows wheel は cp311〜cp313 までで cp314 が存在せず、そのままでは依存解決が破綻する【外部】。不成立。 |

## 証拠等級付き比較

| 論点 | uv + 3.12 pin | pip + venv | 何もしない(3.14) |
|---|---|---|---|
| システム Python バージョン | 開発は 3.12 に固定(uv が別途取得・管理) | システムの py launcher が持つバージョンに依存 | 3.14.3(システムにインストール済み)【実測: `python --version` → 3.14.3、2026-07-10】 |
| py launcher の既存バージョン | 3.12 は未保有のため uv が新規取得 | 同左。py launcher は 3.14/3.10/3.9 のみで 3.11/3.12 が無い【実測: 2026-07-10 環境確認】 | 3.14/3.10/3.9 のみ【実測】 |
| xatlas wheel 対応 | 0.0.11 は Windows cp311〜cp313 wheel を提供【外部: https://pypi.org/project/xatlas/、2026-07-10 参照】→ 3.12 は圏内 | 同上(pip でも wheel 範囲は同じ) | cp314 wheel なし → ソースビルド必須になり導入コスト増大【外部: 同URL】 |
| libigl wheel 対応 | 2.6.2 は cp38〜cp312 wheel を提供【外部: https://pypi.org/project/libigl/、2026-07-10 参照】→ 3.12 が上限かつ圏内 | 同上 | cp314 は範囲外【外部: 同URL】 |
| lockfile | `uv.lock` を標準搭載、コミット対象として build-and-verify.md に明記【実測: 文書確認、2026-07-10】 | 標準搭載なし。pip-tools 等の追加ツール導入が必要【推測: 一般的な pip 運用経験に基づく】 | 該当なし |
| 導入コスト(当環境) | winget で導入済み【実測: 2026-07-10 `winget install astral-sh.uv` 成功】 | venv 自体は標準ライブラリで追加コスト無し【推測】。バージョン切替は別途 pyenv-win 等が必要【推測】 | ゼロ(既存のまま) |
| プロジェクト方針との整合 | build-and-verify.md が第一候補として指定【実測: 文書確認】 | 明示的な推奨なし | 明示的な推奨なし |

## プロトタイプコスト
- 本命候補を安く検証する方法: Step 0-3(依存インストール後)で `uv run python -c "import trimesh, numpy, xatlas, PIL"` を実行し、全依存の import が通ることを実測する。これにより wheel 事故(cp312 対応漏れ等)を実装着手前に検出できる。
- 判定基準: import が例外なく完了すれば合格。`ModuleNotFoundError` やビルドエラーが出た場合は、対象パッケージの wheel 対応バージョンを再調査し、必要なら `requires-python` の下限(3.11)へフォールバックする。

## Advisor往復
- 相談日時・方向・要旨: 2026-07-10、オーケストレーターが本記録案(uv+3.12+hatchling)を含む計画をユーザーへ提示し、AskUserQuestion で承認を得た。パートナーAI(非メイン側)による計画二次レビューは `2026-07-10-phase0-1-plan.md` の v2/v3 改訂プロセス内で実施済み(B4: pyproject に `[build-system]` + `version` 追加、判断7: build backend 選定をユーザー承認対象に含めるべきとの指摘)。本記録はその指摘への対応として作成している。反例として「システム Python(3.14)のまま運用できないか」を検討したが、xatlas wheel 欠如により不成立と結論した。

## 決定記録
- 採用: uv によるツールチェーン・依存管理 + Python 3.12 pin(開発環境)+ build backend は hatchling。理由: xatlas(cp311〜cp313)と libigl(cp38〜cp312)の両方の wheel 提供範囲を満たす唯一の安全圏が 3.12 であり、uv がその pin 管理・lockfile・build-and-verify.md 指定の第一候補という条件を満たすため。hatchling はデファクト標準で src レイアウト対応・MIT ライセンスであり、uv_build(uv 密結合・新しい)や setuptools(レガシー寄り)より枯れていて保守しやすい。
- 却下:
  - pip + venv(+pyenv-win 等): lockfile・バージョン切替を別ツールで補う必要がありツール数が増えるため不採用。
  - 何もしない(システム 3.14 のまま): xatlas 0.0.11 に cp314 wheel が存在せず依存解決が破綻するため不成立。
  - build backend: uv_build(uv 密結合で新しく実績が薄いため次点)、setuptools(レガシー寄りで src レイアウト対応がやや煩雑なため次点)。
- ロールバック手順: `winget uninstall astral-sh.uv` で uv を除去 → `uv python uninstall 3.12` で管理下の Python 3.12 を除去 → リポジトリ内は `.python-version` と `uv.lock` を該当コミット前の状態へ revert。
- **再評価トリガー**: (1) xatlas が cp314 wheel をリリースした場合、(2) libigl が cp313 以降の wheel をリリースした場合、(3) uv のメンテナンスが停止・非推奨化した場合、(4) Phase 2 で PartField(torch 系依存)導入時に環境要件(Python バージョン上限・GPU 依存等)が変わる場合。

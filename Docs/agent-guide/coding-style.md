# Atlasmith — コーディング規約

<!-- 「リンターで機械的に検出できない約束」を優先して書く。 -->

> **状態: 初期合意(コード着地前、2026-07-10)。** 実コードが増えたらコードを正とし、
> 食い違いはこの文書を直す。ruff / pytest は確定(ユーザー確認済み)、
> それ以外の候補(uv、型チェッカー)は初回スキャフォールドのユーザー承認で確定させる。

## 型・ライブラリの規約

- **Python 3.11+**(正確な下限は pyproject.toml が正)。
- **公開関数・メソッドには型ヒント必須。** `Any` への逃げは WHY コメント付きでのみ許可。
- 標準ライブラリで足りるものに依存を増やさない。依存追加は技術選定
  ([technique-selection.md](technique-selection.md))+ユーザー承認を通す
  (architecture.md 危険地帯 1)。
- print デバッグをコミットしない。ロギングは標準 `logging` 経由
  (ロギング基盤を選定したらここを更新)。

## スタイル・命名

- **ruff が正**(format + lint)。ruff format 既定 = 4スペースインデント、行長 88。
- 命名は PEP 8: モジュール/関数/変数 `snake_case`、クラス `PascalCase`、定数 `UPPER_SNAKE_CASE`。
- import はファイル先頭に集約し、ruff の isort ルール(`I`)に従う。関数内 import は
  循環回避などの理由がある場合のみ、WHY コメント付きで。
- 識別子・API 名は英語。コメント・docstring の本文は日本語可(CLAUDE.md の言語ポリシー)。

## ファイル配置

- src レイアウト: パッケージ本体は `src/atlasmith/`、テストは `tests/`
  (ミラー構成: `src/atlasmith/foo.py` ↔ `tests/test_foo.py`)。
- スキャフォールド時に確定 — 変更する場合はこの文書を更新。

## 行末・エンコーディング

- **LF + UTF-8(BOM なし)**。Windows 上でも LF(スキャフォールド時に `.gitattributes` で
  `* text=auto eol=lf` を設定する)。
- 編集後の確認: `git diff --numstat` と `git diff --ignore-cr-at-eol --numstat` の結果が
  食い違ったら行末を壊している — 修復してから報告する。

## コメント契約(全レイヤー共通)

- 非自明なブロックには**WHYコメント**(何をするかではなく、なぜそうするか・何に縛られているか)。
- モジュール/ファイル冒頭に**目的ヘッダ**(このファイルの責務1〜3行)。
- 検証は規約の読み合わせではなく**コールドリーダー理解テスト**で行う
  (文脈ゼロのエージェントがdiffとコメントだけから意図を再構成できるか —
  詳細は workflow-core/review-lenses.md)。

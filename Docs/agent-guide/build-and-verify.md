# Atlasmith — ビルドと検証

<!-- コマンドは全てコピペで動く形で書く。「証拠」の基準(何を貼れば Done と言えるか)も定義する。 -->

> **状態: コード未着地(2026-07-10)— 現時点で実行可能なゲートは存在しない。**
> 下記は確定済みツール(ruff / pytest)前提の雛形。初回スキャフォールドで実際に動くことを
> 確認してからこの注記を外す。パッケージ/環境マネージャは **uv を第一候補** とする
> (未確定 — technique-selection.md で確定させる)。

## 前提・セットアップ

```
# Python 3.12+ / uv(候補)
uv sync              # 依存の同期(pyproject.toml + uv.lock)
```

## ビルド

```
# 純 Python のためビルド工程は無し。配布物が必要になったら: uv build
```

## テスト

```
uv run pytest                                   # 全テスト
uv run pytest tests/test_foo.py -k case_name    # 単一テスト
```

## 品質ゲート(フェーズを「Done」にする前に、証拠付きで)

```
uv run ruff format --check .    # フォーマット
uv run ruff check .             # リント
uv run pytest                   # テスト
# 型チェッカー(mypy / pyright)は未選定 — technique-selection.md で選定後にここへ追加
```

変更に関係するゲートだけ実行し、どれを実行したか明示する。

## 証拠の基準

- 「テストが通った」ではなく、実行したコマンドと実際の出力を貼る。
- 失敗し得ない出力は証拠にならない。特に pytest は **収集件数を確認する** —
  0 件収集の green はテスト未登録の疑いであり、証拠ではない。
- コード未着地の現段階では「ゲート実行済み」を Done 条件にしない — 何を実行し、
  何が実行できなかったか(理由付き)を明示する。

## コミットしてはいけない生成物

- `.venv/`、`__pycache__/`、`.pytest_cache/`、`.ruff_cache/`、`dist/`、`*.egg-info/`
- `.claude/settings.local.json`(マシンローカル)、`.claude/worktrees/`

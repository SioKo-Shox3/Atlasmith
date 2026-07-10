# Atlasmith agent guide — 索引

エージェント(Claude / Codex)向けの詳細ナレッジ。`CLAUDE.md` / `AGENTS.md` は「働き方の合意」だけを
持ち、プロジェクトの知識・規約はすべてここに置く。

**運用ルール:**
- あるレイヤー/工程に触る前に、該当する文書を読む。
- コードと文書が食い違ったら **コードを正** とし、文書を直すか食い違いを報告する。
- 知識が増えたら CLAUDE.md ではなく **ここに** 追記する(CLAUDE.md は薄く保つ)。
- セッションメモリ(Claude 専用)に溜まった安定した知識は、Codex からも見えるようここへ昇格させる。

| 文書 | 内容 |
|---|---|
| [architecture.md](architecture.md) | 全体構造・所有権/寿命・スレッド・依存方向・危険地帯 |
| [coding-style.md](coding-style.md) | スタイル・命名・型規約・行末/エンコーディング |
| [build-and-verify.md](build-and-verify.md) | ビルド/テスト/品質ゲートのコマンドと証拠の基準 |
| [orchestration.md](orchestration.md) | ワークフロー詳細・計画/レビューのチェックリスト・モデル選定 |
| [codex-delegation.md](codex-delegation.md) | Codex への引き継ぎテンプレートと運用の落とし穴(Codex 自身もこれを読む) |
| [model-playbook.md](model-playbook.md) | モデル特性の観察記録と役割ルーティング・弱点相殺ルール |
| [technique-selection.md](technique-selection.md) | 技術選定の記入用テンプレート(選定のたびにコピーして記録) |

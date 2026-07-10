# Atlasmith — パートナーAI連携ガイド(Codex ⇄ Claude)

Claude と Codex は**どちらもメイン(オーケストレーター)になれる**。このファイルは、
メインでない側(パートナーAI)への委譲・レビュー依頼・相談の定型を定める。

## どちらがメインか(デュアルメイン運用)

- **メイン = ユーザーが起動した側。** Claude Code は `CLAUDE.md`、Codex は `AGENTS.md` を読むが
  中身は完全同一なので、**切り替え操作は存在しない** — Codex に指示を出せば Codex がメインとして
  この合意どおりに振る舞い、Claude 側の資産(この Docs、`.claude/agents/` の役割定義)を活用する。
- サブエージェント: Claude メイン時は `.claude/agents/`、Codex メイン時は `.codex/agents/`(1対1対応)。
- **二次レビューは常に非メイン側の AI**(Claude メイン → Codex 二次、Codex メイン → Claude 二次)。
  ベンダーの異なるモデルは失敗モードも異なるため、これでクロスチェックが構造的に保証される。
  **呼び出しは必ず直接 CLI で行い、プラグインを経由しない**(下記)。
- **実装はメイン側で完結する**(Claude メイン = `implementer` サブエージェント、Codex メイン = 自前の
  implementer)。Codex への実装委譲は任意オプション(2026-07-05 に既定から外した)。
- **フックは両メインで作動する(2026-07-07 に Codex 側の正規移植が完了)**: Claude メイン時は
  `.claude/hooks/`、Codex メイン時は `.codex/hooks/`(apply_patch 実装ガード / Stop ミラーガード /
  SessionStart リマインダー — codex-cli 0.142.5 で E2E 実証済み)。ただし Codex 側フックは
  trust 確立(`~/.codex/config.toml` の [hooks.state])が無いと**警告なしに無効**のままなので、
  導入・変更時は trust-hooks.mjs による再信頼までが完了条件(「運用の落とし穴」参照)。
  フックは安全網にすぎない — CLAUDE.md / AGENTS.md を編集したターンは、指摘される前に
  自分で `cp <編集した方> <もう一方>` → `diff` 一致確認まで済ませる。

## Claude がメインのとき: Codex を直接 CLI で使う(二次レビューは必須ゲート)

`codex exec` は**同期実行**(結果がその場で stdout に返る)なので、ポーリングも状態管理も不要。

```bash
# 二次レビュー(読み取り専用サンドボックス) — 「二次レビューの引き継ぎテンプレート」を埋めて渡す
codex exec --sandbox read-only -C <リポジトリ絶対パス> "<テンプレートを埋めた内容>"

# 長いブリーフは stdin で渡す(- が stdin 読み込み)
codex exec --sandbox read-only -C <リポジトリ絶対パス> - < review-brief.md

# 相談(セカンドオピニオン)も同じ形。結果をファイルにも残す場合:
codex exec --sandbox read-only -o <書き込み可能な絶対パス>/codex-answer.md "<相談ブリーフ>"
```

注意:
- `-o` の出力先は**書き込み可能な絶対パス**にする(不正なパスだと書き込みだけ失敗するが、stdout には結果が出る)。
- **プラグイン(`codex:rescue`)経由で呼ばない。** それは Agent ツールのサブエージェントとして動くため、
  Agent ツールをフックするプラグイン(context-mode 等)にブロックされ、ポーリング失敗・呼び出し失敗の原因になる
  (2026-07-05 に実運用で確認)。直接 CLI は Bash 実行なので干渉を受けない。
- 呼び出しが失敗した場合: メイン側の独立二段(`impl-reviewer` + `verifier` 反証)で代替し、
  Codex ゲートを省略したことを必ずユーザーへ報告する。

## Codex が読むもの

- `AGENTS.md`(リポジトリ直下) — `CLAUDE.md` の完全同一ミラー。働き方の合意。
- `Docs/agent-guide/` — プロジェクト知識はすべてここ。**Claude のセッションメモリは Codex から見えない**ので、共有すべき知識は必ずここに置く。

## Codex がメインのとき: Claude を補助に使う

Claude はヘッドレスモード(`claude -p`)で呼び出せる。相談・レビューは読み取り専用で行う。
**重要(2026-07-06 実測): Codex のサンドボックス内からは `claude -p` はネットワーク遮断で失敗する
(ConnectionRefused)。** 対話セッションの Codex では「サンドボックス外での実行」を承認する形で
呼び出すこと(Codex が承認を求めてくる)。非対話 `codex exec` の中からは呼び出せない。
モデルはエイリアス指定 — バージョンIDを書かない。

```bash
# 相談(セカンドオピニオン) — 読み取り専用・計画モード
claude -p "<下の相談ブリーフ>" --model opus --permission-mode plan

# 二次レビュー — Claude 側の役割定義を継承させる
claude -p "まず .claude/agents/impl-reviewer.md を読み、その役割を完全に引き受けよ。その上で対象 diff(<ブランチ/範囲>)を承認済み計画(<所在>)と突き合わせてレビューせよ。" \
  --model opus --allowedTools "Read" "Grep" "Glob" "Bash(git diff:*)" "Bash(git log:*)"

# 調査を安く外出しする場合
claude -p "<調査依頼>" --model sonnet --permission-mode plan
```

## 相互相談プロトコル(セカンドオピニオン)

**発動条件**(どちらがメインでも同じ):
- 同一フェーズで verifier の反証 or レビュー往復 or 実装やり直しが **2回** 続いた(エスカレーション・ラダー)
- レビュー一次と二次の指摘が矛盾して裁定に迷う
- 自分のベンダーのモデルが系統的に苦手な領域だと感じる(Atlasmith では観測未蓄積 —
  事例が出たら model-playbook.md と併せてここに記録する)

**相談ブリーフ(これを埋めて渡す — 空欄のまま渡さない):**

```text
## 相談
[1〜2文で問題を要約]

## 状況
[何を作ろうとして、何が起きているか。エラー・diff・テスト出力など生の証拠を貼る]

## 試したこと
[試行とその結果を時系列で。「〜のはず」ではなく実際の出力]

## 仮説
[現時点の仮説と、その仮説に自信が持てない理由]

## 質問
[答えてほしい具体的な問い(複数可)。yes/no や選択肢で答えられる形が望ましい]

## 回答形式
根拠付きの見解 + 推奨する次の一手(1つ)+ 却下した代替案(あれば)
```

**ルール:**
- 相談の回答は**助言であって決定ではない** — 採否はメインが判断し、理由をフェーズ記録に残す。
- 相談は読み取り専用で行う(相談相手にコードを書かせない — 書くなら正規の委譲手順に乗せる)。
- 相談で解決しなければ、モデル昇格 or ユーザー相談へ(黙って試行を重ねない)。

## 実装タスクの引き継ぎテンプレート

**仕様が確定した実装は Codex が既定担当**(指示追従・コーディング品質 → model-playbook.md)。
以下を埋めて `codex exec --sandbox workspace-write` で渡す。空欄のまま渡さない。
**ループ禁止条項と代替案報告は削らないこと**(GPT の拘泥ループへの補償装置)。

```text
## タスク
[フェーズの目的と期待される挙動変化。承認済み計画の該当部を貼る]

## 書き込み許可パス
[許可: 具体的なファイル/ディレクトリ]
[禁止: それ以外すべて。特に触ってはいけない共有ファイルを明記]

## 守るべき規約
- Docs/agent-guide/coding-style.md の規約に従う(特に: 公開関数に型ヒント必須 /
  依存追加はユーザー承認なしに行わない / ruff format・ruff check をパスさせる)
- 行末は LF、UTF-8(BOM なし)。編集後に `git diff --numstat` と
  `git diff --ignore-cr-at-eol --numstat` を突き合わせ、食い違いがあれば修復してから報告

## 検証
[実装後に実行するコマンド(Docs/agent-guide/build-and-verify.md より)。現行の既定:
`uv run ruff format --check .` → `uv run ruff check .` → `uv run pytest`(収集件数も報告)。
コード未着地の間は実行可能なもののみ実行し、実行できなかったものは理由付きで報告]

## 進め方の制約
- 最初の報告で「採用した手法と、検討して却下した代替案1つ」を述べること
- **同一アプローチが2回失敗したら停止し、試行内容と失敗を報告して戻すこと(ループ禁止)**。
  手法の転換はオーケストレーター側で判断する

## 報告形式
- 変更ファイル一覧
- 実行した検証コマンドと実際の出力
- 未解決リスク・前提にしたこと
```

## 二次レビューの引き継ぎテンプレート

一次レビュー(最上位 Claude)とは独立に、**クリーンな文脈の別 Codex** に依頼する。

```text
## レビュー依頼
以下の diff を承認済み計画と突き合わせて独立レビューせよ。あなたはこの変更の作者ではない。
実装者・一次レビューの結論は渡さない(引きずらせない)。

## 対象
[git diff の範囲 / ブランチ]

## 計画
[承認済み計画の要点]

## 観点
- 計画との差分(計画外の変更 = スコープ逸脱として指摘)
- [architecture.md の危険地帯に対応する観点]
- 生成物混入・regression の可能性

## 報告形式
指摘ごとに: 位置(path:line)/ 何が問題か / 具体的な帰結 / 重大度(blocker/should-fix/nit)
```

## Codex 単体運用時のサブエージェント

Codex をメインエージェントとして直接使う場合も、Claude 側と同じ役割構成で動ける。
役割定義は `.codex/agents/*.toml`(プロジェクト同梱・コミット対象):

- researcher / planner / plan_reviewer / test_designer / implementer / impl_reviewer / verifier
  — Claude 側 `.claude/agents/` と1対1対応。知識は AGENTS.md + `Docs/agent-guide/` が単一ソースなので、
  役割ファイル自体は薄く保ち、詳細はこの Docs を読ませる。
- レビュー・調査系は `sandbox_mode = "read-only"` で読み取り専用を**サンドボックスレベルで強制**
  (Claude 側の tools 制限と同じ役割)。verifier はゲート実行のため workspace-write だが、ソース編集禁止を指示で縛る。
- モデル名はハードコードせず親セッションから継承し、`model_reasoning_effort` で品質役(xhigh)と
  作業役(medium/high)を分ける。
- 並列度は `.codex/config.toml` の `[agents]`(`max_threads` / `max_depth`)で調整。
  `multi_agent` 機能フラグが有効であること(`codex features list`)。
- Codex 側でも原則は同じ: 実装役と自分の変更のレビュー役を同一エージェントにしない。
  (注: カスタムエージェントのファイル形式は発展途上で変わり得る — 動かなくなったら公式ドキュメント
  https://developers.openai.com/codex/subagents を確認する。)

## 運用の落とし穴(実プロジェクトで実際に踏んだもの)

- **プラグイン経由の呼び出しは壊れる**: `codex:rescue`(Agent ツールのサブエージェント)は、
  Agent ツールをフックするプラグイン(context-mode 等)にブロックされ、タスクのポーリング失敗・
  呼び出し失敗を起こす(2026-07-05 確認)。**常に直接 CLI(`codex exec`)を使う。**
  Claude 側サブエージェントの起動まで失敗する場合も同プラグインを疑い、そのセッションでは
  プラグインを無効化するか新セッションで再開する。
- **スコープ逸脱**: Codex は頼んでいない範囲まで直すことがある。受け入れ前に必ず
  `git diff --stat` で変更ファイル一覧を確認してから中身を見る。
- **プロセス残留**: Codex がスラッシング(同じ失敗の繰り返し)を始めたら殺して、
  タスクを分割・明確化してから再投入する。残留 codex プロセスにも注意。
- **サンドボックス設定**: ヘッドレス実行で shell spawn が失敗する/何も動かない場合は
  `~/.codex/config.toml` の sandbox 設定(Windows は `unelevated`)と書き込み許可を確認。
- **コミットは渡さない**: コミット境界はオーケストレーターが所有する。Codex の成果は
  ワーキングツリーへの提案として受け取り、検査後にオーケストレーター側でコミットする。
- **Codex フックは trust 不一致で警告なしに全滅する**(codex-cli 0.142.5 実測):
  `.codex/hooks.json` の内容を変えると `~/.codex/config.toml` [hooks.state] の trusted_hash と
  不一致になり、該当フックは**エラーも警告も出さずスキップ**される。hooks.json を編集したら
  `.codex/hooks/trust-hooks.mjs`(テンプレート側では `codex-templates/hooks/trust-hooks.mjs`)で
  ハッシュを再計算して [hooks.state] を更新するまでが完了条件(実例: Claude スキーマを未検証転植した matcher "Edit|Write|MultiEdit" は Codex に
  実在しないツール名で、ガードは一度も発火していなかった)。
- **プロジェクト未信頼でも hooks.json は黙って無視される**(codex-cli 0.144.0 実測・
  2026-07-10 Atlasmith オンボードで発見): `~/.codex/config.toml` の
  `[projects.'<小文字の絶対パス>']` に `trust_level = "trusted"` が無いプロジェクトでは、
  [hooks.state] の trust が合っていても `.codex/hooks.json` 自体が読み込まれない(エラーなし)。
  新規プロジェクトのオンボードでは hooks trust と **projects trust の両方**を config.toml に
  登録し、SessionStart の逐語引用テストで発火を実証するまでが完了条件。
- **Codex のツール実名**(0.142.5 実測): ファイル編集は `apply_patch` 一本
  (`tool_input.command` にパッチ文字列、対象は `*** Update/Add/Delete File:` 行)。
  Edit/Write/MultiEdit というツールは存在しない。シェルは Windows でも `Bash`。
  **PostToolUse は apply_patch に発火しない** — 編集の事後検査は Stop フックで行う。
- **Windows の exit code 罠**(実測): フックコマンドは `pwsh -Command` 経由で走り、
  node の exit 2 が 1 に潰れてブロックが黙って fail-open になる。フックコマンドは必ず
  `; exit $LASTEXITCODE` で終端する(hooks.json テンプレートは対応済み)。
- **フックでのメイン/サブ判別**: フックペイロードのトップレベル `agent_id` はサブエージェント
  スレッドのツール呼び出しにだけ付く(実測)。Claude Code と同じ判別設計が使える。
- **trust が守るのは hooks.json の定義だけ**(Codex 二次レビュー指摘・2026-07-07): 実行される
  `.codex/hooks/*.mjs` の中身は trusted_hash の対象外で、スクリプトを書き換えられても trust は
  生きたまま自動実行される(0.142.5 の上流設計)。補償は機構で: フックスクリプトは必ず
  **git 管理下に置き**(改竄・変更が diff に出る)、フック変更はレビュー対象に含める。
  リポジトリ外・未追跡のスクリプトを hooks.json から参照しない。

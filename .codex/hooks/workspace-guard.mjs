#!/usr/bin/env node
// PreToolUse guard (Bash) for Atlasmith: creating a NEW WORKSPACE
// (git worktree add) silently exits the harness — the fresh tree carries a
// STALE AGENTS.md/CLAUDE.md snapshot from the branch point, and its
// .codex/hooks.json path has no trust entries in ~/.codex/config.toml, so
// every hook goes silently inert (measured 2026-07-12: the
// Sembazuru-speed-monitor worktree ran with dead hooks and pre-revision docs).
//
// Workspace creation therefore requires (1) explicit USER approval and
// (2) harness re-establishment as part of the SAME task: fresh mirror files,
// trust entries for the new hooks.json path (trust-hooks.mjs), and a
// session_start firing probe.
//
// Applies to ALL threads (main and subagents — a child creating a workspace
// is just as unharnessed). Fails OPEN on any internal error.
// Approved override for ONE session: set ATLASMITH_ALLOW_WORKSPACE=1.

import process from "node:process";

const OVERRIDE_ENV = "ATLASMITH_ALLOW_WORKSPACE";
// `git worktree add` だけを塞ぐ(外部リポジトリの read-only clone は正当用途があるため
// 対象外)。`git -C <dir> worktree add` 形も拾えるよう git〜worktree 間の隙間を許す。
const WORKSPACE_RE = /\bgit\b[^\n]{0,160}?\bworktree\s+add\b/i;

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  raw += chunk;
});
process.stdin.on("end", () => {
  try {
    if (/^(1|true|yes|on)$/i.test(process.env[OVERRIDE_ENV] || "")) {
      process.exit(0);
    }
    const data = JSON.parse(raw || "{}");
    // ツール入力の形が変わっても検出できるよう tool_input 全体を文字列化して見る。
    const hay = JSON.stringify(data.tool_input || {});
    if (WORKSPACE_RE.test(hay)) {
      process.stderr.write(
        "BLOCKED by the Atlasmith workspace guard: creating a new " +
          "workspace (git worktree add) silently exits the harness — the new " +
          "tree gets a STALE AGENTS.md/CLAUDE.md snapshot and its " +
          ".codex/hooks.json path has NO trust entries, so all hooks go " +
          "silently inert (measured 2026-07-12).\n\n" +
          "Do this instead:\n" +
          "  1. Prefer working on a branch in the CURRENT tree.\n" +
          "  2. If isolation is genuinely needed, get the USER's approval " +
          "first, then treat harness re-establishment as part of the task: " +
          "copy the current AGENTS.md/CLAUDE.md + .codex/hooks into the new " +
          "tree, register trust for the new hooks.json path " +
          "(.codex/hooks/trust-hooks.mjs), and probe that session_start " +
          "fires there.\n\n" +
          `Approved override for ONE session: relaunch with ${OVERRIDE_ENV}=1.\n`,
      );
      process.exit(2);
    }
    process.exit(0);
  } catch {
    // Fail open: never let a guard bug wedge the tool.
    process.exit(0);
  }
});

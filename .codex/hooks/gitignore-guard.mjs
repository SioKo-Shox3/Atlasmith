#!/usr/bin/env node
// PreToolUse guard (Bash) for Atlasmith: force-adding IGNORED paths.
//
// `.gitignore` is a USER decision about what belongs in the repository.
// `git add -f` / `--force` overrides that decision silently, and the result is
// only visible much later (measured 2026-07-19 on NorvesLib: `Docs/Plans/` was
// ignored by the user in June as "working memos", then agents force-added 9
// files under it — including a 1.3 MB screenshot — across several sessions;
// the user found out from the repository state, not from a report).
//
// Rule: never push a change past .gitignore. If an ignored path genuinely has
// to be tracked, that is a request to change .gitignore — take it to the USER.
//
// Applies to ALL threads (main and subagents). Fails OPEN on any internal error.
// Approved override for ONE session: set ATLASMITH_ALLOW_FORCE_ADD=1.

import process from "node:process";

const OVERRIDE_ENV = "ATLASMITH_ALLOW_FORCE_ADD";
// `git add` に -f / --force が付く形だけを塞ぐ。`add` は git のサブコマンド位置に
// 限定する(間に許すのは `-C <dir>` 等のグローバルオプションだけ)— そうしないと
// `git commit -m "add -f support"` のようなメッセージ本文で誤爆する(実測)。
// force フラグの探索は `&&`/`;`/`|` を跨がない(`git add bar && rm -f foo` の誤爆防止)。
// `git push --force` や `git clean -fd` など別サブコマンドは対象外。
const FORCE_ADD_RE =
  /\bgit\b(?:\s+(?:-C\s+\S+|-c\s+\S+|--[A-Za-z][\w-]*(?:=\S+)?|-[A-Za-z]))*\s+add\b(?=[^\n&;|]*?(?:\s-[A-Za-z]*f[A-Za-z]*\b|\s--force\b))/i;

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
    if (FORCE_ADD_RE.test(hay)) {
      process.stderr.write(
        "BLOCKED by the Atlasmith gitignore guard: `git add -f` " +
          "overrides a USER decision about what belongs in the repository, " +
          "and the damage only surfaces later (measured 2026-07-19: 9 files " +
          "force-added under a user-ignored directory across several " +
          "sessions).\n\n" +
          "Do this instead:\n" +
          "  1. If the file is a build artifact / scratch output — do NOT " +
          "commit it. Leave it untracked.\n" +
          "  2. If the file genuinely belongs in the repository, that is a " +
          "request to change `.gitignore`. Report to the USER: which path, " +
          "why it must be tracked, and which ignore rule is in the way. " +
          "Change `.gitignore` only after approval, then `git add` normally.\n" +
          "  3. Never keep a durable, agent-read artifact (roadmap, plan, " +
          "decision record) inside an ignored directory — move it to a " +
          "tracked location instead of forcing it past the rule.\n\n" +
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

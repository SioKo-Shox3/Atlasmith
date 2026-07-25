#!/usr/bin/env node
// Stop guard for Atlasmith: files that are tracked by git despite being
// matched by .gitignore.
//
// `.gitignore` is a USER decision about what belongs in the repository.
// Measured 2026-07-19 on NorvesLib: a directory the user ignored in June as
// "working memos" ended up with 9 tracked files under it (including a 1.3 MB
// screenshot), force-added across several sessions. The user found out from the
// repository state, not from a report.
//
// WHY THIS IS A STOP HOOK AND NOT A COMMAND MATCHER (2026-07-25):
// The first version of this guard blocked `git add -f` by pattern-matching the
// Bash command string. A stop-time review rejected it: a command matcher both
// misfires on innocent text (a commit message that merely mentions the flag)
// and is trivially evaded by rewording (line continuations, aliases,
// `git update-index --add --force`, …). Checking the OUTCOME instead —
// "is anything tracked that .gitignore says should not be?" — has neither
// weakness: it cannot be reworded around, and it stays silent unless the
// repository is actually in the bad state.
//
// Blocks turn completion while the condition holds. Fails OPEN on any error
// (not a repo, git missing, timeout) — a guard bug must never wedge a session.
// Approved override for ONE session: set ATLASMITH_ALLOW_TRACKED_IGNORED=1.

import process from "node:process";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const OVERRIDE_ENV = "ATLASMITH_ALLOW_TRACKED_IGNORED";
const MAX_LISTED = 15;

function main() {
  if (/^(1|true|yes|on)$/i.test(process.env[OVERRIDE_ENV] || "")) return;

  // フック自身の位置からリポジトリ根を導く(cwd に依存しない —
  // 子プロセスや別 cwd から呼ばれても同じ判定になるように)。
  const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");

  let out = "";
  try {
    out = execFileSync(
      "git",
      ["-C", root, "ls-files", "--cached", "--ignored", "--exclude-standard"],
      { encoding: "utf8", timeout: 10000, windowsHide: true, stdio: ["ignore", "pipe", "ignore"] },
    );
  } catch {
    return; // fail open: リポジトリでない/git不在/タイムアウト
  }

  const files = out.split("\n").map((s) => s.trim()).filter(Boolean);
  if (files.length === 0) return;

  const shown = files.slice(0, MAX_LISTED);
  const rest = files.length - shown.length;
  process.stderr.write(
    `BLOCKED by the Atlasmith ignored-tracked guard: ${files.length} ` +
      "file(s) are tracked by git even though .gitignore says they do not " +
      "belong in the repository.\n\n" +
      shown.map((f) => `  - ${f}`).join("\n") +
      (rest > 0 ? `\n  … and ${rest} more` : "") +
      "\n\nThis is a USER decision that was overridden. Resolve it before " +
      "ending the turn:\n" +
      "  1. If these are build artifacts / scratch output, untrack them " +
      "(they stay on disk):\n" +
      "       git rm --cached -- <paths>\n" +
      "     then commit that removal.\n" +
      "  2. If they genuinely belong in the repository, that is a request to " +
      "change `.gitignore` — report to the USER which path, why it must be " +
      "tracked, and which ignore rule is in the way. Change `.gitignore` only " +
      "after approval.\n" +
      "  3. Never keep a durable, agent-read artifact (roadmap, plan, " +
      "decision record) inside an ignored directory — move it to a tracked " +
      "location instead of forcing it past the rule.\n\n" +
      `Approved override for ONE session: relaunch with ${OVERRIDE_ENV}=1.\n`,
  );
  process.exit(2);
}

try {
  main();
} catch {
  // fail open
}
process.exit(0);

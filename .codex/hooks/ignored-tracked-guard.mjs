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
// WIRED FROM BOTH CLIs — this file is the single implementation:
//   Codex main : .codex/hooks.json           → Stop
//   Claude main: .claude/settings.local.json → Stop  (points at THIS path,
//                i.e. `node .codex/hooks/ignored-tracked-guard.mjs`)
// It lives under .codex/ only for historical reasons; the check itself is
// CLI-agnostic (it reads git state, nothing else). Do NOT "fix" this by
// copying it into .claude/hooks/ — a second copy is exactly how mirror-guard
// style duplicates drift apart. A guard wired on only one side is worse than
// no guard, because it reads as covered (measured 2026-07-25: the first
// version of this guard shipped Codex-only and a stop-time review caught it).
//
// ONE NUDGE, THEN LET GO. A Stop hook that keeps blocking can make a session
// impossible to end — a stop-hook feedback loop already wrecked a session once
// in this harness (see failure taxonomy; mirror-guard carries the same brake).
// Unlike the mirror drift, this condition is NOT always fixable by the agent:
// untracking a file may be exactly the decision that belongs to the user. So
// the guard nudges once and then steps aside:
//   brake 1 — `stop_hook_active`: this continuation was already forced by a
//             stop hook, so do not force another one.
//   brake 2 — fingerprint marker (OS temp, keyed by repo path): if the exact
//             same set of offending files was already reported, stay silent.
//             A CHANGED set means new information, so it may nudge again.
// Brake 2 exists because brake 1 depends on a field the host may not send;
// a guard that can hang a session must not rely on a single mechanism.
//
// Blocks turn completion while the condition holds. Fails OPEN on any error
// (not a repo, git missing, timeout) — a guard bug must never wedge a session.
// Approved override for ONE session: set ATLASMITH_ALLOW_TRACKED_IGNORED=1.

import process from "node:process";
import { execFileSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync, rmSync, readdirSync } from "node:fs";
import { tmpdir } from "node:os";

const OVERRIDE_ENV = "ATLASMITH_ALLOW_TRACKED_IGNORED";
const MAX_LISTED = 15;

const sha = (s) => createHash("sha256").update(s).digest("hex");

// ブレーキ2: 同じ違反集合を二度は突きつけない(OS temp のマーカー)。
// 失敗は握りつぶす — マーカーが読めない/書けないだけでガードを壊さない。
//
// マーカーは **リポジトリ×セッション** 単位に分ける。1リポジトリ1スロットにすると、
// 同じリポジトリで2セッションが並行したとき互いのfingerprintを上書きし合い、
// どちらも抑止されずループブレーキが破れる(実測ではなくレビュー指摘 2026-07-25)。
const markerPrefix = (root) => `ignored-tracked-guard-${sha(root).slice(0, 16)}-`;
function markerPath(root, sessionKey) {
  return join(tmpdir(), `${markerPrefix(root)}${sha(sessionKey).slice(0, 16)}.marker`);
}
function alreadyReported(root, sessionKey, fingerprint) {
  try {
    return readFileSync(markerPath(root, sessionKey), "utf8").trim() === fingerprint;
  } catch {
    return false;
  }
}
function rememberReported(root, sessionKey, fingerprint) {
  try {
    writeFileSync(markerPath(root, sessionKey), fingerprint);
  } catch {
    /* ignore */
  }
}
// 違反が解消したら、このリポジトリの **全セッション分** のマーカーを消して再武装する。
// これが無いと、いったん報告した違反集合は「解消 → 同じ集合が再発」しても黙ったまま
// になる(= ガードが静かに無効化される最悪の壊れ方)。解消は全セッションに共通の
// 事実なので、他セッション分も一緒に消すのが正しい。
function forgetReported(root) {
  const prefix = markerPrefix(root);
  try {
    for (const f of readdirSync(tmpdir())) {
      if (f.startsWith(prefix)) {
        try {
          rmSync(join(tmpdir(), f));
        } catch {
          /* ignore */
        }
      }
    }
  } catch {
    /* ignore */
  }
}

function main(payload) {
  if (/^(1|true|yes|on)$/i.test(process.env[OVERRIDE_ENV] || "")) return;

  // 注意: ブレーキ1(stop_hook_active)は **git 検査のあと** に効かせる。
  // 先に return してしまうと、最も普通の解消経路 —「ブロック → 指摘どおり修正 →
  // 再 Stop(このとき stop_hook_active が立つ)」— でマーカーが消えず、
  // 再武装が働かない(レビュー指摘 2026-07-25)。ブレーキ1が抑えるのは
  // 「もう一度止めること」だけで、状態の観測と後片付けは常に行う。

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
  if (files.length === 0) {
    // 解消 → 全セッション分のマーカーを消して再武装。stop_hook_active の有無に
    // かかわらずここを通す(上の注意書き参照)。
    forgetReported(root);
    return;
  }

  // ブレーキ1: この継続自体が stop hook に強制されたものなら、二度は止めない。
  // (後片付けは上で済ませてある)
  if (payload && payload.stop_hook_active) return;

  // ブレーキ2: 同一の違反集合を既に報告済みなら黙る(集合が変われば再度促す)。
  // マーカーはリポジトリ×セッション単位。識別子が取れないホストでは空文字となり、
  // そのホスト内では従来どおりリポジトリ単位の抑止に縮退する。
  const sessionKey = String(
    (payload && (payload.session_id || payload.sessionId || payload.thread_id)) || "",
  );
  const fingerprint = sha(files.slice().sort().join("\n"));
  if (alreadyReported(root, sessionKey, fingerprint)) return;
  rememberReported(root, sessionKey, fingerprint);

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

// stdin から Stop フックのペイロードを読む(stop_hook_active を見るため)。
// stdin が来ない環境でも固まらないよう、'end' が来なければ空扱いで進む。
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  raw += chunk;
});
process.stdin.on("end", () => {
  try {
    let payload = {};
    try {
      payload = JSON.parse(raw || "{}");
    } catch {
      payload = {};
    }
    main(payload);
  } catch {
    // fail open
  }
  process.exit(0);
});
process.stdin.on("error", () => process.exit(0));

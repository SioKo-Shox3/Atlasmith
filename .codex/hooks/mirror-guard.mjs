#!/usr/bin/env node
// Stop-hook mirror guard for Codex-main sessions (Atlasmith working
// agreement): CLAUDE.md and AGENTS.md must stay byte-identical (Claude reads
// one, Codex reads the other).
//
// Why Stop and not PostToolUse (the Claude-side design): probed live on
// codex-cli 0.142.5 (2026-07-07) — PostToolUse does NOT fire for
// `apply_patch`, so an after-edit check is impossible there. Checking at
// turn end instead is MORE robust: it catches drift regardless of how the
// edit happened (apply_patch or Bash).
//
// Semantics: on Stop, if <cwd>/CLAUDE.md and <cwd>/AGENTS.md both exist and
// differ, exit 2 with fix instructions — Codex is forced to continue and
// mirror before finishing the turn. ONE nudge per turn: when
// stop_hook_active is set (this continuation was already forced by a stop
// hook) we exit 0 — a stop-hook feedback loop once wrecked a session (see
// failure taxonomy), and the fix is a single cp, so one nudge is enough.
// Fails OPEN on any internal error.

import process from "node:process";
import fs from "node:fs";
import path from "node:path";

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  raw += chunk;
});
process.stdin.on("end", () => {
  try {
    const data = JSON.parse(raw || "{}");

    // Loop brake: never force continuation twice in a row.
    if (data.stop_hook_active) {
      process.exit(0);
    }

    const cwd = data.cwd || process.cwd();
    const claudeMd = path.join(cwd, "CLAUDE.md");
    const agentsMd = path.join(cwd, "AGENTS.md");
    if (!fs.existsSync(claudeMd) || !fs.existsSync(agentsMd)) {
      process.exit(0);
    }
    const a = fs.readFileSync(claudeMd);
    const b = fs.readFileSync(agentsMd);
    if (!a.equals(b)) {
      process.stderr.write(
        "MIRROR DRIFT: CLAUDE.md and AGENTS.md differ. They must stay " +
          "byte-identical (Claude reads one, Codex reads the other).\n" +
          "Before finishing this turn, copy the file you intentionally " +
          "edited over its mirror and re-verify:\n" +
          `  cp "${claudeMd}" "${agentsMd}"   (or the reverse)\n` +
          `  diff "${claudeMd}" "${agentsMd}"   (must print nothing)\n` +
          "If you did not edit either file this turn, report the drift to " +
          "the user instead of guessing which side is authoritative.\n",
      );
      process.exit(2);
    }
    process.exit(0);
  } catch {
    // Fail open: never let a guard bug wedge the session.
    process.exit(0);
  }
});

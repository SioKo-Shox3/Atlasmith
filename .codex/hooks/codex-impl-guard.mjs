#!/usr/bin/env node
// PreToolUse guard for Codex-main sessions (Atlasmith working
// agreement): the main thread never TYPES implementation code — it plans,
// integrates, and reviews. Implementation belongs to subagents (spawn_agent
// -> .codex/agents/implementer.toml), followed by the mandatory double
// review (Codex-side first review + non-main-AI second review).
//
// Codex facts this guard is built on (probed live, codex-cli 0.142.5,
// 2026-07-07 — see Docs/agent-guide/codex-delegation.md):
// - File edits arrive as ONE tool named `apply_patch`. There is no
//   Edit/Write/MultiEdit; matchers watching those names never fire.
// - The patch text is in tool_input.command; per-file targets appear as
//   "*** Update File: <path>" / "*** Add File: <path>" /
//   "*** Delete File: <path>" (and "*** Move to: <path>") lines.
// - Subagent tool calls carry a top-level `agent_id`; main-thread calls do
//   not — same discrimination as the Claude-side guard.
// - exit 2 + non-empty stderr blocks the call and the stderr text is shown
//   to the model. Any other non-zero exit is treated as a hook FAILURE and
//   fails open. On Windows the hook command runs via `pwsh -Command`, which
//   flattens native exit codes to 1 — hooks.json MUST append
//   `; exit $LASTEXITCODE` or blocking silently degrades to fail-open.
// - Known gap (accepted, same as the Claude-side guard): file writes done
//   through Bash (redirects, heredocs) are not intercepted.
//
// Deliberate, operator-approved override for one session: set
// ATLASMITH_ALLOW_DIRECT_EDIT=1 before launching Codex.
// Fails OPEN on any internal error — the guard redirects, it never bricks.

import process from "node:process";

// ---- per-project configuration -------------------------------------------
const PROJECT = "Atlasmith";
const OVERRIDE_ENV = "ATLASMITH_ALLOW_DIRECT_EDIT";
// Implementation source the main thread must not TYPE. Atlasmith is a
// Python project (decided 2026-07-10). Docs (.md), config (.json/.toml/
// .yaml), and build scripts are NOT blocked — the main thread legitimately
// edits those. Keep this set IDENTICAL to the Claude-side guard
// (.claude/hooks/enforce-codex-impl.mjs).
const IMPL_SOURCE_RE = /\.(py|pyi)$/i;
// ---------------------------------------------------------------------------

const FILE_LINE_RE = /^\*\*\* (?:Update|Add|Delete) File: (.+)$/gm;
const MOVE_LINE_RE = /^\*\*\* Move to: (.+)$/gm;

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  raw += chunk;
});
process.stdin.on("end", () => {
  try {
    const data = JSON.parse(raw || "{}");

    // Subagent calls are ALLOWED — implementation belongs to subagents.
    if (data.agent_id) {
      process.exit(0);
    }

    const override = /^(1|true|yes|on)$/i.test(
      process.env[OVERRIDE_ENV] || "",
    );
    if (override) {
      process.exit(0);
    }

    const patch = (data.tool_input && data.tool_input.command) || "";
    const targets = [];
    for (const m of patch.matchAll(FILE_LINE_RE)) targets.push(m[1].trim());
    for (const m of patch.matchAll(MOVE_LINE_RE)) targets.push(m[1].trim());
    const blocked = targets.filter((p) => IMPL_SOURCE_RE.test(p));

    if (blocked.length > 0) {
      process.stderr.write(
        `BLOCKED by the ${PROJECT} workflow guard: the main thread must NOT ` +
          "type implementation code directly.\n\n" +
          `Refused patch targets: ${blocked.join(", ")}\n\n` +
          "The main session is the ORCHESTRATOR: it plans, integrates, and " +
          "reviews — implementation is delegated. Do this instead:\n" +
          "  1. Delegate the change to the implementer subagent (spawn_agent " +
          "-> .codex/agents/implementer.toml) with the approved plan, allowed " +
          "write paths, and conventions.\n" +
          "  2. First review: impl-reviewer agent (never the author). Second " +
          "review (MANDATORY, non-main AI): interactive session -> " +
          "`claude -p \"<review brief>\" --model opus --permission-mode plan` " +
          "(needs out-of-sandbox approval; sandboxed codex exec has no " +
          "network for claude -p). If unavailable, run a second independent " +
          "Codex review and REPORT the skipped gate.\n" +
          "  3. Only then stage + commit.\n\n" +
          "Do NOT retry this patch on the main thread. Deliberate, " +
          `user-approved override for ONE session: relaunch with env ` +
          `${OVERRIDE_ENV}=1.\n`,
      );
      process.exit(2);
    }

    process.exit(0);
  } catch {
    // Fail open: never let a guard bug wedge the tool.
    process.exit(0);
  }
});

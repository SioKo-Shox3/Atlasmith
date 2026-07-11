#!/usr/bin/env node
// PreToolUse guard (Atlasmith working agreement).
// (Filename is historical — the rule it enforces is "the main thread never
// types implementation code"; implementation belongs to the implementer
// subagent, with a mandatory Codex second review via direct CLI.)
//
// CLAUDE.md workflow: the main thread PLANS, INTEGRATES, and REVIEWS —
// "delegate the typing, never the judgment."
//
// Text in CLAUDE.md alone does not hold (proven on Sembazuru: a session typed
// load-bearing code directly), so this hook makes the harness enforce it: a
// main-thread Edit/Write/MultiEdit/NotebookEdit targeting implementation
// source is BLOCKED, with instructions to route the change through the
// `implementer` subagent instead (subagent calls carry an `agent_id` and are
// allowed through below).
//
// Deliberate, operator-approved override for one session: set the environment
// variable ATLASMITH_ALLOW_DIRECT_EDIT=1 before launching Claude Code.
//
// Exit codes: 2 => block the tool call and surface stderr to the model; 0 =>
// allow. Any internal error fails OPEN (exit 0) so the guard can never wedge
// the session — its job is to redirect, not to brick the tool.

import process from "node:process";

// ---- per-project configuration -------------------------------------------
const PROJECT = "Atlasmith";
const OVERRIDE_ENV = "ATLASMITH_ALLOW_DIRECT_EDIT";
// Implementation source the main thread must not TYPE. Atlasmith is a
// Python project (decided 2026-07-10). Docs (.md), config (.json/.toml/
// .yaml), and build scripts are NOT blocked — the main thread legitimately
// edits those.
const IMPL_SOURCE_RE = /\.(py|pyi)$/i;
// ---------------------------------------------------------------------------

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => {
  raw += chunk;
});
process.stdin.on("end", () => {
  try {
    const data = JSON.parse(raw || "{}");
    const ti = (data && data.tool_input) || {};
    const filePath = ti.file_path || ti.notebook_path || "";

    // Subagent calls are ALLOWED — implementation belongs to the implementer
    // subagent. Per the hooks reference, `agent_id` is present only when the
    // hook fires inside a subagent call; the guard targets the MAIN thread.
    if (data.agent_id) {
      process.exit(0);
    }

    // A deliberate, operator-approved escape hatch (one session).
    const override = /^(1|true|yes|on)$/i.test(
      process.env[OVERRIDE_ENV] || "",
    );
    if (override) {
      process.exit(0);
    }

    if (IMPL_SOURCE_RE.test(filePath)) {
      process.stderr.write(
        `BLOCKED by the ${PROJECT} workflow guard: the main thread must NOT ` +
          "type implementation code directly.\n\n" +
          "CLAUDE.md workflow: the main thread plans, integrates, and REVIEWS; " +
          "implementation is delegated to the `implementer` subagent " +
          "(.claude/agents/implementer.md). Cross-CLI implementation handoff " +
          "is abolished (2026-07-12) — the partner AI only does second " +
          "reviews and consultations.\n\n" +
          `Refused edit: ${filePath}\n\n` +
          "Hand the change to the `implementer` subagent with the approved " +
          "plan, allowed write paths, and conventions. Reviews and scope " +
          "verification happen at the integration point, per CLAUDE.md.\n\n" +
          "Deliberate, user-approved override for ONE session: relaunch " +
          `Claude Code with env ${OVERRIDE_ENV}=1.\n`,
      );
      process.exit(2);
    }

    process.exit(0);
  } catch {
    // Fail open: never let a guard bug wedge the tool.
    process.exit(0);
  }
});

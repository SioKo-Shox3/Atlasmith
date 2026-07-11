#!/usr/bin/env node
// SessionStart hook (Atlasmith working agreement): surface the
// model-policy workflow at the top of every session so it is never just buried
// in CLAUDE.md. It also tells the agent that the PreToolUse guard will
// physically block main-thread code edits, so the agent routes implementation
// to the implementer subagent from the start instead of discovering the wall
// mid-task.
//
// Emits the reminder as additionalContext. Fails open (exit 0, no context) on
// any error so it can never wedge session startup.

import process from "node:process";

const context = [
  "Atlasmith workflow policy (ENFORCED by hooks, not just CLAUDE.md):",
  "- The main session is the ORCHESTRATOR: it splits phases, decides, integrates, and reviews — it does not write implementation itself.",
  "- Implementation is delegated to the `implementer` subagent (top model for load-bearing areas; include the no-loop clause: stop after 2 failures of one approach). Do NOT hand implementation to the partner AI's CLI — cross-CLI implementation handoff is abolished (2026-07-12); the partner AI only does second reviews and consultations. The main thread plans, integrates, and reviews; it never types code.",
  "- A PreToolUse guard BLOCKS main-thread Edit/Write of implementation source. Do NOT try to type code directly — hand it to the implementer subagent.",
  "- Every non-trivial change gets a DOUBLE review: top-model Claude first review (plan-reviewer / impl-reviewer, never the author) + a MANDATORY independent Codex second review via DIRECT CLI: `codex exec --sandbox read-only ...` (synchronous, no polling). NEVER via plugin (`codex:rescue`) — Agent-hooking plugins (context-mode) break it. If the CLI call fails, fall back to impl-reviewer + verifier and REPORT the skipped gate.",
  "- Quality roles (planner, plan-reviewer, test-designer, impl-reviewer, verifier) use model:inherit — they follow the main session's model, so keep the main session on the top model available. Research/mechanical work runs cheap (sonnet/haiku aliases). Escalate after 2x refute/rework (thrash ladder).",
  "- Before any non-trivial phase, invoke the `fable-reasoning` skill (recon -> evidence -> decomposition -> stop conditions). Before the first file edit of a phase AND before reporting anything as done, invoke the `phase-gates` skill (request/entry/exit gates). They encode the top-model discipline this workflow assumes.",
  "- Consult triggers are CONCRETE (workflow-core/consult-triggers.md) and ORCHESTRATOR-ONLY: same-signature failure twice, third fix for one symptom, guard block + rewording urge, 2x declared budget, out-of-declaration changes. Window: node ~/.agent-workflow/ask-advisor.mjs <claude|codex> (arg REQUIRED; convention: pick the NON-main AI). The 5-line phase declaration (falsifier + ```scope block) is written by the orchestrator, per phase; check-scope.mjs verifies ONCE against the integrated diff. Subagents never write declarations, never run check-scope, never call ask-advisor.",
  "- Subagent rule (the ONLY discipline delegated agents carry): stay inside assigned paths, run the verification commands and return real output, stop after 2 failures of the same approach and return evidence to the parent.",
  "- Show evidence, not assertions: paste commands and real output.",
  "- Deliberate one-session override (rare, user-approved only): relaunch with env ATLASMITH_ALLOW_DIRECT_EDIT=1.",
].join("\n");

try {
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: context,
      },
    }),
  );
} catch {
  // fail open
}
process.exit(0);

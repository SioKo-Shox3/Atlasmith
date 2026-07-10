#!/usr/bin/env node
// SessionStart hook for Codex-main sessions (Atlasmith working
// agreement): surface the workflow policy at the top of every session so it
// is never just buried in AGENTS.md, and warn that the PreToolUse guard will
// physically block main-thread implementation patches.
//
// Output channel: plain text on stdout. (Claude Code's
// hookSpecificOutput.additionalContext JSON is Claude-specific — verify the
// Codex injection path with `codex exec` before trusting changes here; see
// Docs/agent-guide/codex-delegation.md.)
// Fails open (exit 0, no output) on any error.

const context = [
  "Atlasmith workflow policy (ENFORCED by hooks, not just AGENTS.md):",
  "- The main session is the ORCHESTRATOR: it splits phases, decides, integrates, and reviews — it does not write implementation itself.",
  "- Implementation is delegated to subagents: spawn_agent -> .codex/agents (implementer etc.) with the approved plan, allowed write paths, and conventions. A PreToolUse guard BLOCKS main-thread apply_patch on implementation source — do not try to type code directly.",
  "- Every non-trivial change gets a DOUBLE review: Codex-side first review (impl-reviewer agent, never the author) + a MANDATORY second review by the non-main AI (Claude): `claude -p \"<review brief>\" --model opus --permission-mode plan` (interactive session only — sandboxed codex exec has no network for claude -p; approve out-of-sandbox execution). If unavailable, run a second independent Codex review and REPORT the skipped gate.",
  "- CLAUDE.md and AGENTS.md must stay byte-identical. A Stop hook blocks turn completion while they drift: after editing either, cp over the mirror and diff (must print nothing) before finishing.",
  "- Discipline skills (installed in ~/.codex/skills): invoke $fable-reasoning before any non-trivial phase (recon -> evidence hierarchy -> decomposition -> stop conditions), and $phase-gates before the first file edit of a phase AND before reporting anything as done. Mention them by name — skills do not carry across turns.",
  "- Consult triggers are CONCRETE (workflow-core/consult-triggers.md): same-signature failure twice, third fix for one symptom, guard block + rewording urge, 2x declared budget, out-of-declaration changes. Window: node ~/.agent-workflow/ask-advisor.mjs <claude|codex> (arg REQUIRED; convention: pick the NON-main AI). Write the 5-line phase declaration (with falsifier + ```scope block) before the first edit; check-scope.mjs verifies it.",
  "- Stop conditions: after 2 failures of the same approach, change method or consult the non-main AI — never loop. Escalate after 2x refute/rework.",
  "- Show evidence, not assertions: paste commands and real output.",
  "- Deliberate one-session override (rare, user-approved only): relaunch with env ATLASMITH_ALLOW_DIRECT_EDIT=1.",
].join("\n");

try {
  process.stdout.write(context);
} catch {
  // fail open
}
process.exit(0);

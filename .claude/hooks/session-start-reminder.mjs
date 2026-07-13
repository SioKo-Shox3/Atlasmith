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

// Cadence: 探索期の品質返済(code-gardening+統合一括レビュー)の期限監視。
// <project>/.harness/cadence.json を読み、超過時だけ1行注入する。失敗したら黙る(fail-open)。
import { readFileSync as _read } from "node:fs";
import { execSync as _exec } from "node:child_process";
import { dirname as _dir, join as _join } from "node:path";
import { fileURLToPath as _furl } from "node:url";
function cadenceLine() {
  try {
    const root = _join(_dir(_furl(import.meta.url)), "..", "..");
    const cfg = JSON.parse(_read(_join(root, ".harness", "cadence.json"), "utf8"));
    const days = Math.floor((Date.now() - Date.parse(cfg.last_gardening.date)) / 86400000);
    let commits = null;
    try {
      commits = Number(
        _exec(`git rev-list --count ${cfg.last_gardening.commit}..HEAD`, {
          cwd: root, encoding: "utf8", timeout: 5000, windowsHide: true,
        }).trim(),
      );
    } catch {}
    const maxD = cfg.max_days ?? 14;
    const maxC = cfg.max_commits ?? 40;
    if (days > maxD || (commits !== null && commits > maxC)) {
      return (
        `- CADENCE: quality repayment OVERDUE — last code-gardening ${days}d` +
        (commits !== null ? ` / ${commits} commits` : "") +
        ` ago (limits ${maxD}d/${maxC}). Schedule code-gardening + ONE integrated review at the next natural boundary, distill one representative task into evals (model-evals.md), then update .harness/cadence.json.`
      );
    }
    return null;
  } catch {
    return "- CADENCE: .harness/cadence.json not initialized — create it at the first code-gardening pass ({\"last_gardening\":{\"date\":\"<ISO>\",\"commit\":\"<HEAD>\"},\"max_days\":14,\"max_commits\":40}).";
  }
}

const context = [
  "Atlasmith workflow policy (ENFORCED by hooks, not just CLAUDE.md):",
  "- The main session is the ORCHESTRATOR: it splits phases, decides, integrates, and reviews — it does not write implementation itself.",
  "- DEV-STAGE DIAL (2026-07-13): check the 開発段階 line in CLAUDE.md. In EXPLORATION stage (pre-alpha), the LIGHT path is the DEFAULT for everything outside danger zones; cross-AI review and plan/test-design stages run ONLY for danger-zone work; quality debt is repaid in BATCH (code-gardening + one integrated review) at milestones, not per task. Speed beats per-task QA at this stage — user policy.",
  "- TRIAGE FIRST (cost guard, 2026-07-12): classify the task before starting and declare the path. LIGHT (<=2 files, ~<=50 changed lines, no load-bearing area, no public API/schema/ownership change): go STRAIGHT to one implementer with conventions+verification embedded in the brief, run the gates, commit — the cross-AI second review MAY BE SKIPPED (state 'light path' in the report). STANDARD: full workflow, but the cross-AI second review runs ONCE on the task's integrated diff, not per phase. HEAVY (load-bearing/public API/data formats/concurrency): all gates + heavy-artillery. When the user asks for speed, default to LIGHT and say so — do not gold-plate.",
  "- Implementation is delegated to the `implementer` subagent (top model for load-bearing areas; include the no-loop clause: stop after 2 failures of one approach). Do NOT hand implementation to the partner AI's CLI — cross-CLI implementation handoff is abolished (2026-07-12); the partner AI only does second reviews and consultations. The main thread plans, integrates, and reviews; it never types code.",
  "- A PreToolUse guard BLOCKS main-thread Edit/Write of implementation source. Do NOT try to type code directly — hand it to the implementer subagent.",
  "- STANDARD/HEAVY changes get a DOUBLE review: top-model Claude first review (plan-reviewer / impl-reviewer, never the author) + an independent Codex second review via DIRECT CLI, ONCE per task on the integrated diff: `codex exec --sandbox read-only ...` (synchronous; do NOT wrap in a shell-tool timeout — cut only on failure evidence, never on elapsed time). NEVER via plugin (`codex:rescue`). If the CLI call fails, fall back to impl-reviewer + verifier and REPORT the skipped gate.",
  "- Quality roles (planner, plan-reviewer, test-designer, impl-reviewer, verifier) use model:inherit — they follow the main session's model, so keep the main session on the top model available. Research/mechanical work runs cheap (sonnet/haiku aliases). Escalate after 2x refute/rework (thrash ladder).",
  "- Before any non-trivial phase, invoke the `fable-reasoning` skill (recon -> evidence -> decomposition -> stop conditions). Before the first file edit of a phase AND before reporting anything as done, invoke the `phase-gates` skill (request/entry/exit gates). They encode the top-model discipline this workflow assumes.",
  "- Consult triggers are CONCRETE (workflow-core/consult-triggers.md) and ORCHESTRATOR-ONLY: same-signature failure twice, third fix for one symptom, guard block + rewording urge, 2x declared budget, out-of-declaration changes. Window: node ~/.agent-workflow/ask-advisor.mjs <claude|codex> (arg REQUIRED; convention: pick the NON-main AI). The 5-line phase declaration (falsifier + ```scope block) is written by the orchestrator, per phase; check-scope.mjs verifies ONCE against the integrated diff. Subagents never write declarations, never run check-scope, never call ask-advisor.",
  "- Subagent rule (the ONLY discipline delegated agents carry): stay inside assigned paths, run the verification commands and return real output, stop after 2 failures of the same approach and return evidence to the parent.",
  "- Show evidence, not assertions: paste commands and real output.",
  "- Deliberate one-session override (rare, user-approved only): relaunch with env ATLASMITH_ALLOW_DIRECT_EDIT=1.",
].join("\n");

try {
  const cad = cadenceLine();
  process.stdout.write(
    JSON.stringify({
      hookSpecificOutput: {
        hookEventName: "SessionStart",
        additionalContext: context + (cad ? "\n" + cad : ""),
      },
    }),
  );
} catch {
  // fail open
}
process.exit(0);

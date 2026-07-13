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
//
// Mode signal: CODEX_MODE=ultra (set by ~/.agent-workflow/codex-ultra.ps1)
// switches to the fleet-mode policy text. Editing THIS file does not break
// hook trust — trusted_hash covers hooks.json only. Sessions launched from
// Codex Desktop do not receive the env signal and get the standard text;
// both texts are compatible with autonomous delegation by design.

const ULTRA = (process.env.CODEX_MODE || "").toLowerCase() === "ultra";

// Cadence: 探索期の品質返済(code-gardening+統合一括レビュー)の期限監視。
// <project>/.harness/cadence.json {last_gardening:{date,commit}, max_days, max_commits} を読み、
// 超過時だけ1行注入する。ファイル読取もgitも失敗したら黙る(fail-open)。
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

const ultraContext = [
  "Atlasmith ULTRA fleet mode (CODEX_MODE=ultra) — outcome gates only:",
  "- KICKOFF, once: write the standard 5-line declaration EXTENDED with fleet-wide allowed globs and stop conditions; get ONE user approval for the whole package. No per-phase re-declarations after that.",
  "- CHILDREN carry only 3 rules: stay inside assigned paths; run the verification commands (build/test) and return real output; stop after 2 failures of the same approach and return evidence to the parent. Children never write declarations, never run check-scope, never call ask-advisor.",
  "- EXIT, once, on the INTEGRATED diff: check-scope.mjs against the kickoff globs; minimality + WHY-comment pass/fail items (review-lenses.md); heavy-artillery only on load-bearing phases. Completion report keeps the 2 mandatory sections (hotspots by risk / needs-human-judgment).",
  "- Cross-AI second review: run a short connectivity check first; if the partner AI is unreachable, fall back to a fresh-context Codex review and REPORT the fallback (never silently skip).",
  "- Unchanged: CLAUDE.md/AGENTS.md byte-identical mirror (Stop hook enforces), evidence-not-assertions, main thread does not type implementation source.",
].join("\n");

const context = [
  "Atlasmith workflow policy (ENFORCED by hooks, not just AGENTS.md):",
  "- The main session is the ORCHESTRATOR: it splits phases, decides, integrates, and reviews — it does not write implementation itself.",
  "- DEV-STAGE DIAL (2026-07-13): check the 開発段階 line in AGENTS.md. In EXPLORATION stage (pre-alpha), the LIGHT path is the DEFAULT for everything outside danger zones; cross-AI review and plan/test-design stages run ONLY for danger-zone work; quality debt is repaid in BATCH (code-gardening + one integrated review) at milestones, not per task. Speed beats per-task QA at this stage — user policy.",
  "- TRIAGE FIRST (cost guard, 2026-07-12): classify the task before starting and declare the path. LIGHT (<=2 files, ~<=50 changed lines, no load-bearing area, no public API/schema/ownership change): go STRAIGHT to one implementer with conventions+verification embedded in the brief, run the gates, commit — the cross-AI second review MAY BE SKIPPED (state 'light path' in the report). STANDARD: full workflow, but the cross-AI second review runs ONCE on the task's integrated diff, not per phase. HEAVY (load-bearing/public API/data formats/concurrency): all gates + heavy-artillery. When the user asks for speed, default to LIGHT and say so — do not gold-plate.",
  "- Implementation is delegated to subagents: spawn_agent -> .codex/agents (implementer etc.) with the approved plan, allowed write paths, and conventions. A PreToolUse guard BLOCKS main-thread apply_patch on implementation source — do not try to type code directly.",
  "- STANDARD/HEAVY changes get a DOUBLE review: Codex-side first review (impl-reviewer agent, never the author) + a second review by the non-main AI (Claude), ONCE per task on the integrated diff: `claude -p \"<review brief>\" --model opus --permission-mode plan`. Do NOT wrap the call in a shell-tool timeout — the wrapper cuts only on failure evidence, never on elapsed time. If Claude is unavailable (limit/refused), run a second independent Codex review and REPORT the skipped gate. CONVERGENCE: max 2 review rounds — reviewers classify findings blocking/non-blocking, only blocking requires fixes, round 2 verifies ONLY the fix-diff (new findings accepted only if blocking). There is NO round 3: leftovers are logged as 残課題 and routed to the repayment cycle. Never loop review↔fix.",
  "- CLAUDE.md and AGENTS.md must stay byte-identical. A Stop hook blocks turn completion while they drift: after editing either, cp over the mirror and diff (must print nothing) before finishing.",
  "- Discipline skills (installed in ~/.codex/skills): invoke $fable-reasoning before any non-trivial phase (recon -> evidence hierarchy -> decomposition -> stop conditions), and $phase-gates before the first file edit of a phase AND before reporting anything as done. Mention them by name — skills do not carry across turns.",
  "- Consult triggers are CONCRETE (workflow-core/consult-triggers.md) and ORCHESTRATOR-ONLY: same-signature failure twice, third fix for one symptom, guard block + rewording urge, 2x declared budget, out-of-declaration changes. Window: node ~/.agent-workflow/ask-advisor.mjs <claude|codex> (arg REQUIRED; convention: pick the NON-main AI). The 5-line phase declaration (falsifier + ```scope block) is written by the orchestrator, per phase; check-scope.mjs verifies ONCE against the integrated diff. Subagents never write declarations, never run check-scope, never call ask-advisor.",
  "- Subagent rule (the ONLY discipline delegated agents carry): stay inside assigned paths, run the verification commands and return real output, stop after 2 failures of the same approach and return evidence to the parent.",
  "- Stop conditions: after 2 failures of the same approach, change method or consult the non-main AI — never loop. Escalate after 2x refute/rework.",
  "- Show evidence, not assertions: paste commands and real output.",
  "- Deliberate one-session override (rare, user-approved only): relaunch with env ATLASMITH_ALLOW_DIRECT_EDIT=1.",
].join("\n");

try {
  const cad = cadenceLine();
  process.stdout.write((ULTRA ? ultraContext : context) + (cad ? "\n" + cad : ""));
} catch {
  // fail open
}
process.exit(0);

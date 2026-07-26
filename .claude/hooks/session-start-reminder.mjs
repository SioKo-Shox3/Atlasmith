#!/usr/bin/env node
// SessionStart hook (Atlasmith working agreement): surface the
// model-policy workflow at the top of every session so it is never just buried
// in CLAUDE.md. It also tells the agent that the PreToolUse guard will
// physically block main-thread code edits, so the agent routes implementation
// to the implementer subagent from the start instead of discovering the wall
// mid-task.
//
// MODEL PROFILES (2026-07-26). This hook is the enforcement point for the
// two-layer design in Docs/agent-guide/model-playbook.md: a FIXED CORE that
// every model gets, plus a per-model PROFILE that tunes only the density of
// process instructions. The profile is selected from the `model` field of the
// SessionStart payload (documented input field; the host may omit it).
//   - override for testing / a deliberate run: CLAUDE_MODEL_PROFILE=<name>
//   - model absent or unrecognised -> `default` (FAIL SAFE: an unknown model
//     must never silently receive a tuned-down ruleset)
// The selected profile name is ALWAYS printed, so a live session shows which
// rules it actually got instead of leaving detection unobservable.
//
// Emits the reminder as additionalContext. Fails open (exit 0) on any error so
// it can never wedge session startup — including the case where the host never
// closes stdin, which is why the read below is bounded by a timer.

import process from "node:process";

// Cadence: 探索期の品質返済(code-gardening+統合一括レビュー)の期限監視。
// <project>/.harness/cadence.json を読み、超過時だけ1行注入する。失敗したら黙る(fail-open)。
import { readFileSync as _read } from "node:fs";
import { execFileSync as _execFile } from "node:child_process";
import { dirname as _dir, join as _join } from "node:path";
import { fileURLToPath as _furl } from "node:url";
function cadenceLine() {
  try {
    const root = _join(_dir(_furl(import.meta.url)), "..", "..");
    const cfg = JSON.parse(_read(_join(root, ".harness", "cadence.json"), "utf8"));
    const days = Math.floor((Date.now() - Date.parse(cfg.last_gardening.date)) / 86400000);
    let commits = null;
    try {
      // cadence.json is repo-local (untrusted) input: never hand its values to a
      // shell, and accept only a hex commit id so the value cannot become a git
      // option or range expression. Invalid value => commits stays null and the
      // days-only check still applies.
      const commit = String(cfg.last_gardening.commit || "");
      if (/^[0-9a-f]{7,40}$/i.test(commit)) {
        commits = Number(
          _execFile("git", ["rev-list", "--count", `${commit}..HEAD`], {
            cwd: root, encoding: "utf8", timeout: 5000, windowsHide: true,
          }).trim(),
        );
      }
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

// stdin は「来ないかもしれない」前提で読む。SessionStart はセッション開始を待たせるので、
// end が来なければ打ち切って既定プロファイルで続行する(ハングは絶対に作らない)。
const STDIN_TIMEOUT_MS = 1500;
function readStdin() {
  return new Promise((resolve) => {
    let raw = "";
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(raw);
    };
    const timer = setTimeout(finish, STDIN_TIMEOUT_MS);
    try {
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (c) => { raw += c; });
      process.stdin.on("end", finish);
      process.stdin.on("error", finish);
    } catch {
      finish();
    }
  });
}

// プロファイル選択。判別できないときは default(安全側)。
function pickProfile(payload) {
  const override = String(process.env.CLAUDE_MODEL_PROFILE || "").trim().toLowerCase();
  if (override) return { name: override in PROFILES ? override : "default", why: `env CLAUDE_MODEL_PROFILE=${override}` };
  const model = String((payload && payload.model) || "").toLowerCase();
  if (!model) return { name: "default", why: "host did not report a model" };
  if (/opus-?5/.test(model)) return { name: "opus-5", why: `session model: ${model}` };
  return { name: "default", why: `session model: ${model}` };
}

// ---- 調整層: プロファイルごとに差し替える行 ----------------------------------

const SKILLS_LINE = {
  default:
    "- Before any non-trivial phase, invoke the `fable-reasoning` skill (recon -> evidence -> decomposition -> stop conditions). Before the first file edit of a phase AND before reporting anything as done, invoke the `phase-gates` skill (request/entry/exit gates). They encode the top-model discipline this workflow assumes.",
  "opus-5":
    "- Before any non-trivial phase, invoke the `fable-reasoning` skill (recon -> evidence -> decomposition -> stop conditions) — it is about facts in THIS repo and THIS working tree, which no model can supply from inside itself. Invoke `phase-gates` for its ENTRY gate only (scope and plan before the first edit); SKIP its exit/completion gate — see the MODEL PROFILE block below.",
};

const QUALITY_ROLES_LINE = {
  default:
    "- Quality roles (planner, plan-reviewer, test-designer, impl-reviewer, verifier) use model:inherit — they follow the main session's model, so keep the main session on the top model available. Research/mechanical work runs cheap (sonnet/haiku aliases). Escalate after 2x refute/rework (thrash ladder).",
  "opus-5":
    "- Quality roles (planner, plan-reviewer, test-designer, impl-reviewer) use model:inherit — they follow the main session's model, so keep the main session on the top model available. The `verifier` role is NOT used to re-check your own output on this profile; it remains available only for checking ANOTHER agent's work. Research/mechanical work runs cheap (sonnet/haiku aliases). Escalate after 2x refute/rework (thrash ladder).",
};

// プロファイル固有の追加ブロック。既定は空 = 従来どおりの運転。
const PROFILES = {
  default: [],
  "opus-5": [
    "MODEL PROFILE opus-5 — tuned from Anthropic's official Opus 5 prompting guide (2026-07-26). Where these conflict with a line above, THESE WIN:",
    "- NO ADDED VERIFICATION PASS. Opus 5 verifies its own work without being told to, so bolted-on verification steps burn tokens without improving quality. Skip the phase-gates exit gate, do not spawn a subagent to verify or double-check YOUR OWN output, and do not re-read your own answer 'to be safe'.",
    "- STILL REQUIRED — these are not self-verification and are NOT removable scaffolding: run the build/test gates and paste REAL output (that is a fact about this repo, not a self-check); independent review of ANOTHER agent's work (the guide rates writer-verifier patterns as effective); the 5-line declaration + check-scope on the integrated diff; and every repo-safety guard (branch discipline, push approval, ignored-tracked, CLAUDE/AGENTS mirror).",
    "- DELEGATION CAP. Opus 5 delegates more readily than earlier models, and delegation multiplies cost and time on small tasks. Delegate only for large, genuinely independent, parallelizable tracks — e.g. a wide multi-file investigation. Do not delegate work you can finish yourself in a handful of tool calls. Cap at 3 concurrent subagents unless the user explicitly asks for a fleet.",
    "- SCOPE. Deliver what was asked, at the scope intended. If the request seems mistaken or a better approach exists, say so in one sentence and continue with the task as asked, rather than quietly widening, narrowing, or transforming it.",
    "- LENGTH. Opus 5 writes longer responses AND longer files than earlier models, and lowering effort does not shorten them — length is controlled here, by instruction. Match plans, phase declarations, completion reports, review summaries, and any document written to disk to what the task needs: cover the substance, do not pad with filler sections, redundant summaries, or boilerplate.",
    "- NARRATION. Before the first tool call, say in one sentence what you are about to do. While working, give a brief update only when you find something important or change direction. Lead the final message with the outcome.",
  ],
};

function buildContext(profile) {
  const p = profile.name in PROFILES ? profile.name : "default";
  const lines = [
    "Atlasmith workflow policy (ENFORCED by hooks, not just CLAUDE.md):",
    `- MODEL PROFILE: ${p} (${profile.why}). The fixed core below applies to every model; the profile tunes only how much process instruction is layered on top. Rationale and evidence: Docs/agent-guide/model-playbook.md.`,
    "- The main session is the ORCHESTRATOR: it splits phases, decides, integrates, and reviews — it does not write implementation itself.",
    "- USER-FACING LANGUAGE (2026-07-25): every document the USER is asked to read or approve — kickoff/phase declarations, plans, completion reports, review summaries — is written in JAPANESE (code, identifiers, commands stay English). A decision document the decider can't read is not evidence.",
    "- DEV-STAGE DIAL (2026-07-13): check the 開発段階 line in CLAUDE.md. In EXPLORATION stage (pre-alpha), the LIGHT path is the DEFAULT for everything outside danger zones; cross-AI review and plan/test-design stages run ONLY for danger-zone work; quality debt is repaid in BATCH (code-gardening + one integrated review) at milestones, not per task. Speed beats per-task QA at this stage — user policy.",
    "- TRIAGE FIRST (cost guard, 2026-07-12): classify the task before starting and declare the path. LIGHT (<=2 files, ~<=50 changed lines, no load-bearing area, no public API/schema/ownership change): go STRAIGHT to one implementer with conventions+verification embedded in the brief, run the gates, commit — the cross-AI second review MAY BE SKIPPED (state 'light path' in the report). STANDARD: full workflow, but the cross-AI second review runs ONCE on the task's integrated diff, not per phase. HEAVY (load-bearing/public API/data formats/concurrency): all gates + heavy-artillery. When the user asks for speed, default to LIGHT and say so — do not gold-plate.",
    "- Implementation is delegated to the `implementer` subagent (top model for load-bearing areas; include the no-loop clause: stop after 2 failures of one approach). Do NOT hand implementation to the partner AI's CLI — cross-CLI implementation handoff is abolished (2026-07-12); the partner AI only does second reviews and consultations. The main thread plans, integrates, and reviews; it never types code.",
    "- A PreToolUse guard BLOCKS main-thread Edit/Write of implementation source. Do NOT try to type code directly — hand it to the implementer subagent.",
    "- STANDARD/HEAVY changes get a DOUBLE review: top-model Claude first review (plan-reviewer / impl-reviewer, never the author) + an independent Codex second review via DIRECT CLI, ONCE per task on the integrated diff: `codex exec --sandbox read-only ...` (synchronous; do NOT wrap in a shell-tool timeout — cut only on failure evidence, never on elapsed time). NEVER via plugin (`codex:rescue`). If the CLI call fails, fall back to impl-reviewer + verifier and REPORT the skipped gate. CONVERGENCE: max 2 review rounds — reviewers classify findings blocking/non-blocking, only blocking requires fixes, round 2 verifies ONLY the fix-diff (new findings accepted only if blocking). There is NO round 3: leftovers are logged as 残課題 and routed to the repayment cycle. Never loop review↔fix.",
    QUALITY_ROLES_LINE[p] || QUALITY_ROLES_LINE.default,
    SKILLS_LINE[p] || SKILLS_LINE.default,
    "- Consult triggers are CONCRETE (workflow-core/consult-triggers.md) and ORCHESTRATOR-ONLY: same-signature failure twice, third fix for one symptom, guard block + rewording urge, 2x declared budget, out-of-declaration changes. Window: node ~/.agent-workflow/ask-advisor.mjs <claude|codex> (arg REQUIRED; convention: pick the NON-main AI). The 5-line phase declaration (falsifier + ```scope block) is written by the orchestrator, per phase; check-scope.mjs verifies ONCE against the integrated diff. Subagents never write declarations, never run check-scope, never call ask-advisor.",
    "- Subagent rule (the ONLY discipline delegated agents carry): stay inside assigned paths, run the verification commands and return real output, stop after 2 failures of the same approach and return evidence to the parent.",
    "- Show evidence, not assertions: paste commands and real output.",
    "- Deliberate one-session override (rare, user-approved only): relaunch with env ATLASMITH_ALLOW_DIRECT_EDIT=1.",
  ];
  const block = PROFILES[p] || [];
  return lines.join("\n") + (block.length ? "\n\n" + block.join("\n") : "");
}

let out = "";
try {
  let payload = {};
  try {
    payload = JSON.parse((await readStdin()) || "{}") || {};
  } catch {
    payload = {};
  }
  const cad = cadenceLine();
  out = JSON.stringify({
    hookSpecificOutput: {
      hookEventName: "SessionStart",
      additionalContext: buildContext(pickProfile(payload)) + (cad ? "\n" + cad : ""),
    },
  });
} catch {
  out = ""; // fail open: 何も注入しない方が、壊れた注入より安全
}
try {
  if (out) process.stdout.write(out);
} catch {
  /* fail open */
}
process.exit(0);

#!/usr/bin/env node
// trust-hooks.mjs — compute Codex hook trust entries for a hooks.json.
//
// Codex silently SKIPS any hook whose trusted_hash in ~/.codex/config.toml
// [hooks.state] does not match the hook definition — including after every
// legitimate edit to hooks.json. This helper recomputes the entries so trust
// can be re-established without the interactive TUI (/hooks).
//
// Usage:
//   node trust-hooks.mjs <path-to-hooks.json>
// Prints ready-to-paste TOML for ~/.codex/config.toml. Review before pasting
// — trusting a hook means Codex will execute it without further review.
//
// Hash algorithm (reverse-engineered from codex-rs tag rust-v0.142.5 —
// hooks/src/engine/discovery.rs command_hook_hash + config/src/
// fingerprint.rs version_for_toml — and verified against two live
// trusted_hash entries, 2026-07-07):
//   identity = { event_name: <snake_case event>,
//                matcher: <string, omitted when absent>,
//                hooks: [{ type: "command",
//                          command: <commandWindows ?? command, pre-substitution>,
//                          timeout: <timeout ?? 600>,
//                          async: false,
//                          statusMessage: <omitted when absent> }] }
//   hash = "sha256:" + sha256(compact JSON of identity with all object keys
//                             recursively sorted)
//   state key = "<absolute hooks.json path>:<snake_case event>:<group idx>:<handler idx>"

import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";

const EVENT_SNAKE = {
  SessionStart: "session_start",
  UserPromptSubmit: "user_prompt_submit",
  PreToolUse: "pre_tool_use",
  PostToolUse: "post_tool_use",
  PermissionRequest: "permission_request",
  PreCompact: "pre_compact",
  PostCompact: "post_compact",
  SubagentStart: "subagent_start",
  SubagentStop: "subagent_stop",
  Stop: "stop",
};

function sortKeysDeep(value) {
  if (Array.isArray(value)) return value.map(sortKeysDeep);
  if (value && typeof value === "object") {
    const out = {};
    for (const key of Object.keys(value).sort()) {
      out[key] = sortKeysDeep(value[key]);
    }
    return out;
  }
  return value;
}

function handlerIdentity(handler) {
  const isWindows = process.platform === "win32";
  const command =
    (isWindows && (handler.commandWindows ?? handler.command_windows)) ||
    handler.command;
  const identity = {
    type: "command",
    command,
    timeout: handler.timeout ?? 600,
    async: false,
  };
  if (handler.statusMessage !== undefined || handler.status_message !== undefined) {
    identity.statusMessage = handler.statusMessage ?? handler.status_message;
  }
  return identity;
}

function main() {
  const target = process.argv[2];
  if (!target) {
    console.error("usage: node trust-hooks.mjs <path-to-hooks.json>");
    process.exit(1);
  }
  const absPath = path.resolve(target);
  const doc = JSON.parse(fs.readFileSync(absPath, "utf8"));
  const events = (doc && doc.hooks) || {};
  const lines = [];
  for (const [eventName, groups] of Object.entries(events)) {
    const snake = EVENT_SNAKE[eventName];
    if (!snake) {
      console.error(`warning: unknown event "${eventName}" skipped`);
      continue;
    }
    groups.forEach((group, groupIdx) => {
      (group.hooks || []).forEach((handler, handlerIdx) => {
        const identity = { event_name: snake, hooks: [handlerIdentity(handler)] };
        if (group.matcher !== undefined) identity.matcher = group.matcher;
        const json = JSON.stringify(sortKeysDeep(identity));
        const hash = createHash("sha256").update(json, "utf8").digest("hex");
        const key = `${absPath}:${snake}:${groupIdx}:${handlerIdx}`;
        lines.push(`[hooks.state.'${key}']`);
        lines.push(`trusted_hash = "sha256:${hash}"`);
        lines.push("");
      });
    });
  }
  lines.push(
    "# NOTE: trust covers ONLY the hooks.json definition above — the referenced",
    "# .codex/hooks/*.mjs script contents are NOT hashed. Keep hook scripts under",
    "# git so tampering shows up in diffs, and review hook changes like code.",
    "",
  );
  process.stdout.write(lines.join("\n"));
}

main();

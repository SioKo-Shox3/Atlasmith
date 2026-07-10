#!/usr/bin/env node
// PostToolUse guard (Atlasmith working agreement): CLAUDE.md and
// AGENTS.md must stay byte-identical (Claude reads one, Codex reads the
// other). Editing one without mirroring the other has repeatedly been left
// to memory — this hook makes the drift visible the moment it happens.
//
// Fires after Edit/Write/MultiEdit. If the edited file is CLAUDE.md or
// AGENTS.md and its sibling now differs, exit 2 so the mismatch message is
// fed back to the model (non-blocking — the edit already happened; the model
// is told to mirror before finishing). Fails OPEN on any internal error.

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
    const filePath = (data.tool_input && data.tool_input.file_path) || "";
    const base = path.basename(filePath);
    if (base !== "CLAUDE.md" && base !== "AGENTS.md") {
      process.exit(0);
    }
    const dir = path.dirname(filePath);
    const sibling = path.join(
      dir,
      base === "CLAUDE.md" ? "AGENTS.md" : "CLAUDE.md",
    );
    if (!fs.existsSync(sibling)) {
      // No mirror in this directory (e.g. docs example) — nothing to check.
      process.exit(0);
    }
    const a = fs.readFileSync(filePath);
    const b = fs.readFileSync(sibling);
    if (!a.equals(b)) {
      process.stderr.write(
        `MIRROR DRIFT: ${base} was edited but ${path.basename(sibling)} now ` +
          "differs. CLAUDE.md and AGENTS.md must stay byte-identical " +
          "(Claude reads one, Codex reads the other).\n" +
          `Before finishing this task, copy the edited file over its mirror ` +
          `and re-verify:  cp "${filePath}" "${sibling}"  then  ` +
          `diff "${filePath}" "${sibling}"\n`,
      );
      process.exit(2);
    }
    process.exit(0);
  } catch {
    // Fail open: never let a guard bug wedge the session.
    process.exit(0);
  }
});

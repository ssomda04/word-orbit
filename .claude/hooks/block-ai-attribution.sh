#!/bin/sh
# Claude Code PreToolUse (Bash) guard for the Contextle repo.
#
# Denies any Bash command that would introduce Claude/AI attribution into a git
# commit or PR (co-author trailer, "Generated with Claude Code", an @anthropic.com
# author, or a Claude/Anthropic git identity). This is the harness-level tripwire;
# the git hooks in .githooks/ are the tool-agnostic backstop.
#
# Reads the hook payload JSON from stdin and inspects it as raw text (no jq
# dependency): the forbidden signatures appear literally inside .tool_input.command.

input="$(cat)"

if printf '%s' "$input" | grep -qiE 'co-authored-by:[^"]*(claude|anthropic)|generated with claude|🤖[[:space:]]*generated|noreply@anthropic\.com|--author=[^"]*(claude|anthropic)|user\.(name|email)[^"]*(claude|anthropic)'; then
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"AI attribution is forbidden in this repo (AGENTS.md). Remove any \"Co-Authored-By: Claude ...\", \"Generated with Claude Code\", noreply@anthropic.com, or Claude/Anthropic git author/identity before committing."}}'
  exit 0
fi

exit 0

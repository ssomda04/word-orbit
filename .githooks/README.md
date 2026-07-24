# .githooks

Repo-managed Git hooks that **strongly enforce** the "no Claude/AI attribution"
rule (see [../AGENTS.md](../AGENTS.md)). Tracked in Git so every contributor
shares them.

| Hook         | Blocks                                                                 |
| ------------ | --------------------------------------------------------------------- |
| `commit-msg` | Commit messages containing `Co-Authored-By: Claude …`, "Generated with Claude Code", `noreply@anthropic.com`, etc. |
| `pre-commit` | Commits whose author/committer is Claude or an `@anthropic.com` email. |

## Activation (one-time, per clone)

Git does not use a tracked hooks directory automatically. Each clone must run:

```bash
git config core.hooksPath .githooks
```

On macOS/Linux, ensure they're executable (Git for Windows runs them as-is):

```bash
chmod +x .githooks/*
```

Verify:

```bash
git config --get core.hooksPath      # -> .githooks
```

## Notes

- These are a **backstop** for any tool/contributor. Claude Code additionally
  enforces the rule natively (attribution disabled) and via a PreToolUse hook in
  [`../.claude/settings.json`](../.claude/settings.json).
- `git commit --no-verify` bypasses these hooks — intended only for the rare case
  of a human genuinely named "Claude".

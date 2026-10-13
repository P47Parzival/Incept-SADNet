# Improvement 6 — W&B Credential Hygiene

## Checklist item

Addresses the first pre-training checklist item: revoke any W&B API key that was exposed in repository content.

## Repository change made

- The local working copy removes the credential-bearing HTML comment from `SETUP_AND_RUN.md`.
- The credential value is intentionally not reproduced in this note, command output, or documentation.
- A working-tree scan no longer finds a W&B token-shaped value in current project files.
- The removal is intentionally not included in the new branch commit: a public GitHub deletion diff could display the removed credential. The branch inherits the historical file from its parent commit. Coordinate repository-history cleanup with the owner after revocation; do not assume this branch removes historical exposure.

## Required account action

Removing a credential from the current file does **not** revoke it. The user confirmed that the exposed key has been revoked in their W&B account. Create a replacement only if needed and configure it outside source control, for example for the current PowerShell session:

```powershell
$env:WANDB_API_KEY = "<new key>"
```

Do not put the replacement key in a source file, notebook, Markdown document, shell history, or commit.

## Git history warning

A token-shaped value was also found in reachable repository history. Removing the current file line does not remove historical copies. The credential must be revoked regardless of whether history is later rewritten.

History rewriting and force-pushing can disrupt collaborators and other branches. This change does not rewrite Git history. Coordinate a repository-wide history cleanup with the repository owner/collaborators after revocation; ensure all relevant branches and clones are addressed before considering the exposure fully removed.

## Verification and status

- Current working-copy line removed; the current commit still has the historical copy until coordinated cleanup.
- User confirmed that the exposed key has been revoked.
- Historical copies remain until an explicitly coordinated history rewrite.

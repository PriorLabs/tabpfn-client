# Changelog Fragments

This directory holds changelog "fragments" — small files describing the
user-visible change in each PR. At release time the `release-create-pr.yml`
workflow runs `towncrier build`, which assembles the fragments into
`CHANGELOG.md` and deletes them from here.

## How to add an entry

Create a file named `<PR_NUMBER>.<category>.md` containing one short,
user-facing sentence.

```bash
# Recommended — validates the category
uvx towncrier create 361.added.md --content "Support Parquet payloads in the hosted client"

# Or write the file directly
echo "Support Parquet payloads in the hosted client" > changelog/361.added.md
```

Write for someone reading release notes, not for a reviewer reading the diff:
say what changed for the user, not which functions moved.

## Categories

| Filename suffix | Section | When to use |
|---|---|---|
| `<PR>.breaking.md` | Breaking Changes | Removed or renamed public API, changed default behaviour |
| `<PR>.added.md` | Added | New features |
| `<PR>.changed.md` | Changed | Changes to existing behaviour |
| `<PR>.fixed.md` | Fixed | Bug fixes |
| `<PR>.deprecated.md` | Deprecated | Features scheduled for removal |

A PR that needs no entry — a version bump, a CI-only change — can carry the
`no changelog needed` label instead, which satisfies `check-changelog.yml`.

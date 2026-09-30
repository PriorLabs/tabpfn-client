# CLAUDE.md

## Docstrings

Every public module, class, function and method in `src/` needs a Google-style
docstring (`Args:`, `Returns:`, `Raises:` sections; no NumPy-style
`Parameters` / `----------` blocks). Document constructor arguments in the
`__init__` docstring. Types live in the signature's annotations, never in the
docstring: write `name: description`, not `name (type): description`, and
annotate the parameter if it isn't yet. Follow the style of the
[TabPFN](https://github.com/PriorLabs/TabPFN) repo, which exposes largely the
same API.

Ruff enforces this via the `D` rules with `convention = "google"` in `ruff.toml`
(tests and scripts are exempt). Check before pushing:

```bash
uv run ruff check . && uv run ruff format --check .
```

CI runs the same check through `trunk check`.

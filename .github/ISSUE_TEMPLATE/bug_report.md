---
name: Bug report
about: Report a defect in YAF (solver result wrong, demo broken, test failing, ...)
title: "[bug] "
labels: bug
---

## Summary

<!-- One or two sentences. What did you do, what happened, what did you expect? -->

## Reproduction

<!--
Smallest reproducer you can give. If the bug only triggers when running
a specific demo, the exact command line is usually enough. If it's an
adapter / API call, paste the minimum Python snippet.
-->

```bash
# command(s) you ran
```

```python
# or: minimum python snippet
```

## What I expected

<!-- "Half-wave dipole at 300 MHz, expect R ≈ 73 Ω", "/health returns 200", etc. -->

## What I observed

<!-- Exact output, including stack trace if any. Use a code block. -->

```
<paste output here>
```

## Environment

- YAF commit (short SHA): `git rev-parse --short HEAD` → `...`
- Python version: `python --version` →
- OS / arch:
- Backend availability (if relevant):
  - `necpp` installed? yes / no — `python -c "import necpp; print(necpp.__file__)"`
  - `openems` installed? yes / no
- Anything non-default in your `.env` / dependency versions:

## Severity / impact

<!-- "Wrong solver output", "Demo crashes", "Test fails intermittently",
"Documentation contradicts code". Mark the one closest to your case. -->

## Additional context

<!--
Have you already cross-checked HONEST_STATUS.md to see if this is a
known-limitation rather than a regression? If yes, please link the line.
-->

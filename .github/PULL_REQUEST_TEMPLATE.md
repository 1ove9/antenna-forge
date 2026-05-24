<!--
Thanks for sending a PR! Please fill the sections below; the first two
are required, the rest are conditionally required.
-->

## What changed and why

<!--
One or two paragraphs. What did this PR do, and what motivates it? Link
any related issue (#NN). If the PR closes an issue, say "closes #NN".
-->

## Acceptance commands rerun

<!--
Required. Tick the boxes that apply and paste a short summary of the
output (last 3-5 lines is plenty).
-->

- [ ] `pytest tests/ -x -q` → all passed
- [ ] `mypy yaf_core yaf_ai yaf_solvers --strict` → 0 issues
- [ ] `python3 scripts/verify_dipole.py` (if you touched `yaf_solvers/nec2_adapter/`)
- [ ] `python3 scripts/case_yagi.py` (if you touched the Yagi case or any solver in the loop)
- [ ] `docker compose up -d && curl -fsS http://localhost:8000/health` (if you touched `yaf_api/`)

```
<paste last few lines of relevant output here>
```

## Solver-adapter / physics changes

<!--
Required IF this PR touches yaf_solvers/, yaf_core/domain/simulation.py,
or anything that influences a physics result. Skip otherwise.

For physics changes, include the truth-check numbers (R, gain, F/B,
S11) before and after, against a textbook value. The reviewer will
expect at least one ±tolerance assertion in the test suite.
-->

| Quantity | Before this PR | After this PR | Textbook target |
|---|---|---|---|
|  |  |  |  |

## Backwards-incompatible changes

<!--
List any API / file-format / behaviour changes that break existing
callers, even if you think nobody is calling that API yet. If none,
write "none".
-->

## Known regressions or limitations

<!--
Anything you noticed during development that this PR does NOT fix and
that would be worth a follow-up issue.
-->

## HONEST_STATUS.md update

- [ ] This PR does not affect HONEST_STATUS.md.
- [ ] This PR includes an HONEST_STATUS.md update that reflects the
      change in module reality (🟢 / 🟡 / 🔴 tier shift, new
      pending-legal-review item, etc.).

## License / NOTICE

- [ ] This PR adds no new third-party dependencies.
- [ ] This PR adds a third-party dependency and `NOTICE` has been
      updated with its license and combined-work caveats.

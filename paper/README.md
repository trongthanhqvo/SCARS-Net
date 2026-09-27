# Result reporting helpers

This directory contains an empty result template and JSON validation/LaTeX export
utilities used by the contract tests. It does not contain the complete manuscript.

The experiment output is defined by `src/scars/results/` and
`results/schema/results_schema.json`. Validators preserve the distinction between
synthetic, exploratory, and confirmatory evidence; missing measurements must not
be replaced with invented values. See [Protocol and scope](../docs/PROTOCOL.md).

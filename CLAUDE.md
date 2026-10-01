# CLAUDE.md - Guidelines for VisionProf

Please refer to `AGENTS.md` and the `agent-rules/` folder for comprehensive rules.

## Core Rules Summary
- Max 300 lines of code per file (ideal <= 250). Single Responsibility Principle.
- 3-Tier Canonical Modular ownership:
  - TV1: `src/layer/` (events, timer, memory, flops, overhead, hw_specs, plugin)
  - TV2: `src/model/`, `src/zoo/`, `configs/`, `scripts/` (benchmark, resources, env, metrics, 8 adapters)
  - TV3: `src/phase/`, `src/analyzer/`, `src/dashboard/` (timer, dataloader, 19 rules v2.0, A/B engine, tabs)
  - Shared: `src/core/` (interfaces, types, constants, paths)
- Git: Never push to `main`. Use `<type>/<scope>-<short-description>`. Conventional Commits.
- Scientific rigor: Deferred CUDA Events for GPU (no CPU clock), warmup >= 10-20, P50/P90/P95/P99 latency, no NVML in hooks.
- Verification: Always run `python3 scripts/verify_rules.py` before completing a task.

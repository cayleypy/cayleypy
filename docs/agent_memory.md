# Agent Memory

## 2026-04-18

- task_id=beam_search_multigpu_torchrun_rewrite
- prompt_summary=Preserve existing single-GPU beam search logic; add reliable torchrun multi-GPU beam search for 2-8 ranks and scalable owner-partitioned routing for large rank counts.
- technical_constraints=No local early return inside distributed iteration; all ranks must execute collectives in identical order; owner-partitioned default; all-gather strategy retained as explicit diagnostic strategy; pre-k budget must use destination owner rank.
- changed_files=cayleypy/algo/beam_search_multigpu.py,cayleypy/algo/beam_search.py,cayleypy/algo/__init__.py,cayleypy/__init__.py,docs/api.rst,cayleypy/algo/beam_search_multigpu_test.py,docs/agent_memory.md

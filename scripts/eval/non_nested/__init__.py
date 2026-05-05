"""Legacy non-nested utility modules (post-cascade-redesign).

The ``non_nested`` subcommand has been removed in favor of ``cascade``.
The ``metrics_non_nested.py`` module is retained as a utility surface
because :mod:`scripts.eval.completeness` and other reductions still
import its helpers (e.g. ``schema_conformance``). New code should
import the cascade reductions in :mod:`scripts.eval.cascade.reductions`
or the canonical stat primitives in
:mod:`digital_registrar_research.benchmarks.eval.stats`.
"""

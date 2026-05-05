"""Legacy nested-list utility modules (post-cascade-redesign).

The ``nested`` subcommand has been removed in favor of ``cascade``.
The ``biomarkers.py`` per-case scorer is retained because the cascade
imports it through :mod:`scripts.eval.cascade.biomarkers`. New code
should depend on the cascade surface, not import directly here.
"""

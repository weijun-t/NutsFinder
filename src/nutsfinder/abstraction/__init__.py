"""Abstraction of heads-up no-limit hold'em for CFR (docs/phase6_research.md section 3).

- :mod:`.config`: the spec (bet sizes, raise caps, bucket counts) as one frozen dataclass.
- :mod:`.strength`: vectorised 7-card strength and exact river hand strength.
- :mod:`.cards`: card buckets (169 preflop classes; exact E[HS] percentile
  buckets per postflop street, suit-canonical caches).
- :mod:`.tree`: the abstract betting tree, built once through ``engine.betting``.
- :mod:`.holdem_game`: :class:`AbstractHoldem`, the tree + real cards as a
  ``gametheory.Game`` for :class:`~nutsfinder.gametheory.mccfr.MCCFRSolver`.
- :mod:`.translation`: pseudo-harmonic mapping of off-tree bet sizes (for bots).

Submodules are imported explicitly (``from nutsfinder.abstraction.tree import
build_tree``); this file imports nothing, so loading one part never pulls in
the numpy tables of another.
"""

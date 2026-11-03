"""Honest FID re-evaluation for models whose ``metrics.csv`` FID values were never measured.

``tools/recalibrate_fid.py`` (removed in v16.9.67) rewrote the FID and ``Quality_Score`` columns
of four models (``codeformer``, ``ffanet_indoor``, ``mprnet_deraining``, ``mirnet_lowlight``)
with formulas. This package records those rows as ``unverified`` and measures real FID from a
checkpoint with the training validation FID path (``feature=2048``, chunks of at most 2).

Run ``python -m training.fid_reeval --help``.

Results go to an append-only ledger, ``LemGendaryModels/<model>/fid_provenance.csv``, next to
``metrics.csv``. The tool never writes ``metrics.csv`` (its 28-column schema guard in
``training/telemetry.py`` archives any file with extra columns, and the lifecycle manager reads it
for vault baselines), never writes checkpoints and never touches governor or rung state, so the
Monotonic Ratchet (``_commit_rung``) is out of its reach.
"""

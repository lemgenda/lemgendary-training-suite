"""Run settings across a resume (R4-C5): frozen for the run, switched at epoch boundaries.

* :mod:`~training.run_settings.record`: the checkpoint record, its hash and diffs;
* :mod:`~training.run_settings.capture`: the frozen and epoch settings of a run;
* :mod:`~training.run_settings.policy`: the frozen guard and the pending epoch switch;
* :mod:`~training.run_settings.pinning`: settings written back into config and CLI args;
* :mod:`~training.run_settings.ledger`: the ``settings_changes.jsonl`` log.

The epoch loop applies a due switch through :mod:`training.training.settings_switch`.
The package imports no submodule, and none of them imports torch.
"""

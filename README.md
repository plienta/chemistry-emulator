# chemistry-emulator

Neural emulators (MLP, DeepONet) that replace stiff chemical-kinetics ODE solvers.
Current benchmark: ROBER (Robertson). Long-term target: the per-cell chemistry solve in MEGATRON.

## Layout

```
src/chemistry_emulator/
  systems/rober.py   ODE, reference solver, data generation (direct + one-step)
  models.py          MLP, DeepONet
  training.py        prepare (split by trajectory + normalise), train, save/load
  evaluate.py        direct / one-step metrics, rollout vs solver
  plotting.py        data sketch, rollout error plot
notebooks/
  01_rober_direct.ipynb   (state, t) -> y(t)
  02_rober_step.ipynb     (state, dt) -> next state, rollout
tests/                    pytest sanity checks
```

New system (e.g. more species): add `systems/<name>.py` with the same functions as `rober.py`,
change `n_in` / `n_out` of the models. Everything else is reused.

## Workflow with Colab + GitHub

1. Edit code locally or on GitHub, `git push`.
2. In Colab, open a notebook from `notebooks/` (File > Open > GitHub). The first cell clones
   (or `git pull`s) the repo, installs it with `pip install -e .`, and enables autoreload.
3. Edit a `.py` file in the Colab file browser, re-run the cell that uses it: autoreload picks it up.
   To keep the fix, copy it back to your repo (or push from Colab with a token, see the notebook).
4. Data and trained models are NOT in git. Save them to Google Drive from the notebook.

Local: `pip install -e ".[jax,dev]"` then `python -m pytest -q`.

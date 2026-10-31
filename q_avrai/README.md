# Q_Avrai

A geopolitical map and history generator, in which a network of qubits decides how nations act. This is a new version of the project from "A quantum procedure for map generation" (J. R. Wootton, IEEE Conference on Games 2020, [arXiv:2005.10327](https://arxiv.org/abs/2005.10327)), rebuilt on the current QuantumGraph.

The game rules are those of the 2020 code (on the `CoG` branch of qiskit-community/QuantumGraph), kept exactly, quirks included. One thing changes. When a city changes hands between nations whose qubits are coupled, the 2020 code rotated both to ⟨X⟩=1 and applied a CZ, to make ⟨XZ⟩=⟨ZX⟩=1. The paper explains this was a workaround, because a method that targets ⟨XZ⟩=⟨ZX⟩=1 directly while otherwise keeping the state was not yet reliable. That method is now `set_relationship`, which is used here with `--war motte`. `--war cz` gives the 2020 behaviour.

## Files

- `game.py`: the game, ported from the 2020 `main.py`. Runs headless and saves each year.
- `layout.py`: initial city positions and the opponent bicolouring, from the 2020 `islands.py`.
- `experiment.py`: the paper's test, standard nations against static opponents.
- `analysis.py`: the paper's results figure, for the 2020 data and the new runs.
- `war.py`: whether a nation that loses a city turns on the nation that took it.
- `render.py`: draws the saved maps.
- `poster.py`, `names.py`: a poster version of a map, with names from the 2020 name generator.
- `replay.py`: replays a logged run through the map rules, to check them against 2020.
- `test_reproduction.py`, `check.py`, `reproduce.sh`: the guards that keep results paper-like (below).
- `showcase/`: two 53-nation, 30-year runs on the Rochester coupling map, with posters.
- `data_2020/`: the logs of the runs used for the 2020 paper.
- `PLAN.md`: the plan, including the list of places where the 2020 code and paper differ.

## Rules and status

- **Default: `--rules final`.** This is the final 2020 code, with its own single-qubit rotation rule, and it reproduces the paper's emulated results. It keeps the 2020 quirks, including the attack update that uses lost rather than gained area.
- **`--rules feb2020` and `--rules final_gain`** are earlier and fixed variants, kept for comparison.
- **The war step uses `set_relationship` with fraction 1** (`--war motte`, the default). It gives a stronger war response than the 2020 CZ method, and the standard nations keep their advantage over static opponents.
- `PLAN.md` has the plan and the list of places where the 2020 code and paper differ.
- `runs_setbloch/` holds earlier runs made with the current `set_bloch` rotation rule. They differ from 2020 and are not used for results.

## Skirmishes (an optional variant, not in the 2020 game)

In the 2020 game, nations become entangled only when a city changes hands. That is rare, so the circuits stay easy to simulate classically: a chunkiness of about 2 effective qubits, scored with github.com/moth-quantum/chunkiness. Skirmishes add a small step towards the war state, `set_relationship({'XZ':1,'ZX':1}, fraction=f)`, for every pair of nations between which territory changed hands that round. To run them:

    python experiment.py --civs 7 9 11 --runs 20 --war motte --skirmish scaled --spare_opponents

- The fraction is f = min(1, 0.5 × area moved / πr²), the same scaling the 2020 game uses for its defend and attack steps. `--skirmish_fraction` changes the 0.5, and `--skirmish const` uses a fixed fraction instead.
- `--spare_opponents` skips skirmishes that involve a frozen opponent. A two-qubit step changes both nations' qubits, so without it the static opponents in the standard-vs-opponent test start to adapt.
- In standard games (`--opponent_type none`), every pair skirmishes. Chunkiness then reaches n−1, the most possible at these sizes.

The same variant, with the same options, is in the QDrive version of Q_Avrai (the `q_avrai` branch of moth-quantum/QDrive). Skirmishes are off by default, so the defaults still reproduce the paper.

## Keeping the results paper-like

The defaults reproduce the paper's emulated results: `--rules final`, the 2020 rotation rule (`single='2020'` in `game.py`), all-to-all coupling, 8192 shots, a 256x256 map, r=5, 20 years, and static opponents alternating between the two halves of the Rochester bicolouring. Changing any of these breaks the comparison with the 2020 data.

- `python -m pytest test_reproduction.py` (about 5 s) checks three things:
  - the map rules replay 2020 logs exactly;
  - the 2020 rotation rule equals `set_bloch` for full steps;
  - every standard nation explores in year 0.
- `python check.py` compares the runs in `runs/` with `data_2020/`, and needs every criterion to pass. Its end-of-run criteria catch the wrong rules version (`feb2020`). Its year-0 criterion catches the wrong rotation rule, which end-of-run numbers alone do not.
- `./reproduce.sh` regenerates everything and runs both. Versions are pinned in `requirements.txt`.

## Running

From this folder, with QuantumGraph installed (or `PYTHONPATH=..`):

    python experiment.py --civs 7 9 11 --runs 20 --war motte
    python experiment.py --civs 7 9 11 --runs 20 --war cz
    python analysis.py --sources 2020 final_cz final_motte
    python war.py --sources 2020 final_cz final_motte
    python render.py runs/11_final_motte_static_0 --years 4 9 14 19

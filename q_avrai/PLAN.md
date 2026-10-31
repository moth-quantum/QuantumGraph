# Q_Avrai 2026: plan

A new version of Q_Avrai, the geopolitical map generator from "A quantum procedure for map generation" (J. R. Wootton, IEEE CoG 2020, arXiv:2005.10327). It is built on the current QuantumGraph, on the `q_avrai` branch.

## Sources

- Paper: arXiv:2005.10327 (source tarball in Downloads).
- Blog: "Introducing: A Quantum Procedure for Map Generation", Qiskit Medium, 24 Aug 2020.
- Original code: `CoG` branch of github.com/qiskit-community/QuantumGraph.
  - `main.py`: the game loop and AI, using pygame.
  - `islands.py`: the Rochester city layout (`get_points`), plus quantum terrain code that the paper does not use.
  - `QuantumGraph/QuantumGraph.py`: the 2020 QuantumGraph.
  - `analysis.ipynb`: makes the paper's figures.
  - `maps/`: 2020 data from 76 runs (7, 9 and 11 nations on the simulator, 28 on Cambridge, 53 on Rochester). The run data is in each `data.txt`.

## What the 2020 version does

- Each nation is a qubit. ⟨X⟩ is defence, ⟨Z⟩ is attack and ⟨Y⟩ is exploration.
- Each turn, a nation scores E(P) = ⟨P_j⟩ + Σ_Q ⟨P_j Q_k⟩ for each neighbour k on the map. It takes the tactic and neighbour with the largest score, and places a city to match.
- After the turn:
  - A nation that loses a city is set to ⟨X⟩=1. If the winner is coupled to it, the winner is also set to ⟨X⟩=1, then a CZ is applied, giving ⟨XZ⟩=⟨ZX⟩=1.
  - Otherwise, each nation is rotated towards exploration (fraction 1/4), defence or attack (fraction based on area). Which one depends on whether frontier, losses or gains are largest.
- The test: "standard" nations against "static" opponents, which get no single-qubit updates. The two sides are chosen by bicolouring the coupling map. The measure is mean area over 20 rounds.

The paper says the two-qubit step ought to set ⟨XZ⟩=⟨ZX⟩=1 while otherwise keeping the state, but in 2020 tomography noise made that unworkable. So it uses the "rotate both to X, then CZ" workaround. Today's `set_relationship` is exactly the deferred method. That is the main scientific point of a new version.

## Code vs paper differences found

The new version needs a decision on each of these. My suggested default is in brackets.

1. **Influence.** The paper gives 1+min(1,1/d) inside r and 1/d out to 2r. The code uses 1+1/max(1,d)² and 1/d². [Follow the code, since the paper's results came from it. Make it a parameter.]
2. **Attack fraction.** When gains dominate, the code rotates towards attack using `loss/(πr²)` rather than `gain/(πr²)`. Losses are then the smaller number, so attack updates are tiny or zero. The paper says gained area. [Treat it as a bug: fix it, with a flag to reproduce the 2020 behaviour.]
3. **Attack placement.** The paper says the point where the neighbour's influence is least, with the attacker's own influence capped at 1+r/2. The code picks the point where the neighbour's influence is greatest, subject to that influence being at most `get_influence(r/2)`. [Follow the code. Correct the description.]
4. **Opponents and transfers.** The paper says two-qubit operations apply to opponents as normal. In the code, the `if civ not in opponent` check wraps the transfer update too, so a city lost by an opponent triggers nothing. [Follow the paper, with a flag.]
5. **Choosing the update.** The paper says "whichever is greatest" of frontier, loss and gain. The code compares frontier with loss+gain. [Follow the code. It is minor.]
6. **Random opponents.** In the code, `opponent=='random'` compares a list with a string, so random opponents never happen. Not used in the paper. [Drop it, or fix it as an extra baseline.]
7. **Coupling and tomography.** The 2020 simulator runs used all-to-all coupling and tomography of every pair. The hardware runs used the device coupling map, but still did tomography of every pair. Today's QuantumGraph only does tomography on coupling-map pairs, and fills in other pairs as a product of marginals. So geographic neighbours that are not coupled would lose their correlations. [Allow tomography of every pair alongside a restricted gate map. Check how many circuits that costs.]
8. **The paper's own slip.** One sentence calls ⟨Y⟩=1 "fully defensive". Elsewhere defence is ⟨X⟩, and the code uses ⟨X⟩.

## Status (31 Oct 2026)

- **Phases 1 and 2 are done, and phase 3 has started.** James chose: exactly the 2020 game, except that the war step uses `set_relationship` with fraction 1. Real QPUs are out of scope for now. The output maps should be good enough to hang on a wall.
- **The paper's emulated data is reproduced by the final 2020 code with its own single-qubit rotation rule** (`--rules final`). The `feb2020` rules, which commit dates first suggested, do not reproduce it. For differences 2 and 4 below, the port follows the 2020 code, as that is what produced the paper's results.
- **Results:**
  - Standard nations beat static opponents with both war steps.
  - With `set_relationship` (f=1), a nation that loses a city turns on the winner 71% of the time, against 36% before the transfer. The 2020 CZ method gives 60% against 34%.
  - Figures are in `figures/`.

## Phases

### 1. Faithful port (no new science)

Port the 2020 code to the current QuantumGraph and Qiskit 2, under `q_avrai/` in this repo:
- `world.py`: map state, influence, ownership, borders, cities, ruins. Use numpy arrays instead of pure-Python loops over 65k cells.
- `nations.py`: the quantum AI (tactic choice, state updates) on top of QuantumGraph. 2020 `set_state`/`get_state` map to `set_bloch`/`get_bloch`.
- `layouts.py`: the Rochester layout from `islands.py`, plus the opponent bicolouring.
- `render.py`: PNG frame per round and an animated GIF, with matplotlib or PIL. No pygame and no window.
- `run.py`: command line. Number of nations, backend, opponent mode, rounds, seed, the flags above. Writes JSON logs, not `str(dict)`.

Done when the port reproduces the 2020 rules (with the 2020-behaviour flags on) and runs headless on Aer for 7 to 11 nations.

### 2. Reproduce the 2020 result

- Port `analysis.ipynb`: mean area per round, standard against opponent, with standard deviation.
- Rerun 7, 9 and 11 nations, 10 runs each, on Aer. Compare with the 2020 `data.txt` files from the CoG branch, both in plots and with a simple significance test.
- Then run with the bug fixes on and see what changes.

### 3. Motte version (the new science)

- Replace "rotate to X, then CZ" with `set_relationship({'XZ': 1, 'ZX': 1}, loser, winner, fraction=...)`. This is the state-preserving two-qubit target that the paper deferred.
- Optionally make single-qubit updates partial for the loser too, rather than a hard reset to ⟨X⟩=1.
- Compare standard against opponent with the 2020 method and the Motte method. Also look at whether war pairs target each other more, as in the `war` count in the 2020 analysis.

### 4. Scale up

- Use the `ExpectationValue` backend (k=2, or k=3 for a check) to run 53 nations and beyond, cheaply.
- Before doing so, make the 2020 rotation rule (`single='2020'`) handle a Bloch vector of exactly zero. It currently gives NaN, and the exact backend produces zero vectors after an f=1 war step.
- A general layout: initial cities from any coupling map, such as a graph embedding with heavy-hex coordinates. Then current IBM devices (Heron, 133/156 qubits) can be used instead of Rochester.
- Real hardware run: only with James's sign-off on device, budget and account.

### 5. Tidy up

- Tests for the world mechanics and the AI rules, using a fixed seed and a tiny map.
- README: what it is, how to run it, and how it relates to the paper.
- Add a worked example and the lessons learned to `motte-me`.

## Questions for James

1. Main aim: a faithful re-run of the paper, the Motte version as new science, or a playable game/demo? This plan assumes the first two, in that order.
2. Bugs 2 and 4: fix them by default, or follow the 2020 code by default?
3. Is a hardware run in scope? If so, which device and budget?
4. Should terrain be in scope? The 2020 maps are flat, and the "quantum island" code in `islands.py` is unused.
5. Should Q_Avrai live inside the QuantumGraph repo (as here), or in a separate repo that depends on it?

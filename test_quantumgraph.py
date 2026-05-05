# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been changed from the originals.
#
# Copyright IBM Quantum 2020
# Copyright Moth Quantum 2025-2026

import unittest
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

from quantumgraph import QuantumGraph, ExpectationValue
from quantumgraph.GraphMitigation import pairwise_mitigation_circuits, PairwiseMitigationFitter


# ---------------------------------------------------------------------------
# ExpectationValue — exact, no shots, fast
# ---------------------------------------------------------------------------

class TestExpectationValue(unittest.TestCase):

    def test_single_qubit_gates(self):
        n = 2
        qc = QuantumCircuit(n)
        qc.h(0)
        qc.ry(-np.pi / 8, 1)
        ev = ExpectationValue(n)
        ev.apply_circuit(qc)
        self.assertAlmostEqual(ev.pauli_decomp['XI'],  1.0)
        self.assertAlmostEqual(ev.pauli_decomp['XX'], -0.3826834323650898)
        self.assertAlmostEqual(ev.pauli_decomp['XZ'],  0.9238795325112867)

    def test_cx_propagates_x(self):
        """CX with control |1⟩ flips the target: IZ should be -1."""
        n = 2
        qc = QuantumCircuit(n)
        qc.x(0)
        qc.cx(0, 1)
        ev = ExpectationValue(n)
        ev.apply_circuit(qc)
        self.assertAlmostEqual(ev.pauli_decomp['IZ'], -1.0)

    def test_initial_state_all_z_up(self):
        n = 4
        ev = ExpectationValue(n)
        ev.apply_circuit(QuantumCircuit(n))
        for q in range(n):
            key = ['I'] * n
            key[q] = 'Z'
            self.assertAlmostEqual(ev.pauli_decomp[''.join(key)], 1.0)

    def test_ghz_zz_correlators_k2(self):
        n = 3
        qc = QuantumCircuit(n)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        ev = ExpectationValue(n)
        ev.apply_circuit(qc)
        for key in ['ZZI', 'ZIZ', 'IZZ']:
            self.assertAlmostEqual(ev.pauli_decomp[key], 1.0, msg=key)

    def test_ghz_three_body_correlators_k3(self):
        n = 3
        qc = QuantumCircuit(n)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        ev = ExpectationValue(n, k=3)
        ev.apply_circuit(qc)
        for key in ['ZZI', 'ZIZ', 'IZZ']:
            self.assertAlmostEqual(ev.pauli_decomp[key],  1.0, msg=key)
        for key in ['XYY', 'YXY', 'YYX']:
            self.assertAlmostEqual(ev.pauli_decomp[key], -1.0, msg=key)
        self.assertAlmostEqual(ev.pauli_decomp['XXX'], 1.0)


# ---------------------------------------------------------------------------
# Shared QuantumGraph logic — runs against any backend
# ---------------------------------------------------------------------------

class QuantumGraphTests:
    """
    Mixin containing all backend-agnostic QuantumGraph tests.
    Concrete subclasses must set self.graph and self.EPS in setUp.
    """

    # --- initial state ---

    def test_initial_state_z_up(self):
        """All qubits start in |0⟩: Z=1, X=Y=0."""
        for q in range(self.n):
            bloch = self.graph.get_bloch(q)
            self.assertAlmostEqual(bloch['Z'],  1.0, delta=self.EPS, msg=f'qubit {q} Z')
            self.assertAlmostEqual(bloch['X'],  0.0, delta=self.EPS, msg=f'qubit {q} X')
            self.assertAlmostEqual(bloch['Y'],  0.0, delta=self.EPS, msg=f'qubit {q} Y')

    # --- set_bloch ---

    def test_set_bloch_rotates_target_qubit(self):
        """set_bloch moves the target qubit toward the given Bloch vector."""
        self.graph.set_bloch({'X': 1.0}, 1)
        bloch = self.graph.get_bloch(1)
        self.assertAlmostEqual(bloch['X'],  1.0, delta=self.EPS)
        self.assertAlmostEqual(bloch['Y'],  0.0, delta=self.EPS)
        self.assertAlmostEqual(bloch['Z'],  0.0, delta=self.EPS)

    def test_set_bloch_leaves_other_qubits_unchanged(self):
        """set_bloch should not disturb qubits it does not target."""
        self.graph.set_bloch({'X': 1.0}, 1)
        for q in [0, 2]:
            bloch = self.graph.get_bloch(q)
            self.assertAlmostEqual(bloch['Z'],  1.0, delta=self.EPS, msg=f'qubit {q} Z')
            self.assertAlmostEqual(bloch['X'],  0.0, delta=self.EPS, msg=f'qubit {q} X')

    def test_set_bloch_update_false_defers_tomography(self):
        """update=False modifies the circuit but does not re-run tomography."""
        bloch_before = self.graph.get_bloch(0)
        self.graph.set_bloch({'X': 1.0}, 0, update=False)
        bloch_stale = self.graph.get_bloch(0)
        self.assertAlmostEqual(bloch_stale['Z'], bloch_before['Z'], delta=self.EPS)
        self.graph.update_tomography()
        bloch_fresh = self.graph.get_bloch(0)
        self.assertAlmostEqual(bloch_fresh['X'], 1.0, delta=self.EPS)

    def test_set_bloch_fraction(self):
        """fraction=0.5 applies half the rotation."""
        self.graph.set_bloch({'X': -1.0}, 0, fraction=0.5)
        bloch = self.graph.get_bloch(0)
        self.assertAlmostEqual(bloch['X'], -1.0 / np.sqrt(2), delta=self.EPS)
        self.assertAlmostEqual(bloch['Z'],  1.0 / np.sqrt(2), delta=self.EPS)

    # --- get_relationship ---

    def test_get_relationship_product_state(self):
        """For a product state, two-qubit correlators equal products of marginals."""
        self.graph.set_bloch({'X': 1.0}, 0)
        b0 = self.graph.get_bloch(0)
        b1 = self.graph.get_bloch(1)
        rel = self.graph.get_relationship(0, 1)
        for p in ['XZ', 'ZX', 'XX', 'ZZ']:
            expected = b0[p[0]] * b1[p[1]]
            self.assertAlmostEqual(rel[p], expected, delta=2 * self.EPS, msg=p)

    def test_get_relationship_argument_order(self):
        """get_relationship(j,k)['AB'] == get_relationship(k,j)['BA']."""
        self.graph.set_bloch({'X': 1.0}, 0)
        for j, k in [(0, 1), (0, 2), (1, 2)]:
            rel_jk = self.graph.get_relationship(j, k)
            rel_kj = self.graph.get_relationship(k, j)
            for p in ['XZ', 'ZX', 'XY', 'YX', 'ZZ', 'XX']:
                self.assertAlmostEqual(
                    rel_jk[p], rel_kj[p[::-1]],
                    delta=2 * self.EPS, msg=f'({j},{k}) {p}'
                )

    def test_relationship_after_x_rotation(self):
        """After rotating qubit 0 to X=1: XZ=1 for (0,q), ZX=1 for (q,0)."""
        self.graph.set_bloch({'X': 1.0}, 0)
        for q in [1, 2]:
            self.assertAlmostEqual(
                self.graph.get_relationship(0, q)['XZ'], 1.0, delta=self.EPS,
                msg=f'XZ on (0,{q})'
            )
            self.assertAlmostEqual(
                self.graph.get_relationship(q, 0)['ZX'], 1.0, delta=self.EPS,
                msg=f'ZX on ({q},0)'
            )

    # --- set_relationship ---

    def test_set_relationship_zz(self):
        self.graph.set_relationship({'ZZ': +1}, 0, 1)
        self.assertAlmostEqual(self.graph.get_relationship(0, 1)['ZZ'], 1.0, delta=2 * self.EPS)

    def test_set_relationship_xx(self):
        self.graph.set_relationship({'XX': +1}, 0, 1)
        self.assertAlmostEqual(self.graph.get_relationship(0, 1)['XX'], 1.0, delta=2 * self.EPS)

    def test_set_relationship_commuting_pair(self):
        rel = {'ZX': +1, 'XZ': +1}
        self.graph.set_relationship(rel, 0, 1)
        result = self.graph.get_relationship(0, 1)
        for pauli, sign in rel.items():
            self.assertAlmostEqual(result[pauli], sign, delta=2 * self.EPS, msg=pauli)

    def test_set_relationship_zz_entangles_x_product_state(self):
        """ZZ=+1 on |+⟩|+⟩ projects into span{|00⟩,|11⟩}, yielding |Φ+⟩: ZZ=XX=+1, XI=IX=0."""
        self.graph.set_bloch({'X': 1.0}, 0)
        self.graph.set_bloch({'X': 1.0}, 1)
        self.graph.set_relationship({'ZZ': +1}, 0, 1)
        rel = self.graph.get_relationship(0, 1)
        self.assertAlmostEqual(rel['ZZ'],  1.0, delta=2 * self.EPS, msg='ZZ')
        self.assertAlmostEqual(rel['XX'],  1.0, delta=2 * self.EPS, msg='XX')
        self.assertAlmostEqual(self.graph.get_bloch(0)['X'], 0.0, delta=2 * self.EPS, msg='XI')
        self.assertAlmostEqual(self.graph.get_bloch(1)['X'], 0.0, delta=2 * self.EPS, msg='IX')

    def test_set_relationship_noncommuting_finds_closest(self):
        """Non-commuting or non-physical constraints should not raise; the closest
        physical state is found by clipping negative eigenvalues of the target RDM."""
        self.graph.set_relationship({'XX': +1, 'ZI': +1}, 0, 1)
        rel = self.graph.get_relationship(0, 1)
        # Result is unspecified but should be a valid dict with all Pauli keys
        for pauli in ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']:
            self.assertIn(pauli, rel)

    def test_set_relationship_fraction(self):
        """fraction<1 performs a partial rotation."""
        self.graph.set_relationship({'ZZ': +1}, 0, 1, fraction=0.5)
        result = self.graph.get_relationship(0, 1)
        self.assertGreater(result['ZZ'], 0.0)
        self.assertLess(result['ZZ'], 1.0 + self.EPS)


# ---------------------------------------------------------------------------
# AerSimulator backend
# ---------------------------------------------------------------------------

class TestQuantumGraph(QuantumGraphTests, unittest.TestCase):

    EPS = 0.05   # ~4σ at 8192 shots
    n = 3

    def setUp(self):
        self.graph = QuantumGraph(self.n)

    # --- circuit introspection (AerSimulator only) ---

    def test_tomo_circs_stored(self):
        """tomo_circs is populated after construction and has the right length."""
        self.assertTrue(hasattr(self.graph, 'tomo_circs'))
        self.assertIsInstance(self.graph.tomo_circs, list)
        # n=3: 3 + 6*ceil(log3(3)) = 9
        self.assertEqual(len(self.graph.tomo_circs), 9)

    def test_tomo_circs_refreshed_after_update(self):
        """Calling set_bloch (update=True) replaces tomo_circs with new circuits."""
        old_circs = self.graph.tomo_circs[:]
        self.graph.set_bloch({'X': 1.0}, 0)
        self.assertIsNot(self.graph.tomo_circs, old_circs)

    def test_coupling_map_bipartite_gives_9_circuits(self):
        """A bipartite coupling map triggers the 9-circuit optimisation."""
        g = QuantumGraph(4, coupling_map=[[0, 1], [1, 2], [2, 3], [0, 3]])
        self.assertEqual(len(g.tomo_circs), 9)

    def test_square_lattice_coupling_map(self):
        """3x3 square lattice: bipartite so 9 circuits; on-map gates work; diagonal uses marginals."""
        # 0-1-2
        # | | |
        # 3-4-5
        # | | |
        # 6-7-8
        nn_pairs = [(0,1),(1,2),(3,4),(4,5),(6,7),(7,8),
                    (0,3),(1,4),(2,5),(3,6),(4,7),(5,8)]
        g = QuantumGraph(9, coupling_map=nn_pairs)

        self.assertEqual(len(g.tomo_circs), 9)

        # on-map pair is accessible
        self.assertIn('ZZ', g.get_relationship(0, 1))

        # set_relationship works for an on-map pair
        g.set_relationship({'ZZ': +1}, 0, 1)
        self.assertAlmostEqual(g.get_relationship(0, 1)['ZZ'], 1.0, delta=self.EPS)

        # diagonal pair (0,2) is off-map: get_relationship returns product of marginals
        rel = g.get_relationship(0, 2)
        b0, b2 = g.get_bloch(0), g.get_bloch(2)
        for pauli in ['ZZ', 'XZ', 'ZX']:
            self.assertAlmostEqual(rel[pauli], b0[pauli[0]] * b2[pauli[1]],
                                   delta=self.EPS, msg=pauli)

        # set_relationship raises for off-map pair
        with self.assertRaises(ValueError):
            g.set_relationship({'ZZ': +1}, 0, 2)

    def test_coupling_map_non_bipartite_falls_back(self):
        """A non-bipartite coupling map uses the full coloring scheme."""
        g = QuantumGraph(4, coupling_map=[[0, 1], [1, 2], [0, 2]])
        self.assertGreater(len(g.tomo_circs), 9)

    def test_coupling_map_pairs_accessible(self):
        """get_relationship works for pairs in the coupling map."""
        g = QuantumGraph(4, coupling_map=[[0, 1], [2, 3]])
        for q0, q1 in [(0, 1), (2, 3)]:
            rel = g.get_relationship(q0, q1)
            self.assertIn('ZZ', rel)

    def test_get_relationship_outside_coupling_map_uses_marginals(self):
        """Pairs outside the coupling map return product-of-marginals approximation."""
        g = QuantumGraph(4, coupling_map=[[0, 1], [2, 3]])
        # (0, 2) is not in the coupling map; should return product of marginals
        rel = g.get_relationship(0, 2)
        b0 = g.get_bloch(0)
        b2 = g.get_bloch(2)
        for pauli in ['ZZ', 'XX', 'XZ']:
            expected = b0[pauli[0]] * b2[pauli[1]]
            self.assertAlmostEqual(rel[pauli], expected, delta=self.EPS, msg=pauli)

    def test_set_relationship_outside_coupling_map_raises(self):
        """set_relationship raises ValueError for a pair not in the coupling map."""
        g = QuantumGraph(4, coupling_map=[[0, 1], [2, 3]])
        with self.assertRaises(ValueError):
            g.set_relationship({'ZZ': +1}, 0, 2)


# ---------------------------------------------------------------------------
# ExpectationValue backend — same logical tests, exact results
# ---------------------------------------------------------------------------

class TestQuantumGraphEV(QuantumGraphTests, unittest.TestCase):

    EPS = 1e-3   # EV is exact up to floating-point; tight but forgiving of rounding
    n = 3

    def setUp(self):
        self.graph = QuantumGraph(self.n, backend=ExpectationValue(self.n))


# ---------------------------------------------------------------------------
# PairwiseMitigationFitter
# ---------------------------------------------------------------------------

class TestPairwiseMitigation(unittest.TestCase):

    def setUp(self):
        self.n = 6
        self.qc = QuantumCircuit(self.n)
        self.circs = pairwise_mitigation_circuits(self.qc)

    def test_circuit_names_cover_all_pairs(self):
        """Every pair of qubits sees all four bitstring combinations across the circuits."""
        names = [eval(c.name) for c in self.circs]
        for j in range(self.n - 1):
            for k in range(j + 1, self.n):
                seen = {name[j] + name[k] for name in names}
                self.assertEqual(seen, {'00', '01', '10', '11'},
                                 msg=f'pair ({j},{k}) missing combinations')

    def test_circuit_count(self):
        """Number of circuits is 2 * (ceil(log2(N)) + 1)."""
        import math
        expected = 2 * (math.ceil(math.log2(self.n)) + 1)
        self.assertEqual(len(self.circs), expected)

    def test_mitigate_counts_runs(self):
        """mitigate_counts returns a dict with positive total weight."""
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            pauli_error([('X', 0.05), ('I', 0.95)]), 'measure'
        )
        backend = AerSimulator()
        result = backend.run(
            transpile(self.circs, backend), noise_model=noise_model, shots=512
        ).result()
        fitter = PairwiseMitigationFitter(result, self.circs)
        mitigated = fitter.mitigate_counts({'00': 80, '01': 10, '10': 10, '11': 0}, (0, 1))
        self.assertIsInstance(mitigated, dict)
        self.assertGreater(sum(mitigated.values()), 0)

    def test_mitigate_counts_reduces_error(self):
        """Mitigation moves counts closer to the ideal distribution under readout noise."""
        p = 0.1
        noise_model = NoiseModel()
        noise_model.add_all_qubit_quantum_error(
            pauli_error([('X', p), ('I', 1 - p)]), 'measure'
        )
        backend = AerSimulator()
        result = backend.run(
            transpile(self.circs, backend), noise_model=noise_model, shots=2048
        ).result()
        fitter = PairwiseMitigationFitter(result, self.circs)

        # Counts for an ideal |00⟩ state degraded by p=0.1 readout error.
        # mitigate_counts mutates its input, so record the noisy fraction first.
        noisy = {'00': int((1 - p) ** 2 * 1000),
                 '01': int(p * (1 - p) * 1000),
                 '10': int(p * (1 - p) * 1000),
                 '11': int(p ** 2 * 1000)}
        noisy_frac = noisy['00'] / sum(noisy.values())
        mitigated = fitter.mitigate_counts(dict(noisy), (0, 1))
        mitigated_frac = mitigated['00'] / sum(mitigated.values())
        self.assertGreater(mitigated_frac, noisy_frac)


if __name__ == '__main__':
    unittest.main()

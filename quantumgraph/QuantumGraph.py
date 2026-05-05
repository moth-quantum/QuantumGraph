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

from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CZGate

from pairwise_tomography import pairwise_state_tomography_circuits, PairwiseStateTomographyFitter

from quantumgraph.ExpectationValue import ExpectationValue

from numpy import sqrt, conj, kron, dot, outer, nan, isnan
import numpy as np

from scipy import linalg as la
from scipy.linalg import fractional_matrix_power as pwr

from random import random

# define the Pauli matrices in a dictionary
matrices = {}
matrices['I'] = np.identity(2,dtype='complex')
matrices['X'] = np.array([[0,1+0j],[1+0j,0]])
matrices['Y'] = np.array([[0,-1j],[1j,0]])
matrices['Z'] = np.array([[1+0j,0],[0,-1+0j]])
for pauli1 in ['I','X','Y','Z']:
    for pauli2 in ['I','X','Y','Z']:
        matrices[pauli1+pauli2] = kron(matrices[pauli2],matrices[pauli1])

class QuantumGraph ():

    def __init__ (self,num_qubits,coupling_map=None,backend=None):
        '''
        Args:
            num_qubits: The number of qubits, and hence the number of nodes in the graph.
            coupling_map: A list of pairs of qubits, corresponding to edges in the graph.
                If none is given, the backend's coupling map is used if available,
                otherwise a fully connected graph is used.
            backend: The backend on which the graph will be run as a Qiskit backend object.
                If none is given, a local simulator is used.
        '''

        self.num_qubits = num_qubits

        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

        # the coupling map consists of pairs (j,k) with the convention j<k
        if coupling_map is not None:
            self.coupling_map = [(min(a, b), max(a, b)) for a, b in coupling_map]
        elif type(self.backend) == ExpectationValue:
            self.coupling_map = [(min(a, b), max(a, b)) for a, b in self.backend.coupling_map]
        elif hasattr(self.backend, 'coupling_map') and self.backend.coupling_map is not None:
            self.coupling_map = sorted({(min(a, b), max(a, b)) for a, b in self.backend.coupling_map.get_edges()})
        else:
            self.coupling_map = [
                (j, k)
                for j in range(self.num_qubits - 1)
                for k in range(j + 1, self.num_qubits)
            ]

        self.qc = QuantumCircuit(self.num_qubits)
        self.update_tomography()

    def update_tomography(self, shots=8192, verbose=False):
        '''
        Runs the pairwise tomography circuits for the current state and stores
        the results. After this call, self.tomo_circs contains the full list of
        circuits that were executed, and self.tomography is the fitted fitter.

        Args:
            shots: Number of shots per circuit.
            verbose: If True, print circuit count, depth, 2-qubit gate depth,
                     gate counts, and the circuits themselves.
        '''
        if verbose:
            from collections import Counter

            def range_str(vals):
                lo, hi = min(vals), max(vals)
                return str(lo) if lo == hi else f'{lo}-{hi}'

            circs = pairwise_state_tomography_circuits(
                self.qc, self.qc.qregs[0], pairs_list=self.coupling_map
            )
            depths    = [c.depth() for c in circs]
            depths_2q = [c.depth(lambda x: x.operation.num_qubits > 1) for c in circs]
            ops = Counter()
            for c in circs:
                ops.update(c.count_ops())

            print(f'Circuits: {len(circs)}')
            print(f'Depth: {range_str(depths)}')
            print(f'2-qubit gate depth: {range_str(depths_2q)}')
            print('Gate counts (total across all circuits): ' +
                  ', '.join(f'{k}: {v}' for k, v in sorted(ops.items())))
            if type(self.backend) == ExpectationValue:
                print('ExpectationValue backend: exact simulation, no circuits run.')

        if type(self.backend) == ExpectationValue:
            self.backend.apply_circuit(self.qc)
            return []
        else:
            self.tomo_circs = pairwise_state_tomography_circuits(
                self.qc, self.qc.qregs[0], pairs_list=self.coupling_map
            )
            result = self.backend.run(
                transpile(self.tomo_circs, self.backend), shots=shots
            ).result()
            self.tomography = PairwiseStateTomographyFitter(
                result, self.tomo_circs, self.qc.qregs[0]
            )
            self.exp = self.tomography.fit(output='expectation', pairs_list=self.coupling_map)
            return self.tomo_circs

    def get_bloch(self, qubit):
        '''
        Returns the X, Y and Z expectation values for the given qubit.
        '''
        if type(self.backend) == ExpectationValue:
            full_pauli = ['I'] * self.num_qubits
            expect = {}
            for pauli in ['X', 'Y', 'Z']:
                full_pauli[qubit] = pauli
                expect[pauli] = self.backend.pauli_decomp[''.join(full_pauli)]
                full_pauli[qubit] = 'I'
            return expect
        else:
            return self.exp[qubit]

    def get_relationship(self, qubit0, qubit1):
        '''
        Returns the two-qubit Pauli expectation values for a given pair of qubits.
        '''
        if type(self.backend) == ExpectationValue:
            full_pauli = ['I'] * self.num_qubits
            relationship = {}
            for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                full_pauli[qubit0] = pauli[0]
                full_pauli[qubit1] = pauli[1]
                relationship[pauli] = self.backend.pauli_decomp[''.join(full_pauli)]
                full_pauli[qubit0] = 'I'
                full_pauli[qubit1] = 'I'
            return relationship
        else:
            lo, hi = min(qubit0, qubit1), max(qubit0, qubit1)
            reverse = (lo != qubit0)
            relationship = {}
            if (lo, hi) in self.exp:
                pair_exp = self.exp[(lo, hi)]
                for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                    key = (pauli[1], pauli[0]) if reverse else (pauli[0], pauli[1])
                    relationship[pauli] = pair_exp[key]
            else:
                # Pair outside the coupling map; approximate as product of marginals
                b0 = self.get_bloch(qubit0)
                b1 = self.get_bloch(qubit1)
                for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                    relationship[pauli] = b0[pauli[0]] * b1[pauli[1]]
            return relationship

    def set_bloch(self, target_expect, qubit, fraction=1, update=True):
        '''
        Rotates the given qubit towards the given target state.

        Args:
            target_expect: Expectation values of the target state.
            qubit: Qubit on which the operation is applied.
            fraction: Fraction of the rotation toward the target state to apply.
            update: Whether to update the tomography after the rotation is added to the circuit.
        '''

        def bloch_to_rho(expect):
            x = expect.get('X', 0)
            y = expect.get('Y', 0)
            z = expect.get('Z', 0)
            return np.array([[(1 + z) / 2, (x - 1j * y) / 2],
                             [(x + 1j * y) / 2, (1 - z) / 2]])

        def get_eigenvecs(rho):
            _, vecs = la.eigh(rho)
            vecs = vecs[:, ::-1]  # descending eigenvalue order
            for k in range(vecs.shape[1]):
                idx = np.argmax(np.abs(vecs[:, k]))
                vecs[:, k] *= np.exp(-1j * np.angle(vecs[idx, k]))
            return vecs

        current_vecs = get_eigenvecs(bloch_to_rho(self.get_bloch(qubit)))
        target_vecs = get_eigenvecs(bloch_to_rho(target_expect))

        U = target_vecs @ current_vecs.conj().T

        if fraction != 1:
            U = pwr(U, fraction)

        the, phi, lam = OneQubitEulerDecomposer().angles(U)
        self.qc.u(the, phi, lam, qubit)

        if update:
            self.update_tomography()

    def set_relationship(self, relationships, qubit0, qubit1, fraction=1, update=True):
        '''
        Rotates the given pair of qubits towards the given target expectation values.

        Args:
            relationships: Dictionary of Pauli relationships to enforce, e.g. {'ZX': +1, 'XZ': +1}.
            qubit0, qubit1: Qubits on which the operation is applied.
            fraction: Fraction of the rotation toward the target state to apply.
            update: Whether to update the tomography after the rotation is added to the circuit.
        '''
        zero = 0.001

        def inner(vec1, vec2):
            out = 0
            for j in range(len(vec1)):
                out += conj(vec1[j]) * vec2[j]
            return out

        def normalize(vec):
            renorm = sqrt(inner(vec, vec))
            if abs(renorm * conj(renorm)) > zero:
                return np.copy(vec) / renorm
            else:
                return [nan for _ in vec]

        def random_vector():
            vec = np.array([2 * random() - 1 for _ in range(4)], dtype='complex')
            vec[0] = abs(vec[0])
            return normalize(vec)

        def is_valid(vec):
            return not any(isnan(vec[j]) for j in range(4))

        def make_vec(projector, seed_vec, ortho_vecs=None, max_tries=100):
            if ortho_vecs is None:
                ortho_vecs = []
            vec = dot(projector, seed_vec)
            for basis_vec in ortho_vecs:
                vec -= inner(basis_vec, vec) * basis_vec
            new_vec = normalize(vec)
            tries = 0
            while not is_valid(new_vec) and tries < max_tries:
                vec = dot(projector, random_vector())
                for basis_vec in ortho_vecs:
                    vec -= inner(basis_vec, vec) * basis_vec
                new_vec = normalize(vec)
                tries += 1
            if not is_valid(new_vec):
                raise ValueError(
                    "Could not construct another independent vector in the requested projector subspace. "
                    "This usually means the projector rank is too small for the number of vectors requested."
                )
            return new_vec

        def get_rho(qubit0, qubit1):
            rel = self.get_relationship(qubit0, qubit1)
            b0 = self.get_bloch(qubit0)
            b1 = self.get_bloch(qubit1)
            rho = np.identity(4, dtype='complex128')
            for pauli in ['X', 'Y', 'Z']:
                rho += b0[pauli] * matrices[pauli + 'I']
                rho += b1[pauli] * matrices['I' + pauli]
            for pauli in ['XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ']:
                rho += rel[pauli] * matrices[pauli]
            return rho / 4

        if (min(qubit0, qubit1), max(qubit0, qubit1)) not in self.coupling_map:
            raise ValueError(
                f"Pair ({qubit0}, {qubit1}) is not in the coupling map."
            )

        # Diagonalise inferred RDM with phase convention
        raw_vals, raw_vecs = la.eigh(get_rho(qubit0, qubit1))
        order = np.argsort(raw_vals)[::-1]
        vecs = []
        for k in order:
            vec = raw_vecs[:, k].copy()
            idx = np.argmax(np.abs(vec))
            vec *= np.exp(-1j * np.angle(vec[idx]))
            vecs.append(vec)

        # Build target RDM from supplied constraints (unspecified terms zero)
        target_rho = np.identity(4, dtype='complex128')
        for pauli, val in relationships.items():
            target_rho += val * matrices[pauli]
        target_rho /= 4

        # Diagonalise target RDM; clip negative eigenvalues to find closest physical state
        target_vals, target_vecs = la.eigh(target_rho)
        target_vals = np.maximum(target_vals, 0)
        total = target_vals.sum()
        if total > 0:
            target_vals /= total

        tol = 1e-8
        sorted_idx = np.argsort(target_vals)[::-1]
        groups = []
        i = 0
        while i < 4:
            j = i + 1
            while j < 4 and abs(target_vals[sorted_idx[i]] - target_vals[sorted_idx[j]]) < tol:
                j += 1
            groups.append(sorted_idx[i:j])
            i = j

        # For each degenerate eigenspace, form its projector and map current eigenstates in
        new_vecs = [None] * 4
        current_idx = 0
        for group in groups:
            P = sum(np.outer(target_vecs[:, k], target_vecs[:, k].conj()) for k in group)
            for _ in group:
                placed = [v for v in new_vecs[:current_idx] if v is not None]
                new_vecs[current_idx] = make_vec(P, vecs[current_idx], ortho_vecs=placed)
                current_idx += 1

        U = np.zeros((4, 4), dtype=complex)
        for j in range(4):
            U += outer(new_vecs[j], conj(vecs[j]))

        if fraction != 1:
            U = pwr(U, fraction)

        try:
            decomposer = TwoQubitBasisDecomposer(CZGate())
            circuit = decomposer(U)
            gate = circuit.to_instruction()
        except Exception as e:
            print(e)
            gate = None

        if gate:
            self.qc.append(gate, [qubit0, qubit1])

        if update:
            self.update_tomography()

        return gate

from qiskit import transpile, QuantumCircuit
from qiskit_aer import AerSimulator

from qiskit.synthesis import OneQubitEulerDecomposer
from qiskit.synthesis import TwoQubitBasisDecomposer
from qiskit.circuit.library import CXGate

from pairwise_tomography import pairwise_state_tomography_circuits, PairwiseStateTomographyFitter

from quantumgraph.ExpectationValue import ExpectationValue

from numpy import pi, cos, sin, sqrt, exp, arccos, arctan2, conj, array, kron, dot, outer, nan, isnan
import numpy as np
from numpy.random import normal

from scipy import linalg as la
from scipy.linalg import fractional_matrix_power as pwr

from random import random, choice

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

    def __init__ (self,num_qubits,coupling_map=[],backend=None):
        '''
        Args:
            num_qubits: The number of qubits, and hence the number of nodes in the graph.
            coupling_map: A list of pairs of qubits, corresponding to edges in the graph.
                If none is given, a fully connected graph is used.
            backend: The backend on which the graph will be run as a Qiskit backend object.
                If none is given, a local simulator is used.
        '''

        self.num_qubits = num_qubits

        # the coupling map consists of pairs [j,k] with the convention j<k
        self.coupling_map = []
        for j in range(self.num_qubits-1):
            for k in range(j+1,self.num_qubits):
                if ([j,k] in coupling_map) or ([j,k] in coupling_map) or (not coupling_map):
                    self.coupling_map.append([j,k])

        if backend is None:
            self.backend = AerSimulator()
        else:
            self.backend = backend

        self.qc = QuantumCircuit(self.num_qubits)
        self.update_tomography()

    def update_tomography(self, shots=8192):
        '''
        Runs the pairwise tomography circuits for the current state and stores
        the results. After this call, self.tomo_circs contains the full list of
        circuits that were executed, and self.tomography is the fitted fitter.

        Args:
            shots: Number of shots per circuit.
        '''
        if type(self.backend) == ExpectationValue:
            self.backend = ExpectationValue(self.backend.n,
                                            k=self.backend.k,
                                            pairs=self.backend.pairs)
            self.backend.apply_circuit(self.qc)
        else:
            pairs_list = [tuple(p) for p in self.coupling_map] if self.coupling_map else None
            self.tomo_circs = pairwise_state_tomography_circuits(
                self.qc, self.qc.qregs[0], pairs_list=pairs_list
            )
            result = self.backend.run(
                transpile(self.tomo_circs, self.backend), shots=shots
            ).result()
            self.tomography = PairwiseStateTomographyFitter(
                result, self.tomo_circs, self.qc.qregs[0]
            )
            self.exp = self.tomography.fit(output='expectation', pairs_list=pairs_list)

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
            pair_exp = self.exp[(lo, hi)]
            relationship = {}
            for pauli in ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']:
                # key (a, b): a = basis for lo, b = basis for hi
                key = (pauli[1], pauli[0]) if reverse else (pauli[0], pauli[1])
                relationship[pauli] = pair_exp[key]
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

        def normalize(expect):
            R = sqrt(expect['X']**2 + expect['Y']**2 + expect['Z']**2)
            return {pauli: expect[pauli]/R for pauli in expect}

        def get_basis(expect):
            normalized_expect = normalize(expect)
            theta = arccos(normalized_expect['Z'])
            phi = arctan2(normalized_expect['Y'], normalized_expect['X'])
            state0 = [cos(theta/2), exp(1j*phi)*sin(theta/2)]
            state1 = [conj(state0[1]), -conj(state0[0])]
            return [state0, state1]

        for pauli in ['X', 'Y', 'Z']:
            if pauli not in target_expect:
                target_expect[pauli] = 0

        current_basis = get_basis(self.get_bloch(qubit))
        target_basis = get_basis(target_expect)
        U = array([[0 for _ in range(2)] for _ in range(2)], dtype=complex)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U[j][k] += target_basis[i][j] * conj(current_basis[i][k])

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

        def projector_rank(P, tol=1e-8):
            vals = la.eigvalsh(P)
            return sum(abs(val) > tol for val in vals)

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

        def commute(pauli1, pauli2):
            noncommuting_pairs = {
                ('X', 'Y'), ('Y', 'X'),
                ('Y', 'Z'), ('Z', 'Y'),
                ('Z', 'X'), ('X', 'Z')
            }
            flips = 0
            for a, b in zip(pauli1, pauli2):
                if (a, b) in noncommuting_pairs:
                    flips += 1
            return (flips % 2) == 0

        paulis = list(relationships.keys())
        for j in range(len(paulis)):
            for k in range(j + 1, len(paulis)):
                if not commute(paulis[j], paulis[k]):
                    raise ValueError(
                        f"Noncommuting relationships supplied: {paulis[j]} and {paulis[k]}"
                    )

        raw_vals, raw_vecs = la.eigh(get_rho(qubit0, qubit1))
        vals = sorted([(val, k) for k, val in enumerate(raw_vals)], reverse=True)
        vecs = [[raw_vecs[j][k] for j in range(4)] for (_, k) in vals]

        Pup = np.identity(4, dtype='complex')
        for (pauli, sign) in relationships.items():
            Pup = dot(Pup, (matrices['II'] + sign * matrices[pauli]) / 2)
        Pdown = matrices['II'] - Pup

        rank_up = projector_rank(Pup)
        rank_down = 4 - rank_up

        new_vecs = [[nan for _ in range(4)] for _ in range(4)]

        for j in range(rank_up):
            new_vecs[j] = make_vec(Pup, vecs[j], ortho_vecs=new_vecs[:j])

        for j in range(rank_up, 4):
            new_vecs[j] = make_vec(Pdown, vecs[j], ortho_vecs=new_vecs[rank_up:j])

        U = np.zeros((4, 4), dtype=complex)
        for j in range(4):
            U += outer(new_vecs[j], conj(vecs[j]))

        if fraction != 1:
            U = pwr(U, fraction)

        try:
            decomposer = TwoQubitBasisDecomposer(CXGate())
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

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

import numpy as np
from qiskit import QuantumCircuit, transpile
import itertools, copy

class ExpectationValue():
    
    def __init__(self,n,k=2,coupling_map=None):

        self.n = n
        self.k = k

        if coupling_map:
            self.coupling_map = coupling_map
        else:
            self.coupling_map = []
            for i in range(n-1):
                for j in range(i,n):
                    self.coupling_map.append([i,j])

        self.neighbours = self._get_neighbours()

        self.paths = [[j] for j in range(self.n)]
        for i in range(self.k-1):
            new_paths = []
            for path in self.paths:
                q0 = path[-1]
                for q1 in self.neighbours[q0]:
                    if q1 not in path:
                        new_path = copy.copy(path)
                        new_path.append(q1)
                        new_paths.append(new_path)
            self.paths = new_paths

        self.initialize_decomp()

        self._supported_by = [{p:[] for p in ['I','X','Y','Z']} for _ in range(n)]
        for pauli_n in self.pauli_decomp:
            for q in range(n):
                self._supported_by[q][pauli_n[q]].append(pauli_n)

    def initialize_decomp(self):
        paulis = ['I','X','Y','Z']
        self.pauli_decomp = {}
        for path in self.paths:
            for pauli_k in itertools.product(*(paulis for _ in range(self.k))):
                pauli_n = ['I']*self.n
                for j,q in enumerate(path):
                    pauli_n[q] = pauli_k[j]
                self.pauli_decomp[''.join(pauli_n)] = float('X' not in pauli_k and 'Y' not in pauli_k)
                    
    def _get_neighbours(self):
        neighbours = {j:[] for j in range(self.n)}
        for [j,k] in self.coupling_map:
            neighbours[j].append(k)
            neighbours[k].append(j)
        return neighbours
    
    def _get_gates(self,qc):
        '''
        Turns a circuit into a list of tuples describing rx, ry, rz and cz gates.
        '''
        
        # compile to rx, ry, rz and cz
        qc = transpile(qc,basis_gates=['u3','cz'])

        gates = []
        for instruction in qc.data:
            op, qargs = instruction.operation, instruction.qubits
            if op.name == 'cz':
                q0 = qc.qubits.index(qargs[0])
                q1 = qc.qubits.index(qargs[1])
                gates.append(('cz', q0, q1))
            elif op.name == 'u3':
                angles = [float(a) for a in op.params]
                angles = [angles[2], angles[0], angles[1]]
                for j, rotation in enumerate(['rz', 'ry', 'rz']):
                    if angles[j] != 0:
                        gates.append((rotation, angles[j], qc.qubits.index(qargs[0])))
            elif op.name in ['rx', 'ry', 'rz']:
                angle = float(op.params[0])
                gates.append((op.name, angle, qc.qubits.index(qargs[0])))
            else:
                print(op.name + ' not supported.')

        return gates
        
    def _rotate(self,q,pp0,pp1,theta):
        e0,e1 = self.pauli_decomp[pp0],self.pauli_decomp[pp1]
        self.pauli_decomp[pp0] = np.cos(theta)*e0 + np.sin(theta)*e1
        self.pauli_decomp[pp1] = -np.sin(theta)*e0 + np.cos(theta)*e1
    
    def _change_char(self,string,char,index):
        list_string = list(string)
        list_string[index] = char
        return ''.join(list_string)
    
    def apply_circuit(self,qc,verbose=False,reinitialize=True):
        if reinitialize:
            self.initialize_decomp()
        
        gates = self._get_gates(qc)
        
        flips = {'XI':'XZ',
                 'YI':'YZ',
                 'IX':'ZX',
                 'XX':'YY',
                 'YX':'XY',
                 'ZX':'IX',
                 'IY':'ZY',
                 'XY':'YX',
                 'YY':'XX',
                 'ZY':'IY',
                 'XZ':'XI',
                 'YZ':'YI'}
                
        for gate in gates:
            
            if verbose:
                print()
                print(gate)
            
            if gate[0]!='cz':
                
                # unpack gate
                rotation,theta,q = gate
                
                # determine which paulis it will affect
                if rotation=='rx':
                    p0,p1 = 'Z','Y'
                if rotation=='ry':
                    p0,p1 = 'X','Z'
                if rotation=='rz':
                    p0,p1 = 'Y','X'

                for pp0 in self._supported_by[q][p0]:
                    pp1 = pp0[:q] + p1 + pp0[q+1::]
                    self._rotate(q,pp0,pp1,theta)
                    
                    if verbose:
                        if abs(self.pauli_decomp[pp0])>0.01 or abs(self.pauli_decomp[pp1])>0.01:
                            print('Rotate',pp0,'and',pp1)
                    
            else:
                
                decomp_diff = {}
                
                # unpack gate
                q0,q1 = gate[1],gate[2]
                                
                # loop through all the possible paulis on q0
                for p0_0 in ['I','X','Y','Z']: 
                    # loop through all n-qubit paulis that contain it
                    for pp0 in self._supported_by[q0][p0_0]:
                        
                        # get the 2-qubit pauli on q0 and q1
                        p0 = p0_0 + pp0[q1]
                        
                        if p0 in flips:
                            # get the 2-qubit pauli that cz flips this to
                            p1 = flips[p0]
                            # construct the corresponding n-qubit paulis
                            pp1 = self._change_char(pp0,p1[0],q0)
                            pp1 = self._change_char(pp1,p1[1],q1)

                            # if it is in pauli_decomp, flip them
                            if pp1 in self.pauli_decomp:                               
                                if verbose:
                                    if abs(self.pauli_decomp[pp0])>0.01 or abs(self.pauli_decomp[pp1])>0.01:
                                        if pp0 not in decomp_diff:
                                            print('Swap',pp0,'with',pp1)
                                decomp_diff[pp0] = self.pauli_decomp[pp1] * (-1)**(p0 in ['XY','YX'])
                                decomp_diff[pp1] = self.pauli_decomp[pp0] * (-1)**(p1 in ['XY','YX'])
                                
                            # otherwise infer a value and use that
                            else:
                                # determine the qubits on which pp1 has support
                                support = []
                                for j,char in enumerate(pp1):
                                    if char!='I':
                                        support.append(j)
                                
                                # get all possible bipartitions
                                bipartitions = []
                                for partition in itertools.product(*(range(2) for _ in range(len(support)))):
                                    parts = [[],[]]
                                    if 0 in partition and 1 in partition:
                                        for j,s in enumerate(support):
                                            parts[partition[j]].append(s)
                                        bipartitions.append(parts)
                                
                                # make a list of pairs of paulis products that combine to pp1
                                partitioned_pp1s = []
                                for parts in bipartitions:
                                    partitioned_pp1 = []
                                    for j,part in enumerate(parts):
                                        sub_pp1 = ['I']*self.n
                                        for j in part:
                                            sub_pp1[j] = pp1[j]
                                        partitioned_pp1.append(''.join(sub_pp1))
                                    partitioned_pp1s.append(partitioned_pp1)
                                
                                e1 = 0
                                partion_used = []
                                for [subpp1_a,subpp1_b] in partitioned_pp1s:
                                    if subpp1_a in self.pauli_decomp and subpp1_b in self.pauli_decomp:
                                        e = self.pauli_decomp[subpp1_a]*self.pauli_decomp[subpp1_b]
                                    else:
                                        e = 0
                                    if abs(e)>abs(e1):
                                        e1 = e      
                                        partition_used = [subpp1_a,subpp1_b]
                                decomp_diff[pp0] = e1 * (-1)**(p0 in ['XY','YX'])
                                
                                if verbose:
                                    if abs(self.pauli_decomp[pp0])>0.01:
                                        print('Effectively swap',pp0,'with',pp1,'using the value',round(e1,2),'inferred from',partition_used[0],'and',partition_used[1])
                
                for pp in decomp_diff:
                    self.pauli_decomp[pp] = decomp_diff[pp]
            
            if verbose:
                print('Resulting state:')
                for pp in self.pauli_decomp:
                    if abs(self.pauli_decomp[pp]) > 0.01:
                        print(pp, self.pauli_decomp[pp])

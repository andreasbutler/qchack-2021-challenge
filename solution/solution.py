from typing import List, Tuple

import numpy as np
import cirq
import scipy


def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    converter = cirq.google.ConvertToSycamoreGates()
    xmon_converter = cirq.google.ConvertToXmonGates()
    
    increment_unitaries = []
    for i in range(1, 10):
        inp = np.empty((2 ** i, 2 ** i))
        inp[1:] = np.eye(2 ** i)[:-1]
        inp[:1] = np.eye(2 ** i)[-1:]   
        increment_unitaries.append(inp)
        
        
    oneQ_unitaries = [
        (cirq.unitary(cirq.X), cirq.X),
        (cirq.unitary(cirq.Y), cirq.Y),
        (cirq.unitary(cirq.Z), cirq.Z),
        (cirq.unitary(cirq.H), cirq.H),
        (cirq.unitary(cirq.S), cirq.S),
        (cirq.unitary(cirq.T), cirq.T)
    ]
    
    twoQ_sycamore_unitaries = [
        (cirq.unitary(cirq.google.SycamoreGate()), cirq.google.SycamoreGate()),
        (cirq.unitary(cirq.IdentityGate(num_qubits=2)), cirq.IdentityGate(num_qubits=2)),
        (cirq.unitary(cirq.XX), cirq.XX),
        (cirq.unitary(cirq.YY), cirq.YY)
    ]
    
    twoQ_xmon_unitaries = [
        (cirq.unitary(cirq.ZZ), cirq.ZZ),
        (cirq.unitary(cirq.CX), cirq.CX)
    ]
    
    threeQ_sycamore_unitaries = [
        (cirq.unitary(cirq.CCX), cirq.CCX),
        (cirq.unitary(cirq.CSWAP), cirq.CSWAP),
        # (cirq.unitary(cirq.ControlledGate(cirq.ISWAP ** 0.5)), cirq.ControlledGate(cirq.ISWAP ** 0.5)),
        (cirq.unitary(cirq.CCZ), cirq.CCZ),
        (cirq.unitary(cirq.IdentityGate(num_qubits=3)), cirq.IdentityGate(num_qubits=3))
    ]
    
    fourQ_sycamore_unitaries = [
        (cirq.unitary(cirq.IdentityGate(num_qubits=4)), cirq.IdentityGate(num_qubits=4))
    ]
    
    
    def controlled_sqrt_iswap(qs):
        c = cirq.Circuit()
        c.append(cirq.ControlledGate(cirq.Y ** -0.5).on(qs[0], qs[2]))
        c.append(cirq.CCZ.on(qs[0], qs[1], qs[2]))
        c.append(cirq.ControlledGate(cirq.Y ** 0.5).on(qs[0], qs[2]))
        c.append(cirq.ControlledGate(cirq.Y ** 0.5).on(qs[0], qs[1]))
        c.append(cirq.CX.on(qs[0], qs[1]))
        c.append(cirq.ControlledGate(cirq.Y ** -0.5).on(qs[0], qs[1]))
        c.append(cirq.CCZ(qs[0], qs[1], qs[2]))
        c.append(cirq.ControlledGate(cirq.Y**0.5).on(qs[0], qs[1]))
        c.append(cirq.ControlledGate(cirq.T).on(qs[0], qs[1]))
        c.append(cirq.ControlledGate(cirq.Y**-0.5).on(qs[0], qs[1]))
        c.append(cirq.CCZ(qs[0], qs[1], qs[2]))
        c.append(cirq.ControlledGate(cirq.Y ** -0.5).on(qs[0], qs[2]))
        c.append(cirq.ControlledGate(cirq.Y**0.5).on(qs[0], qs[1]))
        c.append(cirq.ControlledGate(cirq.Z ** -0.25).on(qs[0], qs[1]))
        c.append(cirq.ControlledGate(cirq.Y**0.5).on(qs[0], qs[1]))
        c.append(cirq.CX.on(qs[0], qs[1]))
        c.append(cirq.CCZ.on(qs[0], qs[1], qs[2]))
        c.append(cirq.ControlledGate(cirq.Y ** 0.5).on(qs[0], qs[2]))
        return c
    
    def keep_func(gate):
        if cirq.unitary(gate).shape[0] < 5:
            return True
        else:
            return False
       
    def twoQ_sycamore_optimize(target_qubits, gate):
        g = gate.on(target_qubits[0], target_qubits[1])
        g = converter.convert(g)
        c = cirq.Circuit(g)
        c = cirq.google.optimized_for_sycamore(c, optimizer_type="sycamore")
        return c, []
    
    
    def twoQ_xmon_optimize(target_qubits, gate):
        g = gate.on(target_qubits[0], target_qubits[1])
        g = converter.convert(g)
        c = cirq.Circuit(g)
        c = cirq.google.optimized_for_sycamore(c, optimizer_type="xmon")
        return c, []
    
    
    # TODO: Fix up the CISWAP
    def threeQ_sycamore_optimize(target_qubits, gate):
        g = gate.on(*target_qubits)
        g = converter.convert(g)
        c = cirq.Circuit(g)
        c = cirq.google.optimized_for_sycamore(c, optimizer_type="xmon")
        return c, []
    
    
    def sycamore_optimize(target_qubits, gate):
        g = gate.on(*target_qubits)
        g = cirq.decompose(g, keep=keep_func)
        c = cirq.Circuit(g)
        c = cirq.google.optimized_for_sycamore(c, optimizer_type="sycamore")
        return c, []
    
    if matrix.shape[0] == 4:
        for twoQ_sycamore_unitary in twoQ_sycamore_unitaries:
            if np.isclose(matrix, twoQ_sycamore_unitary[0]).all():
                return twoQ_sycamore_optimize(target_qubits, twoQ_sycamore_unitary[1])

        for twoQ_xmon_unitary in twoQ_xmon_unitaries:
            if np.isclose(matrix, twoQ_xmon_unitary[0]).all():
                return twoQ_xmon_optimize(target_qubits, twoQ_xmon_unitary[1])
        
    if matrix.shape[0] == 8:
#         if np.isclose(matrix, cirq.unitary(cirq.ControlledGate(cirq.ISWAP ** 0.5))).all():
# #             g = cirq.ControlledGate(cirq.ISWAP ** 0.5).on(*target_qubits)
# #             g = cirq.decompose(g)
# #             c = controlled_sqrt_iswap(target_qubits)
#             g = cirq.FSimGate(-np.pi/4, 0)
#             g = cirq.decompose(g.on(*target_qubits))
#             c = cirq.Circuit(g)
#             return cirq.decompose(c, keep=keep_func), []     
        for threeQ_sycamore_unitary in threeQ_sycamore_unitaries:
            if np.isclose(matrix, threeQ_sycamore_unitary[0]).all():
                return threeQ_sycamore_optimize(target_qubits, threeQ_sycamore_unitary[1])
            
    
    if len(target_qubits) == 1:
        for g in oneQ_unitaries:
            if np.isclose(matrix, g[0]).all():
                gate = converter.convert(g[1].on(*target_qubits))
                return gate, []
            
            
    def CNOT_toffoli_decomp(bitset, garbage, t, d):
        target = t
        gates = []
        for i in range(1, len(bitset) - 1):
            gates.append(cirq.CCX(garbage[-i - d], bitset[-i], target))
            target = garbage[-i - d]
        for gate in gates:
            yield gate
        gates.reverse()
        yield cirq.CCX(bitset[0], bitset[1], target)
        for gate in gates[:-1]:
            yield gate
            
    
    def CNOT_toffoli_decomp(bitset, garbage, t, d):
        target = t
        gates = []
        for i in range(1, len(bitset) - 1):
            gates.append(cirq.CCX(garbage[-i - d], bitset[-i], target))
            target = garbage[-i - d]
        gates.append(cirq.CCX(bitset[0], bitset[1], target))
        for gate in gates:
            yield gate
        gates.reverse()
        for gate in gates[1:]:
            yield gate
        gates.reverse()
        for gate in gates[1:]:
            yield gate
        gates.reverse()
        for gate in gates[1:-1]:
            yield gate
    
    
    def decompose_large_CNOT(control_qubits, target, ancilla):
        if len(control_qubits) == 0:
            return cirq.X(target)

        if len(control_qubits) == 1:
            return cirq.CX(control_qubits[0], target)

        if len(control_qubits) == 2:
            return cirq.CCX(control_qubits[0], control_qubits[1], target)

        bitset1 = [tq for i, tq in enumerate(control_qubits) if i%2 == 0]
        bitset2 = [tq for i, tq in enumerate(control_qubits) if i%2 == 1]
        bitset2.append(ancilla)
        gate1 = CNOT_toffoli_decomp(bitset1, bitset2, ancilla, 1)
        gate2 = CNOT_toffoli_decomp(bitset2, bitset1, target, 0)
        gate3 = CNOT_toffoli_decomp(bitset1, bitset2, ancilla, 1)
        gate4 = CNOT_toffoli_decomp(bitset2, bitset1, target, 0)
        return cirq.Circuit([gate1, gate2, gate3, gate4])
    
    
    if len(target_qubits) == 4:
        for fourQ_sycamore_unitary in fourQ_sycamore_unitaries:
            if np.isclose(matrix, fourQ_sycamore_unitary[0]).all():
                return sycamore_optimize(target_qubits, fourQ_sycamore_unitary[1])
    
        if np.isclose(matrix, cirq.unitary(cirq.ControlledGate(cirq.CCX))).all():
            control_qubits = target_qubits[:-1]
            target = target_qubits[-1]
            ancilla = cirq.GridQubit(2, 3)
            c = decompose_large_CNOT(control_qubits, target, ancilla)
            return cirq.decompose(c), [ancilla]


    def increment(qubits):
        n_qubits = len(qubits)
        ancilla = cirq.GridQubit(2, 3)
        c = cirq.Circuit()
        for i in range(n_qubits):
            control_qubits = qubits[i + 1:]
            target = qubits[i]
            c.append(decompose_large_CNOT(control_qubits, target, ancilla))
        if n_qubits < 3:
            return c, []
        return c, [ancilla]
    
    
    def is_adjacent(q1, q2):
        return (q1.row == q2.row and np.abs(q1.col - q2.col) == 1) or (np.abs(q1.row - q2.row) == 1 and q1.col == q2.col)

    def swap_path(qs, q1, q2):
        path = []
        if q1 == q2:
            return path
        q = q1
        while not is_adjacent(q, q2):
            if not q2.row - q.row == 0:
                row_dir = int((q2.row - q.row)/np.abs(q2.row - q.row))
                if cirq.GridQubit(q.row + row_dir, q.col) in qs:
                    q = cirq.GridQubit(q.row + row_dir, q.col)
                    path.append(q)
                    continue
            if not q2.col - q.col == 0:
                col_dir = int((q2.col - q.col)/np.abs(q2.col - q.col))
                q = cirq.GridQubit(q.row, q.col + col_dir)
                path.append(q)
        return path
    
    def two_q_sycamore(g, q1, q2, qs):
        sp = swap_path(qs, q1, q2)
        if len(sp) == 0:
            yield g
            return
        q = q1
        for i in range(len(sp)):
            yield cirq.SWAP(q, sp[i])
            q = sp[i]
        yield g.gate(q, q2)
        sp.reverse()
        for i in range(1, len(sp)):
            yield cirq.SWAP(q, sp[i])
            q = sp[i]
        yield cirq.SWAP(q, q1)
    
    
    for U in increment_unitaries:
        if U.shape == matrix.shape:
            if np.isclose(U, matrix).all():
                if len(target_qubits) >= 7:
                        c, a = increment(target_qubits)
                        c = cirq.decompose(c)
                        # return c, a
                        cn = cirq.Circuit()
                        for g in c:
                            if len(g.qubits)==1:
                                cn.append(g)
                            else:
                                q1 = g.qubits[0]
                                q2 = g.qubits[1]
                                cn.append(cirq.Circuit(two_q_sycamore(g, q1, q2, target_qubits)))
                        return converter.convert(cn), a
                else:
                    c, a = increment(target_qubits)
                    c = cirq.decompose(c)
                    return c, a
    
    def isDiag(M):
        i, j = np.nonzero(M)
        return np.all(i == j)
    
    if isDiag(matrix):
        if np.isclose(matrix.diagonal(), 1).all():
            return cirq.Circuit(), []
    
    if isDiag(matrix):
        diagonal = matrix.diagonal()
        diagangles = (-1j * np.log(diagonal)).real

        diag_gate = cirq.DiagonalGate(diagangles)
        gate = diag_gate(*target_qubits)._decompose_()[1:]
        gate = converter.convert(gate)
        diag_circuit = cirq.Circuit(gate)
        return diag_circuit, []
    
    def pulverize_matrix(m):
        # N = 2^n
        N = len(m)

        L, CS, R = scipy.linalg.cossin(m, p=N / 2, q=N / 2)

        if len(m) == 2:
            stacked_matrices = np.stack([L, CS, R])
            return stacked_matrices

        # BREAK DOWN L MATRIX
        L1 = L[:round(N / 2), :round(N / 2)]
        L2 = L[round(N / 2):, round(N / 2):]

        L1_decomp = pulverize_matrix(L1)
        L2_decomp = pulverize_matrix(L2)

        left_matrices = 1j * np.zeros([L1_decomp.shape[0], L1_decomp.shape[1] * 2, L1_decomp.shape[2] * 2])
        left_matrices[:, :round(N / 2), :round(N / 2)] += L1_decomp
        left_matrices[:, round(N / 2):, round(N / 2):] += L2_decomp

        # BREAK DOWN R MATRIX
        R1 = R[:round(N / 2), :round(N / 2)]
        R2 = R[round(N / 2):, round(N / 2):]

        R1_decomp = pulverize_matrix(R1)
        R2_decomp = pulverize_matrix(R2)

        right_matrices = 1j * np.zeros([R1_decomp.shape[0], R1_decomp.shape[1] * 2, R1_decomp.shape[2] * 2])
        right_matrices[:, :round(N / 2), :round(N / 2)] += R1_decomp
        right_matrices[:, round(N / 2):, round(N / 2):] += R2_decomp

        all_matrices = np.concatenate([left_matrices, CS[None], right_matrices])
        return all_matrices


    def get_S_matrix(m):
        S = np.zeros([round(2 ** int(m)), round(2 ** int(m))]) + 1

        cols = [np.sign(np.cos(2 ** (m - 1 - i) * np.pi * np.arange(0, 2 ** int(m), 1) / 2 ** int(m) + 10 ** -6)) for i in range(int(m))]

        for bit in range(len(cols)):
            for i in range(len(S)):
                if round(2 ** bit) & i:
                    S[:, i] *= cols[bit]

        return S


    def gray_code_changes(m):
        return gray_code_changes_r(m) + [m]


    def gray_code_changes_r(m):
        if m < 0:
            return []
        return gray_code_changes_r(m - 1) + [m] + gray_code_changes_r(m - 1)


    def UCRy(ctrls, target, alphas, S_T_inv, seq):

        thetas = np.matmul(S_T_inv, alphas)

        gates = []
        for i in range(len(seq)):
            theta_i = thetas[i]
            q_i = ctrls[int(seq[i])]
            gates += [cirq.Ry(rads=theta_i)(target)]
            gates += [cirq.CNOT(q_i, target)]

        return gates

    def ry_angle(u):
        u_log = 1j*scipy.linalg.logm(u)
        return((2j*u_log[0][1]).real)

    def extract_ry_angles(matrix, target_bit, n):
        alphas = []
        for i in range(len(matrix)):
            if not int(2**(n - 1 - target_bit))&i:
                ry_gate = matrix[(i, i + int(2**(n - 1 - target_bit))),:][:,(i, i + int(2**(n - 1 - target_bit)))]
                alphas += [ry_angle(ry_gate)]
        alphas = np.array(alphas)
        return alphas

    def UCRy_gates_from_matrix(matrix, target_bit, qs, n, S_T_inv):
        alphas = extract_ry_angles(matrix, target_bit, n)
        ctrl_qs = qs[0:int(target_bit)] + qs[int(target_bit)+1:]
        gates = UCRy(ctrl_qs[::-1], qs[int(target_bit)], alphas, S_T_inv, gray_code_changes(n-2))
        return gates

    def arbitrary_diagonal_gate_from_matrix(matrix, qs):
        diagonal = matrix.diagonal()
        diagangles = (-1j * np.log(diagonal)).real
        diag_gate = cirq.DiagonalGate(diagangles)
        gates = diag_gate(*qs)._decompose_()[0:]
        return(gates)


    def arbitrary_operation_from_matrix(U, qs):
        n = round(np.log2(len(U)))
        S = get_S_matrix(n - 1)
        S_T_inv = np.linalg.inv(S.transpose())

        matrices = pulverize_matrix(U)
        ucry_target_bits = [n - 1 - i for i in gray_code_changes(n - 1)[:-1]]

        gates = []

        for m_i in range(len(matrices)):
            # print(m_i, "/", len(matrices))
            if m_i % 2 == 0:
                gates = arbitrary_diagonal_gate_from_matrix(matrices[m_i], qs) + gates
                # print(np.sum(np.abs(cirq.unitary(cirq.Circuit(arbitrary_diagonal_gate_from_matrix(matrices[m_i], qs))) - matrices[m_i])))
                # print(matrices[m_i])
            else:
                target_bit = ucry_target_bits[int(m_i / 2)]
                gates = UCRy_gates_from_matrix(matrices[m_i], target_bit, qs, n, S_T_inv) + gates
                # print(np.sum(np.abs(cirq.unitary(cirq.Circuit(UCRy_gates_from_matrix(matrices[m_i], target_bit, qs, n, S_T_inv))) - matrices[m_i])))

        return gates
    
    gates = arbitrary_operation_from_matrix(matrix, target_qubits)
    
    c = cirq.Circuit()
    for g in gates:
        if len(g.qubits)==1:
            c.append(g)
        elif len(g.qubits)==2:
            q1 = g.qubits[0]
            q2 = g.qubits[1]
            c.append(cirq.Circuit(two_q_sycamore(g, q1, q2, target_qubits)))
    return converter.convert(c), []
    
        
    return NotImplemented, []

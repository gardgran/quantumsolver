"""
Shor's Algorithm Quantum Solver

This is a quantum solver simulation for shor's algorthim.
The quantum simulation is done during the Quantum Phase Estimation (QPE) to find the period.
Once the period is found. The factors of N are classically solved.

The quantum simulation only works (in a reasonable time frame) for small numbers ( < 7 bits). 
Above that, the program will not attempt to run.

George Lake - 12/2025

Notes about selecting N:
N=15 (4 bits): 3 \times 4 = 12 qubits.
Memory: Negligible (KB). Instant.

N=35 (6 bits): 3 \times 6 = 18 qubits.
Memory: ~4 MB. Very Fast.

N=143 (8 bits): 3 \times 8 = 24 qubits.
Memory: ~256 MB. Slower (seconds to minutes).

N=511 (9 bits): 3 \times 9 = 27 qubits.
Memory: ~2 GB. Heavy.

N=1023 (10 bits): 3 \times 10 = 30 qubits.
Memory: ~16 GB. Maximum for most laptops.

N=2047 (11 bits): 3 \times 11 = 33 qubits.
Memory: ~128 GB. Impossible for most.
"""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit_aer import AerSimulator
from fractions import Fraction

import numpy as np
import random
import math
import sympy
import matplotlib.pyplot as plt

class ShorCircuit(QuantumCircuit):
    def __init__(self, a, N):
        """
        Initialize the Quantum Circuit for Shor's Algorithm

        Arguments:
            a (int): the base integer for modular exponentiation
            N (int): The number to be factored.
        """
        self.n_target = N.bit_length()
        self.n_count = 2 * self.n_target
        total_qubits = self.n_count + self.n_target

        super().__init__(total_qubits, self.n_count)
        
        self.a = a 
        self.N = N 

        self._create_circuit()
        
    def _get_controlled_unitary_matrix(self, power_of_a):
        """
        Constructs the matrix for the operation: Control-U^(2^j)
        U|y> = |(a^x * y) mod N>

        Note: In an actual large quantum computer, this matrix would not be built. We would use adder and multiplier gates. 
        However, for simulation (using a small N), this matrix is easier and faster.
        """
        dim_target = 2 ** self.n_target
        U_matrix = np.zeros((dim_target, dim_target), dtype=complex)

        effective_multiplier = pow(self.a, power_of_a, self.N)

        for y in range(dim_target):
            if y < self.N:
                target_y = (effective_multiplier * y) % self.N
            else:
                target_y = y
            U_matrix[target_y, y] = 1

        CU_matrix = np.block([
            [np.eye(dim_target), np.zeros((dim_target, dim_target))],
            [np.zeros((dim_target, dim_target)), U_matrix]
        ])

        return UnitaryGate(CU_matrix, label=f"CU {self.a}^{power_of_a}")

    def _create_circuit(self):
        """
        
        """
        self.h(range(self.n_count))
        self.x(self.num_qubits - 1)

        for i in range(self.n_count):
            power_of_a = 2 ** i
            CU_gate = self._get_controlled_unitary_matrix(power_of_a)

            target_qubits = list(range(self.n_count, self.num_qubits))
            control_qubit = i

            self.append(CU_gate, target_qubits + [control_qubit])
        
        qft_gate = QFTGate(self.n_count).inverse()
        self.append(qft_gate, range(self.n_count))

        self.measure(range(self.n_count), range(self.n_count))

    def run_simulator(self, simulator):
        """
        Runs the circuit on the Qiskit Aer Simulator
        """
        transpiled_circuit = transpile(self, simulator)
        result = simulator.run(transpiled_circuit, shots=1, memory=True).result()
        return result


class ShorAlgorithm:
    def __init__(self, N, max_attempts=-1, random_coprime_only=False, simulator=None):
        self.N = N
        self.simulator = simulator
        self.max_attempts = max_attempts
        self.random_coprime_only = random_coprime_only
        self.chosen_a = None
        self.r = None
        self.qpe_circuit = None

    def execute(self):
        """
        
        """
        is_N_invalid = self._is_N_invalid()
        if is_N_invalid:
            print(f"[INFO] N = {self.N} is trivially factorable: {is_N_invalid}")
            return is_N_invalid

        a_values = [a for a in range(2, self.N) if not self.random_coprime_only or (math.gcd(a, self.N) == 1)]
        print(f"[INFO] {len(a_values)} possible values of a: {a_values}")

        if not a_values: return None

        limit = len(a_values) if self.max_attempts <= -1 else min(self.max_attempts, len(a_values))
        attempts_count = 0

        while attempts_count < limit:
            print(f'\\n===== Attempt {attempts_count + 1} =======')
            attempts_count += 1

            self.chosen_a = random.choice(a_values)
            self.r = None
            print(f"[START] Chosen base a: {self.chosen_a}")

            if not self.random_coprime_only:
                gcd = math.gcd(self.chosen_a, self.N)
                if gcd != 1:
                    factor2 = self.N // gcd
                    print(f"=> Luck Guess! {self.chosen_a} shars a factor with {self.N}")
                    print(f"[SUCCESS] Found factors: {gcd} and {factor2}")
                    return gcd, factor2

            print(f">>> {self.chosen_a} and {self.N} are coprime. Running Quantum Circuit...")
            success = self._quantum_period_finding()

            if not success:
                if self.chosen_a in a_values: a_values.remove(self.chosen_a)
                continue

            factors = self._classical_postprocess()
            if factors:
                return factors

            if self.chosen_a in a_values: a_values.remove(self.chosen_a)

        print(f"[FAIL] No factors found after {limit} attempts.")
        return None

    def _is_N_invalid(self):
        """
        
        """
        if self.N <= 3: return 1, self.N
        if self.N % 2 == 0: return 2, self.N // 2
        if sympy.isprime(self.N): return 1, self.N
        for k in range(int(math.log2(self.N)), 1, -1):
            p = round(self.N ** (1 / k))
            if p ** k == self.N: return p, k
        return False

    def _quantum_period_finding(self):
        """
        
        """
        self.qpe_circuit = ShorCircuit(self.chosen_a, self.N)

        try:
            result = self.qpe_circuit.run_simulator(self.simulator)
        except Exception as e:
            print(f"[ERROR] Simulation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        state_bin = result.get_memory()[0]
        state_dec = int(state_bin, 2)

        phase = state_dec / (2 ** self.qpe_circuit.n_count)

        self.r = Fraction(phase).limit_denominator(self.N).denominator

        print(f"     -> Measured: |{state_dec}> (binary{state_bin})")
        print(f"     -> Phase: {phase:.4f}")
        print(f"     -> Estimated Period r: {self.r}")

        if self.r > self.N or self.r == 1:
            print(f"[WARNING] Invalid period r={self.r}. Retrying...")
            return False

        if pow(self.chosen_a, self.r, self.N) != 1:
            print(f"[WARNING] a^r != 1 (mod N) [a={self.chosen_a}, r={self.r}]. Phase estimation failed. Retrying...")
            return False

        return True

    def _classical_postprocess(self):
        """
        
        """
        if self.r % 2 != 0:
            print(f"[INFO] Period r={self.r} is odd. Cannot spit to find factors.")
            return None

        guess_1 = pow(self.chosen_a, self.r // 2, self.N) - 1
        guess_2 = pow(self.chosen_a, self.r // 2, self.N) + 1

        factor1 = math.gcd(guess_1, self.N)
        factor2 = math.gcd(guess_2, self.N)

        if factor1 not in [1, self.N]:
            print(f"[SUCCESS] Found factors: {factor1} and {self.N // factor1}")
            return factor1, self.N // factor1
        if factor2 not in [1, self.N]:
            print(f"[SUCCESS] Found factors: {factor2} and {self.N // factor2}")
            return factor2, self.N // factor2

        print(f"[INFO] Found trivial factors. Retrying...")
        return None


if __name__ == "__main__":
    N = 2222

    simulator = AerSimulator()
    shor = ShorAlgorithm(N, simulator=simulator)
    factors = shor.execute()

    print(f"\nFinal Result for N={N}: {factors}")

    if shor.qpe_circuit:
        print("\nCircuit Diagram(Last Attempt):")
        shor.qpe_circuit.draw(output='mpl', fold=-1, style="iqp")
        plt.show()
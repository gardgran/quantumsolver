"""
Shor's Algorithm Quantum Solver

This is a quantum solver simulation for shor's algorthim.
The quantum simulation is done during the Quantum Phase Estimation (QPE) to find the period.
Once the period is found. The factors of N are classically solved.

The quantum simulation only works (in a reasonable time frame) for small numbers ( < 7 bits). 
Above that, the program will not attempt to run.

George Lake - 12/2025

Notes about selecting N:
N=15 (4 bits): This case is special. It is set up to manually build a Quantum Circuit
so that the internal gates to the UGate are visibile. All other cases the UGate is a black box.

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
import qiskit.qasm2
from fractions import Fraction
import numpy as np
import random
import math
import sympy
import matplotlib.pyplot as plt
import argparse

class ShorCircuit(QuantumCircuit):
    """
    Generic Quantum Circuit for Shor's Algorithm
    Uses unitary matrices to implement modular exponentiation for any N
    """
    def __init__(self, a, N):
        self.n_target = N.bit_length()
        self.n_count = 2 * self.n_target 
        total_qubits = self.n_count + self.n_target 
        
        super().__init__(total_qubits, self.n_count)
        
        self.a = a
        self.N = N
        
        self._create_circuit()

    def _get_controlled_unitary_matrix(self, power_of_a):
        """
        Creates the matrix for the operation U^x |y> = |(a^x * y) mod N>.
        This is built manually as a UnitaryGate, and is a black box gate derived mathematically
        """
        dim_target = 2 ** self.n_target
        U_matrix = np.zeros((dim_target, dim_target), dtype=complex)
        
        # Calculate a^(power_of_a) mod N efficiently
        effective_multiplier = pow(self.a, power_of_a, self.N)
        
        for y in range(dim_target):
            if y < self.N:
                target_y = (effective_multiplier * y) % self.N
            else:
                target_y = y 
            U_matrix[target_y, y] = 1
            
        # Create the Controlled-U matrix
        CU_matrix = np.block([
            [np.eye(dim_target), np.zeros((dim_target, dim_target))],
            [np.zeros((dim_target, dim_target)), U_matrix]
        ])
            
        return UnitaryGate(CU_matrix, label=f"C-{self.a}^{power_of_a}")

    def _create_circuit(self):
        """
        Creates the quantum circuit for Shor's algorithm.
        """
        # 1. Initialize counting qubits to superposition |+>
        self.h(range(self.n_count))
        
        # 2. Initialize target register to |1> (eigenstate of the unitary operator)
        self.x(self.num_qubits - 1) 
        
        # 3. Apply Controlled-U operations (Phase Kickback)
        for i in range(self.n_count):
            power_of_a = 2**i
            CU_gate = self._get_controlled_unitary_matrix(power_of_a)
            
            target_qubits = list(range(self.n_count, self.num_qubits))
            control_qubit = i
            
            self.append(CU_gate, target_qubits + [control_qubit])

        # 4. Apply Inverse QFT to the counting qubits
        qft_gate = QFTGate(self.n_count).inverse()
        self.append(qft_gate, range(self.n_count))
        
        # 5. Measure the counting qubits
        self.measure(range(self.n_count), range(self.n_count))

    def run_simulation(self, simulator):
        """
        Transpiles and runs the circuit on the provided simulator.
        """
        transpiled_circuit = transpile(self, simulator)
        result = simulator.run(transpiled_circuit, shots=1, memory=True).result()
        return result
    

class ShorN15Circuit(QuantumCircuit):
    """
    Specific implementation for N=15, a=2
    Uses Controlled-SWAP gates to implement modular exponentiation
    """
    def __init__(self, a, N):
        # Safety check for this specific hardcoded circuit
        if N != 15 or a != 2:
            raise ValueError("ShorN15Circuit is strictly for N=15 and a=2.")

        self.n_target = 4
        self.n_count = 8
        total_qubits = self.n_count + self.n_target

        super().__init__(total_qubits, self.n_count)

        self.a = a
        self.N = N

        self._create_circuit()

    def _apply_manual_gates(self, ctrl_qubit, stage_index):
        """
        Applies U^(2^i) for N=15, a=2 using C-SWAP gates.
        Operation is multiplication by 2^(2^i) mod 15.
        """
        # Target qubits are the last 4
        t = list(range(self.n_count, self.num_qubits))
        
        # Stage 0: Multiplier = 2^(2^0) = 2
        # Permutation: 1->2->4->8->1 (Cyclic shift left by 1)
        if stage_index == 0:
            self.cswap(ctrl_qubit, t[2], t[3])
            self.cswap(ctrl_qubit, t[1], t[2])
            self.cswap(ctrl_qubit, t[0], t[1])
            
        # Stage 1: Multiplier = 2^(2^1) = 4
        # Permutation: 1->4->1, 2->8->2 (Two swaps)
        elif stage_index == 1:
            self.cswap(ctrl_qubit, t[0], t[2])
            self.cswap(ctrl_qubit, t[1], t[3])

    def _create_circuit(self):
        # 1. Initialize counting qubits
        self.h(range(self.n_count))
        
        # 2. Initialize target register to |1> (0001)
        self.x(self.num_qubits - 1) 
        
        # 3. Apply Controlled-U operations manually
        for i in range(self.n_count):
            self._apply_manual_gates(i, i)

        # 4. Apply Inverse QFT
        qft_gate = QFTGate(self.n_count).inverse()
        self.append(qft_gate, range(self.n_count))
        
        # 5. Measure counting qubits
        self.measure(range(self.n_count), range(self.n_count))

    def run_simulation(self, simulator):
        """
        Transpiles and runs the circuit on the provided simulator.
        """
        transpiled_circuit = transpile(self, simulator)
        result = simulator.run(transpiled_circuit, shots=1, memory=True).result()
        return result


class ShorAlgorithm:
    def __init__(self, N, circuit_class=ShorCircuit, max_attempts=-1, simulator=None):
        self.N = N
        self.circuit_class = circuit_class
        self.simulator = simulator if simulator else AerSimulator()
        self.max_attempts = max_attempts
        self.chosen_a = None
        self.r = None
        self.qpe_circuit = None

    def execute(self):
        print(f"--- Solving for N={self.N} ---")
        
        # 1. Trivial Factor Checks
        if self.N % 2 == 0: return 2, self.N // 2
        
        # 2. Filter valid 'a' candidates
        candidates = [a for a in range(2, self.N) if math.gcd(a, self.N) == 1]
        
        if not candidates:
            print("[Error] No coprime candidates found (N might be prime).")
            return None
            
        print(f'[INFO] Coprime candidates for a: {candidates}')

        if self.max_attempts > 0:
            limit = min(self.max_attempts, len(candidates))
        else:
            limit = len(candidates)

        attempts_count = 0

        while attempts_count < limit:
            attempts_count += 1
            print(f'\n[Attempt {attempts_count}/{limit}]')

            # Select 'a'
            if self.circuit_class == ShorN15Circuit:
                self.chosen_a = 2
                print(f"[INFO] N=15 Special Mode: Forcing a=2 for visualization.")
            else:
                self.chosen_a = random.choice(candidates)
                print(f'[Step 1] Chosen base a: {self.chosen_a}')

            # 3. Quantum Period Finding
            print(f'[Step 2] {self.chosen_a} is coprime to {self.N}. Generating Quantum Circuit...')
            
            # Run the circuit (Visualization object created here)
            success = self._quantum_period_finding()
            
            if success:
                # 4. Classical Post-Processing
                factors = self._classical_postprocess()
                if factors:
                    return factors
            
            # If we failed (either QPE failed or Factors were trivial), remove 'a' and try again
            if self.chosen_a in candidates:
                candidates.remove(self.chosen_a)
            
            # Check if we ran out of candidates prematurely
            if not candidates:
                print("[WARN] Ran out of unique candidates to try.")
                break
            
        print(f'[FAIL] Could not find factors after {attempts_count} attempts.')
        return None
    
    def _quantum_period_finding(self):
        # Build Circuit
        self.qpe_circuit = self.circuit_class(self.chosen_a, self.N)
        
        # Run Simulation
        try:
            result = self.qpe_circuit.run_simulation(self.simulator)
        except Exception as e:
            print(f"[ERR] Simulation error: {e}")
            return False

        # Parse Result
        readout = result.get_memory()[0] 
        state_dec = int(readout, 2)
        
        # Phase = measured_val / 2^n_count
        phase = state_dec / (2 ** self.qpe_circuit.n_count)
        
        # Continued Fractions to find 'r'
        frac = Fraction(phase).limit_denominator(self.N)
        self.r = frac.denominator
        
        print(f'   -> Measurement: |{readout}⟩ (Decimal: {state_dec})')
        print(f'   -> Calculated Phase: {phase:.4f} (~ {frac.numerator}/{frac.denominator})')
        print(f'   -> Estimated Period r: {self.r}')

        # Validation
        if self.r == 0:
            print("   -> Period r=0 is invalid.")
            return False
            
        if pow(self.chosen_a, self.r, self.N) != 1:
            print(f'   -> Check failed: {self.chosen_a}^{self.r} = {pow(self.chosen_a, self.r, self.N)} (mod {self.N}).')
            return False

        print(f'   -> Period Verified!')
        return True

    def _classical_postprocess(self):
        if self.r % 2 != 0:
            print(f'[Info] Period r={self.r} is odd. Cannot split N. Retrying...')
            return None

        # Guess factors using gcd(a^(r/2) ± 1, N)
        val = pow(self.chosen_a, self.r // 2, self.N)
        
        guess_1 = math.gcd(val - 1, self.N)
        guess_2 = math.gcd(val + 1, self.N)
        
        print(f'[Step 3] Derived guesses: gcd({val}-1, {self.N})={guess_1}, gcd({val}+1, {self.N})={guess_2}')

        if guess_1 not in [1, self.N]:
            print(f'[SUCCESS] Factors found: {guess_1} and {self.N // guess_1}')
            return guess_1, self.N // guess_1
        
        if guess_2 not in [1, self.N]:
            print(f'[SUCCESS] Factors found: {guess_2} and {self.N // guess_2}')
            return guess_2, self.N // guess_2
            
        print("[Info] Guesses were trivial factors (1 or N). Retrying...")
        return None
    
def solve(N: int) -> dict:
    """
    Main entry point
    Determine which case is given by the input

    Returns a JSON-ready dictionary with:
    {
        "answer": "...",
        "qasm" : "..."    
    }
    """
    if not isinstance(N, int):
        return {"answer": "Input must be an integer", "qasm": "NA"}
    
    n_bits = N.bit_length()
    
    if n_bits > 9:
        return {"answer": "Bit Length too Long", "qasm": "NA"}  
    
    if N == 15:
        selected_circuit = ShorN15Circuit
    else:
        selected_circuit = ShorCircuit

    simulator = AerSimulator()
    shor = ShorAlgorithm(N, circuit_class=selected_circuit, simulator=simulator)
    factors = shor.execute()

    # Check if quantum circuit was made
    if shor.qpe_circuit:
        try:
            qasm_string = qiskit.qasm2.dumps(shor.qpe_circuit)
        except Exception:
            qasm_string = "Error generating QASM"

    return {"answer": factors, "qasm": qasm_string}



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Shor's Algorithm Quantum Solver")
    parser.add_argument('N', type=int, nargs='?', default=15, help="The integer N to factor (default: 15)")
    args = parser.parse_args()

    N = args.N

    # Calculate bit length
    n_bits = N.bit_length()

    if n_bits > 9:
        print(f"\n[ERROR] Input N={N} is {n_bits} bits long")
        print(f"To prevent program crashing, this algorithm is limited to 9 bits")
        exit(1)
    
    # Select the circuit class
    if N == 15:
        selected_circuit = ShorN15Circuit
        print(f"--- N={N} Detected: Using Specialized ShorN15Circuit (Visual Gates) ---")
    else:
        selected_circuit = ShorCircuit
        print(f"--- N={N} Detected: Using Generic ShorCircuit (Matrix Blocks) ---")
    
    simulator = AerSimulator()
    
    shor = ShorAlgorithm(N, circuit_class=selected_circuit, simulator=simulator)
    factors = shor.execute()
    
    print(f"\nFinal Result: Factors = {factors}")
    
    # Visualize the circuit from the last attempt
    if shor.qpe_circuit:
        print("\nCircuit Diagram (Last Attempt):")
        shor.qpe_circuit.draw(output='mpl', fold=-1, style="iqp")
        plt.show()
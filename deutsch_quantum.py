"""
Deutsch Quantum Solver

Input - dictionary
{'case' : 'f1'} (f1, f2, f3, f4)
TODO: Coordinate how Problem group will send oracle
Code will need to be modified based on that


Output - dictionary
{'answer': 'constant' or 'balanced'}
TODO: Coordinate what information Visualization group needs,
Possible to send quantum circuit to them.
"""

import io
import base64
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def deutsch_oracle(case: str) -> QuantumCircuit:
    """
    2-qubit oracle U_f acting on |x,y> where y is the target.
    This is just for testing purposes,
    for the project this will be provided by the problem team.
    """
    oracle = QuantumCircuit(2)
    
    if case == "f1":    # constant 0: do nothing
        pass
    elif case == "f2":  # constant 1: flip the target bit
        oracle.x(1)
    elif case == "f3":  # balanced: f(x) = x, use CNOT
        oracle.cx(0,1)
    elif case == "f4":  # balanced: f(x) = not x
        oracle.x(0)
        oracle.cx(0, 1)
        oracle.x(0)
    else:
        raise ValueError(f"Unknown case '{case}'. Use one of f1, f2, f3, f4.")
    return oracle

def deutsch_solver(case: str, shots: int = 1024) -> dict:
    """
    Run Deutsch's algorithm for the given oracle case and return the dictionary
    {'answer': 'constant'|'balanced'} 
    """
    # create circuit with 2 qubits and 1 classial bit.
    qc = QuantumCircuit(2, 1)

    # prepare |1>, and put both qubits in superpostition
    qc.x(1)
    qc.h(0)
    qc.h(1)

    # apply the oracle
    qc.compose(deutsch_oracle(case), inplace=True)

    # hadamard on the first qubit and measure
    qc.h(0)
    qc.measure(0,0)

    # run on simulator - note some code he is overkill for a simulator, 
    # but included in the event it would run on an actual QPC.
    sim = AerSimulator(seed_simulator=42)
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Majority vote on the measured bit of qubit-0
    bit = max(counts, key=counts.get)   # '0' or '1'
    answer = "constant" if bit == '0' else "balanced"

    # Create Diagram 
    # uncomment here and in return statement to implement
    # png_b64 = circuit_png_base64(qc)

    # Return
    return {
        "answer": answer,
        # "diagram_format": "png_base64",
        # "diagram_png": png_b64
        }

def circuit_png_base64(qc: QuantumCircuit) -> str:
    """
    Create diagram of circuit
    Currently just for fun and testing
    Output is a 64bit encoding. It works with Postman
    """
    fig = qc.draw(output='mpl')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

def solve(data) -> dict:
    """
    Main entry point
    Interface between deutsch solver and flask
    """
    case = data.get("case")
    if case not in {"f1", "f2", "f3", "f4"}:
        return {"error": "invalid or missing case (use f1,f2,f3,f4)"}
    return deutsch_solver(case)



"""
TODO: right now it implements via selecting which case to send.
Research how we will get the information via the problem group.
"""
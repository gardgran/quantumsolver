"""
Deutsch Quantum Solver

Input - list
Representing the states of 2 bits
[0,0], [0,1], [1,0], [1,1]
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

def deutsch_function(case: int) -> QuantumCircuit:
    """
    Creates a quantum circuit just for the oracle
    Depending on the state given:
    f1 - Do nothing
    f2 - CNOT
    f3 - CNOT, Not
    f4 - Not
    """
    if case not in [1, 2, 3, 4]:
        raise ValueError("'case' must be 1, 2, 3, or 4")
    
    f = QuantumCircuit(2)
    if case in [2, 3]:
        f.cx(0, 1)
    if case in [3, 4]:
        f.x(1)
    return f

def compile_circuit(function: QuantumCircuit) -> QuantumCircuit:
    """
    Builds the circuit around the oracle
    """
    # create circuit with 2 qubits and 1 classial bit.
    n = function.num_qubits - 1
    qc = QuantumCircuit(n + 1, n)

    # prepare |1>, and put both qubits in superpostition
    qc.x(n)
    qc.h(range(n + 1))

    # apply the oracle
    qc.compose(function, inplace=True)

    # hadamard on the first qubit and measure
    qc.h(range(n))
    qc.measure(range(n),range(n))

    return qc

def deutsch_algorithm(function: QuantumCircuit) -> dict:
    """
    Runs the compiled circuit (with oracle) in the Qiskit simulator
    Returns constant if measurements are the same
    Returns balanced if measurements are opposite
    """
    qc = compile_circuit(function)

    result = AerSimulator().run(qc, shots=1, memory=True).result()
    measurements = result.get_memory()
    answer = "constant" if measurements[0] == '0' else "balanced"
    return {"answer": answer}

def circuit_png_base64(qc: QuantumCircuit) -> str:
    """
    Create diagram of circuit
    Currently just for fun and testing
    """
    fig = qc.draw(output='mpl')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=200)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return data

def solve(data: list) -> dict:
    """
    Main entry point
    Determine which case is given by the input
    """
    
    if data[0]:
        if data[1]:
            f = deutsch_function(4)  # [1, 1] f4
        else:
            f = deutsch_function(3)  # [1, 0] f3
    else:
        if data[1]:
            f = deutsch_function(2)  # [0, 1] f2
        else:
            f = deutsch_function(1)  # [0, 0] f1 
    return(deutsch_algorithm(f))


if __name__ == "__main__":
    assert solve([0, 0])["answer"] == "constant"
    assert solve([1, 1])["answer"] == "constant"
    assert solve([0, 1])["answer"] == "balanced"
    assert solve([1, 0])["answer"] == "balanced"
    print("All tests passed")


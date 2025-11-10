"""This is a classical solver for the Deutsch problem"""


def solve(data: list) -> dict:
    """We expect the incoming json schema to be a list of length two.
    Each entry is either 0 or 1. The first number represents
    f(0), and the second represents f(1)
    """

    # convert input data to booleans
    data = [bool(x) for x in data]

    def f(x: bool) -> bool:
        """This is the hidden function f we are trying to analyze"""
        return data[1] if x else data[0]

    # f has been defined. Now we treat it as a black box.
    # Past this point, we pretend that we don't know which function
    # it is and we try to determine if it is constant or balanced.

    result = "constant" if f(False) == f(True) else "balanced"
    return {"answer": result}


if __name__ == "__main__":
    assert solve([1, 1])["answer"] == "constant"
    assert solve([0, 0])["answer"] == "constant"
    assert solve([1, 0])["answer"] == "balanced"
    assert solve([0, 1])["answer"] == "balanced"
    print("All tests passed")

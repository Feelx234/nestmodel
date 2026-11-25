import numpy as np
from nestmodel import _core


def g6_read_bytes(p):
    """Read the file at path p into a byte array"""
    with open(p, "rb") as file:
        lines = file.read()
        arr = np.fromiter(lines, np.uint8)
    with open(p, "rb") as f:
        first_line = f.readline()
    line_length = len(first_line)
    m, r = divmod(len(arr), line_length)
    if r != 0:
        raise ValueError("not all lines have the same length")
    arr = arr.reshape(m, line_length)
    assert np.all(arr[:, line_length - 1] == 10)  # 10 in ASCII is linefeed
    arr = arr[:, : line_length - 1]
    return arr


def g6_bytes_to_edges(input_arr):
    """Convert a g6 bytes array into a list of graph edges"""
    arr = np.ascontiguousarray(input_arr, dtype=np.int64) - 63
    return _core.g6_bytes_to_edges(arr)

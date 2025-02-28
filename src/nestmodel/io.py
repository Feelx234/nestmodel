import numpy as np
from numba import njit, bool_


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


@njit(cache=True)
def _g6_data_to_n(data):
    """Read initial one-, four- or eight-unit value from graph6
    integer sequence.

    Return (value, rest of seq.)

    From https://networkx.org/documentation/stable/_modules/networkx/readwrite/graph6.html
    """
    if data[0] <= 62:
        return data[0], data[1:]
    if data[1] <= 62:
        return (data[1] << 12) + (data[2] << 6) + data[3], data[4:]
    return (
        (data[2] << 30)
        + (data[3] << 24)
        + (data[4] << 18)
        + (data[5] << 12)
        + (data[6] << 6)
        + data[7],
        data[8:],
    )


@njit(cache=True)
def g6_bytes_to_edges(input_arr):
    """Convert a g6 bytes array into a list of graph edges"""
    arr = input_arr - 63
    out = []

    print(arr.shape[0])
    bits_buf = np.empty(arr.shape[1] * 6, dtype=bool_)
    for i in range(arr.shape[0]):
        edges = _g6_line_to_edges(arr[i, :], bits_buf)
        out.append(edges)
    return out


@njit(cache=True)
def _g6_bits(data, bits_buf):
    """Returns sequence of individual bits from 6-bit-per-value
    list of data values. From https://networkx.org/documentation/stable/_modules/networkx/readwrite/graph6.html#read_graph6
    """
    n = 0
    for d in data:
        for i in [5, 4, 3, 2, 1, 0]:
            bits_buf[n] = (d >> i) & 1
            n += 1


@njit(cache=True)
def _g6_line_to_edges(input_line, bits_buf):
    """Convert the g6 byte representation on the input line into an array of edges"""
    n, data = _g6_data_to_n(input_line)
    _g6_bits(data, bits_buf)
    m = 0
    edges = []
    for j in range(1, n):
        for i in range(j):
            if bits_buf[m]:
                edges.append((i, j))
            m += 1
    return np.array(edges, dtype=np.int16)

import sys
import shutil
from pathlib import Path


py_files = [
    "centralities.py",
    "fast_graph.py",
    "unified_functions.py",
    "fast_rewire.py",
    "fast_rewire2.py",
    "fast_wl.py",
    "utils.py",
    "wl.py",
    "load_datasets.py",
    "tests/test_centralities.py",
    "tests/test_fast_wl.py",
    "tests/test_utils.py",
    "tests/test_pagerank.py",
    "tests/testing.py",
    "ERGM_experiments.py",
    "dict_graph.py",
    "ergm.py",
]


script_files = [
    "convergence2_katz.ipynb",
    "convergence2_jaccard.ipynb",
    "convergence2_eigenvector.ipynb",
    "convergence2_hits.ipynb",
    "convergence2_eigenvector.ipynb",
    "convergence2_pagerank.ipynb",
    "convergence2_plot.ipynb",
    "convergence_helper.py",
    "convergence2_runner.py",
    "convergence2_example.ipynb",
    "get_datasets.py",
    "Baselines.ipynb",
    "Baselines_plot.ipynb",
]

other_files = [
    "setup.py",
    ".gitignore",
    ".pylintrc",
]


def inplace_change(filename, old_string, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            # print('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # Safely write the changed content, if found in the file
    with open(filename, "w") as f:
        # print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


def copy_py_file(origin, target, files):
    print(f"copying from {origin} to {target}")
    origin = Path(origin).absolute()
    target = Path(target).absolute()
    assert origin.exists(), origin
    assert target.exists(), target
    for file in files:
        shutil.copyfile(origin / file, target / file)
        inplace_change(target / file, "cc_model", "nestmodel")


if __name__ == "__main__":
    assert len(sys.argv) == 3, "no parameters provided"
    # print(f"Arguments count: {len(sys.argv)}")
    for i, arg in enumerate(sys.argv):
        print(f"Argument {i:>6}: {arg}")
    origin = Path(sys.argv[1]).resolve()
    target = Path(sys.argv[2]).resolve()
    copy_py_file(origin, target, py_files)
    copy_py_file(origin.parent / "scripts", target.parent / "scripts", script_files)
    copy_py_file(origin.parent, target.parent, other_files)


# python copy_files.py ../colorful_configuration/cc_model ./nestmodel

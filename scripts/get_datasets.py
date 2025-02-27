# pylint: disable=missing-function-docstring, invalid-name
import subprocess
from pathlib import Path
import os


hep_link = r"https://snap.stanford.edu/data/cit-HepPh.txt.gz"
astro_link = r"https://snap.stanford.edu/data/ca-AstroPh.txt.gz"
# http://networksciencebook.com/translations/en/resources/data.html
networksciencebook_link = (
    r"https://networksciencebook.com/translations/en/resources/networks.zip"
)
google_link = r"https://snap.stanford.edu/data/web-Google.txt.gz"
pokec_link = r"https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

networkscience_files = [
    "collaboration.edgelist.txt",
    "powergrid.edgelist.txt",
    "actor.edgelist.txt",
    "www.edgelist.txt",
    "phonecalls.edgelist.txt",
    "internet.edgelist.txt",
    "metabolic.edgelist.txt",
    "email.edgelist.txt",
    "citation.edgelist.txt",
    "protein.edgelist.txt",
]
print(Path(__file__).resolve())
print(Path(__file__).parent.absolute())


parent = Path(__file__).parent.absolute()
print(parent.name)
if str(parent.name) == "scripts":
    parent = parent.parent
if str(parent.name) != "datasets":
    parent = parent / "datasets"

if not parent.is_dir():
    print("creating ", parent)
    parent.mkdir(exist_ok=True)
# assert parent.exists(), str(parent)
links = [hep_link, astro_link, networksciencebook_link, google_link, pokec_link]
download_names = [
    "cit-HepPh.txt.gz",
    "ca-AstroPh.txt.gz",
    "networks.zip",
    "web-Google.txt.gz",
    "soc-pokec-relationships.txt.gz",
]
final_files = [
    ("cit-HepPh.txt",),
    ("ca-AstroPh.txt",),
    networkscience_files,
    ("web-Google.txt",),
    ("soc-pokec-relationships.txt",),
]

combinatorical_prefix = "https://users.cecs.anu.edu.au/~bdm/data/"
combinatorial_names = [f"ge{i}d1.g6" for i in range(2, 16)]
combinatorial_links = [combinatorical_prefix + name for name in combinatorial_names]
combinatorial_final = [(name, name[:-3] + ".npy") for name in combinatorial_names]
links.extend(combinatorial_links)
download_names.extend(combinatorial_names)
final_files.extend(combinatorial_final)

if os.name == "nt":
    # if you want to know more about the download command for windows use
    # https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/invoke-webrequest?view=powershell-7.3
    def download_command_windows(_link, _parent, file):
        return "powershell wget " + "-Uri " + _link + " -OutFile " + str(_parent / file)

    dowload_command = download_command_windows

    def unzip_command_windows(_parent, file):
        return "unzip " + str(_parent / file) + " -d " + str((parent / file).stem)

    unzip_command = unzip_command_windows
else:

    def download_command_linux(_link, _parent, file):  # pylint: disable=unused-argument
        return "wget " + _link + " -P " + str(_parent)

    dowload_command = download_command_linux

    def unzip_command_linux(_parent, file):  # pylint: disable=unused-argument
        return "unzip " + str(file)

    unzip_command = unzip_command_linux


def process_g6(path):
    import networkx as nx  # pylint: disable=import-outside-toplevel
    import numpy as np  # pylint: disable=import-outside-toplevel

    Graphs = nx.readwrite.read_graph6(path)
    print(Graphs[0])
    out = [np.array([e for e in G.edges], dtype=np.int32) for G in Graphs]
    out = np.array(out, dtype=np.int32)
    new_path = Path(path).resolve().with_suffix(".npy")
    print("saving numpy array to", new_path)
    np.save(str(new_path), out, allow_pickle=False)


if True:
    files = list(file.name for file in parent.iterdir())
    for link, download_name, finals in zip(links, download_names, final_files):
        if all(final in files for final in finals):
            continue

        if download_name not in files:
            # download file
            command = dowload_command(link, parent, download_name)
            print()
            print("<<< downloading " + download_name)
            print(command)
            subprocess.call(command, shell=True, cwd=str(parent))

        if download_name.endswith(".gz"):
            command = "gzip -d " + str(download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True, cwd=str(parent))

        if download_name.endswith(".zip"):
            command = unzip_command(parent, download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True, cwd=str(parent))

        if download_name.endswith(".g6"):
            process_g6(Path(parent) / download_name)

print()
print("done")
print()

dataset_path_file = Path(__file__).parent.absolute() / "datasets_path.txt"
if not dataset_path_file.is_file():
    with open(dataset_path_file, "w", encoding="utf-8") as f:
        f.write(str(parent))

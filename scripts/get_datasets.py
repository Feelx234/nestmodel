
import subprocess
from pathlib import Path



hep_link = r"http://snap.stanford.edu/data/cit-HepPh.txt.gz"
astro_link = r"http://snap.stanford.edu/data/ca-AstroPh.txt.gz"
# http://networksciencebook.com/translations/en/resources/data.html
networksciencebook_link = r"http://networksciencebook.com/translations/en/resources/networks.zip"
google_link = r"https://snap.stanford.edu/data/web-Google.txt.gz"
pokec_link = r"https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"

networkscience_files = ['collaboration.edgelist.txt', 'powergrid.edgelist.txt', 'actor.edgelist.txt', 'www.edgelist.txt', 'phonecalls.edgelist.txt', 'internet.edgelist.txt', 'metabolic.edgelist.txt', 'email.edgelist.txt', 'citation.edgelist.txt', 'protein.edgelist.txt']
print(Path(__file__).resolve())
print(Path(__file__).parent.absolute())


parent = Path(__file__).parent.absolute()
print(parent.name)
if str(parent.name) == "scripts":
    parent = parent.parent
if str(parent.name)!="datasets":
    parent = parent/"datasets"

if not parent.is_dir():
    print("creating ", parent)
    parent.mkdir(exist_ok=True)
#assert parent.exists(), str(parent)
links = [hep_link, astro_link, networksciencebook_link, google_link, pokec_link]
download_names = ["cit-HepPh.txt.gz", "ca-AstroPh.txt.gz", "networks.zip", "web-Google.txt.gz", "soc-pokec-relationships.txt.gz"]
final_files = [("cit-HepPh.txt",), ("ca-AstroPh.txt",), networkscience_files, ("web-Google.txt",), ("soc-pokec-relationships.txt",)]


if True:
    files = list(file.name for file in parent.iterdir())
    for link, download_name, finals in zip(links, download_names, final_files):
        if all(final in files for final in finals):
            continue

        if not download_name in files:
            # download file
            command = "wget " + link +" -P " + str(parent)
            print()
            print("<<< downloading " + download_name)
            subprocess.call(command, shell=True, cwd=str(parent))

        if download_name.endswith(".gz"):
            command = "gzip -d " + str(download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True , cwd=str(parent))

        if download_name.endswith(".zip"):
            command = "unzip " + str(download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True , cwd=str(parent))

print()
print("done")
print()

dataset_path_file = Path(__file__).parent.absolute()/"datasets_path.txt"
if not dataset_path_file.is_file():
    with open(dataset_path_file, "w") as f:
        f.write(str(parent))
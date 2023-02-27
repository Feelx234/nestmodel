from pathlib import Path
import numpy as np
import pandas as pd
from nestmodel.utils import graph_tool_from_edges
from nestmodel.fast_graph import FastGraph

def relabel_edges(edges):
    """relabels nodes such that they start from 0 consecutively"""
    unique = np.unique(edges.ravel())
    mapping = {key:val for key, val in zip(unique, range(len(unique)))}
    out_edges = np.empty_like(edges)
    for i,(e1,e2) in enumerate(edges):
        out_edges[i,0] = mapping[e1]
        out_edges[i,1] = mapping[e2]
    return out_edges

def check_is_directed(edges):
    """Checks whether for all edges u-v the edge v-u is also in edges"""
    d = {(a,b) for a,b in edges}
    for a,b in edges:
        assert (b,a) in d

class Dataset:
    """Simple structure to store information on datasets"""
    def __init__(self, name, file_name, is_directed=False, delimiter=None):
        self.name=name
        self.file_name = file_name
        self.get_edges = self.get_edges_pandas
        self.skip_rows = 0
        self.is_directed=is_directed
        self.delimiter = delimiter
        self.requires_node_renaming=False



    def get_edges_pandas(self, datasets_dir):
        """Reads edges using pands read_csv function"""
        df = pd.read_csv(datasets_dir/self.file_name, skiprows=self.skip_rows, header=None, sep=self.delimiter)
        edges = np.array([df[0].to_numpy(), df[1].to_numpy()],dtype=np.uint64).T


        if self.requires_node_renaming:
            return relabel_edges(edges)
        else:
            return edges

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        elif isinstance(other, Dataset):
            return self.name==other.name
        else:
            raise ValueError()

    def get_edges_karate(self, datasets_dir): # pylint: disable=unused-argument, missing-function-docstring
        import networkx as nx # pylint: disable=import-outside-toplevel
        G = nx.karate_club_graph()
        edges = np.array(list(G.edges), dtype=int)
        return edges

Phonecalls = Dataset("phonecalls", "phonecalls.edgelist.txt", delimiter="\t")

AstroPh = Dataset("AstroPh", "ca-AstroPh.txt", delimiter="\t", is_directed=False)
AstroPh.skip_rows=4
AstroPh.requires_node_renaming=True

HepPh = Dataset("HepPh", "cit-HepPh.txt", delimiter="\t", is_directed=True)
HepPh.skip_rows=4
HepPh.requires_node_renaming=True

Karate = Dataset("karate", "karate")
Karate.get_edges = Karate.get_edges_karate

Google= Dataset("web-Google", "web-Google.txt", delimiter="\t", is_directed=True)
Google.skip_rows=4
Google.requires_node_renaming=True

Pokec= Dataset("soc-Pokec", "soc-pokec-relationships.txt", delimiter="\t", is_directed=True)
Pokec.skip_rows=0
Pokec.requires_node_renaming=True

Netscience= Dataset("netscience", "ca-netscience.edges", delimiter=" ", is_directed=False)
Netscience.skip_rows=0
Netscience.requires_node_renaming=True

all_datasets = [Karate, Phonecalls, AstroPh, HepPh, Google, Pokec, Netscience]

def find_dataset(dataset_name):
    """Finds dataset object by dataset_name as str"""
    dataset = None
    for potential_dataset in all_datasets:
        if potential_dataset == dataset_name:
            dataset = potential_dataset
            break
    assert not dataset is None, f"You have specified an unknown dataset {dataset}"
    return dataset

def load_fast_graph(dataset_path, dataset, verbosity=0):
    """Loads the dataset as specified by the dataset_str"""
    g_base = load_gt_dataset_cached(dataset_path,
                                    dataset,
                                    verbosity=verbosity,
                                    force_reload=True)
    edges = np.array(g_base.get_edges(), dtype=np.uint32)

    G = FastGraph(edges, g_base.is_directed())
    return G

def load_dataset(datasets_dir, dataset_name):
    """Loads dataset in as edge_list """
    #"deezer_HR", "deezer_HU", "deezer_RO","tw_musae_DE",
    #            "tw_musae_ENGB","tw_musae_FR","lastfm_asia","fb_ath",
    #            "fb_pol","phonecalls", "facebook_sc"]

    dataset = find_dataset(dataset_name)
    edges = dataset.get_edges(datasets_dir)

    if dataset.is_directed is False:
        edges = edges[edges[:,0] < edges[:,1],:]
        #[(e1, e2) for e1, e2 in edges if e1 < e2]
    #print("A", dataset.is_directed)
    return edges, dataset.is_directed

def get_datasets_path():
    """Try to read dataset path from file"""
    folders = [".", "./scripts", "./nest_model/scripts"]
    for folder in folders:
        p = Path(folder)/"datasets_path.txt"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                return Path(f.read())

    raise ValueError("Could not find datasets_path.txt")


def load_fg_dataset_cached(datasets_dir, dataset_name, verbosity=0, force_reload=False):
    """Loads a dataset using the binary file format from graph-tool"""
    if datasets_dir is None:
        datasets_dir = get_datasets_path()
    else:
        datasets_dir = Path(datasets_dir)
    dataset = find_dataset(dataset_name)
    cache_file = datasets_dir/(dataset.file_name+".npz")
    if cache_file.is_file() and not force_reload:
        if verbosity>1:
            print("loading cached")
        npzfile = np.load(cache_file)
        return FastGraph(npzfile["edges"], bool(npzfile["is_directed"]))
    else:
        if verbosity>1:
            print("loading raw")
        edges, is_directed = load_dataset(datasets_dir, dataset_name)
        if edges.max() < np.iinfo(np.uint32).max:
            edges = edges.astype(np.uint32)
        print(edges.dtype)
        g = FastGraph(edges, is_directed)
        g.save_npz(str(cache_file.absolute()))
        return g




def load_gt_dataset_cached(datasets_dir, dataset_name, verbosity=0, force_reload=False):
    """Loads a dataset using the binary file format from graph-tool"""
    dataset = find_dataset(dataset_name)
    cache_file = datasets_dir/(dataset.file_name+".gt")
    if cache_file.is_file() and not force_reload:
        if verbosity>1:
            print("loading cached")
        import graph_tool.all as gt # pylint: disable=import-outside-toplevel # type: ignore
        return gt.load_graph(str(cache_file.absolute()))
    else:
        if verbosity>1:
            print("loading raw")
        edges, is_directed = load_dataset(datasets_dir, dataset_name)
        g = graph_tool_from_edges(edges, None, is_directed=is_directed)
        g.save(str(cache_file.absolute()))
        return g

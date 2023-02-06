# pylint: disable=missing-function-docstring, missing-class-docstring
from itertools import chain, repeat
from datetime import datetime
from pathlib import Path
import sys
import warnings
import pickle

import numpy as np
import pandas as pd

from nestmodel.load_datasets import load_fg_dataset_cached

class ParameterWrapper:
    def __init__(self, dataset_path, cent_func, cent_kwargs, rewire_kwargs, wl_kwargs, number_of_samples):
        self.dataset_path=dataset_path
        self._cent_func=cent_func
        self.cent_kwargs=cent_kwargs
        self.rewire_kwargs=rewire_kwargs
        self.wl_kwargs=wl_kwargs
        self.verbosity=1
        self.force_reload=False
        self.number_of_samples=number_of_samples
        self.tqdm=None

    def cent_func(self, *args, **kwargs):
        centralities = self._cent_func(*args, **kwargs)
        if isinstance(centralities, tuple):
            return centralities
        else:
            return (centralities,)

    def important_to_dict(self):
        return {
            "cent_kwargs" : self.cent_kwargs,
            "rewire_kwargs":self.rewire_kwargs,
            "wl_kwargs" : self.wl_kwargs,
            "number_of_samples": self.number_of_samples
        }

    def range_over_samples(self):
        the_range = range(self.number_of_samples)
        if self.tqdm:
            try:
                from tqdm.auto import tqdm
                the_range = tqdm(the_range, leave=False, desc="samples")
            except ModuleNotFoundError:
                warnings.warn("Could not find tqdm, not displaying progressbar")
        return the_range
    def wl_range(self, wl_iterations):
        the_range =  range(wl_iterations-1,-1,-1)
        if self.tqdm:
            try:
                from tqdm.auto import tqdm
                the_range = tqdm(the_range, desc="wl_rounds", leave=False)
            except ModuleNotFoundError:
                warnings.warn("Could not find tqdm, not displaying progressbar")
        return the_range




def SAE(v0, v1):
    return np.sum(np.abs(v0-v1))

class Tracker:
    def __init__(self, tag=None, func=SAE, base_centrality=None):
        self.data = []
        self.curr_round=None
        self.curr_dataset=None
        self.tag=tag
        self.func=func
        self.set_base_centrality(base_centrality)

    def set_base_centrality(self, base_centrality):
        self.base_centrality = base_centrality

    def new_round(self, round_id):
        self.curr_round=round_id


    def new_dataset(self, dataset_id):
        self.curr_round=None
        self.curr_dataset=dataset_id

    def add_centrality(self, centrality):
        assert not self.curr_round is None
        assert not self.curr_dataset is None

        tpl = (self.curr_dataset, self.curr_round, self.func(centrality, self.base_centrality), self.tag)
        self.data.append(tpl)



class MultiTracker:
    def __init__(self, tags=None, funcs=None):
        if funcs is None:
            funcs = repeat(SAE)
        else:
            assert len(funcs)==len(tags)
        self.trackers = [Tracker(tag=tag, func=func) for tag ,func in zip(tags, funcs)]
        self.verbosity = 0

    def set_base_centrality(self, *centralities):
        assert len(centralities) == len(self.trackers)
        for tracker, centrality in zip(self.trackers, centralities):
            tracker.set_base_centrality(centrality)

    def new_round(self, round_id):
        if self.verbosity > 1:
            print("\tround:", round_id)
        for tracker in self.trackers:
            tracker.new_round(round_id)

    def new_dataset(self, dataset_id):
        if self.verbosity > 0:
            print("dataset:", dataset_id)
        for tracker in self.trackers:
            tracker.new_dataset(dataset_id)

    def add_centrality(self, *centralities):

        assert len(centralities) == len(self.trackers)
        for tracker, centrality in zip(self.trackers, centralities):
            tracker.add_centrality(centrality)

    @property
    def data(self):
        return chain.from_iterable(tracker.data for tracker in self.trackers)

    def data_to_pandas(self):
        return pd.DataFrame.from_records(list(self.data), columns=("dataset", "wl_round", "value", "tag"))
MultiTracker.to_df = MultiTracker.data_to_pandas
MultiTracker.track = MultiTracker.add_centrality


def load_dataset(dataset, params):
    return load_fg_dataset_cached(params.dataset_path,
                                    dataset,
                                    verbosity=params.verbosity,
                                    force_reload=params.force_reload)

def process_dataset(dataset, tracker, params):
    tracker.new_dataset(dataset)
    G = load_dataset(dataset, params)

    G.ensure_edges_prepared(**params.wl_kwargs)

    tracker.set_base_centrality(*params.cent_func(G, **params.cent_kwargs))

    print("\ttotal WL iterations ", G.wl_iterations)
    for wl_round in params.wl_range(G.wl_iterations):
        tracker.new_round(wl_round)

        for _ in params.range_over_samples():
            G.rewire(wl_round, **params.rewire_kwargs)
            tracker.track(*params.cent_func(G, **params.cent_kwargs))


def compute_on_all_datasets(datasets, tracker, params):
    """ computes
    """

    for dataset in datasets:
        process_dataset(dataset, tracker, params)


def save_results(prefix, tracker, params, min_samples=50):
    now = datetime.now()
    if params.number_of_samples >= min_samples:
        date_suffix = now.strftime("%Y_%m_%d__%H_%M_%S")
        folder = Path("./results/")
        if not folder.exists():
            folder = Path("./scripts/results/")
        out_name = folder/(f"{prefix}_"+date_suffix+".pkl")

        data = tracker.data_to_pandas()
        if len(data)<min_samples:
            return

        print(out_name)
        params_dict = params.important_to_dict()
        with open(out_name, "wb") as f:
            pickle.dump((data, params_dict), f)

def get_datasets(datasets):
    if "-all" in sys.argv:
        datasets = ["karate",
            "phonecalls",
            "HepPh",
            "AstroPh",
            "web-Google",
            "soc-Pokec"
           ]
    elif "datasets" in sys.argv:
        import json

        i = sys.argv.index("datasets")+1
        if i < len(sys.argv):
            print("datasets="+sys.argv[i])
            datasets = json.loads(sys.argv[i].replace("'", '"'))

    return datasets

def get_samples(samples):
    if "n" in sys.argv:
        i = sys.argv.index("n")+1
        if i < len(sys.argv):
            samples = int(sys.argv[i])
    return samples
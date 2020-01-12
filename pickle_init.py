import pickle
import scipy
import os
import imageio

class PickleList():
    def __init__(self, list_to_pickle):
        self.list_to_pickle = list_to_pickle

    def get_list(self):
        return self.list_to_pickle

def pickle_it(pickle_list, picklepath):
    pkl_list = PickleList(pickle_list)
    with open(picklepath, "wb") as f:
        pickle.dump(pkl_list, f)

from interface import Regressor
from utils import get_data, create_ui_matrix
from tqdm import tqdm
from config import *
import pickle
import numpy as np
from os import path

class SlopeOne(Regressor):
    def __init__(self):
        self.popularity_differences = {}
        self.ui_mtx = None

    def fit(self, X: np.array):
        # check if params are already uploaded
        if path.exists(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH):
            self.upload_params()
            self.ui_mtx = create_ui_matrix(X)
        else:
            self.save_params(X)

    def build_popularity_difference_dict(self, data):
        # build user item matrix
        self.ui_mtx = create_ui_matrix(data)
        # loop over all items -> I*(I-1)/2
        for i in tqdm(range(self.ui_mtx.shape[1])):
            for j in range(i+1, self.ui_mtx.shape[1]):
                ij_mtx = self.ui_mtx[:, [i,j]].copy()
                # get rows where both items are rated
                ij_mtx = ij_mtx[(ij_mtx > 0).all(axis=1)]
                # num of users who rated both items
                C_ij = ij_mtx.shape[0]
                # calculate mean difference
                if C_ij > 0:
                    PD_ij = (ij_mtx[:, 0] - ij_mtx[:, 1]).mean().astype(np.float16)
                    self.popularity_differences[(i, j)] = (PD_ij, C_ij)




    def predict_on_pair(self, user: int, item: int):
        r_ui = 0
        total_C = 0
        for v in range(self.ui_mtx.shape[1]):
            # take only items the user rated
            if v != item and self.ui_mtx[user,v] > 0:
                # get the popularity difference, and the number of users who rated both items
                if item < v:
                    PD, C = self.popularity_differences.get((item, v), (0, 0))
                else:
                    PD, C = self.popularity_differences.get((v, item), (0, 0))
                    PD = -PD

                r_ui += (PD + self.ui_mtx[user,v]) * C
                total_C += C

        return r_ui / total_C

    def upload_params(self):
        # read existing pickle file with params
        with open(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH, 'rb') as f:
            self.popularity_differences = pickle.load(f)

    def save_params(self,data):
        self.build_popularity_difference_dict(data)
        # save to pickle file for later use
        with open(POPULARITY_DIFFERENCES_PARAMS_FILE_PATH, 'wb') as f:
            pickle.dump(self.popularity_differences, f)
    

if __name__ == '__main__':
    train, validation = get_data()
    slope_one = SlopeOne()
    slope_one.fit(train)
    print(slope_one.calculate_rmse(validation))

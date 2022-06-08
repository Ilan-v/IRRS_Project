from interface import Regressor
from utils import get_data, create_ui_matrix
from tqdm import tqdm
from config import *
import pickle
import numpy as np

class SlopeOne(Regressor):
    def __init__(self):
        self.popularity_differences = {}
        self.ui_mtx = None

    def fit(self, X: np.array):
       raise NotImplementedError

    def build_popularity_difference_dict(self, data):
        # build user item matrix
        if self.ui_mtx is None:
            self.ui_mtx = create_ui_matrix(data)

        for i in tqdm(range(self.ui_matrix.shape[0])):
            for j in range(self.ui_matrix.shape[1]):
                if i!=j:
                    ij_mtx = self.ui_matrix[:,[i,j]].copy()
                    #get rows where both items are rated
                    ij_mtx = ij_mtx[(ij_mtx>0).all(axis=1)]
                    # calculate mean difference and update dict
                    mean_diff = (ij_mtx[:,0] - ij_mtx[:,1]).mean()
                    self.popularity_differences[(i,j)] = mean_diff

        



    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError

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

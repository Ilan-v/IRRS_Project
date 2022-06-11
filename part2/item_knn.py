import numpy as np
import pickle
from utils import get_data, Config, create_ui_matrix
from user_knn import KnnUserSimilarity
from os import path
from config import *

class KnnItemSimilarity(KnnUserSimilarity):
    def __init__(self, config):
        super().__init__(config)


    def fit(self, X: np.array):
        """
        Ovveride the fit method of the parent class, for inverted user item matrix
        """
        ui_mtx = create_ui_matrix(X)
        self.ui_mtx = ui_mtx.T
        # Calculate average rating per pseudo user - meaning item
        rating_mtx = self.ui_mtx.copy().astype(np.float32)
        rating_mtx[rating_mtx == 0] = np.nan
        self.user_means = np.nanmean(rating_mtx, axis=1)
        # upload the params if they exist
        if path.exists(ITEM_CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()
        else:
            self.save_params()

    def build_user_to_user_corr_dict(self):
        super().build_item_to_itm_corr_dict()

    def predict_on_pair(self, user, item):
        pseudo_user = item
        pseudo_item = user
        return super().predict_on_pair(pseudo_user, pseudo_item)

    def upload_params(self):
        with open(ITEM_CORRELATION_PARAMS_FILE_PATH, 'rb') as f:
            self.similarity_dict = pickle.load(f)

    def save_params(self):
        self.build_user_to_user_corr_dict()
        with open(ITEM_CORRELATION_PARAMS_FILE_PATH, 'wb') as f:
            pickle.dump(self.similarity_dict, f)


if __name__ == '__main__':
    knn_config = Config(k=25)
    train, validation = get_data()
    knn = KnnItemSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))

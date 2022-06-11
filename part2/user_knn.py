import numpy as np
import pickle
from tqdm import tqdm
from interface import Regressor
from utils import get_data, Config, create_ui_matrix
from config import *
from os import path


class KnnUserSimilarity(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.ui_mtx = None  # sparse matrix of ratings with dimensions usersXitems
        self.user_means = None
        self.similarity_dict = {}

    def fit(self, X: np.array):
        # create the user item sparse matrix
        self.ui_mtx = create_ui_matrix(X)
        # Calculate average rating per user
        rating_mtx = self.ui_mtx.copy().astype(np.float32)
        rating_mtx[rating_mtx == 0] = np.nan
        self.user_means = np.nanmean(rating_mtx, axis=1)

        # upload the params if they exist
        if path.exists(USER_CORRELATION_PARAMS_FILE_PATH):
            self.upload_params()
        else:
            self.save_params()

    def build_item_to_itm_corr_dict(self):
        """
        Calculate the similarity between every two users.
        """
        for i in tqdm(range(self.ui_mtx.shape[0])):
            for j in range(i + 1, self.ui_mtx.shape[0]):
                # find shared items between the two users
                shared_items = self.ui_mtx[[i,j],:]
                shared_items = shared_items[:, (shared_items != 0).all(axis=0)]
                # check if there are any shared items
                if shared_items.shape[0] > 0:
                    # calculate the similarity between the two users
                    a = shared_items[0] - self.user_means[i]
                    b = shared_items[1] - self.user_means[j]
                    mechane = (np.sqrt(np.sum(a ** 2)) * np.sqrt(np.sum(b ** 2)))

                    if mechane > 0:
                        sim = np.sum(a*b) / mechane
                        # save similiraty if its bigger than 0
                        if sim > 0:
                            self.similarity_dict[(i,j)] = sim.astype(np.float16)


    def predict_on_pair(self, user: int, item: int):
        sims = np.zeros(self.ui_mtx.shape[0])
        for i in range(self.ui_mtx.shape[0]):
            # using symmetry to find the similarity between the two users
            if user < i:
                sims[i] = self.similarity_dict.get((user,i),0)
            else:
                sims[i] = self.similarity_dict.get((i,user),0)

        # don't include the user itself
        sims[user] = -1
        # sort similarities in descending order
        sorted_sims_idx = np.argsort(sims)[::-1]

        # get the top k users, that rated the item
        curr_k = 0  # current neighbor
        i = 0  # iteration counter
        mone = 0
        mechane = 0

        # run until we reach the k-th neighbor, or we reach the end of the list (not including the user itself)
        while curr_k < self.k and i < sorted_sims_idx.shape[0]-1:
            # if current neighbor has rated the item
            nb = sorted_sims_idx[i]
            if self.ui_mtx[nb, item] != 0:
                mone += sims[nb] * self.ui_mtx[nb,item]
                mechane += sims[nb]
                curr_k += 1
            i += 1
        if mechane > 0:
            return mone / mechane
        else:
            return self.user_means[user]

    def upload_params(self):
        # read existing params from pickle file
        with open(USER_CORRELATION_PARAMS_FILE_PATH, 'rb') as f:
            self.similarity_dict = pickle.load(f)

    def save_params(self):
        self.build_item_to_itm_corr_dict()
        # save to pickle file for later use
        with open(USER_CORRELATION_PARAMS_FILE_PATH, 'wb') as f:
            pickle.dump(self.similarity_dict, f)


if __name__ == '__main__':
    knn_config = Config(k=10)
    train, validation = get_data()
    knn = KnnUserSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))

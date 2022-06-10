import numpy as np
from tqdm import tqdm
from interface import Regressor
from utils import get_data, Config, create_ui_matrix

class KnnUserSimilarity(Regressor):
    def __init__(self, config):
        self.k = config.k
        self.ui_mtx = None # sparse matrix of ratings with dimensions usersXitems
        self.user_means = None

    def fit(self, X: np.array):
        # create the user item sparse matrix
        self.ui_mtx = create_ui_matrix(X)
        # Calculate average rating per user
        rating_mtx = self.ui_mtx.copy().astype(np.float32)
        rating_mtx[rating_mtx == 0] = np.nan
        self.user_means = np.nanmean(rating_mtx, axis=1)


    def build_item_to_itm_corr_dict(self, data):
        """
        Calculate the similarity between every two users.
        """
        for i in tqdm(self.ui_mtx.shape[0]):
            for j in range(i + 1, self.ui_mtx.shape[0]):


    def predict_on_pair(self, user: int, item: int):
        raise NotImplementedError

    def upload_params(self):
        raise NotImplementedError


if __name__ == '__main__':
    knn_config = Config(k=10)
    train, validation = get_data()
    train = train[:200000, :]
    knn = KnnUserSimilarity(knn_config)
    knn.fit(train)
    print(knn.calculate_rmse(validation))

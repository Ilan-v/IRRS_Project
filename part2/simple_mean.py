from interface import Regressor
from utils import get_data, create_ui_matrix
import np

class SimpleMean(Regressor):
    def __init__(self):
        # We will use user means as a numpy array instead of a dictionary
        self.user_means = {}

    def fit(self, X):
        ui_mtx = create_ui_matrix(X)
        # ui matrix but all zero values are converted to nan, for mean calculation
        rating_mtx = ui_mtx.copy().astype(float)
        rating_mtx[rating_mtx == 0] = np.nan
        # calculate mean but ignore nans
        self.user_means = np.nanmean(rating_mtx, axis=1)

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))

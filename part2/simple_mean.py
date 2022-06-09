from interface import Regressor
from utils import get_data, create_ui_matrix


class SimpleMean(Regressor):
    def __init__(self):
        # We will use user means as a numpy array instead of a dictionary
        self.user_means = {}

    def fit(self, X):
        ui_mtx = create_ui_matrix(X)
        self.user_means = ui_mtx.mean(axis=1)

    def predict_on_pair(self, user: int, item: int):
        return self.user_means[user]


if __name__ == '__main__':
    train, validation = get_data()
    baseline_model = SimpleMean()
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))

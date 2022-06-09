from interface import Regressor
from utils import Config, get_data, create_ui_matrix
from tqdm import tqdm
import numpy as np


class MatrixFactorization(Regressor):
    def __init__(self, config):
        self.lr = config.lr
        self.gamma = config.gamma
        self.k = config.k
        self.epochs = config.epochs
        self.epoch = 0
        self.user_bias = None
        self.item_bias = None
        self.mu = None
        self.p = None
        self.q = None

    def record(self, covn_dict):
        epoch = "{:02d}".format(self.epoch)
        temp = f"| epoch   # {epoch} :"
        for key, value in covn_dict.items():
            key = f"{key}"
            val = '{:.4}'.format(value)
            result = "{:<32}".format(F"  {key} : {val}")
            temp += result

    def calc_regularization(self):
        return self.gamma * (
                np.sum(self.item_bias ** 2) +
                np.sum(self.user_bias ** 2) +
                np.sum(self.q ** 2) +
                np.sum(self.p ** 2))

    def fit(self, X):
        # Initialize the model parameters
        ui_mtx = create_ui_matrix(X)

        n_users = ui_mtx.shape[0]
        n_items = ui_mtx.shape[1]
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.p = np.random.normal(0, 0.1, (self.k, n_users))
        self.q = np.random.normal(0, 0.1, (self.k, n_items))
        self.mu = np.mean(ui_mtx)

        # Calculating the RMSE for each epoch using SGD
        for epoch in range(self.epochs):
            self.epoch = epoch
            self.run_epoch(X)

            mse_train = np.square(self.calculate_rmse(X))
            train_objective = mse_train * X.shape[0] + self.calc_regularization()
            epoch_conv = {"train_objective": train_objective,
                          "train_mse": mse_train}
            self.record(epoch_conv)

    def run_epoch(self, data: np.array):
        for row in tqdm(data, desc=f"Epoch {self.epoch}"):
            user, item, rating = row
            # calculate the derivatives
            pred_y = self.predict_on_pair(user, item)
            derivative = -2 * (rating - pred_y)
            deriv_bu = 2 * self.gamma * self.user_bias[user]
            deriv_bi = 2 * self.gamma * self.user_bias[item]
            deriv_p = derivative * self.q[:, item] + 2 * self.gamma * self.p[:, user]
            deriv_q = derivative * self.p[:, user] + 2 * self.gamma * self.q[:, item]

            # updating the bias parameters to the opposite direction of the gradient- SGD
            self.user_bias[user] -= self.lr * (derivative + deriv_bu)
            self.item_bias[item] -= self.lr * (derivative + deriv_bi)
            self.p[:, user] -= deriv_p * self.lr
            self.q[:, item] -= deriv_q * self.lr

    def predict_on_pair(self, user, item):
        vec_prob = np.dot(self.q[:, item], self.p[:, user])
        return self.mu + self.user_bias[user] + self.item_bias[item] + vec_prob


if __name__ == '__main__':
    baseline_config = Config(
        lr=0.01,
        gamma=0.001,
        k=24,
        epochs=10)

    train, validation = get_data()
    baseline_model = MatrixFactorization(baseline_config)
    baseline_model.fit(train)
    print(baseline_model.calculate_rmse(validation))

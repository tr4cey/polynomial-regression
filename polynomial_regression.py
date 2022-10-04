import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')

class LinearModel(object):

    def __init__(self, theta=None):
        # theta: Weights vector for the model.
        self.theta = theta

    def create_poly(self, k, X):
        x1 = X[:, [1]]
        xmtrx = X
        if k >= 1:
            for i in range(2, k + 1):
                xi = np.power(x1, i)
                xmtrx = np.append(xmtrx, xi, axis=1)
        return xmtrx

    def create_cosine(self, k, X):
        x1 = X[:, [1]]
        xmtrx = X
        if k >= 1:
            for i in range(2, k + 1):
                xi = np.power(x1, i)
                xmtrx = np.append(xmtrx, xi, axis=1)
        x_cos = np.cos(x1)
        xmtrx = np.append(xmtrx, x_cos, axis=1)
        return xmtrx

    def predict(self, X, cosine):
        pred = []
        for j in range(len(X)):
            value = self.theta[0]
            for t in range(1, len(self.theta)):
                if t == len(self.theta)-1 and cosine :
                    value += self.theta[t] * (np.cos(X[j]))
                else:
                    value += self.theta[t] * (X[j] ** t)
            pred.append(value)
        return pred


def run_exp(train_path, cosine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.pdf'):

    train_x, train_y = load_dataset(train_path, add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-0.1, 1.1, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        lm = LinearModel()
        xmtrx = []
        if cosine:
            xmtrx = lm.create_cosine(k, train_x)
        else:
            xmtrx = lm.create_poly(k, train_x)

        xT = np.transpose(xmtrx)
        xT_dot_xmtrx = xT.dot(xmtrx)

        xt_dot_y = xT.dot(train_y)

        lm.theta = np.linalg.solve(xT_dot_xmtrx, xt_dot_y)
        predX = plot_x[:, 1]

        plot_y = lm.predict(predX, cosine)

        plt.ylim(-2.5, 2.5)
        f_type = 'normal'
        plt.plot(plot_x[:, 1], plot_y,
                 label='k={:d}, fit={:s}'.format(k, f_type))

    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.clf()

def load_dataset(csv_path, label_col='y', add_intercept=False):
    # Load dataset from a CSV file.

    def add_intercept_fn(x):
        global add_intercept
        return add_intercept(x)

    allowed_label_cols = ('y', 't')
    if label_col not in allowed_label_cols:
        raise ValueError('Invalid label_col: {} (expected {})'
                         .format(label_col, allowed_label_cols))

    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels

def add_intercept(x):
    # Add intercept to matrix x.
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def plot(x, y, theta, save_path, correction=1.0):
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Plot decision boundary
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1
           + np.log((2 - correction) / correction) / theta[2])
    plt.plot(x1, x2, c='red', linewidth=2)
    plt.xlim(x[:, -2].min()-.1, x[:, -2].max()+.1)
    plt.ylim(x[:, -1].min()-.1, x[:, -1].max()+.1)

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


def main(data):
    run_exp(data, ks=[3,5,10,25,50], filename='poly_plot.png')

    run_exp(data, cosine=True, ks=[3,5,10,25,50], filename='cosine_plot.png')


if __name__ == '__main__':
    main(data='data.csv')

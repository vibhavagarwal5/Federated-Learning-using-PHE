import argparse
import time
from statistics import mean

import numpy as np
import phe as paillier

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed = 10
np.random.seed(seed)


def generate_data(dataset, n=3):
    '''
    Loads and returns the data in horizontal splits 
    based on the number of clients.

    Parameters:
    ----------
    dataset: Choose between the breast_cancer or grad_school dataset
    n: No of clients

    Return:
    ----------
    X_train: List of n splits training set of X
    y_train: List of n splits training set of y
    X_test: Testing set of X
    y_test: Testing set of y
    '''
    # Loading the data
    if dataset == 'breast_cancer':
        X, y = load_breast_cancer(return_X_y=True)
    elif dataset == 'grad_school':
        X = np.genfromtxt('../data/grad_school.csv', delimiter=',')
        X = X[~np.isnan(X).any(axis=1)]
        num_features = X.shape[1] - 1
        y, X = X[:, -1], X[:, :num_features]
        if type(y[0]) != int:
            def f(x):
                if x >= 0.7:
                    return 1
                return 0
            y = np.array([f(val) for val in y])

    # Preprocessing the data to adjust the mean and variance.
    scalar = StandardScaler()
    X = scalar.fit_transform(X)

    # Splitting the data into train and test(0.25 of the total)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, shuffle=True)

    # Splitting the training data into 'n' chunks for 'n' clients
    X_train = np.array(np.array_split(X_train, n))
    y_train = np.array(np.array_split(y_train, n))

    return X_train, X_test, y_train, y_test


class Server:
    '''
    Server generate the public and private keys for the data encryption
    Decrypts the sum of gradients provided by the client

    Parameters:
    ----------
    key_length: Used to generate keys of bit length key_length
    '''

    # Generates the public and private keys of key_length bits
    def __init__(self, key_length):
        self.pubkey, self.privkey = paillier.generate_paillier_keypair(
            n_length=key_length)

    def decrypt_gradients(self, grad, n_clients):
        '''
        Returns decrypted average gradients (using private key)

        Parameters:
        ----------
        grad: The aggregated gradients to be decrypted
        n_clients: The number of clients
        '''
        return Server.decrypt_vector(self.privkey, grad) / n_clients

    @staticmethod
    def decrypt_vector(private_key, x):
        '''
        Decrypts vector x using the private key
        '''
        return np.array(list(map(private_key.decrypt, x)))


class Client:
    """
    Runs Logistic regression on local data
    Public key encrypts locally computed gradients to be passed on.

    Parameters:
    ----------
    X: Feature set of the data
    y: Label set of the data
    pubkey: Public key used to encrypt the gradients
    """

    def __init__(self, X, y, pubkey):
        self.pubkey = pubkey
        self.X = X
        self.y = y
        self.weights = np.zeros(X.shape[1])
        self.loss = list()

    def fit(self, n_iter, lr=0.01):
        '''
        Logistic Regression gradient calculation and descent optimization.
        '''
        for _ in range(n_iter):
            # Compute gradient
            gradient = self._compute_gradient()
            # Gradient descent optimization
            self.gradient_descent(gradient, lr)

            # loss estimation using cross_entropy
            y_pred = self.predict(self.X)
            loss = self.cross_entropy(y_pred, self.y)
            self.loss.append(loss)

    def cross_entropy(self, y_pred, y):
        '''
        Cross entropy loss between predicted and actual values.
        '''
        epsilon = 1e-5  # Used for log(0) adjustment
        # -y*log(y')-(1-y)log(1-y')
        return (-np.dot(y, np.log(y_pred + epsilon)) - np.dot(1 - y, np.log(1 - y_pred + epsilon))) / len(y)

    def sigmoid(self, x):
        '''
        Sigmoid Activation Function
        '''
        return 1 / (1 + np.exp(-x))

    def gradient_descent(self, gradient, lr=0.01):
        '''
        Gradient Step Function to adjust the weights using approximate learning rate

        Parameters: 
        ----------
        gradient: Computed gradient
        lr: Learning Rate required for gradient descent
        '''
        self.weights -= lr * gradient

    def _compute_gradient(self):
        '''
        Computing gradient from the predicted and actual values
        '''
        y_pred = self.predict(self.X)
        delta = y_pred - self.y
        return np.dot(self.X.T, delta) / len(X)

    def predict(self, X):
        '''
        Predicting the label using the model weights
        '''
        pred = np.round(self.sigmoid(np.dot(X, self.weights)))
        return pred

    def encrypted_gradient(self, curr_aggr=None):
        '''
        Compute and encrypt gradient. Adds computed gradient to current aggregated value.

        Parameters:
        ----------
        curr_aggr: Holds the current sum of gradients
        '''
        gradient = self._compute_gradient()
        encrypted_gradient = Client.encrypt_vector(self.pubkey, gradient)

        if curr_aggr is not None:
            return Client.sum_encrypted_vectors(curr_aggr, encrypted_gradient)
        else:
            return encrypted_gradient

    @staticmethod
    def encrypt_vector(public_key, x):
        '''
        Encryptes vector x using the public key
        '''
        return list(map(public_key.encrypt, x))

    @staticmethod
    def sum_encrypted_vectors(x, y):
        '''
        Homomorphic Addition of two encrypted vectors x and y.
        '''
        assert len(x) == len(y), \
            f"Found inputs of the size: {len(x)}, {len(y)}. Encrypted vectors must have the same size"
        return [x[i] + y[i] for i in range(len(x))]


def independent_learning(X, y, X_test, y_test, args):
    # Instantiate clients. Each client gets its own local set of training data
    clients = [Client(X[i], y[i], None) for i in range(args.n_clients)]

    # Each client run Logistic Regression on its own local dataset
    acc = list()
    for i, client in enumerate(clients):
        client.fit(args.n_iter, args.lr)
        y_pred = client.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc.append(accuracy)
        # print('Client {:d}:\t{:.2f}'.format(i, accuracy))
    print('Avg accuracy (independent training): {:.4f}'.format(mean(acc)))


def federated_learning(X, y, X_test, y_test, args):
    # Instantiate a Server which generates public  and private keys.
    server = Server(key_length=args.key_length)

    # Instantiate clients. Each gets its own training dataset and a public key for encryption
    clients = [Client(X[i], y[i], server.pubkey)
               for i in range(args.n_clients)]

    # Federated Learning with gradient descent
    for i in range(args.n_iter):

        # Compute gradient, then encrypt it and aggregate it over all clients
        encrypt_aggr = clients[0].encrypted_gradient(curr_aggr=None)
        for c in clients[1:]:
            encrypt_aggr = c.encrypted_gradient(curr_aggr=encrypt_aggr)

        # Aggregated gradient is decrypted to be used by clients
        aggr = server.decrypt_gradients(encrypt_aggr, args.n_clients)

        # Performs the gradient descent optimization using the decrypted average gradient sum
        for c_no, c in enumerate(clients):
            c.gradient_descent(aggr, args.lr)

            # loss aggregation
            loss = c.cross_entropy(c.predict(c.X), c.y)
            c.loss.append(loss)

    acc = list()
    for i, client in enumerate(clients):
        y_pred = client.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        acc.append(accuracy)
        # print('Client {:d}:\t{:.2f}'.format(i, accuracy))
    print('Avg accuracy (federated training): {:.4f}'.format(mean(acc)))


def get_args():
    parser = argparse.ArgumentParser(
        description='Basic Federated learning using Pallier HE')
    parser.add_argument('-d', '--dataset', type=str,
                        default='breast_cancer', help='Choose between breast_cancer or grad_school')
    parser.add_argument('-n', '--n_clients', type=int,
                        default=3, help='No of clients for training')
    parser.add_argument('-kl', '--key_length', type=int,
                        default=1024, help='Key length for the Pallier system')
    parser.add_argument('-i', '--n_iter', type=int,
                        default=15, help='Number of iterations')
    parser.add_argument('--lr', type=int, default=0.05,
                        help='Learning rate for the Gradient Descent')
    return parser.parse_args()


if __name__ == "__main__":
    # Get passed arguments
    args = get_args()
    assert args.dataset in [
        'breast_cancer', 'grad_school'], 'Choose between grad_school or breast_cancer'

    # Generate data splits
    X, X_test, y, y_test = generate_data(
        dataset=args.dataset, n=args.n_clients)

    # Perform independent training
    start = time.time()
    independent_learning(X, y, X_test, y_test, args)
    indep_time = time.time() - start
    print(f"Time taken to run independent training: {indep_time}\n")

    # Perform federated training
    start = time.time()
    federated_learning(X, y, X_test, y_test, args)
    fed_time = time.time() - start
    print(f"Time taken to run federated training: {fed_time}\n")

    # Compare it with sklearn's Logistic Regression model
    start = time.time()
    clf = LogisticRegression()
    clf.fit(np.concatenate(X), np.concatenate(y))
    score = clf.score(X_test, y_test)
    sklearn_time = time.time() - start
    print("Sklearn Logistic Regression Accuracy: {:.4f}".format(score))
    print(f"Time taken to run sklearn logistic regression: {sklearn_time}")

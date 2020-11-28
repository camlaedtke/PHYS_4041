import os
import imageio
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt



def make_gif(path, output_path):
    
    filenames = []
    for root, directories, files in os.walk(path, topdown=True):
        for name in files:
            filenames.append(os.path.join(root, name))

    filenames = sorted(filenames)

    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(output_path, images)
    
    
# the function making up the graph of a dataset
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$X_1$", ylabel="$X_2$")
    plt.title(plot_name, fontsize=30)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Greys", vmin=0, vmax=.6)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors='black')
    if(file_name):
        plt.savefig(file_name)
        plt.close()
        
        
# an auxiliary function that converts probability into class
def convert_prob_into_class(probs):
    probs_ = np.copy(probs)
    probs_[probs_ > 0.5] = 1
    probs_[probs_ <= 0.5] = 0
    return probs_


def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    return (Y_hat_ == Y).all(axis=0).mean()


def generate_batches(x: np.array, y: np.array, batch_size: int):
    """
    :param x - features array with (..., n) shape
    :param y - one hot ground truth array with (k, n) shape
    :batch_size - number of elements in single batch
    ----------------------------------------------------------------------------
    n - number of examples in data set
    k - number of classes
    """
    for i in range(0, x.shape[1], batch_size):
        yield (
            x.take(indices=range(i, min(i + batch_size, x.shape[1])), axis=1),
            y.take(indices=range(i, min(i + batch_size, y.shape[1])), axis=1)
        )

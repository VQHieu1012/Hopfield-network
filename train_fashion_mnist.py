import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
from keras.datasets import fashion_mnist
import hopfield
from numba import njit
from tqdm import tqdm

# Utils
def reshape(data):
    dim = int(np.sqrt(len(data)))
    return np.reshape(data, (dim, dim))

@njit
def preprocess_binary_image(img, thresh):
    """Preprocess the image by applying thresholding and flattening."""
    w, h = img.shape
    binary = img > thresh
    shift = 2 * (binary.astype(np.int32)) - 1  # Boolean to int
    
    # Reshape
    flatten = np.reshape(shift, (w * h))
    return flatten

def preprocess_image(img):
    """Calculate threshold and preprocess the image."""
    # Thresholding
    thresh = threshold_mean(img)
    return preprocess_binary_image(img, thresh)

def plot(data, test, predicted, figsize=(3, 3)):
    """Plot the train data, input data, and output predictions."""
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
        
        axarr[i, 0].imshow(data[i], cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i], cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i], cmap='gray')
        axarr[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("result_mnist.png")
    plt.show()

def main():
    # Load data
    (x_train, y_train), (_, _) = fashion_mnist.load_data()
    
    # Select training data
    data = []
    for i in range(3):
        xi = x_train[y_train == i]
        data.append(xi[0])

    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocess_image(d) for d in data]
    
    # Create Hopfield Network Model
    model = hopfield.HopfieldNetwork()
    model.train_weights(data)

    # Prepare test data
    test = []
    for i in range(5):
        xi = x_train[y_train == i]
        test.append(xi[1])
    test = [preprocess_image(d) for d in test]
    
    predicted = model.predict(test, threshold=60, asynchronous=False)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(5, 5))
    # Uncomment to show network weights matrix
    # model.plot_weights()

if __name__ == '__main__':
    main()

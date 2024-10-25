import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
from skimage import data as skdata
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
from numba import njit, prange
import hopfield

# Utility Functions with Numba optimization
@njit(parallel=True, fastmath=True)
def get_corrupted_input(input_data, corruption_level):
    """Corrupt the input by inverting values based on the corruption level."""
    corrupted_data = np.copy(input_data).astype(np.float64)  # Ensure float64
    inversion = np.random.binomial(n=1, p=corruption_level, size=len(input_data))
    for idx in prange(len(input_data)):
        if inversion[idx]:
            corrupted_data[idx] = -input_data[idx]
    return corrupted_data

@njit
def reshape_image(data):
    """Reshape a flat array into a square 2D array."""
    dim = int(np.sqrt(len(data)))
    return np.reshape(data, (dim, dim))

# Keeping plot_results without Numba as plotting doesn’t benefit from it
def plot_results(original, corrupted, predicted, figsize=(5, 6)):
    """Plot and compare original, corrupted, and predicted data."""
    original_images = [reshape_image(d) for d in original]
    corrupted_images = [reshape_image(d) for d in corrupted]
    predicted_images = [reshape_image(d) for d in predicted]

    fig, axarr = plt.subplots(len(original_images), 3, figsize=figsize)
    for i in range(len(original_images)):
        if i == 0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Corrupted data")
            axarr[i, 2].set_title('Reconstructed data')

        axarr[i, 0].imshow(original_images[i], cmap='gray')
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(corrupted_images[i], cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted_images[i], cmap='gray')
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

# Preprocessing can stay as it is since it contains non-loop operations
def preprocess_image(img, width=128, height=128):
    """Resize, binarize, and flatten an image for Hopfield Network processing."""
    # Resize image
    resized_img = resize(img, (width, height), mode='reflect').astype(np.float64)

    # Apply threshold to create binary image
    thresh = threshold_mean(resized_img)
    binary_img = resized_img > thresh
    binary_shifted = 2 * binary_img.astype(np.float64) - 1  # Convert to -1, 1 format

    return binary_shifted.flatten()

@njit
def reshape_to_square(data):
    """Reshape a flat array into a square 2D array."""
    dim = int(np.sqrt(len(data)))
    return np.reshape(data, (dim, dim))

# Optimized main functions
def main():
    # Load sample images
    images = [
        skdata.camera(), 
        skdata.horse(), 
        skdata.moon(),
        skdata.coins(),
        rgb2gray(skdata.coffee())
    ]

    # Preprocess images
    print("Starting data preprocessing...")
    processed_data = [preprocess_image(img) for img in images]

    # Initialize and train Hopfield Network
    model = hopfield.HopfieldNetwork()
    model.train_weights(processed_data)

    # Create corrupted test data
    corrupted_data = [get_corrupted_input(d, corruption_level=0.3) for d in processed_data]

    # Predict reconstructed images
    predicted_data = model.predict(corrupted_data, threshold=0, asynchronous=False)
    print("Displaying prediction results...")
    plot_results(processed_data, corrupted_data, predicted_data)
    
    # Optional: Display weights matrix
    # model.plot_weights()

if __name__ == '__main__':
    main()

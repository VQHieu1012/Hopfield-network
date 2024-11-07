import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
from skimage.transform import resize
from numba import njit, prange
import hopfield
from PIL import Image
import os

# Utility Functions with Numba optimization
@njit(parallel=True, fastmath=True)
def get_corrupted_input(input_data, corruption_level):
    """Corrupt the input by inverting values based on the corruption level."""
    corrupted_data = np.copy(input_data).astype(np.float32)  # Ensure float32
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

    # plt.tight_layout()
    plt.savefig("result.png")
    plt.show()


# Preprocessing can stay as it is since it contains non-loop operations
def preprocess_image(img, width, height):
    """Resize, binarize, and flatten an image for Hopfield Network processing."""
    # Resize image
    resized_img = resize(img, (width, height), mode='reflect').astype(np.float32)
    thresh = threshold_mean(resized_img)
    binary_img = resized_img > thresh
    binary_shifted = 2 * binary_img.astype(np.float32) - 1  # Convert to -1, 1 format
    return binary_shifted.flatten()


def load_images(image_path):
    image = Image.open(image_path)
    numpy_image = np.array(image).astype(np.float32)
    print(numpy_image.shape)
    return numpy_image

# Optimized main functions
def main():
    input_directory = "sample_data"
    min_size = 100
    images = []
    for filename in os.listdir(input_directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            input_path = os.path.join(input_directory, filename)
            images.append(load_images(input_path))
            
    print("Starting data preprocessing...")
    processed_data = [preprocess_image(img, min_size, min_size) for img in images]
    # Initialize and train Hopfield Network
    model = hopfield.HopfieldNetwork()
    model.train_weights(processed_data)

    # Create corrupted test data
    corrupted_data = [get_corrupted_input(d, corruption_level=0.35) for d in processed_data]

    # Predict reconstructed images
    predicted_data = model.predict(corrupted_data, num_iter=50, threshold=70, asynchronous=False)
    print("Displaying prediction results...")
    plot_results(processed_data, corrupted_data, predicted_data)
    
    # Optional: Display weights matrix
    # model.plot_weights()

if __name__ == '__main__':
    main()

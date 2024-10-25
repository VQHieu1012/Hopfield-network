import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from numba import njit

class HopfieldNetwork:
    def __init__(self):
        self.W = None
        self.num_neuron = 0

    def train_weights(self, train_data):
        """Train the weight matrix W using Hebbian learning."""
        print("Training weights...")
        
        num_samples = len(train_data)
        self.num_neuron = train_data[0].shape[0]
        
        # Initialize weight matrix W with zeros
        W = np.zeros((self.num_neuron, self.num_neuron), dtype=np.float64)
        avg_activity = np.sum([np.sum(sample) for sample in train_data]) / (num_samples * self.num_neuron)
        
        # Hebbian learning rule with Numba optimization
        W = self._hebbian_learning(W, train_data, avg_activity, num_samples)
        
        # Set self.W to the computed weight matrix
        self.W = W

    @staticmethod
    @njit
    def _hebbian_learning(W, train_data, avg_activity, num_samples):
        """Applies the Hebbian learning rule to compute weights."""
        for sample in train_data:
            centered_sample = sample - avg_activity
            W += np.outer(centered_sample, centered_sample)
        
        # Zero out diagonal to prevent self-connection
        np.fill_diagonal(W, 0)
        
        # Normalize weights by number of samples
        return W / num_samples

    def predict(self, data, num_iter=20, threshold=0, asynchronous=False):
        """Predict the output for each input pattern."""
        print("Starting prediction...")
        
        self.num_iter = num_iter
        self.threshold = threshold
        self.asynchronous = asynchronous
        
        # Predict each sample
        predictions = [self._run_pattern(sample) for sample in tqdm(data, desc="Predicting Samples")]
        return predictions
    
    def _run_pattern(self, initial_state):
        """Run the Hopfield network until convergence for a given initial state."""
        state = np.copy(initial_state)
        prev_energy = self.compute_energy(state.astype(np.float64))  # Cast to float64
        
        for _ in range(self.num_iter):
            if self.asynchronous:
                # Asynchronous update
                state = self._asynchronous_update(state.astype(np.float64), self.W, self.threshold, self.num_neuron)
            else:
                # Synchronous update
                state = np.sign(self.W @ state.astype(np.float64) - self.threshold)

            current_energy = self.compute_energy(state.astype(np.float64))  # Cast to float64
            if current_energy == prev_energy:
                break  # Convergence achieved
            prev_energy = current_energy
        return state

    @staticmethod
    @njit
    def _asynchronous_update(state, W, threshold, num_neuron):
        """Perform asynchronous updates on the network state."""
        for _ in range(100):
            neuron = np.random.randint(0, num_neuron)
            state[neuron] = np.sign(W[neuron] @ state.astype(np.float64) - threshold)
        return state

    @staticmethod
    @njit
    def compute_energy(state):
        """Calculate the energy of a given state."""
        return -0.5 * state @ state + np.sum(state)

    def plot_weights(self):
        """Plot and save the weight matrix as a heatmap."""
        plt.figure(figsize=(6, 5))
        plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar()
        plt.title("Network Weights")
        plt.tight_layout()
        plt.savefig("weights.png")
        plt.show()

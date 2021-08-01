# Implementation of the ICA algorithm from scratch
# following the guide at: https://towardsdatascience.com/independent-component-analysis-ica-in-python-a0ef0db0955e

# STEPS
# for a signal x
# 1. Center x by subtracting the mean
# 2. Whiten x
# 3. Choose random initial value for de-mixing matrix W
# 4. Calculate new value for W
# 5. Normalize W
# 6. Check if algorithm has converged - if not - return to step 4
# 7. Take dot product of W and x to get the independent source signals S = Wx

import numpy as np
from scipy import signal
from scipy.io import wavfile
from matplotlib import pyplot as plt
import seaborn as sns

# Random Seed + Figure Sizes
np.random.seed(0)
sns.set(rc={'figure.figsize':(11.7,8.27)})

# g and g' for re-estimation of W
def g(x):
    return np.tanh(x)

def g_der(x):
    return 1 - g(x) * g(x)

# function to center the signal
def center(X):
    X = np.array(X)
    mean = X.mean(axis=1, keepdims=True)
    return X - mean

# function to whiten the signal
def whitening(X):
    cov = np.cov(X)
    d, E = np.linalg.eigh(cov)
    D = np.diag(d)
    D_inv = np.sqrt(np.linalg.inv(D))
    X_whitened = np.dot(E, np.dot(D_inv, np.dot(E.T, X)))
    return X_whitened


# function to update the de-mixing matrix W
def calculate_new_w(w, X):
    w_new = (X * g(np.dot(w.T, X))).mean(axis = 1) - g_der(np.dot(w.T, X)).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


# main method ICA
def ica(X, iterations, tolerance = 1e-5):
    # preprocessing
    X = center(X)
    X = whitening(X)
    components_nr = X.shape[0]

    # initialise W
    W = np.zeros((components_nr, components_nr), dtype = X.dtype)

    # iteratively update W
    for i in range(components_nr):
        w = np.random.rand(components_nr)
        for j in range(iterations):
            w_new = calculate_new_w(w, X)
            
            if i >= 1:
                w_new -= np.dot(np.dot(w_new, W[:i].T), W[:i])

            distance = np.abs(np.abs((w*w_new).sum()) - 1)

            w = w_new

            if distance < tolerance:
                break

        W[i, :] = w

    # extract components by taking dot
    S = np.dot(W, X)
    return S


# function for plotting and comparing the original, mixed and predicted signals
def plot_comparison(X, sources, S):
    fig = plt.figure()

    plt.subplot(3, 1, 1)
    for x in X:
        plt.plot(x)
    plt.title("Mixtures")

    plt.subplot(3, 1, 2)
    for s in sources:
        plt.plot(s)
    plt.title("Actual Sources")

    plt.subplot(3, 1, 3)
    for s in S:
        plt.plot(s)
    plt.title("Predicted Sources")

    fig.tight_layout()
    plt.show()


# method for artificially mixing sources, with optional noise
def mix_sources(mixtures, apply_noise = False, noise_mag = 0.02):
    for i in range(len(mixtures)):
        max_val = np.max(mixtures[i])
        min_val = np.min(mixtures[i])
        if max_val > 1 or min_val < 1:
            mixtures[i] = mixtures[i] / (max_val / 2) - 0.5

    X = np.c_[[mix for mix in mixtures]]

    if apply_noise:
        X += noise_mag * np.random.normal(size = X.shape)

    return X


# test method - generate 3 signals, combine and perform ICA, and plot
def test_method():
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    
    # signals
    s1 = np.sin(2*time) #sinusoidal
    s2 = np.sign(np.sin(3*time)) #square
    s3 = signal.sawtooth(2*np.pi * time) # sawtooth
    sources = [s1, s2, s3]

    # combine signal
    X = mix_sources(sources)
    A = np.array(([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]]))
    X = np.dot(X.T, A.T)
    X = X.T

    # perform ICA
    S = ica(X, iterations= 1000)

    # plot result
    plot_comparison(X, sources, S)

# test method - on audio files from 2 mixed signal sources
def audio_test_method():
    # load files
    sample_rate, mix1 = wavfile.read('Independent Component Analysis\AudioFiles\mix1.wav')
    sample_rate, mix2 = wavfile.read('Independent Component Analysis\AudioFiles\mix2.wav')
    sample_rate, source1 = wavfile.read('Independent Component Analysis\AudioFiles\source1.wav')
    sample_rate, source2 = wavfile.read('Independent Component Analysis\AudioFiles\source2.wav')

    # artificially mix sources
    X = mix_sources([mix1, mix2])

    # perform ICA
    S = ica(X, iterations=1000)

    # plot
    plot_comparison(X, [source1, source2], S)

    # cast outputs to float32 arrays (for proper wav file)
    out1 = np.float32(S[0])
    out2 = np.float32(S[1])

    # write file to test results
    wavfile.write('Independent Component Analysis\AudioFiles\out\out1.wav', sample_rate, out1)
    wavfile.write('Independent Component Analysis\AudioFiles\out\out2.wav', sample_rate, out2)


if __name__ == "__main__":
    #test_method()
    audio_test_method()
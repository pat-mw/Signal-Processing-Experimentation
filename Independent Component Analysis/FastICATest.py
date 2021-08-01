import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy import signal

#  --- SIGNAL GENERATION ---

randomSeed = 69
N_Samples = 3000
signalDuration = 8
noiseAmplitude = 0.2

np.random.seed(randomSeed)
time = np.linspace(0, signalDuration, N_Samples)

# test signal 1 -- sinusoidal
s1 = np.sin(time)

# test signal 2 -- square wave
s2 = np.sign(np.sin(2*time))

# test signal 3 -- sawtooth
s3 = signal.sawtooth(2 * np.pi * time)


# --- SIGNAL MIXING ---
S = np.c_[s1, s2, s3]

# add random noise
Sn = S + noiseAmplitude * np.random.normal(size = S.shape)

# Standardize signal and take basis vector
Sn /= Sn.std(axis = 0)

# initial guess for our basis vector (mixing matrix)
A = np.array([[1,1,1],[0.5,2,1.0],[1.5,1.0,2.0]])
# A = np.array([[1,1,1],[1,1,1.0],[1,1.0,1]])

# create observation data for ICA (dot product of Signal and the transposed basis vector)
X = np.dot(Sn, A.T)

# --- INDEPENDENT COMPONENT ANALYSIS ---
ica = FastICA(n_components=3)

# decomposed signal
S_ = ica.fit_transform(X)

# decomposed mixing matrix
A_ = ica.mixing_

# assertion for unmixing
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)


# --- PRINCIPLE COMPONENT ANALYSIS (for comparison) ---
pca = PCA(n_components=3)
H = pca.fit_transform(X)


# --- PLOTTING THE RESULTS ---
plt.figure(figsize=[5, 10])

# our models: ground truth, noise added, mixed signals, ICA recovered, PCA recovered
models = [S, Sn, X, S_, H]
names = ['Ground Truth', 'Noise Added', 'Observation (Mixed Signals)', 'ICA decomposition', 'PCA decomposition (For Comparison)']
signal_colors = ['red', 'blue', 'orange']

rows = len(models)
columns = 1

# subplots for each model
for i, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(rows, columns, i)
    plt.title(name)
    for sig, color in zip(model.T, signal_colors):
        plt.plot(sig, color = color)

plt.tight_layout()

plt.show()


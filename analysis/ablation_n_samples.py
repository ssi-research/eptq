from matplotlib import pyplot as plt
import numpy as np

div = 100

m_samples = [128, 256, 512, 1024, 2048]
accuracy = [69.059, 69.2, 69.28, 69.326, 69.374]

plt.plot(m_samples, accuracy, "--x")
for a, b in zip(m_samples, accuracy):
    plt.text(a, b, str(np.round(b * div) / div))
plt.grid()
plt.xlabel("Accuracy[%]")
plt.ylabel("Number of samples")
plt.savefig("dataset_size.svg")
plt.show()

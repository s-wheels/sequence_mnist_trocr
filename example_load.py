from sequence_mnist.model import SequenceMNIST

import matplotlib.pylab as plt


train = SequenceMNIST(train=True, root="/tmp/data", download=True)
test = SequenceMNIST(train=False, root="/tmp/data", download=True)

train_it = iter(train)
image, labels = next(train_it)

title = "".join(str(v) for v in labels)
plt.title(title)
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.show()

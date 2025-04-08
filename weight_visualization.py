import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
model.load_model(r'results\DataAug_3+Jitter\best_models\best_model.pickle')

# test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
# test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

# with gzip.open(test_images_path, 'rb') as f:
#         magic, num, rows, cols = unpack('>4I', f.read(16))
#         test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
# with gzip.open(test_labels_path, 'rb') as f:
#         magic, num = unpack('>2I', f.read(8))
#         test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# test_imgs = test_imgs / test_imgs.max()

# logits = model(test_imgs)

mats = []
mats.append(model.layers[0].params['W'])
mats.append(model.layers[2].params['W'])

_, axes = plt.subplots(3, 100, figsize=(250, 7.5))
_.set_tight_layout(1)
# axes = axes.reshape(-1)
plt.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.01, hspace=0.01)
for i in range(100, 200):
        j = i - 100
        axes[0, j].set_title(i)
        axes[0, j].matshow(mats[0].T[i].reshape(3,32,32)[0])
        axes[1, j].matshow(mats[0].T[i].reshape(3,32,32)[1])
        axes[2, j].matshow(mats[0].T[i].reshape(3,32,32)[2])
        axes[0, j].set_xticks([])
        axes[0, j].set_yticks([])
        axes[1, j].set_xticks([])
        axes[1, j].set_yticks([])
        axes[2, j].set_xticks([])
        axes[2, j].set_yticks([])
        
plt.savefig("test.svg")
# plt.figure()
# plt.matshow(mats[1])
# plt.xticks([])
# plt.yticks([])
# plt.show()
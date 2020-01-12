import pickle
import scipy
import os
import imageio
import pickle_init
from pickle_init import PickleList
from pickle_init import pickle_it

folder_path = "CIFAR/augmented_cifar_test_gray/"
pkl_path = "CIFAR/augmented_cifar_test_gray.pkl"

classes_list = ["airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"]


def get_class(img_dir):
    idx = img_dir.find('_')
    img_class = img_dir[:idx]
    class_id = classes_list.index(img_class)
    return class_id


img_list = []
label_list = []
img_label_list = []
for subdir, dirs, files in os.walk(folder_path):
    for file in files:
        path = os.path.join(subdir, file)
        img = imageio.imread(path)
        label = get_class(file)
        img_list.append(img)
        label_list.append(label)

img_label_list.append(img_list)
img_label_list.append(label_list)

pickle_it(img_label_list, pkl_path)
#with open(pkl_path, "wb") as f:
#    pickle.dump(img_label_list, f)

import scipy
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
from PIL import Image
from check_dir import check_dir

train_imgs_dir = "CIFAR/cifar_train_classes/"
test_imgs_dir = "CIFAR/cifar_test_classes/"

augmented_train_dir = 'CIFAR/augmented_cifar_train'
augmented_test_dir = 'CIFAR/augmented_cifar_test'

check_dir(augmented_test_dir)
check_dir(augmented_train_dir)

left = 2
right = 30
top = 2
bottom = 30

train_datagen = ImageDataGenerator(rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

def get_class(img_dir):
    idx = img_dir.rfind('/') + 1
    img_class = img_dir[idx:]
    return img_class

for subdir, dirs, files in os.walk(train_imgs_dir):
    for file in files:
        i = 0
        img = load_img(f"{subdir}/{file}")
        #img = img.crop((left, top, right, bottom))
        #img = img.convert('L') # create grayscale img
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        class_img = get_class(subdir)
        prefix = f"{class_img}_{file[:-4]}"

        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=augmented_train_dir,
                                        save_prefix=prefix, save_format='png'):
            i += 1
            if i == 1:
                break


for subdir, dirs, files in os.walk(test_imgs_dir):
    for file in files:
        i = 0
        img = load_img(f"{subdir}/{file}")
        #img = img.crop((left, top, right, bottom)) # take part of image
        #img = img.convert('L') # create grayscale img
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        class_img = get_class(subdir)
        prefix = f"{class_img}_{file[:-4]}"

        for batch in test_datagen.flow(x, batch_size=1, save_to_dir=augmented_test_dir,
                                       save_prefix=prefix, save_format='png'):
            i += 1
            if i == 1:
                break

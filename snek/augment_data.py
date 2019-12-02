# example of horizontal shift image augmentation
import sys
import os
import time
import PIL
import progressbar
from pathlib import Path
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

def check_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    root = str(Path(__file__).parent.parent)
    dataset_path = root + "/datasets"
    #where all class folders are
    train_path = sys.argv[1]
    #the new training folder
    train_out_path = sys.argv[2]
    check_create_dir(train_out_path)
    num_images = 4
    classes = os.listdir(train_path)
   
    for label in classes:
        start_time = time.time()
        print(label)
        out_dir = train_out_path + '/' + label
        check_create_dir(out_dir)
        in_dir = train_path + '/' + label
        allimages = os.listdir(in_dir)
        for img_name in progressbar.progressbar(allimages):
            img = load_img(in_dir + "/" + img_name)
            data = img_to_array(img)
            samples = expand_dims(data, 0)
            datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=36, brightness_range=[0.5, 1.0], zoom_range=[0.4, 1])
            it = datagen.flow(samples, batch_size=1)
            for i in range(num_images):
                # pyplot.subplot(330 + 1 + i)
                batch = it.next()
                image = batch[0].astype('uint8')
                image = PIL.Image.fromarray(image)
                image.save(out_dir + "/" + str(i) + "_" + img_name)
        print("Time taken", time.time() - start_time, "---")
    print("done")

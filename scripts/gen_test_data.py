import os
import time
from pathlib import Path

def check_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    root = str(Path(__file__).parent.parent)
    dataset_path = root + "/datasets"
    #where all class folders are
    train_path = dataset_path + "/train_resized"
    test_path = dataset_path + "/test"

    classes = os.listdir(train_path)
    start_time = time.time()
    for label in classes:
        check_create_dir(test_path + '/' + label)
        in_dir = train_path + '/' + label
        out_dir = test_path + '/' + label
        allimages = os.listdir(in_dir)
        num_img = len(allimages)
        # image_ind = 0
        for image_ind in range(len(allimages)):
            if image_ind/len(allimages) >= 0.1:
                break
            in_image_path = in_dir + '/' + allimages[image_ind]
            out_image_path = out_dir + '/' + allimages[image_ind]
            os.rename(in_image_path, out_image_path)
        print("Time taken", time.time() - start_time, "---")
        # print(image_ind, num_img)

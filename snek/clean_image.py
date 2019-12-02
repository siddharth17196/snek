import os
import cv2
import time
from pathlib import Path

def check_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    root = str(Path(__file__).parent.parent)
    dataset_path = root + "/datasets"
    #where all class folders are
    train_path = dataset_path + "/train"
    #the new training folder
    train_out_path = train_path + "_resized"
    check_create_dir(train_out_path)

    h = w = 128 # height and width

    classes = os.listdir(train_path)
    start_time = time.time()
    for label in classes:
        check_create_dir(train_out_path + '/' + label)
        in_dir = train_path + '/' + label
        allimages = os.listdir(in_dir)
        for image in allimages:
            img = cv2.imread(in_dir + '/' + image, cv2.IMREAD_UNCHANGED)
            dim=(w, h)
            # print(image)
            img_new = cv2.resize(img, dim)
            # convert to grayscale
            gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(train_out_path + '/' + label + '/' + image, gray)

            """
            TODO: Highlight snake from the image
            something like: 
                img = cv2.findsnake(img)
                gray = cv2.cvtColor(img_new, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(train_out_path + '/' + label + '/highlighted_' + image, gray) ## VVVIP Save the file with a different name as shown in this example.

            """

    print("Time taken", time.time() - start_time, "---")

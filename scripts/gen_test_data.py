import os
import time
import random
from pathlib import Path

def check_create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

if __name__ == "__main__":
    train = 0.8
    check_create_dir("./datasets/valid")
    check_create_dir("./datasets/test")
    train_path = './datasets/train'
    valid_path = './datasets/valid'
    for label in os.listdir(train_path):
        train_label = os.path.join(train_path, label)
        valid_label = os.path.join(valid_path, label)
        # print(valid_label)
        check_create_dir(valid_label)
        total_samples = len(os.listdir(train_label))
        # print(total_samples)
        datalist = os.listdir(train_label)
        random_idx = random.sample(range(total_samples), total_samples)
        train_idx = random_idx[0:int(total_samples*train)]
        valid_idx = random_idx[int(total_samples*train):total_samples]
        for idx in valid_idx:
            os.rename(os.path.join(train_label, datalist[idx]), os.path.join(valid_label, datalist[idx]))


# if __name__ == "__main__":
#     root = str(Path(__file__).parent.parent)
#     dataset_path = root + "/datasets"
#     #where all class folders are
#     train_path = dataset_path + "/train_resized"
#     test_path = dataset_path + "/valid"

#     classes = os.listdir(train_path)
#     start_time = time.time()
#     for label in classes:
#         check_create_dir(test_path + '/' + label)
#         in_dir = train_path + '/' + label
#         out_dir = test_path + '/' + label
#         allimages = os.listdir(in_dir)
#         num_img = len(allimages)
#         # image_ind = 0
#         for image_ind in range(len(allimages)):
#             if image_ind/len(allimages) >= 0.1:
#                 break
#             in_image_path = in_dir + '/' + allimages[image_ind]
#             out_image_path = out_dir + '/' + allimages[image_ind]
#             os.rename(in_image_path, out_image_path)
#         print("Time taken", time.time() - start_time, "---")
#         # print(image_ind, num_img)

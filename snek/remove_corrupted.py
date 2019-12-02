import os
import argparse


parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-d', '--datadir', required=True)

args = parser.parse_args()

place = args.datadir
print(place)
for root, dirs, files in os.walk(place):
    for f in files:
        file_path = os.path.join(root, f) #Full path to the file
        size = os.path.getsize(file_path) #pass the full path to getsize()
        if size == 0:
            print(f, file_path)
            os.remove(file_path)
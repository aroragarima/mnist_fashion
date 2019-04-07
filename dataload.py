from zipfile import ZipFile
import numpy as np

"""load your data here"""


class DataLoader(object):
    def __init__(self):
        DIR = "./data/"

    # Returns images and labels corresponding for training and testing. Default mode is train.
    # For retrieving test data pass mode as 'test' in function call.
    def load_data(self, mode):
        label_filename = mode + "_labels"
        image_filename = mode + "_images"
        label_zip = "./data/" + label_filename + ".zip"
        image_zip = "./data/" + image_filename + ".zip"
        if mode == "train":
            with ZipFile(label_zip, "r") as lblzip:
                labels = np.frombuffer(
                    lblzip.read(label_filename), dtype=np.uint8, offset=8
                )
            with ZipFile(image_zip, "r") as imgzip:
                images = np.frombuffer(
                    imgzip.read(image_filename), dtype=np.uint8, offset=16
                ).reshape(len(labels), 784)

            return images, labels
        else:
            with ZipFile(image_zip, "r") as imgzip:
                images = np.frombuffer(
                    imgzip.read(image_filename), dtype=np.uint8, offset=16
                ).reshape(10000, 784)
            return images


if __name__ == "__main__":
    data_loader = DataLoader()
    X, y = data_loader.load_data("train")
    y = y.reshape(y.shape[0], 1)
    print("The shape of X: {} \nThe shape of y: {}".format(X.shape, y.shape))

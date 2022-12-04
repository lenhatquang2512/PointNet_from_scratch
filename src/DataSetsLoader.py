import os
import sys
import glob
import trimesh
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class DataSetsLoader:
    def __init__(self, num_points=2048, num_class=10, use_internet=True):
        self.num_points = num_points
        self.num_class = num_class
        self.data_dir = "./datasets/ModelNet10.zip"
        self.use_internet = use_internet
        self.load_datasets()

    def load_datasets(self):
        if self.use_internet:
            # 3D Vision Dataset: 128 each: http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
            # Latest Datasets - 98 each: http://modelnet.cs.princeton.edu/ModelNet10.zip
            self.data_dir = tf.keras.utils.get_file(
                "modelnet.zip",
                "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
                extract=True,
            )
        else:
            folder = os.getcwd()
            path = os.path.join(folder, "datasets/ModelNet10.zip")
            self.data_dir = tf.keras.utils.get_file(fname="modelnet.zip", origin=path, extract=True)
        self.data_dir = os.path.join(os.path.dirname(self.data_dir), "ModelNet10")

    def show_sample_data(self):
        mesh = trimesh.load(os.path.join(self.data_dir, "sofa/train/sofa_0001.off"))
        mesh.show()

        points = mesh.sample(self.num_points)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.set_axis_off()
        plt.show()

    def transform_to_tensorflow_dataset(self, num_points=2048):
        train_points = []
        train_labels = []
        test_points = []
        test_labels = []
        class_map = {}
        folders = glob.glob(os.path.join(self.data_dir, "[!README]*"))
        # Check if the system is Windows or not
        is_windows = sys.platform.startswith('win')

        for i, folder in enumerate(folders):
            print("processing class: {}".format(os.path.basename(folder)))

            # For Windows OS, the path delimiter is "\\". For Unix OS, the path delimiter is "/"
            if is_windows:
                class_map[i] = folder.split("\\")[-1]
            else:
                class_map[i] = folder.split("/")[-1]
            # gather all files
            train_files = glob.glob(os.path.join(folder, "train/*"))
            test_files = glob.glob(os.path.join(folder, "test/*"))

            for f in train_files:
                train_points.append(trimesh.load(f).sample(num_points))
                train_labels.append(i)

            for f in test_files:
                test_points.append(trimesh.load(f).sample(num_points))
                test_labels.append(i)
        return (
            np.array(train_points),
            np.array(test_points),
            np.array(train_labels),
            np.array(test_labels),
            class_map,
        )


if __name__ == "__main__":
    datasets_loader = DataSetsLoader()
    # Show a example dataset
    datasets_loader.show_sample_data()
    # Get the tensorflow compatiable dataset
    # datasets_loader.transform_to_tensorflow_dataset()

import torchvision
import numpy as np

def download_mnist():
    torchvision.datasets.MNIST(root='./data', train=True, transform=None, download=True)
    torchvision.datasets.MNIST(root='./data', train=False, transform=None, download=True)

def load_mnist_images(file_path: str):
    with open(file_path, 'rb') as f:
        # Read header information from the file
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')

        # Read pixel data from the file
        raw_data = np.fromfile(f, dtype=np.uint8)
        images = raw_data.reshape(num_images, num_rows, num_cols)
    return images


def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # 读取文件头部信息
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        num_labels = int.from_bytes(f.read(4), byteorder='big')

        # 读取标签数据
        raw_data = np.fromfile(f, dtype=np.uint8)
        labels = raw_data
    return labels


if __name__ == '__main__':
    download_mnist()
    train_images = load_mnist_images('./data/MNIST/raw/train-images-idx3-ubyte') # List of images, each image is represented as a 28 x 28 array
    train_labels = load_mnist_labels('./data/MNIST/raw/train-labels-idx1-ubyte') # List of labels, each label is a number in [0, 9]
    test_images = load_mnist_images('./data/MNIST/raw/t10k-images-idx3-ubyte')
    test_labels = load_mnist_labels('./data/MNIST/raw/t10k-labels-idx1-ubyte')  
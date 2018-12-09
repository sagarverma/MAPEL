import sys
import os, csv, random

from PIL import Image
import numpy as np

import skimage.io
from scipy.ndimage import zoom
from skimage.transform import resize
import cv2

import torch
import torch.utils.data as data
from torch.autograd import Variable

from torchvision.transforms import functional


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(images_list):

    classes = {}
    class_id = 0
    for image in images_list:
        if image[1] not in classes:
            classes[image[1]] = class_id
            class_id += 1

    return classes.keys(), classes

def make_dataset(dir, images_list, class_to_idx):
    images = []

    for image in images_list:
        images.append((dir + image[0], int(image[1])))

    return images

def make_dataset_without_idx(dir, images_list):
    images = []

    for image in images_list:
        images.append((dir + image[0], int(image[1])))

    return images

def make_sequence_dataset(dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        images = []
        for image in sequence[0]:
            images.append(dir + image)

        sequences.append([images, int(sequence[1])])

    return sequences

def make_bimode_sequence_dataset(rgb_dir, flow_dir, sequences_list):
    sequences = []

    for sequence in sequences_list:
        rgb_images = []
        flow_images = []

        for image in sequence[0]:
            rgb_images.append(rgb_dir + image)
            flow_images.append(flow_dir + image)

        sequences.append([rgb_images, flow_images, int(sequence[1])])

    return sequences

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def valid_loc(grid, loc):
    if loc[0] > 0 and loc[1] > 0 and loc[0] < grid.shape[0] and loc[1] < grid.shape[1]:
        return True

    return False

def print_grid(grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            print grid[i][j],
        print
    print "##############################################################################"

def get_adjacent_locs(current_loc):
    adjacent_locs = [[current_loc[0] - 1, current_loc[1] - 1],
                    [current_loc[0] - 1, current_loc[1]],
                    [current_loc[0] - 1, current_loc[1] + 1],
                    [current_loc[0], current_loc[1] - 1],
                    current_loc,
                    [current_loc[0], current_loc[1] + 1],
                    [current_loc[0] + 1, current_loc[1] - 1],
                    [current_loc[0] + 1, current_loc[1]],
                    [current_loc[0] + 1, current_loc[1] + 1]]
    return adjacent_locs

def populate_adjacent_locs(img, color, value_map):
    nmap = np.zeros((img.shape[0], img.shape[1]))
    x, y = np.where((img == color).all(axis = 2))
    if len(x) == 1:
        adjacent_locs = get_adjacent_locs([x, y])
        for i in value_map.keys():
            if valid_loc(nmap, adjacent_locs[i]):
                nmap[adjacent_locs[i][0], adjacent_locs[i][1]] = value_map[i]
    return nmap

def populate_adjacent_locs2(current_loc, size):
    nmap = np.ones(size)
    adjacent_locs = get_adjacent_locs(current_loc)
    for adjacent_loc in adjacent_locs:
        if adjacent_loc[0] < size[0] and adjacent_loc[1] < size[1] and adjacent_loc[0] >= 0 and adjacent_loc[1] >= 0:
            nmap[adjacent_loc[0], adjacent_loc[1]] = 0

    return nmap

def sim_loader(path):
    img = cv2.imread(path)
    maps = []

    nmap = np.zeros((img.shape[0], img.shape[1]))
    nmap[np.where((img == [255,255,255]).all(axis = 2))] = 1
    maps.append(nmap)

    maps.append(populate_adjacent_locs(img, [255,0,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}))
    maps.append(populate_adjacent_locs(img, [0,255,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}))
    maps.append(populate_adjacent_locs(img, [0,0,255], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1}))

    return np.asarray(maps)

def sim_loader2(path):
    img = cv2.imread(path)
    maps = []

    nmap = np.zeros((img.shape[0], img.shape[1]))
    nmap[np.where((img == [255,255,255]).all(axis = 2))] = 1
    maps.append(nmap)

    guard_map = populate_adjacent_locs(img, [255,0,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    invader_map = populate_adjacent_locs(img, [0,255,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    target_map = populate_adjacent_locs(img, [0,0,255], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})

    maps.append(np.logical_or(invader_map, target_map))
    maps.append(invader_map)
    maps.append(target_map)
    maps.append(np.logical_or(guard_map, np.logical_or(invader_map, target_map)))
    maps.append(guard_map)
    maps.append(invader_map)
    maps.append(target_map)

    return np.asarray(maps)

def partial_sim_loader2(path):
    """
    Guard can see both target and invader.
    """

    img = cv2.imread(path)
    img[np.where((img == [0,0,0]).all(axis = 2))] = 1
    img[np.where((img == [128,5,5]).all(axis = 2))] = 1
    img[np.where((img == [5,128,5]).all(axis = 2))] = 0
    maps = []

    nmap = np.zeros((img.shape[0], img.shape[1]))
    nmap[np.where((img == [255,255,255]).all(axis = 2))] = 1
    maps.append(nmap)

    guard_map = populate_adjacent_locs(img, [255,0,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    invader_map = populate_adjacent_locs(img, [0,255,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    target_map = populate_adjacent_locs(img, [0,0,255], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})

    maps.append(np.logical_or(invader_map, target_map))
    maps.append(invader_map)
    maps.append(target_map)
    maps.append(np.logical_or(guard_map, np.logical_or(invader_map, target_map)))
    maps.append(guard_map)
    maps.append(invader_map)
    maps.append(target_map)

    return np.asarray(maps)

def featurize(img):
    img[np.where((img == [0,0,0]).all(axis = 2))] = 1
    img[np.where((img == [128,5,5]).all(axis = 2))] = 1
    img[np.where((img == [5,128,5]).all(axis = 2))] = 0
    maps = []

    nmap = np.zeros((img.shape[0], img.shape[1]))
    nmap[np.where((img == [255,255,255]).all(axis = 2))] = 1
    maps.append(nmap)

    guard_map = populate_adjacent_locs(img, [255,0,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    invader_map = populate_adjacent_locs(img, [0,255,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    target_map = populate_adjacent_locs(img, [0,0,255], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})

    maps.append(np.logical_or(invader_map, target_map))
    maps.append(invader_map)
    maps.append(target_map)
    maps.append(np.logical_or(guard_map, np.logical_or(invader_map, target_map)))
    maps.append(guard_map)
    maps.append(invader_map)
    maps.append(target_map)

    return np.asarray([maps])

def partial_featurize(img):
    maps = []

    nmap = np.zeros((img.shape[0], img.shape[1]))
    nmap[np.where((img == [255,255,255]).all(axis = 2))] = 1
    maps.append(nmap)

    guard_map = populate_adjacent_locs(img, [255,0,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    invader_map = populate_adjacent_locs(img, [0,255,0], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})
    target_map = populate_adjacent_locs(img, [0,0,255], {0:1, 1:1, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1})

    maps.append(np.logical_or(invader_map, target_map))
    maps.append(invader_map)
    maps.append(target_map)
    maps.append(np.logical_or(guard_map, np.logical_or(invader_map, target_map)))
    maps.append(guard_map)
    maps.append(invader_map)
    maps.append(target_map)

    return np.asarray([maps])

def npy_seq_loader(seq):
    out = []
    for s in seq:
        out.append(np.load(s))
    out = np.asarray(out)

    return out


def rgb_sequence_loader(paths, mean, std, inp_size, rand_crop_size, resize_size):
    irand = random.randint(0, inp_size[0] - rand_crop_size[0])
    jrand = random.randint(0, inp_size[1] - rand_crop_size[1])
    flip = random.random()
    batch = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.center_crop(img, (inp_size[0], inp_size[1]))
        img = functional.crop(img, irand, jrand, rand_crop_size[0], rand_crop_size[1])
        img = functional.resize(img, resize_size)
        if flip < 1:
            img = functional.hflip(img)
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        batch.append(tensor)

    batch = torch.stack(batch)

    return batch

def flow_sequence_loader(paths, mean, std, inp_size, rand_crop_size, resize_size):
    irand = random.randint(0, inp_size[0] - rand_crop_size[0])
    jrand = random.randint(0, inp_size[1] - rand_crop_size[1])
    flip = random.random()
    batch = []
    for path in paths:
        img = Image.open(path)
        img = img.convert('RGB')
        img = functional.resize(img, resize_size)
        img = functional.crop(img, irand, jrand, rand_crop_size[0], rand_crop_size[1])
        if flip < 1:
            img = functional.hflip(img)
        tensor = functional.to_tensor(img)
        tensor = functional.normalize(tensor, mean, std)
        batch.append(tensor)

    batch = torch.stack(batch)

    return batch

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



class ImagePreloader(data.Dataset):

    def __init__(self, root, csv_file, class_map, transform=None, target_transform=None,
                 loader=default_loader):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append([row[0],row[1]])


        random.shuffle(images_list)
        classes, class_to_idx = class_map.keys(), class_map
        imgs = make_dataset(root, images_list, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class SequencePreloader(data.Dataset):

    def __init__(self, root, csv_file, mean, std, inp_size, rand_crop_size, resize_size):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        sequences_list = []

        for row in r:
            sequences_list.append([row[0:-1],row[-1]])

        random.shuffle(sequences_list)
        sequences = make_sequence_dataset(root, sequences_list)


        self.root = root
        self.sequences = sequences
        if 'flow' in root:
            self.loader = flow_sequence_loader
        else:
            self.loader = rgb_sequence_loader
        self.mean = mean
        self.std = std
        self.inp_size = inp_size
        self.rand_crop_size = rand_crop_size
        self.resize_size = resize_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        paths, target = self.sequences[index]
        sequence = self.loader(paths, self.mean, self.std, self.inp_size, self.rand_crop_size, self.resize_size)

        return sequence, target

    def __len__(self):
        return len(self.sequences)

class NpySequencePreloader(data.Dataset):

    def __init__(self, root, csv_file):

        r = csv.reader(open(root + csv_file, 'r'), delimiter=',')

        sequence_list = []
        for row in r:
            sequence_list.append([row[0:-1], int(row[-1])])

        sequences = make_sequence_dataset(root, sequence_list)

        self.root = root
        self.sequences = sequences
        self.loader = npy_seq_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        paths, target = self.sequences[index]
        seq = self.loader(paths)

        return seq, target

    def __len__(self):
        return len(self.sequences)

class BiModeSequencePreloader(data.Dataset):

    def __init__(self, rgb_param, flow_param):

        r = csv.reader(open(rgb_param[1], 'r'), delimiter=',')

        sequences_list = []

        for row in r:
            sequences_list.append([row[0:-1],row[-1]])

        random.shuffle(sequences_list)
        bimode_sequences = make_bimode_sequence_dataset(rgb_param[0], flow_param[0], sequences_list)

        self.bimode_sequences = bimode_sequences
        self.flow_loader = flow_sequence_loader
        self.rgb_loader = rgb_sequence_loader
        self.rgb_mean = rgb_param[2]
        self.flow_mean = flow_param[2]
        self.rgb_std = rgb_param[3]
        self.flow_std = flow_param[3]
        self.rgb_inp_size = rgb_param[4]
        self.flow_inp_size = flow_param[4]
        self.rgb_rand_crop_size = rgb_param[5]
        self.flow_rand_crop_size = flow_param[5]
        self.rgb_resize_size = rgb_param[6]
        self.flow_resize_size = flow_param[6]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        rgb_paths, flow_paths, target = self.bimode_sequences[index]
        rgb_sequence = self.rgb_loader(rgb_paths, self.rgb_mean, self.rgb_std, self.rgb_inp_size, self.rgb_rand_crop_size, self.rgb_resize_size)
        flow_sequence = self.flow_loader(flow_paths, self.flow_mean, self.flow_std, self.flow_inp_size, self.flow_rand_crop_size, self.flow_resize_size)

        return rgb_sequence, flow_sequence, target

    def __len__(self):
        return len(self.bimode_sequences)

class BiModeNpySequencePreloader(data.Dataset):

    def __init__(self, rgb_param, flow_param):

        r = csv.reader(open(rgb_param[1], 'r'), delimiter=',')

        sequence_list = []
        for row in r:
            sequence_list.append([row[0:-1], int(row[-1])])

        bimode_sequences = make_bimode_sequence_dataset(rgb_param[0], flow_param[0], sequence_list)

        self.bimode_sequences = bimode_sequences
        self.loader = npy_seq_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        rgb_paths, flow_paths, target = self.bimode_sequences[index]
        rgb_seq = self.loader(rgb_paths)
        flow_seq = self.loader(flow_paths)

        return rgb_seq, flow_seq, target

    def __len__(self):
        return len(self.bimode_sequences)

class SimImagePreloader(data.Dataset):
    def __init__(self, root, csv_file, loader=sim_loader2):

        r = csv.reader(open(csv_file, 'r'), delimiter=',')

        images_list = []

        for row in r:
            images_list.append([row[0],row[1]])


        random.shuffle(images_list)
        imgs = make_dataset_without_idx(root, images_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]

        img = self.loader(path)

        return img, target

    def __len__(self):
        return len(self.imgs)

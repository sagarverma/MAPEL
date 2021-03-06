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

def print_grid(grid):
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            print (grid[i][j], end=',')
        print ()
    print ("##############################################################################")

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

def action_to_loc(current_loc, action):
    return get_adjacent_locs(current_loc)[action]

def valid_loc(grid, loc):
    if loc[0] > 0 and loc[1] > 0 and loc[0] < grid.shape[0] and loc[1] < grid.shape[1]:
        if grid[loc[0], loc[1]] != 1:
            return True

    return False

def loc_to_action(current_loc, next_loc):
    adjacent_locs = get_adjacent_locs(current_loc)

    for i in range(len(adjacent_locs)):
        if adjacent_locs[i][0] == next_loc[0] and adjacent_locs[i][1] == next_loc[1]:
            return i

def loc_online_inbetween(loc, source, target):
    dxc = loc[0] - source[0]
    dyc = loc[1] - source[1]

    dxl = target[0] - source[0]
    dyl = target[1] - source[1]

    cross = dxc * dyl - dyc * dxl

    if cross == 0:
        if abs(dxl) >= abs(dyl):
            if dxl > 0:
                return (source[0] <= loc[0]) & (loc[0] <= target[0])
            else:
                return (target[0] <= loc[0]) & (loc[0] <= source[0])
        else:
            if dyl > 0 :
                return (source[1] <= loc[1]) & (loc[1] <= target[1])
            else:
                return (target[1] <= loc[1]) & (loc[1] <= source[1])
    else:
        return False

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

    return np.asarray(maps)

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

def get_sequence_list(root, run_ids, history_length):
    sequence_lists = []
    for run_id in run_ids:
        runlen = list(range(int(run_id[1])))
        history_runlen = [0 for x in range(history_length)] + runlen

        for i in range(history_length+1,len(history_runlen)+1):
            sequence_lists.append([root + run_id[0], history_runlen[i-history_length:i]])
            
    return sequence_lists
    
def run_loader(run_path, frame_ids):
    history = []
    npy = np.load(run_path)
    for _id in frame_ids:
        history.append(npy['runs'][_id].transpose(2,0,1))
    
    guard_action = npy['guard_actions'][frame_ids[-1]]
    invader_action = npy['invader_actions'][frame_ids[-1]]

    history = np.asarray(history) / 255.
    history = history.reshape(-1,history.shape[-1],history.shape[-1])
    return history, guard_action, invader_action
    
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

class SimRunPreloader(data.Dataset):
    def __init__(self, root, phase, history_length, loader=run_loader):

        self.root = root
        stats = np.load(root + 'stats.npz')
        run_ids = stats[phase]
        
        self.phase = phase
        self.sequence_list = get_sequence_list(self.root, run_ids, history_length)
        
        random.shuffle(self.sequence_list)
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        run_path, frame_ids = self.sequence_list[index]

        history, guard_action, invader_action = self.loader(run_path, frame_ids)

        return history, guard_action, invader_action

    def __len__(self):
        return len(self.sequence_list)
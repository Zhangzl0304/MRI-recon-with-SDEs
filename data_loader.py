from os import listdir
from os.path import join
import pandas as pd
import torch.utils.data as data
import h5py
from k_space import ifft2c, fft2c
import numpy as np
import random
import torch
import matplotlib.pyplot as plt

def c2r(x, axis=1):
    """Convert complex data to pseudo-complex data (2 real channels)

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    shape = x.shape

    dtype = np.float32 if x.dtype == np.complex64 else np.float64

    x = np.ascontiguousarray(x).view(dtype=dtype).reshape(shape + (2,))

    n = x.ndim

    if axis < 0: axis = n + axis
    if axis < n:
        newshape = tuple([i for i in range(0, axis)]) + (n-1,) \
                   + tuple([i for i in range(axis, n-1)])

        x = x.transpose(newshape)

    return x

def to_tensor_format(x, mask=False):
    """
    Assumes data is of shape (n[, nt], nx, ny).
    Reshapes to (n, n_channels, nx, ny[, nt])
    Note: Depth must be the last axis, the dimensions will be reordered
    """
    if x.ndim == 4:  # n 3D inputs. reorder axes
        x = np.transpose(x, (0, 2, 3, 1))


    if mask:  # Hacky solution

        x = x*(1+1j)

    x = c2r(x)

    return x

def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    x_real = x[:,0,...]
    x_img = x[:,1,...]
    x = x_real + 1j*x_img

    return x

def from_tensor_format(x):
    x = r2c(x)
    return x

def load_mat(file_path:str):
    with h5py.File(file_path, 'r') as f:
        key0 = list(f.keys())[0]

        assert len(list(f.keys())) == 1, "There is more than 1 key in the mat file."
        try:
            dataset = f[key0][:]
        except KeyError:
            print(f'Key Error, options:{f.keys()}')
    if dataset.ndim > 3:
        dataset = dataset["real"] + 1j*dataset["imag"]
    return dataset

np.random.seed(42)

def random_list(range1, num):
    l = []
    for i in range(num):
        l.append(np.random.randint(0, range1))
    return l




def norm_01_im(x):
    x = ifft2c(x)
    x = x / np.max(x)
    x = fft2c(x)
    return np.max(x), x

def norm_Im(x):
    x = x / np.max(x)
    return x


def crop_cmrx(im):
    # input: kt, kx, ky
    if len(im.shape) >= 3:
        kx, ky = im.shape[-2:]
        im_crop = im[..., ky//4:3*ky//4]
    elif len(im.shape) == 2:
        kx, ky = im.shape
        im_crop = im[:, ky//4:3*ky//4]
    return im_crop

def gen_patch(input, patch, rand):
        train = np.concatenate([input[..., i:i+patch] for i in rand])
        return train


def reshap_mask(input, mask):

    b, t, x, y = input.shape
    mask = np.tile(mask[np.newaxis, np.newaxis,...], (b, t, 1, 1))
    return mask

def reshape2(mask):
    mask = np.transpose(mask, (0,2,3,1))
    mask = np.tile(mask[:, np.newaxis,...], (1, 2, 1, 1, 1))
    return mask

def r2c(x, axis=1):
    """Convert pseudo-complex data (2 real channels) to complex data

    x: ndarray
        input data
    axis: int
        the axis that is used to represent the real and complex channel.
        e.g. if axis == i, then x.shape looks like (n_1, n_2, ..., n_i-1, 2, n_i+1, ..., nm)
    """
    x_real = x[:,0,...]
    x_img = x[:,1,...]
    x = x_real + 1j*x_img

    return x


class TrainDatasetFolder(data.Dataset):
    def __init__(self, Acc, ax, patch):
        super(TrainDatasetFolder, self).__init__()
        self.path = r'/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.norm = norm_01_im
        self.patch = 36
        self.patch_num = 3
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        # print(len(self.patient))
        self.n_suj = len(self.patient)
        self.is_patch = patch



    def __getitem__(self, index):

        dir_und = join(self.path, self.patient[index], self.lax)
        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        dir_mask = join(self.path, self.patient[index], self.mask)
        k_und = load_mat(dir_und)
        k_gnd = load_mat(dir_gnd)
        mask_org = load_mat(dir_mask)

        # self.shape = k_gnd.shape
        k_und = np.transpose(k_und, (1, 0, 2, 3))
        k_gnd = np.transpose(k_gnd, (1, 0, 2, 3))
        # _, k_und = self.norm(k_und)
        # _, k_gnd = self.norm(k_gnd)

        im_gnd = ifft2c(k_gnd)
        # im_und = ifft2c(k_und)

        im_gnd = crop_cmrx(im_gnd) #(10, 12, 204, 256)
        im_gnd = norm_Im(im_gnd)
        # im_und = crop_cmrx(im_und)
        # im_und = norm_Im(im_und)
        mask = crop_cmrx(mask_org)
        mask = reshap_mask(im_gnd, mask)

        k_gnd_c = fft2c(im_gnd)



        k_und = mask*k_gnd_c


        im_und = ifft2c(k_und)


        # _, k_und = self.norm(k_und)
        # _, k_gnd = self.norm(k_gnd)

        shape = k_und.shape
        randseed = random_list(shape[-2] - self.patch, self.patch_num)

        if self.is_patch:

           k_und = gen_patch(k_und, self.patch, randseed)
           mask = gen_patch(mask, self.patch, randseed)
           im_und = gen_patch(im_und, self.patch, randseed)
           im_gnd = gen_patch(im_gnd, self.patch, randseed)


        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd))  #[10, 2, 204, 256, 12]
        im_und_in = torch.from_numpy(to_tensor_format(im_und))
        k_und_in = torch.from_numpy(to_tensor_format(k_und))


        mask = reshape2(mask)
        mask_in = torch.from_numpy(mask)


        return k_und_in, im_und_in, im_gnd_in, mask_in

    def __len__(self):
        return self.n_suj



class TestDatasetFolder(data.Dataset):
    def __init__(self, Acc, ax):
        super(TestDatasetFolder, self).__init__()
        self.path = r'/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.data_gnd = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/TrainingSet/FullSample'
        self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            # print(self.patient[i],i)
            # print([x for x in listdir(join(self.path, self.patient[i]))], i)
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)



    def __getitem__(self, index):

        # dir_und = join(self.path, self.patient[index], self.lax)
        dir_gnd = join(self.data_gnd, self.patient[index], self.lax)
        dir_mask = join(self.path, self.patient[index], self.mask)
        # k_und = load_mat(dir_und)
        k_gnd = load_mat(dir_gnd)
        mask_org = load_mat(dir_mask)

        # k_und = np.transpose(k_und, (1, 0, 2, 3))
        k_gnd = np.transpose(k_gnd, (1, 0, 2, 3))
        # _, k_und = self.norm(k_und)
        # _, k_gnd = self.norm(k_gnd)

        im_gnd = ifft2c(k_gnd)
        # im_und = ifft2c(k_und)

        im_gnd = crop_cmrx(im_gnd)

        mask = crop_cmrx(mask_org)

        mask = reshap_mask(im_gnd, mask)

        k_gnd_c = fft2c(im_gnd)
        k_und = mask*k_gnd_c
        im_und = ifft2c(k_und)

        im_gnd_in = torch.from_numpy(to_tensor_format(im_gnd))
        im_und_in = torch.from_numpy(to_tensor_format(im_und))
        k_und_in = torch.from_numpy(to_tensor_format(k_und))
        mask = reshape2(mask)
        mask_in = torch.from_numpy(mask)

        return k_und_in[0:1, ...], im_und_in[0:1, ...], im_gnd_in[0:1, ...], mask_in[0:1, ...]


def slice_num(item, ax):
    path = '/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/ValidationSet/AccFactor04'
    filename = 'cine_{}.mat'.format(ax)
    join_p = join(path, item, filename)
    data = load_mat(join_p)
    return data.shape[1]



class TestData_for_rank(data.Dataset):
    def __init__(self, Acc, ax):
        super(TestData_for_rank, self).__init__()
        self.path = r'/home/zhenlin/CMRxRecon2023/dataset/SingleCoil/Cine/ValidationSet/AccFactor{}'.format(Acc)
        self.patient = [x for x in listdir(self.path)]
        self.norm = norm_01_im
        self.lax = 'cine_{}.mat'.format(ax)
        self.mask = 'cine_{}_mask.mat'.format(ax)
        del_list = []
        for i in range(len(self.patient)):
            if self.lax not in [x for x in listdir(join(self.path, self.patient[i]))]:
                del_list.append(self.patient[i])
        for ite in del_list:
            self.patient.remove(ite)
        print('num of patient',len(self.patient), self.patient)
        # self.num = num



    def __getitem__(self, index):

        dir_und = join(self.path, index, self.lax)
        dir_mask = join(self.path, index, self.mask)
        k_und = load_mat(dir_und)
        print(k_und.shape, 'input shape')
        slice = k_und.shape[1]
        print(slice,'11slicc')

        mask_org = load_mat(dir_mask)

        k_und = np.transpose(k_und, (1, 0, 2, 3))
        _, k_und = self.norm(k_und)
        mask = reshap_mask(k_und, mask_org)

        im_und = ifft2c(k_und)

        im_und_in = torch.from_numpy(to_tensor_format(im_und))
        k_und_in = torch.from_numpy(to_tensor_format(k_und))
        mask = reshape2(mask)
        mask_in = torch.from_numpy(mask)

        return k_und_in, im_und_in, mask_in

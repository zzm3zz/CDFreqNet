import random
import numpy as np
import torch
import cv2
from torch.utils import data
from scipy.special import comb


def pre(x, clip_window=None):
    if clip_window is not None:
        x = np.clip(x, clip_window[0], clip_window[1])
    else:
        b = np.percentile(x, 99.5)
        t = np.percentile(x, 0.5)
        x = np.clip(x, t, b)

    if np.max(x) == np.min(x):
        return x - np.min(x)

    x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
    return x


def load_npz(path, load_label=True):
    data = np.load(path)

    img1 = data['struct'] if 'struct' in data else data['high']
    img2 = data['style'] if 'style' in data else data['low']

    if load_label:
        label = data['seg']
    else:
        label = None

    return img1, img2, label


def get_density_guided_mapping(densities, eta=0.3, max_trials=100):
    M = len(densities)
    nonempty_indices = np.where(densities > 0)[0]
    empty_indices = np.where(densities == 0)[0]

    if len(nonempty_indices) > 0:
        sorted_nonempty = nonempty_indices[np.argsort(densities[nonempty_indices])[::-1]]
    else:
        sorted_nonempty = []

    for _ in range(max_trials):
        available_slots = list(range(M))
        final_mapping = [-1] * M
        success = True

        for src_idx in sorted_nonempty:
            rho_i = densities[src_idx]
            radius = max(1, int(np.floor(M * (1.0 - (1.0 - eta) * rho_i))))
            candidates = [slot for slot in available_slots if abs(slot - src_idx) <= radius]

            if len(candidates) == 0:
                success = False
                break

            target_slot = random.choice(candidates)
            final_mapping[src_idx] = target_slot
            available_slots.remove(target_slot)

        if success:
            random.shuffle(available_slots)
            for src_idx, target_slot in zip(empty_indices, available_slots):
                final_mapping[src_idx] = target_slot
            return torch.tensor(final_mapping, dtype=torch.long)

    return torch.arange(M, dtype=torch.long)


def density_guided_remap(data, ranges=[-1, 1], rand_point=[2, 50], eta=0.3, eps=1e-8):
    device = data.device
    M = random.randint(rand_point[0], rand_point[1])

    internal_points = torch.rand(M - 1, device=device) * (ranges[1] - ranges[0]) + ranges[0]
    boundaries = torch.cat([
        torch.tensor([ranges[0]], dtype=data.dtype, device=device),
        internal_points,
        torch.tensor([ranges[1]], dtype=data.dtype, device=device)
    ])
    boundaries, _ = torch.sort(boundaries)

    valid_pixels = data[torch.isfinite(data)]

    if valid_pixels.numel() == 0:
        return data

    valid_buckets = torch.bucketize(valid_pixels.contiguous(), boundaries[1:-1])
    counts = torch.bincount(valid_buckets, minlength=M).float()

    rho_bar = counts / (counts.sum() + eps)
    rho = rho_bar / (rho_bar.max() + eps)

    mapping = get_density_guided_mapping(rho.detach().cpu().numpy(), eta=eta).to(device)

    data_buckets = torch.bucketize(data.contiguous(), boundaries[1:-1])

    src_min = boundaries[data_buckets]
    src_max = boundaries[data_buckets + 1]

    target_indices = mapping[data_buckets]
    tgt_min = boundaries[target_indices]
    tgt_max = boundaries[target_indices + 1]

    denom = torch.clamp(src_max - src_min, min=eps)
    new_image = tgt_min + (data - src_min) * (tgt_max - tgt_min) / denom

    return torch.clamp(new_image, ranges[0], ranges[1])


class AppearanceFrequencyIntervention(object):
    def __init__(self, p=1.0, rmmax=50, eta=0.3):
        self.p = p
        self.rmmax = rmmax
        self.eta = eta

    def __call__(self, img):
        if random.random() >= self.p:
            return img

        return density_guided_remap(img, ranges=[-1, 1], rand_point=[2, self.rmmax], eta=self.eta)


def bernstein_poly(i, n, t):
    return comb(n, i) * (1 - t) ** (n - i) * t ** i


def bezier_curve(points, nTimes=100000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)
    polynomial_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(x, prob=1.0):
    if random.random() >= prob:
        return x

    is_tensor = torch.is_tensor(x)

    if is_tensor:
        device = x.device
        dtype = x.dtype
        x_np = x.detach().cpu().numpy()
    else:
        x_np = np.asarray(x)

    x_normalized = np.clip((x_np + 1.0) / 2.0, 0.0, 1.0)

    points = np.array([
        [0.0, 0.0],
        [random.random(), random.random()],
        [random.random(), random.random()],
        [1.0, 1.0]
    ], dtype=np.float64)

    points[:, 0] = np.sort(points[:, 0])

    if random.random() < 0.5:
        points[:, 1] = np.sort(points[:, 1])

    xvals, yvals = bezier_curve(points, nTimes=100000)
    nonlinear_x_normalized = np.interp(x_normalized, xvals, yvals)
    nonlinear_x = nonlinear_x_normalized * 2.0 - 1.0
    nonlinear_x = np.clip(nonlinear_x, -1.0, 1.0).astype(np.float32)

    if is_tensor:
        return torch.from_numpy(nonlinear_x).to(device=device, dtype=dtype)

    return nonlinear_x


class Bezier_curve(object):
    def __init__(self, p=1.0):
        self.p = p

    def __call__(self, *inputs):
        outputs = []

        for _input in inputs:
            outputs.append(nonlinear_transformation(_input, prob=self.p))

        if len(outputs) == 1:
            return outputs[0]

        return outputs


class BilateralFilter(object):
    def __init__(self, prob=1.0, d=7, sigma_int=0.2, sigma_sp=2):
        self.prob = prob
        self.d = d
        self.sigma_int = sigma_int
        self.sigma_sp = sigma_sp

    def __call__(self, x):
        if random.random() >= self.prob:
            return x

        is_tensor = torch.is_tensor(x)

        if is_tensor:
            device = x.device
            dtype = x.dtype
            img = x.detach().cpu().numpy()
        else:
            img = np.asarray(x)

        img = img.astype(np.float32)
        original_shape = img.shape
        flat_img = img.reshape(-1, original_shape[-2], original_shape[-1])

        out_list = []

        for i in range(flat_img.shape[0]):
            filtered_slice = cv2.bilateralFilter(flat_img[i], d=self.d, sigmaColor=self.sigma_int, sigmaSpace=self.sigma_sp)
            out_list.append(filtered_slice)

        out_img = np.stack(out_list).reshape(original_shape)

        if is_tensor:
            return torch.from_numpy(out_img).to(device=device, dtype=dtype)

        return out_img


class Dataset3D(data.Dataset):
    def __init__(self, dir_):
        super(Dataset3D, self).__init__()
        self.filenames = dir_

    def __getitem__(self, index):
        rootfile = self.filenames[index]
        img1, img2, label = load_npz(rootfile, load_label=True)

        img1 = pre(img1)
        img2 = pre(img2)

        img1 = img1.astype(np.float32)[None, ...]
        img2 = img2.astype(np.float32)[None, ...]
        label = label.astype(np.int64)[None, ...]

        return img1, img2, label

    def __len__(self):
        return len(self.filenames)


class Dataset3D_DFI(data.Dataset):
    def __init__(self, dir_, rmmax, eta=0.3, use_label=True):
        super(Dataset3D_DFI, self).__init__()
        self.filenames = dir_
        self.use_label = use_label
        self.sfi_1 = BilateralFilter(prob=1.0, d=7, sigma_int=0.2, sigma_sp=2)
        self.sfi_2 = Bezier_curve(p=1.0)
        self.afi = AppearanceFrequencyIntervention(p=1.0, rmmax=rmmax, eta=eta)

    def __getitem__(self, index):
        rootfile = self.filenames[index]
        img1, img2, label = load_npz(rootfile, load_label=self.use_label)

        img1 = pre(img1)
        img2 = pre(img2)

        img1 = img1.astype(np.float32)[None, ...]
        img2 = img2.astype(np.float32)[None, ...]

        img1_r = torch.from_numpy(img1)
        img2_r = torch.from_numpy(img2)

        img1_step1 = self.sfi_1(img1_r)
        img1_step2 = self.sfi_2(img1_step1)
        img2_step1 = self.afi(img2_r)

        if self.use_label:
            label = label.astype(np.int64)[None, ...]
        else:
            label = np.zeros((1,) + img1.shape[1:], dtype=np.int64)

        return img1, img2, img1_step2, img2_step1, label

    def __len__(self):
        return len(self.filenames)

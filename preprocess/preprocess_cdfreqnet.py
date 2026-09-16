import os
import argparse
import multiprocessing as mp
from functools import partial

import numpy as np

if not hasattr(np, 'asfarray'):
    def _mock_asfarray(a, dtype=float):
        return np.asarray(a, dtype=dtype)
    np.asfarray = _mock_asfarray

import SimpleITK as sitk
from scipy.ndimage import zoom
import dtcwt


TRANSFORMER_3D = dtcwt.Transform3d()


def align_to_rai(image):
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAI")
    return orienter.Execute(image)


def resample_image_to_spacing(image, target_xy_spacing=(1.0, 1.0)):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    target_spacing = (target_xy_spacing[0], target_xy_spacing[1], original_spacing[2])
    target_size = [int(round(original_size[i] * original_spacing[i] / target_spacing[i])) for i in range(3)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(target_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetInterpolator(sitk.sitkBSpline)

    return resampler.Execute(image)


def resample_label_to_reference(label, reference_image):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return resampler.Execute(label)


def normalize_intensity(volume, modality):
    volume = volume.astype(np.float32)
    modality = modality.upper()

    if modality == 'CT':
        volume = np.clip(volume, -125, 275)
    elif modality == 'MRI':
        upper = np.percentile(volume, 99.5)
        volume = np.clip(volume, volume.min(), upper)
    else:
        raise ValueError("modality must be 'CT' or 'MRI'")

    v_min = volume.min()
    v_max = volume.max()

    if v_max - v_min < 1e-8:
        return np.zeros_like(volume, dtype=np.float32)

    volume = (volume - v_min) / (v_max - v_min) * 2.0 - 1.0
    return volume.astype(np.float32)


def get_bbox_from_mask(mask, margin=15):
    pos = np.where(mask > 0)

    if len(pos[0]) == 0:
        return 0, mask.shape[0], 0, mask.shape[1], 0, mask.shape[2]

    z1 = max(0, int(pos[0].min()) - margin)
    z2 = min(mask.shape[0], int(pos[0].max()) + margin + 1)
    y1 = max(0, int(pos[1].min()) - margin)
    y2 = min(mask.shape[1], int(pos[1].max()) + margin + 1)
    x1 = max(0, int(pos[2].min()) - margin)
    x2 = min(mask.shape[2], int(pos[2].max()) + margin + 1)

    return z1, z2, y1, y2, x1, x2


def resize_volume(volume, target_shape, order):
    if tuple(volume.shape) == tuple(target_shape):
        return volume

    factors = np.array(target_shape, dtype=np.float64) / np.array(volume.shape, dtype=np.float64)
    resized = zoom(volume, factors, order=order)

    if tuple(resized.shape) != tuple(target_shape):
        fixed = np.zeros(target_shape, dtype=resized.dtype)
        z = min(target_shape[0], resized.shape[0])
        y = min(target_shape[1], resized.shape[1])
        x = min(target_shape[2], resized.shape[2])
        fixed[:z, :y, :x] = resized[:z, :y, :x]
        resized = fixed

    return resized


def dtcwt_decompose_numpy(volume, nlevels=2, alpha=2.0):
    volume = volume.astype(np.float32)
    pyramid = TRANSFORMER_3D.forward(volume, nlevels=nlevels)
    Pyramid = type(pyramid)

    original_shape = volume.shape
    actual_levels = len(pyramid.highpasses)

    reconstructed_lows = []

    for level in range(1, actual_levels + 1):
        masked_highpasses = []
        for high_idx in range(actual_levels):
            if high_idx < level:
                masked_highpasses.append(np.zeros_like(pyramid.highpasses[high_idx]))
            else:
                masked_highpasses.append(pyramid.highpasses[high_idx])

        low_pyramid = Pyramid(pyramid.lowpass, tuple(masked_highpasses))
        low_reconstruction = TRANSFORMER_3D.inverse(low_pyramid)
        low_reconstruction = resize_volume(low_reconstruction, original_shape, order=1)
        reconstructed_lows.append(low_reconstruction)

    if len(reconstructed_lows) > 0:
        low_frequency = np.mean(np.stack(reconstructed_lows, axis=0), axis=0)
    else:
        low_frequency = volume.copy()

    reconstructed_highs = []

    for level in range(actual_levels):
        isolated_highpasses = []

        for high_idx in range(actual_levels):
            if high_idx == level:
                isolated_highpasses.append(pyramid.highpasses[high_idx])
            else:
                isolated_highpasses.append(np.zeros_like(pyramid.highpasses[high_idx]))

        high_pyramid = Pyramid(np.zeros_like(pyramid.lowpass), tuple(isolated_highpasses))
        high_reconstruction = TRANSFORMER_3D.inverse(high_pyramid)
        high_reconstruction = resize_volume(high_reconstruction, original_shape, order=1)
        reconstructed_highs.append(high_reconstruction)

    if len(reconstructed_highs) > 0:
        high_frequency = np.mean(np.stack(reconstructed_highs, axis=0), axis=0)
    else:
        high_frequency = np.zeros_like(volume)

    high_enhanced = volume + alpha * high_frequency

    low_frequency = np.clip(low_frequency, -1.0, 1.0)
    high_enhanced = np.clip(high_enhanced, -1.0, 1.0)

    return high_enhanced.astype(np.float32), low_frequency.astype(np.float32)


def get_label_id(data_id, replace_from="", replace_to=""):
    if replace_from:
        return data_id.replace(replace_from, replace_to, 1)
    return data_id


def process_single_file(data_id, args):
    try:
        save_file = os.path.join(args.save_root, data_id + ".npz")

        if os.path.exists(save_file) and not args.overwrite:
            return True, data_id

        image_path = os.path.join(args.image_root, data_id + args.image_suffix)
        label_id = get_label_id(data_id, args.label_replace_from, args.label_replace_to)
        label_path = os.path.join(args.label_root, label_id + args.label_suffix)

        if not os.path.isfile(image_path):
            raise FileNotFoundError("Image not found: " + image_path)

        if not os.path.isfile(label_path):
            raise FileNotFoundError("Label not found: " + label_path)

        image = align_to_rai(sitk.ReadImage(image_path))
        label = align_to_rai(sitk.ReadImage(label_path))

        image = resample_image_to_spacing(image, target_xy_spacing=(args.spacing_x, args.spacing_y))
        label = resample_label_to_reference(label, image)

        image_np = sitk.GetArrayFromImage(image).astype(np.float32)
        label_np = sitk.GetArrayFromImage(label).astype(np.int32)

        image_np = normalize_intensity(image_np, args.modality)

        if args.use_roi:
            z1, z2, y1, y2, x1, x2 = get_bbox_from_mask(label_np, margin=args.margin)
            image_np = image_np[z1:z2, y1:y2, x1:x2]
            label_np = label_np[z1:z2, y1:y2, x1:x2]

        target_shape = (args.depth, args.height, args.width)
        image_np = resize_volume(image_np, target_shape, order=1).astype(np.float32)
        label_np = resize_volume(label_np, target_shape, order=0).astype(np.uint8)

        high_np, low_np = dtcwt_decompose_numpy(image_np, nlevels=args.nlevels, alpha=args.alpha)

        os.makedirs(args.save_root, exist_ok=True)
        np.savez_compressed(save_file, data=image_np.astype(np.float32), high=high_np.astype(np.float32), low=low_np.astype(np.float32), seg=label_np.astype(np.uint8))

        return True, data_id

    except Exception as e:
        print("Error processing " + str(data_id) + ": " + str(e))
        return False, data_id


def collect_data_ids(args):
    files = [f for f in os.listdir(args.image_root) if f.endswith(args.image_suffix)]
    data_ids = [f[:-len(args.image_suffix)] for f in files]
    return sorted(data_ids)


def build_parser():
    parser = argparse.ArgumentParser(description='CDFreqNet preprocessing')

    parser.add_argument('--image_root', type=str, default='./raw_data/images/')
    parser.add_argument('--label_root', type=str, default='./raw_data/labels/')
    parser.add_argument('--save_root', type=str, default='./data/preprocessed/')

    parser.add_argument('--modality', type=str, choices=['CT', 'MRI'], required=True)

    parser.add_argument('--image_suffix', type=str, default='_0000.nii.gz')
    parser.add_argument('--label_suffix', type=str, default='.nii.gz')
    parser.add_argument('--label_replace_from', type=str, default='')
    parser.add_argument('--label_replace_to', type=str, default='')

    parser.add_argument('--spacing_x', type=float, default=1.0)
    parser.add_argument('--spacing_y', type=float, default=1.0)

    parser.add_argument('--use_roi', action='store_true')
    parser.add_argument('--margin', type=int, default=15)

    parser.add_argument('--depth', type=int, default=64)
    parser.add_argument('--height', type=int, default=160)
    parser.add_argument('--width', type=int, default=160)

    parser.add_argument('--nlevels', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=2.0)

    parser.add_argument('--num_workers', type=int, default=max(1, mp.cpu_count() // 2))
    parser.add_argument('--overwrite', action='store_true')

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    data_ids = collect_data_ids(args)

    print("Modality: " + args.modality)
    print("RAI orientation: Enabled")
    print("XY spacing: " + str((args.spacing_x, args.spacing_y)))
    print("ROI cropping: " + str(args.use_roi))
    print("ROI margin: " + str(args.margin if args.use_roi else "N/A"))
    print("Final shape: " + str((args.depth, args.height, args.width)))
    print("DTCWT levels: " + str(args.nlevels))
    print("HF sharpening alpha: " + str(args.alpha))
    print("Files: " + str(len(data_ids)))

    worker_func = partial(process_single_file, args=args)

    if args.num_workers == 1:
        results = [worker_func(data_id) for data_id in data_ids]
    else:
        with mp.Pool(processes=args.num_workers) as pool:
            results = list(pool.imap_unordered(worker_func, data_ids))

    success_count = sum(r[0] for r in results)
    failed_ids = [r[1] for r in results if not r[0]]

    print("Success: " + str(success_count) + "/" + str(len(data_ids)))

    if len(failed_ids) > 0:
        print("Failed cases:")
        for data_id in failed_ids:
            print("  " + data_id)

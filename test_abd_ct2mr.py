import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, binary_erosion
from models.network import CDFreqNet

def pre(x, clip_window=None):
    if clip_window is not None:
        x = np.clip(x, clip_window[0], clip_window[1])
        x = (x - np.min(x))/(np.max(x) - np.min(x)) * 2 -1
    else:
        b = np.percentile(x, 99.5) 
        t = np.percentile(x, 00.5) 
        x = np.clip(x, t, b)
        if np.max(x) == np.min(x):
            return x - np.min(x)
        x = (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 -1
    return x

def calculate_metrics(pred, gt, num_classes):
    dices = []
    asds = []
    
    for c in range(1, num_classes):
        p = (pred == c)
        g = (gt == c)

        intersection = np.logical_and(p, g).sum()
        union = p.sum() + g.sum()
        
        if union == 0:
            dices.append(1.0)
            asds.append(0.0) 
            continue
        elif g.sum() == 0 or p.sum() == 0:
            dices.append(0.0)
            asds.append(np.nan) 
            continue
        else:
            dices.append(2.0 * intersection / union)

        struct = np.ones((3, 3, 3))
        p_border = p ^ binary_erosion(p, structure=struct)
        g_border = g ^ binary_erosion(g, structure=struct)
        
        dt_p = distance_transform_edt(~p_border)
        dt_g = distance_transform_edt(~g_border)
        
        dist_p_to_g = dt_g[p_border]
        dist_g_to_p = dt_p[g_border]
        
        sum_dist = dist_p_to_g.sum() + dist_g_to_p.sum()
        total_voxels = len(dist_p_to_g) + len(dist_g_to_p)
        
        if total_voxels == 0:
            asds.append(0.0)
        else:
            asds.append(sum_dist / total_voxels)
            
    return dices, asds

def test():
    weight_path = ""

    test_dir = "./data/MR/CHAOS/dataTr/"
    
    save_dir = os.path.join(os.path.dirname(weight_path), "test_predictions")
    os.makedirs(save_dir, exist_ok=True)
    
    num_classes = 5 
    
    print(f"Loading weights from: {weight_path}")
    model = CDFreqNet(input_channels=1, num_classes=num_classes).cuda()

    model.load_state_dict(torch.load(weight_path, map_location='cuda'), strict=True)
    
    model.eval()
    
    test_files = glob.glob(os.path.join(test_dir, "*.npz"))
    print(f"Found {len(test_files)} files to evaluate.\n")
    
    all_dices = []
    all_asds = []
    
    with torch.no_grad():
        for file_path in test_files:
            file_name = os.path.basename(file_path)
            
            npz_data = np.load(file_path)
            img_np = npz_data['data']
            img_np1 = npz_data['high']
            img_np2 = npz_data['low']
            seg_np = npz_data['seg']

            img_pre = pre(img_np)
            img_pre1 = pre(img_np1)
            img_pre2 = pre(img_np2)

            img_tensor = torch.from_numpy(img_pre.astype(np.float32))[None, None, ...].cuda()
            img_tensor1 = torch.from_numpy(img_pre1.astype(np.float32))[None, None, ...].cuda()
            img_tensor2 = torch.from_numpy(img_pre2.astype(np.float32))[None, None, ...].cuda()
            
            pred_probs, _ = model(x_high=img_tensor1, x_low=img_tensor2)
            
            pred_mask = torch.argmax(pred_probs[0], dim=0).cpu().numpy().astype(np.uint8)
            
            volume_dices, volume_asds = calculate_metrics(pred_mask, seg_np, num_classes)
            
            all_dices.append(volume_dices)
            all_asds.append(volume_asds)
            
            save_name = file_name.replace('.npz', '_pred.nii.gz')
            save_path = os.path.join(save_dir, save_name)
            sitk_img = sitk.GetImageFromArray(pred_mask)
            sitk.WriteImage(sitk_img, save_path)
            
            dice_str = " | ".join([f"C{c+1}: {d:.4f}" for c, d in enumerate(volume_dices)])
            asd_str = " | ".join([f"C{c+1}: {a:.4f} vox" if not np.isnan(a) else f"C{c+1}: NaN" for c, a in enumerate(volume_asds)])
            print(f"Processed: {file_name}")
            print(f"    Dice: {dice_str}")
            print(f"    ASD:  {asd_str}")

    all_dices = np.array(all_dices)
    all_asds = np.array(all_asds)

    mean_dices = np.nanmean(all_dices, axis=0)
    std_dices = np.nanstd(all_dices, axis=0)
    
    mean_asds = np.nanmean(all_asds, axis=0)
    std_asds = np.nanstd(all_asds, axis=0)

    patient_mean_dices = np.nanmean(all_dices, axis=1)
    overall_mean_dice = np.nanmean(patient_mean_dices)
    overall_std_dice = np.nanstd(patient_mean_dices)
    
    patient_mean_asds = np.nanmean(all_asds, axis=1)
    overall_mean_asd = np.nanmean(patient_mean_asds)
    overall_std_asd = np.nanstd(patient_mean_asds)
    
    print("\n" + "="*50)
    print("Testing Complete!")
    print(f"Predictions saved at: {save_dir}")
    print("-" * 50)
    print("Dice Score (Mean +- STD):")
    for c in range(1, num_classes):
        print(f"  Class {c}: {mean_dices[c-1]:.4f} +- {std_dices[c-1]:.4f}")
    print(f"Overall:   {overall_mean_dice:.4f} +- {overall_std_dice:.4f}")
    
    print("-" * 50)
    print("ASD in Voxels (Mean +- STD):")
    for c in range(1, num_classes):
        print(f"  Class {c}: {mean_asds[c-1]:.4f} +- {std_asds[c-1]:.4f} voxels")
    print(f"Overall:   {overall_mean_asd:.4f} +- {overall_std_asd:.4f} voxels")
    print("="*50)

if __name__ == '__main__':
    test()

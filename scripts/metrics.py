import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff

def maybe_convert_tensor_to_numpy(array):
    """Chuyển đổi Tensor sang NumPy nếu cần"""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return array

def compute_hausdorff(seg, gt):
    """Tính Hausdorff Distance giữa hai binary masks"""
    seg_points = np.argwhere(seg > 0)
    gt_points = np.argwhere(gt > 0)

    if len(seg_points) == 0 or len(gt_points) == 0:
        return np.inf  # Nếu một trong hai mask rỗng, trả về vô cùng

    hausdorff_dist = max(directed_hausdorff(seg_points, gt_points)[0], 
                         directed_hausdorff(gt_points, seg_points)[0])
    return hausdorff_dist

def compute_ged(seg=None, gt=None):
    """Tính Generalized Energy Distance (GED) giữa segment và ground truth"""
    hausdorff = compute_hausdorff(seg, gt)
    dice = dice_score(torch.tensor(seg), torch.tensor(gt))  # Sử dụng Dice Score đã định nghĩa trước đó

    # Công thức GED có thể thay đổi theo ứng dụng, đây là một cách tiếp cận phổ biến
    ged = hausdorff * (1 - dice)
    return ged

def dice_score(preds, targets, smooth=1e-6):

    preds = (preds > 0.5).float()  # Chuyển dự đoán thành nhị phân (nếu chưa phải)
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def compute_pairwise_iou(seg, gt):
    """
    Tính toán IoU (Intersection over Union) giữa từng cặp phân vùng dự đoán và ground truth.
    
    Args:
        seg (np.ndarray): Mảng nhị phân dự đoán (H, W) hoặc (N, H, W).
        gt (np.ndarray): Mảng nhị phân ground truth (H, W) hoặc (N, H, W).
    
    Returns:
        float: Giá trị IoU trung bình giữa các cặp.
    """
    intersection = np.logical_and(seg, gt).sum(axis=(1, 2))
    union = np.logical_or(seg, gt).sum(axis=(1, 2))
    
    # Tránh chia cho 0
    iou = intersection / (union + 1e-6)
    return np.mean(iou)

def expected_calibration_error(probabilities, ground_truth, bins=10):
    """
    Tính Expected Calibration Error (ECE) để đo mức độ tin cậy của mô hình phân đoạn.

    Args:
        probabilities (np.ndarray): Mảng xác suất dự đoán (H, W).
        ground_truth (np.ndarray): Mảng nhãn thực tế (H, W).
        bins (int): Số lượng bins để nhóm giá trị xác suất.

    Returns:
        float: Giá trị ECE.
    """
    bin_boundaries = np.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (probabilities >= bin_lower) & (probabilities < bin_upper)
        if np.any(mask):
            acc = np.mean(ground_truth[mask])
            conf = np.mean(probabilities[mask])
            ece += np.abs(acc - conf) * np.mean(mask)

    return ece

def variance_ncc_dist(sample_arr, gt_arr):
    """
    Tính độ lệch phương sai của NCC giữa các mẫu phân vùng và ground truth.
    
    Args:
        sample_arr (np.ndarray): Mảng dự đoán có kích thước (N, H, W).
        gt_arr (np.ndarray): Mảng ground truth có kích thước (H, W).
    
    Returns:
        float: Phương sai của NCC.
    """
    n_samples = sample_arr.shape[0]
    ncc_values = []

    for i in range(n_samples):
        pred = sample_arr[i]
        mean_pred = np.mean(pred)
        mean_gt = np.mean(gt_arr)
        
        numerator = np.sum((pred - mean_pred) * (gt_arr - mean_gt))
        denominator = np.sqrt(np.sum((pred - mean_pred) ** 2) * np.sum((gt_arr - mean_gt) ** 2) + 1e-6)
        
        ncc = numerator / denominator
        ncc_values.append(ncc)

    return np.var(ncc_values)  # Tính phương sai của NCC

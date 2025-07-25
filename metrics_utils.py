from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import cv2
import os
import numpy as np

def compute_metrics(real_dir, fake_dir):
    real_files = sorted([f for f in os.listdir(real_dir) if f.endswith('.jpg') or f.endswith('.png')])
    fake_files = sorted([f for f in os.listdir(fake_dir) if f.endswith('.jpg') or f.endswith('.png')])
    psnr_vals, ssim_vals, mae_vals, mse_vals = [], [], [], []
    for rf, ff in zip(real_files, fake_files):
        real_img = cv2.imread(os.path.join(real_dir, rf))
        fake_img = cv2.imread(os.path.join(fake_dir, ff))
        if real_img is None or fake_img is None:
            continue
        real_img = cv2.resize(real_img, (fake_img.shape[1], fake_img.shape[0]))
        psnr_vals.append(psnr(real_img, fake_img, data_range=255))
        ssim_vals.append(ssim(real_img, fake_img, channel_axis=-1, data_range=255))
        mae_vals.append(np.mean(np.abs(real_img.astype(np.float32) - fake_img.astype(np.float32))))
        mse_vals.append(np.mean((real_img.astype(np.float32) - fake_img.astype(np.float32)) ** 2))
    return {
        'PSNR': round(sum(psnr_vals)/len(psnr_vals), 2) if psnr_vals else 'N/A',
        'SSIM': round(sum(ssim_vals)/len(ssim_vals), 3) if ssim_vals else 'N/A',
        'MAE': round(sum(mae_vals)/len(mae_vals), 2) if mae_vals else 'N/A',
        'MSE': round(sum(mse_vals)/len(mse_vals), 2) if mse_vals else 'N/A',
    } 
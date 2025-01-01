import torch
import cv2
import numpy as np
from SSPSR import SSPSR  # Assuming this is your model architecture file
from common import default_conv  # Assuming this is needed for model construction
import scipy.io as sio
import utils

def preprocess(load_dir, augment=False, aug_num=0, use_3D=False):
    """
    Preprocesses the data by loading the .mat file, applying augmentations,
    and formatting it for model input.
    
    Args:
    - load_dir (str): Path to the .mat file.
    - augment (bool): Whether to apply data augmentation.
    - aug_num (int): Augmentation mode (0 to 7 if augmenting).
    - use_3D (bool): Whether to format data for 3D convolutions.

    Returns:
    - ms (torch.Tensor): Multispectral input.
    - lms (torch.Tensor): Low-resolution multispectral input.
    - gt (torch.Tensor): Ground truth data.
    """

    # Load .mat file
    data = sio.loadmat(load_dir)

    # Extract data arrays
    ms = np.array(data['ms'][...], dtype=np.float32)
    lms = np.array(data['ms_bicubic'][...], dtype=np.float32)
    gt = np.array(data['gt'][...], dtype=np.float32)

    # Apply augmentations if required
    if augment:
        ms = utils.data_augmentation(ms, mode=aug_num)
        lms = utils.data_augmentation(lms, mode=aug_num)
        gt = utils.data_augmentation(gt, mode=aug_num)

    # Prepare for 3D convolution or regular 2D conv
    if use_3D:
        ms, lms, gt = ms[np.newaxis, :, :, :], lms[np.newaxis, :, :, :], gt[np.newaxis, :, :, :]
        ms = torch.from_numpy(ms.copy()).permute(0, 3, 1, 2)
        lms = torch.from_numpy(lms.copy()).permute(0, 3, 1, 2)
        gt = torch.from_numpy(gt.copy()).permute(0, 3, 1, 2)
    else:
        ms = torch.from_numpy(ms.copy()).permute(2, 0, 1)
        lms = torch.from_numpy(lms.copy()).permute(2, 0, 1)
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1)

    return ms, lms, gt


# 1. Load Model
def load_model(model_path, device, n_subs, n_ovls, n_blocks, n_feats, n_scale, use_share=True):
    model = SSPSR(n_subs=n_subs, n_ovls=n_ovls, n_colors=128, n_blocks=n_blocks, 
                  n_feats=n_feats, n_scale=n_scale, res_scale=0.1, use_share=use_share, conv=default_conv)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

# 2. Preprocess Image
def preprocess_image(image_path, n_scale):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image.shape[1]//n_scale, image.shape[0]//n_scale))  # Downscale
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = image / 255.0  # Normalize
    image = torch.tensor(image).float().unsqueeze(0)
    return image

# 3. Inference
def infer(model, lr_image, device):
    lr_image = lr_image.to(device)
    with torch.no_grad():
        sr_image = model(lr_image, lr_image)
    sr_image = sr_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    sr_image = np.clip(sr_image * 255.0, 0, 255).astype(np.uint8)
    return sr_image

# 4. Save Output
def save_image(image, output_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    model_path = './checkpoints/Chikusei_Chikusei_SSPSR_Blocks=3_Subs8_Ovls2_Feats=256_ckpt_epoch_40.pth'
    image_path = './test_images/low_res_image.jpg'
    output_path = './results/super_resolved_image.jpg'
    
    # Model parameters (match those from training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_subs, n_ovls, n_blocks, n_feats, n_scale = 8, 2, 3, 256, 4  # Adjust to match your training

    # Load model and image
    model = load_model(model_path, device, n_subs, n_ovls, n_blocks, n_feats, n_scale)
    lr_image = preprocess_image(image_path, n_scale)
    
    # Perform inference
    sr_image = infer(model, lr_image, device)

    # Save or display the output
    save_image(sr_image, output_path)
    print(f"Super-resolved image saved at {output_path}")

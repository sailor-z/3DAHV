import os, random
from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import pickle

def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    try: torch.manual_seed(s)
    except NameError: pass
    try: torch.cuda.manual_seed_all(s)
    except NameError: pass
    try: np.random.seed(s%(2**32-1))
    except NameError: pass
    random.seed(s)
    if reproducible:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_cuda(data):
    if type(data)==list:
        results = []
        for i, item in enumerate(data):
            results.append(to_cuda(item))
        return results
    elif type(data)==dict:
        results={}
        for k,v in data.items():
            results[k]=to_cuda(v)
        return results
    elif type(data).__name__ == "Tensor":
        return data.cuda()
    else:
        return data

def one_batch(dl):
    return next(iter(dl))

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def mk_folders(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def get_permutations(num_images, eval_time=False):
    if not eval_time:
        permutations = []
        for i in range(1, num_images):
            for j in range(num_images - 1):
                if i > j:
                    permutations.append((j, i))
    else:
        permutations = []
        for i in range(0, num_images):
            for j in range(0, num_images):
                if i != j:
                    permutations.append((j, i))

    return permutations

def get_calibration_matrix_K_from_blender(lens, sensor_width, resolution_x, resolution_y):
    f_in_mm = lens
    resolution_x_in_px = resolution_x
    resolution_y_in_px = resolution_y
    scale = 1
    sensor_width_in_mm = sensor_width
    sensor_height_in_mm = sensor_width

    pixel_aspect_ratio = 1

    s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
    s_v = resolution_y_in_px * scale / sensor_height_in_mm

    # Parameters of intrinsic calibration matrix K
    alpha_u = f_in_mm * s_u
    alpha_v = f_in_mm * s_v
    u_0 = resolution_x_in_px * scale / 2
    v_0 = resolution_y_in_px * scale / 2
    skew = 0 # only use rectangular pixels

    K = np.array(
        ((alpha_u, skew,    u_0),
        (    0  , alpha_v, v_0),
        (    0  , 0,        1 )))
    return K

def rotate_volume(volume, rotation_matrix, padding_mode='zeros'):
    """
    Rotate a 3D volume by applying a 3D rotation matrix to it.
    Args:
        volume (torch.Tensor): a tensor representing the 3D volume with shape (N, C, D, H, W)
        rotation_matrix (torch.Tensor): a 3x3 rotation matrix with shape (N, 3, 3)
    Returns:
        A rotated 3D volume with the same shape as the input volume.
    """
    # Reshape the rotation matrix to shape (N, 3, 4)
    rotation_matrix = torch.cat([rotation_matrix, torch.zeros(rotation_matrix.size(0), 3, 1, device=rotation_matrix.device)], dim=-1)

    # Create a grid of coordinates with the same shape as the input volume
    grid = F.affine_grid(rotation_matrix, volume.size(), align_corners=False)

    # Apply the rotation to the input volume using bilinear interpolation
    rotated_volume = F.grid_sample(volume, grid, padding_mode=padding_mode, align_corners=False)

    return rotated_volume

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, C, L], sequence
    """
    x = x.flatten(2)
    N, C, L = x.shape  # batch, dim, length

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    gate = (torch.rand(N, device=x.device) > 0.5).float()
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

    # generate the binary mask: 1 is keep, 0 is remove
    mask = torch.zeros([N, L], device=x.device)
    mask[:, :len_keep] = 1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_shuffle)

    mask = (mask + gate[:, None]) > 0

    return mask.float()

def skew_symmetric(T):
    T = T.reshape(3)
    Tx = np.array([[0, -T[2], T[1]],
                    [T[2], 0, -T[0]],
                    [-T[1], T[0], 0]])
    return Tx

def resize_pad(im, dim, mode=T.InterpolationMode.BILINEAR):
    _, h, w = im.shape
    im = T.functional.resize(im, int(dim * min(w, h) / max(w, h)), interpolation=mode)
    left = int(np.ceil((dim - im.shape[2]) / 2))
    right = int(np.floor((dim - im.shape[2]) / 2))
    top = int(np.ceil((dim - im.shape[1]) / 2))
    bottom = int(np.floor((dim - im.shape[1]) / 2))
    im = T.functional.pad(im, (left, top, right, bottom))

    return im

def bbx_resize(bbx, img_w, img_h, scale_ratio=1.0):
    w, h = bbx[2] - bbx[0], bbx[3] - bbx[1]
    dim = scale_ratio * max(w, h)

    left = int(np.ceil((dim - w) / 2))
    right = int(np.floor((dim - w) / 2))
    top = int(np.ceil((dim - h) / 2))
    bottom = int(np.floor((dim - h) / 2))

    bbx[0] = max(bbx[0] - left, 0)
    bbx[1] = max(bbx[1] - top, 0)
    bbx[2] = min(bbx[2] + right, img_w)
    bbx[3] = min(bbx[3] + bottom, img_h)

    return bbx

def square_bbox(bbox, padding=0.0, astype=None):
    """
    Computes a square bounding box, with optional padding parameters.

    Args:
        bbox: Bounding box in xyxy format (4,).

    Returns:
        square_bbox in xyxy format (4,).
    """
    if astype is None:
        astype = type(bbox[0])
    bbox = np.array(bbox)
    center = (bbox[:2] + bbox[2:]) / 2
    extents = (bbox[2:] - bbox[:2]) / 2
    s = max(extents) * (1 + padding)
    square_bbox = np.array(
        [center[0] - s, center[1] - s, center[0] + s, center[1] + s],
        dtype=astype,
    )
    return square_bbox


def crop(img, bbx):
    if len(img.shape) < 4:
        crop_img = img[int(bbx[1]):int(bbx[3]), int(bbx[0]):int(bbx[2])]
    else:
        crop_img = [img[i, int(bbx[i, 1]):int(bbx[i, 3]), int(bbx[i, 0]):int(bbx[i, 2])] for i in range(img.shape[0])]
    return crop_img

def jitter_bbox(bbox, jitter_scale, jitter_trans, img_shape):
    s = (jitter_scale[1] - jitter_scale[0]) * torch.rand(1).item() + jitter_scale[0]
    tx = (jitter_trans[1] - jitter_trans[0]) * torch.rand(1).item() + jitter_trans[0]
    ty = (jitter_trans[1] - jitter_trans[0]) * torch.rand(1).item() + jitter_trans[0]

    side_length = np.array([bbox[2] - bbox[0], bbox[3] - bbox[1]])
    center = (bbox[:2] + bbox[2:]) / 2 + np.array([tx, ty]) * side_length
    extent = side_length / 2 * s

    # Final coordinates need to be integer for cropping.
    ul = center - extent
    lr = ul + 2 * extent

    ul = np.maximum(ul, np.zeros(2))
    lr = np.minimum(lr, np.array([img_shape[1], img_shape[0]]))

    return np.concatenate((ul, lr))

def load(model_cpkt_path):
    checkpoint = torch.load(model_cpkt_path)
    self.feature_aligner.module.load_state_dict(checkpoint['predictor_state_dict'])
    self.feature_extractor.module.load_state_dict(checkpoint['feature_extractor_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def draw_pose_axis(img, K, R, T, bbx_3d=None, color=None, crop_params=None, centered=False):
    thickness = 5

    if bbx_3d is not None:
        radius = 0.6 * np.linalg.norm(bbx_3d, axis=1).mean()
    else:
        radius = 0.4

    aPts = np.array([[0,0,0],[0,0,radius],[0,radius,0],[radius,0,0]])

    rep = np.matmul(K, np.matmul(R, aPts.T) + T)
    rep = rep / rep[2:]
    rep = rep[:2]

    if crop_params is not None:
        rep = (rep - crop_params[:2, None]) / crop_params[2:, None]

    if centered is True:
        delta_x = 0.5*img.shape[1] - rep[0, 0]
        delta_y = 0.5*img.shape[0] - rep[1, 0]

        x = np.int32(rep[0] + delta_x + 0.5)
        y = np.int32(rep[1] + delta_y + 0.5)
    else:
        x = np.int32(rep[0] + 0.5)
        y = np.int32(rep[1] + 0.5)

    img = np.ascontiguousarray(img)
    if color is None:
        img = cv2.arrowedLine(img, (x[0],y[0]), (x[1],y[1]), (0,0,255), thickness, cv2.LINE_AA)
        img = cv2.arrowedLine(img, (x[0],y[0]), (x[2],y[2]), (0,255,0), thickness, cv2.LINE_AA)
        img = cv2.arrowedLine(img, (x[0],y[0]), (x[3],y[3]), (255,0,0), thickness, cv2.LINE_AA)
    else:
        img = cv2.arrowedLine(img, (x[0],y[0]), (x[1],y[1]), color, thickness, cv2.LINE_AA)
        img = cv2.arrowedLine(img, (x[0],y[0]), (x[2],y[2]), color, thickness, cv2.LINE_AA)
        img = cv2.arrowedLine(img, (x[0],y[0]), (x[3],y[3]), color, thickness, cv2.LINE_AA)
    return img

def visualization(cfg, img_src, img_tgt, pred_delta_R, R_src, R_tgt, T_tgt, crop_params_tgt, K, bbox_3d):
    R_pred = torch.bmm(pred_delta_R, R_src)[0].cpu().numpy()
    R_tgt, T_tgt = R_tgt[0].cpu().numpy(), T_tgt[0].cpu().numpy()
    crop_params_tgt = crop_params_tgt[0].cpu().numpy()

    image_norm_mean = np.array(cfg["DATA"]["PIXEL_MEAN"])
    image_norm_std = np.array(cfg["DATA"]["PIXEL_STD"])
    img_tgt = img_tgt[0].permute(1, 2, 0).cpu().numpy() * image_norm_std + image_norm_mean
    img_tgt = (255*img_tgt).astype(np.uint8)
    img_src = img_src[0].permute(1, 2, 0).cpu().numpy() * image_norm_std + image_norm_mean
    img_src = (255*img_src).astype(np.uint8)

    viz = np.ascontiguousarray(img_tgt)
    viz = draw_pose_axis(viz, K, R_tgt, T_tgt, bbox_3d, (0, 255, 0), crop_params_tgt, centered=True)
    viz = draw_pose_axis(viz, K, R_pred, T_tgt, bbox_3d, (255, 0, 0), crop_params_tgt, centered=True)

    return img_src, img_tgt, viz

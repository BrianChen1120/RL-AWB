import numpy as np
from scipy.stats import skew
import cv2
import os
from scipy.ndimage import laplace, convolve
from scipy.signal import convolve2d
import scipy.io
from numpy.lib.stride_tricks import sliding_window_view

try:
    from utils.WBsRGB import rgb_uv_hist
except:
    from WBsRGB import rgb_uv_hist


def angerr2(l1, l2):
    l1 = l1 / (np.linalg.norm(l1) + 1e-12)
    l2 = l2 / (np.linalg.norm(l2) + 1e-12)
    rec = np.degrees(np.arccos(np.clip(np.sum(l1 * l2), -1, 1)))
    LL = l2 / (l1 + 1e-12)
    rep = np.degrees(np.arccos(np.dot(LL, np.ones(3)) /
                               (np.sqrt(3) * np.sqrt(np.sum(LL ** 2)))))
    return rec, rep


def set_border(inp, width, method=1):
    temp = np.ones_like(inp)
    rr, cc = inp.shape
    y, x = np.ogrid[:rr, :cc]
    temp *= ((x < (cc - width)) & (x + 1 > width))
    temp *= ((y < (rr - width)) & (y + 1 > width))
    out = temp * inp
    if method == 1:
        if np.sum(temp) != 0:
            avg_val = np.sum(out) / np.sum(temp)
        else:
            avg_val = 0
        out = out + avg_val * (np.ones_like(inp) - temp)
    return out


def dilation33(inp, it=1):
    inp = np.array(inp)
    hh, ll = inp.shape

    for _ in range(it):
        channel0 = np.vstack((inp[1:, :], inp[-1:, :]))
        channel1 = inp.copy()
        channel2 = np.vstack((inp[0:1, :], inp[:-1, :]))

        temp = np.stack((channel0, channel1, channel2), axis=2)
        out2 = np.max(temp, axis=2)

        channel0_h = np.hstack((out2[:, 1:], out2[:, -1:]))
        channel1_h = out2.copy()
        channel2_h = np.hstack((out2[:, 0:1], out2[:, :ll - 1]))

        temp2 = np.stack((channel0_h, channel1_h, channel2_h), axis=2)
        inp = np.max(temp2, axis=2)

    return inp


def updated_saliency_map(sRGBImage, mask, VarThreshold, ColorThreshold, kernel_size=11):
    mask = mask.astype(bool)

    r_ln = np.zeros_like(sRGBImage[:, :, 0], dtype=float)
    g_ln = np.zeros_like(sRGBImage[:, :, 1], dtype=float)
    b_ln = np.zeros_like(sRGBImage[:, :, 2], dtype=float)

    r_ln[mask] = np.log(sRGBImage[:, :, 0][mask] + 1)
    g_ln[mask] = np.log(sRGBImage[:, :, 1][mask] + 1)
    b_ln[mask] = np.log(sRGBImage[:, :, 2][mask] + 1)

    stacked = np.stack((r_ln, g_ln, b_ln), axis=2)
    variance_map = np.var(stacked, axis=2, ddof=1)

    updated_saliencyMap = np.zeros_like(variance_map, dtype=float)
    updated_saliencyMap[(variance_map > VarThreshold) & mask] = 1

    Mr = np.mean(r_ln[mask]) if np.any(mask) else 0
    Mg = np.mean(g_ln[mask]) if np.any(mask) else 0
    Mb = np.mean(b_ln[mask]) if np.any(mask) else 0
    Minimum = np.min([Mr, Mg, Mb])

    Xr = np.zeros_like(r_ln)
    Xg = np.zeros_like(g_ln)
    Xb = np.zeros_like(b_ln)

    Xr[mask] = np.abs(r_ln[mask] - Mr)
    Xg[mask] = np.abs(g_ln[mask] - Mg)
    Xb[mask] = np.abs(b_ln[mask] - Mb)

    threshold = ColorThreshold * Minimum
    difference_map = np.maximum(np.maximum(Xr, Xg), Xb)

    not_important_mask = (difference_map > threshold) & mask

    updated_saliencyMap2 = updated_saliencyMap.copy()
    updated_saliencyMap2[not_important_mask] = 0

    return updated_saliencyMap, updated_saliencyMap2


def compute_edge_confidence(image, mask, bitDepth, E, edge_threshold):
    image = image / (2**bitDepth - 1)
    image = np.clip(image, 0, 1)

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    OW = (R + G + B) / 3.0
    OW = OW * mask

    nonzero = OW[OW != 0]
    if nonzero.size == 0:
        OWskew = 0
        m = 1.0
    else:
        OWskew = skew(nonzero)
        m = np.mean(nonzero)

    if OWskew > 1.5:
        E = 1.0
    elif OWskew > 0.2:
        E = 2.0
    else:
        E = 4.0

    edge_weights = 1 - np.exp(-((OW / m) ** E))
    edge_weights[edge_weights < edge_threshold] = 0

    if np.sum(edge_weights) == 0:
        edge_weights = 1 - np.exp(-((OW / m) ** E))

    return edge_weights


def compute_derivative(channel, order, sigma):
    kernel_size = int(np.ceil(6 * sigma))
    if kernel_size % 2 == 0:
        kernel_size += 1

    k = cv2.getGaussianKernel(kernel_size, sigma)
    G = k * k.T

    smoothed = cv2.filter2D(channel, -1, G, borderType=cv2.BORDER_REPLICATE)

    if order == 1:
        grad_y, grad_x = np.gradient(smoothed)
        derivative = np.sqrt(grad_x ** 2 + grad_y ** 2)
    else:
        derivative = np.abs(laplace(smoothed))

    return derivative


def estimate_illuminant_pixelwise_accelerated(image, order, p, sigma, mask, E, edge_threshold, window_size=3, bitDepth=14):
    if image.dtype != np.float64:
        image = image.astype(np.float64) / 255.0

    edge_weights = compute_edge_confidence(image, mask, bitDepth, E, edge_threshold)

    pad_size = window_size // 2
    padded_derivatives = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='edge')

    numerator = np.zeros(3)
    denominator = np.zeros(3)

    for c in range(3):
        windows = sliding_window_view(padded_derivatives[:, :, c], (window_size, window_size))

        max_vals = np.max(windows, axis=(-1, -2))

        nonzero_mask = windows != 0
        sum_nonzero = np.sum(np.where(nonzero_mask, windows, 0), axis=(-1, -2))
        count_nonzero = np.sum(nonzero_mask, axis=(-1, -2))
        mean_vals = np.divide(sum_nonzero, count_nonzero, out=np.zeros_like(sum_nonzero), where=(count_nonzero != 0))

        valid = max_vals > 0

        center_weight = edge_weights
        norm_center = np.zeros_like(mean_vals)
        norm_center[valid] = mean_vals[valid] / max_vals[valid]

        weighted_val = mean_vals * center_weight
        weighted_norm = norm_center * center_weight

        numerator[c] = np.sum(np.abs(weighted_val[valid]) ** p)
        denominator[c] = np.sum(np.abs(weighted_norm[valid]) ** p)

    illuminant = np.zeros(3)
    for c in range(3):
        if denominator[c] > 0:
            illuminant[c] = (numerator[c] / denominator[c]) ** (1 / p)

    norm_val = np.linalg.norm(illuminant)
    if norm_val > 0:
        illuminant = illuminant / norm_val

    return illuminant


def deriv_gauss(img, sigma):
    GaussianDieOff = 1e-6
    pw = np.arange(1, 51)
    ssq = sigma ** 2
    exp_vals = np.exp(-(pw ** 2) / (2 * ssq))
    valid = np.where(exp_vals > GaussianDieOff)[0]
    if valid.size > 0:
        width = valid[-1] + 1
    else:
        width = 1

    xs = np.arange(-width, width + 1)
    ys = np.arange(-width, width + 1)
    x, y = np.meshgrid(xs, ys)

    dgau2D = -x * np.exp(-(x ** 2 + y ** 2) / (2 * ssq)) / (np.pi * ssq)

    ax = convolve(img, dgau2D, mode='nearest')
    ay = convolve(img, dgau2D.T, mode='nearest')

    mag = np.sqrt(ax ** 2 + ay ** 2)
    return mag


def normr(data):
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms[norms == 0] = np.finfo(float).eps
    return data / norms


def gray_index_angular(img, mask, sigma, percentage):
    eps_val = np.finfo(float).eps
    rr, cc, dd = img.shape

    R = img[:, :, 0].copy()
    G = img[:, :, 1].copy()
    B = img[:, :, 2].copy()

    R[R == 0] = eps_val
    G[G == 0] = eps_val
    B[B == 0] = eps_val

    Mr = deriv_gauss(np.log(R), sigma)
    Mg = deriv_gauss(np.log(G), sigma)
    Mb = deriv_gauss(np.log(B), sigma)

    data = np.column_stack((Mr.ravel(order='F'), Mg.ravel(order='F'), Mb.ravel(order='F'))).astype(np.float64)

    data[data[:, 0] == 0, 0] = eps_val
    data[data[:, 1] == 0, 1] = eps_val
    data[data[:, 2] == 0, 2] = eps_val

    data_normed = normr(data)
    gt1 = normr(np.ones_like(data))

    dot_product = np.sum(data_normed * gt1, axis=1)
    dot_product = np.clip(dot_product, -1, 1)
    angular_error = np.arccos(dot_product)

    Greyidx_angular = angular_error.reshape((rr, cc), order='F').astype(np.float64)

    max_val = np.max(Greyidx_angular)
    Greyidx = Greyidx_angular / (max_val + eps_val)

    condition = (Mr < eps_val) & (Mg < eps_val) & (Mb < eps_val)
    Greyidx[condition] = np.max(Greyidx)
    Greyidx_angular[condition] = np.max(Greyidx_angular)

    kernel = np.ones((7, 7), dtype=np.float64) / 49.0
    Greyidx = convolve2d(Greyidx, kernel, mode='same', boundary='wrap')
    Greyidx_angular = convolve2d(Greyidx_angular, kernel, mode='same', boundary='wrap')

    if mask is not None and mask.size > 0:
        Greyidx_angular[mask.astype(bool)] = np.max(Greyidx_angular)

    threshold1 = np.percentile(Greyidx_angular.ravel(order='F'), percentage)
    threshold2 = np.percentile(Greyidx_angular.ravel(order='F'), 0.5)

    binary_mask1 = np.zeros_like(Greyidx_angular).astype(np.float64)
    binary_mask1[Greyidx_angular <= threshold1] = 1

    binary_mask2 = np.zeros_like(Greyidx_angular).astype(np.float64)
    binary_mask2[Greyidx_angular <= threshold2] = 1

    return binary_mask1, binary_mask2


def imresize_nearest(img, scale):
    in_h, in_w = img.shape[:2]
    out_h = int(np.round(in_h * scale))
    out_w = int(np.round(in_w * scale))

    row_indices = np.clip(np.round((np.arange(out_h) + 0.5) / scale - 0.5).astype(int), 0, in_h - 1)
    col_indices = np.clip(np.round((np.arange(out_w) + 0.5) / scale - 0.5).astype(int), 0, in_w - 1)

    if img.ndim == 3:
        resized = img[row_indices[:, np.newaxis], col_indices, :]
    else:
        resized = img[row_indices[:, np.newaxis], col_indices]

    return resized


def RGB_estimation(index, action, load_img=None, dataset='NCC'):
    base_path = f"./dataset/{dataset}_dataset"

    gt_mat_path = os.path.join(base_path, "gt.mat")
    gt_data = scipy.io.loadmat(gt_mat_path)

    if "gts" in gt_data:
        gt = gt_data["gts"]
    elif "gt" in gt_data:
        gt = gt_data["gt"]
    elif "real_rgb" in gt_data:
        gt = gt_data["real_rgb"]
    else:
        gt = None

    img_path = os.path.join(base_path, "img")
    msk_path = os.path.join(base_path, "msk")

    imname = f"{index}.png"
    img_full_path = os.path.join(img_path, imname)
    mask_full_path = os.path.join(msk_path, imname)

    img = cv2.imread(img_full_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_full_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)

    mask = cv2.imread(mask_full_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_full_path}")
    mask = mask > 0

    if dataset == 'LEVI':
        if index < 371:
            img = imresize_nearest(img, 0.25)
            mask = imresize_nearest(mask.astype(np.uint8), 0.25).astype(bool)
            bitDepth = 12
        else:
            img = imresize_nearest(img, 0.125)
            mask = imresize_nearest(mask.astype(np.uint8), 0.125).astype(bool)
            bitDepth = 14
    elif dataset == 'NCC':
        img = imresize_nearest(img, 0.125)
        mask = imresize_nearest(mask.astype(np.uint8), 0.125).astype(bool)
        bitDepth = 14
    elif dataset == 'Gehler':
        img = imresize_nearest(img, 0.125)
        mask = imresize_nearest(mask.astype(np.uint8), 0.125).astype(bool)
        bitDepth = 14
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if load_img is not None:
        img = load_img.astype(np.float64)

    saturation_threshold = np.max(img) * 0.95
    max_img = np.max(img, axis=2)
    dilated = dilation33((max_img >= saturation_threshold).astype(np.float64))
    mask_im2 = mask.astype(np.float64) + dilated
    mask_im2 = (mask_im2 == 0).astype(np.float64)
    mask_proc = set_border(mask_im2, 1, method=0)
    mask_proc = 1 - mask_proc

    binary_mask1, binary_mask2 = gray_index_angular(img, mask_proc, action[0], action[1])

    saliencyMap1, saliencyMap2 = updated_saliency_map(img, 1 - mask, action[2], action[3])

    if dataset == 'Gehler':
        a1 = binary_mask2
        a2 = binary_mask1
    else:
        a1 = binary_mask2 * saliencyMap1
        a2 = binary_mask1 * saliencyMap2

    Npixels = img.shape[0] * img.shape[1]
    numGPs = int(np.floor(0.05 * Npixels / 100))
    if np.count_nonzero(a2) < numGPs:
        a = a1
        for_RL = False
    else:
        a = a2
        for_RL = True

    img_masked = img * a[:, :, np.newaxis]

    order = 2
    illuminant = estimate_illuminant_pixelwise_accelerated(
        img_masked, order, action[4], action[5], a, action[6],
        action[7], int(action[9]), bitDepth=bitDepth
    )
    EvaLum = illuminant
    cc_mask_img = img * np.expand_dims(~mask, axis=-1)
    feature_vec = rgb_uv_hist(cc_mask_img)

    if gt is not None:
        gt_val = gt[index - 1, :].flatten()
        arr, arr_rep = angerr2(EvaLum, gt_val)
    else:
        arr, arr_rep = None, None
        gt_val = None

    return arr, arr_rep, EvaLum, gt_val, [for_RL, None], feature_vec.flatten()
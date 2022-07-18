import cv2
import numpy as np


def color_transfer(target):
    target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    # compute color statistics for the source and target images
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = (
        145.32921, 22.719374, 143.36488, 2.6790402, 147.16614, 2.9267197)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = image_stats(target)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    # scale by the standard deviations
    l = (lStdSrc / lStdTar) * l
    a = (aStdSrc / aStdTar) * a
    b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip the pixel intensities to [0, 255] if they fall outside
    # this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # return the color transferred image
    return transfer


def image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return (lMean, lStd, aMean, aStd, bMean, bStd)


def tensor2im(image_tensor, size=None, imtype=np.uint16, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    if size:
        image_numpy = cv2.resize(image_numpy, size)
    if normalize:
        image_numpy = (image_numpy + 1) / 2.0 * 65535.0
    else:
        image_numpy = image_numpy * 65535.0
    image_numpy = np.clip(image_numpy, 0, 65535)
    if len(image_numpy.shape) == 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)

def dpmap2verts(verts, tris, texcoords, dpmap, scale=0.914):
    dpmap = np.array(dpmap).astype(int)
    normals = np.zeros(verts.shape)
    tri_verts = verts[tris]
    n0 = np.cross(tri_verts[::, 1] - tri_verts[::, 0], tri_verts[::, 2] - tri_verts[::, 0])
    n0 = n0 / np.linalg.norm(n0, axis=1)[:, np.newaxis]
    for i in range(tris.shape[0]):
        normals[tris[i, 0]] += n0[i]
        normals[tris[i, 1]] += n0[i]
        normals[tris[i, 2]] += n0[i]
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

    pos_u = dpmap.shape[0] - (texcoords[:, 1] * dpmap.shape[0]).astype(int)
    pos_v = (texcoords[:, 0] * dpmap.shape[1]).astype(int)
    verts += normals * (dpmap[pos_u, pos_v] - 32768)[:, np.newaxis] / 32768 * scale
    return verts


def preserve_largest_area(faces):
    if len(faces) <= 1:
        return faces
    areas = []
    for i in range(len(faces)):
        left, top, right, bottom = faces[i][:4]
        areas.append((right - left) * (bottom - top))
    largest_idx = np.argmax(np.array(areas))
    return faces[largest_idx: largest_idx + 1]


def rescale_disp(disp):
    return disp / 0.914 * 32768 + 32768


def scale_disp(disp):
    return (disp - 32768) / 32768 * 0.914


def filter_dpmap(img):
    """
    Args:
        img: uint16
    Returns:
        uint16
    """
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    img = scale_disp(img.astype(np.float32))
    img_filtered = cv2.bilateralFilter(img, 7, 2, 0.6)  # Remove high frequency noise
    img_blur = cv2.GaussianBlur(img_filtered, (11, 11), 10)  # Remove low frequency error
    img_residual = img_filtered - img_blur
    img_residual = rescale_disp(img_residual).astype(np.uint16)
    return img_residual


def get_device(gpu_ids):
    if len(gpu_ids) > 0:
        return f'cuda:{gpu_ids[0]}'
    else:
        return 'cpu'

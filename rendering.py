import os
import argparse
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from joblib import Parallel, delayed

from pbrdf import PBRDF


pangles = [0, 45, 90, 135]


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def render(i, pbrdf, n, L, stokes, H, theta_d, Q):
    nL = L @ n
    theta_h = np.arccos(np.clip(H @ n, -1., 1.))
    norm = np.linalg.norm(H - n, axis=1, keepdims=True)
    norm[norm == 0] = 1
    P = (H - n) / norm
    phi_d = np.arccos(np.clip(np.sum(P * Q, axis=1), -1., 1.))

    ret = np.zeros((4,) + L.shape, dtype=float)
    D = np.array([
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0.5, -0.5, 0],
        [0.5, 0, -0.5],
        ], dtype=float)
    for j in range(len(L)):
        if nL[j] <= 0: continue

        matrix = np.array(pbrdf.get_interpolated_mueller_matrix(theta_h=theta_h[j], theta_d=theta_d[j], phi_d=phi_d[j]), dtype=float)
        blue = matrix[0] @ stokes[j]
        green = matrix[2] @ stokes[j]
        red = matrix[4] @ stokes[j]
        ret[:, j, 0] = D @ red[:3]
        ret[:, j, 1] = D @ green[:3]
        ret[:, j, 2] = D @ blue[:3]
        ret = np.nan_to_num(ret)
        # ret[ret < 0] = 0

    return i, ret


def main(args):
    obj_names = np.loadtxt(args.obj_file, dtype=str)
    N_map = np.load(args.N_map_file)
    mask = cv2.imread(args.mask_file, 0)
    N = N_map[mask > 0]

    L = np.loadtxt(args.L_file)
    if args.stokes_file is None:
        stokes = np.tile(np.array([[1, 0, 0, 0]]), (len(L), 1))
    else:
        stokes = np.loadtxt(args.stokes_file)
    v = np.array([0., 0., 1.], dtype=float)
    H = (L + v) / np.linalg.norm(L + v, axis=1, keepdims=True)
    theta_d = np.arccos(np.sum(L * H, axis=1))
    norm = np.linalg.norm(L - H, axis=1, keepdims=True)
    norm[norm == 0] = 1
    Q = (L - H) / norm

    for i_obj, obj_name in enumerate(obj_names[args.obj_range[0]:args.obj_range[1]]):
        print('===== {} - {} start ====='.format(i_obj, obj_name))

        obj_name = str(obj_name)

        pbrdf = PBRDF(os.path.join(args.pbrdf_dir, obj_name + '_matlab', obj_name + '_pbrdf.mat'))

        ret = Parallel(n_jobs=args.n_jobs, verbose=5, prefer='threads')([delayed(render)(i, pbrdf, n, L, stokes, H, theta_d, Q) for i, n in enumerate(N)])
        ret.sort(key=lambda x: x[0])
        M = np.array([x[1] for x in ret], dtype=float)
        if args.save_type != 'raw':
            M = M / M.max()

        pimgs = np.zeros((len(L), 4) + N_map.shape)
        pimgs[:, :, mask > 0] = M.transpose(2, 1, 0, 3)

        out_path = os.path.join(args.out_dir, obj_name)
        makedirs(out_path)

        print('Saving images...')
        fnames = []
        for i, imgs in enumerate(tqdm(pimgs)):
            if args.save_type == 'npy' or args.save_type == 'raw':
                for img, pangle in zip(imgs, pangles):
                    fname = '{:03d}_{:03d}.npy'.format(i + 1, pangle)
                    fnames.append(fname)
                    np.save(os.path.join(out_path, fname), img)
            elif args.save_type == 'png':
                for img, pangle in zip(imgs, pangles):
                    fname = '{:03d}_{:03d}.png'.format(i + 1, pangle)
                    fnames.append(fname)
                    img = img * np.iinfo(np.uint16).max
                    img = img[..., ::-1]
                    cv2.imwrite(os.path.join(out_path, fname), img.astype(np.uint16))
        np.save(os.path.join(out_path, 'normal_gt.npy'), N_map)
        shutil.copyfile(args.mask_file, os.path.join(out_path, 'mask.png'))
        shutil.copyfile(args.L_file, os.path.join(out_path, 'light_directions.txt'))

        print('===== {} - {} done ====='.format(i_obj, obj_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pbrdf_dir', type=str)
    parser.add_argument('--obj_file', type=str)
    parser.add_argument('--obj_range', type=int, nargs=2)
    parser.add_argument('--N_map_file', type=str)
    parser.add_argument('--mask_file', type=str)
    parser.add_argument('--L_file', type=str)
    parser.add_argument('--stokes_file', type=str, default=None)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--save_type', type=str, default='npy')
    parser.add_argument('--n_jobs', type=int)
    args = parser.parse_args()

    main(args)

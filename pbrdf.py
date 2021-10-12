import numpy as np
from scipy.io import loadmat


class PBRDF:
    def __init__(self, filepath):
        data = loadmat(filepath)
        self.pbrdf = data["pbrdf_T_f"]
        self.theta_h = data["theta_h"][0]
        self.theta_d = data["theta_d"][0]
        self.phi_d = data["phi_d"][0]
        self.wvls = data["wvls"][0]
        self.num_theta_h = len(self.theta_h)
        self.num_theta_d = len(self.theta_d)
        self.num_phi_d = len(self.phi_d)
    
    def get_raw_mueller_matrix(self, theta_h, theta_d, phi_d):
        idx_theta_h = self.theta_h_idx(theta_h)
        idx_theta_d = self.theta_d_idx(theta_d)
        idx_phi_d = self.phi_d_idx(phi_d)
        
        return self.pbrdf[idx_phi_d, idx_theta_d, idx_theta_h]

    def get_interpolated_mueller_matrix(self, theta_h, theta_d, phi_d):
        idx_th_p = self.theta_h_idx(theta_h)
        idx_td_p = self.theta_d_idx(theta_d)
        idx_pd_p = self.phi_d_idx(phi_d)

        # Calculate the indexes for interpolation
        idx_th_p = idx_th_p if idx_th_p < self.num_theta_h - 1 else self.num_theta_h - 2
        idx_td_p = idx_td_p if idx_td_p < self.num_theta_d - 1 else self.num_theta_d - 2

        idx_th = [idx_th_p, idx_th_p + 1]
        idx_td = [idx_td_p, idx_td_p + 1]
        idx_pd = [idx_pd_p, idx_pd_p + 1]

        # Calculate the weights
        weight_th = [abs(self.theta_h[i] - theta_h) for i in idx_th]
        weight_td = [abs(self.theta_d[i] - theta_d) for i in idx_td]
        weight_pd = [abs(self.phi_d[i] - phi_d) for i in idx_pd]

        # Normalize the weights
        weight_th = [1 - w / sum(weight_th) for w in weight_th]
        weight_td = [1 - w / sum(weight_td) for w in weight_td]
        weight_pd = [1 - w / sum(weight_pd) for w in weight_pd]

        idx_pd[1] = idx_pd[1] if idx_pd[1] < self.num_phi_d else 0

        matrix = 0
        for ith, wth in zip(idx_th, weight_th):
            for itd, wtd in zip(idx_td, weight_td):
                for ipd, wpd in zip(idx_pd, weight_pd):
                    matrix += wth * wtd * wpd * self.pbrdf[ipd, itd, ith]

        return matrix
    
    def theta_h_idx(self, theta_h):
        if theta_h < 0:
            return 0
        idx = np.argmin(np.abs(self.theta_h - theta_h))
        if theta_h < self.theta_h[idx] and idx != 0:
            idx = idx - 1
        return idx
        
    def theta_d_idx(self, theta_d):
        idx = np.argmin(np.abs(self.theta_d - theta_d))
        if theta_d < self.theta_d[idx] and idx != 0:
            idx = idx - 1
        return idx

    def phi_d_idx(self, phi_d):
        while phi_d < -np.pi:
            phi_d += 2 * np.pi
        while phi_d > np.pi:
            phi_d -= 2 * np.pi

        idx = np.argmin(np.abs(self.phi_d - phi_d))
        if phi_d < self.phi_d[idx] and idx != 0:
            idx = idx - 1
        return idx


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pbrdf_dir', type=str, help='Path to a directory storing the MERL binary files.')
    parser.add_argument('--material_name_file', type=str, help='Path to a text file of material names.')
    args = parser.parse_args()

    mat_name = np.loadtxt(args.material_name_file, dtype=str)
    pbrdf = PBRDF(os.path.join(args.pbrdf_dir, mat_name[0] + '_matlab', mat_name[0] + '_pbrdf.mat'))

    print(pbrdf.phi_d.min(), pbrdf.phi_d.max())
    print(np.count_nonzero(np.isnan(pbrdf.pbrdf)))

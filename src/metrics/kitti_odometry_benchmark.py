# pylint: skip-file

# Copyright (C) Huangying Zhan 2019. All rights reserved.
# https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py

# Edited by Anton Klochkov.

import enum

import numpy as np
import wandb
from matplotlib import pyplot as plt


class MetricType(enum.Enum):
    TRANS_ERROR = "Translational error (%)"
    ROT_ERROR = "Rotational error (deg/100m)"
    ATE = "ATE (m)"
    RPE_M = "RPE (m)"
    RPE_DEG = "RPE (deg)"


def scale_lse_solver(X, Y):
    """Least-sqaure-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y) / np.sum(X ** 2)
    return scale


def umeyama_alignment(x, y, with_scale=False):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


class KittiEvalOdom:
    """Evaluate odometry result
    Usage example:
        vo_eval = KittiEvalOdom()
        vo_eval.eval(gt_pose_txt_dir, result_pose_txt_dir)
    """

    def __init__(self):
        self.lengths = [100, 200, 300, 400, 500, 600, 700, 800]
        self.num_lengths = len(self.lengths)

    @staticmethod
    def trajectory_distances(poses):
        """Compute distance for each pose w.r.t frame-0
        Args:
            poses (list): [4x4 array]
        Returns:
            dist (float list): distance of each pose w.r.t frame-0
        """
        dist = [0]
        for i in range(len(poses) - 1):
            cur_frame_idx = i
            next_frame_idx = i + 1
            P1 = poses[cur_frame_idx]
            P2 = poses[next_frame_idx]
            dx = P1[0, 3] - P2[0, 3]
            dy = P1[1, 3] - P2[1, 3]
            dz = P1[2, 3] - P2[2, 3]
            dist.append(dist[i] + np.sqrt(dx ** 2 + dy ** 2 + dz ** 2))
        return dist

    @staticmethod
    def rotation_error(pose_error):
        """Compute rotation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            rot_error (float): rotation error
        """
        a = pose_error[0, 0]
        b = pose_error[1, 1]
        c = pose_error[2, 2]
        d = 0.5 * (a + b + c - 1.0)
        rot_error = np.arccos(max(min(d, 1.0), -1.0))
        return rot_error

    @staticmethod
    def translation_error(pose_error):
        """Compute translation error
        Args:
            pose_error (4x4 array): relative pose error
        Returns:
            trans_error (float): translation error
        """
        dx = pose_error[0, 3]
        dy = pose_error[1, 3]
        dz = pose_error[2, 3]
        trans_error = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return trans_error

    @staticmethod
    def last_frame_from_segment_length(dist, first_frame, length):
        """Find frame (index) that away from the first_frame with
        the required distance
        Args:
            dist (float list): distance of each pose w.r.t frame-0
            first_frame (int): start-frame index
            length (float): required distance
        Returns:
            i (int) / -1: end-frame index. if not found return -1
        """
        for i in range(first_frame, len(dist), 1):
            if dist[i] > (dist[first_frame] + length):
                return i
        return -1

    def calc_sequence_errors(self, poses_gt, poses_result):
        """calculate sequence error
        Args:
            poses_gt (list): {idx: 4x4 array}, ground truth poses
            poses_result (list): {idx: 4x4 array}, predicted poses
        Returns:
            err (list list): [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        """
        err = []
        dist = self.trajectory_distances(poses_gt)
        step_size = 10

        for first_frame in range(0, len(poses_gt), step_size):
            for i in range(self.num_lengths):
                len_ = self.lengths[i]
                last_frame = self.last_frame_from_segment_length(
                    dist, first_frame, len_
                )

                # Continue if sequence not long enough
                if last_frame == -1 or not (last_frame < len(poses_result)) or not (last_frame < len(poses_result)):
                    continue

                # compute rotational and translational errors
                pose_delta_gt = np.dot(
                    np.linalg.inv(poses_gt[first_frame]),
                    poses_gt[last_frame]
                )
                pose_delta_result = np.dot(
                    np.linalg.inv(poses_result[first_frame]),
                    poses_result[last_frame]
                )
                pose_error = np.dot(
                    np.linalg.inv(pose_delta_result),
                    pose_delta_gt
                )

                r_err = self.rotation_error(pose_error)
                t_err = self.translation_error(pose_error)

                # compute speed
                num_frames = last_frame - first_frame + 1.0
                speed = len_ / (0.1 * num_frames)

                err.append([first_frame, r_err / len_, t_err / len_, len_, speed])
        return err

    @staticmethod
    def compute_overall_err(seq_err):
        """Compute average translation & rotation errors
        Args:
            seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
                - r_err (float): rotation error
                - t_err (float): translation error
        Returns:
            ave_t_err (float): average translation error
            ave_r_err (float): average rotation error
        """
        t_err = 0
        r_err = 0

        seq_len = len(seq_err)

        if seq_len > 0:
            for item in seq_err:
                r_err += item[1]
                t_err += item[2]
            ave_t_err = t_err / seq_len
            ave_r_err = r_err / seq_len
            return ave_t_err, ave_r_err
        else:
            return 0, 0

    def plot_error(self, avg_segment_errs, track_name: str):
        """Plot per-length error
        Args:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
            track_name (int): sequence index.
        """
        # Translation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][0] * 100)
            else:
                plot_y.append(0)
        fontsize = 10

        ax1 = plt.subplot()
        plt.plot(plot_x, plot_y, "bs-", label="Translation Error")
        plt.ylabel('Translation Error (%)', fontsize=fontsize)
        plt.xlabel('Path Length (m)', fontsize=fontsize)
        plt.legend(loc="upper right", prop={'size': fontsize})
        image_1 = wandb.Image(ax1, caption=f"Translation Error: {track_name}")
        plt.cla()
        plt.clf()

        # Rotation error
        plot_y = []
        plot_x = []
        for len_ in self.lengths:
            plot_x.append(len_)
            if len(avg_segment_errs[len_]) > 0:
                plot_y.append(avg_segment_errs[len_][1] / np.pi * 180 * 100)
            else:
                plot_y.append(0)
        ax2 = plt.subplot()
        plt.plot(plot_x, plot_y, "bs-", label="Rotation Error")
        plt.ylabel('Rotation Error (deg/100m)', fontsize=fontsize)
        plt.xlabel('Path Length (m)', fontsize=fontsize)
        plt.legend(loc="upper right", prop={'size': fontsize})
        image_2 = wandb.Image(ax2, caption=f"Translation Error: {track_name}")
        plt.cla()
        plt.clf()

        return [image_1, image_2]

    def compute_segment_error(self, seq_errs):
        """This function calculates average errors for different segment.
        Args:
            seq_errs (list list): list of errs; [first_frame, rotation error, translation error, length, speed]
                - first_frame: frist frame index
                - rotation error: rotation error per length
                - translation error: translation error per length
                - length: evaluation trajectory length
                - speed: car speed (#FIXME: 10FPS is assumed)
        Returns:
            avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
        """

        segment_errs = {}
        avg_segment_errs = {}
        for len_ in self.lengths:
            segment_errs[len_] = []

        # Get errors
        for err in seq_errs:
            len_ = err[3]
            t_err = err[2]
            r_err = err[1]
            segment_errs[len_].append([t_err, r_err])

        # Compute average
        for len_ in self.lengths:
            if segment_errs[len_]:
                avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
                avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
                avg_segment_errs[len_] = [avg_t_err, avg_r_err]
            else:
                avg_segment_errs[len_] = []
        return avg_segment_errs

    @staticmethod
    def compute_ate(gt, pred):
        """Compute RMSE of ATE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        """
        errors = []

        for i in range(len(pred)):
            # cur_gt = np.linalg.inv(gt_0) @ gt[i]
            cur_gt = gt[i]
            gt_xyz = cur_gt[:3, 3]

            # cur_pred = np.linalg.inv(pred_0) @ pred[i]
            cur_pred = pred[i]
            pred_xyz = cur_pred[:3, 3]

            align_err = gt_xyz - pred_xyz

            # print('i: ', i)
            # print("gt: ", gt_xyz)
            # print("pred: ", pred_xyz)
            # input("debug")
            errors.append(np.sqrt(np.sum(align_err ** 2)))
        ate = np.sqrt(np.mean(np.asarray(errors) ** 2))
        return ate

    @staticmethod
    def compute_rpe(gt, pred):
        """Compute RPE
        Args:
            gt (4x4 array dict): ground-truth poses
            pred (4x4 array dict): predicted poses
        Returns:
            rpe_trans
            rpe_rot
        """
        trans_errors = []
        rot_errors = []
        for i in range(len(pred) - 1):
            gt1 = gt[i]
            gt2 = gt[i + 1]
            gt_rel = np.linalg.inv(gt1) @ gt2

            pred1 = pred[i]
            pred2 = pred[i + 1]
            pred_rel = np.linalg.inv(pred1) @ pred2
            rel_err = np.linalg.inv(gt_rel) @ pred_rel

            trans_errors.append(KittiEvalOdom.translation_error(rel_err))
            rot_errors.append(KittiEvalOdom.rotation_error(rel_err))
        # rpe_trans = np.sqrt(np.mean(np.asarray(trans_errors) ** 2))
        # rpe_rot = np.sqrt(np.mean(np.asarray(rot_errors) ** 2))
        rpe_trans = np.mean(np.asarray(trans_errors))
        rpe_rot = np.mean(np.asarray(rot_errors))
        return rpe_trans, rpe_rot

    def eval(self, poses_result, poses_gt, track_name: str):
        """Evaulate required/available sequences
        Args:
            poses_gt (list): ground truth poses
            poses_result (list): pose predictions
            track_name: id of track
        """

        # evaluation

        # Pose alignment to first frame
        pred_0 = poses_result[0]
        gt_0 = poses_gt[0]
        for cnt in range(len(poses_result)):
            poses_result[cnt] = np.linalg.inv(pred_0) @ poses_result[cnt]
            poses_gt[cnt] = np.linalg.inv(gt_0) @ poses_gt[cnt]

        # compute sequence errors
        seq_err = self.calc_sequence_errors(poses_gt, poses_result)

        # Compute segment errors
        avg_segment_errs = self.compute_segment_error(seq_err)

        # compute overall error
        ave_t_err, ave_r_err = KittiEvalOdom.compute_overall_err(seq_err)

        # Compute ATE
        ate = KittiEvalOdom.compute_ate(poses_gt, poses_result)

        # Compute RPE
        rpe_trans, rpe_rot = KittiEvalOdom.compute_rpe(poses_gt, poses_result)

        metrics = {
            f"{MetricType.TRANS_ERROR.value}/{track_name}": round(ave_t_err * 100, 5),
            f"{MetricType.ROT_ERROR.value}/{track_name}": round(ave_r_err / np.pi * 180 * 100, 5),
            f"{MetricType.ATE.value}/{track_name}": round(ate, 5),
            f"{MetricType.RPE_M.value}/{track_name}": round(rpe_trans, 5),
            f"{MetricType.RPE_DEG.value}/{track_name}": round(rpe_rot * 180 / np.pi, 5),
        }

        # Plotting
        return metrics, self.plot_error(avg_segment_errs, track_name)

    def __call__(self, poses_result, poses_gt, track_id):
        return self.eval(poses_result, poses_gt, track_id)

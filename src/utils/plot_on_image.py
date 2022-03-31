import multiprocessing
from operator import itemgetter

import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure

from src.datamodules.datasets.kitti.kitti_reader import Odometry
from src.datamodules.datasets.kitti.kitti_track_dataset import KittiTrackDataset


class PlotTrackVideoWriter:
    def __init__(self, base_path: str, track_id: str):
        # self._base_path = base_path
        self._track_id = track_id
        self._track_dataset = Odometry(
            base_path=base_path, sequence=self._track_id
        )

        self._poses_coords = np.array([pose[:3, -1] for pose in self._track_dataset.poses])
        self._target_coord_min = np.min(self._poses_coords, axis=0)
        self._target_coord_max = np.max(self._poses_coords, axis=0)

    def _plot_tracks(self, target_coord: np.ndarray, track_id: str):
        fig: figure.Figure = plt.figure(dpi=90, figsize=(4, 4))
        ax: axes.Axes = fig.add_subplot()

        ax.plot(
            target_coord[:, 0], target_coord[:, 2], marker='.',
            markevery=[len(target_coord) - 1], label="target", c='r', mec='b', mfc='b', linewidth=0.5
        )
        ax.set_xlabel("x (m)", fontsize='x-small')
        ax.set_ylabel("x (m)", fontsize='x-small')

        diff_x = (self._target_coord_max[0] - self._target_coord_min[0]) / 10.
        diff_z = (self._target_coord_max[2] - self._target_coord_min[2]) / 10.
        ax.set_xlim(self._target_coord_min[0] - diff_x, self._target_coord_max[0] + diff_x)
        ax.set_ylim(self._target_coord_min[2] - diff_z, self._target_coord_max[2] + diff_z)
        ax.set_title(f"KITTI TRACK: {track_id}", fontsize='x-small')
        ax.tick_params(axis='both', which='major', labelsize='x-small')
        ax.margins(x=0, y=0)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1, wspace=0, hspace=0)
        fig.tight_layout(pad=0.3)

        fig_canvas: figure.FigureCanvasBase = fig.canvas
        fig_canvas.draw()

        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8, )
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        plt.close(fig)

        return img

    def draw_on_image(self, index):
        print(f"{self._track_id}: {index}")
        plot_img = self._plot_tracks(self._poses_coords[:index], f"{self._track_id}. Frame id: {index}")
        image = KittiTrackDataset._load_images([self._track_dataset.cam2_files[index - 1]])[0]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return index, np.hstack([plot_img, image])

    def start(self):
        pool = multiprocessing.Pool()
        values = pool.map(self.draw_on_image, range(1, len(self._poses_coords) + 1))
        sorted(values, key=itemgetter(0))

        out = cv2.VideoWriter(f'{self._track_id}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1560, 360))
        for (index, img) in values:
            print(f"Video write {self._track_id}: {index}")
            out.write(img)
        out.release()


def main(track_id: str):
    PlotTrackVideoWriter("/mnt/sda1/datasets/kitti/dataset", track_id).start()


if __name__ == '__main__':
    tracks = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    for track_id in tracks:
        main(track_id)

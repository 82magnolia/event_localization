import sys
import numpy as np

class VoxelGrid:
    def __init__(self, num_bins: int=5, width: int=640, height: int=480, upsample_rate: int=1):
        assert num_bins > 1
        assert height > 0
        assert width > 0
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.upsample_rate = upsample_rate

    def events_to_voxel_grid(self, events):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """
        event_array = events
        assert(event_array.shape[1] == 4)

        voxel_grid = np.zeros((self.num_bins, self.height, self.width), np.float32).ravel()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = event_array[-1, 0]
        first_stamp = event_array[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        event_array[:, 0] = (self.num_bins - 1) * (event_array[:, 0] - first_stamp) / deltaT
        ts = event_array[:, 0]
        xs = event_array[:, 1].astype(np.int)
        ys = event_array[:, 2].astype(np.int)
        pols = event_array[:, 3]
        pols[pols == 0] = -1  # polarity should be +1 / -1
        

        tis = ts.astype(np.int)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < self.num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * self.width +
                tis[valid_indices] * self.width * self.height, vals_left[valid_indices])

        valid_indices = (tis + 1) < self.num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * self.width +
                (tis[valid_indices] + 1) * self.width * self.height, vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (self.num_bins, self.height, self.width))
        return voxel_grid, last_stamp

    def normalize_voxel(self, voxel_grid, normalize=True):
        if normalize:
            mask = np.nonzero(voxel_grid)
            if mask[0].size > 0:
                mean, stddev = voxel_grid[mask].mean(), voxel_grid[mask].std()
                if stddev > 0:
                    voxel_grid[mask] = (voxel_grid[mask] - mean) / stddev
        return voxel_grid

# Code excerpted from https://github.com/uzh-rpg/rpg_e2vid
from . import base
from . import model
from . import options
from . import utils
from . import image_reconstructor

import torch
import urllib
import os


def download_checkpoint(path_to_model):
    print('Downloading E2VID checkpoint to {} ...'.format(path_to_model))
    e2vid_model = urllib.request.urlopen('http://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar')
    with open(path_to_model, 'w+b') as f:
        f.write(e2vid_model.read())
    print('Done with downloading!')


class E2VID:
    def __init__(self, args, force_no_recurrent=False, force_path_to_model=None, force_no_load=False):
            # Load model to device
        if force_path_to_model is not None:
            pth = force_path_to_model
        else:
            pth = args.path_to_model

        if not os.path.isfile(pth):
            download_checkpoint(pth)
        assert os.path.isfile(pth)
        self.model = utils.loading_utils.load_model(pth, force_no_load=force_no_load)
        use_gpu = torch.cuda.is_available()
        self.device = utils.loading_utils.get_device(use_gpu, 0)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_reconstructor = image_reconstructor.ImageReconstructor(
            self.model, args.height, args.width, self.model.num_bins, args, force_no_recurrent
        )

    def __call__(self, voxel_grid):
        assert len(voxel_grid.shape) == 3
        assert voxel_grid.shape[0] == 5

        if not isinstance(voxel_grid, torch.Tensor):
            event_tensor = torch.from_numpy(voxel_grid)
        else:
            event_tensor = voxel_grid
        image = self.image_reconstructor.update_reconstruction(event_tensor)

        return image


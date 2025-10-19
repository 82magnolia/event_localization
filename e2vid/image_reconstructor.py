import torch
from e2vid.model.model import *
from e2vid.utils.inference_utils import CropParameters, IntensityRescaler, ImageFilter, ImageWriter, UnsharpMaskFilter
from e2vid.utils.event_tensor_utils import EventPreprocessor
from e2vid.utils.image_display_utils import ImageDisplay
from e2vid.utils.inference_utils import upsample_color_image, merge_channels_into_color_image  # for color reconstruction


class ImageReconstructor:
    def __init__(self, model, height, width, num_bins, options, force_no_recurrent=False):

        self.model = model
        self.use_gpu = torch.cuda.is_available()
        self.gpu_id = 0
        self.device = torch.device(self.gpu_id) if self.use_gpu else torch.device('cpu')
        self.height = height
        self.width = width
        self.num_bins = num_bins
        self.options = options
        self.events = None
        self.reconstructed_image = None
        self.perform_color_reconstruction = False
        self.force_no_recurrent = force_no_recurrent

        self.initialize(self.height, self.width, self.options)

    def initialize(self, height, width, options):
        self.last_stamp = None

        if self.force_no_recurrent:
            self.no_recurrent = True
        else:
            self.no_recurrent = options.no_recurrent

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)

        self.last_states_for_each_channel = {'grayscale': None}

        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.image_writer = ImageWriter(options)
        self.image_display = ImageDisplay(options)

    def update_reconstruction(self, event_tensor, event_tensor_id=None, stamp=None):

        # max duration without events before we reinitialize
        self.max_duration_before_reinit_s = 5.0

        # we reinitialize if stamp < last_stamp, or if stamp > last_stamp + max_duration_before_reinit_s
        if stamp is not None and self.last_stamp is not None:
            if stamp < self.last_stamp or stamp > self.last_stamp + self.max_duration_before_reinit_s:
                self.initialize(self.height, self.width, self.options)

        self.last_stamp = stamp

        with torch.no_grad():
            events = event_tensor.unsqueeze(dim=0)
            events = events.to(self.device)

            if self.options.use_fp16:
                events = events.half()

            self.events = self.event_preprocessor(events)

            # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
            events_for_each_channel = {'grayscale': self.crop.pad(events)}
            reconstructions_for_each_channel = {}
            if self.perform_color_reconstruction:
                events_for_each_channel['R'] = self.crop_halfres.pad(self.events[:, :, 0::2, 0::2])
                events_for_each_channel['G'] = self.crop_halfres.pad(self.events[:, :, 0::2, 1::2])
                events_for_each_channel['W'] = self.crop_halfres.pad(self.events[:, :, 1::2, 0::2])
                events_for_each_channel['B'] = self.crop_halfres.pad(self.events[:, :, 1::2, 1::2])

            # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
            for channel in events_for_each_channel.keys():
                new_predicted_frame, states = self.model(events_for_each_channel[channel],
                                                            self.last_states_for_each_channel[channel])

                if self.no_recurrent:
                    self.last_states_for_each_channel[channel] = None
                else:
                    self.last_states_for_each_channel[channel] = states

                # Output reconstructed image
                crop = self.crop if channel == 'grayscale' else self.crop_halfres

                # Unsharp mask (on GPU)
                new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                # Intensity rescaler (on GPU)
                new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                reconstructions_for_each_channel[channel] = new_predicted_frame[0, 0, crop.iy0:crop.iy1,
                                                                                crop.ix0:crop.ix1].cpu().numpy()

            if self.perform_color_reconstruction:
                out = merge_channels_into_color_image(reconstructions_for_each_channel)
            else:
                out = reconstructions_for_each_channel['grayscale']

        # Post-processing, e.g bilateral filter (on CPU)
        self.reconstructed_image = self.image_filter(out)
        self.image_display(self.reconstructed_image, self.events)
    
        return self.reconstructed_image

    def save_reconstruction(self, event_tensor_id):
        self.image_writer(self.reconstructed_image, event_tensor_id, events=self.events)

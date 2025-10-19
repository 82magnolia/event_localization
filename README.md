# Privacy-Preserving Visual Localization with Event Cameras
Official repository of **Privacy-Preserving Visual Localization with Event Cameras (IEEE TIP 2025)** [[Paper]](https://ieeexplore.ieee.org/document/11175560) [[Video]](https://www.youtube.com/watch?v=kk0q690NMXc) [[Project Page]](https://82magnolia.github.io/event_localization/).

## Installation
Run the following commands.
```
conda create --name evloc python=3.7
conda activate evloc
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html 
conda install cudatoolkit=10.2
```

In addition, install pytorch scatter from the following repo for torch 1.10.1 and cuda 10.2: [link](https://github.com/rusty1s/pytorch_scatter).

## Downloading EvRooms and EvHumans
EvRooms and EvHumans can be downloaded from Huggingface Datasets.
- Link to EvRooms: [Click](https://huggingface.co/datasets/82magnolia/ev_rooms)
- Link to EvHumans: [Click](https://huggingface.co/datasets/82magnolia/ev_humans)

After downloading, unzip each folder and organize the folders in the following structure.

    event_localization/data
    └── ev_rooms (EvRooms Dataset)
    │   ├── boxes (Name of dataset in EvRooms)
    │   │   ├── events.dat (Raw events)
    │   │   └── images (Image reconstruction from events - provided for illustration)
    │   │       └── *.png
    │   │   ├── images.txt (Timestamp of each image reconstruction)
    │   │   └── raw_images (Raw images from DAVIS camera)
    │   │       └── *.png
    │   │   ├── raw_images.txt (Timestamp of each image)
    │   │   ├── scale.txt (Scale of each 3D reconstruction to match metric scale)
    │   │   └── sparse/ (Folder containing COLMAP reconstructions)
    │   ├── cabinet (Another name of dataset in EvRooms)
    │   │   ⋮
    └── ev_humans (EvHumans Dataset)
        ├── boxes_human (Name of dataset in EvHumans)
        │   ├── events.dat (Raw events)
        │   └── images (Image reconstruction from events - provided for illustration)
        │       └── *.png
        │   ├── images.txt (Timestamp of each image reconstruction)
        │   └── raw_images (Raw images from DAVIS camera)
        │       └── *.png
        │   ├── raw_images.txt (Timestamp of each image)
        ├── cabinet_human (Another name of dataset in EvHumans)
        │   ⋮

## Running Localization on EvRooms
As stated in the [paper](https://arxiv.org/abs/2212.03177), our localization pipelines operates using [NetVLAD](https://github.com/Relja/netvlad) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork).
Please download the pre-trained weights of SuperGlue through the following [link](https://github.com/magicleap/SuperGluePretrainedNetwork/tree/master/models/weights), and place it under `superglue_models/weights/`.

Also, our localization pipeline uses [E2Vid](https://github.com/uzh-rpg/rpg_e2vid) for event-to-image conversion.
Please download the pre-trained weights or E2Vid through the following [link](https://github.com/uzh-rpg/rpg_e2vid/tree/master?tab=readme-ov-file#run) and place in under `e2vid/pretrained/`.

### Licensing
Note that the repositories from which code is excerpted to build our localization pipeline have the following licenses, so please ensure that all the conditions specified from the licenses are met when using this codebase for other purposes.
- NetVLAD: [License Link](https://github.com/Relja/netvlad?tab=MIT-1-ov-file#readme)
- SuperGlue: [License Link](https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/LICENSE)
- E2Vid: [License Link](https://github.com/uzh-rpg/rpg_e2vid/blob/master/LICENSE)

### Evaluate Localization
Run the following command to obtain localization results on EvRooms.
```
./run_ev_rooms_loc.sh
```

## Event-to-Image Conversion on EvHumans
Run the following command to run sensor-level protection on EvHumans.
The command below will run event-to-image conversions on the `cabinet_human` sequence.
Go to `./log/filter_results/` to see the reconstructions. 
```
python process_ev_to_img.py --config configs/ev_humans.ini --event_path ./data/ev_humans/cabinet_human/events.dat --data_root ./log/filter_results/ --joint_filter
```

## Citation
If you find this repository useful, please cite

```bibtex
@ARTICLE{11175560,
  author={Kim, Junho and Kim, Young Min and Zahreddine, Ramzi and Welge, Weston A. and Krishnan, Gurunandan and Ma, Sizhuo and Wang, Jian},
  journal={IEEE Transactions on Image Processing}, 
  title={Privacy-Preserving Visual Localization With Event Cameras}, 
  year={2025},
  volume={34},
  number={},
  pages={6215-6230},
  keywords={Location awareness;Cameras;Privacy;Protection;Visualization;Three-dimensional displays;Neural networks;Streaming media;Event detection;Image reconstruction;Event cameras;visual localization;camera pose estimation;privacy-preserving computer vision},
  doi={10.1109/TIP.2025.3607640}}
```

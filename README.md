# Privacy-Preserving Visual Localization with Event Cameras
Official repository of **[Privacy-Preserving Visual Localization with Event Cameras (arXiv 2022)](https://arxiv.org/abs/2212.03177)**.
We will make the code and dataset available soon!

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

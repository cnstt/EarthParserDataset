# EarthParserDataset

> [!IMPORTANT]  
> This repository is a fork of the original repo [EarthParserDataset](https://github.com/romainloiseau/EarthParserDataset). For more informations, check out the original repository. Special thanks to [@romainloiseau](https://github.com/romainloiseau) for his great work!
>
> This fork mainly adds good support for the 3D Lidar scenes of [genBoxes](https://github.com/cnstt/genBoxes).

## Main changes
- `LidarScenes` module in [earthparserdataset/lidardataset.py](earthparserdataset/lidardataset.py): able to load entire 3D Lidar scenes (specifically tailored towards scenes generated by [genBoxes](https://github.com/cnstt/genBoxes), and aiming autonomous driving scenes).
- New options, including for example the `lidar_center` setting (encodes the position of the lidar sensor) or the ability to process the ground truth (`gt`) positions of the objects in the scenes.
- New config files for different experiment setups.
You can for example have a look at [configs/data/5complex_light_velod_allScene_SINGLE.yaml](configs/data/5complex_light_velod_allScene_SINGLE.yaml).
It uses several of the new options.

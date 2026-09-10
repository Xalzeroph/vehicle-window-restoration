# Vehicle Window Restoration

Research code for reflection removal and low-light enhancement through vehicle windows and windshields.

## Scope

This repository contains the single-frame physics-aware reflection/restoration pipeline and training code recovered from the previous `vehicle_ssd` experiment tree. The project is organized to support two research lines:

- single-frame vehicle-window reflection removal and low-light enhancement;
- multi-frame/video extensions with layer-aware temporal consistency.

## Included

- `data_v4.py`: dual-pane glass/Fresnel-inspired synthetic degradation and camera pipeline;
- `models/ojm_net_v10.py`: OJMNet model implementation;
- `train_v11.py`: DDP training/evaluation entry point;
- `src/`: data loading, augmentation, scene preparation, and benchmark utilities;
- `requirements.txt`: core Python dependencies.

Large datasets, checkpoints, virtual environments, logs, and experiment outputs are intentionally kept out of this code repository. They belong in the companion Hugging Face dataset repository.

## Reproducibility

The recovered source originated from `/home/courseliu/vehicle_ssd`. The original tree was an experiment snapshot without Git metadata, so exact dataset manifests, checkpoint hashes, and launch commands should be recorded before reproducing older results.

## License

To be decided after checking the licenses of the source datasets and reflection assets.
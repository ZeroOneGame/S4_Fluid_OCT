# PyTorch Project for the Semi-supervised Fluid Segmentation

This is the official implementation of our Semi-supervised Fluid Segmentation Model.

More detailed comments are coming soon.

# Timeline

```
├── 2023.7.29 Update segmentation models
├── ...
├── ...
```


# In Details
```
├──  config
│    └── defaults.py  - here's the default config file.
│ 
│
├──  data  
│    └── datasets  - here's the datasets folder that is responsible for all data handling.
|        └── S4RF_dataset.py  
│    └── transforms  - here's the data preprocess folder that is responsible for all data augmentation.
|        └── build.py
|        └── transforms.py  
│    └── data_utils.py   - here's the file that is responsible for dataset utils.
│
│
├──  engine
│   ├── trainer.py     - this file contains the train loops.
│   └── inference.py   - this file contains the inference process.
│
│
├── layers              - this folder contains any customed layers of your project.
│   └── conv_layer.py
│   └── seg_base_models.py
│   └── segmentation_models.py
│   └── anatomy_loss.py
|   └── semantic_losses.py
│ 
│ 
└── utils
│   └── logger.py
│ 
│ 
└── tests					- this foler contains unit test of your project.
     ├── test_data_sampler.py
```


# Future Work


# Acknowledgments

Some implementations are referred to the following repositories:

https://github.com/huggingface/pytorch-image-models

https://github.com/qubvel/segmentation_models.pytorch

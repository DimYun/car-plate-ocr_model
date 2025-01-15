## Car plates project. OCR model (part 2/3)

This is the project for car plate OCR recognition, which include:
1. [Neural network segmentation model for car plate area with number selection (part 1/3)](https://github.com/DimYun/car-plate-segm_model)
2. Neural network OCR model for plate character recognition (part 2/3)
3. [API service for these two models (part 3/3)](https://github.com/DimYun/car-plate_service)
4. [Additional exemple how to use API service in Telegram bot](https://github.com/DimYun/car-plate_tg-bot)


### Dataset

Dataset include car plate crops from 27 countries, about 2 000 000 images (include two line numbers) in `.jpg`.
Some data are bad and invalid. For more information about data and backbone selection for CRNN please see [EDA.ipynb](notebooks/EDA.ipynb).

To download data:

```shell
make download_dataset
```


### Environment setup

1. Create and activate python venv
    ```shell
    python3 -m venv venv
    . venv/bin/activate
    ```

2. Install libraries
   ```shell
    make install
   ```
   
3. Run linters
   ```shell
   make lint
   ``` 

4. Tune [config.yaml](configs/config.yaml) and select input size of image and backbone for CRNN, see [EDA notebook](notebooks/EDA.ipynb)

5. Train
   ```shell
   make train
   ```


### Additional information

* Inference example in [notebook](notebooks/inference_onnx_convert.ipynb)
* [Best experiment in ClearML](https://app.clear.ml/projects/cb019a605a934ca1a4d85897c43bec3b/experiments/3b62d34fc96049ee9f82db6c858be152/output/execution)
* [History of experiments](HISTORY.md)
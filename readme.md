# nnU-Net modified for LAScarQS 2022

# Directory structure
```
nnUNet_for_LAScarQS
+-- nnunet
+-- batchgenerators
+-- data_conversion
|   +-- dataset_conversion.ipynb
|   +-- scarPred_conversion.ipynb
+-- trained_models
|   +-- task1cavity_TopK
|   |   +-- cv_niftis_postprocessed
|   |   +-- cv_niftis_raw
|   |   +-- fold_0
|   |   +-- ...
|   +-- task1scar
|   +-- task2cavity_TopK
+-- pred_results
|   +-- task1cavity_TopK
|   +-- task1scar
|   +-- task2cavity_TopK
+-- week_report
|   +-- 22-05-06
|   +-- 22-05-13
|   +-- 22-05-20
|   +-- 22-06-10
|   +-- 22-06-17
|   +-- 22-06-24
```

# Installation and Settings
Please follow [nnUNet  official guide](https://github.com/MIC-DKFZ/nnUNet) to install nnUNet as integrative framework.
**Remember to set your environment variables correctly (Item 4 in the official installation guide). It is very important.**
Then you may substitute the folders listed below:
1) nnUNet/nnunet, which is located at the nnUNet folder cloned from GitHub.
2) batchgenerators. You can get the directory of the package folder by typing `batchgenerators.__file__` in Python.
   It is probably like `~/miniconda3/lib/python3.8/site-packages/batchgenerators`
## Package dependency

- simpleitk = 2.0.2
- nibabel
- scipy

# Dataset for LAScarQS 2022
You can download LAScarQS 2022 dataset from [LAScarQS 2022](https://zmic.fudan.edu.cn/lascarqs22)
or from our BaiduDisk url with conversion code.

# Cavity Segmentation
## Dataset conversion
Please uncompress the downloaded dataset, and customize the following variables in `dataset_conversion.ipynb` for automatic conversion.
For Task1 cavity, setting the **TASK1 Cavity** part with
```
  train_source_dir="{where you uncompress TASK1 TRAIN dataset}/task1/train_data"
  val_source_dir="{where you uncompress TASK1 VAL dataset}"
  database_dir="{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task501_task1cavity"
```
Then you can run the **TASK1 Cavity** part.

For Task2 caivty, setting the **TASK2 Cavity** part with
```
  train_source_dir="{where you uncompress TASK2 TRAIN dataset}/task2/train_data"
  val_source_dir="{where you uncompress TASK2 VAL dataset}"
  database_dir="{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task502_task2cavity"
```
Then you can run the **TASK2 Cavity** part.

## Train
Our best model for cavity prediction is nnUNet with combined loss of dice and topk (k=10)
To train task1 cavity model, do the following steps
1)  Set `self.max_num_epochs = 500` in `nnUNet/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_Loss_DiceTopK10.py`
Then run the command lines:
2) `nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity` to do preprocessing.
3) `nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_DiceTopK10 Task501_task1cavity FOLD`
    Please replace FOLD with 0, 1, 2, 3, 4 to run all five folds.
    Training one fold takes about 10 hours on 3090.
4) `nnUNet_find_best_configuration -m 3d_fullres -t 501` will do postprepossing
    and print the command line to run prediction. It will be like
5) `nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceTopK10 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task501_task1cavity`
    replace FOLDER_WITH_TEST_CASES with `{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task501_task1cavity/imagesTs`
    replace OUTPUT_FOLDER_MODEL1 with your output folder to run prediction.

To train task2 cavity model, do the following steps
1)  Set `self.max_num_epochs = 1000` in `nnUNet/nnunet/training/network_training/nnUNet_variants/loss_function/nnUNetTrainerV2_Loss_DiceTopK10.py`
Then run the command lines:
2) `nnUNet_plan_and_preprocess -t 502 --verify_dataset_integrity` to do preprocessing.
3) `nnUNet_train 3d_fullres nnUNetTrainerV2_Loss_DiceTopK10 Task502_task2cavity FOLD`
    Please replace FOLD with 0, 1, 2, 3, 4 to run all five folds.
    Training one fold takes about 20 hours on 3090.
4) `nnUNet_find_best_configuration -m 3d_fullres -t 502` will do postprepossing
    and print the command line to run prediction. It will be like
5) `nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_Loss_DiceTopK10 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task502_task2cavity`
    replace FOLDER_WITH_TEST_CASES with `{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task502_task2cavity/imagesTs`
    replace OUTPUT_FOLDER_MODEL1 with your output folder to run prediction.

# Scar Segmentation
## Dataset conversion
For Task1 scar, setting the **TASK1 Scar** part with
```
  train_source_dir="{where you uncompress TASK1 TRAIN dataset}/task1/train_data"
  cavity_labels_dir="{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task501_task1cavity/labelsTr"
  database_dir="{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task503_task1scar"
```
Then you can run the **TASK1 Scar** part.

## Train
Our best model for scar prediction is to combine raw MRI and distance map of cavity prediction as input to original nnUNet.

To train task1 scar model, do the following steps
1) `nnUNet_plan_and_preprocess -t 503 --verify_dataset_integrity` to do preprocessing.
2) `nnUNet_train 3d_fullres nnUNetTrainerV2_130 Task503_task1scar FOLD`
    Please replace FOLD with 0, 1, 2, 3, 4 to run all five folds.
    Training one fold takes about 2.5 hours on 3090.
3) `nnUNet_find_best_configuration -m 3d_fullres -t 503` will do postprepossing
    and print the command line to run prediction. It will be like
4) `nnUNet_predict -i FOLDER_WITH_TEST_CASES -o OUTPUT_FOLDER_MODEL1 -tr nnUNetTrainerV2_130 -ctr nnUNetTrainerV2CascadeFullRes -m 3d_fullres -p nnUNetPlansv2.1 -t Task503_task1scar`
    before you run the above command, modify the following part in `scarPred_conversion.ipynb`
    ```
      val_source_dir="{where you uncompress TASK1 VAL dataset}"
      cavity_pred_dir="{where you store task1 cavity prediction results}"
      test_dir="{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task503_task1scar/imagesTs"
    ```
    After running the notebook, you can
    replace FOLDER_WITH_TEST_CASES with `{environment path of nnUNet_raw_data_base}/nnUNet_raw_data/Task503_task1scar/imagesTs`
    replace OUTPUT_FOLDER_MODEL1 with your output folder to run prediction
    and run the command for prediction

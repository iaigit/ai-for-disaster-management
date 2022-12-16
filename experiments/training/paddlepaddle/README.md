# How to train our model PaddlePaddle

`Dataset Link`: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

1. First of all, you must download the repository or run in the google colab

```bash
git clone https://github.com/iaigit/ai-for-disaster-management.git
```

2. After that, go to the directory `experiments/training/paddlepaddle/`
3. Choose the model you want to train and open the notebook inside the model
4. If you run the notebook on your local computer, delete `Cell 2`. If you run the notebook on google colab, you can skip this
5. If you run the notebook on your local computer, change the `train_dataset.dataset_root`, `train_dataset.train_path`, `val_dataset.dataset_root`, `val_dataset.val_path` inside `ai-for-disaster-management/miscellaneous/paddlepaddle/ with configs-paddlepaddle-forest_fire_segmentation-hardnet.yaml` with relative to your local computer. If you run the notebook on google colab, you can skip this
6. For `Cell 3` `Line 3`, change the path annotations to `ai-for-disaster-management/miscellaneous/annotations/annotations_forest_fire.zip`
7. For `Cell 3` `Line 5`, change the link downloads with a link inside the `Dataset Link` and choose `No 9` in the section `DATASET FILES`
8. For `Cell 3` `Line 8`, change the link downloads with a link inside the `Dataset Link` and choose `No 10` in the section `DATASET FILES`
9. For `Cell 5` `Line 3`, change the path to `ai-for-disaster-management/miscellaneous/paddlepaddle/configs-paddlepaddle-forest_fire_segmentation-hardnet.yaml`
10. For `Cell 5` `Line 4`, change the path with the path you want
11. For `Cell 6` `Line 1`, change the path to `ai-for-disaster-management/miscellaneous/paddlepaddle/configs-paddlepaddle-forest_fire_segmentation-hardnet.yaml`
12. For `Cell 6` `Line 2`, change the path to the same as `Cell 5` `Line 4` but with extended directory `best_model/model.pdparams`
13. For `Cell 6` `Line 3`, change the path with the path you want
14. For `Cell 7` `Line 1`, change the path to the same as `Cell 6` `Line 3`
15. For `Cell 7` `Line 2`, change the path with the path you want
16. Run all cell notebooks

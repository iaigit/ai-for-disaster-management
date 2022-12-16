# How to train our model PyTorch

`Dataset Link`: https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs

1. First of all, you must download the repository or run in the google colab

```bash
git clone https://github.com/iaigit/ai-for-disaster-management.git
```

2. After that, go to the directory `experiments/training/pytorch/`
3. Choose the model you want to train and open the notebook inside the model
4. If you run the notebook in your local computer, delete `Cell 2` and if you run the notebook on google colab, you can skip this
5. For `Cell 3` `Line 3`, change the path annotations to `ai-for-disaster-management/miscellaneous/annotations/annotations_forest_fire.zip`
6. For `Cell 3` `Line 5`, change the link downloads with a link inside the `Dataset Link` and choose `No 9` in the section `DATASET FILES`
7. For `Cell 3` `Line 8`, change the link downloads with a link inside the `Dataset Link` and choose `No 10` in the section `DATASET FILES`
8. For `Cell 6` `Line 1`, change the path with the path you want
9. Run all cell notebooks

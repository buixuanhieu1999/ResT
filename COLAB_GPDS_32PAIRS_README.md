# GPDS 32-pairs 10-epochs Colab bundle

Put these files in Google Drive:

```text
MyDrive/ResT/archive (3).zip
MyDrive/ResT/rest_gpds_32pairs_colab_bundle.zip
```

Then upload/open:

```text
REST_GPDS_32PAIRS_10EPOCHS_Colab.ipynb
```

Default training protocol:

- Dataset: GPDS/GDPS subset, 150 writers.
- Writer split: 120 train writers / 30 test writers.
- Training pairs: 32 balanced sampled pairs per train writer per epoch.
- Evaluation pairs: 64 balanced sampled pairs per test writer per epoch.
- Epochs: 10.
- Total train pairs: `120 * 32 * 10 = 38,400`.
- Total eval pairs: `30 * 64 * 10 = 19,200`.
- Pretraining: official ViT-B/16 ImageNet-21k NPZ.
- Optimizer: Adam, learning rate `5e-5`.
- Batch size: 32.
- Augmentation during training: rotation, horizontal flip, random 224x224 crop from 256x256.
- Validation/test preprocessing: resize to 224x224, no augmentation.

Expected Colab T4 time:

```text
about 2-3.5 hours
```

If Colab runs out of memory, set:

```python
BATCH_SIZE = 16
```

The notebook saves:

```text
MyDrive/ResT/checkpoints/rest_gpds_32pairs_10epochs_best.pt
MyDrive/ResT/checkpoints/rest_gpds_32pairs_10epochs_report.json
MyDrive/ResT/checkpoints/rest_gpds_32pairs_10epochs_console.log
MyDrive/ResT/checkpoints/rest_gpds_32pairs_10epochs_process.jsonl
MyDrive/ResT/checkpoints/rest_gpds_32pairs_10epochs_curves.png
```

At the end, it prints:

- final best accuracy;
- selected/best epoch;
- AUC and EER;
- training loss/metric graph.

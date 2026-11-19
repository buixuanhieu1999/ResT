# ResT phone web app

A Gradio app you can open on a phone, with two tabs:

- **Scan**: take a photo of a signature and get clean ink on white or transparent, auto-cropped. Downloads: a transparent PNG and a white-background PNG. Options: sensitivity, speck removal, border-blob removal, black/blue ink.
- **Verify**: take a reference photo and a photo to check. Both are cleaned, run through the model's own eval preprocessing (`train_cedar.SignatureTransform(training=False)`), and scored by the Siamese ResT model. Distance ≤ threshold means genuine. The default threshold comes from the checkpoint's metrics and you can change it with a slider.

> **Mock tests only.** The checkpoint scores ~75% accuracy, AUC 0.825 and EER 25.4% on 30 GPDS test writers. Phone photos of signatures are outside its training data (GPDS scans), so the verdict is not reliable for any real decision.

| File | Purpose |
|---|---|
| `app.py` | Gradio UI, checkpoint loading and inference |
| `cleanup.py` | Illumination-robust ink extraction (numpy, OpenCV, Pillow) |
| `ResT_webapp_Colab.ipynb` | Self-contained Colab launcher |
| `make_notebook.py` | Regenerates the notebook from `cleanup.py` and `app.py` |

## Run on Colab (free GPU, recommended)

1. Open `webapp/ResT_webapp_Colab.ipynb` in Colab. Upload it, or use *File → Open notebook → GitHub* once it's pushed.
2. *Runtime → Change runtime type → T4 GPU*.
3. Optional but faster: put the checkpoint in Google Drive at `MyDrive/ResT/checkpoints/rest_gpds_32pairs_10epochs_best.pt`. If it isn't there, the notebook pulls only that file through Git LFS (~388 MB, which counts against the repo's LFS bandwidth).
4. *Runtime → Run all*. The last cell prints a `https://xxxx.gradio.live` link. Open it on your phone.

The notebook writes `cleanup.py` and `app.py` with `%%writefile`, so it works even if `webapp/` isn't on GitHub yet. After you edit either file, run `python webapp/make_notebook.py` to keep the notebook in sync.

**Why `--share`?** Mobile browsers only allow camera access on HTTPS pages (or `localhost`). The share link gives you HTTPS. It's public to anyone who has the URL and expires after at most 72 hours, or as soon as the Colab cell stops.

## Run locally

```bash
pip install -r requirements.txt gradio opencv-python-headless
git lfs pull --include="checkpoints/rest_gpds_32pairs_10epochs_best.pt"
python webapp/app.py                       # http://localhost:7860
python webapp/app.py --share               # HTTPS link for a phone
python webapp/app.py --checkpoint /path/to/best.pt --device cpu --port 8000
```

- `--checkpoint` defaults to `$REST_CKPT`, then `checkpoints/rest_gpds_32pairs_10epochs_best.pt`.
- `--device auto` (default) uses CUDA with fp16 autocast if available, otherwise CPU fp32 with all cores. A warm-up pair runs at start-up so the first real request isn't slow.
- On CPU the ~97M-parameter model takes about 1 s per pair (measured on an 8-thread desktop CPU). A T4 GPU is much faster. The start-up log prints the measured warm-up time.
- If the checkpoint is missing or is still a Git LFS pointer (a file of about 130 bytes), the app still starts. Scan works, Verify is disabled, and the tab shows the error.
- On a phone, tapping the **upload** area opens the native camera app, which gives much sharper photos than the in-page webcam view.

## Why the cleanup step exists

The model was trained on GPDS scans. Its preprocessing uses a single global Otsu threshold, then inverts, crops and centers the image. On a flat scan that works. On a phone photo, a shadow or lighting gradient is darker than the paper, so global Otsu marks the shadow as ink. The crop then covers the whole shadow, and the model sees a large bright blob instead of strokes.

`cleanup.py` fixes the lighting before the model's preprocessing runs:

1. Downscale to at most 1600 px on the long side. Convert to gray and apply a 3×3 blur.
2. Estimate the paper/shadow brightness locally: an elliptical dilation (size ≈ long/90) removes thin dark strokes, then a median blur (size ≈ long/25) smooths what's left.
3. `darkness = 1 - gray / background`. A shadow darkens the paper and the background estimate by the same factor, so it cancels out and only ink stays dark.
4. Threshold with Otsu over pixels where darkness > 0.04, with a floor of 0.10, divided by the sensitivity setting.
5. Drop specks (area threshold scaled with resolution) and border-touching blobs that are large (paper edges, desk, cast shadows).
6. Build a soft alpha for anti-aliased edges, limited to the kept strokes, then crop with 4% padding.

The white-background result is what Verify sends to `SignatureTransform`. The model's own Otsu step then sees clean black ink on white, which is close to its training data.

## Limits

- ~75% accuracy on in-domain GPDS data. Expect worse on phone photos: different pens, paper, resolution and stroke widths than GPDS.
- The model compares one reference with one query. It doesn't model a writer from several references.
- The cleanup assumes dark ink on lighter paper. Very faint pencil, coloured or patterned paper, or signatures that touch the photo border (with border-blob removal on) can lose strokes. Use the sensitivity and options to adjust.
- The decision threshold was tuned on GPDS validation pairs, so it likely needs adjusting for phone photos. That's why there is a slider.

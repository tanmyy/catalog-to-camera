# From Catalog to Camera - Streamlit demo

AASD4015 final project: fashion product image classification.
Upload a product photo, get its category and confidence.

## What it does

One main action, on the first screen: upload a photo, see the result.
Below that: try an example, results, why real photos failed, how it
works, and limitations. Nothing else to learn first.

## Results

| Model | Catalog test (1,200) | Real photos (345) |
|---|---|---|
| A - standard | 97.8% | 58.6% |
| B - + background swap | 97.4% | 69.6% |
| C - + 598 real photos | 97.8% | 95.4% |

Catalog accuracy stays at 97.8%. The gain is on real phone photos.

## Problem

Sellers pick a product category by hand for every listing. It is slow,
and a wrong category hides the product from search and filters.
The seller already uploads a main photo, so the model reads the
category from that photo, for example "Shoes, 98% confidence", and the
seller only confirms it.

## Dataset

- Catalog: Fashion Product Images (Small), Kaggle, 44,441 images.
  Used: 10 classes, 800 per class, 8,000 total, split 70/15/15.
  Classes: Topwear, Shoes, Bags, Bottomwear, Watches, Innerwear,
  Jewellery, Eyewear, Fragrance, Belts.
- Real photos: clothing-dataset-small, 640 photos, test only, plus
  598 real photos in training for Model C, at most 200 per class.
- Background textures for Model B: Describable Textures Dataset, 470 images.

## Approach

- Model: EfficientNet-B0, pretrained on ImageNet, 4.02M parameters.
  It beat a small CNN from scratch (50%), MobileNetV2 and ResNet50
  (6x larger, slightly worse) in a 3-epoch comparison.
- Preprocessing: resize 128x128, ImageNet normalisation, augmentation
  only on training images (rotate +/-18, zoom 0.9-1.1, flip).
- Training: two stages. Head only first, lr 0.001, then unfreeze the
  last two blocks, lr 0.0001. BatchNorm kept in eval mode.
  Early stopping, patience 2. Batch 32, cross-entropy, Adam.
- Model B swaps white backgrounds (R,G,B > 235) for random textures.
- Model C adds real photos to training.

## Failure cases and limitations

- Real photos only cover 3 of the 10 classes: Topwear, Bottomwear
  and Shoes. The 95.4% real-photo score is on those 3 only.
- Part of the gain could be a shortcut, like "phone photo means one
  of these 3 classes".
- Scores vary by a few points between training runs.
- 779 duplicate catalog images were kept and reported as a limitation.
- Weakest catalog class is Topwear, sometimes mixed up with Innerwear.
- A flat brown sweater was predicted as a bag with 99% confidence.
  High confidence does not mean correct.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Cloud

1. Push this repo to GitHub (it is already here).
2. Go to share.streamlit.io, sign in with GitHub, New app.
3. Pick this repo, branch `main`, main file `app.py`, Deploy.
4. Share the link it gives you. No other setup is needed.

## How to select a different model

On the Streamlit app, at the top under "Live model loaded", there
is a Model dropdown. Pick a model there, then upload a photo.
Models appear in that dropdown automatically, no code change,
if you put their `.pt` files in `models/` with these names:

- `models/fashion_classifier_torch.pt` - the final model
- `models/model_A.pt` - Model A, standard
- `models/model_B.pt` - Model B, background swap
- `models/model_C.pt` - Model C, + 598 real photos

Only `.pt` (PyTorch) files work in this app. A `.keras` file
(TensorFlow/Keras) will not appear in the dropdown, because it
is a different format and needs TensorFlow to load.

This repo ships in Demo mode: no trained weights are in it, because
weights are not committed by default and the notebook has to be run
to make them.

1. Run `final_project_torch.ipynb`. On a Mac it takes about 15-20
   minutes on MPS. On Windows it uses an NVIDIA GPU if you have
   one, otherwise the CPU, which is much slower. Easiest on
   Windows: use Google Colab with a free GPU instead, see below.
2. Find the saved weights file, `fashion_classifier_torch.pt`,
   about 16 MB, in the project folder next to the notebook.
3. Put the file at `models/fashion_classifier_torch.pt` in this
   repo (or rename it to `models/model_C.pt`). You can upload
   it on github.com (Add file, Upload files), or with git.
4. Redeploy on Streamlit Cloud. The app loads it automatically and
   predicts for real, with no code change.

Model A and B weights go at `models/model_A.pt` and
`models/model_B.pt` if you want the selector to offer them.
In Demo mode the app never invents a prediction for an uploaded
photo. It shows the preprocessing and the deck examples instead.

## Files

- `app.py` - the whole demo, one file, one main action
- `requirements.txt` - Streamlit, PyTorch, Pillow, numpy, pandas
- `models/` - put trained weights here, see above
- `README.md` - this file

## Source

Project notebook: `final_project_torch.ipynb` in the team package
(AASD4015_team_package.zip). Deck and speaker script are in the
course submission. All numbers above are from the deck and the
team README, last full run.

"""From Catalog to Camera - Streamlit demo.

AASD4015 final project: fashion product image classification.
One main action: upload a product photo, get its category and confidence.

How it works:
- If models/model_C.pt is present, the app loads EfficientNet-B0
  (10 classes) and predicts for real. That is the live mode.
- If no weights file is present, the app runs in Demo mode. It shows
  the upload and the preprocessing, and it shows the three example
  predictions from the project deck, with their known results.
  It never invents a prediction for an uploaded photo in Demo mode.

To enable live predictions: run final_project_torch.ipynb (about
15-20 minutes on a Mac with MPS, longer on Windows without a GPU),
take the saved weights file, fashion_classifier_torch.pt, put it at
models/fashion_classifier_torch.pt in this repo (or rename it to
models/model_C.pt), and redeploy on Streamlit Cloud.
The weights file is about 16 MB.
"""
import io
from pathlib import Path

import streamlit as st
from PIL import Image

CLASSES = [
    "Topwear", "Shoes", "Bags", "Bottomwear", "Watches",
    "Innerwear", "Jewellery", "Eyewear", "Fragrance", "Belts",
]

MODEL_PATHS = {
    "Final model (from notebook)": Path("models/fashion_classifier_torch.pt"),
    "Final model (repo root)": Path("fashion_classifier_torch.pt"),
    "C - + 598 real photos (final)": Path("models/model_C.pt"),
    "A - standard": Path("models/model_A.pt"),
    "B - + background replacement": Path("models/model_B.pt"),
}

# Deck examples, section 10-11. Known results, not live predictions.
EXAMPLES = [
    {
        "name": "Pink T-shirt, flat on a bed",
        "true": "Topwear",
        "model_a": ("Bags", 39),
        "model_c": ("Topwear", 96),
        "why": "Catalog tops are worn by a model. Real ones lie flat, so the shape changes.",
    },
    {
        "name": "Sandals, shot from above",
        "true": "Shoes",
        "model_a": ("Jewellery", 55),
        "model_c": ("Shoes", 94),
        "why": "Catalog shoes are shot from the side. Real ones are shot from above.",
    },
    {
        "name": "Brown sweater, flat",
        "true": "Topwear",
        "model_a": ("Bags", 99),
        "model_c": ("Topwear", 95),
        "why": "A flat brown sweater looks like a leather bag. 99% confident does not mean correct.",
    },
]

RESULTS = {
    "Model": ["A - standard", "B - + background swap", "C - + 598 real photos"],
    "Catalog test (1,200)": [97.8, 97.4, 97.8],
    "Real photos (345)": [58.6, 69.6, 95.4],
}


@st.cache_resource
def load_model(path):
    """Load a trained EfficientNet-B0 checkpoint. Import torch only here,
    so Demo mode works even before torch is installed."""
    import torch
    from torchvision import models
    import torch.nn as nn

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    # The notebook saves a raw state_dict, not a dict with a
    # "state_dict" key, so use it as it is unless that key exists.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model = models.efficientnet_b0(weights=None)
    # Rebuild the notebook's head exactly (final_project_torch.ipynb,
    # cells 3-2 / 3-3c): BatchNorm -> Dropout -> Linear. The notebook
    # saves that head's state_dict, whose keys are classifier.0.* and
    # classifier.2.*. A plain Dropout -> Linear head would fail to load
    # with missing/unexpected keys, so do not simplify this.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.3),
        nn.Linear(in_features, len(CLASSES)),
    )
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(image):
    """Same recipe as the notebook: resize 128x128, ImageNet normalisation."""
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def predict(model, image):
    import torch

    with torch.no_grad():
        probs = torch.softmax(model(preprocess(image)), dim=1)[0]
    top = torch.topk(probs, 3)
    return [(CLASSES[i], float(v)) for v, i in zip(top.values, top.indices)]


def show_result(label, confidence, top3):
    st.subheader(f"{label} - {confidence:.0%} confidence")
    st.progress(min(confidence, 1.0))
    if confidence < 0.50:
        st.warning("Low confidence. In the real seller flow this would go to "
                   "'Unknown' for a person to check, not auto-filled.")
    else:
        st.success(f"Seller would see: {label}, {confidence:.0%} - tap Confirm to use it.")
    st.write("Top 3:")
    for name, conf in top3:
        st.write(f"{name}: {conf:.1%}")
        st.progress(min(conf, 1.0))


def main():
    st.set_page_config(page_title="From Catalog to Camera", layout="centered")
    st.title("From Catalog to Camera")
    st.write("Upload a fashion product photo. The model reads its category, "
             "so a seller does not have to pick it by hand.")

    # Which weights are available?
    available = {name: p for name, p in MODEL_PATHS.items() if p.exists()}
    keras_files = sorted(Path("models").glob("*.keras")) if Path("models").exists() else []
    if keras_files:
        st.info("Found a .keras file in models/, but this app uses "
                "PyTorch, so it cannot load .keras files. Only the .pt "
                "models below can be selected. To use a .keras model, "
                "it would need a separate TensorFlow version of the app.")
    if available:
        st.success("Live model loaded. Predictions below are real.")
        model_name = st.selectbox("Model", list(available.keys()), index=0)
        model = load_model(str(available[model_name]))
        live = True
    else:
        st.info("Demo mode: no trained weights in this repo yet. "
                "Uploads below show the preprocessing, and the examples "
                "show the project's known results. See 'How to enable live "
                "predictions' at the bottom for the one file to add.")
        model = None
        live = False

    st.header("Try it")
    uploaded = st.file_uploader("Choose a product photo",
                                type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
        except Exception:
            st.error("That file could not be read as an image. Try a JPG or PNG.")
            image = None
        if image is not None:
            st.image(image, caption="Your photo", use_column_width=True)
            small = image.resize((128, 128))
            st.image(small, caption="After preprocessing: resized to 128 x 128",
                     width=128)
            if live:
                with st.spinner("Predicting..."):
                    top3 = predict(model, image)
                label, conf = top3[0]
                show_result(label, conf, top3)
            else:
                st.write("No live prediction yet. Add "
                         "`models/fashion_classifier_torch.pt` "
                         "(about 16 MB, from running the project notebook) "
                         "and redeploy, and this same upload will get a real "
                         "category and confidence.")

    st.header("Try an example")
    st.write("These three are from the project deck. Model A was trained "
             "only on catalog photos. Model C also saw 598 real photos.")
    choice = st.selectbox("Example", [e["name"] for e in EXAMPLES])
    ex = next(e for e in EXAMPLES if e["name"] == choice)
    st.write(f"True category: **{ex['true']}**")
    st.write(f"Model A: {ex['model_a'][0]}, {ex['model_a'][1]}% confidence - wrong")
    st.write(f"Model C: {ex['model_c'][0]}, {ex['model_c'][1]}% confidence - right")
    st.write(ex["why"])

    st.header("Results")
    st.write("Catalog accuracy stays at 97.8%. The gain is on real phone photos.")
    try:
        import pandas as pd

        st.dataframe(pd.DataFrame(RESULTS), hide_index=True, use_container_width=True)
        st.bar_chart(pd.DataFrame(RESULTS).set_index("Model"))
    except Exception:
        for row in zip(RESULTS["Model"], RESULTS["Catalog test (1,200)"],
                       RESULTS["Real photos (345)"]):
            st.write(f"{row[0]}: catalog {row[1]}%, real {row[2]}%")

    with st.expander("Why real photos failed at first"):
        st.write("- Background: 89% of training images are pure white. "
                 "Real photos have beds, floors and carpets.")
        st.write("- Viewpoint: catalog tops are worn by a model, real ones "
                 "lie flat. Catalog shoes are side-on, real ones from above.")
        st.write("- 99% confident does not mean correct.")
        st.write("- In statistics this is domain shift: the training images "
                 "and the test images come from different distributions.")

    with st.expander("How it works"):
        st.write("- Model: EfficientNet-B0, pretrained on ImageNet, 4.02M parameters.")
        st.write("- Data: 8,000 catalog images, 10 classes, 800 per class, "
                 "split 70/15/15. Plus 640 real phone photos for testing.")
        st.write("- Preprocessing: resize 128x128, ImageNet normalisation, "
                 "augmentation only on training images (rotate, zoom, flip).")
        st.write("- Training: two stages. Head only first (lr 0.001), then "
                 "unfreeze the last two blocks (lr 0.0001). BatchNorm kept "
                 "in eval mode. Early stopping, patience 2.")
        st.write("- Model B swaps white backgrounds for textures. Model C "
                 "adds 598 real photos, at most 200 per class.")

    with st.expander("Limitations - read this before trusting a result"):
        st.write("- Real photos only cover 3 of the 10 classes: Topwear, "
                 "Bottomwear and Shoes. The 95.4% real-photo score is on "
                 "those 3 only.")
        st.write("- Part of the gain could be a shortcut, like 'phone photo "
                 "means one of these 3 classes'.")
        st.write("- Scores vary by a few points between training runs.")
        st.write("- 779 duplicate catalog images were kept and reported as "
                 "a limitation.")
        st.write("- Weakest catalog class is Topwear, sometimes mixed up "
                 "with Innerwear.")

    with st.expander("How to enable live predictions"):
        st.write("1. Run `final_project_torch.ipynb` (about 15-20 "
                 "minutes on a Mac, longer on Windows without a GPU).")
        st.write("2. Find the saved weights file, "
                 "`fashion_classifier_torch.pt`, about 16 MB, in the "
                 "project folder next to the notebook.")
        st.write("3. Put the file at `models/fashion_classifier_torch.pt` "
                 "in this repo (or rename it to `models/model_C.pt`).")
        st.write("4. Redeploy on Streamlit Cloud. The app will then load it "
                 "and predict for real, with no code change.")
        st.write("Model A and B weights go at `models/model_A.pt` and "
                 "`models/model_B.pt` if you want the selector to offer them.")

    st.caption("AASD4015 - Advanced Mathematical Concepts for Deep Learning. "
               "Project: From Catalog to Camera.")


if __name__ == "__main__":
    main()

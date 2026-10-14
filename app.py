"""
SignLingo desktop app.

Camera and upload recognition use hand landmarks. The hold-out gallery
uses the Sign Language MNIST CNN. The autoencoder reconstructs 28×28 crops.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import streamlit as st

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from signlingo.data.hands_runtime import HandTracker, draw_hands
from signlingo.history_store import PredictionHistory
from signlingo.models.alphabet_cnn import ASL_LETTERS, index_to_letter
from signlingo.models.landmark_alphabet import predict_landmarks, top_k
from signlingo.sentence.formation import SentenceFormer, _POS_MAP

METRICS_PATH = os.path.join(ROOT, "results", "alphabet_metrics.json")
AE_METRICS_PATH = os.path.join(ROOT, "results", "autoencoder_metrics.json")
SAMPLES_PATH = os.path.join(ROOT, "results", "alphabet_samples.npz")

_POS_LABELS = {
    "PRON": "Pronouns",
    "VERB": "Verbs",
    "MODAL": "Modals",
    "NOUN": "Nouns",
    "ADJ": "Adjectives",
    "ADV": "Time words",
    "DET": "Determiners",
    "WH": "Question words",
    "INTJ": "Responses",
}


@st.cache_resource
def load_predictor():
    from signlingo.models.alphabet_infer import AlphabetPredictor

    return AlphabetPredictor()


@st.cache_resource
def load_tracker() -> HandTracker:
    return HandTracker(mode="image", max_hands=1, min_detection_confidence=0.5)


def read_metrics() -> dict | None:
    if not os.path.isfile(METRICS_PATH):
        return None
    with open(METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_ae_metrics() -> dict | None:
    if not os.path.isfile(AE_METRICS_PATH):
        return None
    with open(AE_METRICS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource
def load_autoencoder():
    from signlingo.models.autoencoder_infer import SignAutoencoder

    return SignAutoencoder()


def _sign_preview(image: np.ndarray, size: int = 220) -> np.ndarray:
    """Nearest-neighbor enlarge a 28×28 crop so the hand is readable."""
    gray = np.asarray(image, dtype=np.float32).reshape(28, 28)
    if float(np.max(gray)) > 1.0:
        canvas = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        canvas = (np.clip(gray, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)


def predict_bgr(image_bgr: np.ndarray, tracker: HandTracker):
    detection = tracker.detect(image_bgr)
    drawn = draw_hands(image_bgr.copy(), detection)
    if not detection.hands:
        return "—", 0.0, np.zeros(24, dtype=np.float32), drawn, False
    letter, confidence, probs = predict_landmarks(detection.hands[0].landmarks)
    return letter, confidence, probs, drawn, True


def open_page(title: str, icon: str, lead: str):
    st.title(title, icon=icon)
    st.markdown(":green-badge[ASL letters] :blue-badge[A–Y except J] :gray-badge[Static signs]")
    st.write(lead)


def page_recognize():
    open_page(
        "Recognize a letter",
        ":material/front_hand:",
        "Hold one hand up with the palm toward the camera and the fingers pointing up. "
        "A letter is saved only after you click Save prediction.",
    )

    pose, scope, save = st.columns(3)
    with pose:
        with st.container(border=True):
            st.subheader("Pose", icon=":material/pan_tool:")
            st.write("Palm faces the camera. Fingers point up. Keep the whole hand inside the frame on a plain background.")
    with scope:
        with st.container(border=True):
            st.subheader("What is read", icon=":material/gesture:")
            st.write("MediaPipe finds 21 hand points. Finger extension, thumb position, and palm direction decide the letter.")
    with save:
        with st.container(border=True):
            st.subheader("What is saved", icon=":material/bookmark:")
            st.write("Save prediction writes one row to History. It does not change the model weights.")

    st.space("small")
    try:
        tracker = load_tracker()
    except Exception as exc:
        st.error(str(exc), icon=":material/error:")
        return

    with st.container(border=True):
        st.subheader("Capture", icon=":material/photo_camera:")
        source = st.segmented_control(
            "Input",
            ["Camera photo", "Upload"],
            default="Camera photo",
            required=True,
            key="recognize_source",
        )
        image_bgr = None
        source_name = ""
        if source == "Camera photo":
            shot = st.camera_input("Take a photo of one hand sign")
            if shot is not None:
                data = np.frombuffer(shot.getvalue(), dtype=np.uint8)
                image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                source_name = "camera"
        else:
            file = st.file_uploader("PNG or JPG", type=["png", "jpg", "jpeg"])
            if file is not None:
                data = np.frombuffer(file.getvalue(), dtype=np.uint8)
                image_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                source_name = "upload"

    st.space("small")
    with st.container(border=True):
        st.subheader("Reading", icon=":material/abc:")
        if image_bgr is None:
            st.info("Take a photo or upload one to see a prediction.", icon=":material/info:")
        else:
            letter, confidence, probs, drawn, found = predict_bgr(image_bgr, tracker)
            left, right = st.columns([1.15, 0.85], gap="large")
            with left:
                st.image(
                    cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB),
                    caption="Detected hand",
                    alt="Photo with the detected hand outlined",
                    width="stretch",
                )
                if not found:
                    st.warning(
                        "No hand was found. Move closer and keep the whole hand in frame.",
                        icon=":material/warning:",
                    )
            with right:
                st.metric("Letter", letter)
                st.metric("Confidence", f"{confidence:.1%}")
                st.caption("Top 3")
                for name, score in top_k(probs, k=3):
                    st.progress(min(max(score, 0.0), 1.0), text=f"{name}  {score:.1%}")
                if st.button("Save prediction", type="primary", icon=":material/save:", disabled=not found):
                    PredictionHistory().add(letter, confidence, source=source_name)
                    st.success("Saved to History.", icon=":material/check_circle:")

    st.space("small")
    with st.container(border=True):
        st.subheader("Letters this page can return", icon=":material/font_download:")
        st.write(" ".join(ASL_LETTERS))
        st.caption(
            "J and Z need motion, so they are not in this set. Closed-fist letters such as A, S, T, M, and N "
            "are easier to confuse because their landmarks look alike. Open shapes such as B, V, L, Y, W, and I are clearer."
        )


def page_gallery():
    open_page(
        "Hold-out samples",
        ":material/grid_view:",
        "Each card is a dataset image the CNN did not train on. The caption is the prediction and the true letter.",
    )
    metrics = read_metrics()
    if metrics:
        one, two, three, four = st.columns(4)
        one.metric("Hold-out accuracy", f"{metrics['holdout_accuracy']:.1%}")
        two.metric("Top 3", f"{metrics['holdout_top3_accuracy']:.1%}")
        three.metric("Train images", f"{metrics['train_images']:,}")
        four.metric("Hold-out images", f"{metrics['holdout_images']:,}")
        st.caption("These scores are for 28×28 Sign Language MNIST crops. They do not describe a camera photo.")
    else:
        st.warning("No measured metrics yet. Run `python scripts/train_alphabet.py`.", icon=":material/warning:")

    if not os.path.isfile(SAMPLES_PATH):
        st.warning("Train the model first so the sample file can be created.", icon=":material/warning:")
        return

    st.space("small")
    st.subheader("One hold-out crop per letter", icon=":material/image:")
    predictor = load_predictor()
    from signlingo.models.alphabet_infer import pixels_to_input

    pack = np.load(SAMPLES_PATH)
    images = pack["images"]
    labels = pack["labels"]
    cols = st.columns(4, gap="medium")
    for i, (img, lab) in enumerate(zip(images, labels)):
        pred, conf, _ = predictor.predict_input(pixels_to_input(img))
        true = index_to_letter(int(lab))
        match = pred == true
        with cols[i % 4]:
            with st.container(border=True):
                st.image(
                    _sign_preview(img, 200),
                    caption=f"Predicted {pred}  ·  {conf:.0%}",
                    alt=f"Hold-out sign for letter {true}, predicted {pred}",
                    width="stretch",
                )
                if match:
                    st.markdown(f":green-badge[True {true}]")
                else:
                    st.markdown(f":red-badge[True {true}]")

    st.space("small")
    with st.container(border=True):
        st.subheader("How this gallery was built", icon=":material/dataset:")
        st.write(
            "Sign Language MNIST rows are a label plus 784 grayscale pixels. "
            "The public file used here is the training CSV only, so the score is a stratified 80/20 hold-out "
            "with random seed 42, not the official Kaggle test split."
        )
        st.write(
            "The network is a small CNN: three convolution blocks, then a dense layer and a 24-way softmax. "
            "Pixel values are divided by 255 inside the network. J is missing from the label set, and Z is absent, "
            "because both letters are drawn in motion."
        )


def page_autoencoder():
    open_page(
        "Sign autoencoder",
        ":material/blur_on:",
        "The encoder compresses a 28×28 letter crop into a 32-number code. "
        "The decoder rebuilds the hand from that code. Noise added to the crop "
        "is dropped on the way back, so the reconstruction is the clean sign.",
    )

    encode, bottleneck, decode = st.columns(3)
    with encode:
        with st.container(border=True):
            st.subheader("1  Encode", icon=":material/compress:")
            st.write("Convolution layers shrink the 28×28 hand until only a short latent code remains.")
    with bottleneck:
        with st.container(border=True):
            st.subheader("2  Latent code", icon=":material/data_array:")
            st.write("Thirty-two numbers stand in for the letter shape. The chart below is that code for the image you pick.")
    with decode:
        with st.container(border=True):
            st.subheader("3  Decode", icon=":material/open_in_full:")
            st.write("The decoder paints the hand back. A lower error than the noisy input means noise was removed.")

    st.space("small")
    try:
        model = load_autoencoder()
    except FileNotFoundError as exc:
        st.warning(str(exc), icon=":material/warning:")
        return

    ae_metrics = read_ae_metrics()
    if ae_metrics:
        one, two, three, four = st.columns(4)
        one.metric("Noisy error", f"{ae_metrics['holdout_noisy_mse']:.4f}")
        two.metric("Reconstructed error", f"{ae_metrics['holdout_denoised_mse']:.4f}")
        three.metric("Clean reconstruction", f"{ae_metrics['holdout_clean_reconstruction_mse']:.4f}")
        four.metric("Latent size", str(ae_metrics["latent_dim"]))
        st.caption("Errors are mean squared error on the same hold-out split as the letter CNN.")

    st.space("small")
    with st.container(border=True):
        st.subheader("Choose a crop", icon=":material/tune:")
        source = st.segmented_control(
            "Image",
            ["Hold-out letter", "Upload"],
            default="Hold-out letter",
            required=True,
            key="ae_source",
        )
        image = None
        if source == "Upload":
            file = st.file_uploader("PNG or JPG of a hand sign", type=["png", "jpg", "jpeg"], key="ae_upload")
            if file is not None:
                data = np.frombuffer(file.getvalue(), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is not None:
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        elif os.path.isfile(SAMPLES_PATH):
            pack = np.load(SAMPLES_PATH)
            images = pack["images"]
            labels = pack["labels"]
            letters = [index_to_letter(int(lab)) for lab in labels]
            choice = st.selectbox("Letter", letters, key="ae_letter")
            image = images[letters.index(choice)]
        else:
            st.warning(
                "Train the alphabet CNN first so a hold-out image of each letter is available.",
                icon=":material/warning:",
            )
            return

        if image is None:
            st.info("Choose a letter or upload an image.", icon=":material/info:")
            return

        noise = st.slider("Noise", min_value=0.0, max_value=0.6, value=0.25, step=0.05)

    clean, noisy, rebuilt, noisy_mse, rebuilt_mse = model.denoise(
        image, noise_std=float(noise), rng=np.random.default_rng(0)
    )
    code = model.encode(clean)[0]

    st.space("small")
    st.subheader("Original, noisy input, reconstruction", icon=":material/compare:")
    original, corrupted, restored = st.columns(3)
    with original:
        with st.container(border=True):
            st.image(
                _sign_preview(clean[0]),
                caption="Original",
                alt="Original 28 by 28 sign image enlarged",
                width="stretch",
            )
    with corrupted:
        with st.container(border=True):
            st.image(
                _sign_preview(noisy[0]),
                caption=f"Noisy input  ·  error {noisy_mse:.4f}",
                alt="Sign image with Gaussian noise added",
                width="stretch",
            )
    with restored:
        with st.container(border=True):
            st.image(
                _sign_preview(rebuilt[0]),
                caption=f"Reconstructed  ·  error {rebuilt_mse:.4f}",
                alt="Hand sign reconstructed by the autoencoder",
                width="stretch",
            )
            if rebuilt_mse < noisy_mse:
                st.markdown(":green-badge[Noise reduced]")
            else:
                st.markdown(":orange-badge[Error did not fall]")

    st.space("small")
    with st.container(border=True):
        st.subheader("Latent code", icon=":material/bar_chart:")
        st.caption("Each bar is one of the 32 numbers the encoder kept for this sign.")
        st.bar_chart(
            {"value": code.astype(np.float64)},
            x_label="Latent unit",
            y_label="Activation",
            sort=False,
            height=280,
            alt="Bar chart of the 32-number latent code for this sign",
        )
        st.caption(
            "A lower reconstructed error than the noisy-input error means the decoder "
            "removed noise instead of copying it. This page does not classify the letter."
        )


def page_spell():
    open_page(
        "Build a sentence",
        ":material/translate:",
        "Type fingerspelled words. If every word is a known gloss, such as I GO SCHOOL, "
        "SignLingo rewrites it as an English sentence.",
    )

    with st.container(border=True):
        st.subheader("Gloss line", icon=":material/edit:")
        text = st.text_input("Spelled text", value=st.session_state.get("spell", "I GO"), key="spell_input")
        st.session_state["spell"] = text
        tokens = [t for t in text.strip().upper().split() if t]
        if tokens:
            st.caption("Tokens  ·  " + "   ".join(tokens))

    st.space("small")
    with st.container(border=True):
        st.subheader("English line", icon=":material/subject:")
        if not tokens:
            st.caption("Type at least one word.")
        elif all(tok in _POS_MAP for tok in tokens):
            st.success(SentenceFormer().form_sentence(tokens), icon=":material/check_circle:")
        else:
            unknown = [t for t in tokens if t not in _POS_MAP]
            st.write(" ".join(tokens))
            st.warning(
                "These words are not in the gloss list: " + ", ".join(unknown),
                icon=":material/warning:",
            )

    st.space("small")
    st.subheader("Words the rewriter knows", icon=":material/menu_book:")
    st.caption("The English line runs only when every token is in this list. Unknown words stay as typed text.")
    grouped: dict[str, list[str]] = defaultdict(list)
    for word, pos in _POS_MAP.items():
        grouped[pos].append(word)
    order = ["PRON", "VERB", "MODAL", "NOUN", "ADJ", "ADV", "DET", "WH", "INTJ"]
    for pos in order:
        words = sorted(grouped.get(pos, []))
        if not words:
            continue
        with st.container(border=True):
            st.markdown(f"**{_POS_LABELS.get(pos, pos)}**")
            st.write(", ".join(words))


def page_history():
    open_page(
        "Prediction history",
        ":material/history:",
        "Saved camera and upload readings stay in this list. Clearing it removes the file contents. "
        "Saving a row does not train the model.",
    )
    store = PredictionHistory()
    rows = list(reversed(store.records))
    one, two, three = st.columns(3)
    one.metric("Saved rows", f"{len(rows):,}")
    two.metric("Kept at most", "500")
    latest = rows[0]["label"] if rows else "—"
    three.metric("Latest letter", latest)

    st.space("small")
    with st.container(border=True):
        st.subheader("Records", icon=":material/table:")
        if st.button("Clear history", icon=":material/delete:"):
            store.clear()
            st.rerun()
        if not rows:
            st.info("No predictions saved yet.", icon=":material/info:")
        else:
            st.dataframe(rows, width="stretch", hide_index=True)

    st.space("small")
    with st.container(border=True):
        st.subheader("Where the rows live", icon=":material/folder:")
        st.write(
            "Each save stores the time, the letter, the confidence, and whether the photo came from the camera or an upload. "
            "The file is data/prediction_history.json, and only the latest 500 rows are kept."
        )


def page_about():
    open_page(
        "About this demo",
        ":material/info:",
        "SignLingo shows three separate readings of a static ASL letter: hand landmarks from a photo, "
        "a CNN on dataset crops, and an autoencoder that rebuilds those crops.",
    )

    camera, gallery, auto = st.columns(3)
    with camera:
        with st.container(border=True):
            st.subheader("Camera", icon=":material/photo_camera:")
            st.write("A still photo or upload is read from 21 hand landmarks. This is the Recognize page.")
    with gallery:
        with st.container(border=True):
            st.subheader("Gallery", icon=":material/grid_view:")
            st.write("Hold-out 28×28 images go through the trained CNN. That accuracy is not a webcam score.")
    with auto:
        with st.container(border=True):
            st.subheader("Autoencoder", icon=":material/blur_on:")
            st.write("The same crops are compressed to 32 numbers and reconstructed, including from a noisy copy.")

    st.space("small")
    with st.container(border=True):
        st.subheader("What this folder does not do", icon=":material/block:")
        st.write("J and Z need motion, so they are outside the letter set.")
        st.write("The English sentence line is a fixed gloss list, not a trained translator.")
        st.write("This folder does not include the WLASL videos or models/sign_language_model.h5, so the original word-level accuracy table was not measured here.")
        st.write("Saving a prediction does not update any weights.")

    metrics = read_metrics()
    ae = read_ae_metrics()
    st.space("small")
    with st.expander("Measured CNN record", icon=":material/analytics:"):
        if metrics:
            st.json(metrics)
        else:
            st.caption("Run python scripts/train_alphabet.py to write results/alphabet_metrics.json.")
    with st.expander("Measured autoencoder record", icon=":material/blur_on:"):
        if ae:
            st.json(ae)
        else:
            st.caption("Run python scripts/train_autoencoder.py to write results/autoencoder_metrics.json.")


def main():
    st.set_page_config(
        page_title="SignLingo",
        page_icon=":material/front_hand:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.logo(".streamlit/logo.png", size="large")

    page = st.navigation(
        {
            "Camera": [
                st.Page(
                    page_recognize,
                    title="Recognize",
                    icon=":material/front_hand:",
                    url_path="recognize",
                    default=True,
                ),
            ],
            "Dataset": [
                st.Page(
                    page_gallery,
                    title="Hold-out samples",
                    icon=":material/grid_view:",
                    url_path="samples",
                ),
                st.Page(
                    page_autoencoder,
                    title="Autoencoder",
                    icon=":material/blur_on:",
                    url_path="autoencoder",
                ),
            ],
            "Language": [
                st.Page(
                    page_spell,
                    title="Sentence",
                    icon=":material/translate:",
                    url_path="sentence",
                ),
                st.Page(
                    page_history,
                    title="History",
                    icon=":material/history:",
                    url_path="history",
                ),
                st.Page(
                    page_about,
                    title="About",
                    icon=":material/info:",
                    url_path="about",
                ),
            ],
        },
        position="sidebar",
        expanded=True,
    )
    page.run()


if __name__ == "__main__":
    main()

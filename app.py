
import streamlit as st
import tensorflow as tf
import numpy as np
import plotly.express as px
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="AI Handwritten Digit Recognition",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "models/emnist_model.h5"
    )

model = load_model()

# ----------------------------------
# CLASSES
# ----------------------------------

classes = [str(i) for i in range(10)]

# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.title("🧠 Project Info")

    st.metric(
        "Model Accuracy",
        "99%+"
    )

    st.markdown("---")

    st.write("Dataset")
    st.info("MNIST Digits")

    st.write("Classes")
    st.info("Digits 0-9")

    st.markdown("---")

    st.write(
        "Built with TensorFlow, CNN and Streamlit"
    )

# ----------------------------------
# HEADER
# ----------------------------------

st.title("🧠 Handwritten Digit Recognition")

st.caption(
    "Draw a digit (0-9) and let the CNN predict it."
)

# ----------------------------------
# SESSION STATE
# ----------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# ----------------------------------
# LAYOUT
# ----------------------------------

col1, col2 = st.columns([2, 1])

# ----------------------------------
# CANVAS
# ----------------------------------

with col1:

    st.subheader("✏️ Draw Here")

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas"
    )

# ----------------------------------
# PREDICTION
# ----------------------------------

with col2:

    st.subheader("Prediction")

    if st.button("Predict"):

        if canvas_result.image_data is not None:

            image = Image.fromarray(
                canvas_result.image_data.astype("uint8")
            )

            image = image.convert("L")

            img = np.array(image)

            coords = np.argwhere(img > 20)

            if len(coords) > 0:

                y0, x0 = coords.min(axis=0)
                y1, x1 = coords.max(axis=0)

                img = img[y0:y1+1, x0:x1+1]

                image = Image.fromarray(img)
                image = image.resize((20, 20))

                canvas = Image.new(
                    "L",
                    (28, 28),
                    0
                )

                canvas.paste(
                    image,
                    (4, 4)
                )

                img = np.array(canvas)

                st.image(
                    img,
                    caption="Processed Digit",
                    width=120
                )

                img = img.astype("float32") / 255.0

                img = img.reshape(
                    1,
                    28,
                    28,
                    1
                )

                prediction = model.predict(
                    img,
                    verbose=0
                )

                pred_idx = np.argmax(
                    prediction
                )

                confidence = (
                    np.max(prediction) * 100
                )

                predicted_digit = classes[
                    pred_idx
                ]

                st.success(
                    f"Prediction: {predicted_digit}"
                )

                st.progress(
                    int(confidence)
                )

                st.write(
                    f"Confidence: {confidence:.2f}%"
                )

                st.session_state.history.append(
                    predicted_digit
                )

                st.markdown(
                    "### Top 3 Predictions"
                )

                top3 = np.argsort(
                    prediction[0]
                )[-3:]

                for idx in reversed(top3):

                    st.write(
                        f"{classes[idx]} → {prediction[0][idx]*100:.2f}%"
                    )

                st.markdown(
                    "### Probability Distribution"
                )

                fig = px.bar(
                    x=classes,
                    y=prediction[0],
                    labels={
                        "x": "Digit",
                        "y": "Probability"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:
                st.warning(
                    "Please draw a digit first."
                )

# ----------------------------------
# HISTORY
# ----------------------------------

st.markdown("---")

st.subheader(
    "Prediction History"
)

if len(st.session_state.history) > 0:

    st.write(
        " → ".join(
            st.session_state.history
        )
    )

else:

    st.info(
        "No predictions yet."
    )

# ----------------------------------
# DATASET INFO
# ----------------------------------

with st.expander(
    "Dataset Information"
):

    st.write("""
    MNIST Dataset

    • 70,000 handwritten digits
    • 10 classes (0-9)
    • 28×28 grayscale images
    • Standard benchmark dataset
    """)


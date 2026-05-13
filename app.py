import streamlit as st

from PIL import Image

from utils import (
    load_efficientnet,
    load_resnet,
    predict_image,
    generate_gradcam
)

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Brain Tumor AI Platform",
    page_icon="🧠",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown(
    """
    <style>

    .main {
        background-color: #F5F7FA;
    }

    .hero-title {
        font-size: 52px;
        font-weight: 800;
        color: #111827;
        text-align: center;
    }

    .hero-subtitle {
        font-size: 22px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ======================================
# LOAD MODELS
# ======================================

efficientnet_model = load_efficientnet()

resnet_model = load_resnet()

# ======================================
# SIDEBAR
# ======================================

st.sidebar.title("🧠 Brain Tumor AI")

st.sidebar.success(
    "Reliability-Aware Explainable AI"
)

st.sidebar.write("---")

st.sidebar.markdown(
    """
    ## 🚀 Features

    ✅ EfficientNet
    
    ✅ ResNet18
    
    ✅ Ensemble Learning
    
    ✅ Monte Carlo Dropout
    
    ✅ Uncertainty Estimation
    
    ✅ Reliability Framework
    
    ✅ GradCAM Explainability
    
    ✅ PDF Report Generation
    
    ✅ Multi-Page Dashboard
    """
)

st.sidebar.write("---")

st.sidebar.info(
    "Medical AI Research Prototype"
)

# ======================================
# HERO SECTION
# ======================================

st.markdown(
    '<div class="hero-title">🧠 Brain Tumor AI Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-subtitle">Reliability-Aware Explainable Deep Learning Framework</div>',
    unsafe_allow_html=True
)

st.write("---")

# ======================================
# TOP METRICS
# ======================================

metric1, metric2, metric3, metric4 = st.columns(4)

with metric1:
    st.metric(
        "Models",
        "2"
    )

with metric2:
    st.metric(
        "Framework",
        "Ensemble"
    )

with metric3:
    st.metric(
        "Explainability",
        "GradCAM"
    )

with metric4:
    st.metric(
        "Reliability",
        "High"
    )

st.write("---")

# ======================================
# FILE UPLOAD
# ======================================

st.markdown(
    '<div class="section-title">📌 Upload MRI Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ======================================
# MAIN PREDICTION
# ======================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    (
        prediction,
        confidence,
        uncertainty
    ) = predict_image(
        efficientnet_model,
        resnet_model,
        image
    )

    # ==================================
    # RELIABILITY
    # ==================================

    if confidence >= 90:

        reliability = "Highly Reliable ✅"

    elif confidence >= 70:

        reliability = "Moderately Reliable ⚠️"

    else:

        reliability = "Low Reliability ❌"

    # ==================================
    # TABS
    # ==================================

    tab1, tab2, tab3 = st.tabs([
        "🧠 Prediction",
        "🔥 GradCAM",
        "📊 AI Insights"
    ])

    # ==================================
    # PREDICTION TAB
    # ==================================

    with tab1:

        col1, col2 = st.columns([1,1])

        with col1:

            st.image(
                image,
                caption="Uploaded MRI Image",
                use_container_width=True
            )

        with col2:

            st.subheader(
                "📊 Ensemble Prediction Results"
            )

            st.metric(
                "Predicted Tumor",
                prediction
            )

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.metric(
                "Uncertainty Score",
                f"{uncertainty:.4f}"
            )

            st.success(
                f"Reliability Status: {reliability}"
            )

            st.info(
                "Low uncertainty indicates stable ensemble prediction."
            )

    # ==================================
    # GRADCAM TAB
    # ==================================

    with tab2:

        st.subheader(
            "🔥 GradCAM Explainability"
        )

        gradcam_image = generate_gradcam(
            efficientnet_model,
            image
        )

        st.image(
            gradcam_image,
            caption="Model Attention Heatmap",
            use_container_width=True
        )

        st.warning(
            """
            🔴 Red and yellow regions indicate strong model attention.

            🔵 Blue regions indicate lower attention regions.

            GradCAM improves explainability and trustworthiness.
            """
        )

    # ==================================
    # AI INSIGHTS TAB
    # ==================================

    with tab3:

        insight1, insight2, insight3 = st.columns(3)

        with insight1:

            st.metric(
                "Framework",
                "Ensemble AI"
            )

            st.caption(
                "EfficientNet + ResNet18"
            )

        with insight2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Prediction Stability"
            )

        with insight3:

            st.metric(
                "Uncertainty",
                f"{uncertainty:.4f}"
            )

            st.caption(
                "Monte Carlo Dropout"
            )

        st.write("---")

        st.subheader(
            "🧪 Reliability Analysis"
        )

        st.success(
            """
            The ensemble deep learning framework combines:

            • EfficientNet

            • ResNet18

            • Monte Carlo Dropout

            • GradCAM Explainability

            to improve robustness, reliability,
            and interpretability in brain tumor classification.
            """
        )

# ======================================
# FOOTER
# ======================================

st.write("---")

st.caption(
    "Developed using Ensemble Learning, Explainable AI, and Reliability-Aware Deep Learning"
)
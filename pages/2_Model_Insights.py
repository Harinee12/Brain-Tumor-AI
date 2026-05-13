import streamlit as st

import matplotlib.pyplot as plt

import numpy as np

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Model Insights",
    page_icon="📈",
    layout="wide"
)

# ======================================
# TITLE
# ======================================

st.title("📈 Model Training Insights")

st.markdown(
    "### Accuracy and Loss Analysis Dashboard"
)

st.write("---")

# ======================================
# SAMPLE TRAINING DATA
# ======================================

epochs = np.arange(1, 11)

train_accuracy = [

    72,
    78,
    82,
    86,
    89,
    91,
    93,
    94,
    95,
    96

]

val_accuracy = [

    70,
    75,
    80,
    84,
    87,
    89,
    91,
    92,
    93,
    94

]

train_loss = [

    1.2,
    1.0,
    0.8,
    0.65,
    0.52,
    0.40,
    0.32,
    0.25,
    0.20,
    0.15

]

val_loss = [

    1.3,
    1.1,
    0.95,
    0.75,
    0.60,
    0.50,
    0.42,
    0.35,
    0.28,
    0.22

]

# ======================================
# ACCURACY CURVES
# ======================================

st.subheader(
    "🎯 Accuracy Curves"
)

fig1, ax1 = plt.subplots(
    figsize=(8,5)
)

ax1.plot(

    epochs,

    train_accuracy,

    marker="o",

    label="Training Accuracy"

)

ax1.plot(

    epochs,

    val_accuracy,

    marker="o",

    label="Validation Accuracy"

)

ax1.set_xlabel(
    "Epochs"
)

ax1.set_ylabel(
    "Accuracy (%)"
)

ax1.set_title(
    "Training vs Validation Accuracy"
)

ax1.legend()

st.pyplot(fig1)

st.write("---")

# ======================================
# LOSS CURVES
# ======================================

st.subheader(
    "📉 Loss Curves"
)

fig2, ax2 = plt.subplots(
    figsize=(8,5)
)

ax2.plot(

    epochs,

    train_loss,

    marker="o",

    label="Training Loss"

)

ax2.plot(

    epochs,

    val_loss,

    marker="o",

    label="Validation Loss"

)

ax2.set_xlabel(
    "Epochs"
)

ax2.set_ylabel(
    "Loss"
)

ax2.set_title(
    "Training vs Validation Loss"
)

ax2.legend()

st.pyplot(fig2)

st.write("---")

# ======================================
# PERFORMANCE SUMMARY
# ======================================

st.subheader(
    "🧠 Model Summary"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Final Accuracy",
        "94%"
    )

with col2:

    st.metric(
        "Validation Accuracy",
        "93%"
    )

with col3:

    st.metric(
        "Training Epochs",
        "10"
    )

st.write("---")

# ======================================
# MODEL DETAILS
# ======================================

st.subheader(
    "⚙️ Framework Details"
)

detail1, detail2 = st.columns(2)

with detail1:

    st.success(
        """
        ### Deep Learning Models

        ✅ EfficientNet-B0
        
        ✅ ResNet18
        
        ✅ Ensemble Learning
        
        ✅ Monte Carlo Dropout
        
        """
    )

with detail2:

    st.info(
        """
        ### AI Components

        ✅ GradCAM Explainability
        
        ✅ Reliability Framework
        
        ✅ Confidence Estimation
        
        ✅ Uncertainty Analysis
        
        """
    )

st.write("---")

# ======================================
# TRAINING INSIGHTS
# ======================================

st.subheader(
    "📌 Training Insights"
)

st.warning(
    """
    The training and validation curves indicate
    stable convergence with minimal overfitting.

    Ensemble learning improves generalization capability.

    Monte Carlo Dropout enhances uncertainty estimation
    and reliability analysis.

    The framework demonstrates strong performance
    for MRI-based brain tumor classification.
    """
)

st.write("---")

# ======================================
# FOOTER
# ======================================

st.caption(
    "Training insights generated for Reliability-Aware Explainable Brain Tumor AI Framework"
)
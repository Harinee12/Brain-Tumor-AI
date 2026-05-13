import streamlit as st

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Project Overview",
    page_icon="🧠",
    layout="wide"
)

# ======================================
# TITLE
# ======================================

st.title("🧠 Project Overview")

st.markdown(
    "### Reliability-Aware Explainable Brain Tumor Detection Framework"
)

st.write("---")

# ======================================
# INTRODUCTION
# ======================================

st.subheader(
    "📌 Project Introduction"
)

st.info(
    """
    This project presents a deep learning-based framework
    for automated brain tumor classification using MRI images.

    The system combines:
    
    • Ensemble Learning
    
    • Explainable AI
    
    • Reliability Analysis
    
    • Uncertainty Estimation
    
    to improve trustworthiness and performance
    in medical image analysis.
    """
)

st.write("---")

# ======================================
# WORKFLOW
# ======================================

st.subheader(
    "⚙️ System Workflow"
)

st.code(
    """
MRI Upload
     ↓
Image Preprocessing
     ↓
EfficientNet Prediction
     ↓
ResNet Prediction
     ↓
Ensemble Learning
     ↓
Confidence Estimation
     ↓
Monte Carlo Dropout
     ↓
Uncertainty Analysis
     ↓
GradCAM Explainability
     ↓
Final Reliable Prediction
    """
)

st.write("---")

# ======================================
# TECHNOLOGIES
# ======================================

st.subheader(
    "🛠️ Technologies Used"
)

col1, col2 = st.columns(2)

with col1:

    st.success(
        """
        ### Deep Learning

        ✅ PyTorch
        
        ✅ EfficientNet
        
        ✅ ResNet18
        
        ✅ Ensemble Learning
        
        """
    )

with col2:

    st.info(
        """
        ### Deployment & Visualization

        ✅ Streamlit
        
        ✅ OpenCV
        
        ✅ GradCAM
        
        ✅ Matplotlib
        
        """
    )

st.write("---")

# ======================================
# FEATURES
# ======================================

st.subheader(
    "🚀 Key Features"
)

feature1, feature2, feature3 = st.columns(3)

with feature1:

    st.metric(
        "Models",
        "2"
    )

    st.caption(
        "EfficientNet + ResNet"
    )

with feature2:

    st.metric(
        "Framework",
        "Ensemble"
    )

    st.caption(
        "Reliability-Aware AI"
    )

with feature3:

    st.metric(
        "Explainability",
        "GradCAM"
    )

    st.caption(
        "Visual Attention Mapping"
    )

st.write("---")

# ======================================
# AI COMPONENTS
# ======================================

st.subheader(
    "🧪 AI Components"
)

st.warning(
    """
    ✅ Ensemble Learning improves robustness
    
    ✅ Monte Carlo Dropout estimates uncertainty
    
    ✅ GradCAM improves interpretability
    
    ✅ Reliability framework enhances trustworthiness
    
    ✅ Confidence estimation validates predictions
    """
)

st.write("---")

# ======================================
# APPLICATIONS
# ======================================

st.subheader(
    "🏥 Applications"
)

st.success(
    """
    • Brain Tumor Classification
    
    • Medical Image Analysis
    
    • AI-Assisted Diagnosis
    
    • Clinical Decision Support
    
    • Explainable Healthcare AI
    """
)

st.write("---")

# ======================================
# FOOTER
# ======================================

st.caption(
    "Developed as a Reliability-Aware Explainable Deep Learning Framework for Brain Tumor MRI Classification"
)
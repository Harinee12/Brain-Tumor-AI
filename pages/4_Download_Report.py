import streamlit as st

from fpdf import FPDF

from datetime import datetime

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Download Report",
    page_icon="📄",
    layout="wide"
)

# ======================================
# TITLE
# ======================================

st.title("📄 Download AI Prediction Report")

st.markdown(
    "### Generate Professional MRI Analysis Report"
)

st.write("---")

# ======================================
# SAMPLE REPORT DATA
# ======================================

prediction = "Meningioma"

confidence = "97.64%"

uncertainty = "0.0000"

reliability = "Highly Reliable"

timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

# ======================================
# PDF GENERATION
# ======================================

def create_pdf():

    pdf = FPDF()

    pdf.add_page()

    pdf.set_font(
        "Arial",
        "B",
        18
    )

    pdf.cell(
        200,
        10,
        txt="Brain Tumor AI Analysis Report",
        ln=True,
        align="C"
    )

    pdf.ln(10)

    pdf.set_font(
        "Arial",
        size=12
    )

    pdf.cell(
        200,
        10,
        txt=f"Prediction: {prediction}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Confidence Score: {confidence}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Uncertainty Score: {uncertainty}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Reliability Status: {reliability}",
        ln=True
    )

    pdf.cell(
        200,
        10,
        txt=f"Generated On: {timestamp}",
        ln=True
    )

    pdf.ln(10)

    pdf.multi_cell(
        0,
        10,
        txt=
        "This report was generated using a "
        "Reliability-Aware Explainable Deep "
        "Learning Framework for Brain Tumor "
        "MRI Classification."
    )

    pdf.output(
        "reports/Brain_Tumor_Report.pdf"
    )

# ======================================
# BUTTON
# ======================================

if st.button(
    "📥 Generate PDF Report"
):

    create_pdf()

    with open(
        "reports/Brain_Tumor_Report.pdf",
        "rb"
    ) as file:

        st.download_button(

            label="⬇ Download Report",

            data=file,

            file_name="Brain_Tumor_Report.pdf",

            mime="application/pdf"
        )

    st.success(
        "PDF Report Generated Successfully!"
    )

st.write("---")

# ======================================
# REPORT PREVIEW
# ======================================

st.subheader(
    "📋 Report Preview"
)

st.info(
    f"""
    Prediction: {prediction}
    
    Confidence Score: {confidence}
    
    Uncertainty Score: {uncertainty}
    
    Reliability Status: {reliability}
    
    Generated On: {timestamp}
    """
)

# ======================================
# FOOTER
# ======================================

st.caption(
    "Professional AI report generation system for Brain Tumor MRI Analysis"
)
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from config import FEATURES, OUTPUT_DIR, METRICS_PATH
from src.data_loader import load_dataset
from src.training import train_and_evaluate, load_trained_model
from src.prediction import predict_single

st.set_page_config(page_title="GlucoSense | Diabetes Analytics", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp {background: #0b1220; color: #e5edf8;}
[data-testid="stSidebar"] {background: #111c2e;}
.block-container {padding-top: 2rem; max-width: 1400px;}
div[data-testid="stMetric"] {background:#142238; padding:18px; border:1px solid #263a55; border-radius:14px;}
.hero {padding:26px 30px; border-radius:18px; background:linear-gradient(110deg,#12304a,#17243d); border:1px solid #29445f;}
.hero h1 {color:#f4f8ff; margin:0;} .hero p {color:#a9c0d8; margin:8px 0 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_dataset()

st.markdown('<div class="hero"><h1>GlucoSense</h1><p>Diabetes risk analytics • Machine learning decision-support demonstration</p></div>', unsafe_allow_html=True)
st.caption("Educational use only. This application is not a diagnostic tool and must not replace professional medical advice.")

with st.sidebar:
    st.title("Navigation")
    page = st.radio("Choose a workspace", ["Risk Assessment", "Model Performance", "Dataset Explorer"])
    st.divider()
    st.caption("Model training uses a stratified hold-out test set and cross-validation.")

try:
    df = get_data()
except Exception as e:
    st.error(str(e))
    st.stop()

if page == "Risk Assessment":
    st.subheader("Individual risk assessment")
    st.write("Enter the measurements below. Use values from a valid clinical report.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose (mg/dL)", 0.0, 300.0, 120.0)
    with col2:
        blood_pressure = st.number_input("Blood pressure (mm Hg)", 0.0, 200.0, 70.0)
        skin = st.number_input("Skin thickness (mm)", 0.0, 100.0, 20.0)
    with col3:
        insulin = st.number_input("Insulin (mu U/mL)", 0.0, 900.0, 80.0)
        bmi = st.number_input("BMI", 0.0, 80.0, 28.0)
    with col4:
        pedigree = st.number_input("Diabetes pedigree function", 0.0, 3.0, 0.5, step=0.01)
        age = st.number_input("Age (years)", 18, 120,  30)
    if st.button("Analyze risk", type="primary", use_container_width=True):
        try:
            model = load_trained_model()
        except FileNotFoundError:
            with st.spinner("Training and evaluating models for the first time..."):
                model, _, _ = train_and_evaluate(df)
            st.success("Model training completed.")
        result = predict_single(model, {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": blood_pressure,
            "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree, "Age": age
        })
        score = result["risk_probability"] * 100
        st.metric("Estimated model probability", f"{score:.1f}%")
        if result["prediction"] == 1:
            st.warning("The model classified this record in the positive class. Please consult a qualified healthcare professional for proper testing and interpretation.")
        else:
            st.info("The model classified this record in the negative class. This does not rule out diabetes; consult a healthcare professional if concerned.")
        st.progress(min(max(result["risk_probability"], 0.0), 1.0))

elif page == "Model Performance":
    st.subheader("Model performance")
    if not METRICS_PATH.exists():
        if st.button("Train and compare models", type="primary"):
            with st.spinner("Training models with stratified cross-validation..."):
                _, payload, _ = train_and_evaluate(df)
            st.success("Training complete. Reopen this page to view metrics.")
            st.rerun()
        st.info("Train the models to generate the evaluation report.")
    else:
        payload = json.loads(METRICS_PATH.read_text())
        m = payload["test_metrics"]
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Accuracy", f'{m["accuracy"]*100:.1f}%')
        c2.metric("Precision", f'{m["precision"]*100:.1f}%')
        c3.metric("Recall", f'{m["recall"]*100:.1f}%')
        c4.metric("F1 score", f'{m["f1"]*100:.1f}%')
        c5.metric("ROC-AUC", f'{m["roc_auc"]:.3f}')
        st.write("Selected model:", f'**{payload["best_model"]}**')
        board = pd.DataFrame(payload["leaderboard"])
        st.plotly_chart(px.bar(board, x="model", y="cv_roc_auc_mean",
                               error_y="cv_roc_auc_std", title="Cross-validation ROC-AUC comparison"),
                        use_container_width=True)
        for filename, title in [("confusion_matrix.png","Confusion matrix"),
                                ("roc_curve.png","ROC curve"),
                                ("correlation_heatmap.png","Feature correlation")]:
            path = OUTPUT_DIR / filename
            if path.exists():
                st.image(str(path), caption=title, use_container_width=True)

else:
    st.subheader("Dataset explorer")
    a,b,c = st.columns(3)
    a.metric("Records", f"{len(df):,}")
    b.metric("Input features", len(FEATURES))
    c.metric("Positive class", f'{df["Outcome"].mean()*100:.1f}%')
    st.dataframe(df.head(30), use_container_width=True)
    st.plotly_chart(px.histogram(df, x="Glucose", color="Outcome",
                                 barmode="overlay", title="Glucose distribution by outcome"),
                    use_container_width=True)
    st.plotly_chart(px.scatter(df, x="BMI", y="Glucose", color="Outcome",
                               title="BMI vs glucose"),
                    use_container_width=True)
    st.download_button("Download dataset CSV", df.to_csv(index=False),
                       file_name="diabetes_dataset.csv", mime="text/csv")

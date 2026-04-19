import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(
    page_title="AI Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️"
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
}

h1 {
    font-size: 42px !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
    letter-spacing: 1px;
}

h2, h3 {
    font-size: 26px !important;
    color: #7dd3fc !important;
}

p, label, span {
    font-size: 16px !important;
    color: #cbd5f5 !important;
}

section[data-testid="stSidebar"] {
    background: #020617;
}

section[data-testid="stSidebar"] h1 {
    font-size: 24px !important;
    color: #38bdf8 !important;
}

.stTextInput > div > div > input {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    padding: 12px !important;
    font-size: 15px !important;
}

.stSelectbox div[data-baseweb="select"] {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 12px !important;
}

.stButton > button {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    border: none !important;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}

.stAlert {
    border-radius: 12px !important;
    font-size: 16px !important;
}

.block-container {
    padding-top: 2rem;
}

.css-1r6slb0 {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
}

.stColumns {
    gap: 1.5rem;
}

</style>
""", unsafe_allow_html=True)

working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model(path):
    return pickle.load(open(path, 'rb'))

diabetes_model = load_model(f'{working_dir}/save_models/diabetes_model.sav')
heart_model = load_model(f'{working_dir}/save_models/heart.sav')
parkinsons_model = load_model(f'{working_dir}/save_models/parkinsons_model.sav')

with st.sidebar:
    st.title("🧠 AI Health Assistant")
    st.markdown("Predict diseases using Machine Learning")

    selected = option_menu(
        "Navigation",
        ["Diabetes", "Heart Disease", "Parkinson's"],
        icons=['activity', 'heart', 'person'],
        menu_icon='hospital-fill'
    )

    st.markdown("---")
    st.info("⚠️ This tool is for educational purposes only.")

def get_float_inputs(inputs):
    try:
        return [float(x) for x in inputs]
    except:
        return None

def show_result(pred, label_positive, label_negative):
    if pred == 1:
        st.error(f"🚨 {label_positive}")
    else:
        st.success(f"✅ {label_negative}")

if selected == "Diabetes":
    st.title("🩸 Diabetes Prediction")
    st.markdown("Enter patient details below:")

    col1, col2, col3 = st.columns(3)

    Pregnancies = col1.text_input("Pregnancies")
    Glucose = col2.text_input("Glucose")
    BloodPressure = col3.text_input("Blood Pressure")
    SkinThickness = col1.text_input("Skin Thickness")
    Insulin = col2.text_input("Insulin")
    BMI = col3.text_input("BMI")
    DPF = col1.text_input("Diabetes Pedigree Function")
    Age = col2.text_input("Age")

    if st.button("🔍 Predict Diabetes"):
        inputs = get_float_inputs([Pregnancies, Glucose, BloodPressure,
                                  SkinThickness, Insulin, BMI, DPF, Age])

        if inputs:
            pred = diabetes_model.predict([inputs])[0]
            show_result(pred,
                        "High chance of Diabetes",
                        "Low chance of Diabetes")
        else:
            st.warning("⚠️ Please enter valid numeric values")

elif selected == "Heart Disease":
    st.title("❤️ Heart Disease Prediction")

    col1, col2, col3 = st.columns(3)

    age = col1.text_input("Age")
    sex = col2.selectbox("Sex", [0, 1])
    cp = col3.selectbox("Chest Pain Type", [0,1,2,3])

    trestbps = col1.text_input("Resting BP")
    chol = col2.text_input("Cholesterol")
    fbs = col3.selectbox("Fasting Blood Sugar >120", [0,1])

    restecg = col1.selectbox("ECG", [0,1,2])
    thalach = col2.text_input("Max Heart Rate")
    exang = col3.selectbox("Exercise Angina", [0,1])

    oldpeak = col1.text_input("ST Depression")
    slope = col2.selectbox("Slope", [0,1,2])
    ca = col3.selectbox("Major Vessels", [0,1,2,3])
    thal = col1.selectbox("Thal", [0,1,2])

    if st.button("🔍 Predict Heart Disease"):
        inputs = get_float_inputs([
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak,
            slope, ca, thal
        ])

        if inputs:
            pred = heart_model.predict([inputs])[0]
            show_result(pred,
                        "Risk of Heart Disease detected",
                        "Heart looks healthy")
        else:
            st.warning("⚠️ Invalid input detected")

elif selected == "Parkinson's":
    st.title("🧬 Parkinson's Prediction")

    cols = st.columns(5)

    fields = [
        "Fo","Fhi","Flo","Jitter%","JitterAbs",
        "RAP","PPQ","DDP","Shimmer","Shimmer_dB",
        "APQ3","APQ5","APQ","DDA","NHR",
        "HNR","RPDE","DFA","spread1","spread2",
        "D2","PPE"
    ]

    values = []
    for i, field in enumerate(fields):
        values.append(cols[i % 5].text_input(field))

    if st.button("🔍 Predict Parkinson's"):
        inputs = get_float_inputs(values)

        if inputs:
            pred = parkinsons_model.predict([inputs])[0]
            show_result(pred,
                        "Parkinson's detected",
                        "No Parkinson's detected")
        else:
            st.warning("⚠️ Fill all fields correctly")
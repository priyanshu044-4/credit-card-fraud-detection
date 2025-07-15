import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detector", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction features below to check if it's **Fraud** or **Legit**.")

# --- Input keys ---
input_keys = [f"V{i}" for i in range(1, 29)] + ["Amount_Norm"]

# --- Form state control ---
if "form_version" not in st.session_state:
    st.session_state.form_version = 0

# --- Fraud sample for autofill ---
fraud_sample = {
    "V1": -2.3122, "V2": 1.9519, "V3": -1.6098, "V4": 3.9979, "V5": -0.5222,
    "V6": -1.4265, "V7": -2.5374, "V8": 1.3916, "V9": -2.7700, "V10": -2.7722,
    "V11": 3.2020, "V12": -1.8574, "V13": -1.2246, "V14": -2.5458, "V15": -1.6820,
    "V16": 1.4761, "V17": 0.0851, "V18": 0.2451, "V19": 0.5200, "V20": 0.2031,
    "V21": 0.1653, "V22": 0.0627, "V23": 0.0615, "V24": -0.0773, "V25": 0.1782,
    "V26": 0.5080, "V27": -0.0266, "V28": -0.8731, "Amount_Norm": 0.6203,
}
st.markdown("### 🧪 Demo Autofill")

if st.button("✅ Autofill Legit Transaction"):
    legit_sample = {
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3781, "V5": -0.3383,
        "V6": 0.4624, "V7": 0.2396, "V8": 0.0987, "V9": 0.3638, "V10": 0.0908,
        "V11": -0.5516, "V12": -0.6178, "V13": -0.9914, "V14": -0.3111, "V15": 1.4682,
        "V16": -0.4704, "V17": 0.2076, "V18": 0.0257, "V19": 0.4039, "V20": 0.2514,
        "V21": -0.0183, "V22": 0.2778, "V23": -0.1105, "V24": 0.0669, "V25": 0.1285,
        "V26": -0.1891, "V27": 0.1336, "V28": -0.0210,
        "Amount_Norm": 0.0013
    }
    for key, value in legit_sample.items():
        st.session_state[key] = value
    st.rerun()
# --- Buttons outside form ---
colA, colB = st.columns([1, 1])
if colA.button("🎲 Autofill Fraud Sample"):
    for k, v in fraud_sample.items():
        st.session_state[k] = v
    st.session_state.form_version += 1
    st.rerun()

if colB.button("🔄 Reset Form"):
    for k in input_keys:
        st.session_state[k] = 0.0
    st.session_state.form_version += 1
    st.rerun()

# --- Input form ---
with st.form("fraud_form"):
    cols = st.columns(3)
    inputs = {}

    for i, key in enumerate(input_keys):
        widget_key = f"{key}_{st.session_state.form_version}"
        inputs[key] = cols[i % 3].number_input(
            key, format="%.4f", value=st.session_state.get(key, 0.0), key=widget_key
        )

    submitted = st.form_submit_button("🔍 Check Fraud")

# --- Prediction ---
if submitted:
    try:
        response = requests.post("http://127.0.0.1:5000/predict", json=inputs)
        if response.status_code == 200:
            result = response.json()
            if result["prediction"] == 1:
                st.error("❌ This transaction is likely **Fraud**")
            else:
                st.success("✅ This transaction is **Legit**")
        else:
            st.warning("⚠️ API error: " + response.text)
    except Exception as e:
        st.error(f"🚫 Could not connect to API: {e}")

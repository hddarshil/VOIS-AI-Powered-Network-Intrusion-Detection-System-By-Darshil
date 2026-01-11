import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from groq import Groq

# --- ⚙️ PAGE CONFIG ---
st.set_page_config(page_title="AI-NIDS Student Project", page_icon="🛡️", layout="wide")

# --- 📄 FILE NAMES ---
CSV_FILES = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"
]

# --- 🚀 HEADER ---
st.title("🛡️ AI-Based Network Intrusion Detection System")
st.markdown("""
**Student Project**: Merging Random Forest Detection with **Groq AI Analyst** for deep packet explanation.
""")

# --- 🛠️ DATA ENGINE ---
@st.cache_data
def load_data_from_local():
    all_dfs = []
    found_files = []
    
    for file in CSV_FILES:
        if os.path.exists(file):
            # Reading a subset (e.g., 20k rows) to keep it fast for student project
            df_temp = pd.read_csv(file, nrows=20000, low_memory=False)
            all_dfs.append(df_temp)
            found_files.append(file)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, axis=0, ignore_index=True)
        combined_df.columns = combined_df.columns.str.strip()
        combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined_df.dropna(inplace=True)
        return combined_df, found_files
    return None, []

# --- 🕹️ SIDEBAR ---
st.sidebar.header("🔑 1. API Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key (gsk_...)", type="password")
st.sidebar.caption("[Get a free key here](https://console.groq.com/keys)")

st.sidebar.header("🕹️ 2. Controls")
if st.sidebar.button("🚀 INITIALIZE AI ENGINE"):
    with st.spinner("Processing Dataset..."):
        data, files = load_data_from_local()
        if data is not None:
            st.session_state['nids_final_data'] = data
            st.session_state['files_found'] = files
            st.sidebar.success(f"Loaded {len(data)} rows from {len(files)} files!")
            st.balloons()
        else:
            st.sidebar.error("No CSV files found in the folder!")

# --- 📊 DASHBOARD DISPLAY ---
if 'nids_final_data' in st.session_state:
    df = st.session_state['nids_final_data']
    
    # 💎 Top Analytics
    st.subheader("📊 Network Traffic Intelligence")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Logs Analyzed", f"{len(df):,}")
    c2.metric("Detected Attack Classes", df['Label'].nunique())
    c3.metric("System Health", "Optimal")

    # 🧠 MACHINE LEARNING TRAINING
    st.divider()
    
    # Features selection
    features = ['Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 
                'Total Length of Fwd Packets', 'Fwd Packet Length Max', 'Flow IAT Mean']
    
    X = df[features]
    y = df['Label']
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42)
    
    # Train Model (Using Session State to avoid retraining on every click)
    if 'rf_model' not in st.session_state:
        with st.spinner("🤖 Training AI Intelligence..."):
            rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
            rf.fit(X_train, y_train)
            st.session_state['rf_model'] = rf
            st.session_state['le'] = le
            st.session_state['X_test'] = X_test
            st.session_state['y_test_enc'] = y_test

    # --- PERFORMANCE VISUALS ---
    col_a, col_b = st.columns([1.5, 1])
    with col_a:
        st.markdown("#### 🎯 Model Performance")
        y_pred = st.session_state['rf_model'].predict(st.session_state['X_test'])
        acc = accuracy_score(st.session_state['y_test_enc'], y_pred)
        st.success(f"Model Accuracy: {acc*100:.2f}%")
        
        fig_cm, ax_cm = plt.subplots(figsize=(8, 5))
        cm = confusion_matrix(st.session_state['y_test_enc'], y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Threat Detection Heatmap")
        st.pyplot(fig_cm)

    with col_b:
        st.markdown("#### 📊 Threat Distribution")
        fig_bar, ax_bar = plt.subplots()
        df['Label'].value_counts().plot(kind='bar', color='teal')
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)

    # --- 🤖 AI AGENT SECTION (THE PACKET ANALYST) ---
    st.divider()
    st.header("🔬 Live Threat Analyst (AI Agent)")
    
    ana1, ana2 = st.columns([1, 1.5])
    
    with ana1:
        st.info("Pick a packet to analyze with Groq AI")
        if st.button("🎲 Capture & Analyze Random Packet"):
            # Select random packet
            random_idx = np.random.randint(0, len(X_test))
            sample_packet = X_test.iloc[random_idx]
            actual_idx = y_test[random_idx]
            actual_label = le.inverse_transform([actual_idx])[0]
            
            # Predict
            pred_idx = st.session_state['rf_model'].predict([sample_packet])[0]
            pred_label = le.inverse_transform([pred_idx])[0]
            
            st.session_state['current_sample'] = sample_packet
            st.session_state['current_pred'] = pred_label
            st.session_state['current_actual'] = actual_label

    if 'current_sample' in st.session_state:
        with ana1:
            st.write("**Packet Technical Features:**")
            st.dataframe(st.session_state['current_sample'], use_container_width=True)
            
            if st.session_state['current_pred'] == "BENIGN":
                st.success(f"✅ Detection: {st.session_state['current_pred']}")
            else:
                st.error(f"🚨 Detection: {st.session_state['current_pred']}")
            st.caption(f"Ground Truth: {st.session_state['current_actual']}")

        with ana2:
            st.subheader("🤖 Groq AI Analysis")
            if st.button("Ask AI Agent to Explain"):
                if not groq_api_key:
                    st.warning("Please enter Groq API Key in the sidebar!")
                else:
                    try:
                        client = Groq(api_key=groq_api_key)
                        
                        prompt = f"""
                        You are a Senior Cybersecurity Analyst. 
                        Our NIDS system detected a packet as: {st.session_state['current_pred']}.
                        
                        Packet Details:
                        {st.session_state['current_sample'].to_string()}
                        
                        Please explain to a student:
                        1. Why these specific values might indicate a {st.session_state['current_pred']} status.
                        2. If it is an attack, what are the risks? If BENIGN, why does it look safe?
                        3. Keep the explanation professional but easy to understand.
                        """

                        with st.spinner("AI Agent is thinking..."):
                            chat_completion = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama-3.3-70b-versatile",
                                temperature=0.5
                            )
                            st.markdown("### Agent Report:")
                            st.write(chat_completion.choices[0].message.content)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

else:
    st.info("👋 Welcome! Make sure the 4 CSV files are in the folder and click 'INITIALIZE AI ENGINE'.")
    
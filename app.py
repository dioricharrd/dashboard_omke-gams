import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Prediksi Depresi Mahasiswa",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Model Neural Network
class DepressionClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, dropout=0.3):
        super(DepressionClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.bn2 = nn.BatchNorm1d(hidden_size//2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size//2, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Fungsi untuk memuat dan memproses data
@st.cache_data
def load_and_process_data():
    """Load dan preprocess dataset"""
    try:
        # Membaca dataset dari file yang diupload
        data = pd.read_csv('student_depression_dataset.csv')
        
        # Data cleaning sesuai laporan
        columns_to_drop = ['id', 'City', 'Profession', 'Work Pressure', 'Job Satisfaction']
        data_clean = data.drop(columns=columns_to_drop, errors='ignore')
        
        # Encoding categorical variables
        le_gender = LabelEncoder()
        le_sleep = LabelEncoder()
        le_dietary = LabelEncoder()
        le_degree = LabelEncoder()
        le_suicidal = LabelEncoder()
        le_family = LabelEncoder()
        
        data_clean['Gender'] = le_gender.fit_transform(data_clean['Gender'])
        data_clean['Sleep Duration'] = le_sleep.fit_transform(data_clean['Sleep Duration'])
        data_clean['Dietary Habits'] = le_dietary.fit_transform(data_clean['Dietary Habits'])
        data_clean['Degree'] = le_degree.fit_transform(data_clean['Degree'])
        data_clean['Have you ever had suicidal thoughts ?'] = le_suicidal.fit_transform(data_clean['Have you ever had suicidal thoughts ?'])
        data_clean['Family History of Mental Illness'] = le_family.fit_transform(data_clean['Family History of Mental Illness'])
        
        # Convert Financial Stress to numeric if it's string
        if data_clean['Financial Stress'].dtype == 'object':
            data_clean['Financial Stress'] = pd.to_numeric(data_clean['Financial Stress'], errors='coerce')
        
        # Remove any rows with NaN values
        data_clean = data_clean.dropna()
        
        return data_clean, {
            'gender': le_gender,
            'sleep': le_sleep,
            'dietary': le_dietary,
            'degree': le_degree,
            'suicidal': le_suicidal,
            'family': le_family
        }
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Fungsi untuk training model
def train_models(X_train, X_test, y_train, y_test):
    """Training PyTorch dan XGBoost models"""
    models = {}
    metrics = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors for PyTorch
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1)
    y_test_tensor = torch.FloatTensor(y_test.values).reshape(-1, 1)
    
    # Train PyTorch model
    pytorch_model = DepressionClassifier(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)
    
    # Training loop
    pytorch_model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = pytorch_model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
    
    # Evaluate PyTorch model
    pytorch_model.eval()
    with torch.no_grad():
        y_pred_pytorch = pytorch_model(X_test_tensor)
        y_pred_pytorch_binary = (y_pred_pytorch > 0.5).float()
        pytorch_accuracy = (y_pred_pytorch_binary == y_test_tensor).float().mean().item()
        pytorch_auc = roc_auc_score(y_test, y_pred_pytorch.numpy())
    
    models['pytorch'] = {'model': pytorch_model, 'scaler': scaler}
    metrics['pytorch'] = {
        'accuracy': pytorch_accuracy,
        'auc': pytorch_auc,
        'predictions': y_pred_pytorch.numpy().flatten()
    }
    
    # Train XGBoost model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    y_pred_xgb_binary = xgb_model.predict(X_test_scaled)
    xgb_accuracy = (y_pred_xgb_binary == y_test).mean()
    xgb_auc = roc_auc_score(y_test, y_pred_xgb)
    
    models['xgboost'] = {'model': xgb_model, 'scaler': scaler}
    metrics['xgboost'] = {
        'accuracy': xgb_accuracy,
        'auc': xgb_auc,
        'predictions': y_pred_xgb,
        'feature_importance': xgb_model.feature_importances_
    }
    
    return models, metrics

# Main app
def main():
    st.markdown('<h1 class="main-header">🧠 Dashboard Prediksi Depresi Mahasiswa</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Sistem Prediksi Berbasis Machine Learning untuk Deteksi Dini Depresi pada Mahasiswa</p>', unsafe_allow_html=True)
    
    # Load data
    data, encoders = load_and_process_data()
    
    if data is None:
        st.error("❌ Gagal memuat dataset. Pastikan file 'student_depression_dataset.csv' tersedia.")
        return
    
    # Sidebar
    st.sidebar.header("🔧 Pengaturan Dashboard")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Pilih Halaman:",
        ["📊 Exploratory Data Analysis", "🤖 Model Training & Evaluation", "🎯 Prediksi Individual", "📈 Feature Analysis", "📋 Model Comparison"]
    )
    
    # EDA Page
    if page == "📊 Exploratory Data Analysis":
        st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
        
        # Dataset overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Sampel", f"{len(data):,}")
        with col2:
            st.metric("Jumlah Fitur", data.shape[1]-1)
        with col3:
            depression_rate = (data['Depression'].sum() / len(data)) * 100
            st.metric("Tingkat Depresi", f"{depression_rate:.1f}%")
        with col4:
            missing_values = data.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        
        # Distribution plots
        col1, col2 = st.columns(2)
        
        with col1:
            # Depression distribution
            fig_dist = px.pie(
                values=data['Depression'].value_counts().values,
                names=['Tidak Depresi', 'Depresi'],
                title="Distribusi Kasus Depresi",
                color_discrete_sequence=['#2ecc71', '#e74c3c']
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Age distribution by depression
            fig_age = px.histogram(
                data, x='Age', color='Depression',
                title="Distribusi Usia berdasarkan Status Depresi",
                nbins=20,
                color_discrete_sequence=['#3498db', '#e74c3c']
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Matriks Korelasi Fitur")
        
        # Select numeric columns for correlation
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            title="Korelasi Antar Fitur",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig_corr.update_layout(height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature distributions
        st.subheader("📈 Distribusi Fitur Utama")
        
        feature_cols = ['Academic Pressure', 'Financial Stress', 'Work/Study Hours', 'CGPA']
        
        for i in range(0, len(feature_cols), 2):
            col1, col2 = st.columns(2)
            
            for j, col in enumerate([col1, col2]):
                if i + j < len(feature_cols):
                    feature = feature_cols[i + j]
                    if feature in data.columns:
                        fig_feat = px.box(
                            data, y=feature, color='Depression',
                            title=f"Distribusi {feature}",
                            color_discrete_sequence=['#3498db', '#e74c3c']
                        )
                        col.plotly_chart(fig_feat, use_container_width=True)
    
    # Model Training Page
    elif page == "🤖 Model Training & Evaluation":
        st.markdown('<h2 class="sub-header">🤖 Model Training & Evaluation</h2>', unsafe_allow_html=True)
        
        # Prepare features and target
        feature_cols = [col for col in data.columns if col != 'Depression']
        X = data[feature_cols]
        y = data['Depression']
        
        # Train/test split
        test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        if st.button("🚀 Mulai Training Model", type="primary"):
            with st.spinner("Training models... Mohon tunggu..."):
                models, metrics = train_models(X_train, X_test, y_train, y_test)
                st.session_state.models = models
                st.session_state.metrics = metrics
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
        
        if 'models' in st.session_state:
            # Model comparison metrics
            st.subheader("📊 Performa Model")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**PyTorch Neural Network**")
                pytorch_metrics = st.session_state.metrics['pytorch']
                st.metric("Accuracy", f"{pytorch_metrics['accuracy']:.4f}")
                st.metric("AUC Score", f"{pytorch_metrics['auc']:.4f}")
            
            with col2:
                st.markdown("**XGBoost**")
                xgb_metrics = st.session_state.metrics['xgboost']
                st.metric("Accuracy", f"{xgb_metrics['accuracy']:.4f}")
                st.metric("AUC Score", f"{xgb_metrics['auc']:.4f}")
            
            # ROC Curves
            st.subheader("📈 ROC Curves Comparison")
            
            fig_roc = go.Figure()
            
            # PyTorch ROC
            fpr_pt, tpr_pt, _ = roc_curve(st.session_state.y_test, pytorch_metrics['predictions'])
            fig_roc.add_trace(go.Scatter(
                x=fpr_pt, y=tpr_pt,
                mode='lines',
                name=f"PyTorch (AUC = {pytorch_metrics['auc']:.3f})",
                line=dict(color='#e74c3c', width=2)
            ))
            
            # XGBoost ROC
            fpr_xgb, tpr_xgb, _ = roc_curve(st.session_state.y_test, xgb_metrics['predictions'])
            fig_roc.add_trace(go.Scatter(
                x=fpr_xgb, y=tpr_xgb,
                mode='lines',
                name=f"XGBoost (AUC = {xgb_metrics['auc']:.3f})",
                line=dict(color='#3498db', width=2)
            ))
            
            # Random classifier line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=1, dash='dash')
            ))
            
            fig_roc.update_layout(
                title="ROC Curve Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                width=800, height=500
            )
            
            st.plotly_chart(fig_roc, use_container_width=True)
            
            # Confusion matrices
            st.subheader("🎯 Confusion Matrices")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # PyTorch confusion matrix
                y_pred_pt_binary = (pytorch_metrics['predictions'] > 0.5).astype(int)
                cm_pt = confusion_matrix(st.session_state.y_test, y_pred_pt_binary)
                
                fig_cm_pt = px.imshow(
                    cm_pt,
                    text_auto=True,
                    title="PyTorch Confusion Matrix",
                    color_continuous_scale="Blues",
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(fig_cm_pt, use_container_width=True)
            
            with col2:
                # XGBoost confusion matrix
                y_pred_xgb_binary = (xgb_metrics['predictions'] > 0.5).astype(int)
                cm_xgb = confusion_matrix(st.session_state.y_test, y_pred_xgb_binary)
                
                fig_cm_xgb = px.imshow(
                    cm_xgb,
                    text_auto=True,
                    title="XGBoost Confusion Matrix",
                    color_continuous_scale="Reds",
                    labels=dict(x="Predicted", y="Actual")
                )
                st.plotly_chart(fig_cm_xgb, use_container_width=True)
    
    # Individual Prediction Page
    elif page == "🎯 Prediksi Individual":
        st.markdown('<h2 class="sub-header">🎯 Prediksi Individual</h2>', unsafe_allow_html=True)
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Mohon training model terlebih dahulu di halaman 'Model Training & Evaluation'")
            return
        
        st.subheader("📝 Input Data Mahasiswa")
        
        # Create input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.number_input("Usia", min_value=17, max_value=40, value=20)
            academic_pressure = st.slider("Tekanan Akademik (1-5)", 1, 5, 3)
            cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        
        with col2:
            study_satisfaction = st.slider("Kepuasan Belajar (1-5)", 1, 5, 3)
            sleep_duration = st.selectbox("Durasi Tidur", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
            dietary_habits = st.selectbox("Kebiasaan Makan", ["Healthy", "Moderate", "Unhealthy"])
            degree = st.selectbox("Tingkat Pendidikan", ["Bachelor", "Master", "PhD"])
        
        with col3:
            suicidal_thoughts = st.selectbox("Pernah Berpikiran Bunuh Diri?", ["No", "Yes"])
            work_study_hours = st.number_input("Jam Kerja/Belajar per Hari", min_value=0, max_value=24, value=8)
            financial_stress = st.slider("Stress Finansial (1-5)", 1, 5, 3)
            family_history = st.selectbox("Riwayat Kesehatan Mental Keluarga", ["No", "Yes"])
        
        if st.button("🔮 Prediksi Depresi", type="primary"):
            # Prepare input data
            input_data = {
                'Gender': 1 if gender == "Male" else 0,
                'Age': age,
                'Academic Pressure': academic_pressure,
                'CGPA': cgpa,
                'Study Satisfaction': study_satisfaction,
                'Sleep Duration': ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"].index(sleep_duration),
                'Dietary Habits': ["Healthy", "Moderate", "Unhealthy"].index(dietary_habits),
                'Degree': ["Bachelor", "Master", "PhD"].index(degree),
                'Have you ever had suicidal thoughts ?': 1 if suicidal_thoughts == "Yes" else 0,
                'Work/Study Hours': work_study_hours,
                'Financial Stress': financial_stress,
                'Family History of Mental Illness': 1 if family_history == "Yes" else 0
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make predictions
            models = st.session_state.models
            
            # PyTorch prediction
            pytorch_model = models['pytorch']['model']
            scaler = models['pytorch']['scaler']
            input_scaled = scaler.transform(input_df)
            input_tensor = torch.FloatTensor(input_scaled)
            
            pytorch_model.eval()
            with torch.no_grad():
                pytorch_pred = pytorch_model(input_tensor).item()
            
            # XGBoost prediction
            xgb_model = models['xgboost']['model']
            xgb_pred = xgb_model.predict_proba(input_scaled)[0, 1]
            
            # Display results
            st.subheader("🎯 Hasil Prediksi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**PyTorch Neural Network**")
                pytorch_risk = pytorch_pred * 100
                if pytorch_pred > 0.7:
                    st.error(f"⚠️ RISIKO TINGGI: {pytorch_risk:.1f}%")
                elif pytorch_pred > 0.4:
                    st.warning(f"⚡ RISIKO SEDANG: {pytorch_risk:.1f}%")
                else:
                    st.success(f"✅ RISIKO RENDAH: {pytorch_risk:.1f}%")
            
            with col2:
                st.markdown("**XGBoost**")
                xgb_risk = xgb_pred * 100
                if xgb_pred > 0.7:
                    st.error(f"⚠️ RISIKO TINGGI: {xgb_risk:.1f}%")
                elif xgb_pred > 0.4:
                    st.warning(f"⚡ RISIKO SEDANG: {xgb_risk:.1f}%")
                else:
                    st.success(f"✅ RISIKO RENDAH: {xgb_risk:.1f}%")
            
            # Risk gauge chart
            avg_risk = (pytorch_pred + xgb_pred) / 2
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_risk * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Skor Risiko Depresi (%)"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgreen"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Recommendations
            if avg_risk > 0.7:
                st.markdown("""
                <div class="warning-box">
                <h4>🚨 Rekomendasi Tindakan Segera:</h4>
                <ul>
                <li>Konsultasi dengan konselor atau psikolog kampus</li>
                <li>Bergabung dengan program support group</li>
                <li>Pertimbangkan untuk mengurangi beban akademik sementara</li>
                <li>Cari dukungan dari keluarga dan teman terdekat</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            elif avg_risk > 0.4:
                st.markdown("""
                <div class="warning-box">
                <h4>⚡ Rekomendasi Pencegahan:</h4>
                <ul>
                <li>Lakukan aktivitas relaksasi dan mindfulness</li>
                <li>Atur jadwal tidur yang teratur (7-8 jam)</li>
                <li>Perbaiki pola makan dan olahraga rutin</li>
                <li>Manajemen waktu yang lebih baik</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-box">
                <h4>✅ Tetap Jaga Kesehatan Mental:</h4>
                <ul>
                <li>Pertahankan pola hidup sehat yang sudah baik</li>
                <li>Tetap aktif bersosialisasi dengan teman</li>
                <li>Lakukan hobi dan aktivitas yang menyenangkan</li>
                <li>Jaga keseimbangan antara akademik dan kehidupan pribadi</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
    
    # Feature Analysis Page
    elif page == "📈 Feature Analysis":
        st.markdown('<h2 class="sub-header">📈 Feature Analysis</h2>', unsafe_allow_html=True)
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Mohon training model terlebih dahulu di halaman 'Model Training & Evaluation'")
            return
        
        # Feature importance from XGBoost
        feature_importance = st.session_state.metrics['xgboost']['feature_importance']
        feature_names = [col for col in data.columns if col != 'Depression']
        
        # Create feature importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=True)
        
        # Feature importance plot
        fig_importance = px.bar(
            importance_df.tail(10),  # Top 10 features
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 10 Feature Importance (XGBoost)",
            color='Importance',
            color_continuous_scale='viridis'
        )
        fig_importance.update_layout(height=500)
        st.plotly_chart(fig_importance, use_container_width=True)
        
        # Feature correlation with target
        st.subheader("🔗 Korelasi Fitur dengan Depresi")
        
        correlations = []
        for feature in feature_names:
            if feature in data.columns:
                corr = data[feature].corr(data['Depression'])
                correlations.append({'Feature': feature, 'Correlation': corr})
        
        corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
        
        fig_corr = px.bar(
            corr_df.head(10),
            x='Correlation',
            y='Feature',
            orientation='h',
            title="Top 10 Korelasi Fitur dengan Depresi",
            color='Correlation',
            color_continuous_scale='RdBu'
        )
        fig_corr.update_layout(height=500)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Risk factor analysis
        st.subheader("🎯 Analisis Faktor Risiko")
        
        # Key insights dari laporan
        insights = {
            "Pemikiran Bunuh Diri": "31.27% kontribusi terhadap prediksi - faktor risiko tertinggi",
            "Tekanan Akademik": "18.45% kontribusi - mahasiswa dengan skor ≥4 berisiko 2.7x lebih tinggi",
            "Stres Finansial": "12.53% kontribusi - mahasiswa dengan skor 4-5 berisiko 2.3x lebih tinggi",
            "Durasi Tidur": "8.91% kontribusi - tidur <6 jam meningkatkan risiko 1.9x",
            "Jam Kerja/Belajar": "7.83% kontribusi - >10 jam/hari meningkatkan risiko 1.7x"
        }
        
        for factor, description in insights.items():
            st.markdown(f"**{factor}:** {description}")
        
        # Risk threshold visualization
        st.subheader("⚠️ Threshold Risiko untuk Intervensi")
        
        thresholds = {
            "Tekanan Akademik": {"threshold": 4, "scale": "1-5", "action": "Konseling akademik"},
            "Stres Finansial": {"threshold": 4, "scale": "1-5", "action": "Bantuan finansial"},
            "Durasi Tidur": {"threshold": 6, "scale": "jam/hari", "action": "Edukasi sleep hygiene"},
            "Jam Kerja/Belajar": {"threshold": 10, "scale": "jam/hari", "action": "Manajemen waktu"}
        }
        
        threshold_df = pd.DataFrame([
            {"Faktor": k, "Threshold": v["threshold"], "Skala": v["scale"], "Tindakan": v["action"]}
            for k, v in thresholds.items()
        ])
        
        st.dataframe(threshold_df, use_container_width=True)
    
    # Model Comparison Page
    elif page == "📋 Model Comparison":
        st.markdown('<h2 class="sub-header">📋 Model Comparison</h2>', unsafe_allow_html=True)
        
        if 'models' not in st.session_state:
            st.warning("⚠️ Mohon training model terlebih dahulu di halaman 'Model Training & Evaluation'")
            return
        
        # Comprehensive model comparison
        st.subheader("🔄 Perbandingan Komprehensif Model")
        
        pytorch_metrics = st.session_state.metrics['pytorch']
        xgb_metrics = st.session_state.metrics['xgboost']
        
        # Calculate additional metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_true = st.session_state.y_test
        
        # PyTorch metrics
        y_pred_pt = (pytorch_metrics['predictions'] > 0.5).astype(int)
        pt_precision = precision_score(y_true, y_pred_pt)
        pt_recall = recall_score(y_true, y_pred_pt)
        pt_f1 = f1_score(y_true, y_pred_pt)
        
        # XGBoost metrics
        y_pred_xgb = (xgb_metrics['predictions'] > 0.5).astype(int)
        xgb_precision = precision_score(y_true, y_pred_xgb)
        xgb_recall = recall_score(y_true, y_pred_xgb)
        xgb_f1 = f1_score(y_true, y_pred_xgb)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
            'PyTorch': [pytorch_metrics['accuracy'], pt_precision, pt_recall, pt_f1, pytorch_metrics['auc']],
            'XGBoost': [xgb_metrics['accuracy'], xgb_precision, xgb_recall, xgb_f1, xgb_metrics['auc']]
        })
        
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # Radar chart comparison
        fig_radar = go.Figure()
        
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        pytorch_values = [pytorch_metrics['accuracy'], pt_precision, pt_recall, pt_f1, pytorch_metrics['auc']]
        xgb_values = [xgb_metrics['accuracy'], xgb_precision, xgb_recall, xgb_f1, xgb_metrics['auc']]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=pytorch_values,
            theta=metrics_names,
            fill='toself',
            name='PyTorch',
            line_color='#e74c3c'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=xgb_values,
            theta=metrics_names,
            fill='toself',
            name='XGBoost',
            line_color='#3498db'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Radar Chart: Model Performance Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Model characteristics
        st.subheader("🔍 Karakteristik Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PyTorch Neural Network**")
            st.markdown("""
            **Kelebihan:**
            - Fleksibilitas arsitektur tinggi
            - Dapat menangkap pola non-linear kompleks
            - Batch normalization dan dropout untuk regularisasi
            - Gradient-based optimization
            
            **Kekurangan:**
            - Membutuhkan tuning hyperparameter lebih detail
            - Waktu training lebih lama
            - Interpretabilitas lebih rendah
            - Sensitif terhadap inisialisasi
            """)
        
        with col2:
            st.markdown("**XGBoost**")
            st.markdown("""
            **Kelebihan:**
            - Feature importance yang jelas
            - Robust terhadap outliers
            - Built-in regularization
            - Performa stabil dengan default parameters
            
            **Kekurangan:**
            - Kurang fleksibel untuk pola sangat kompleks
            - Memory intensive untuk dataset besar
            - Overfitting pada dataset kecil
            - Parameter tuning bisa rumit
            """)
        
        # Prediction confidence analysis
        st.subheader("📊 Analisis Confidence Prediksi")
        
        # Confidence distribution
        fig_conf = make_subplots(rows=1, cols=2, subplot_titles=(['PyTorch Confidence', 'XGBoost Confidence']))
        
        fig_conf.add_trace(
            go.Histogram(x=pytorch_metrics['predictions'], name='PyTorch', nbinsx=20),
            row=1, col=1
        )
        
        fig_conf.add_trace(
            go.Histogram(x=xgb_metrics['predictions'], name='XGBoost', nbinsx=20),
            row=1, col=2
        )
        
        fig_conf.update_layout(title="Distribusi Confidence Score", height=400)
        st.plotly_chart(fig_conf, use_container_width=True)
        
        # Error analysis summary
        st.subheader("⚠️ Ringkasan Error Analysis")
        
        # Calculate error regions
        pytorch_preds = pytorch_metrics['predictions']
        xgb_preds = xgb_metrics['predictions']
        
        # High confidence errors
        pt_high_conf_errors = np.sum((pytorch_preds > 0.8) & (y_pred_pt != y_true)) + np.sum((pytorch_preds < 0.2) & (y_pred_pt != y_true))
        xgb_high_conf_errors = np.sum((xgb_preds > 0.8) & (y_pred_xgb != y_true)) + np.sum((xgb_preds < 0.2) & (y_pred_xgb != y_true))
        
        # Uncertain predictions
        pt_uncertain = np.sum((pytorch_preds > 0.4) & (pytorch_preds < 0.6))
        xgb_uncertain = np.sum((xgb_preds > 0.4) & (xgb_preds < 0.6))
        
        error_summary = pd.DataFrame({
            'Model': ['PyTorch', 'XGBoost'],
            'High Confidence Errors': [pt_high_conf_errors, xgb_high_conf_errors],
            'Uncertain Predictions': [pt_uncertain, xgb_uncertain],
            'Total Errors': [np.sum(y_pred_pt != y_true), np.sum(y_pred_xgb != y_true)]
        })
        
        st.dataframe(error_summary, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Rekomendasi Implementasi")
        
        st.markdown("""
        **Berdasarkan analisis performa:**
        
        1. **Untuk Skrining Massal:** Gunakan XGBoost karena:
           - Interpretabilitas tinggi (feature importance jelas)
           - Performa stabil dan konsisten
           - Waktu prediksi lebih cepat
           
        2. **Untuk Penelitian Lanjutan:** Gunakan PyTorch karena:
           - Fleksibilitas untuk eksperimen arsitektur
           - Potensi improvement dengan tuning lebih detail
           - Dapat diintegrasikan dengan teknik deep learning lain
           
        3. **Ensemble Approach:** Kombinasi kedua model untuk:
           - Meningkatkan robustness prediksi
           - Mengurangi bias individual model
           - Confidence scoring yang lebih reliable
        """)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center'>
            <p><strong>Dashboard Prediksi Depresi Mahasiswa</strong></p>
            <p>Dikembangkan menggunakan PyTorch, XGBoost, dan Streamlit</p>
            <p style='font-size: 0.8rem; color: #666;'>
                Muhammad Zaky Darajat (202210370311052) & Dio Richard Prastiyo (202210370311061)
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
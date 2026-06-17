import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Prediksi Iklan", layout="wide")

# --- 1. PROSES DATA & MESIN ML (BERJALAN DI BACKGROUND) ---
@st.cache_resource
def siapkan_sistem():
    df = pd.read_csv('Social_Network_Ads.csv')
    X = df[['Gender', 'Age', 'EstimatedSalary']].copy()
    X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})
    
    X_train, X_test, y_train, y_test = train_test_split(X, df['Purchased'], test_size=0.25, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
    X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_scaled, y_train) #Agar data seimbang
    
    models = {
        "KNN": GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [5, 7]}, cv=3).fit(X_train_bal, y_train_bal).best_estimator_,
        "Decision Tree": GridSearchCV(DecisionTreeClassifier(random_state=42), {'max_depth': [3, 5]}, cv=3).fit(X_train_bal, y_train_bal).best_estimator_,
        "Naïve Bayes": GaussianNB().fit(X_train_bal, y_train_bal)
    }
    return df, models, scaler, X_test_scaled, y_test

df_asli, mesin_ml, scaler, X_test, y_test = siapkan_sistem()

# --- 2. ANTARMUKA WEB ---
st.title("Aplikasi Prediksi Konversi Iklan")
tab1, tab2, tab3 = st.tabs(["Eksplorasi Data", "Kinerja Model", "Coba Prediksi"])

with tab1:
    col1, col2 = st.columns(2)
    col1.dataframe(df_asli.head(6)) 
    
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x='Purchased', data=df_asli, hue='Purchased', palette='viridis', legend=False, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Tidak Beli', 'Beli'])
    col2.pyplot(fig)

with tab2:
    pilihan_model = st.selectbox("Pilih Model untuk Dievaluasi:", list(mesin_ml.keys()))
    prediksi_uji = mesin_ml[pilihan_model].predict(X_test)
    
    st.metric("Akurasi Pengujian", f"{accuracy_score(y_test, prediksi_uji) * 100:.2f}%")
    st.write("**Detail Laporan Klasifikasi:**")
    st.dataframe(pd.DataFrame(classification_report(y_test, prediksi_uji, output_dict=True)).T)

with tab3:
    # --- BAGIAN YANG DITAMBAHKAN ---
    model_prediksi = st.selectbox("Gunakan Metode Machine Learning:", list(mesin_ml.keys()))
    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["Female", "Male"])
    umur = c2.slider("Umur", 18, 60, 30)
    gaji = c3.number_input("Gaji (USD)", 15000, 150000, 50000, 1000)

    # --- PARAMETER WARNING DIHAPUS ---
    if st.button("Analisis Audiens"):
        input_df = pd.DataFrame([[1 if gender == "Male" else 0, umur, gaji]], columns=['Gender', 'Age', 'EstimatedSalary'])
        hasil = mesin_ml[model_prediksi].predict(scaler.transform(input_df))
        
        st.markdown("---")
        if hasil[0] == 1:
            st.success(f"**BERPOTENSI BELI** - Menurut metode **{model_prediksi}**, audiens ini layak ditargetkan iklan!")
        else:
            st.error(f"**BERPOTENSI ABAI** - Menurut metode **{model_prediksi}**, audiens ini kemungkinan besar tidak akan membeli.")
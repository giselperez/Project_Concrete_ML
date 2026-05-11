import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# CONFIGURACIÓN
st.set_page_config(
    page_title="Concrete Strength AI",
    page_icon="🏗️",
    layout="wide"
)

# Cargar modelo
with open("models/modelo.pkl", "rb") as f:
    model = pickle.load(f)

# SIDEBAR
st.sidebar.title("📊 Información del Modelo")

st.sidebar.success("Modelo entrenado correctamente")

st.sidebar.metric("R² Score", "0.87")
st.sidebar.metric("RMSE", "5.66")
st.sidebar.metric("Algoritmo", "Random Forest")

st.sidebar.write("---")

st.sidebar.info("""
Este sistema utiliza Machine Learning para predecir la resistencia del concreto en MPa según la composición de la mezcla.
""")

# TÍTULO
st.title("🏗️ Sistema Inteligente de Predicción de Resistencia del Concreto")

st.markdown("""
### Plataforma basada en Machine Learning para estimar resistencia a compresión del concreto
""")

# COLUMNAS
col1, col2 = st.columns(2)

with col1:

    st.subheader("✍️ Ingreso Manual")

    cement = st.number_input("Cemento (kg/m³)", min_value=0.0)
    slag = st.number_input("Escoria (kg/m³)", min_value=0.0)
    fly_ash = st.number_input("Ceniza volante (kg/m³)", min_value=0.0)
    water = st.number_input("Agua (kg/m³)", min_value=0.0)
    superplasticizer = st.number_input("Superplastificante (kg/m³)", min_value=0.0)
    coarse_agg = st.number_input("Agregado grueso (kg/m³)", min_value=0.0)
    fine_agg = st.number_input("Agregado fino (kg/m³)", min_value=0.0)
    age = st.number_input("Edad (días)", min_value=1)

    if st.button("🔍 Predecir Resistencia"):

        data = pd.DataFrame({
            "cement": [cement],
            "slag": [slag],
            "fly_ash": [fly_ash],
            "water": [water],
            "superplasticizer": [superplasticizer],
            "coarse_agg": [coarse_agg],
            "fine_agg": [fine_agg],
            "age": [age]
        })

        pred = model.predict(data)

        st.success(f"✅ Resistencia estimada: {pred[0]:.2f} MPa")

        # Gráfica
        fig, ax = plt.subplots()

        materiales = [
            "Cemento",
            "Escoria",
            "Ceniza",
            "Agua",
            "Superplast.",
            "Ag. grueso",
            "Ag. fino"
        ]

        valores = [
            cement,
            slag,
            fly_ash,
            water,
            superplasticizer,
            coarse_agg,
            fine_agg
        ]

        ax.bar(materiales, valores)

        plt.xticks(rotation=20)

        st.pyplot(fig)

with col2:

    st.subheader("📂 Predicción mediante Excel")

    archivo = st.file_uploader(
        "Sube un archivo Excel",
        type=["xlsx"]
    )

    if archivo is not None:

        df = pd.read_excel(archivo)

        st.write("### Vista previa")
        st.dataframe(df.head())

        try:

            predicciones = model.predict(df)

            df["Predicción MPa"] = predicciones

            st.write("### Resultados")
            st.dataframe(df)

            st.download_button(
                "⬇️ Descargar resultados",
                df.to_csv(index=False),
                "predicciones.csv",
                "text/csv"
            )

        except Exception as e:
            st.error(f"Error en archivo: {e}")

# FOOTER
st.write("---")

st.caption("""
Proyecto desarrollado con Python, Streamlit y Machine Learning.
Modelo predictivo para ingeniería civil y análisis de materiales.
""")
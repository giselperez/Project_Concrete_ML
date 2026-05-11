import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# =========================================
# CONFIGURACIÓN GENERAL
# =========================================

st.set_page_config(
    page_title="Concrete Strength AI",
    page_icon="🏗️",
    layout="wide"
)

# =========================================
# CARGAR MODELO
# =========================================

with open("models/modelo.pkl", "rb") as f:
    model = pickle.load(f)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.title("📊 Información del Modelo")

st.sidebar.success("Modelo cargado correctamente")

st.sidebar.metric("Algoritmo", "Random Forest")
st.sidebar.metric("R² Score", "0.87")
st.sidebar.metric("RMSE", "5.66")

st.sidebar.write("---")

st.sidebar.info("""
Sistema basado en Machine Learning para predecir
la resistencia del concreto según la composición
de la mezcla.
""")

# =========================================
# TÍTULO
# =========================================

st.title("🏗️ Sistema Inteligente de Predicción de Resistencia del Concreto")

st.markdown("""
### Plataforma de Machine Learning aplicada a Ingeniería Civil
Predicción automática de resistencia a compresión del concreto (MPa)
""")

st.write("---")

# =========================================
# COLUMNAS
# =========================================

col1, col2 = st.columns(2)

# =========================================
# COLUMNA 1
# =========================================

with col1:

    st.subheader("✍️ Predicción Manual")

    cement = st.number_input(
        "Cemento (kg/m³)",
        min_value=0.0,
        value=300.0
    )

    slag = st.number_input(
        "Escoria (kg/m³)",
        min_value=0.0,
        value=0.0
    )

    fly_ash = st.number_input(
        "Ceniza volante (kg/m³)",
        min_value=0.0,
        value=0.0
    )

    water = st.number_input(
        "Agua (kg/m³)",
        min_value=0.0,
        value=180.0
    )

    superplasticizer = st.number_input(
        "Superplastificante (kg/m³)",
        min_value=0.0,
        value=0.0
    )

    coarse_agg = st.number_input(
        "Agregado grueso (kg/m³)",
        min_value=0.0,
        value=1000.0
    )

    fine_agg = st.number_input(
        "Agregado fino (kg/m³)",
        min_value=0.0,
        value=700.0
    )

    age = st.number_input(
        "Edad (días)",
        min_value=1,
        value=28
    )

    # =====================================
    # BOTÓN PREDICCIÓN
    # =====================================

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

        resistencia = pred[0]

        # =================================
        # RESULTADO
        # =================================

        st.success(
            f"✅ Resistencia estimada: {resistencia:.2f} MPa"
        )

        # =================================
        # CLASIFICACIÓN
        # =================================

        if resistencia < 20:
            st.error("⚠️ Concreto de baja resistencia")

        elif resistencia < 40:
            st.warning("🟡 Concreto de resistencia media")

        else:
            st.success("🟢 Concreto de alta resistencia")

        # =================================
        # GRÁFICA DE MATERIALES
        # =================================

        st.subheader("📈 Composición de la mezcla")

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

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(materiales, valores)

        ax.set_ylabel("Cantidad kg/m³")

        ax.set_title("Distribución de materiales")

        plt.xticks(rotation=20)

        st.pyplot(fig)

# =========================================
# COLUMNA 2
# =========================================

with col2:

    st.subheader("📂 Predicción mediante Excel")

    st.write("""
    El archivo debe contener estas columnas:

    - cement
    - slag
    - fly_ash
    - water
    - superplasticizer
    - coarse_agg
    - fine_agg
    - age
    """)

    archivo = st.file_uploader(
        "Sube un archivo Excel",
        type=["xlsx"]
    )

    # =====================================
    # SI SUBEN ARCHIVO
    # =====================================

    if archivo is not None:

        try:

            df = pd.read_excel(archivo)

            st.subheader("👀 Vista previa del archivo")

            st.dataframe(df.head())

            # =============================
            # PREDICCIONES
            # =============================

            predicciones = model.predict(df)

            df["Predicción MPa"] = predicciones

            st.subheader("📋 Resultados")

            st.dataframe(df)

            # =============================
            # MÉTRICAS
            # =============================

            st.subheader("📊 Estadísticas")

            met1, met2, met3 = st.columns(3)

            met1.metric(
                "Promedio",
                f"{df['Predicción MPa'].mean():.2f} MPa"
            )

            met2.metric(
                "Máxima",
                f"{df['Predicción MPa'].max():.2f} MPa"
            )

            met3.metric(
                "Mínima",
                f"{df['Predicción MPa'].min():.2f} MPa"
            )

            # =============================
            # HISTOGRAMA
            # =============================

            st.subheader("📈 Distribución de resistencias")

            fig2, ax2 = plt.subplots(figsize=(8, 5))

            ax2.hist(
                df["Predicción MPa"],
                bins=10
            )

            ax2.set_xlabel("Resistencia MPa")

            ax2.set_ylabel("Frecuencia")

            ax2.set_title("Distribución de resistencias")

            st.pyplot(fig2)

            # =============================
            # DESCARGA CSV
            # =============================

            csv = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="⬇️ Descargar resultados CSV",
                data=csv,
                file_name="predicciones_concreto.csv",
                mime="text/csv"
            )

        except Exception as e:

            st.error(f"❌ Error en archivo: {e}")

# =========================================
# FOOTER
# =========================================

st.write("---")

# =========================================
# FOOTER
# =========================================

st.write("---")

st.caption("""
Proyecto desarrollado con Python, Streamlit y Machine Learning.

Aplicación enfocada en analítica predictiva para Ingeniería Civil.
""")
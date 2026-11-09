
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Big Five - Personalidad & Clusters", layout="centered")

st.title("🧠 Clasificador de Personalidad (Big Five) & Clusters")
st.write("Ingresa tus resultados del test OCEAN (Big Five) para descubrir tu clúster y compararte con famosos de tu grupo.")

# 1. Cargar el modelo, escalador y datos
@st.cache_resource
def cargar_recursos():
    kmeans = joblib.load('kmeans_Bigfive_model.joblib')
    scaler = joblib.load('scaler_bigfive.joblib')
    # El dataset de famosos original
    df = pd.read_csv("https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv")
    df["cluster"] = kmeans.labels_ # Asignar las etiquetas del modelo entrenado
    return kmeans, scaler, df

try:
    kmeans, scaler, df = cargar_recursos()
except Exception as e:
    st.error(f"Error cargando los recursos: {e}. Asegúrate de tener los archivos .joblib generados en el entorno.")
    st.stop()

# 2. Entrada de datos de usuario
st.sidebar.header("Tus Puntuaciones (OCEAN)")
op = st.sidebar.slider("Apertura (op)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
co = st.sidebar.slider("Conciencia (co)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
ex = st.sidebar.slider("Extroversión (ex)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
ag = st.sidebar.slider("Amabilidad (ag)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
ne = st.sidebar.slider("Neuroticismo (ne)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)

# 3. Predicción del Cluster
mis_datos = pd.DataFrame([[op, co, ex, ag, ne]], columns=['op', 'co', 'ex', 'ag', 'ne'])
mis_datos_scaled = scaler.transform(mis_datos)
mi_cluster = kmeans.predict(mis_datos_scaled)[0]

# 4. Mostrar Resultados del Cluster
st.subheader(f"🎯 Cluster Asignado: **Cluster {mi_cluster}**")

# Obtener famosos del clúster
famosos_cluster = df[df['cluster'] == mi_cluster]

st.write(f"Famosos que comparten tu mismo clúster ({len(famosos_cluster)} en total):")
st.dataframe(famosos_cluster[['usuario', 'op', 'co', 'ex', 'ag', 'ne']].head(5))

# 5. Obtener promedios del clúster asignado
features = ['op', 'co', 'ex', 'ag', 'ne']
promedio_cluster = famosos_cluster[features].mean().values.tolist()
mis_valores = [op, co, ex, ag, ne]

# Cerrar el círculo para el gráfico de radar de Plotly
features_radar = features + [features[0]]
promedio_cluster_radar = promedio_cluster + [promedio_cluster[0]]
mis_valores_radar = mis_valores + [mis_valores[0]]

# 6. Crear el Gráfico de Radar Interactivo con Plotly
fig = go.Figure()

# Trazo de los valores promedio del Cluster
fig.add_trace(go.Scatterpolar(
    r=promedio_cluster_radar,
    theta=features_radar,
    fill='toself',
    name=f'Promedio Cluster {mi_cluster}',
    line_color='#1f77b4'
))

# Trazo de los valores del usuario
fig.add_trace(go.Scatterpolar(
    r=mis_valores_radar,
    theta=features_radar,
    fill='toself',
    name='Mis Resultados',
    line_color='#ff7f0e'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 100]
        )
    ),
    showlegend=True,
    title=f"Comparativa de Perfil: Mis Resultados vs. Promedio Cluster {mi_cluster}"
)

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 1. IMPORTACIONES NECESARIAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score

# Configuración de la página Streamlit
st.set_page_config(layout="wide")
st.title("🚨 SICOPS: Tablero Predictivo de Seguridad (Versión Integrada)")
st.markdown("Modelo de Decision Tree entrenado y visualización de riesgo en tiempo real.")

# ==============================================================================
# 2. MODELO DE MACHINE LEARNING Y PIPELINE DE DATOS (OE1)
#    Usamos st.cache_resource para entrenar el modelo UNA SOLA VEZ
# ==============================================================================

@st.cache_resource
def entrenar_modelo_y_obtener_features():
    """
    Simula el pipeline completo de ML: ingestión, ingeniería de features y entrenamiento.
    """
    st.info("Iniciando entrenamiento del modelo Decision Tree... (Solo se ejecuta la primera vez)")
    
    # --- SIMULACIÓN DE DATOS HISTÓRICOS (INCLUYE COORDENADAS) ---
    N_REGISTROS = 50000 
    
    # 1. Simulación de la Variable Objetivo (Y) - 10% de riesgo (desbalanceo)
    np.random.seed(42) # Fija la semilla para reproducibilidad
    y_historico = np.random.choice([0, 1], size=N_REGISTROS, p=[0.90, 0.10])
    
    # 2. Simulación de Features (X) - Simula 15 features
    data = {
        'latitud': 7.1132 + (np.random.rand(N_REGISTROS) - 0.5) * 0.15,
        'longitud': -73.1190 + (np.random.rand(N_REGISTROS) - 0.5) * 0.15,
        'franja_horaria': np.random.randint(0, 24, N_REGISTROS),
        'historico_7d': np.random.poisson(lam=2, size=N_REGISTROS), 
        'kde_vecino_1km': np.random.rand(N_REGISTROS) * 10,       
    }
    # Completar con 10 features adicionales
    for i in range(1, 11):
        data[f'feature_{i}'] = np.random.rand(N_REGISTROS)
        
    df_historico = pd.DataFrame(data)
    
    # 3. Preparación para Scikit-Learn
    # Definimos las features (X) que se usaron para entrenar y se usarán para predecir
    nombres_features = [col for col in df_historico.columns if col not in ['latitud', 'longitud', 'franja_horaria']]
    X = df_historico[nombres_features]
    
    # 4. División y Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y_historico, test_size=0.2, random_state=42)
    
    # Configuración del modelo (basado en Untitled.ipynb)
    modelo = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
    modelo.fit(X_train, y_train)
    
    # 5. Evaluación 
    pred_test = modelo.predict(X_test)
    f1 = f1_score(y_test, pred_test, average='weighted')
    
    st.success(f"Modelo Decision Tree entrenado. F1-Score (Test): {f1:.4f}")
    
    return modelo, nombres_features, f1

# Ejecutar el pipeline de entrenamiento
modelo_entrenado, nombres_features, f1_score_modelo = entrenar_modelo_y_obtener_features()

# ==============================================================================
# 3. FUNCIÓN DE PREDICCIÓN DIARIA (Simula la ejecución del servidor)
#    Usamos st.cache_data para no regenerar el mapa en cada interacción simple
# ==============================================================================

@st.cache_data
def generar_datos_prediccion(_modelo, nombres_features): # CORRECCIÓN: Usar '_modelo'
    """Genera los datos de predicción para el Tablero."""
    
    # --- SIMULACIÓN DE DATOS FUTUROS (Celda y Franja Horaria) ---
    N_CELDAS_PREDICCION = 15000 
    
    df_future = pd.DataFrame({
        'latitud': 7.1132 + (np.random.rand(N_CELDAS_PREDICCION) - 0.5) * 0.1,
        'longitud': -73.1190 + (np.random.rand(N_CELDAS_PREDICCION) - 0.5) * 0.1,
        'modalidad': np.random.choice(['Hurto a Persona', 'Hurto de Vehículo', 'Lesiones', 'Otro'], N_CELDAS_PREDICCION),
        'franja_horaria': np.random.choice(['00:00-06:00', '06:00-12:00', '12:00-18:00', '18:00-00:00'], N_CELDAS_PREDICCION)
    })
    
    # --- SIMULACIÓN DE FEATURES FUTURAS (X_future) ---
    
    # Crear un DataFrame de Features Futuras con la misma estructura (mismas 15 columnas)
    X_future_data = np.random.rand(N_CELDAS_PREDICCION, len(nombres_features))
    X_future = pd.DataFrame(X_future_data, columns=nombres_features)
    
    # --- PREDICCIÓN ---
    
    # El modelo predice la probabilidad de la clase 1 (Riesgo Alto)
    probabilidades = _modelo.predict_proba(X_future)[:, 1] # CORRECCIÓN: Uso de '_modelo'
    
    # Añadir los resultados de la predicción al DataFrame de coordenadas
    df_future['probabilidad_riesgo'] = probabilidades
    
    # Convertir a intensidad para visualización en el Heatmap
    df_future['intensidad_riesgo'] = df_future['probabilidad_riesgo'] * 100
    
    return df_future

# Generar los datos que se van a visualizar
# CORRECCIÓN: Llamada con el keyword argument
df = generar_datos_prediccion(_modelo=modelo_entrenado, nombres_features=nombres_features)


# ==============================================================================
# 4. CONSTRUCCIÓN DEL TABLERO WEB STREAMLIT (OE2)
# ==============================================================================

# --- 4.1 BARRA LATERAL Y FILTROS ---
st.sidebar.header("🔍 Filtros de Predicción")

franja_seleccionada = st.sidebar.selectbox(
    "1. Selecciona Franja Horaria:",
    options=df['franja_horaria'].unique()
)

modalidades_seleccionadas = st.sidebar.multiselect(
    "2. Filtrar por Modalidad de Riesgo:",
    options=df['modalidad'].unique(),
    default=df['modalidad'].unique() 
)

# Aplicar filtros
df_filtrado = df[
    (df['franja_horaria'] == franja_seleccionada) &
    (df['modalidad'].isin(modalidades_seleccionadas))
]


# --- 4.2 MÉTRICAS CLAVE (KPIs) ---
st.header("📊 Indicadores de Riesgo")
col1, col2, col3 = st.columns(3)

riesgo_critico = df_filtrado[df_filtrado['probabilidad_riesgo'] > 0.8].shape[0]
total_celdas = df_filtrado.shape[0]

with col1:
    st.metric(
        label="Celdas en Riesgo Crítico (> 80% Prob.)",
        value=f"{riesgo_critico} de {total_celdas}"
    )

with col2:
    st.metric(
        label="Riesgo Promedio General",
        value=f"{df_filtrado['probabilidad_riesgo'].mean() * 100:.2f}%",
    )

with col3:
    st.metric(
        label="F1-Score del Modelo (OE1)",
        value=f"{f1_score_modelo:.4f}",
        delta="Modelo Decision Tree (Línea Base)"
    )


# --- 4.3 MAPA DE CALOR INTERACTIVO ---
st.header(f"🗺️ Mapa de Riesgo para la Franja: {franja_seleccionada}")

if df_filtrado.empty:
    st.warning("No hay datos de predicción para los filtros seleccionados.")
else:
    # Coordenadas iniciales (Centro de Santander/Bucaramanga simulado)
    view_state = pdk.ViewState(
        latitude=df_filtrado['latitud'].mean(),
        longitude=df_filtrado['longitud'].mean(),
        zoom=11,
        pitch=0,
    )

    # Definición de la capa de Heatmap
    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_filtrado,
        opacity=0.9,
        threshold=0.3,
        get_position=['longitud', 'latitud'],
        get_weight='intensidad_riesgo', # Usa el campo calculado (0-100) como peso
    )

    # Renderizar el mapa
    r = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v9' 
    )

    st.pydeck_chart(r)

# ==============================================================================
# 5. INTEGRACIÓN CON EL CHATBOT (OE3 - Placeholder)
# ==============================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Chatbot Comunitario (OE3)")
st.sidebar.info("El chatbot se integraría aquí como un widget o enlace, proporcionando información local y preventiva.")
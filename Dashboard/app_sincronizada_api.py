# ==============================================================================
# 1. IMPORTACIONES NECESARIAS
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import requests 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import pydeck as pdk
import geopandas as gpd
from shapely.geometry import Point, box
import json

# Configuración de la página Streamlit
st.set_page_config(layout="wide")
st.title("🚨 SICOPS: Tablero Predictivo de Seguridad (Consumo de APIs)")
st.markdown("Modelo Decision Tree entrenado con datos en tiempo real (vía API).")

# ==============================================================================
# 2. CONFIGURACIÓN DE LAS APIS Y CARGA DE DATOS
# ==============================================================================

# Máximo de registros a cargar para asegurar un funcionamiento rápido en Streamlit
MAX_REGISTROS = 50000 
BASE_URL = "https://www.datos.gov.co/resource/"

# IDs de recurso proporcionados por el usuario
RECURSO_SEXUAL_ID = "fpe5-yrmw" 
RECURSO_VIOLENCIA_ID = "vuyt-mqpw"
# ID CORREGIDO: d4fr-sbn2
RECURSO_HURTO_ID = "d4fr-sbn2" 

def cargar_datos_desde_api(recurso_id):
    """Llama a la API de Socrata y devuelve un DataFrame de Pandas."""
    # Se limita la carga a MAX_REGISTROS para evitar tiempos de espera excesivos
    url = f"{BASE_URL}{recurso_id}.json?$limit={MAX_REGISTROS}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Lanza un error para códigos de estado HTTP 4xx/5xx
        data = response.json()
        return pd.DataFrame(data)
    except requests.exceptions.RequestException as e:
        st.error(f"Error al cargar datos de la API {recurso_id}: {e}. Verifica tu conexión o el ID del recurso.")
        return pd.DataFrame()

def normalizar_dep_mun(df):
    """Aplica la normalización de DEPARTAMENTO y MUNICIPIO del notebook."""
    # Los nombres de columna de la API suelen ser en minúsculas
    for col in ["departamento", "municipio"]: 
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.upper()
                .str.replace(r"\s+\(CT\)$", "", regex=True)
            )
    return df

@st.cache_resource
def cargar_y_preprocesar_datos_api():
    """Carga los datos de las APIs y aplica la limpieza inicial."""
    st.info("Conectando y cargando datos desde las APIs de Policía Nacional...")
    
    # --- Carga de datos desde las APIs ---
    delitos_sex = cargar_datos_desde_api(RECURSO_SEXUAL_ID)
    hurto = cargar_datos_desde_api(RECURSO_HURTO_ID)
    violencia = cargar_datos_desde_api(RECURSO_VIOLENCIA_ID)
    
    if delitos_sex.empty or hurto.empty or violencia.empty:
        st.error("Una o más cargas de API fallaron o no devolvieron datos. El script se detiene.")
        st.stop() 

    # --- Normalización de texto (Fiel al Untitled.ipynb) ---
    delitos_sex = normalizar_dep_mun(delitos_sex)
    hurto = normalizar_dep_mun(hurto)
    violencia = normalizar_dep_mun(violencia)
    
    # --- Parseo de fechas (Fiel al Untitled.ipynb) ---
    # Asume que la columna de fecha de la API es 'fecha_hecho'
    for df in [delitos_sex, hurto, violencia]:
        if "fecha_hecho" in df.columns:
            df["fecha_hecho"] = pd.to_datetime(
                df["fecha_hecho"],
                dayfirst=True,
                errors="coerce" 
            )
        
    st.success("Datos cargados por API y limpieza inicial completada.")
    
    return delitos_sex, hurto, violencia

delitos_sex_clean, hurto_clean, violencia_clean = cargar_y_preprocesar_datos_api()


def combinar_delitos(delitos_sex, hurto, violencia):
    """Combina los 3 DataFrames para un análisis de riesgo unificado."""
    
    # Definición de la modalidad de delito para cada dataset
    delitos_sex['modalidad_delito'] = 'DELITO SEXUAL'
    # La columna de modalidad en la API de Hurto es "modalidad_del_hecho"
    hurto['modalidad_delito'] = hurto.get('modalidad_del_hecho', 'HURTO (sin detalle)') 
    violencia['modalidad_delito'] = 'VIOLENCIA INTRAFAMILIAR'
    
    # Columnas comunes para la concatenación
    cols_base = ['departamento', 'municipio', 'fecha_hecho', 'modalidad_delito']
    
    # Filtrar columnas existentes antes de concatenar
    df_combined = pd.concat([
        delitos_sex[[c for c in cols_base if c in delitos_sex.columns]], 
        hurto[[c for c in cols_base if c in hurto.columns]], 
        violencia[[c for c in cols_base if c in violencia.columns]]
    ], ignore_index=True)
    
    df_combined.dropna(subset=['fecha_hecho'], inplace=True)
    
    return df_combined


# ==============================================================================
# 3. ENTRENAMIENTO DEL MODELO (OE1) - Mantiene la estructura y el Warning
# ==============================================================================

@st.cache_resource
def entrenar_modelo_y_obtener_features(delitos_sex_clean, hurto_clean, violencia_clean):
    
    # 1. COMBINACIÓN DE DATOS
    df_combined = combinar_delitos(delitos_sex_clean, hurto_clean, violencia_clean)
    st.success(f"Datos combinados: {df_combined.shape[0]} incidentes listos para Feature Engineering.")


    st.markdown("""
        <div style="background-color: #ffcccc; padding: 15px; border-radius: 5px; border: 2px solid #cc0000;">
            <h3>⚠️ PASO PENDIENTE: GEOCODIFICACIÓN y FEATURE ENGINEERING REAL</h3>
            <p><strong>El modelo predice en una cuadrícula de 200m x 200m, pero los datos de la API solo tienen el municipio.</strong></p>
            <p><strong>DEBES REEMPLAZAR EL CÓDIGO DE SIMULACIÓN ABAJO</strong> con tu lógica de:</p>
            <ol>
                <li>Geocodificación (convertir Municipio a Lat/Lon).</li>
                <li>Agregación a Cuadrícula (crear la unidad de análisis de 200m).</li>
                <li>Cálculo de las 15 Features (KDE, Recencia, etc.).</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    # --- INICIO DE SIMULACIÓN ESTRUCTURAL (DEBES REEMPLAZAR DESDE AQUÍ) ---
    st.warning("Usando una SIMULACIÓN ESTRUCTURAL DE COORDENADAS/FEATURES para hacer el código ejecutable.")

    N_REGISTROS = df_combined.shape[0] 
    
    # Simulación del resultado del Feature Engineering
    np.random.seed(42)
    df_combined['latitud'] = 7.1132 + (np.random.rand(N_REGISTROS) - 0.5) * 0.15
    df_combined['longitud'] = -73.1190 + (np.random.rand(N_REGISTROS) - 0.5) * 0.15
    df_combined['target_riesgo'] = np.random.choice([0, 1], size=N_REGISTROS, p=[0.90, 0.10])
    
    nombres_features = []
    for i in range(1, 15):
        feature_name = f'feature_{i}'
        df_combined[feature_name] = np.random.rand(N_REGISTROS)
        nombres_features.append(feature_name)
    
    df_combined['franja_horaria_cod'] = np.random.randint(0, 24, N_REGISTROS)
    nombres_features.append('franja_horaria_cod')
    
    X = df_combined[nombres_features]
    y_historico = df_combined['target_riesgo']
    
    # --- FIN DE SIMULACIÓN ESTRUCTURAL ---
    
    # 4. División y Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y_historico, test_size=0.2, random_state=42)
    
    st.info("Entrenando modelo con los parámetros fieles de tu `Untitled.ipynb`...")
    
    # Parámetros del modelo: ¡Decision Tree FIEL AL NOTEBOOK!
    modelo = DecisionTreeClassifier(
        max_depth=8, 
        min_samples_leaf=10, 
        random_state=42
    )
    modelo.fit(X_train, y_train)
    
    # 5. Evaluación 
    pred_test = modelo.predict(X_test)
    f1 = f1_score(y_test, pred_test, average='weighted')
    
    st.success(f"Modelo Decision Tree entrenado. F1-Score (Test): {f1:.4f}")
    
    return modelo, nombres_features, f1

modelo_entrenado, nombres_features, f1_score_modelo = entrenar_modelo_y_obtener_features(delitos_sex_clean, hurto_clean, violencia_clean)


# ==============================================================================
# 4. FUNCIÓN DE PREDICCIÓN Y TABLERO (OE2)
# ==============================================================================

@st.cache_data
def generar_datos_prediccion(_modelo, nombres_features):
    """Genera los datos de predicción para el Tablero (simulando la cuadrícula futura)."""
    
    N_CELDAS_PREDICCION = 15000 
    
    df_future = pd.DataFrame({
        'latitud': 7.1132 + (np.random.rand(N_CELDAS_PREDICCION) - 0.5) * 0.1,
        'longitud': -73.1190 + (np.random.rand(N_CELDAS_PREDICCION) - 0.5) * 0.1,
        'modalidad': np.random.choice(['Hurto a Persona', 'Hurto de Vehículo', 'Lesiones', 'Otro'], N_CELDAS_PREDICCION),
        'franja_horaria': np.random.choice(['00:00-06:00', '06:00-12:00', '12:00-18:00', '18:00-00:00'], N_CELDAS_PREDICCION)
    })
    
    # Simulación de Features Futuras (X_future)
    X_future_data = np.random.rand(N_CELDAS_PREDICCION, len(nombres_features))
    X_future = pd.DataFrame(X_future_data, columns=nombres_features)
    
    # PREDICCIÓN
    probabilidades = _modelo.predict_proba(X_future)[:, 1] 
    
    df_future['probabilidad_riesgo'] = probabilidades
    df_future['intensidad_riesgo'] = df_future['probabilidad_riesgo'] * 100
    
    return df_future

df = generar_datos_prediccion(_modelo=modelo_entrenado, nombres_features=nombres_features)


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
        delta="Decision Tree (Parámetros Fieles)"
    )


# --- 4.3 MAPA DE CALOR INTERACTIVO ---
st.header(f"🗺️ Mapa de Riesgo para la Franja: {franja_seleccionada}")


if df_filtrado.empty:
    st.warning("No hay datos de predicción para los filtros seleccionados.")
else:
    view_state = pdk.ViewState(
        latitude=df_filtrado['latitud'].mean(),
        longitude=df_filtrado['longitud'].mean(),
        zoom=11,
        pitch=0,
    )

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=df_filtrado,
        opacity=0.9,
        threshold=0.3,
        get_position=['longitud', 'latitud'],
        get_weight='intensidad_riesgo', 
    )

    r = pdk.Deck(
        layers=[heatmap_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v9' 
    )

    st.pydeck_chart(r)

st.sidebar.markdown("---")
st.sidebar.subheader("💬 Chatbot Comunitario (OE3)")
st.sidebar.info("El chatbot se integraría aquí como un widget o enlace, proporcionando información local y preventiva.")

# ==============================================================================
# 5. GEOCODIFICACIÓN REAL Y AGREGACIÓN A CUADRÍCULAS
# ==============================================================================

def cargar_geojson_municipios(file_path):
    """Carga el archivo GeoJSON y devuelve un GeoDataFrame."""
    try:
        gdf = gpd.read_file(file_path)
        return gdf
    except Exception as e:
        st.error(f"Error al cargar el archivo GeoJSON: {e}")
        return gpd.GeoDataFrame()

@st.cache_resource
def geocodificar_municipios(df, _gdf_municipios):
    """Asocia coordenadas geográficas a los municipios en el DataFrame."""
    municipios_coords = {
        row['MPIO_CNMBR']: (row['LONGITUD'], row['LATITUD'])
        for _, row in _gdf_municipios.iterrows()
    }

    df['latitud'] = df['municipio'].map(lambda x: municipios_coords.get(x, (None, None))[1])
    df['longitud'] = df['municipio'].map(lambda x: municipios_coords.get(x, (None, None))[0])

    return df.dropna(subset=['latitud', 'longitud'])

@st.cache_resource
def agregar_a_cuadriculas(df, tam_celda=0.002):
    """Divide el área en cuadrículas y asigna cada incidente a una celda."""
    # Crear cuadrículas
    minx, miny, maxx, maxy = df['longitud'].min(), df['latitud'].min(), df['longitud'].max(), df['latitud'].max()
    grid = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            grid.append(box(x, y, x + tam_celda, y + tam_celda))
            y += tam_celda
        x += tam_celda

    grid_gdf = gpd.GeoDataFrame({'geometry': grid})

    # Asignar incidentes a cuadrículas
    incidentes_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitud'], df['latitud']))
    incidentes_gdf = gpd.sjoin(incidentes_gdf, grid_gdf, how='left', predicate='within')

    return incidentes_gdf

@st.cache_resource
def calcular_features(_df):
    """Calcula características avanzadas para el modelo."""
    # Ejemplo de cálculo de densidad de incidentes
    _df['densidad_incidentes'] = _df.groupby('geometry')['geometry'].transform('count')

    # Recencia de eventos
    _df['recencia'] = (pd.Timestamp.now() - _df['fecha_hecho']).dt.days

    # Verificar si la columna 'franja_horaria_cod' existe
    if 'franja_horaria_cod' not in _df.columns:
        _df['franja_horaria_cod'] = 0  # Inicializar con valores predeterminados

    # Distribución por franja horaria
    _df['franja_horaria_cod'] = pd.cut(_df['franja_horaria_cod'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3])

    return _df

# Cargar GeoJSON
geojson_path = "d:\\Mintic\\santander_municipios.geojson"
gdf_municipios = cargar_geojson_municipios(geojson_path)

# Geocodificar municipios
if not gdf_municipios.empty:
    delitos_sex_clean = geocodificar_municipios(delitos_sex_clean, gdf_municipios)
    hurto_clean = geocodificar_municipios(hurto_clean, gdf_municipios)
    violencia_clean = geocodificar_municipios(violencia_clean, gdf_municipios)

    # Agregar a cuadrículas
    delitos_sex_clean = agregar_a_cuadriculas(delitos_sex_clean)
    hurto_clean = agregar_a_cuadriculas(hurto_clean)
    violencia_clean = agregar_a_cuadriculas(violencia_clean)

    # Calcular features avanzadas
    delitos_sex_clean = calcular_features(delitos_sex_clean)
    hurto_clean = calcular_features(hurto_clean)
    violencia_clean = calcular_features(violencia_clean)

    st.success("Geocodificación, agregación a cuadrículas y cálculo de features completados.")

def agregar_franja_horaria(df):
    """Agrega la columna 'franja_horaria_cod' basada en la hora de 'fecha_hecho'."""
    if 'fecha_hecho' in df.columns:
        df['hora'] = df['fecha_hecho'].dt.hour
        df['franja_horaria_cod'] = pd.cut(
            df['hora'], bins=[0, 6, 12, 18, 24], labels=[0, 1, 2, 3], right=False
        )
        df.drop(columns=['hora'], inplace=True)
    return df

# Aplicar la función a los datasets
if not gdf_municipios.empty:
    delitos_sex_clean = agregar_franja_horaria(delitos_sex_clean)
    hurto_clean = agregar_franja_horaria(hurto_clean)
    violencia_clean = agregar_franja_horaria(violencia_clean)

# Verificar que la columna se haya agregado correctamente
st.write("Delitos sexuales - franja horaria:")
st.write(delitos_sex_clean[['fecha_hecho', 'franja_horaria_cod']].head())

st.write("Hurto - franja horaria:")
st.write(hurto_clean[['fecha_hecho', 'franja_horaria_cod']].head())

st.write("Violencia intrafamiliar - franja horaria:")
st.write(violencia_clean[['fecha_hecho', 'franja_horaria_cod']].head())

# ==============================================================================
# 6. MAPA DE SANTANDER CON MUNICIPIOS Y MAPA DE CALOR
# ==============================================================================

# Cargar datos geográficos de Santander
santander_geojson = "santander_municipios.geojson"
with open(santander_geojson, 'r') as f:
    santander_data = json.load(f)

# Crear capa de municipios
municipios_layer = pdk.Layer(
    "GeoJsonLayer",
    santander_data,
    pickable=True,
    stroked=True,
    filled=True,
    extruded=False,
    line_width_min_pixels=1,
    get_fill_color="[200, 100, 240, 200]",
    get_line_color="[0, 0, 0, 255]",
)

# Crear capa de mapa de calor
heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=delitos_sex_clean,
    get_position="[longitud, latitud]",
    get_weight="frecuencia",
    radius=100,
)

# Configurar vista inicial del mapa
view_state = pdk.ViewState(
    latitude=7.125,
    longitude=-73.1198,
    zoom=8,
    pitch=50,
)

# Renderizar el mapa
mapa_santander = pdk.Deck(
    layers=[municipios_layer, heatmap_layer],
    initial_view_state=view_state,
    tooltip={"text": "Municipio: {name}"},
)

st.pydeck_chart(mapa_santander)

# --- Corrección para usar datos reales en el mapa de calor ---

# Verificar que los datos de las APIs estén cargados y geocodificados
if not delitos_sex_clean.empty and not hurto_clean.empty and not violencia_clean.empty:
    # Combinar los datos geocodificados
    df_combined_real = pd.concat([delitos_sex_clean, hurto_clean, violencia_clean], ignore_index=True)

    # Configurar vista inicial del mapa
    view_state = pdk.ViewState(
        latitude=df_combined_real['latitud'].mean(),
        longitude=df_combined_real['longitud'].mean(),
        zoom=8,
        pitch=50,
    )

    # Crear capa de mapa de calor con datos reales
    heatmap_layer_real = pdk.Layer(
        "HeatmapLayer",
        data=df_combined_real,
        get_position=['longitud', 'latitud'],
        get_weight='densidad_incidentes',  # Usar densidad calculada como peso
        radius=100,
    )

    # Crear capa de municipios
    municipios_layer_real = pdk.Layer(
        "GeoJsonLayer",
        santander_data,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=False,
        line_width_min_pixels=1,
        get_fill_color="[200, 100, 240, 200]",
        get_line_color="[0, 0, 0, 255]",
    )

    # Renderizar el mapa con datos reales
    mapa_santander_real = pdk.Deck(
        layers=[municipios_layer_real, heatmap_layer_real],
        initial_view_state=view_state,
        tooltip={"text": "Municipio: {municipio}"},
    )

    st.pydeck_chart(mapa_santander_real)
else:
    st.error("No se pueden visualizar los datos reales porque las APIs no devolvieron datos válidos o no se completó la geocodificación.")
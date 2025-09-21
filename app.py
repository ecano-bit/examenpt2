import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
from pathlib import Path

# Agregar el directorio padre al path para importar módulos
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))

# Configuración minimalista
st.set_page_config(
    page_title="Predictor de Gasolina",
    layout="centered"
)

# CSS minimalista
st.markdown("""
<style>
.main {
    background-color: #ffffff;
    font-family: 'Helvetica', sans-serif;
}
.stButton > button {
    background-color: #000000;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 0.5rem 1rem;
}
.stSelectbox > div > div {
    border: 1px solid #cccccc;
}
h1 {
    font-weight: 300;
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

# Título minimalista
st.title("Predictor de Precios de Gasolina")

# Descripción simple
st.write("Modelo de predicción basado en estado, mes y año.")

@st.cache_data
def cargar_datos():
    try:
        archivo_datos = parent_dir / "Gasolina_expandido.csv"
        if archivo_datos.exists():
            df = pd.read_csv(archivo_datos)
            return df
        else:
            st.error(f"Archivo no encontrado: {archivo_datos}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

@st.cache_resource
def cargar_modelo():
    try:
        archivo_modelo = parent_dir / "modelo_gasolina.pkl"
        if archivo_modelo.exists():
            with open(archivo_modelo, 'rb') as f:
                modelo_data = pickle.load(f)
            return modelo_data
        else:
            st.error(f"Modelo no encontrado: {archivo_modelo}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def predecir_precio(modelo_data, entidad, mes, año):
    try:
        modelo = modelo_data['modelo']
        encoder_entidad = modelo_data['encoder_entidad']
        encoder_mes = modelo_data['encoder_mes']
        
        if entidad not in encoder_entidad.classes_:
            return None, f"Entidad '{entidad}' no válida"
        
        if mes not in encoder_mes.classes_:
            return None, f"Mes '{mes}' no válido"
        
        entidad_encoded = encoder_entidad.transform([entidad])[0]
        mes_encoded = encoder_mes.transform([mes])[0]
        
        X_pred = np.array([[entidad_encoded, mes_encoded, año]])
        precio_predicho = modelo.predict(X_pred)[0]
        
        return precio_predicho, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

# Cargar datos y modelo
df_gasolina = cargar_datos()
modelo_data = cargar_modelo()

if df_gasolina is not None and modelo_data is not None:
    
    # Controles de entrada
    entidades = sorted(df_gasolina['Entidad'].unique())
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    años = sorted(df_gasolina['Año'].unique())
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entidad_seleccionada = st.selectbox(
            "Estado",
            entidades,
            index=entidades.index('Nacional') if 'Nacional' in entidades else 0
        )
    
    with col2:
        mes_seleccionado = st.selectbox(
            "Mes",
            meses
        )
    
    with col3:
        año_seleccionado = st.selectbox(
            "Año",
            list(range(min(años), max(años) + 5)),
            index=len(años) - 1 if años else 0
        )
    
    # Botón de predicción
    if st.button("Predecir"):
        precio_predicho, error = predecir_precio(
            modelo_data, entidad_seleccionada, mes_seleccionado, año_seleccionado
        )
        
        if error:
            st.error(error)
        else:
            # Resultado
            st.markdown("---")
            st.subheader("Resultado")
            st.metric("Precio predicho", f"${precio_predicho:.2f} MXN")
            
            # Información adicional
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Estado:** {entidad_seleccionada}")
                st.write(f"**Período:** {mes_seleccionado} {año_seleccionado}")
            
            with col2:
                # Comparación histórica
                precio_historico = df_gasolina[
                    (df_gasolina['Entidad'] == entidad_seleccionada) &
                    (df_gasolina['Mes'] == mes_seleccionado)
                ]['Precio'].mean()
                
                if not pd.isna(precio_historico):
                    diferencia = precio_predicho - precio_historico
                    st.write(f"**Promedio histórico:** ${precio_historico:.2f}")
                    st.write(f"**Diferencia:** {diferencia:+.2f}")
    
    # Datos históricos
    st.markdown("---")
    st.subheader("Datos históricos")
    
    datos_entidad = df_gasolina[df_gasolina['Entidad'] == entidad_seleccionada].copy()
    
    if not datos_entidad.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            precio_min = datos_entidad['Precio'].min()
            precio_max = datos_entidad['Precio'].max()
            precio_promedio = datos_entidad['Precio'].mean()
            
            st.write(f"**Mínimo:** ${precio_min:.2f}")
            st.write(f"**Máximo:** ${precio_max:.2f}")
            st.write(f"**Promedio:** ${precio_promedio:.2f}")
            st.write(f"**Registros:** {len(datos_entidad)}")
        
        with col2:
            datos_grafico = datos_entidad.groupby(['Año', 'Mes'])['Precio'].mean().reset_index()
            datos_grafico['Fecha'] = datos_grafico['Año'].astype(str) + '-' + datos_grafico['Mes']
            
            st.line_chart(
                data=datos_grafico.set_index('Fecha')['Precio'],
                height=200
            )
    
    # Métricas del modelo
    st.markdown("---")
    st.subheader("Métricas del modelo")
    
    col1, col2, col3 = st.columns(3)
    
    metricas = modelo_data.get('metricas', {})
    
    with col1:
        r2 = metricas.get('r2_test', 0)
        st.metric("R² Score", f"{r2:.3f}")
    
    with col2:
        rmse = metricas.get('rmse_test', 0)
        st.metric("RMSE", f"{rmse:.3f}")
    
    with col3:
        mae = metricas.get('mae_test', 0)
        st.metric("MAE", f"{mae:.3f}")
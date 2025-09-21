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

# Configuración básica
st.set_page_config(
    page_title="App Gasolina",
    page_icon="🚗",
    layout="centered"
)

# CSS 
st.markdown("""
<style>
.main {
    background-color: #f0f8ff;
}
.stButton > button {
    background-color: #ff6b6b;
    color: white;
    border: 3px solid #000;
    border-radius: 20px;
    font-weight: bold;
}
.stSelectbox > div > div {
    background-color: #ffffcc;
    border: 2px dashed #333;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center; color: #ff1493; font-family: Comic Sans MS; 
text-shadow: 2px 2px 4px #000000; background-color: #ffff00; 
padding: 10px; border: 5px solid #ff0000; border-radius: 15px;'>
Analísis predictivo del precio de la gasolina a través de los años 
</h1>
""", unsafe_allow_html=True)

st.balloons()

# Descripción amateur
st.markdown("""
<div style='background-color: #98fb98; padding: 15px; border: 3px dotted #8b4513; 
border-radius: 10px; font-family: Arial; margin: 20px 0;'>
<h3 style='color: #4b0082;'>🔮 ¿Qué hace esta app?</h3>
<p style='font-size: 16px; color: #000080;'>
Esta aplicación INCREÍBLE usa inteligencia artificial para adivinar cuánto va a costar la gasolina!!! 🤖✨
</p>
<ul style='color: #8b0000;'>
<li>Escoge tu estado</li>
<li>Elige el mes</li>
<li>Selecciona el año</li>
</ul>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def cargar_datos():
    """Carga los datos expandidos de gasolina"""
    try:
        # Intentar cargar desde el directorio padre
        archivo_datos = parent_dir / "Gasolina_expandido.csv"
        if archivo_datos.exists():
            df = pd.read_csv(archivo_datos)
            return df
        else:
            st.error(f"No se encontró el archivo de datos en: {archivo_datos}")
            return None
    except Exception as e:
        st.error(f"Error al cargar los datos: {str(e)}")
        return None

@st.cache_resource
def cargar_modelo():
    """Carga el modelo entrenado y los encoders"""
    try:
        # Intentar cargar desde el directorio padre
        archivo_modelo = parent_dir / "modelo_gasolina.pkl"
        if archivo_modelo.exists():
            with open(archivo_modelo, 'rb') as f:
                modelo_data = pickle.load(f)
            return modelo_data
        else:
            st.error(f"No se encontró el archivo del modelo en: {archivo_modelo}")
            return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

def predecir_precio(modelo_data, entidad, mes, año):
    """Realiza la predicción del precio"""
    try:
        modelo = modelo_data['modelo']
        encoder_entidad = modelo_data['encoder_entidad']
        encoder_mes = modelo_data['encoder_mes']
        
        # Verificar que la entidad existe
        if entidad not in encoder_entidad.classes_:
            return None, f"Error: La entidad '{entidad}' no existe en el dataset"
        
        # Verificar que el mes existe
        if mes not in encoder_mes.classes_:
            return None, f"Error: El mes '{mes}' no existe en el dataset"
        
        # Codificar las variables
        entidad_encoded = encoder_entidad.transform([entidad])[0]
        mes_encoded = encoder_mes.transform([mes])[0]
        
        # Crear el vector de características
        X_pred = np.array([[entidad_encoded, mes_encoded, año]])
        
        # Realizar la predicción
        precio_predicho = modelo.predict(X_pred)[0]
        
        return precio_predicho, None
        
    except Exception as e:
        return None, f"Error en la predicción: {str(e)}"

# Cargar datos y modelo
df_gasolina = cargar_datos()
modelo_data = cargar_modelo()

if df_gasolina is not None and modelo_data is not None:
    
    st.markdown("""
    <div style='background-color: #ffd700; padding: 10px; border: 4px solid #ff4500; 
    border-radius: 25px; margin: 20px 0;'>
    <h2 style='text-align: center; color: #8b0000; font-family: Impact;'>
    CONTROLES DE LA MÁQUINA PREDICTORA 
    </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Obtener opciones únicas
    entidades = sorted(df_gasolina['Entidad'].unique())
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    años = sorted(df_gasolina['Año'].unique())
    
    # Crear columnas para los controles
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4 style='color: #ff6347; text-align: center;'>🏠 ESTADO</h4>", unsafe_allow_html=True)
        entidad_seleccionada = st.selectbox(
            "Elige tu estado:",
            entidades,
            index=entidades.index('Nacional') if 'Nacional' in entidades else 0
        )
    
    with col2:
        st.markdown("<h4 style='color: #32cd32; text-align: center;'>📅 MES</h4>", unsafe_allow_html=True)
        mes_seleccionado = st.selectbox(
            "Elige el mes:",
            meses,
            index=0
        )
    
    with col3:
        st.markdown("<h4 style='color: #4169e1; text-align: center;'>📆 AÑO</h4>", unsafe_allow_html=True)
        año_seleccionado = st.selectbox(
            "Elige el año:",
            list(range(min(años), max(años) + 5)),
            index=len(años) - 1 if años else 0
        )
    
    # Botón gigante y llamativo
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 ¡PREDECIR AHORA! 🚀", key="predict_btn"):
            precio_predicho, error = predecir_precio(
                modelo_data, entidad_seleccionada, mes_seleccionado, año_seleccionado
            )
            
            if error:
                st.markdown(f"""
                <div style='background-color: #ff6b6b; padding: 20px; border: 3px solid #000; 
                border-radius: 15px; text-align: center;'>
                <h3 style='color: white;'>❌ ERROR ❌</h3>
                <p style='color: white;'>{error}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
                padding: 30px; border: 5px dashed #ff1493; border-radius: 20px; 
                text-align: center; margin: 20px 0; box-shadow: 0 0 20px #ff69b4;'>
                    <h1 style='color: #8b008b; font-family: Impact; text-shadow: 2px 2px 4px #000;'>
                    💵 PRECIO: ${precio_predicho:.2f} PESOS 💵
                    </h1>
                    <h3 style='color: #4b0082; font-family: Comic Sans MS;'>
                    📍 {entidad_seleccionada} | 📅 {mes_seleccionado} {año_seleccionado}
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.snow()

                st.markdown("""
                <div style='background-color: #87ceeb; padding: 15px; border: 3px solid #4682b4; 
                border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: #191970; text-align: center;'>📊 DATOS EXTRA GENIALES 📊</h3>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div style='background-color: #ffb6c1; padding: 10px; border: 2px solid #ff69b4; 
                    border-radius: 10px; text-align: center;'>
                    <h4 style='color: #8b008b;'>🏠 ESTADO</h4>
                    <p style='color: #4b0082; font-weight: bold;'>{entidad_seleccionada}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div style='background-color: #98fb98; padding: 10px; border: 2px solid #32cd32; 
                    border-radius: 10px; text-align: center;'>
                    <h4 style='color: #006400;'>📅 FECHA</h4>
                    <p style='color: #228b22; font-weight: bold;'>{mes_seleccionado} {año_seleccionado}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    precio_historico = df_gasolina[
                        (df_gasolina['Entidad'] == entidad_seleccionada) &
                        (df_gasolina['Mes'] == mes_seleccionado)
                    ]['Precio'].mean()
                    
                    if not pd.isna(precio_historico):
                        diferencia = precio_predicho - precio_historico
                        color = "#ff6347" if diferencia > 0 else "#32cd32"
                        st.markdown(f"""
                        <div style='background-color: #ffd700; padding: 10px; border: 2px solid #ffa500; 
                        border-radius: 10px; text-align: center;'>
                        <h4 style='color: #b8860b;'>📈 COMPARACIÓN</h4>
                        <p style='color: {color}; font-weight: bold;'>{diferencia:+.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color: #dda0dd; padding: 10px; border: 2px solid #9370db; 
                        border-radius: 10px; text-align: center;'>
                        <h4 style='color: #4b0082;'>💰 PRECIO</h4>
                        <p style='color: #8b008b; font-weight: bold;'>${precio_predicho:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background: linear-gradient(45deg, #ff6b35, #f7931e); padding: 20px; 
    border: 4px solid #000; border-radius: 15px; margin: 30px 0;'>
    <h2 style='text-align: center; color: white; font-family: Impact; 
    text-shadow: 3px 3px 6px #000;'>
    DATOS DEL PASADO 
    </h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Filtrar datos para la entidad seleccionada
    datos_entidad = df_gasolina[df_gasolina['Entidad'] == entidad_seleccionada].copy()
    
    if not datos_entidad.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            precio_min = datos_entidad['Precio'].min()
            precio_max = datos_entidad['Precio'].max()
            precio_promedio = datos_entidad['Precio'].mean()
            
            st.markdown(f"""
            <div style='background-color: #ffcccb; padding: 15px; border: 3px dashed #ff0000; 
            border-radius: 10px;'>
            <h3 style='color: #8b0000; text-align: center;'>📊 NÚMEROS LOCOS</h3>
            <p style='color: #000080; font-size: 18px;'><b>🔻 Mínimo:</b> ${precio_min:.2f}</p>
            <p style='color: #000080; font-size: 18px;'><b>🔺 Máximo:</b> ${precio_max:.2f}</p>
            <p style='color: #000080; font-size: 18px;'><b>📊 Promedio:</b> ${precio_promedio:.2f}</p>
            <p style='color: #000080; font-size: 18px;'><b>📝 Registros:</b> {len(datos_entidad)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #e6e6fa; padding: 15px; border: 3px solid #9370db; 
            border-radius: 10px;'>
            <h3 style='color: #4b0082; text-align: center;'>📉 GRÁFICO CHIDO</h3>
            </div>
            """, unsafe_allow_html=True)
            
            datos_grafico = datos_entidad.groupby(['Año', 'Mes'])['Precio'].mean().reset_index()
            datos_grafico['Fecha'] = datos_grafico['Año'].astype(str) + '-' + datos_grafico['Mes']
            
            st.line_chart(
                data=datos_grafico.set_index('Fecha')['Precio'],
                height=250
            )
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    padding: 20px; border: 5px ridge #4b0082; border-radius: 20px; margin: 30px 0;'>
    <h2 style='text-align: center; color: #ffff00; font-family: Impact; 
    text-shadow: 2px 2px 4px #000;'>
    ESTADÍSTICAS DEL MODELO
    </h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    metricas = modelo_data.get('metricas', {})
    
    with col1:
        r2 = metricas.get('r2_test', 0)
        st.markdown(f"""
        <div style='background-color: #ff69b4; padding: 15px; border: 3px solid #ff1493; 
        border-radius: 15px; text-align: center;'>
        <h3 style='color: white; font-family: Comic Sans MS;'>🎯 R² SCORE</h3>
        <h2 style='color: #ffff00; text-shadow: 1px 1px 2px #000;'>{r2:.3f}</h2>
        <p style='color: #ffe4e1; font-size: 12px;'>Qué tan bueno es (0-1)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        rmse = metricas.get('rmse_test', 0)
        st.markdown(f"""
        <div style='background-color: #32cd32; padding: 15px; border: 3px solid #228b22; 
        border-radius: 15px; text-align: center;'>
        <h3 style='color: white; font-family: Comic Sans MS;'>📉 RMSE</h3>
        <h2 style='color: #ffff00; text-shadow: 1px 1px 2px #000;'>{rmse:.3f}</h2>
        <p style='color: #f0fff0; font-size: 12px;'>Error cuadrático</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mae = metricas.get('mae_test', 0)
        st.markdown(f"""
        <div style='background-color: #ff6347; padding: 15px; border: 3px solid #dc143c; 
        border-radius: 15px; text-align: center;'>
        <h3 style='color: white; font-family: Comic Sans MS;'>📊 MAE</h3>
        <h2 style='color: #ffff00; text-shadow: 1px 1px 2px #000;'>{mae:.3f}</h2>
        <p style='color: #ffe4e1; font-size: 12px;'>Error absoluto</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #000; padding: 20px; border-radius: 10px; 
    text-align: center; margin: 30px 0;'>
    <h3 style='color: #00ff00; font-family: Courier New;'>
    💻 Hecho con amor y mucho café ☕ 💻
    </h3>
    <p style='color: #ffff00;'>Powered by Streamlit & Python 🐍</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error("""
    ❌ **Error de Configuración**
    
    No se pudieron cargar los datos o el modelo necesarios. 
    
    **Pasos para solucionar:**
    1. Asegúrate de que el archivo `Gasolina_expandido.csv` existe en el directorio padre
    2. Asegúrate de que el archivo `modelo_gasolina.pkl` existe en el directorio padre
    3. Ejecuta primero el notebook `analisis_gasolina.ipynb` para generar estos archivos
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Modelo de Regresión Lineal Múltiple</p>
    <p>Datos históricos de precios de gasolina en México</p>
</div>
""", unsafe_allow_html=True)
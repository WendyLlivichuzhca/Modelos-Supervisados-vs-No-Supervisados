# app.py - IMPLEMENTACIÓN COMPLETA EN ESPAÑOL - DISEÑO MEJORADO
import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import FastICA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, silhouette_score, davies_bouldin_score
)
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
from scipy.stats import kurtosis

# =============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="ML: Árboles Extra + FastICA",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal con mejor formato
st.title("🌲 Clasificador Árboles Extra + ⚡ FastICA")
st.markdown("---")

# =============================================================================
# SIDEBAR - CONFIGURACIÓN
# =============================================================================
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Selección de dataset en español
    dataset_option = st.selectbox(
        "📊 Seleccionar Conjunto de Datos",
        ["Flores Iris", "Vinos", "Cáncer de Mama"]
    )
    
    # Cargar dataset seleccionado con nombres en español
    @st.cache_data
    def load_data(dataset_name):
        if dataset_name == "Flores Iris":
            data = load_iris()
            # Traducir nombres de características al español
            feature_names_es = [
                "Longitud del Sépalo (cm)",
                "Ancho del Sépalo (cm)", 
                "Longitud del Pétalo (cm)",
                "Ancho del Pétalo (cm)"
            ]
            target_names_es = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"]
            
        elif dataset_name == "Vinos":
            data = load_wine()
            feature_names_es = [
                "Alcohol", "Ácido Málico", "Cenizas", "Alcalinidad de Cenizas",
                "Magnesio", "Fenoles Totales", "Flavonoides", "Fenoles No Flavonoides",
                "Proantocianinas", "Intensidad de Color", "Matiz", "OD280/OD315",
                "Prolina"
            ]
            target_names_es = ["Vino Tipo 1", "Vino Tipo 2", "Vino Tipo 3"]
            
        else:  # Cáncer de Mama
            data = load_breast_cancer()
            feature_names_es = [
                "Radio Medio", "Textura Media", "Perímetro Medio", "Área Media",
                "Suavidad Media", "Compacidad Media", "Concavidad Media", "Puntos Cóncavos Medios",
                "Simetría Media", "Dimensión Fractal Media", "Radio Error", "Textura Error",
                "Perímetro Error", "Área Error", "Suavidad Error", "Compacidad Error", 
                "Concavidad Error", "Puntos Cóncavos Error", "Simetría Error", "Dimensión Fractal Error",
                "Radio Peor", "Textura Peor", "Perímetro Peor", "Área Peor",
                "Suavidad Peor", "Compacidad Peor", "Concavidad Peor", "Puntos Cóncavos Peor",
                "Simetría Peor", "Dimensión Fractal Peor"
            ]
            target_names_es = ["Benigno", "Maligno"]
        
        X = pd.DataFrame(data.data, columns=feature_names_es)
        y = pd.Series(data.target, name='target')
        return X, y, feature_names_es, target_names_es

    X, y, feature_names, target_names = load_data(dataset_option)

    # Mostrar información del dataset
    st.subheader("📋 Información del Dataset")
    info_container = st.container()
    with info_container:
        st.write(f"**Dataset:** {dataset_option}")
        st.write(f"**Muestras:** {X.shape[0]}")
        st.write(f"**Características:** {X.shape[1]}")
        st.write(f"**Clases:** {len(target_names)}")
    
    st.markdown("---")
    st.info("💡 Selecciona un dataset y explora las diferentes pestañas para analizar los datos y entrenar modelos.")

# =============================================================================
# PESTAÑAS PRINCIPALES
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Análisis Exploratorio", 
    "🌲 Modo Supervisado", 
    "⚡ Modo No Supervisado",
    "📤 Zona de Exportación"
])

# =============================================================================
# PESTAÑA 1: ANÁLISIS EXPLORATORIO
# =============================================================================
with tab1:
    st.header("📊 Análisis Exploratorio de Datos")
    
    # Contenedor para estadísticas generales
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Muestras", X.shape[0])
        with col2:
            st.metric("Total de Características", X.shape[1])
        with col3:
            st.metric("Número de Clases", len(target_names))
        with col4:
            st.metric("Datos Ausentes", "0")
    
    st.markdown("---")
    
    # Sección de datos y estadísticas
    col_data, col_stats = st.columns([1, 1])
    
    with col_data:
        st.subheader("👀 Vista Previa de los Datos")
        with st.expander("Ver datos", expanded=True):
            preview_df = X.copy()
            preview_df['Clase'] = y
            preview_df['Nombre Clase'] = [target_names[i] for i in y]
            st.dataframe(preview_df.head(10), use_container_width=True)
            st.caption(f"Mostrando 10 de {X.shape[0]} registros")
    
    with col_stats:
        st.subheader("📈 Estadísticas Descriptivas")
        with st.expander("Ver estadísticas", expanded=True):
            st.dataframe(X.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Sección de visualizaciones
    st.subheader("📈 Visualizaciones")
    
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        with st.container():
            st.subheader("🎯 Distribución de Clases")
            class_dist = y.value_counts()
            fig_pie = px.pie(
                names=[target_names[i] for i in class_dist.index],
                values=class_dist.values,
                title="Distribución de Clases en el Dataset"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with viz_col2:
        with st.container():
            st.subheader("📊 Distribución de Características")
            selected_feature = st.selectbox("Seleccionar característica:", feature_names, key="hist_feature")
            fig_hist = px.histogram(
                X, 
                x=selected_feature,
                title=f"Distribución de {selected_feature}",
                color_discrete_sequence=['#3366CC']
            )
            st.plotly_chart(fig_hist, use_container_width=True)

# =============================================================================
# PESTAÑA 2: MODO SUPERVISADO - EXTRA TREES CLASSIFIER
# =============================================================================
with tab2:
    st.header("🌲 Modo Supervisado: Clasificador Árboles Extra")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo Supervisado", expanded=False):
        st.markdown("""
        **🌲 Clasificador Árboles Extra (Extra Trees Classifier)**
        
        **Descripción del algoritmo:**
        - Es un método de conjunto que construye múltiples árboles de decisión
        - Combina las predicciones de todos los árboles para mejorar la precisión
        - Más aleatorizado que Random Forest (selecciona puntos de corte aleatorios)
        
        **Tipo de problema que resuelve:** Clasificación supervisada
        
        **Ventajas:**
        - Menos propenso al sobreajuste
        - Más rápido en entrenamiento que Random Forest
        - Maneja bien características no lineales
        - Robustez frente a ruido en los datos
        """)
    
    # Configuración y resultados en columnas
    config_col, results_col = st.columns([2, 1])
    
    with config_col:
        with st.container():
            st.subheader("🛠️ Configuración del Modelo")
            
            # Parámetros del modelo en columnas
            param_col1, param_col2, param_col3 = st.columns(3)
            
            with param_col1:
                n_estimators = st.slider("Número de árboles", 10, 200, 100, key="et_trees")
            with param_col2:
                max_depth = st.slider("Profundidad máxima", 2, 20, 10, key="et_depth")
            with param_col3:
                test_size = st.slider("Tamaño de prueba (%)", 10, 40, 20, key="et_test")
            
            train_button = st.button("🚀 Entrenar Modelo Árboles Extra", 
                                   type="primary", 
                                   key="et_train",
                                   use_container_width=True)
            
            if train_button:
                with st.spinner("Entrenando modelo Árboles Extra..."):
                    # Dividir datos (80% entrenamiento, 20% prueba)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=42, stratify=y
                    )
                    
                    # Entrenar modelo
                    et_model = ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=42
                    )
                    et_model.fit(X_train, y_train)
                    
                    # Predicciones
                    y_pred = et_model.predict(X_test)
                    y_pred_proba = et_model.predict_proba(X_test)
                    
                    # Calcular métricas REQUERIDAS
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    
                    # Guardar en session state
                    st.session_state.et_model = et_model
                    st.session_state.current_dataset = dataset_option
                    st.session_state.current_features = feature_names
                    st.session_state.et_metrics = {
                        'accuracy': float(accuracy),
                        'precision': float(precision),
                        'recall': float(recall),
                        'f1_score': float(f1)
                    }
                    st.session_state.et_predictions = {
                        'y_test': y_test.tolist(),
                        'y_pred': y_pred.tolist(),
                        'y_pred_proba': y_pred_proba.tolist()
                    }
                    
                    st.success("✅ Modelo Árboles Extra entrenado exitosamente!")
    
    with results_col:
        with st.container():
            st.subheader("📊 Métricas de Evaluación")
            
            if 'et_metrics' in st.session_state:
                metrics = st.session_state.et_metrics
                
                # Mostrar métricas REQUERIDAS
                st.metric("Exactitud", f"{metrics['accuracy']:.4f}", 
                         delta="Alto" if metrics['accuracy'] > 0.9 else "Bueno" if metrics['accuracy'] > 0.7 else "Regular")
                st.metric("Precisión", f"{metrics['precision']:.4f}")
                st.metric("Sensibilidad", f"{metrics['recall']:.4f}")
                st.metric("Puntuación F1", f"{metrics['f1_score']:.4f}")
                
                # Interpretación de métricas
                with st.expander("💡 Interpretación de Métricas"):
                    st.markdown(f"""
                    **Exactitud ({metrics['accuracy']:.2%}):** 
                    - Proporción de predicciones correctas sobre el total
                    
                    **Precisión ({metrics['precision']:.2%}):**
                    - De las predicciones positivas, cuántas son realmente positivas
                    
                    **Sensibilidad ({metrics['recall']:.2%}):**
                    - De los casos realmente positivos, cuántos logró identificar
                    
                    **Puntuación F1 ({metrics['f1_score']:.2%}):**
                    - Media armónica entre Precisión y Sensibilidad
                    """)
            else:
                st.info("ℹ️ Entrena el modelo primero para ver las métricas")
                st.metric("Exactitud", "0.0000")
                st.metric("Precisión", "0.0000")
                st.metric("Sensibilidad", "0.0000")
                st.metric("Puntuación F1", "0.0000")
    
    # PREDICCIÓN INTERACTIVA REQUERIDA
    st.markdown("---")
    if 'et_model' in st.session_state:
        st.subheader("🔮 Predicción Interactiva")
        
        with st.container():
            st.write(f"**Dataset actual:** {st.session_state.current_dataset}")
            st.write(f"**Características esperadas:** {len(st.session_state.current_features)}")
            st.write("Ingresa los valores de las características para realizar una predicción:")
            
            # Verificar que el modelo fue entrenado con el mismo dataset
            if st.session_state.current_dataset != dataset_option:
                st.error(f"⚠️ El modelo fue entrenado con el dataset '{st.session_state.current_dataset}' pero actualmente estás viendo '{dataset_option}'. Cambia al dataset correcto o entrena un nuevo modelo.")
            else:
                # Crear inputs para características específicas del dataset actual
                input_features = []
                
                # Organizar sliders en columnas
                num_cols = 4
                cols = st.columns(num_cols)
                
                # Mostrar sliders para todas las características del dataset actual
                for i, feature in enumerate(feature_names):
                    with cols[i % num_cols]:
                        min_val = float(X[feature].min())
                        max_val = float(X[feature].max())
                        default_val = float(X[feature].mean())
                        
                        value = st.slider(
                            feature,
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"pred_{feature}"
                        )
                        input_features.append(value)
                
                predict_col1, predict_col2 = st.columns([1, 4])
                with predict_col1:
                    predict_btn = st.button("🎯 Predecir Clase", type="secondary", key="predict_btn", use_container_width=True)
                
                if predict_btn:
                    try:
                        # Crear DataFrame con las características en el orden correcto
                        input_df = pd.DataFrame([input_features], columns=feature_names)
                        
                        # Realizar predicción
                        prediction = st.session_state.et_model.predict(input_df)
                        probability = st.session_state.et_model.predict_proba(input_df)
                        
                        # Guardar predicción actual para exportación
                        st.session_state.current_prediction = {
                            'input': input_features,
                            'output_class': int(prediction[0]),
                            'output_label': target_names[prediction[0]],
                            'probabilities': probability[0].tolist()
                        }
                        
                        with predict_col2:
                            st.success(f"**🎉 Clase predicha:** {target_names[prediction[0]]}")
                        
                        # Mostrar probabilidades
                        prob_df = pd.DataFrame({
                            'Clase': target_names,
                            'Probabilidad': probability[0]
                        })
                        
                        fig_proba = px.bar(
                            prob_df, 
                            x='Clase', 
                            y='Probabilidad',
                            color='Probabilidad',
                            color_continuous_scale='Viridis',
                            title="Probabilidades por Clase"
                        )
                        st.plotly_chart(fig_proba, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error en la predicción: {str(e)}")

# =============================================================================
# PESTAÑA 3: MODO NO SUPERVISADO - FASTICA
# =============================================================================
with tab3:
    st.header("⚡ Modo No Supervisado: FastICA")
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo No Supervisado", expanded=False):
        st.markdown("""
        **⚡ FastICA (Análisis de Componentes Independientes)**
        
        **Descripción del algoritmo:**
        - Algoritmo de separación ciega de fuentes que encuentra componentes estadísticamente independientes
        - Basado en la maximización de la no-gaussianidad mediante negentropía
        
        **Tipo de problema que resuelve:** Reducción de dimensionalidad no supervisada
        
        **Ventajas:**
        - No requiere distribución gaussiana de los datos
        - Eficiente computacionalmente
        - Encuentra componentes no lineales
        - Robustez frente a outliers
        """)
    
    # Configuración del modelo
    config_col, results_col = st.columns([2, 1])
    
    with config_col:
        with st.container():
            st.subheader("🛠️ Configuración de FastICA")
            
            # Parámetros de FastICA en columnas
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                n_components = st.slider("Número de Componentes", 2, min(5, X.shape[1]), 3, key="ica_comp")
            with param_col2:
                max_iter = st.slider("Máximo de iteraciones", 100, 1000, 200, key="ica_iter")
            
            apply_button = st.button("🚀 Aplicar FastICA", 
                                   type="primary", 
                                   key="ica_apply",
                                   use_container_width=True)
            
            if apply_button:
                with st.spinner("Aplicando FastICA..."):
                    try:
                        # Normalizar datos
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Entrenar FastICA
                        ica = FastICA(
                            n_components=n_components,
                            max_iter=max_iter,
                            random_state=42
                        )
                        
                        # Aplicar transformación
                        X_ica = ica.fit_transform(X_scaled)
                        
                        # Calcular kurtosis (medida de no-gaussianidad)
                        kurtosis_values = kurtosis(X_ica, axis=0)
                        avg_kurtosis = np.mean(np.abs(kurtosis_values))
                        
                        # CALCULAR MÉTRICAS DE CLUSTERING REQUERIDAS
                        # Aplicar K-Means en los componentes para evaluación
                        kmeans = KMeans(n_clusters=len(target_names), random_state=42)
                        cluster_labels = kmeans.fit_predict(X_ica)
                        
                        # Calcular métricas REQUERIDAS
                        silhouette_avg = silhouette_score(X_ica, cluster_labels)
                        db_score = davies_bouldin_score(X_ica, cluster_labels)
                        
                        # Guardar en session state
                        st.session_state.ica_model = ica
                        st.session_state.ica_dataset = dataset_option
                        st.session_state.ica_results = {
                            'X_ica': X_ica.tolist(),
                            'kurtosis_values': kurtosis_values.tolist(),
                            'avg_kurtosis': float(avg_kurtosis),
                            'n_components': n_components,
                            'cluster_labels': cluster_labels.tolist(),
                            'silhouette_score': float(silhouette_avg),
                            'davies_bouldin_score': float(db_score)
                        }
                        st.session_state.scaler = scaler
                        
                        st.success("✅ FastICA aplicado exitosamente!")
                        
                    except Exception as e:
                        st.error(f"❌ Error en FastICA: {str(e)}")
    
    with results_col:
        with st.container():
            st.subheader("📊 Métricas de Calidad")
            
            if 'ica_results' in st.session_state:
                ica_results = st.session_state.ica_results
                
                # Mostrar métricas REQUERIDAS
                st.metric("Puntuación Silueta", f"{ica_results['silhouette_score']:.4f}",
                         delta="Excelente" if ica_results['silhouette_score'] > 0.7 else "Bueno" if ica_results['silhouette_score'] > 0.5 else "Regular")
                st.metric("Índice Davies-Bouldin", f"{ica_results['davies_bouldin_score']:.4f}",
                         delta="Excelente" if ica_results['davies_bouldin_score'] < 0.5 else "Bueno" if ica_results['davies_bouldin_score'] < 1.0 else "Regular")
                st.metric("Curtosis Promedio", f"{ica_results['avg_kurtosis']:.4f}")
                
                # Interpretación de métricas
                with st.expander("💡 Interpretación de Métricas"):
                    st.markdown(f"""
                    **Puntuación Silueta ({ica_results['silhouette_score']:.4f}):** 
                    - Mide cuán similar es cada punto a su cluster vs otros clusters
                    - **Rango:** -1 a 1 (1 es mejor)
                    
                    **Índice Davies-Bouldin ({ica_results['davies_bouldin_score']:.4f}):**
                    - Mide separación y compactación de clusters
                    - **Valores más bajos = mejor separación**
                    
                    **Curtosis ({ica_results['avg_kurtosis']:.4f}):**
                    - Mide no-gaussianidad de componentes
                    - FastICA busca maximizar la no-gaussianidad
                    """)
            else:
                st.info("ℹ️ Aplica FastICA primero para ver las métricas")
                st.metric("Puntuación Silueta", "0.0000")
                st.metric("Índice Davies-Bouldin", "0.0000")
                st.metric("Curtosis Promedio", "0.0000")
    
    # VISUALIZACIÓN DE CLUSTERS REQUERIDA
    st.markdown("---")
    if 'ica_results' in st.session_state and st.session_state.ica_dataset == dataset_option:
        st.subheader("📈 Visualización de Clusters")
        
        ica_results = st.session_state.ica_results
        
        # Crear DataFrame para visualización
        ica_data = []
        for i in range(len(ica_results['X_ica'])):
            row_data = {}
            for j in range(ica_results['n_components']):
                row_data[f'IC_{j+1}'] = ica_results['X_ica'][i][j]
            row_data['cluster'] = ica_results['cluster_labels'][i]
            row_data['target'] = y.iloc[i]  # Clase real para comparación
            ica_data.append(row_data)
        
        ica_df = pd.DataFrame(ica_data)
        
        # VISUALIZACIÓN REQUERIDA: Scatter plot de clusters
        if ica_results['n_components'] >= 2:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                with st.container():
                    # Coloreado por cluster
                    fig_clusters = px.scatter(
                        ica_df,
                        x='IC_1',
                        y='IC_2',
                        color='cluster',
                        title="Clusters en Componentes Independientes",
                        labels={'cluster': 'Cluster'},
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True)
            
            with viz_col2:
                with st.container():
                    # Coloreado por clase real (para comparación)
                    fig_real = px.scatter(
                        ica_df,
                        x='IC_1',
                        y='IC_2',
                        color='target',
                        title="Clases Reales en Componentes Independientes",
                        labels={'target': 'Clase Real'},
                        color_continuous_scale='plasma'
                    )
                    st.plotly_chart(fig_real, use_container_width=True)
            
            # Análisis de resultados
            with st.expander("🔍 Análisis de Resultados"):
                st.write("**Observaciones:**")
                st.write("- Los clusters muestran cómo FastICA separa los datos en componentes independientes")
                st.write("- Compara la visualización por clusters (izquierda) con las clases reales (derecha)")
                st.write("- Un buen resultado mostraría clusters bien definidos y separados")
                
    elif 'ica_results' in st.session_state:
        st.warning(f"⚠️ Los resultados de FastICA son del dataset '{st.session_state.ica_dataset}'. Cambia al dataset correcto o aplica FastICA nuevamente.")

# =============================================================================
# PESTAÑA 4: ZONA DE EXPORTACIÓN
# =============================================================================
with tab4:
    st.header("📤 Zona de Exportación")
    
    with st.container():
        st.info("""
        **Integración Python-React mediante JSON:**
        
        Esta sección permite exportar los resultados para ser consumidos por una aplicación React.
        Python/Streamlit entrena los modelos y calcula métricas, luego exporta archivos JSON 
        estructurados que React puede leer sin necesidad de "saber" Python, permitiendo la 
        interoperabilidad entre backend (Python/IA) y frontend (React).
        """)
    
    st.markdown("---")
    
    # Exportación en columnas
    json_col, model_col = st.columns(2)
    
    with json_col:
        with st.container():
            st.subheader("📄 Exportar a JSON (para React)")
            
            # PREPARAR DATOS PARA EXPORTACIÓN JSON REQUERIDA
            if 'et_metrics' in st.session_state and st.session_state.current_dataset == dataset_option:
                # JSON para modelo supervisado (FORMATO REQUERIDO)
                supervised_data = {
                    "model_type": "Supervised",
                    "model_name": "Extra Trees Classifier",
                    "metrics": st.session_state.et_metrics,
                    "current_prediction": st.session_state.current_prediction if 'current_prediction' in st.session_state else {
                        "input": [],
                        "output_class": 0,
                        "output_label": "Sin predicción aún"
                    }
                }
                
                supervised_json = json.dumps(supervised_data, indent=2)
                
                download_col1, preview_col1 = st.columns([1, 1])
                with download_col1:
                    st.download_button(
                        label="📥 Descargar JSON - Árboles Extra",
                        data=supervised_json,
                        file_name="arboles_extra_resultados.json",
                        mime="application/json",
                        use_container_width=True,
                        help="Descarga los resultados del modelo supervisado en formato JSON para React"
                    )
                with preview_col1:
                    with st.expander("👀 Previsualizar JSON"):
                        st.code(supervised_json, language="json")
            
            if 'ica_results' in st.session_state and st.session_state.ica_dataset == dataset_option:
                # JSON para modelo no supervisado (FORMATO REQUERIDO)
                unsupervised_data = {
                    "model_type": "Unsupervised",
                    "algorithm": "FastICA",
                    "parameters": {
                        "n_components": n_components,
                        "max_iter": max_iter
                    },
                    "metrics": {
                        "silhouette_score": st.session_state.ica_results['silhouette_score'],
                        "davies_bouldin": st.session_state.ica_results['davies_bouldin_score']
                    },
                    "cluster_labels": st.session_state.ica_results['cluster_labels']
                }
                
                unsupervised_json = json.dumps(unsupervised_data, indent=2)
                
                download_col2, preview_col2 = st.columns([1, 1])
                with download_col2:
                    st.download_button(
                        label="📥 Descargar JSON - FastICA",
                        data=unsupervised_json,
                        file_name="fastica_resultados.json",
                        mime="application/json",
                        use_container_width=True,
                        help="Descarga los resultados del modelo no supervisado en formato JSON para React"
                    )
                with preview_col2:
                    with st.expander("👀 Previsualizar JSON"):
                        st.code(unsupervised_json, language="json")
            
            if ('et_metrics' not in st.session_state or st.session_state.current_dataset != dataset_option) and \
               ('ica_results' not in st.session_state or st.session_state.ica_dataset != dataset_option):
                st.warning("⚠️ Entrena los modelos con el dataset actual para exportar JSON")
    
    with model_col:
        with st.container():
            st.subheader("🤖 Exportar Modelos (.pkl)")
            
            # EXPORTACIÓN DE MODELOS .PKL REQUERIDA
            model_export_col1, model_export_col2 = st.columns(2)
            
            with model_export_col1:
                if 'et_model' in st.session_state and st.session_state.current_dataset == dataset_option:
                    et_model_pkl = pickle.dumps(st.session_state.et_model)
                    st.download_button(
                        label="🌲 Descargar Árboles Extra",
                        data=et_model_pkl,
                        file_name="modelo_arboles_extra.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                        help="Descarga el modelo Árboles Extra entrenado en formato Pickle"
                    )
                else:
                    st.button("🌲 Descargar Árboles Extra", 
                             disabled=True,
                             use_container_width=True,
                             help="Entrena el modelo primero para descargar")
            
            with model_export_col2:
                if 'ica_model' in st.session_state and st.session_state.ica_dataset == dataset_option:
                    ica_model_pkl = pickle.dumps(st.session_state.ica_model)
                    st.download_button(
                        label="⚡ Descargar FastICA",
                        data=ica_model_pkl,
                        file_name="modelo_fastica.pkl",
                        mime="application/octet-stream",
                        use_container_width=True,
                        help="Descarga el modelo FastICA entrenado en formato Pickle"
                    )
                else:
                    st.button("⚡ Descargar FastICA", 
                             disabled=True,
                             use_container_width=True,
                             help="Aplica FastICA primero para descargar")
            
            # Información sobre el uso de los archivos
            with st.expander("📚 Instrucciones de Uso", expanded=False):
                st.markdown("""
                **Para usar los archivos exportados:**
                
                **Archivos JSON (para React):**
                ```javascript
                // En tu componente React
                import React, { useState, useEffect } from 'react';
                
                function VisorResultados() {
                    const [datosModelo, setDatosModelo] = useState(null);
                    
                    useEffect(() => {
                        fetch('/arboles_extra_resultados.json')
                            .then(respuesta => respuesta.json())
                            .then(datos => setDatosModelo(datos));
                    }, []);
                    
                    return (
                        <div>
                            <h2>{datosModelo?.model_name}</h2>
                            <p>Exactitud: {datosModelo?.metrics.accuracy}</p>
                        </div>
                    );
                }
                ```
                
                **Modelos .pkl (para Python):**
                ```python
                import pickle
                import pandas as pd
                
                # Cargar modelo
                with open('modelo_arboles_extra.pkl', 'rb') as f:
                    modelo = pickle.load(f)
                
                # Hacer predicciones
                predicciones = modelo.predict(nuevos_datos)
                ```
                """)


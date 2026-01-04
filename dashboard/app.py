"""
Dashboard Streamlit pour YOLO QA Framework
==========================================

Lancement :
    streamlit run dashboard/app.py

Fonctionnalités :
    - Upload et détection d'images
    - Comparaison de modèles
    - Visualisation des performances
    - Historique des tests
"""

import streamlit as st
import sys
from pathlib import Path
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
image_path=Path(__file__).parent / "assets" / "yolo.webp"

# Ajouter le répertoire racine au path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.yolo_detector import YOLODetector


# Configuration page
st.set_page_config(
    page_title="YOLO QA Dashboard",
    page_icon="",
    layout="wide"
)

st.title(" YOLO Quality Assurance Dashboard") 
st.markdown("---")


# ========================================
# SIDEBAR : Configuration
# ========================================
with st.sidebar:
    st.header("Configuration")
    
    # Sélection du modèle
    model_options = {
        "YOLOv8n (nano)": "yolov8n.pt",
        "YOLOv8s (small)": "yolov8s.pt",
        "YOLOv8m (medium)": "yolov8m.pt"
    }
    
    selected_model_name = st.selectbox(
        "Modèle YOLO",
        list(model_options.keys())
    )
    
    # Seuil de confiance
    conf_threshold = st.slider(
        "Seuil de confiance",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )
    
    st.markdown("---")
    
    # Statistiques rapides
    st.subheader(" Statistiques")
    
    # Charger baseline si existe
    baseline_file = Path("tests/baseline_metrics.json")
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        st.metric("Version Baseline", baseline['version'])
        st.metric("Latence Baseline", 
                  f"{baseline['metrics']['avg_latency_ms']:.1f} ms")
    else:
        st.warning("Aucune baseline trouvée")


# ========================================
# TAB 1 : Détection sur Image
# ========================================
tab1, tab2, tab3, tab4 = st.tabs([
    " Détection Image", 
    " Performance", 
    " Comparaison Modèles",
    " Historique"
])

with tab1:
    st.header("Détection sur Image")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choisir une image",
            type=['jpg', 'jpeg', 'png']
        )
        
        if uploaded_file:
            # Afficher image originale
            image = Image.open(uploaded_file)
            st.image(image, caption="Image originale", use_column_width=True)
            
            # Bouton détection
            if st.button(" Lancer Détection", type="primary"):
                with st.spinner("Détection en cours..."):
                    # Sauvegarder temporairement
                    temp_path = Path("temp_upload.jpg")
                    image.save(temp_path)
                    
                    # Détection
                    model_path = model_options[selected_model_name]
                    detector = YOLODetector(model_path=model_path)
                    
                    start = time.perf_counter()
                    result = detector.detect(
                        str(temp_path), 
                        conf_threshold=conf_threshold
                    )
                    inference_time = time.perf_counter() - start
                    
                    # Stocker résultat en session
                    st.session_state['last_result'] = result
                    st.session_state['inference_time'] = inference_time
    
    with col2:
        st.subheader("Résultats")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            inf_time = st.session_state['inference_time']
            
            # Métriques
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Détections", result['num_detections'])
            col_b.metric("Temps", f"{inf_time*1000:.1f} ms")
            col_c.metric("FPS", f"{1/inf_time:.1f}")
            
            # Liste des détections
            if result['num_detections'] > 0:
                st.subheader(" Objets Détectés")
                
                detections_df = pd.DataFrame(result['detections'])
                detections_df['confidence'] = detections_df['confidence'].apply(
                    lambda x: f"{x:.2%}"
                )
                
                st.dataframe(
                    detections_df[['class_name', 'confidence']],
                    use_container_width=True
                )
                
                # Graphique des confiances
                fig = px.bar(
                    result['detections'],
                    x='class_name',
                    y='confidence',
                    title="Confiances par Classe",
                    labels={'confidence': 'Confiance', 'class_name': 'Classe'}
                )
                st.plotly_chart(fig, use_container_width=True)


# ========================================
# TAB 2 : Performance
# ========================================
with tab2:
    st.header(" Analyse de Performance")
    
    if st.button(" Lancer Benchmark"):
        with st.spinner("Benchmark en cours (cela peut prendre 30s)..."):
            
            # Préparer une image de test
            test_image = Path("data/images/normal/cats_sofa.jpg")
            
            if not test_image.exists():
                st.error("Image de test non trouvée")
            else:
                detector = YOLODetector(
                    model_path=model_options[selected_model_name]
                )
                
                # Benchmark latence
                latencies = []
                progress_bar = st.progress(0)
                
                num_runs = 50
                for i in range(num_runs):
                    start = time.perf_counter()
                    detector.detect(str(test_image))
                    latencies.append(time.perf_counter() - start)
                    progress_bar.progress((i + 1) / num_runs)
                
                progress_bar.empty()
                
                # Statistiques
                avg_latency = sum(latencies) / len(latencies)
                p50 = sorted(latencies)[len(latencies)//2]
                p95 = sorted(latencies)[int(len(latencies)*0.95)]
                p99 = sorted(latencies)[int(len(latencies)*0.99)]
                
                # Affichage métriques
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Moyenne", f"{avg_latency*1000:.1f} ms")
                col2.metric("P50", f"{p50*1000:.1f} ms")
                col3.metric("P95", f"{p95*1000:.1f} ms")
                col4.metric("P99", f"{p99*1000:.1f} ms")
                
                # Graphique distribution
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=[l*1000 for l in latencies],
                    nbinsx=30,
                    name="Latence"
                ))
                fig.update_layout(
                    title="Distribution des Latences",
                    xaxis_title="Latence (ms)",
                    yaxis_title="Fréquence"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Graphique temporel
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    y=[l*1000 for l in latencies],
                    mode='lines+markers',
                    name="Latence"
                ))
                fig2.add_hline(
                    y=avg_latency*1000, 
                    line_dash="dash", 
                    annotation_text="Moyenne"
                )
                fig2.update_layout(
                    title="Latence au Fil du Temps",
                    xaxis_title="Run #",
                    yaxis_title="Latence (ms)"
                )
                st.plotly_chart(fig2, use_container_width=True)


# ========================================
# TAB 3 : Comparaison Modèles
# ========================================
with tab3:
    st.header(" Comparaison de Modèles")
    
    st.markdown("""
    Compare les performances de différents modèles YOLO sur la même image.
    """)
    
    # Sélection des modèles à comparer
    models_to_compare = st.multiselect(
        "Sélectionner les modèles",
        list(model_options.keys()),
        default=list(model_options.keys())[:2]
    )
    
    # Image de test
    test_image_path = st.text_input(
        "Chemin image de test",
        value="data/images/normal/image_0.jpg"
    )
    
    if st.button("🏁 Comparer Modèles"):
        if not Path(test_image_path).exists():
            st.error("Image non trouvée")
        else:
            comparison_results = []
            
            progress = st.progress(0)
            for idx, model_name in enumerate(models_to_compare):
                with st.spinner(f"Test de {model_name}..."):
                    detector = YOLODetector(
                        model_path=model_options[model_name]
                    )
                    
                    # Benchmark
                    latencies = []
                    for _ in range(10):
                        start = time.perf_counter()
                        result = detector.detect(test_image_path)
                        latencies.append(time.perf_counter() - start)
                    
                    comparison_results.append({
                        'Modèle': model_name,
                        'Latence Moy (ms)': sum(latencies)/len(latencies)*1000,
                        'Détections': result['num_detections'],
                        'FPS': 1 / (sum(latencies)/len(latencies))
                    })
                
                progress.progress((idx + 1) / len(models_to_compare))
            
            progress.empty()
            
            # Tableau comparatif
            df = pd.DataFrame(comparison_results)
            st.dataframe(df, use_container_width=True)
            
            # Graphiques
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    df, 
                    x='Modèle', 
                    y='Latence Moy (ms)',
                    title="Latence par Modèle"
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    df,
                    x='Modèle',
                    y='Détections',
                    title="Nombre de Détections"
                )
                st.plotly_chart(fig2, use_container_width=True)


# ========================================
# TAB 4 : Historique
# ========================================
with tab4:
    st.header(" Historique des Tests")
    
    # Charger baseline
    baseline_file = Path("tests/baseline_metrics.json")
    
    if baseline_file.exists():
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        st.subheader(" Baseline Actuelle")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.json({
                "Version": baseline['version'],
                "Date": baseline['timestamp']
            })
        
        with col2:
            metrics_df = pd.DataFrame([baseline['metrics']]).T
            metrics_df.columns = ['Valeur']
            st.dataframe(metrics_df)
        
        # Graphique évolution (simulé)
        st.subheader(" Évolution des Métriques")
        
        # Créer données fictives pour démo
        dates = pd.date_range(end=pd.Timestamp.now(), periods=10, freq='D')
        evolution_data = pd.DataFrame({
            'Date': dates,
            'Latence (ms)': [
                baseline['metrics']['avg_latency_ms'] * (1 + (i-5)*0.02)
                for i in range(10)
            ],
            'Détections': [
                baseline['metrics']['avg_detections_per_image'] * (1 + (i-5)*0.01)
                for i in range(10)
            ]
        })
        
        fig = px.line(
            evolution_data,
            x='Date',
            y='Latence (ms)',
            title="Évolution de la Latence"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning(" Aucune baseline trouvée. Exécutez les tests de régression d'abord.")
        
        st.code("""
# Générer une baseline
pytest tests/test_regression.py --baseline-save
        """)


# ========================================
# Footer
# ========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p> YOLO QA Framework v1.0 | Made using Streamlit</p>
</div>
""", unsafe_allow_html=True)
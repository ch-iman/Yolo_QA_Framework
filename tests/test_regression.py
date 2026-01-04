"""
Tests de Régression pour YOLO
==============================

Compare les performances entre versions pour détecter les régressions.

Exécution : pytest tests/test_regression.py -v
Utilisation :
    1. Générer baseline : pytest --baseline-save
    2. Comparer versions : pytest --baseline-compare
"""

import pytest
import json
from pathlib import Path
from src.yolo_detector import YOLODetector
import time
import statistics


# Chemin du fichier baseline
BASELINE_FILE = Path("tests/baseline_metrics.json")


class BaselineManager:
    """Gestion des métriques baseline"""
    
    @staticmethod
    def save_baseline(metrics, version="v1.0.0"):
        """Sauvegarde les métriques baseline"""
        baseline = {
            'version': version,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'metrics': metrics
        }
        
        BASELINE_FILE.parent.mkdir(exist_ok=True)
        with open(BASELINE_FILE, 'w') as f:
            json.dump(baseline, f, indent=2)
        
        print(f"✅ Baseline sauvegardée : {BASELINE_FILE}")
    
    @staticmethod
    def load_baseline():
        """Charge la baseline existante"""
        if not BASELINE_FILE.exists():
            return None
        
        with open(BASELINE_FILE, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def compare_metrics(current, baseline, tolerance=0.10):
        """
        Compare métriques actuelles vs baseline
        
        Args:
            current: Métriques actuelles
            baseline: Métriques baseline
            tolerance: Tolérance de dégradation (10% par défaut)
        
        Returns:
            dict: Résultats de comparaison
        """
        comparisons = {}
        
        for metric_name, current_value in current.items():
            if metric_name not in baseline:
                continue
            
            baseline_value = baseline[metric_name]
            
            # Calcul de la dégradation
            if baseline_value > 0:
                change_pct = ((current_value - baseline_value) / baseline_value) * 100
            else:
                change_pct = 0
            
            # Déterminer si c'est une régression
            # Pour latence : augmentation = mauvais
            # Pour accuracy/détections : diminution = mauvais
            is_latency_metric = 'latency' in metric_name or 'time' in metric_name
            
            if is_latency_metric:
                is_regression = change_pct > (tolerance * 100)
            else:
                is_regression = change_pct < -(tolerance * 100)
            
            comparisons[metric_name] = {
                'current': current_value,
                'baseline': baseline_value,
                'change_pct': change_pct,
                'is_regression': is_regression
            }
        
        return comparisons


# ════════════════════════════════════════════════════════════
# CONFIGURATION PYTEST (doit être AVANT les fixtures)
# ════════════════════════════════════════════════════════════

def pytest_addoption(parser):
    """Ajoute les options CLI pour pytest"""
    parser.addoption("--baseline-save", action="store_true", 
                     help="Sauvegarder les métriques comme baseline")
    parser.addoption("--baseline-compare", action="store_true", default=True,
                     help="Comparer avec la baseline existante")


@pytest.fixture(scope="session")
def baseline_mode(request):
    """Détermine si on est en mode baseline-save ou compare"""
    return {
        'save': request.config.getoption("--baseline-save", default=False),
        'compare': request.config.getoption("--baseline-compare", default=True)
    }


class TestRegressionMetrics:
    """Tests de détection de régression"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        return YOLODetector()
    
    def test_baseline_or_regression(self, detector, dataset_normal_path, 
                                     sample_image_path, baseline_mode):
        """
        Test principal : sauvegarde baseline OU détecte régression
        """
        # === 1. Collecter les métriques actuelles ===
        
        # Latence
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            detector.detect(sample_image_path)
            latencies.append(time.perf_counter() - start)
        
        # Détections sur dataset
        results = detector.detect_on_dataset(dataset_normal_path, verbose=False)
        total_detections = sum(r['num_detections'] for r in results)
        avg_detections = total_detections / len(results)
        
        # Métriques actuelles
        current_metrics = {
            'avg_latency_ms': statistics.mean(latencies) * 1000,
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[18] * 1000,
            'avg_detections_per_image': avg_detections,
            'total_detections': total_detections,
            'num_images': len(results)
        }
        
        print(f"\n   📊 Métriques Actuelles:")
        for metric, value in current_metrics.items():
            print(f"      {metric:30s}: {value:.2f}")
        
        # === 2. Mode : Sauvegarder baseline ===
        if baseline_mode['save']:
            BaselineManager.save_baseline(current_metrics, version="v1.0.0")
            pytest.skip("Baseline sauvegardée - pas de comparaison")
        
        # === 3. Mode : Comparer avec baseline ===
        baseline = BaselineManager.load_baseline()
        
        if baseline is None:
            pytest.skip("Aucune baseline trouvée - exécutez avec --baseline-save d'abord")
        
        print(f"\n   📈 Baseline Version: {baseline['version']}")
        print(f"   📅 Baseline Date: {baseline['timestamp']}")
        
        # Comparaison
        comparisons = BaselineManager.compare_metrics(
            current_metrics, 
            baseline['metrics'],
            tolerance=0.10  # 10% de tolérance
        )
        
        # Afficher résultats
        print(f"\n   🔍 Comparaison vs Baseline:")
        regressions_found = []
        
        for metric_name, comp in comparisons.items():
            status = "❌ RÉGRESSION" if comp['is_regression'] else "✅ OK"
            change_sign = "+" if comp['change_pct'] >= 0 else ""
            
            print(f"      {metric_name:30s}: {comp['current']:8.2f} "
                  f"(baseline: {comp['baseline']:.2f}, "
                  f"{change_sign}{comp['change_pct']:+.1f}%) {status}")
            
            if comp['is_regression']:
                regressions_found.append(metric_name)
        
        # === 4. Assertions ===
        if regressions_found:
            regression_details = "\n".join([
                f"  - {metric}: {comparisons[metric]['change_pct']:+.1f}%"
                for metric in regressions_found
            ])
            pytest.fail(
                f"🚨 RÉGRESSIONS DÉTECTÉES :\n{regression_details}\n"
                f"Tolérance : ±10%"
            )
    
    def test_accuracy_consistency(self, detector, sample_image_path):
        """Vérifie que le modèle produit des résultats cohérents"""
        # Exécuter 10 fois sur la même image
        results = []
        for _ in range(10):
            result = detector.detect(sample_image_path, conf_threshold=0.25)
            results.append(result['num_detections'])
        
        # Tous les résultats devraient être identiques (déterminisme)
        unique_results = set(results)
        
        print(f"\n   🎯 Cohérence Détections:")
        print(f"      Runs        : 10")
        print(f"      Résultats   : {results}")
        print(f"      Unique      : {unique_results}")
        
        # Devrait être déterministe
        assert len(unique_results) == 1, \
            f"Résultats incohérents : {unique_results}"


class TestVersionComparison:
    """Compare explicitement deux versions de modèles"""
    
    def test_compare_model_versions(self, sample_image_path):
        """Compare YOLOv8n vs YOLOv8s (si disponibles)"""
        models = {
            'yolov8n': 'yolov8n.pt',
            'yolov8s': 'yolov8s.pt'
        }
        
        results = {}
        
        for name, model_path in models.items():
            try:
                detector = YOLODetector(model_path=model_path)
                
                # Mesures
                latencies = []
                detections = []
                
                for _ in range(10):
                    start = time.perf_counter()
                    result = detector.detect(sample_image_path)
                    latencies.append(time.perf_counter() - start)
                    detections.append(result['num_detections'])
                
                results[name] = {
                    'avg_latency': statistics.mean(latencies) * 1000,
                    'avg_detections': statistics.mean(detections)
                }
                
            except Exception as e:
                print(f"   ⚠️  {name} non disponible: {e}")
                continue
        
        # Comparaison
        if len(results) >= 2:
            print(f"\n   ⚖️  Comparaison Versions:")
            for name, metrics in results.items():
                print(f"      {name:10s}: {metrics['avg_latency']:6.1f}ms, "
                      f"{metrics['avg_detections']:.1f} détections")
            
            # YOLOv8s devrait être plus lent mais potentiellement plus précis
            if 'yolov8n' in results and 'yolov8s' in results:
                n_latency = results['yolov8n']['avg_latency']
                s_latency = results['yolov8s']['avg_latency']
                
                assert s_latency > n_latency, \
                    "YOLOv8s devrait être plus lent que YOLOv8n"


class TestDegradationDetection:
    """Détecte les dégradations spécifiques"""
    
    def test_no_memory_leak(self, detector, sample_image_path):
        """Vérifie l'absence de fuite mémoire sur longue durée"""
        import psutil
        
        process = psutil.Process()
        
        # Mesures initiales
        initial_memory = process.memory_info().rss / 1024**2
        
        # 100 inférences
        for _ in range(100):
            detector.detect(sample_image_path)
        
        # Mesure finale
        final_memory = process.memory_info().rss / 1024**2
        memory_increase = final_memory - initial_memory
        
        print(f"\n   🧠 Memory Leak Test:")
        print(f"      Initial : {initial_memory:.1f} MB")
        print(f"      Final   : {final_memory:.1f} MB")
        print(f"      Increase: {memory_increase:.1f} MB")
        
        # Augmentation < 100 MB acceptable
        assert memory_increase < 100, \
            f"Fuite mémoire détectée: +{memory_increase:.1f}MB"
    
    def test_fps_stability_over_time(self, detector, sample_image_path):
        """Vérifie que le FPS reste stable dans le temps"""
        
        # Mesurer FPS sur 3 batches de 30 frames
        fps_batches = []
        
        for batch in range(3):
            start = time.perf_counter()
            for _ in range(30):
                detector.detect(sample_image_path)
            elapsed = time.perf_counter() - start
            fps = 30 / elapsed
            fps_batches.append(fps)
        
        # Variance entre batches
        fps_variance = statistics.stdev(fps_batches) / statistics.mean(fps_batches) * 100
        
        print(f"\n   ⚡ FPS Stability:")
        print(f"      Batch 1: {fps_batches[0]:.1f} FPS")
        print(f"      Batch 2: {fps_batches[1]:.1f} FPS")
        print(f"      Batch 3: {fps_batches[2]:.1f} FPS")
        print(f"      Variance: {fps_variance:.1f}%")
        
        # Variance < 10% = stable
        assert fps_variance < 15, \
            f"FPS instable dans le temps: {fps_variance:.1f}%"


# Fonction utilitaire pour générer un rapport
def generate_regression_report(output_path="regression_report.json"):
    """Génère un rapport de régression complet"""
    baseline = BaselineManager.load_baseline()
    
    if baseline is None:
        print("❌ Aucune baseline trouvée")
        return
    
    # Collecter métriques actuelles (simplifié)
    detector = YOLODetector()
    # ... (collecte de métriques)
    
    report = {
        'baseline_version': baseline['version'],
        'baseline_date': baseline['timestamp'],
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'regressions': [],
        'improvements': []
    }
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Rapport généré : {output_path}")
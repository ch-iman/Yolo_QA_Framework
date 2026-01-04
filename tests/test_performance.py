"""
Tests de Performance pour YOLO
===============================

Mesure les performances de latence, CPU/GPU, FPS et comparaison entre modèles.

Exécution : pytest tests/test_performance.py -v -s
Durée attendue : 2-5 minutes

Prérequis :
    pip install psutil py-cpuinfo pytest-benchmark
"""

import pytest
import time
import psutil
import platform
from pathlib import Path
from src.yolo_detector import YOLODetector
import statistics
import json


class TestLatencyPerformance:
    """Tests de latence détaillés"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        return YOLODetector()
    
    def test_cold_start_latency(self, detector, sample_image_path):
        """Mesure le temps de première inférence (cold start)"""
        # Créer un nouveau détecteur pour simuler cold start
        fresh_detector = YOLODetector()
        
        start = time.perf_counter()
        result = fresh_detector.detect(sample_image_path)
        cold_start_time = time.perf_counter() - start
        
        print(f"\n   🥶 Cold start: {cold_start_time*1000:.2f}ms")
        
        # Cold start devrait être < 2 secondes
        assert cold_start_time < 2.0, \
            f"Cold start trop lent: {cold_start_time:.2f}s"
    
    def test_warm_inference_latency(self, detector, sample_image_path):
        """Mesure la latence en inférence chaude (après warm-up)"""
        # Warm-up : 5 inférences
        for _ in range(5):
            detector.detect(sample_image_path, conf_threshold=0.25)
        
        # Mesurer sur 20 inférences
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            detector.detect(sample_image_path, conf_threshold=0.25)
            latencies.append(time.perf_counter() - start)
        
        # Statistiques
        avg_lat = statistics.mean(latencies)
        p50_lat = statistics.median(latencies)
        p95_lat = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        p99_lat = max(latencies)
        
        print(f"\n   📊 Latence (20 runs):")
        print(f"      Moyenne : {avg_lat*1000:.2f}ms")
        print(f"      P50     : {p50_lat*1000:.2f}ms")
        print(f"      P95     : {p95_lat*1000:.2f}ms")
        print(f"      P99     : {p99_lat*1000:.2f}ms")
        
        # Assertions
        assert avg_lat < 0.3, f"Latence moyenne trop élevée: {avg_lat*1000:.2f}ms"
        assert p95_lat < 0.5, f"P95 latence trop élevée: {p95_lat*1000:.2f}ms"
    
    def test_latency_variance(self, detector, sample_image_path):
        """Vérifie la variance de latence (stabilité)"""
        latencies = []
        for _ in range(30):
            start = time.perf_counter()
            detector.detect(sample_image_path)
            latencies.append(time.perf_counter() - start)
        
        avg = statistics.mean(latencies)
        stdev = statistics.stdev(latencies)
        cv = (stdev / avg) * 100  # Coefficient de variation
        
        print(f"\n   📈 Variance:")
        print(f"      Moyenne : {avg*1000:.2f}ms")
        print(f"      Std Dev : {stdev*1000:.2f}ms")
        print(f"      CV      : {cv:.1f}%")
        
        # Coefficient de variation < 15% = stable
        assert cv < 15, f"Latence instable (CV={cv:.1f}%)"


class TestResourceUsage:
    """Tests de consommation CPU/GPU"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        return YOLODetector()
    
    def test_cpu_usage_during_inference(self, detector, sample_image_path):
        """Mesure la consommation CPU pendant l'inférence"""
        # Baseline CPU avant inférence
        psutil.cpu_percent(interval=1)  # Reset
        
        # Mesurer CPU pendant inférences
        cpu_samples = []
        for _ in range(10):
            cpu_before = psutil.cpu_percent(interval=None)
            detector.detect(sample_image_path)
            cpu_after = psutil.cpu_percent(interval=0.1)
            cpu_samples.append(cpu_after)
        
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        print(f"\n   💻 CPU Usage:")
        print(f"      Moyenne : {avg_cpu:.1f}%")
        print(f"      Maximum : {max_cpu:.1f}%")
        
        # CPU ne devrait pas saturer à 100%
        assert max_cpu < 95, f"CPU saturé: {max_cpu:.1f}%"
    
    def test_memory_usage(self, detector, dataset_normal_path):
        """Mesure la consommation mémoire"""
        process = psutil.Process()
        
        # Mémoire avant
        mem_before = process.memory_info().rss / 1024**2  # MB
        
        # Traiter dataset
        results = detector.detect_on_dataset(dataset_normal_path, verbose=False)
        
        # Mémoire après
        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_increase = mem_after - mem_before
        
        print(f"\n   🧠 Memory Usage:")
        print(f"      Avant   : {mem_before:.1f} MB")
        print(f"      Après   : {mem_after:.1f} MB")
        print(f"      Augment.: {mem_increase:.1f} MB")
        
        # Augmentation < 500 MB acceptable
        assert mem_increase < 500, \
            f"Fuite mémoire potentielle: +{mem_increase:.1f}MB"
    
    def test_gpu_availability_check(self):
        """Vérifie si GPU est disponible et utilisé"""
        import torch
        
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"\n   🎮 GPU Détecté:")
            print(f"      Nom     : {gpu_name}")
            print(f"      Mémoire : {gpu_memory:.1f} GB")
        else:
            print(f"\n   ⚠️  Aucun GPU détecté - utilisation CPU")
        
        # Pas d'assertion - juste informatif


class TestFPSRealTime:
    """Tests de FPS en temps réel"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        return YOLODetector()
    
    def test_sustained_fps(self, detector, sample_image_path):
        """Mesure le FPS soutenu sur 100 frames"""
        num_frames = 100
        
        start_time = time.perf_counter()
        for _ in range(num_frames):
            detector.detect(sample_image_path, conf_threshold=0.25)
        total_time = time.perf_counter() - start_time
        
        fps = num_frames / total_time
        
        print(f"\n   🎬 FPS Performance:")
        print(f"      Frames  : {num_frames}")
        print(f"      Durée   : {total_time:.2f}s")
        print(f"      FPS     : {fps:.1f}")
        
        # Sur CPU, on attend au moins 3 FPS
        assert fps >= 3.0, f"FPS trop bas: {fps:.1f} (attendu >= 3)"
    
    def test_batch_vs_single_inference(self, detector, dataset_normal_path):
        """Compare FPS batch vs image par image"""
        from PIL import Image
        import glob
        
        # Charger quelques images
        image_files = list(Path(dataset_normal_path).glob("*.jpg"))[:10]
        
        # Test 1 : Une par une
        start = time.perf_counter()
        for img_path in image_files:
            detector.detect(str(img_path))
        time_single = time.perf_counter() - start
        
        # Test 2 : Via detect_on_dataset (optimisé)
        start = time.perf_counter()
        detector.detect_on_dataset(dataset_normal_path, verbose=False)
        time_batch = time.perf_counter() - start
        
        fps_single = len(image_files) / time_single
        fps_batch = len(image_files) / time_batch
        speedup = fps_batch / fps_single
        
        print(f"\n   ⚡ Batch vs Single:")
        print(f"      Single : {fps_single:.1f} FPS")
        print(f"      Batch  : {fps_batch:.1f} FPS")
        print(f"      Speedup: {speedup:.2f}x")


class TestModelBenchmarking:
    """Comparaison entre différents modèles YOLO"""
    
    def test_yolov8_model_variants(self, sample_image_path):
        """Compare YOLOv8n, YOLOv8s, YOLOv8m"""
        models = ['yolov8n.pt', 'yolov8s.pt']  # Modèles légers pour test
        
        results = {}
        
        for model_name in models:
            try:
                detector = YOLODetector(model_path=model_name)
                
                # Warm-up
                detector.detect(sample_image_path)
                
                # Benchmark
                latencies = []
                for _ in range(10):
                    start = time.perf_counter()
                    result = detector.detect(sample_image_path)
                    latencies.append(time.perf_counter() - start)
                
                results[model_name] = {
                    'avg_latency': statistics.mean(latencies),
                    'p95_latency': statistics.quantiles(latencies, n=20)[18],
                    'num_detections': result['num_detections']
                }
                
            except Exception as e:
                print(f"   ⚠️  {model_name} non disponible: {e}")
                continue
        
        # Afficher comparaison
        print(f"\n   🏆 Comparaison Modèles:")
        for model, metrics in results.items():
            print(f"      {model:15s}: {metrics['avg_latency']*1000:6.1f}ms "
                  f"(P95: {metrics['p95_latency']*1000:.1f}ms, "
                  f"Détections: {metrics['num_detections']})")
        
        # Au moins 1 modèle testé
        assert len(results) > 0, "Aucun modèle n'a pu être testé"


class TestSystemInfo:
    """Collecte des informations système pour contexte"""
    
    def test_collect_system_info(self, tmp_path):
        """Collecte et sauvegarde les infos système"""
        import torch
        import cpuinfo
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'ram_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        }
        
        # Sauvegarder
        output_file = tmp_path / "system_info.json"
        with open(output_file, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n   💾 System Info:")
        for key, value in info.items():
            print(f"      {key:20s}: {value}")
        
        assert output_file.exists()


# Fixture pytest-benchmark (optionnel)
@pytest.fixture
def benchmark_detector(benchmark):
    """Fixture pour benchmarking avec pytest-benchmark"""
    detector = YOLODetector()
    return detector


def test_benchmark_inference(benchmark_detector, benchmark, sample_image_path):
    """Benchmark avec pytest-benchmark (statistiques avancées)"""
    # Utilisation: pytest tests/test_performance.py::test_benchmark_inference --benchmark-only
    result = benchmark(benchmark_detector.detect, sample_image_path)
    assert result is not None
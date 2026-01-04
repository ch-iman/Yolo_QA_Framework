import pytest
from pathlib import Path

class TestSetup:
    """Tests de base installation"""
    
    def test_imports(self):
        """Librairies installées"""
        import torch
        import cv2
        from ultralytics import YOLO
        assert True
    
    def test_project_structure(self):
        """Structure du projet correcte"""
        assert Path('src').exists()
        assert Path('tests').exists()
        assert Path('data/images/normal').exists()

class TestYOLOBasics:
    """Tests YOLO de base"""
    
    def test_model_file_exists(self):
        """Modèle YOLO existe"""
        model_paths = [
            'models/yolov8n.pt',
            '../models/yolov8n.pt'
        ]
        exists = any(Path(p).exists() for p in model_paths)
        assert exists, "Modèle non trouvé. Téléchargez-le avec: python scripts/download_models.py"
    
    def test_model_loads(self):
        """Modèle se charge"""
        from src.yolo_detector import YOLODetector
        detector = YOLODetector()
        assert detector.model is not None

class TestDatasetMinimal:
    
    def test_at_least_one_image_exists(self):
        """Au moins 1 image de test existe"""
        normal_dir = Path('data/images/normal')
        images = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
        
        assert len(images) >= 1, \
            "Aucune image trouvée. Ajoutez des images dans 'data/images/normal'."
    
    def test_can_load_image(self):
        """OpenCV peut charger l'image"""
        import cv2
        normal_dir = Path('data/images/normal')
        images = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
        
        img = cv2.imread(str(images[0]))
        assert img is not None
        assert len(img.shape) == 3
    
    def test_detection_runs(self):
        """Détection fonctionne sans crash"""
        from src.yolo_detector import YOLODetector
        
        normal_dir = Path('data/images/normal')
        images = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
        
        detector = YOLODetector()
        result = detector.detect(str(images[0]))
        
        assert result is not None
        assert 'detections' in result
        assert 'inference_time' in result
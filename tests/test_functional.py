"""
Tests fonctionnels pour valider le comportement de YOLO sur dataset.

Ces tests vérifient que :
- Les détections sont correctes sur images individuelles
- Le traitement de dataset complet fonctionne
- Les edge cases sont gérés correctement
- Les résultats peuvent être sauvegardés

Exécution : pytest tests/test_functional.py -v
Durée attendue : 10-30 secondes (selon taille dataset)
"""

import pytest
from pathlib import Path
from src.yolo_detector import YOLODetector
import json


class TestSingleImageDetection:
    """Tests sur image unique (basiques)"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture : crée un détecteur une fois pour toute la classe"""
        return YOLODetector()
    
    def test_detection_on_known_image(self, detector, sample_image_path):
        """Détecte des objets sur une image connue"""
        result = detector.detect(sample_image_path)
        
        # Doit détecter au moins quelque chose (peut être 0 sur certaines images)
        assert result['num_detections'] >= 0, "num_detections doit être >= 0"
        
        # Si détections présentes, vérifier leur format
        if result['num_detections'] > 0:
            det = result['detections'][0]
            assert 'class_name' in det, "Détection doit avoir 'class_name'"
            assert 'confidence' in det, "Détection doit avoir 'confidence'"
            assert 'bbox' in det, "Détection doit avoir 'bbox'"
            assert len(det['bbox']) == 4, "bbox doit avoir 4 coordonnées [x1,y1,x2,y2]"
    
    def test_detection_result_format(self, detector, sample_image_path):
        """Vérifie le format complet du résultat"""
        result = detector.detect(sample_image_path)
        
        # Clés requises
        required_keys = ['image_path', 'detections', 'inference_time', 
                        'image_shape', 'num_detections']
        for key in required_keys:
            assert key in result, f"Clé '{key}' manquante dans le résultat"
        
        # Types corrects
        assert isinstance(result['detections'], list)
        assert isinstance(result['inference_time'], float)
        assert isinstance(result['num_detections'], int)
        assert result['inference_time'] > 0, "Temps d'inférence doit être > 0"
    
    def test_confidence_threshold_works(self, detector, sample_image_path):
        """Vérifie que le seuil de confiance fonctionne"""
        # Seuil bas
        result_low = detector.detect(sample_image_path, conf_threshold=0.1)
        
        # Seuil haut
        result_high = detector.detect(sample_image_path, conf_threshold=0.8)
        
        # Seuil bas doit donner >= détections que seuil haut
        assert result_low['num_detections'] >= result_high['num_detections'], \
            "Seuil 0.1 devrait donner plus ou autant de détections que 0.8"
        
        # Vérifier que toutes les détections respectent le seuil haut
        for det in result_high['detections']:
            assert det['confidence'] >= 0.8, \
                f"Détection avec confidence {det['confidence']} < 0.8"


class TestDatasetDetection:
    """Tests sur dataset complet"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        """Fixture : détecteur pour toute la classe"""
        return YOLODetector()
    
    def test_detection_on_normal_dataset(self, detector, dataset_normal_path):
        """Test sur TOUTES les images normales"""
        results = detector.detect_on_dataset(dataset_normal_path, verbose=False)
        
        # Au moins 1 image traitée
        assert len(results) > 0, "Aucune image traitée dans le dataset normal"
        
        # Toutes les images doivent avoir un résultat valide
        for result in results:
            assert 'image_path' in result
            assert 'num_detections' in result
            assert 'inference_time' in result
        
        # Au moins 80% des images normales devraient avoir des détections
        images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
        detection_rate = images_with_detections / len(results)
        
        assert detection_rate >= 0.8, \
            f"Seulement {detection_rate*100:.1f}% des images ont des détections (attendu >= 80%)"
        
        print(f"\n   ✅ {len(results)} images traitées")
        print(f"   ✅ {images_with_detections}/{len(results)} avec détections ({detection_rate*100:.1f}%)")
    
    def test_detection_on_edge_cases_dataset(self, detector, dataset_edge_cases_path):
        """Test sur edge cases (images dégradées)"""
        results = detector.detect_on_dataset(dataset_edge_cases_path, verbose=False)
        
        # Au moins quelques images traitées
        assert len(results) > 0, "Aucune image edge case traitée"
        
        # Même sur edge cases, on attend QUELQUES détections (pas 0 partout)
        total_detections = sum(r['num_detections'] for r in results)
        assert total_detections > 0, \
            "Aucune détection sur AUCUN edge case (trop sévère)"
        
        # Mais on accepte que certaines images n'aient pas de détections
        images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
        detection_rate = images_with_detections / len(results)
        
        print(f"\n   ✅ {len(results)} edge cases traitées")
        print(f"   ✅ {total_detections} détections au total")
        print(f"   ⚠️  {detection_rate*100:.1f}% avec détections (dégradation attendue)")
    
    def test_inference_time_reasonable(self, detector, dataset_normal_path):
        """Vérifie que les temps d'inférence sont raisonnables"""
        results = detector.detect_on_dataset(dataset_normal_path, verbose=False)
        
        # Calculer temps moyen
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        max_time = max(r['inference_time'] for r in results)
        
        # Sur CPU, on accepte jusqu'à 500ms par image
        assert avg_time < 0.5, \
            f"Temps moyen trop élevé : {avg_time*1000:.2f}ms (max attendu: 500ms)"
        
        assert max_time < 1.0, \
            f"Temps maximum trop élevé : {max_time*1000:.2f}ms (max attendu: 1000ms)"
        
        print(f"\n   ⏱️  Temps moyen : {avg_time*1000:.2f}ms")
        print(f"   ⏱️  Temps max : {max_time*1000:.2f}ms")


class TestDatasetByCategory:
    """Tests de traitement par catégories"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        return YOLODetector()
    
    def test_detect_by_category(self, detector, project_paths):
        """Test traitement par catégories (normal + edge_cases)"""
        base_dir = project_paths['data'] / 'images'
        
        results_by_cat = detector.detect_on_dataset_by_category(str(base_dir))
        
        # Au moins 1 catégorie trouvée
        assert len(results_by_cat) > 0, "Aucune catégorie trouvée"
        
        # Catégorie "normal" doit exister
        assert 'normal' in results_by_cat, "Catégorie 'normal' manquante"
        assert len(results_by_cat['normal']) > 0, "Aucune image dans 'normal'"
        
        # Afficher résumé
        print(f"\n   📂 Catégories trouvées : {list(results_by_cat.keys())}")
        for cat, results in results_by_cat.items():
            total = sum(r['num_detections'] for r in results)
            print(f"   ✅ {cat:15s} : {len(results)} images, {total} détections")
    
    def test_compare_normal_vs_edge_cases(self, detector, project_paths):
        """Compare les détections normal vs edge cases"""
        base_dir = project_paths['data'] / 'images'
        results_by_cat = detector.detect_on_dataset_by_category(str(base_dir))
        
        # Calculer taux de détection
        def detection_rate(results):
            if not results:
                return 0
            return sum(r['num_detections'] for r in results) / len(results)
        
        normal_rate = detection_rate(results_by_cat.get('normal', []))
        edge_rate = detection_rate(results_by_cat.get('edge_cases', []))
        
        # Edge cases devraient avoir moins de détections (dégradation)
        if 'edge_cases' in results_by_cat and normal_rate > 0:
            degradation = (normal_rate - edge_rate) / normal_rate * 100
            
            print(f"\n   📊 Détections par image :")
            print(f"      Normal     : {normal_rate:.2f}")
            print(f"      Edge cases : {edge_rate:.2f}")
            print(f"      Dégradation: {degradation:.1f}%")
            
            # On accepte jusqu'à 50% de dégradation sur edge cases
            assert degradation <= 50, \
                f"Dégradation trop importante : {degradation:.1f}%"


class TestResultsSaving:
    """Tests de sauvegarde des résultats"""
    
    @pytest.fixture(scope="class")
    def detector(self):
        return YOLODetector()
    
    def test_save_results_creates_file(self, detector, dataset_normal_path, tmp_path):
        """Vérifie que save_results() crée un fichier"""
        results = detector.detect_on_dataset(dataset_normal_path, verbose=False)
        
        output_file = tmp_path / "test_results.json"
        detector.save_results(results, str(output_file))
        
        # Fichier créé
        assert output_file.exists(), "Fichier de résultats non créé"
        
        # Fichier non vide
        assert output_file.stat().st_size > 0, "Fichier de résultats vide"
    
    def test_save_results_json_format(self, detector, dataset_normal_path, tmp_path):
        """Vérifie le format JSON des résultats sauvegardés"""
        results = detector.detect_on_dataset(dataset_normal_path, verbose=False)
        
        output_file = tmp_path / "test_results.json"
        detector.save_results(results, str(output_file))
        
        # Charger et vérifier le JSON
        with open(output_file) as f:
            data = json.load(f)
        
        # Clés requises
        assert 'model' in data, "Clé 'model' manquante"
        assert 'total_images' in data, "Clé 'total_images' manquante"
        assert 'total_detections' in data, "Clé 'total_detections' manquante"
        assert 'avg_inference_time' in data, "Clé 'avg_inference_time' manquante"
        assert 'results' in data, "Clé 'results' manquante"
        
        # Valeurs cohérentes
        assert data['total_images'] == len(results)
        assert len(data['results']) == len(results)
        
        print(f"\n   ✅ JSON valide avec {data['total_images']} images")
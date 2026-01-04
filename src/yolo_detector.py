# src/yolo_detector.py

from ultralytics import YOLO
import cv2
import time
from typing import List, Dict, Tuple
import numpy as np
from pathlib import Path
import json

class YOLODetector:
    """
    Classe principale pour gérer la détection d'objets avec YOLO.
    Supporte détection sur image unique ou dataset complet.
    """
    
    def __init__(self, model_path: str = 'models/yolov8n.pt'):
        """
        Initialise le détecteur YOLO.
        
        Args:
            model_path: Chemin vers le modèle YOLO (.pt)
        """
        self.model = YOLO(model_path)
        self.model_path = model_path
        self.inference_times = []
        
    def detect(self, image_path: str, conf_threshold: float = 0.25) -> Dict:
        """
        Détecte les objets dans UNE image.
        
        Args:
            image_path: Chemin vers l'image
            conf_threshold: Seuil de confiance minimum
            
        Returns:
            Dict avec détections, temps, et métadonnées
        """
        # Charger l'image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Impossible de charger l'image: {image_path}")
        
        # Mesurer le temps d'inférence
        start_time = time.time()
        results = self.model(img, conf=conf_threshold)[0]
        inference_time = time.time() - start_time
        
        self.inference_times.append(inference_time)
        
        # Extraire les détections
        detections = []
        for box in results.boxes:
            detection = {
                'class_id': int(box.cls[0]),
                'class_name': results.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
            }
            detections.append(detection)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'inference_time': inference_time,
            'image_shape': img.shape,
            'num_detections': len(detections)
        }
    
    # ════════════════════════════════════════════════════════════
    # NOUVELLES MÉTHODES POUR DATASET COMPLET
    # ════════════════════════════════════════════════════════════
    
    def detect_on_dataset(self, 
                         dataset_dir: str, 
                         conf_threshold: float = 0.25,
                         recursive: bool = False,
                         verbose: bool = True) -> List[Dict]:
        """
        Détecte les objets sur TOUTES les images d'un dossier.
        
        Args:
            dataset_dir: Chemin vers le dossier d'images
            conf_threshold: Seuil de confiance
            recursive: Chercher dans sous-dossiers aussi ?
            verbose: Afficher progression ?
            
        Returns:
            Liste de résultats (1 dict par image)
            
        Example:e
            detector = YOLODetector()
            results = detector.detect_on_dataset('data/images/normal')
            print(f"Traité {len(results)} images")
        """
        dataset_path = Path(dataset_dir)
        
        if not dataset_path.exists():
            raise ValueError(f"Dossier introuvable : {dataset_dir}")
        
        # Trouver toutes les images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            if recursive:
                image_files.extend(dataset_path.rglob(ext))
            else:
                image_files.extend(dataset_path.glob(ext))
        
        if len(image_files) == 0:
            print(f"⚠️  Aucune image trouvée dans {dataset_dir}")
            return []
        
        if verbose:
            print(f"\n📊 Traitement de {len(image_files)} images...")
            print(f"📁 Dossier : {dataset_dir}")
            print(f"🎯 Seuil de confiance : {conf_threshold}\n")
        
        # Traiter chaque image
        results = []
        for i, img_path in enumerate(image_files, 1):
            try:
                result = self.detect(str(img_path), conf_threshold)
                results.append(result)
                
                if verbose:
                    status = f"[{i}/{len(image_files)}]"
                    detections = result['num_detections']
                    time_ms = result['inference_time'] * 1000
                    print(f"{status} {img_path.name:30s} → "
                          f"{detections:2d} objets ({time_ms:6.2f}ms)")
                    
            except Exception as e:
                print(f"❌ Erreur sur {img_path.name}: {e}")
                continue
        
        if verbose:
            self._print_dataset_summary(results)
        
        return results
    
    def detect_on_dataset_by_category(self, 
                                      base_dir: str = 'data/images',
                                      conf_threshold: float = 0.25) -> Dict:
        """
        Détecte sur dataset organisé en catégories (normal, edge_cases).
        
        Args:
            base_dir: Dossier racine (contient normal/, edge_cases/, etc.)
            conf_threshold: Seuil de confiance
            
        Returns:
            Dict avec résultats par catégorie
            
        Example:
            results = detector.detect_on_dataset_by_category()
            print(f"Normal: {len(results['normal'])} images")
            print(f"Edge cases: {len(results['edge_cases'])} images")
        """
        base_path = Path(base_dir)
        results_by_category = {}
        
        # Détecter les catégories disponibles
        categories = [d for d in base_path.iterdir() if d.is_dir()]
        
        print(f"\n📂 Traitement par catégorie...")
        print(f"   Base : {base_dir}")
        print(f"   Catégories trouvées : {len(categories)}\n")
        
        for category_dir in categories:
            category_name = category_dir.name
            print(f"\n{'='*60}")
            print(f"📁 CATÉGORIE : {category_name.upper()}")
            print(f"{'='*60}")
            
            results = self.detect_on_dataset(
                str(category_dir), 
                conf_threshold=conf_threshold,
                verbose=True
            )
            
            results_by_category[category_name] = results
        
        # Résumé global
        print(f"\n{'='*60}")
        print(f"📊 RÉSUMÉ GLOBAL")
        print(f"{'='*60}")
        for cat, res in results_by_category.items():
            total_detections = sum(r['num_detections'] for r in res)
            avg_time = sum(r['inference_time'] for r in res) / len(res) if res else 0
            print(f"{cat:20s} : {len(res):3d} images, "
                  f"{total_detections:4d} détections, "
                  f"moy={avg_time*1000:.2f}ms")
        
        return results_by_category
    
    def save_results(self, results: List[Dict], output_path: str):
        """
        Sauvegarde les résultats au format JSON.
        
        Args:
            results: Liste de résultats (de detect_on_dataset)
            output_path: Chemin du fichier JSON de sortie
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculer statistiques
        summary = {
            'model': self.model_path,
            'total_images': len(results),
            'total_detections': sum(r['num_detections'] for r in results),
            'avg_inference_time': self.get_average_inference_time(),
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✅ Résultats sauvegardés : {output_path}")
        print(f"   - {summary['total_images']} images traitées")
        print(f"   - {summary['total_detections']} détections au total")
    
    def _print_dataset_summary(self, results: List[Dict]):
        """Affiche un résumé des résultats sur le dataset."""
        if not results:
            return
        
        total_detections = sum(r['num_detections'] for r in results)
        avg_detections = total_detections / len(results)
        avg_time = sum(r['inference_time'] for r in results) / len(results)
        min_time = min(r['inference_time'] for r in results)
        max_time = max(r['inference_time'] for r in results)
        
        print(f"\n{'─'*60}")
        print(f"📊 RÉSUMÉ")
        print(f"{'─'*60}")
        print(f"Images traitées       : {len(results)}")
        print(f"Détections totales    : {total_detections}")
        print(f"Détections par image  : {avg_detections:.2f}")
        print(f"Temps moyen           : {avg_time*1000:.2f}ms")
        print(f"Temps min/max         : {min_time*1000:.2f}ms / {max_time*1000:.2f}ms")
        print(f"{'─'*60}\n")
    
    # ════════════════════════════════════════════════════════════
    # MÉTHODES EXISTANTES (inchangées)
    # ════════════════════════════════════════════════════════════
    
    def get_average_inference_time(self) -> float:
        """Retourne le temps d'inférence moyen."""
        if not self.inference_times:
            return 0.0
        return sum(self.inference_times) / len(self.inference_times)
    
    def visualize_detections(self, image_path: str, save_path: str = None):
        """Visualise les détections sur une image."""
        img = cv2.imread(image_path)
        result = self.detect(image_path)
        if save_path:
         Path(save_path).parent.mkdir(parents=True, exist_ok=True)  # ← AJOUTER CETTE LIGNE
         cv2.imwrite(save_path, img)
        
        for det in result['detections']:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, img)
            print(f"✓ Image sauvegardée : {save_path}")
        else:
            cv2.imshow('Detections', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def visualize_dataset(self, 
                         dataset_dir: str, 
                         output_dir: str = 'res_images',
                         conf_threshold: float = 0.25,
                         max_images: int = None):
        """
        Visualise TOUTES les images d'un dataset.
        
        Args:
            dataset_dir: Dossier contenant les images
            output_dir: Dossier où sauvegarder les images annotées
            conf_threshold: Seuil de confiance
            max_images: Limite du nombre d'images (None = toutes)
        """
        results = self.detect_on_dataset(dataset_dir, conf_threshold, verbose=False)
        
        if not results:
            print("⚠️  Aucune image à visualiser")
            return
        
        if max_images:
            results = results[:max_images]
        
        print(f"\n🎨 Génération des visualisations...")
        print(f"📁 Sortie : {output_dir}")
        print(f"📊 {len(results)} images à traiter\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results, 1):
            input_path = result['image_path']
            filename = Path(input_path).stem
            output_file = output_path / f"{filename}_result.jpg"
            
            img = cv2.imread(input_path)
            
            for det in result['detections']:
                x1, y1, x2, y2 = [int(c) for c in det['bbox']]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']} {det['confidence']:.2f}"
                cv2.putText(img, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imwrite(str(output_file), img)
            print(f"[{i}/{len(results)}] ✅ {output_file.name:40s} "
                  f"({result['num_detections']} objets)")
        
        print(f"\n✅ {len(results)} images annotées sauvegardées dans {output_dir}/")        


# ════════════════════════════════════════════════════════════
# TESTS MANUELS (Section améliorée)
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("="*60)
    print("🔍 TEST MANUEL - YOLODetector avec Dataset")
    print("="*60)
    
    # Créer le détecteur
    try:
        detector = YOLODetector()
        print("\n✅ Modèle chargé avec succès")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)
    
    # ──────────────────────────────────────────────────
    # TEST 1 : Une seule image (comme avant)
    # ──────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("TEST 1 : Détection sur une image")
    print("─"*60)
    
    test_images = [
        "data/images/normal/cats_sofa.jpg",
        "../data/images/normal/dog.jpeg",
    ]
    
    test_image = "data/images/normal/cats_sofa.jpg"  # Par défaut
    for img_path in test_images:
        if Path(img_path).exists():
            test_image = img_path
            break
    
    if test_image:
        result = detector.detect(test_image)
        print(f"📷 Image : {Path(test_image).name}")
        print(f"✅ Détections : {result['num_detections']}")
        print(f"⏱️  Temps : {result['inference_time']*1000:.2f}ms")
        detector.save_results([result], 'reports/one_image.json')
        detector.visualize_detections(test_image,'res_images/cats_sofa_result.jpg')
    else:
        print("❌ Aucune image de test trouvée. Veuillez vérifier le chemin.")
    # ──────────────────────────────────────────────────
    # TEST 2 : Dataset complet (NOUVEAU)
    # ──────────────────────────────────────────────────
    print("\n" + "─"*60)
    print("TEST 2 : Détection sur dataset complet")
    print("─"*60)
    
    # Option A : Un seul dossier
    if Path('data/images/normal').exists():
        results = detector.detect_on_dataset('data/images/normal')
        
        # Sauvegarder les résultats
        detector.save_results(results, 'reports/normal_results.json')
        # Visualiser tout le dataset
        print("\n" + "─"*60)
        print("Génération des visualisations pour le dataset normal")
        print("─"*60)
        detector.visualize_dataset('data/images/normal', 'res_images/normal')
    
    # Option B : Par catégories (normal + edge_cases)
    if Path('data/images').exists():
        print("\n" + "="*60)
        print("TEST 3 : Traitement par catégories")
        print("="*60)
        
        all_results = detector.detect_on_dataset_by_category('data/images')
        
        # Sauvegarder résultats par catégorie
        for category, results in all_results.items():
            output_file = f'reports/{category}_results.json'
            detector.save_results(results, output_file)
            # Générer visualisations
            print(f"\n🎨 Génération visualisations pour catégorie : {category}")
            detector.visualize_dataset(
            f'data/images/{category}', 
            f'res_images/{category}'
          )
    
    print("\n" + "="*60)
    print("✅ TOUS LES TESTS TERMINÉS")
    print("="*60)
    print("\n📂 Vérifiez les dossiers :")
    print("   - reports/        → Résultats JSON")
    print("   - res_images/     → Images annotées")
    print("="*60)
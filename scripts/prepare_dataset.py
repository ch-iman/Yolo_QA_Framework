import os
import requests
from pathlib import Path
import cv2
import numpy as np

def download_coco_samples():
    """
    Télécharge 5 images COCO variées pour tests
    """
    # Images COCO sélectionnées avec soin
    samples = {
        'person_kitchen.jpg': 'http://images.cocodataset.org/val2017/000000397133.jpg',
        'cats_sofa.jpg': 'http://images.cocodataset.org/val2017/000000039769.jpg',
        'kitchen.jpg': 'http://images.cocodataset.org/val2017/000000037777.jpg',
        'toilet.jpg': 'http://images.cocodataset.org/val2017/000000006818.jpg',
        'washbasin.jpg': 'http://images.cocodataset.org/val2017/000000104572.jpg',
    }
    
    # Chemin absolu basé sur l'emplacement du script
    script_dir = Path(__file__).parent.resolve()  # Répertoire du script
    output_dir = script_dir.parent / 'data' / 'images' / 'normal'  # Remonte d'un niveau
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Images seront sauvegardées dans :\n   {output_dir}\n")
    print("📥 Téléchargement des images COCO...")
    
    for name, url in samples.items():
        output_path = output_dir / name
        
        if output_path.exists():
            print(f"  ⏭️  {name} existe déjà")
            continue
            
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ Erreur {response.status_code} pour {name}")
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
    
    print(f"\n✓ {len(list(output_dir.glob('*.jpg')))} images téléchargées\n")

def create_edge_cases():
    """
    POURQUOI chaque transformation :
    - Flou : Caméra en mouvement, défocalisation
    - Sombre : Mauvais éclairage, nuit
    - Bruit : Capteur bas de gamme, interférences
    - Basse résolution : Caméras anciennes, compression
    """
    # Chemins absolus
    script_dir = Path(__file__).parent.resolve()
    normal_dir = script_dir.parent / 'data' / 'images' / 'normal'
    edge_dir = script_dir.parent / 'data' / 'images' / 'edge_cases'
    edge_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Edge cases seront sauvegardés dans :\n   {edge_dir}\n")
    print("🔧 Création des cas limites...")
    
    for img_path in normal_dir.glob('*.jpg'):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        # 1. FLOU - Simule mouvement
        blurred = cv2.GaussianBlur(img, (21, 21), 0)
        cv2.imwrite(str(edge_dir / f'blur_{img_path.name}'), blurred)
        
        # 2. SOMBRE - Simule faible éclairage
        dark = cv2.convertScaleAbs(img, alpha=0.3, beta=0)
        cv2.imwrite(str(edge_dir / f'dark_{img_path.name}'), dark)
        
        # 3. BRUIT - Simule capteur bas de gamme
        noise = np.random.normal(0, 30, img.shape).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.imwrite(str(edge_dir / f'noisy_{img_path.name}'), noisy)
        
        # 4. BASSE RÉSOLUTION - Simule vieille caméra
        h, w = img.shape[:2]
        small = cv2.resize(img, (w//4, h//4))
        lowres = cv2.resize(small, (w, h))  # Upscale = pixelisé
        cv2.imwrite(str(edge_dir / f'lowres_{img_path.name}'), lowres)
        
        # 5. ROTATION - Simule caméra mal orientée
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
        rotated = cv2.warpAffine(img, matrix, (w, h))
        cv2.imwrite(str(edge_dir / f'rotated_{img_path.name}'), rotated)
    
    num_edge = len(list(edge_dir.glob('*.jpg')))
    print(f"✓ {num_edge} images de cas limites créées\n")

def verify_dataset():
    """
    Vérifie que le dataset est complet
    """
    script_dir = Path(__file__).parent.resolve()
    normal_dir = script_dir.parent / 'data' / 'images' / 'normal'
    edge_dir = script_dir.parent / 'data' / 'images' / 'edge_cases'
    
    normal_count = len(list(normal_dir.glob('*.jpg')))
    edge_count = len(list(edge_dir.glob('*.jpg')))
    
    print("📊 État du dataset :")
    print(f"  Normal : {normal_count} images")
    print(f"  Edge cases : {edge_count} images")
    print(f"  Total : {normal_count + edge_count} images")
    print(f"\n📂 Chemins complets :")
    print(f"  Normal: {normal_dir}")
    print(f"  Edge cases: {edge_dir}")
    
    if normal_count == 0:
        print("\n⚠️  ATTENTION : Aucune image normale trouvée !")
        print("   Exécutez d'abord download_coco_samples()")

if __name__ == '__main__':
    download_coco_samples()
    create_edge_cases()
    verify_dataset()
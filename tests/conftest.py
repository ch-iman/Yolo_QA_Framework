"""
Configuration Pytest pour YOLO QA Framework.

Ce fichier définit des fixtures réutilisables par tous les tests.
"""

import sys
from pathlib import Path
import pytest

# Ajouter le répertoire racine au PYTHONPATH
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"\n✅ PYTHONPATH configuré : {project_root}\n")


# ═══════════════════════════════════════════════════════════
# FIXTURES GLOBALES (disponibles pour tous les tests)
# ═══════════════════════════════════════════════════════════

def pytest_addoption(parser):
    """Ajoute les options CLI personnalisées pour pytest"""
    parser.addoption(
        "--baseline-save", 
        action="store_true", 
        default=False,
        help="Sauvegarder les métriques comme baseline"
    )
    parser.addoption(
        "--baseline-compare", 
        action="store_true", 
        default=True,
        help="Comparer avec la baseline existante"
    )


@pytest.fixture(scope="session")
def baseline_mode(request):
    """Détermine si on est en mode baseline-save ou compare"""
    return {
        'save': request.config.getoption("--baseline-save", default=False),
        'compare': request.config.getoption("--baseline-compare", default=True)
    }

@pytest.fixture(scope="session")
def project_paths():
    """
    Retourne les chemins importants du projet.
    Scope "session" = créé une seule fois pour toute la session de tests.
    """
    root = Path(__file__).parent.parent
    return {
        'root': root,
        'src': root / 'src',
        'tests': root / 'tests',
        'data': root / 'data',
        'models': root / 'models',
        'reports': root / 'reports',
    }


@pytest.fixture(scope="session")
def sample_image_path(project_paths):
    """
    Retourne le chemin d'une image de test.
    Utilisé par les tests qui ont besoin d'une image valide.
    """
    normal_dir = project_paths['data'] / 'images' / 'normal'
    images = list(normal_dir.glob('*.jpg')) + list(normal_dir.glob('*.jpeg'))
    
    if not images:
        pytest.skip("Aucune image de test disponible dans data/images/normal/")
    
    return str(images[0])


@pytest.fixture(scope="session")
def dataset_normal_path(project_paths):
    """
    Retourne le chemin du dossier d'images normales.
    """
    path = project_paths['data'] / 'images' / 'normal'
    if not path.exists():
        pytest.skip("Dossier data/images/normal/ introuvable")
    return str(path)


@pytest.fixture(scope="session")
def dataset_edge_cases_path(project_paths):
    """
    Retourne le chemin du dossier d'edge cases.
    """
    path = project_paths['data'] / 'images' / 'edge_cases'
    if not path.exists():
        pytest.skip("Dossier data/images/edge_cases/ introuvable")
    return str(path)

@pytest.fixture(scope="class")
def detector():
    """
    Fixture : Détecteur YOLO partagé entre tous les tests.
    Créé une fois par classe de test pour optimiser les performances.
    """
    from src.yolo_detector import YOLODetector
    return YOLODetector()
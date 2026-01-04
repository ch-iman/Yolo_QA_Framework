from pathlib import Path
import shutil

def clean_dataset():
    """Supprime les images existantes"""
    """base_dir = Path('C:/Users/hp/OneDrive/Documents/my_projects/data/images')"""
    script_dir = Path(__file__).parent.resolve()  # Répertoire du script
    output_dir = script_dir.parent / 'data' / 'images' / 'normal'  # Remonte d'un niveau
    output_dir.mkdir(parents=True, exist_ok=True)
    base_dir = script_dir.parent / 'data' / 'images'
    folders = ['normal', 'edge_cases']
    
    for folder in folders:
        folder_path = base_dir / folder
        if folder_path.exists():
            shutil.rmtree(folder_path)
            print(f"✓ Supprimé : {folder_path}")
        else:
            print(f"⏭️  N'existe pas : {folder_path}")
    
    print("\n✓ Nettoyage terminé !")

if __name__ == '__main__':
    clean_dataset()
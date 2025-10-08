#!/usr/bin/env python3
"""
Script pour trouver votre image photo-velours
"""

import os
import cv2

def search_for_image():
    """Recherche l'image photo-velours sur le système"""
    print("🔍 Recherche de l'image 'photo-velours'...")
    
    # Dossiers à rechercher
    search_dirs = [
        "C:\\Users\\eloua\\",
        "C:\\Users\\eloua\\OneDrive\\",
        "C:\\Users\\eloua\\Desktop\\",
        "C:\\Users\\eloua\\Pictures\\",
        "C:\\Users\\eloua\\Documents\\"
    ]
    
    found_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"Recherche dans: {search_dir}")
            try:
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if "photo-velours" in file.lower() or "velours" in file.lower():
                            full_path = os.path.join(root, file)
                            found_files.append(full_path)
                            print(f"  ✓ Trouvé: {full_path}")
            except PermissionError:
                print(f"  ⚠️ Accès refusé: {search_dir}")
            except Exception as e:
                print(f"  ⚠️ Erreur: {e}")
    
    if found_files:
        print(f"\n✅ {len(found_files)} fichier(s) trouvé(s):")
        for i, file_path in enumerate(found_files, 1):
            print(f"{i}. {file_path}")
            
            # Tester si c'est une image valide
            try:
                img = cv2.imread(file_path)
                if img is not None:
                    print(f"   ✓ Image valide - Taille: {img.shape}")
                else:
                    print(f"   ❌ Fichier corrompu ou non-image")
            except Exception as e:
                print(f"   ❌ Erreur lors du chargement: {e}")
        
        return found_files
    else:
        print("\n❌ Aucun fichier 'photo-velours' trouvé")
        print("\n💡 Suggestions:")
        print("1. Vérifiez le nom exact du fichier")
        print("2. Copiez l'image dans le dossier actuel")
        print("3. Spécifiez le chemin complet manuellement")
        return []

def create_test_image():
    """Crée une image de test si aucune n'est trouvée"""
    print("\n🖼️ Création d'une image de test...")
    
    import numpy as np
    
    # Créer une image de test avec des ombres simulées
    height, width = 400, 600
    
    # Image de base avec gradient
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Ajouter des couleurs de velours
    img[:, :, 0] = 50  # Rouge
    img[:, :, 1] = 30  # Vert  
    img[:, :, 2] = 80  # Bleu
    
    # Ajouter des ombres simulées
    for y in range(height):
        for x in range(width):
            # Ombre en haut à gauche
            if x < width//2 and y < height//2:
                img[y, x] = img[y, x] * 0.3
            # Ombre en bas à droite  
            elif x > width//2 and y > height//2:
                img[y, x] = img[y, x] * 0.4
            # Zone éclairée au centre
            else:
                img[y, x] = img[y, x] * 1.2
    
    # Ajouter du bruit pour simuler la texture
    noise = np.random.randint(-20, 20, (height, width, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Sauvegarder
    test_path = "image_test_velours.jpg"
    cv2.imwrite(test_path, img)
    print(f"✅ Image de test créée: {test_path}")
    
    return test_path

if __name__ == "__main__":
    # Rechercher l'image
    found_files = search_for_image()
    
    if not found_files:
        # Créer une image de test
        test_path = create_test_image()
        print(f"\n🎯 Utilisation de l'image de test: {test_path}")
        
        # Tester le système de suppression d'ombres
        print("\n🧪 Test du système de suppression d'ombres...")
        try:
            from shadow_removing import AdvancedShadowRemover
            
            shadow_remover = AdvancedShadowRemover()
            image = cv2.imread(test_path)
            
            if image is not None:
                result = shadow_remover.remove_shadows(image, method='auto')
                if result is not None:
                    cv2.imwrite("test_resultat.jpg", result)
                    print("✅ Test réussi! Résultat sauvegardé: test_resultat.jpg")
                else:
                    print("❌ Erreur lors de la suppression d'ombres")
            else:
                print("❌ Impossible de charger l'image de test")
        except Exception as e:
            print(f"❌ Erreur: {e}")
    else:
        print(f"\n🎯 Utilisez l'un de ces fichiers avec votre système de suppression d'ombres")





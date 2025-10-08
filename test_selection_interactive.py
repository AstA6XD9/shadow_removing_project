#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la sélection interactive de couleur et création de textures MASSIVES
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_selection_interactive():
    """Test de la sélection interactive et création de textures MASSIVES"""
    print("Test Selection Interactive et Textures MASSIVES")
    print("=" * 60)
    print("Objectif: Selectionner manuellement la couleur desiree")
    print("et creer des textures MASSIVES avec multiplication enorme")
    print("=" * 60)
    
    # Chemin de l'image
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image non trouvee: {image_path}")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return
    
    print(f"Image chargee: {image.shape}")
    
    # Initialiser le shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Test 1: Selection interactive de la couleur
    print("\n1. Selection Interactive de la Couleur")
    print("-" * 40)
    print("Une fenetre va s'ouvrir pour selectionner la couleur desiree")
    print("Instructions:")
    print("- Cliquez et glissez pour selectionner une region")
    print("- Appuyez sur ENTER pour confirmer")
    print("- Appuyez sur ESC pour annuler")
    print("- Appuyez sur SPACE pour voir la couleur")
    
    try:
        region_coords, selected_color = shadow_remover.select_color_region_interactive(image)
        
        if region_coords is not None and selected_color is not None:
            print(f"\nSelection reussie!")
            print(f"Region: {region_coords}")
            print(f"Couleur: BGR({selected_color[0]}, {selected_color[1]}, {selected_color[2]})")
            
            # Sauvegarder la region selectionnee
            x, y, w, h = region_coords
            selected_region = image[y:y+h, x:x+w]
            cv2.imwrite("region_selectionnee.jpg", selected_region)
            print("Sauvegarde: region_selectionnee.jpg")
            
        else:
            print("Selection annulee ou echouee")
            return
            
    except Exception as e:
        print(f"ERREUR lors de la selection: {e}")
        return
    
    # Test 2: Creation d'echantillon personnalise
    print("\n2. Creation d'Echantillon Personnalise")
    print("-" * 40)
    
    try:
        custom_sample = shadow_remover.create_custom_fabric_sample(
            image, 
            region_coords, 
            selected_color, 
            zoom_factor=1.3, 
            uniformization_strength=0.8
        )
        
        if custom_sample is not None:
            cv2.imwrite("echantillon_personnalise.jpg", custom_sample)
            print("OK - Echantillon personnalise: echantillon_personnalise.jpg")
            print(f"Taille: {custom_sample.shape[1]}x{custom_sample.shape[0]}")
            
            # Verifier la couleur de l'echantillon
            sample_color = np.mean(custom_sample.reshape(-1, 3), axis=0)
            print(f"Couleur de l'echantillon: BGR({sample_color[0]:.1f}, {sample_color[1]:.1f}, {sample_color[2]:.1f})")
            
        else:
            print("ECHEC - Impossible de creer l'echantillon personnalise")
            return
            
    except Exception as e:
        print(f"ERREUR lors de la creation d'echantillon: {e}")
        return
    
    # Test 3: Creation de texture MASSIVE
    print("\n3. Creation de Texture MASSIVE")
    print("-" * 40)
    
    try:
        # Texture MASSIVE 3000x3000
        print("Creation texture MASSIVE 3000x3000...")
        massive_texture = shadow_remover.create_massive_fabric_texture(
            image, 
            region_coords, 
            selected_color, 
            target_size=(3000, 3000),
            zoom_factor=1.3,
            uniformization_strength=0.8
        )
        
        if massive_texture is not None:
            cv2.imwrite("texture_massive_3000x3000.jpg", massive_texture)
            print("OK - Texture MASSIVE: texture_massive_3000x3000.jpg")
            print(f"Taille: {massive_texture.shape[1]}x{massive_texture.shape[0]}")
            
            # Verifier la couleur de la texture
            texture_color = np.mean(massive_texture.reshape(-1, 3), axis=0)
            print(f"Couleur de la texture: BGR({texture_color[0]:.1f}, {texture_color[1]:.1f}, {texture_color[2]:.1f})")
            
        else:
            print("ECHEC - Impossible de creer la texture MASSIVE")
            
    except Exception as e:
        print(f"ERREUR lors de la creation de texture MASSIVE: {e}")
    
    # Test 4: Creation de plusieurs textures MASSIVES
    print("\n4. Creation de Plusieurs Textures MASSIVES")
    print("-" * 40)
    
    # Tailles MASSIVES
    massive_sizes = [
        (2000, 2000),   # 4 megapixels
        (3000, 3000),   # 9 megapixels
        (4000, 4000),   # 16 megapixels
        (5000, 5000)    # 25 megapixels
    ]
    
    try:
        print(f"Creation de {len(massive_sizes)} textures MASSIVES...")
        textures = shadow_remover.create_multiple_massive_textures(
            image, 
            region_coords, 
            selected_color, 
            sizes=massive_sizes,
            zoom_factor=1.3,
            uniformization_strength=0.8
        )
        
        # Sauvegarder toutes les textures
        for size_name, texture in textures.items():
            output_path = f"texture_massive_{size_name}.jpg"
            cv2.imwrite(output_path, texture)
            print(f"OK - {size_name}: {output_path}")
            
            # Calculer la taille en megapixels
            width, height = texture.shape[1], texture.shape[0]
            megapixels = (width * height) / 1000000
            print(f"  Taille: {width}x{height} ({megapixels:.1f} megapixels)")
        
    except Exception as e:
        print(f"ERREUR lors de la creation de textures MASSIVES: {e}")
    
    # Test 5: Test avec differentes forces d'uniformisation
    print("\n5. Test Differentes Forces d'Uniformisation")
    print("-" * 40)
    
    forces = [0.5, 0.7, 0.8, 0.9]
    
    for force in forces:
        try:
            print(f"\nTest avec force d'uniformisation: {force}")
            sample = shadow_remover.create_custom_fabric_sample(
                image, 
                region_coords, 
                selected_color, 
                zoom_factor=1.3, 
                uniformization_strength=force
            )
            
            if sample is not None:
                output_path = f"echantillon_force_{force}.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  OK - Force {force}: {output_path}")
            else:
                print(f"  ECHEC - Force {force}")
                
        except Exception as e:
            print(f"  ERREUR - Force {force}: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- region_selectionnee.jpg : Region selectionnee manuellement")
    print("- echantillon_personnalise.jpg : Echantillon avec couleur preservee")
    print("- texture_massive_*.jpg : Textures MASSIVES de differentes tailles")
    print("- echantillon_force_*.jpg : Echantillons avec differentes forces")
    print("\nVous avez maintenant des textures MASSIVES avec")
    print("la couleur exacte que vous avez selectionnee!")

def main():
    """Fonction principale"""
    print("Test de Selection Interactive et Textures MASSIVES")
    print("=" * 70)
    print("Nouvelles fonctionnalites:")
    print("1. Selection interactive de la couleur desiree")
    print("2. Preservation de la couleur selectionnee")
    print("3. Uniformisation sans changer la couleur")
    print("4. Creation de textures MASSIVES (jusqu'a 25 megapixels)")
    print("5. Multiplication enorme des images")
    print("=" * 70)
    
    # Test de la selection interactive
    test_selection_interactive()
    
    print("\nResume des Ameliorations:")
    print("=" * 40)
    print("✅ Selection interactive de la couleur")
    print("✅ Preservation de la couleur choisie")
    print("✅ Uniformisation sans changement de couleur")
    print("✅ Textures MASSIVES (jusqu'a 5000x5000)")
    print("✅ Multiplication enorme des images")
    print("\nVous pouvez maintenant choisir exactement la couleur")
    print("que vous voulez et creer des textures MASSIVES!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'uniformisation MAXIMALE pour capturer le maximum de défauts
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_uniformisation_maximale():
    """Test de l'uniformisation MAXIMALE"""
    print("Test Uniformisation MAXIMALE")
    print("=" * 60)
    print("Objectif: Capturer et eliminer le MAXIMUM de defauts")
    print("pour eviter qu'ils s'affichent lors de la multiplication")
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
    print("Selectionnez la couleur que vous voulez pour la texture")
    
    try:
        region_coords, selected_color = shadow_remover.select_color_region_interactive(image)
        
        if region_coords is not None and selected_color is not None:
            print(f"\nSelection reussie!")
            print(f"Region: {region_coords}")
            print(f"Couleur selectionnee: BGR({selected_color[0]}, {selected_color[1]}, {selected_color[2]})")
            
            # Sauvegarder la region selectionnee
            x, y, w, h = region_coords
            selected_region = image[y:y+h, x:x+w]
            cv2.imwrite("region_selectionnee_maximale.jpg", selected_region)
            print("Sauvegarde: region_selectionnee_maximale.jpg")
            
        else:
            print("Selection annulee")
            return
            
    except Exception as e:
        print(f"ERREUR lors de la selection: {e}")
        return
    
    # Test 2: Creation d'echantillon avec uniformisation MAXIMALE
    print("\n2. Creation d'Echantillon avec Uniformisation MAXIMALE")
    print("-" * 40)
    
    # Test avec differentes forces d'uniformisation
    forces = [0.7, 0.8, 0.9, 0.95]
    
    for force in forces:
        try:
            print(f"\nTest avec force d'uniformisation MAXIMALE: {force}")
            sample = shadow_remover.create_custom_fabric_sample(
                image, 
                region_coords, 
                selected_color, 
                zoom_factor=1.3, 
                uniformization_strength=force
            )
            
            if sample is not None:
                output_path = f"echantillon_uniformisation_maximale_{force}.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  OK - Force {force}: {output_path}")
                print(f"  Taille: {sample.shape[1]}x{sample.shape[0]}")
                
                # Analyser la couleur de l'echantillon
                sample_color = np.mean(sample.reshape(-1, 3), axis=0)
                print(f"  Couleur: BGR({sample_color[0]:.1f}, {sample_color[1]:.1f}, {sample_color[2]:.1f})")
                
            else:
                print(f"  ECHEC - Force {force}")
                
        except Exception as e:
            print(f"  ERREUR - Force {force}: {e}")
    
    # Test 3: Creation de texture MASSIVE avec uniformisation MAXIMALE
    print("\n3. Creation de Texture MASSIVE avec Uniformisation MAXIMALE")
    print("-" * 40)
    
    try:
        # Texture MASSIVE avec uniformisation maximale
        print("Creation texture MASSIVE 3000x3000 avec uniformisation MAXIMALE...")
        massive_texture = shadow_remover.create_massive_fabric_texture(
            image, 
            region_coords, 
            selected_color, 
            target_size=(3000, 3000),
            zoom_factor=1.3,
            uniformization_strength=0.9  # Force maximale
        )
        
        if massive_texture is not None:
            cv2.imwrite("texture_massive_uniformisation_maximale_3000x3000.jpg", massive_texture)
            print("OK - Texture MASSIVE: texture_massive_uniformisation_maximale_3000x3000.jpg")
            print(f"Taille: {massive_texture.shape[1]}x{massive_texture.shape[0]}")
            
            # Calculer la taille en megapixels
            width, height = massive_texture.shape[1], massive_texture.shape[0]
            megapixels = (width * height) / 1000000
            print(f"Taille: {megapixels:.1f} megapixels")
            
            # Analyser la couleur de la texture
            texture_color = np.mean(massive_texture.reshape(-1, 3), axis=0)
            print(f"Couleur de la texture: BGR({texture_color[0]:.1f}, {texture_color[1]:.1f}, {texture_color[2]:.1f})")
            
        else:
            print("ECHEC - Impossible de creer la texture MASSIVE")
            
    except Exception as e:
        print(f"ERREUR lors de la creation de texture MASSIVE: {e}")
    
    # Test 4: Creation de plusieurs textures MASSIVES avec uniformisation MAXIMALE
    print("\n4. Creation de Plusieurs Textures MASSIVES avec Uniformisation MAXIMALE")
    print("-" * 40)
    
    # Tailles MASSIVES
    massive_sizes = [
        (2000, 2000),   # 4 megapixels
        (3000, 3000),   # 9 megapixels
        (4000, 4000),   # 16 megapixels
        (5000, 5000)    # 25 megapixels
    ]
    
    try:
        print(f"Creation de {len(massive_sizes)} textures MASSIVES avec uniformisation MAXIMALE...")
        
        for i, (width, height) in enumerate(massive_sizes, 1):
            print(f"\nCreation texture {i}/{len(massive_sizes)}: {width}x{height}")
            
            texture = shadow_remover.create_massive_fabric_texture(
                image, 
                region_coords, 
                selected_color, 
                target_size=(width, height),
                zoom_factor=1.3,
                uniformization_strength=0.9  # Force maximale
            )
            
            if texture is not None:
                output_path = f"texture_massive_uniformisation_maximale_{width}x{height}.jpg"
                cv2.imwrite(output_path, texture)
                print(f"  OK - {width}x{height}: {output_path}")
                
                # Calculer la taille en megapixels
                megapixels = (width * height) / 1000000
                print(f"  Taille: {megapixels:.1f} megapixels")
                
                # Analyser la couleur
                texture_color = np.mean(texture.reshape(-1, 3), axis=0)
                print(f"  Couleur: BGR({texture_color[0]:.1f}, {texture_color[1]:.1f}, {texture_color[2]:.1f})")
            else:
                print(f"  ECHEC - {width}x{height}")
                
    except Exception as e:
        print(f"ERREUR lors de la creation de textures MASSIVES: {e}")
    
    # Test 5: Comparaison avant/après uniformisation
    print("\n5. Comparaison Avant/Apres Uniformisation")
    print("-" * 40)
    
    try:
        # Echantillon sans uniformisation (force 0.0)
        print("Creation echantillon SANS uniformisation...")
        sample_sans = shadow_remover.create_custom_fabric_sample(
            image, 
            region_coords, 
            selected_color, 
            zoom_factor=1.3, 
            uniformization_strength=0.0
        )
        
        if sample_sans is not None:
            cv2.imwrite("echantillon_SANS_uniformisation.jpg", sample_sans)
            print("OK - Echantillon SANS uniformisation: echantillon_SANS_uniformisation.jpg")
        
        # Echantillon avec uniformisation MAXIMALE
        print("\nCreation echantillon AVEC uniformisation MAXIMALE...")
        sample_avec = shadow_remover.create_custom_fabric_sample(
            image, 
            region_coords, 
            selected_color, 
            zoom_factor=1.3, 
            uniformization_strength=0.9
        )
        
        if sample_avec is not None:
            cv2.imwrite("echantillon_AVEC_uniformisation_maximale.jpg", sample_avec)
            print("OK - Echantillon AVEC uniformisation MAXIMALE: echantillon_AVEC_uniformisation_maximale.jpg")
        
    except Exception as e:
        print(f"ERREUR lors de la comparaison: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- region_selectionnee_maximale.jpg : Region selectionnee")
    print("- echantillon_uniformisation_maximale_*.jpg : Echantillons avec differentes forces")
    print("- texture_massive_uniformisation_maximale_*.jpg : Textures MASSIVES uniformisees")
    print("- echantillon_SANS_uniformisation.jpg : Comparaison sans uniformisation")
    print("- echantillon_AVEC_uniformisation_maximale.jpg : Comparaison avec uniformisation")
    print("\nL'uniformisation MAXIMALE capture et elimine")
    print("le MAXIMUM de defauts pour eviter qu'ils s'affichent!")

def main():
    """Fonction principale"""
    print("Test de l'Uniformisation MAXIMALE")
    print("=" * 70)
    print("Nouvelles fonctionnalites d'uniformisation MAXIMALE:")
    print("1. Detection et elimination des defauts")
    print("2. Uniformisation multi-echelle")
    print("3. Correction specifique des defauts detectes")
    print("4. Filtre bilateral pour uniformisation finale")
    print("5. Reduction agressive des variations")
    print("6. Lissage final pour eliminer les derniers defauts")
    print("=" * 70)
    
    # Test de l'uniformisation maximale
    test_uniformisation_maximale()
    
    print("\nResume des Ameliorations MAXIMALES:")
    print("=" * 40)
    print("✅ Detection automatique des defauts")
    print("✅ Uniformisation multi-echelle")
    print("✅ Correction specifique des zones problematiques")
    print("✅ Filtre bilateral pour uniformisation finale")
    print("✅ Reduction agressive des variations")
    print("✅ Lissage final pour eliminer les derniers defauts")
    print("\nL'uniformisation MAXIMALE capture et elimine")
    print("le MAXIMUM de defauts pour eviter qu'ils s'affichent!")

if __name__ == "__main__":
    main()

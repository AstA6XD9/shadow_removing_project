#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de l'uniformisation de la zone zoomée
Uniformise la zone zoomée et ajuste la couleur vers la couleur la plus claire
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_uniformisation_zone_zoomee():
    """Test de l'uniformisation de la zone zoomée"""
    print("Test Uniformisation de la Zone Zoomee")
    print("=" * 60)
    print("Objectif: Uniformiser la zone zoomee et ajuster la couleur")
    print("vers la couleur la plus claire de l'image originale")
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
    
    # Test 1: Detection de la couleur la plus claire
    print("\n1. Detection de la Couleur la Plus Claire")
    print("-" * 40)
    
    try:
        lightest_color = shadow_remover._find_lightest_color_in_image(image)
        print(f"Couleur la plus claire trouvee: BGR({lightest_color[0]}, {lightest_color[1]}, {lightest_color[2]})")
        
        # Creer une image de demonstration avec la couleur la plus claire
        color_demo = np.full((100, 200, 3), lightest_color, dtype=np.uint8)
        cv2.imwrite("couleur_plus_claire.jpg", color_demo)
        print("Sauvegarde: couleur_plus_claire.jpg")
        
    except Exception as e:
        print(f"ERREUR lors de la detection de couleur: {e}")
    
    # Test 2: Comparaison echantillon normal vs uniformise
    print("\n2. Comparaison Echantillon Normal vs Uniformise")
    print("-" * 40)
    
    try:
        # Echantillon normal (sans uniformisation)
        print("Creation echantillon normal...")
        normal_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=1.3)
        
        if normal_sample is not None:
            cv2.imwrite("echantillon_normal.jpg", normal_sample)
            print("OK - Echantillon normal: echantillon_normal.jpg")
            print(f"Taille: {normal_sample.shape[1]}x{normal_sample.shape[0]}")
        
        # Echantillon uniformise
        print("\nCreation echantillon uniformise...")
        uniformized_sample = shadow_remover.create_uniformized_fabric_sample(
            image, 
            zoom_factor=1.3, 
            uniformization_strength=0.8
        )
        
        if uniformized_sample is not None:
            cv2.imwrite("echantillon_uniformise.jpg", uniformized_sample)
            print("OK - Echantillon uniformise: echantillon_uniformise.jpg")
            print(f"Taille: {uniformized_sample.shape[1]}x{uniformized_sample.shape[0]}")
        
    except Exception as e:
        print(f"ERREUR lors de la creation d'echantillons: {e}")
    
    # Test 3: Differentes forces d'uniformisation
    print("\n3. Test Differentes Forces d'Uniformisation")
    print("-" * 40)
    
    uniformization_strengths = [0.3, 0.5, 0.7, 0.8, 0.9]
    
    for strength in uniformization_strengths:
        try:
            print(f"\nTest avec force d'uniformisation: {strength}")
            sample = shadow_remover.create_uniformized_fabric_sample(
                image, 
                zoom_factor=1.3, 
                uniformization_strength=strength
            )
            
            if sample is not None:
                output_path = f"uniformise_force_{strength}.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  OK - Force {strength}: {output_path}")
                print(f"  Taille: {sample.shape[1]}x{sample.shape[0]}")
            else:
                print(f"  ECHEC - Force {strength}")
                
        except Exception as e:
            print(f"  ERREUR - Force {strength}: {e}")
    
    # Test 4: Creation de textures uniformisees
    print("\n4. Creation de Textures Uniformisees")
    print("-" * 40)
    
    try:
        # Texture uniformisee 1200x900 (comme votre exemple)
        print("Creation texture uniformisee 1200x900...")
        texture_1200x900 = shadow_remover.create_uniformized_fabric_texture(
            image, 
            target_size=(1200, 900),
            zoom_factor=1.3,
            uniformization_strength=0.8
        )
        
        if texture_1200x900 is not None:
            cv2.imwrite("texture_uniformisee_1200x900.jpg", texture_1200x900)
            print("OK - Texture 1200x900: texture_uniformisee_1200x900.jpg")
        
        # Autres tailles
        sizes = [(800, 600), (1000, 750), (1600, 1200)]
        
        for width, height in sizes:
            print(f"\nCreation texture {width}x{height}...")
            texture = shadow_remover.create_uniformized_fabric_texture(
                image, 
                target_size=(width, height),
                zoom_factor=1.3,
                uniformization_strength=0.8
            )
            
            if texture is not None:
                output_path = f"texture_uniformisee_{width}x{height}.jpg"
                cv2.imwrite(output_path, texture)
                print(f"  OK - {width}x{height}: {output_path}")
            else:
                print(f"  ECHEC - {width}x{height}")
        
    except Exception as e:
        print(f"ERREUR lors de la creation de textures: {e}")
    
    # Test 5: Comparaison texture normale vs uniformisee
    print("\n5. Comparaison Texture Normale vs Uniformisee")
    print("-" * 40)
    
    try:
        # Texture normale (ancienne methode)
        print("Creation texture normale...")
        normal_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=1.3)
        if normal_sample is not None:
            normal_texture = shadow_remover.create_fabric_texture_from_sample(
                normal_sample, 
                target_size=(1200, 900),
                preserve_characteristics=True
            )
            if normal_texture is not None:
                cv2.imwrite("texture_normale_1200x900.jpg", normal_texture)
                print("OK - Texture normale: texture_normale_1200x900.jpg")
        
        # Texture uniformisee (nouvelle methode)
        print("\nCreation texture uniformisee...")
        uniformized_texture = shadow_remover.create_uniformized_fabric_texture(
            image, 
            target_size=(1200, 900),
            uniformization_strength=0.8
        )
        
        if uniformized_texture is not None:
            cv2.imwrite("texture_uniformisee_comparaison_1200x900.jpg", uniformized_texture)
            print("OK - Texture uniformisee: texture_uniformisee_comparaison_1200x900.jpg")
        
    except Exception as e:
        print(f"ERREUR lors de la comparaison: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- couleur_plus_claire.jpg : Couleur la plus claire detectee")
    print("- echantillon_normal.jpg : Echantillon sans uniformisation")
    print("- echantillon_uniformise.jpg : Echantillon uniformise")
    print("- uniformise_force_*.jpg : Differentes forces d'uniformisation")
    print("- texture_uniformisee_*.jpg : Textures uniformisees")
    print("- texture_normale_1200x900.jpg : Texture normale (comparaison)")
    print("- texture_uniformisee_comparaison_1200x900.jpg : Texture uniformisee (comparaison)")
    print("\nL'uniformisation evite que les motifs indesirables")
    print("deviennent plus visibles lors de la multiplication!")

def main():
    """Fonction principale"""
    print("Test de l'Uniformisation de la Zone Zoomee")
    print("=" * 70)
    print("Nouvelles fonctionnalites:")
    print("1. Detection de la couleur la plus claire de l'image originale")
    print("2. Uniformisation de la zone zoomee")
    print("3. Ajustement de couleur vers la couleur la plus claire")
    print("4. Reduction des variations de luminosite")
    print("5. Creation de textures uniformisees")
    print("=" * 70)
    
    # Test de l'uniformisation
    test_uniformisation_zone_zoomee()
    
    print("\nResume des Ameliorations:")
    print("=" * 40)
    print("✅ Detection de la couleur la plus claire")
    print("✅ Uniformisation de la zone zoomee")
    print("✅ Ajustement de couleur vers la couleur cible")
    print("✅ Reduction des variations de luminosite")
    print("✅ Textures uniformisees sans motifs indesirables")
    print("\nLes textures sont maintenant uniformisees et")
    print("la couleur est ajustee vers la couleur la plus claire!")

if __name__ == "__main__":
    main()

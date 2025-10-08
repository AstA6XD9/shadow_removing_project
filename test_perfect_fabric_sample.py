#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la création d'échantillons parfaits de tissu
Détecte les zones les plus homogènes et les zoome en préservant les caractéristiques
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_perfect_fabric_sample():
    """Test de la création d'échantillons parfaits"""
    print("Test de Creation d'Echantillons Parfaits de Tissu")
    print("=" * 60)
    print("Objectif: Detectar les zones les plus homogenes et les zoomer")
    print("Applicable uniquement aux tissus unis sans motifs")
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
    
    # Test 1: Detection de tissu uni
    print("\n1. Detection de Tissu Uni")
    print("-" * 30)
    
    is_solid, complexity = shadow_remover._is_solid_color_fabric(image)
    print(f"Complexite de texture: {complexity:.6f}")
    print(f"Tissu uni detecte: {is_solid}")
    
    if not is_solid:
        print("ATTENTION: Cette image ne semble pas etre un tissu uni")
        print("La fonctionnalite est optimisee pour les tissus unis sans motifs")
        print("Continuer le test pour demonstration...")
    
    # Test 2: Detection de la zone parfaite
    print("\n2. Detection de la Zone Parfaite")
    print("-" * 30)
    
    try:
        perfect_region, score = shadow_remover._detect_perfect_fabric_region(image, min_region_size=100)
        
        if perfect_region is not None:
            x, y, w, h = perfect_region
            print(f"Zone parfaite trouvee: ({x}, {y}, {w}, {h})")
            print(f"Score d'homogeneite: {score:.2f}")
            
            # Dessiner un rectangle sur l'image pour montrer la zone
            image_with_region = image.copy()
            cv2.rectangle(image_with_region, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imwrite("zone_parfaite_detectee.jpg", image_with_region)
            print("Sauvegarde: zone_parfaite_detectee.jpg")
            
            # Extraire la zone parfaite
            perfect_zone = image[y:y+h, x:x+w]
            cv2.imwrite("zone_parfaite_extraite.jpg", perfect_zone)
            print("Sauvegarde: zone_parfaite_extraite.jpg")
            
        else:
            print("ERREUR: Aucune zone parfaite trouvee")
            
    except Exception as e:
        print(f"ERREUR lors de la detection: {e}")
    
    # Test 3: Creation d'echantillon parfait
    print("\n3. Creation d'Echantillon Parfait")
    print("-" * 30)
    
    zoom_factors = [1.5, 2.0, 3.0, 4.0]
    
    for zoom_factor in zoom_factors:
        try:
            print(f"\nTest avec zoom x{zoom_factor}:")
            sample = shadow_remover.create_perfect_fabric_sample(
                image, 
                zoom_factor=zoom_factor, 
                min_region_size=100
            )
            
            if sample is not None:
                # Sauvegarder l'echantillon
                output_path = f"echantillon_parfait_zoom_{zoom_factor}x.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  OK - Echantillon cree: {output_path}")
                print(f"  Taille: {sample.shape[1]}x{sample.shape[0]}")
            else:
                print(f"  ECHEC - Impossible de creer l'echantillon")
                
        except Exception as e:
            print(f"  ERREUR: {e}")
    
    # Test 4: Creation de texture a partir d'echantillon
    print("\n4. Creation de Texture a partir d'Echantillon")
    print("-" * 30)
    
    try:
        # Creer un echantillon de base
        base_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=2.0)
        
        if base_sample is not None:
            # Creer differentes tailles de texture
            target_sizes = [
                (400, 300),   # Petit
                (800, 600),   # Moyen
                (1200, 900),  # Grand
                (1600, 1200)  # Tres grand
            ]
            
            for width, height in target_sizes:
                texture = shadow_remover.create_fabric_texture_from_sample(
                    base_sample, 
                    target_size=(width, height),
                    preserve_characteristics=True
                )
                
                if texture is not None:
                    output_path = f"texture_tissu_{width}x{height}.jpg"
                    cv2.imwrite(output_path, texture)
                    print(f"  OK - Texture {width}x{height}: {output_path}")
                else:
                    print(f"  ECHEC - Texture {width}x{height}")
        else:
            print("  ECHEC - Impossible de creer l'echantillon de base")
            
    except Exception as e:
        print(f"  ERREUR: {e}")
    
    # Test 5: Comparaison avec et sans preservation des caracteristiques
    print("\n5. Comparaison avec/sans Preservation des Caracteristiques")
    print("-" * 30)
    
    try:
        base_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=2.0)
        
        if base_sample is not None:
            # Avec preservation
            texture_with_preservation = shadow_remover.create_fabric_texture_from_sample(
                base_sample, 
                target_size=(600, 400),
                preserve_characteristics=True
            )
            
            # Sans preservation
            texture_without_preservation = shadow_remover.create_fabric_texture_from_sample(
                base_sample, 
                target_size=(600, 400),
                preserve_characteristics=False
            )
            
            if texture_with_preservation is not None:
                cv2.imwrite("texture_avec_preservation.jpg", texture_with_preservation)
                print("  OK - Avec preservation: texture_avec_preservation.jpg")
            
            if texture_without_preservation is not None:
                cv2.imwrite("texture_sans_preservation.jpg", texture_without_preservation)
                print("  OK - Sans preservation: texture_sans_preservation.jpg")
                
        else:
            print("  ECHEC - Impossible de creer l'echantillon de base")
            
    except Exception as e:
        print(f"  ERREUR: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- zone_parfaite_detectee.jpg : Image avec zone parfaite marquee")
    print("- zone_parfaite_extraite.jpg : Zone parfaite extraite")
    print("- echantillon_parfait_zoom_*.jpg : Echantillons avec differents zooms")
    print("- texture_tissu_*.jpg : Textures de differentes tailles")
    print("- texture_avec_preservation.jpg : Avec variations subtiles")
    print("- texture_sans_preservation.jpg : Sans variations")
    print("\nCette fonctionnalite est optimisee pour les tissus unis!")

def test_with_different_images():
    """Test avec differentes images pour voir la detection de tissu uni"""
    print("\n" + "=" * 60)
    print("Test avec Differentes Images")
    print("=" * 60)
    
    # Chemins d'images a tester
    image_paths = [
        "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg",
        "C:/Users/eloua/OneDrive/Images/IMG_6020.jpg"
    ]
    
    shadow_remover = AdvancedShadowRemover()
    
    for i, image_path in enumerate(image_paths, 1):
        if os.path.exists(image_path):
            print(f"\nImage {i}: {os.path.basename(image_path)}")
            print("-" * 30)
            
            image = cv2.imread(image_path)
            if image is not None:
                is_solid, complexity = shadow_remover._is_solid_color_fabric(image)
                print(f"  Complexite: {complexity:.6f}")
                print(f"  Tissu uni: {is_solid}")
                
                if is_solid:
                    print("  -> Cette image est adaptee pour la creation d'echantillons parfaits")
                else:
                    print("  -> Cette image contient probablement des motifs")
            else:
                print("  ERREUR: Impossible de charger l'image")
        else:
            print(f"\nImage {i}: {image_path} - NON TROUVEE")

def main():
    """Fonction principale"""
    print("Test de Creation d'Echantillons Parfaits de Tissu")
    print("=" * 70)
    print("Cette fonctionnalite:")
    print("1. Detecte automatiquement les tissus unis (sans motifs)")
    print("2. Trouve la zone la plus homogene de l'image")
    print("3. Zoom sur cette zone en preservant les caracteristiques")
    print("4. Cree des echantillons parfaits du tissu")
    print("5. Genere des textures de differentes tailles")
    print("=" * 70)
    
    # Test principal
    test_perfect_fabric_sample()
    
    # Test avec differentes images
    test_with_different_images()
    
    print("\nResume:")
    print("=" * 40)
    print("Cette fonctionnalite est parfaite pour:")
    print("- Creer des echantillons de tissus unis")
    print("- Generer des textures homogenes")
    print("- Preserver les caracteristiques du tissu")
    print("- Zoomer sur les zones les plus parfaites")
    print("\nElle fonctionne mieux avec les tissus unis sans motifs!")

if __name__ == "__main__":
    main()

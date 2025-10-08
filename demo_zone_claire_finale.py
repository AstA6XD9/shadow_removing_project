#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration finale de la détection de zone CLAIRE avec zoom logique
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def demo_zone_claire_finale():
    """Démonstration finale"""
    print("Demonstration - Zone CLAIRE avec Zoom Logique")
    print("=" * 60)
    print("Ameliorations:")
    print("- Detection de la zone la plus CLAIRE")
    print("- Zoom LOGIQUE (1.1x a 1.5x)")
    print("- Region plus grande (150px)")
    print("- Evite les details indesirables")
    print("=" * 60)
    
    # Chemin de l'image
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image non trouvee: {image_path}")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    print(f"Image chargee: {image.shape}")
    
    # Initialiser le shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Vérifier si c'est un tissu uni
    is_solid, complexity = shadow_remover._is_solid_color_fabric(image)
    print(f"Tissu uni detecte: {is_solid} (complexite: {complexity:.6f})")
    
    if not is_solid:
        print("ATTENTION: Cette image ne semble pas etre un tissu uni")
        return
    
    # Créer un échantillon avec zoom logique
    print("\nCreation d'un echantillon avec zoom LOGIQUE (x1.3)...")
    sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=1.3)
    
    if sample is not None:
        cv2.imwrite("echantillon_zone_claire_final.jpg", sample)
        print(f"OK - Echantillon cree: echantillon_zone_claire_final.jpg")
        print(f"Taille: {sample.shape[1]}x{sample.shape[0]}")
        
        # Analyser la luminosité
        gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_sample)
        print(f"Luminosite moyenne: {brightness:.1f}")
        
        # Créer une texture
        print("\nCreation d'une texture 500x400...")
        texture = shadow_remover.create_fabric_texture_from_sample(
            sample, 
            target_size=(500, 400),
            preserve_characteristics=True
        )
        
        if texture is not None:
            cv2.imwrite("texture_zone_claire_final.jpg", texture)
            print("OK - Texture creee: texture_zone_claire_final.jpg")
        
    else:
        print("ERREUR: Impossible de creer l'echantillon")
    
    # Test avec différents zooms logiques
    print("\nTest avec differents zooms logiques:")
    zoom_factors = [1.1, 1.2, 1.3, 1.4, 1.5]
    
    for zoom_factor in zoom_factors:
        try:
            sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=zoom_factor)
            if sample is not None:
                output_path = f"demo_zoom_{zoom_factor}x.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  Zoom x{zoom_factor}: {output_path} ({sample.shape[1]}x{sample.shape[0]})")
        except Exception as e:
            print(f"  Zoom x{zoom_factor}: ERREUR - {e}")

if __name__ == "__main__":
    demo_zone_claire_finale()

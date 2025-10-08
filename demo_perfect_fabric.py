#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration simple de la création d'échantillons parfaits de tissu
"""

import cv2
import os
from shadow_removing import AdvancedShadowRemover

def demo_perfect_fabric():
    """Démonstration simple"""
    print("Demonstration - Creation d'Echantillons Parfaits de Tissu")
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
        print("La fonctionnalite est optimisee pour les tissus unis sans motifs")
        return
    
    # Créer un échantillon parfait avec zoom x2
    print("\nCreation d'un echantillon parfait (zoom x2)...")
    sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=2.0)
    
    if sample is not None:
        cv2.imwrite("echantillon_parfait_demo.jpg", sample)
        print(f"OK - Echantillon cree: echantillon_parfait_demo.jpg")
        print(f"Taille: {sample.shape[1]}x{sample.shape[0]}")
        
        # Créer une texture plus grande
        print("\nCreation d'une texture 800x600...")
        texture = shadow_remover.create_fabric_texture_from_sample(
            sample, 
            target_size=(800, 600),
            preserve_characteristics=True
        )
        
        if texture is not None:
            cv2.imwrite("texture_demo_800x600.jpg", texture)
            print("OK - Texture creee: texture_demo_800x600.jpg")
        
    else:
        print("ERREUR: Impossible de creer l'echantillon")

if __name__ == "__main__":
    demo_perfect_fabric()

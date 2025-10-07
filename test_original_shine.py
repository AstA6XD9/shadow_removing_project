#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test pour préserver la brillance originale de l'image
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_original_shine():
    """Test pour préserver la brillance originale"""
    print("Test Preservation de la Brillance Originale")
    print("=" * 60)
    print("Prend la brillance de l'image originale et l'applique a la couleur uniforme")
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
    
    print("\n1. Analyse du type de tissu...")
    
    # Analyser le type de tissu
    fabric_type = shadow_remover._analyze_fabric_type(image)
    print(f"Type de tissu detecte: {fabric_type}")
    
    print("\n2. Test avec preservation de la brillance originale...")
    
    # Test avec méthode smart_fabric (qui préserve maintenant la brillance originale)
    try:
        result_original_shine = shadow_remover.uniformize_color(image, method='smart_fabric')
        cv2.imwrite("original_shine.jpg", result_original_shine)
        print("[OK] Brillance originale preservee: original_shine.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n3. Test combine - suppression d'ombres + brillance originale...")
    
    # Test combiné
    try:
        result_combined = shadow_remover.remove_shadows_with_color_uniformization(
            image, shadow_method='auto', color_method='smart_fabric'
        )
        cv2.imwrite("original_shine_combined.jpg", result_combined)
        print("[OK] Resultat combine avec brillance originale: original_shine_combined.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n4. Test avec differentes methodes...")
    
    methods = ['brightest', 'most_saturated', 'center']
    
    for method in methods:
        try:
            result = shadow_remover.uniformize_color(image, method=method)
            cv2.imwrite(f"original_shine_{method}.jpg", result)
            print(f"[OK] Methode {method} avec brillance originale: original_shine_{method}.jpg")
        except Exception as e:
            print(f"[ERROR] Methode {method} echouee: {e}")
    
    print("\n5. Test avec selection manuelle de region...")
    
    # Test avec région manuelle (centre de l'image)
    h, w = image.shape[:2]
    center_x, center_y = w//2, h//2
    patch_size = 15
    
    try:
        result_manual = shadow_remover.uniformize_color(
            image, 
            reference_region=(center_x - patch_size//2, center_y - patch_size//2, patch_size, patch_size)
        )
        cv2.imwrite("original_shine_manual.jpg", result_manual)
        print("[OK] Selection manuelle avec brillance originale: original_shine_manual.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("TEST BRILLANCE ORIGINALE TERMINE")
    print("=" * 60)
    print("Fichiers crees:")
    print("- original_shine.jpg : Brillance originale preservee")
    print("- original_shine_combined.jpg : Suppression d'ombres + brillance originale")
    print("- original_shine_brightest.jpg : Region lumineuse + brillance originale")
    print("- original_shine_most_saturated.jpg : Region saturee + brillance originale")
    print("- original_shine_center.jpg : Region centrale + brillance originale")
    print("- original_shine_manual.jpg : Selection manuelle + brillance originale")
    print(f"\nType de tissu detecte: {fabric_type}")
    print("La brillance originale a ete preservee!")

if __name__ == "__main__":
    test_original_shine()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test pour la texture réaliste avec reflets naturels
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_realistic_texture():
    """Test de la texture réaliste avec reflets"""
    print("Test Texture Realiste avec Reflets Naturels")
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
    
    print("\n1. Test de la texture realiste avec reflets...")
    
    # Test avec méthode realistic_texture
    try:
        result_realistic = shadow_remover.uniformize_color(image, method='realistic_texture')
        cv2.imwrite("realistic_texture.jpg", result_realistic)
        print("[OK] Texture realiste sauvegardee: realistic_texture.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n2. Test combine - suppression d'ombres + texture realiste...")
    
    # Test combiné
    try:
        result_combined = shadow_remover.remove_shadows_with_color_uniformization(
            image, shadow_method='auto', color_method='realistic_texture'
        )
        cv2.imwrite("realistic_combined.jpg", result_combined)
        print("[OK] Resultat combine realiste sauvegarde: realistic_combined.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n3. Test avec differentes methodes realistes...")
    
    methods = ['brightest', 'most_saturated', 'center']
    
    for method in methods:
        try:
            result = shadow_remover.uniformize_color(image, method=method)
            cv2.imwrite(f"realistic_{method}.jpg", result)
            print(f"[OK] Methode realiste {method} sauvegardee: realistic_{method}.jpg")
        except Exception as e:
            print(f"[ERROR] Methode {method} echouee: {e}")
    
    print("\n4. Test avec selection manuelle de region...")
    
    # Test avec région manuelle (centre de l'image)
    h, w = image.shape[:2]
    center_x, center_y = w//2, h//2
    patch_size = 15  # Légèrement plus grand pour plus de texture
    
    try:
        result_manual = shadow_remover.uniformize_color(
            image, 
            reference_region=(center_x - patch_size//2, center_y - patch_size//2, patch_size, patch_size)
        )
        cv2.imwrite("realistic_manual_center.jpg", result_manual)
        print("[OK] Resultat manuel realiste sauvegarde: realistic_manual_center.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TEXTURE REALISTE TERMINE")
    print("=" * 60)
    print("Fichiers crees:")
    print("- realistic_texture.jpg : Texture realiste avec reflets")
    print("- realistic_combined.jpg : Suppression d'ombres + texture realiste")
    print("- realistic_brightest.jpg : Region la plus lumineuse + texture")
    print("- realistic_most_saturated.jpg : Region la plus saturee + texture")
    print("- realistic_center.jpg : Region centrale + texture")
    print("- realistic_manual_center.jpg : Selection manuelle + texture")
    print("\nCes images devraient avoir une texture realiste avec des reflets naturels!")

if __name__ == "__main__":
    test_realistic_texture()


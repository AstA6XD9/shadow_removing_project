#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test final pour la fonctionnalité d'uniformisation des couleurs parfaite
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_final():
    """Test final de la fonctionnalité d'uniformisation des couleurs"""
    print("Test Final - Uniformisation des Couleurs Parfaite")
    print("=" * 60)
    
    # Chemin de l'image
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image non trouvée: {image_path}")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    print(f"Image chargée: {image.shape}")
    
    # Initialiser le shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    print("\n1. Test de l'uniformisation des couleurs parfaite...")
    
    # Test avec méthode perfect_patch
    try:
        result_perfect = shadow_remover.uniformize_color(image, method='perfect_patch')
        cv2.imwrite("final_perfect_color.jpg", result_perfect)
        print("[OK] Couleur parfaitement uniforme sauvegardee: final_perfect_color.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n2. Test de la suppression d'ombres + uniformisation...")
    
    # Test combiné
    try:
        result_combined = shadow_remover.remove_shadows_with_color_uniformization(
            image, shadow_method='auto', color_method='perfect_patch'
        )
        cv2.imwrite("final_combined.jpg", result_combined)
        print("[OK] Resultat combine sauvegarde: final_combined.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n3. Test avec selection manuelle de region...")
    
    # Test avec région manuelle (centre de l'image)
    h, w = image.shape[:2]
    center_x, center_y = w//2, h//2
    patch_size = 10
    
    try:
        result_manual = shadow_remover.uniformize_color(
            image, 
            reference_region=(center_x - patch_size//2, center_y - patch_size//2, patch_size, patch_size)
        )
        cv2.imwrite("final_manual_center.jpg", result_manual)
        print("[OK] Resultat manuel sauvegarde: final_manual_center.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n4. Test avec differentes methodes...")
    
    methods = ['brightest', 'most_saturated', 'center']
    
    for method in methods:
        try:
            result = shadow_remover.uniformize_color(image, method=method)
            cv2.imwrite(f"final_{method}.jpg", result)
            print(f"[OK] Methode {method} sauvegardee: final_{method}.jpg")
        except Exception as e:
            print(f"[ERROR] Methode {method} echouee: {e}")
    
    print("\n" + "=" * 60)
    print("TEST FINAL TERMINÉ")
    print("=" * 60)
    print("Fichiers créés:")
    print("- final_perfect_color.jpg : Couleur parfaitement uniforme")
    print("- final_combined.jpg : Suppression d'ombres + couleur parfaite")
    print("- final_manual_center.jpg : Sélection manuelle du centre")
    print("- final_brightest.jpg : Région la plus lumineuse")
    print("- final_most_saturated.jpg : Région la plus saturée")
    print("- final_center.jpg : Région centrale")
    print("\nCes images devraient avoir une couleur parfaitement uniforme!")

if __name__ == "__main__":
    test_final()

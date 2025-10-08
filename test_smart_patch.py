#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test pour le remplacement intelligent de patches
Remplace les zones avec coefficient élevé par des zones avec coefficient faible
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_smart_patch_replacement():
    """Test pour le remplacement intelligent de patches"""
    print("Test Remplacement Intelligent de Patches")
    print("=" * 60)
    print("Remplace les zones problematiques par des zones similaires")
    print("Coefficient faible = zone uniforme (bonne)")
    print("Coefficient eleve = zone variable (problematique)")
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
    
    print("\n1. Test avec differentes valeurs de seuil...")
    
    # Test avec différents seuils
    thresholds = [0.2, 0.3, 0.4, 0.5]
    
    for threshold in thresholds:
        try:
            result = shadow_remover._uniformize_color_smart_patch(
                image, threshold=threshold
            )
            cv2.imwrite(f"smart_patch_threshold_{threshold}.jpg", result)
            print(f"[OK] Seuil {threshold}: smart_patch_threshold_{threshold}.jpg")
        except Exception as e:
            print(f"[ERROR] Seuil {threshold} echoue: {e}")
    
    print("\n2. Test avec selection manuelle de region...")
    
    # Test avec région manuelle (centre de l'image)
    h, w = image.shape[:2]
    center_x, center_y = w//2, h//2
    patch_size = 30
    
    try:
        result_manual = shadow_remover._uniformize_color_smart_patch(
            image, 
            reference_region=(center_x - patch_size//2, center_y - patch_size//2, patch_size, patch_size),
            threshold=0.3
        )
        cv2.imwrite("smart_patch_manual.jpg", result_manual)
        print("[OK] Selection manuelle: smart_patch_manual.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n3. Test combine - suppression d'ombres + smart patch...")
    
    # Test combiné
    try:
        # D'abord suppression d'ombres
        shadow_removed = shadow_remover.remove_shadows(image, method='auto')
        
        # Puis smart patch replacement
        result_combined = shadow_remover._uniformize_color_smart_patch(
            shadow_removed, threshold=0.3
        )
        cv2.imwrite("smart_patch_combined.jpg", result_combined)
        print("[OK] Resultat combine: smart_patch_combined.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n4. Test avec differentes regions de reference...")
    
    # Test avec différentes régions
    regions = [
        ("center", (w//2 - 15, h//2 - 15, 30, 30)),
        ("top_left", (w//4, h//4, 30, 30)),
        ("top_right", (3*w//4 - 15, h//4, 30, 30)),
        ("bottom_left", (w//4, 3*h//4 - 15, 30, 30)),
        ("bottom_right", (3*w//4 - 15, 3*h//4 - 15, 30, 30))
    ]
    
    for region_name, region in regions:
        try:
            result = shadow_remover._uniformize_color_smart_patch(
                image, reference_region=region, threshold=0.3
            )
            cv2.imwrite(f"smart_patch_{region_name}.jpg", result)
            print(f"[OK] Region {region_name}: smart_patch_{region_name}.jpg")
        except Exception as e:
            print(f"[ERROR] Region {region_name} echoue: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SMART PATCH TERMINE")
    print("=" * 60)
    print("Fichiers crees:")
    print("- smart_patch_threshold_0.2.jpg : Seuil 0.2 (tres strict)")
    print("- smart_patch_threshold_0.3.jpg : Seuil 0.3 (recommandé)")
    print("- smart_patch_threshold_0.4.jpg : Seuil 0.4 (modéré)")
    print("- smart_patch_threshold_0.5.jpg : Seuil 0.5 (permissif)")
    print("- smart_patch_manual.jpg : Selection manuelle")
    print("- smart_patch_combined.jpg : Suppression d'ombres + smart patch")
    print("- smart_patch_center.jpg : Region centrale")
    print("- smart_patch_top_left.jpg : Region haut-gauche")
    print("- smart_patch_top_right.jpg : Region haut-droite")
    print("- smart_patch_bottom_left.jpg : Region bas-gauche")
    print("- smart_patch_bottom_right.jpg : Region bas-droite")
    print("\nPrincipe:")
    print("- Coefficient faible = zone uniforme (bonne zone)")
    print("- Coefficient eleve = zone variable (zone a remplacer)")
    print("- Remplacement par la zone la plus similaire")

if __name__ == "__main__":
    test_smart_patch_replacement()



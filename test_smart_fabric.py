#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test pour l'approche intelligente de traitement des tissus
Analyse le type de tissu et applique le traitement approprié
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_smart_fabric():
    """Test de l'approche intelligente pour les tissus"""
    print("Test Approche Intelligente pour Tissus")
    print("=" * 60)
    print("Analyse le type de tissu et applique le traitement approprie")
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
    
    if fabric_type == 'smooth_solid':
        print("  -> Tissu lisse (velours, soie) - Brillance subtile")
    elif fabric_type == 'matte_solid':
        print("  -> Tissu mat (coton, lin) - Texture mate")
    elif fabric_type == 'textured':
        print("  -> Tissu avec motifs - Preservation des motifs")
    else:
        print("  -> Tissu standard - Equilibre brillance/texture")
    
    print("\n2. Test de l'uniformisation intelligente...")
    
    # Test avec méthode smart_fabric
    try:
        result_smart = shadow_remover.uniformize_color(image, method='smart_fabric')
        cv2.imwrite("smart_fabric.jpg", result_smart)
        print("[OK] Tissu intelligent sauvegarde: smart_fabric.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n3. Test combine - suppression d'ombres + tissu intelligent...")
    
    # Test combiné
    try:
        result_combined = shadow_remover.remove_shadows_with_color_uniformization(
            image, shadow_method='auto', color_method='smart_fabric'
        )
        cv2.imwrite("smart_combined.jpg", result_combined)
        print("[OK] Resultat combine intelligent sauvegarde: smart_combined.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n4. Test avec differentes methodes intelligentes...")
    
    methods = ['brightest', 'most_saturated', 'center']
    
    for method in methods:
        try:
            result = shadow_remover.uniformize_color(image, method=method)
            cv2.imwrite(f"smart_{method}.jpg", result)
            print(f"[OK] Methode intelligente {method} sauvegardee: smart_{method}.jpg")
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
        cv2.imwrite("smart_manual_center.jpg", result_manual)
        print("[OK] Resultat manuel intelligent sauvegarde: smart_manual_center.jpg")
    except Exception as e:
        print(f"[ERROR] Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TISSU INTELLIGENT TERMINE")
    print("=" * 60)
    print("Fichiers crees:")
    print("- smart_fabric.jpg : Tissu intelligent (analyse automatique)")
    print("- smart_combined.jpg : Suppression d'ombres + tissu intelligent")
    print("- smart_brightest.jpg : Region lumineuse + traitement intelligent")
    print("- smart_most_saturated.jpg : Region saturee + traitement intelligent")
    print("- smart_center.jpg : Region centrale + traitement intelligent")
    print("- smart_manual_center.jpg : Selection manuelle + traitement intelligent")
    print(f"\nType de tissu detecte: {fabric_type}")
    print("Le traitement a ete adapte au type de tissu detecte!")

if __name__ == "__main__":
    test_smart_fabric()



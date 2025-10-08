#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test final des améliorations avec seuil adaptatif
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_final_improvements():
    """Test final des améliorations"""
    print("Test Final des Ameliorations avec Seuil Adaptatif")
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
    
    # Test de la detection d'ombres avec seuil adaptatif
    print("\nTest de la Detection d'Ombres avec Seuil Adaptatif")
    print("-" * 50)
    try:
        shadow_mask, shadow_probability = shadow_remover._advanced_shadow_pattern_detection(image)
        
        # Statistiques
        shadow_percentage = np.sum(shadow_mask) / shadow_mask.size * 100
        print(f"Pourcentage de pixels detectes comme ombres: {shadow_percentage:.1f}%")
        
        # Sauvegarder les resultats
        shadow_prob_vis = (shadow_probability * 255).astype(np.uint8)
        cv2.imwrite("carte_probabilite_adaptative.jpg", shadow_prob_vis)
        
        shadow_mask_vis = (shadow_mask * 255).astype(np.uint8)
        cv2.imwrite("masque_ombres_adaptatif.jpg", shadow_mask_vis)
        
        print("OK - Detection adaptative reussie")
        print("Sauvegarde: carte_probabilite_adaptative.jpg")
        print("Sauvegarde: masque_ombres_adaptatif.jpg")
        
    except Exception as e:
        print(f"ERREUR lors de la detection: {e}")
    
    # Test des methodes ameliorees
    print("\nTest des Methodes Ameliorees")
    print("-" * 30)
    
    methods = ['intelligent', 'fabric_preserving', 'auto']
    
    for method in methods:
        try:
            result = shadow_remover.remove_shadows(image, method=method)
            if result is not None:
                metrics = shadow_remover.assess_quality(image, result)
                print(f"{method}:")
                print(f"  Qualite: {metrics['overall_quality']:.3f}")
                print(f"  Contraste: {metrics['contrast_improvement']:.3f}")
                print(f"  Ombres: {metrics['shadow_reduction']:.3f}")
                print(f"  Couleurs: {metrics['color_preservation']:.3f}")
                
                # Sauvegarder
                output_path = f"final_{method}.jpg"
                cv2.imwrite(output_path, result)
                print(f"  Sauvegarde: {output_path}")
            else:
                print(f"{method}: ECHEC")
        except Exception as e:
            print(f"{method}: ERREUR = {e}")
    
    print("\n" + "=" * 60)
    print("TEST FINAL TERMINE")
    print("=" * 60)
    print("Les nouvelles methodes utilisent maintenant un seuil adaptatif")
    print("pour mieux differencier les ombres des motifs du tissu!")

if __name__ == "__main__":
    test_final_improvements()

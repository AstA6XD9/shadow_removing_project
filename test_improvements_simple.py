#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple des améliorations de suppression d'ombres
Version sans emojis pour éviter les problèmes d'encodage
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_improvements():
    """Test des améliorations"""
    print("Test des Ameliorations de Suppression d'Ombres")
    print("=" * 60)
    print("Nouvelles fonctionnalites:")
    print("- Detection intelligente ombres vs motifs")
    print("- Preservation des caracteristiques du tissu")
    print("- Methodes 'intelligent' et 'fabric_preserving'")
    print("=" * 60)
    
    # Chemin de l'image
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"Image non trouvee: {image_path}")
        print("Veuillez verifier le chemin de l'image")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image: {image_path}")
        return
    
    print(f"Image chargee: {image.shape}")
    
    # Initialiser le shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Test 1: Methode intelligente
    print("\nTest 1: Methode 'intelligent'")
    print("-" * 30)
    try:
        result_intelligent = shadow_remover.remove_shadows(image, method='intelligent')
        if result_intelligent is not None:
            cv2.imwrite("resultat_intelligent.jpg", result_intelligent)
            print("OK - Methode 'intelligent' reussie")
            print("Sauvegarde: resultat_intelligent.jpg")
            
            # Evaluer la qualite
            metrics = shadow_remover.assess_quality(image, result_intelligent)
            print(f"Qualite: {metrics['overall_quality']:.3f}")
            print(f"Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"Couleurs: {metrics['color_preservation']:.3f}")
        else:
            print("ERREUR - Methode 'intelligent' echouee")
    except Exception as e:
        print(f"ERREUR avec methode 'intelligent': {e}")
    
    # Test 2: Methode fabric_preserving
    print("\nTest 2: Methode 'fabric_preserving'")
    print("-" * 30)
    try:
        result_fabric = shadow_remover.remove_shadows(image, method='fabric_preserving')
        if result_fabric is not None:
            cv2.imwrite("resultat_fabric_preserving.jpg", result_fabric)
            print("OK - Methode 'fabric_preserving' reussie")
            print("Sauvegarde: resultat_fabric_preserving.jpg")
            
            # Evaluer la qualite
            metrics = shadow_remover.assess_quality(image, result_fabric)
            print(f"Qualite: {metrics['overall_quality']:.3f}")
            print(f"Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"Couleurs: {metrics['color_preservation']:.3f}")
        else:
            print("ERREUR - Methode 'fabric_preserving' echouee")
    except Exception as e:
        print(f"ERREUR avec methode 'fabric_preserving': {e}")
    
    # Test 3: Selection automatique amelioree
    print("\nTest 3: Selection automatique amelioree")
    print("-" * 30)
    try:
        result_auto = shadow_remover.remove_shadows(image, method='auto')
        if result_auto is not None:
            cv2.imwrite("resultat_auto_ameliore.jpg", result_auto)
            print("OK - Selection automatique amelioree reussie")
            print("Sauvegarde: resultat_auto_ameliore.jpg")
            
            # Evaluer la qualite
            metrics = shadow_remover.assess_quality(image, result_auto)
            print(f"Qualite: {metrics['overall_quality']:.3f}")
            print(f"Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"Couleurs: {metrics['color_preservation']:.3f}")
        else:
            print("ERREUR - Selection automatique amelioree echouee")
    except Exception as e:
        print(f"ERREUR avec selection automatique: {e}")
    
    # Test 4: Detection d'ombres avancee
    print("\nTest 4: Detection d'ombres avancee")
    print("-" * 30)
    try:
        # Test de la detection d'ombres
        shadow_mask, shadow_probability = shadow_remover._advanced_shadow_pattern_detection(image)
        
        # Sauvegarder la carte de probabilite d'ombres
        shadow_prob_vis = (shadow_probability * 255).astype(np.uint8)
        cv2.imwrite("carte_probabilite_ombres.jpg", shadow_prob_vis)
        print("OK - Carte de probabilite d'ombres generee")
        print("Sauvegarde: carte_probabilite_ombres.jpg")
        
        # Sauvegarder le masque d'ombres
        shadow_mask_vis = (shadow_mask * 255).astype(np.uint8)
        cv2.imwrite("masque_ombres.jpg", shadow_mask_vis)
        print("OK - Masque d'ombres genere")
        print("Sauvegarde: masque_ombres.jpg")
        
        # Statistiques
        shadow_percentage = np.sum(shadow_mask) / shadow_mask.size * 100
        print(f"Pourcentage de pixels detectes comme ombres: {shadow_percentage:.1f}%")
        
    except Exception as e:
        print(f"ERREUR lors de la detection d'ombres: {e}")
    
    # Test 5: Comparaison avec methodes classiques
    print("\nTest 5: Comparaison avec methodes classiques")
    print("-" * 30)
    
    methods_to_compare = ['retinex', 'texture_aware', 'intelligent', 'fabric_preserving']
    
    for method in methods_to_compare:
        try:
            result = shadow_remover.remove_shadows(image, method=method)
            if result is not None:
                metrics = shadow_remover.assess_quality(image, result)
                print(f"{method}: Qualite = {metrics['overall_quality']:.3f}")
                
                # Sauvegarder le resultat
                output_path = f"comparaison_{method}.jpg"
                cv2.imwrite(output_path, result)
                print(f"  Sauvegarde: {output_path}")
            else:
                print(f"{method}: ECHEC")
        except Exception as e:
            print(f"{method}: ERREUR = {e}")
    
    print("\n" + "=" * 60)
    print("TEST DES AMELIORATIONS TERMINE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- resultat_intelligent.jpg : Methode intelligente")
    print("- resultat_fabric_preserving.jpg : Methode preservant le tissu")
    print("- resultat_auto_ameliore.jpg : Selection automatique amelioree")
    print("- carte_probabilite_ombres.jpg : Carte de probabilite d'ombres")
    print("- masque_ombres.jpg : Masque des zones d'ombres detectees")
    print("- comparaison_*.jpg : Resultats de comparaison")
    print("\nLes nouvelles methodes differencient mieux les ombres des motifs!")

def main():
    """Fonction principale"""
    print("Test des Ameliorations Avancees de Suppression d'Ombres")
    print("Objectif: Differencier efficacement les ombres des motifs du tissu")
    print("=" * 60)
    
    # Test des ameliorations
    test_improvements()
    
    print("\nResume des Ameliorations:")
    print("1. Detection intelligente ombres vs motifs")
    print("2. Preservation des caracteristiques du tissu")
    print("3. Nouvelles methodes 'intelligent' et 'fabric_preserving'")
    print("4. Selection automatique amelioree")
    print("5. Visualisation de la detection d'ombres")
    print("\nLes resultats devraient maintenant mieux preserver:")
    print("- La couleur originale du tissu")
    print("- La brillance et les reflets")
    print("- Les motifs et textures")
    print("- Tout en supprimant efficacement les vraies ombres")

if __name__ == "__main__":
    main()

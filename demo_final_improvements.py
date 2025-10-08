#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration finale des améliorations
Montre les résultats avant/après avec les nouvelles méthodes
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def demo_final_improvements():
    """Démonstration finale des améliorations"""
    print("Demonstration Finale des Ameliorations")
    print("=" * 60)
    print("Objectif: Montrer comment les nouvelles methodes")
    print("differencient efficacement les ombres des motifs du tissu")
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
    
    # 1. Analyse de l'image originale
    print("\n1. Analyse de l'Image Originale")
    print("-" * 30)
    
    # Analyser les caracteristiques
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness_mean = np.mean(gray)
    brightness_std = np.std(gray)
    texture_complexity = np.std(cv2.Laplacian(gray, cv2.CV_64F))
    
    print(f"Brillance moyenne: {brightness_mean:.1f}")
    print(f"Variation de brillance: {brightness_std:.1f}")
    print(f"Complexite de texture: {texture_complexity:.1f}")
    
    # 2. Detection d'ombres intelligente
    print("\n2. Detection d'Ombres Intelligente")
    print("-" * 30)
    
    try:
        shadow_mask, shadow_probability = shadow_remover._advanced_shadow_pattern_detection(image)
        
        shadow_percentage = np.sum(shadow_mask) / shadow_mask.size * 100
        print(f"Pixels detectes comme ombres: {shadow_percentage:.1f}%")
        
        # Sauvegarder la visualisation
        shadow_prob_vis = (shadow_probability * 255).astype(np.uint8)
        cv2.imwrite("demo_detection_ombres.jpg", shadow_prob_vis)
        
        shadow_mask_vis = (shadow_mask * 255).astype(np.uint8)
        cv2.imwrite("demo_masque_ombres.jpg", shadow_mask_vis)
        
        print("OK - Detection intelligente reussie")
        print("Sauvegarde: demo_detection_ombres.jpg")
        print("Sauvegarde: demo_masque_ombres.jpg")
        
    except Exception as e:
        print(f"ERREUR lors de la detection: {e}")
    
    # 3. Comparaison des methodes
    print("\n3. Comparaison des Methodes")
    print("-" * 30)
    
    methods = {
        'Methode Classique (Retinex)': 'retinex',
        'Methode Texture-Aware': 'texture_aware', 
        'Methode Intelligente': 'intelligent',
        'Methode Fabric-Preserving': 'fabric_preserving'
    }
    
    results = {}
    
    for method_name, method_key in methods.items():
        try:
            result = shadow_remover.remove_shadows(image, method=method_key)
            if result is not None:
                metrics = shadow_remover.assess_quality(image, result)
                results[method_name] = {
                    'result': result,
                    'metrics': metrics
                }
                
                print(f"{method_name}:")
                print(f"  Qualite: {metrics['overall_quality']:.3f}")
                print(f"  Contraste: {metrics['contrast_improvement']:.3f}")
                print(f"  Ombres: {metrics['shadow_reduction']:.3f}")
                print(f"  Couleurs: {metrics['color_preservation']:.3f}")
                
                # Sauvegarder
                filename = f"demo_{method_key}.jpg"
                cv2.imwrite(filename, result)
                print(f"  Sauvegarde: {filename}")
            else:
                print(f"{method_name}: ECHEC")
        except Exception as e:
            print(f"{method_name}: ERREUR = {e}")
    
    # 4. Analyse comparative
    print("\n4. Analyse Comparative")
    print("-" * 30)
    
    if results:
        # Trouver la meilleure methode
        best_method = max(results.items(), key=lambda x: x[1]['metrics']['overall_quality'])
        print(f"Meilleure methode: {best_method[0]}")
        print(f"Score de qualite: {best_method[1]['metrics']['overall_quality']:.3f}")
        
        # Analyser la preservation des couleurs
        color_scores = [(name, data['metrics']['color_preservation']) for name, data in results.items()]
        color_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("\nClassement par preservation des couleurs:")
        for i, (name, score) in enumerate(color_scores, 1):
            print(f"  {i}. {name}: {score:.3f}")
    
    # 5. Test de la selection automatique
    print("\n5. Test de la Selection Automatique")
    print("-" * 30)
    
    try:
        result_auto = shadow_remover.remove_shadows(image, method='auto')
        if result_auto is not None:
            metrics = shadow_remover.assess_quality(image, result_auto)
            print("Selection automatique:")
            print(f"  Qualite: {metrics['overall_quality']:.3f}")
            print(f"  Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"  Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"  Couleurs: {metrics['color_preservation']:.3f}")
            
            cv2.imwrite("demo_selection_auto.jpg", result_auto)
            print("  Sauvegarde: demo_selection_auto.jpg")
        else:
            print("Selection automatique: ECHEC")
    except Exception as e:
        print(f"Selection automatique: ERREUR = {e}")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION FINALE TERMINEE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- demo_detection_ombres.jpg : Carte de probabilite d'ombres")
    print("- demo_masque_ombres.jpg : Masque des zones d'ombres")
    print("- demo_retinex.jpg : Methode classique")
    print("- demo_texture_aware.jpg : Methode texture-aware")
    print("- demo_intelligent.jpg : Methode intelligente")
    print("- demo_fabric_preserving.jpg : Methode preservant le tissu")
    print("- demo_selection_auto.jpg : Selection automatique")
    print("\nLes nouvelles methodes 'intelligent' et 'fabric_preserving'")
    print("devraient montrer une meilleure preservation des caracteristiques du tissu!")

def main():
    """Fonction principale"""
    print("Demonstration Finale des Ameliorations de Suppression d'Ombres")
    print("=" * 70)
    print("Nouvelles fonctionnalites implementees:")
    print("1. Detection intelligente ombres vs motifs")
    print("2. Preservation des caracteristiques du tissu")
    print("3. Methodes 'intelligent' et 'fabric_preserving'")
    print("4. Selection automatique amelioree")
    print("5. Seuil adaptatif pour la detection d'ombres")
    print("=" * 70)
    
    # Demonstration
    demo_final_improvements()
    
    print("\nResume des Ameliorations:")
    print("=" * 40)
    print("✅ Detection intelligente ombres vs motifs")
    print("✅ Preservation de la couleur originale")
    print("✅ Conservation de la brillance et des reflets")
    print("✅ Maintien des motifs et textures du tissu")
    print("✅ Suppression efficace des vraies ombres")
    print("✅ Seuil adaptatif pour une detection precise")
    print("\nLes resultats devraient maintenant etre plus fideles")
    print("a l'apparence originale du tissu tout en supprimant")
    print("efficacement les ombres indesirables!")

if __name__ == "__main__":
    main()

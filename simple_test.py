#!/usr/bin/env python3
"""
Test simplifié du système de suppression d'ombres
Évite les problèmes d'optimisation complexes
"""

import cv2
import os
from shadow_removing import AdvancedShadowRemover

def simple_shadow_removal_test():
    """Test simplifié de suppression d'ombres"""
    print("🎯 Test simplifié de suppression d'ombres")
    print("="*50)
    
    # Initialiser le système
    shadow_remover = AdvancedShadowRemover()
    
    # Chemins des images
    image_paths = {
        "IMG_6020": "C:/Users/eloua/OneDrive/Images/IMG_6020.JPG",
        "IMG_6043": "C:/Users/eloua/OneDrive/Images/IMG_6043.JPG"
    }
    
    for image_name, image_path in image_paths.items():
        print(f"\n📸 Traitement de {image_name}")
        print("-" * 30)
        
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Impossible de charger {image_path}")
            continue
        
        print(f"✅ Image chargée - Taille: {image.shape}")
        
        # Test de méthodes simples (sans optimisation)
        methods_to_test = ['retinex', 'adaptive_histogram', 'gradient_domain']
        
        best_method = None
        best_score = -1
        best_result = None
        
        for method in methods_to_test:
            print(f"\n🧪 Test de {method}...")
            try:
                result = shadow_remover.remove_shadows(image, method=method)
                if result is not None:
                    # Évaluer la qualité
                    metrics = shadow_remover.assess_quality(image, result)
                    score = metrics['overall_quality']
                    
                    print(f"  📊 Qualité: {score:.3f}")
                    print(f"  🎨 Contraste: {metrics['contrast_improvement']:.3f}")
                    print(f"  🌑 Ombres: {metrics['shadow_reduction']:.3f}")
                    print(f"  🎨 Couleurs: {metrics['color_preservation']:.3f}")
                    
                    # Sauvegarder le résultat
                    output_path = f"simple_{method}_{image_name}.jpg"
                    cv2.imwrite(output_path, result)
                    print(f"  💾 Sauvegardé: {output_path}")
                    
                    # Garder le meilleur
                    if score > best_score:
                        best_score = score
                        best_method = method
                        best_result = result
                else:
                    print(f"  ❌ Échec de {method}")
            except Exception as e:
                print(f"  ❌ Erreur avec {method}: {e}")
        
        # Test de sélection automatique
        print(f"\n🤖 Test de sélection automatique...")
        try:
            auto_result = shadow_remover.remove_shadows(image, method='auto')
            if auto_result is not None:
                auto_metrics = shadow_remover.assess_quality(image, auto_result)
                auto_score = auto_metrics['overall_quality']
                print(f"  📊 Qualité automatique: {auto_score:.3f}")
                
                output_path = f"simple_auto_{image_name}.jpg"
                cv2.imwrite(output_path, auto_result)
                print(f"  💾 Sauvegardé: {output_path}")
                
                # Comparer avec le meilleur manuel
                if auto_score > best_score:
                    best_score = auto_score
                    best_method = "auto"
                    best_result = auto_result
        except Exception as e:
            print(f"  ❌ Erreur avec sélection automatique: {e}")
        
        # Résumé
        if best_result is not None:
            print(f"\n🏆 Meilleure méthode pour {image_name}: {best_method} (score: {best_score:.3f})")
            
            # Sauvegarder le meilleur résultat
            best_output = f"BEST_{image_name}.jpg"
            cv2.imwrite(best_output, best_result)
            print(f"💎 Meilleur résultat sauvegardé: {best_output}")
        else:
            print(f"\n❌ Aucune méthode n'a fonctionné pour {image_name}")
    
    print(f"\n🎉 Test terminé! Vérifiez les fichiers de résultats.")

if __name__ == "__main__":
    simple_shadow_removal_test()


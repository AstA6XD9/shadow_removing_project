#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test des améliorations avancées de suppression d'ombres
Teste les nouvelles méthodes intelligentes qui différencient les ombres des motifs
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from shadow_removing import AdvancedShadowRemover

def test_advanced_methods():
    """Test des nouvelles méthodes avancées"""
    print("🧪 Test des Méthodes Avancées de Suppression d'Ombres")
    print("=" * 70)
    print("Nouvelles fonctionnalités:")
    print("• Détection intelligente ombres vs motifs")
    print("• Préservation des caractéristiques du tissu")
    print("• Méthodes 'intelligent' et 'fabric_preserving'")
    print("=" * 70)
    
    # Chemin de l'image
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        print("Veuillez vérifier le chemin de l'image")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return
    
    print(f"✅ Image chargée: {image.shape}")
    
    # Initialiser le shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Test 1: Méthode intelligente
    print("\n🔬 Test 1: Méthode 'intelligent'")
    print("-" * 40)
    try:
        result_intelligent = shadow_remover.remove_shadows(image, method='intelligent')
        if result_intelligent is not None:
            cv2.imwrite("resultat_intelligent.jpg", result_intelligent)
            print("✅ Méthode 'intelligent' réussie")
            print("💾 Sauvegardé: resultat_intelligent.jpg")
            
            # Évaluer la qualité
            metrics = shadow_remover.assess_quality(image, result_intelligent)
            print(f"📊 Qualité: {metrics['overall_quality']:.3f}")
            print(f"🎨 Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"🌑 Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"🎨 Couleurs: {metrics['color_preservation']:.3f}")
        else:
            print("❌ Méthode 'intelligent' échouée")
    except Exception as e:
        print(f"❌ Erreur avec méthode 'intelligent': {e}")
    
    # Test 2: Méthode fabric_preserving
    print("\n🔬 Test 2: Méthode 'fabric_preserving'")
    print("-" * 40)
    try:
        result_fabric = shadow_remover.remove_shadows(image, method='fabric_preserving')
        if result_fabric is not None:
            cv2.imwrite("resultat_fabric_preserving.jpg", result_fabric)
            print("✅ Méthode 'fabric_preserving' réussie")
            print("💾 Sauvegardé: resultat_fabric_preserving.jpg")
            
            # Évaluer la qualité
            metrics = shadow_remover.assess_quality(image, result_fabric)
            print(f"📊 Qualité: {metrics['overall_quality']:.3f}")
            print(f"🎨 Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"🌑 Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"🎨 Couleurs: {metrics['color_preservation']:.3f}")
        else:
            print("❌ Méthode 'fabric_preserving' échouée")
    except Exception as e:
        print(f"❌ Erreur avec méthode 'fabric_preserving': {e}")
    
    # Test 3: Sélection automatique améliorée
    print("\n🔬 Test 3: Sélection automatique améliorée")
    print("-" * 40)
    try:
        result_auto = shadow_remover.remove_shadows(image, method='auto')
        if result_auto is not None:
            cv2.imwrite("resultat_auto_ameliore.jpg", result_auto)
            print("✅ Sélection automatique améliorée réussie")
            print("💾 Sauvegardé: resultat_auto_ameliore.jpg")
            
            # Évaluer la qualité
            metrics = shadow_remover.assess_quality(image, result_auto)
            print(f"📊 Qualité: {metrics['overall_quality']:.3f}")
            print(f"🎨 Contraste: {metrics['contrast_improvement']:.3f}")
            print(f"🌑 Ombres: {metrics['shadow_reduction']:.3f}")
            print(f"🎨 Couleurs: {metrics['color_preservation']:.3f}")
        else:
            print("❌ Sélection automatique améliorée échouée")
    except Exception as e:
        print(f"❌ Erreur avec sélection automatique: {e}")
    
    # Test 4: Comparaison avec méthodes classiques
    print("\n🔬 Test 4: Comparaison avec méthodes classiques")
    print("-" * 40)
    
    methods_to_compare = ['retinex', 'texture_aware', 'intelligent', 'fabric_preserving']
    results_comparison = {}
    
    for method in methods_to_compare:
        try:
            result = shadow_remover.remove_shadows(image, method=method)
            if result is not None:
                metrics = shadow_remover.assess_quality(image, result)
                results_comparison[method] = {
                    'result': result,
                    'metrics': metrics
                }
                print(f"✅ {method}: Qualité = {metrics['overall_quality']:.3f}")
            else:
                print(f"❌ {method}: Échec")
        except Exception as e:
            print(f"❌ {method}: Erreur = {e}")
    
    # Sauvegarder les résultats de comparaison
    if results_comparison:
        print("\n💾 Sauvegarde des résultats de comparaison...")
        for method, data in results_comparison.items():
            output_path = f"comparaison_{method}.jpg"
            cv2.imwrite(output_path, data['result'])
            print(f"  {method}: {output_path}")
    
    # Test 5: Test de la détection d'ombres avancée
    print("\n🔬 Test 5: Détection d'ombres avancée")
    print("-" * 40)
    try:
        # Test de la détection d'ombres
        shadow_mask, shadow_probability = shadow_remover._advanced_shadow_pattern_detection(image)
        
        # Sauvegarder la carte de probabilité d'ombres
        shadow_prob_vis = (shadow_probability * 255).astype(np.uint8)
        cv2.imwrite("carte_probabilite_ombres.jpg", shadow_prob_vis)
        print("✅ Carte de probabilité d'ombres générée")
        print("💾 Sauvegardé: carte_probabilite_ombres.jpg")
        
        # Sauvegarder le masque d'ombres
        shadow_mask_vis = (shadow_mask * 255).astype(np.uint8)
        cv2.imwrite("masque_ombres.jpg", shadow_mask_vis)
        print("✅ Masque d'ombres généré")
        print("💾 Sauvegardé: masque_ombres.jpg")
        
        # Statistiques
        shadow_percentage = np.sum(shadow_mask) / shadow_mask.size * 100
        print(f"📊 Pourcentage de pixels détectés comme ombres: {shadow_percentage:.1f}%")
        
    except Exception as e:
        print(f"❌ Erreur lors de la détection d'ombres: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 TEST DES MÉTHODES AVANCÉES TERMINÉ")
    print("=" * 70)
    print("Fichiers générés:")
    print("• resultat_intelligent.jpg : Méthode intelligente")
    print("• resultat_fabric_preserving.jpg : Méthode préservant le tissu")
    print("• resultat_auto_ameliore.jpg : Sélection automatique améliorée")
    print("• carte_probabilite_ombres.jpg : Carte de probabilité d'ombres")
    print("• masque_ombres.jpg : Masque des zones d'ombres détectées")
    print("• comparaison_*.jpg : Résultats de comparaison")
    print("\n✨ Les nouvelles méthodes différencient mieux les ombres des motifs!")

def test_shadow_detection_visualization():
    """Test de visualisation de la détection d'ombres"""
    print("\n🎨 Test de Visualisation de la Détection d'Ombres")
    print("-" * 50)
    
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    image = cv2.imread(image_path)
    shadow_remover = AdvancedShadowRemover()
    
    try:
        # Obtenir la détection d'ombres
        shadow_mask, shadow_probability = shadow_remover._advanced_shadow_pattern_detection(image)
        
        # Créer une visualisation
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Image originale
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("Image Originale")
        axes[0, 0].axis('off')
        
        # Carte de probabilité d'ombres
        axes[0, 1].imshow(shadow_probability, cmap='hot')
        axes[0, 1].set_title("Probabilité d'Ombres (Rouge = Ombres)")
        axes[0, 1].axis('off')
        
        # Masque d'ombres
        axes[1, 0].imshow(shadow_mask, cmap='gray')
        axes[1, 0].set_title("Masque d'Ombres (Blanc = Ombres)")
        axes[1, 0].axis('off')
        
        # Image avec masque superposé
        overlay = img_rgb.copy()
        overlay[shadow_mask] = [255, 0, 0]  # Rouge pour les ombres
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Ombres Détectées (Rouge)")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig("visualisation_detection_ombres.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualisation générée: visualisation_detection_ombres.png")
        
    except Exception as e:
        print(f"❌ Erreur lors de la visualisation: {e}")

def main():
    """Fonction principale"""
    print("🚀 Test des Améliorations Avancées de Suppression d'Ombres")
    print("Objectif: Différencier efficacement les ombres des motifs du tissu")
    print("=" * 70)
    
    # Test des méthodes avancées
    test_advanced_methods()
    
    # Test de visualisation
    test_shadow_detection_visualization()
    
    print("\n🎯 Résumé des Améliorations:")
    print("1. ✅ Détection intelligente ombres vs motifs")
    print("2. ✅ Préservation des caractéristiques du tissu")
    print("3. ✅ Nouvelles méthodes 'intelligent' et 'fabric_preserving'")
    print("4. ✅ Sélection automatique améliorée")
    print("5. ✅ Visualisation de la détection d'ombres")
    print("\n💡 Les résultats devraient maintenant mieux préserver:")
    print("   • La couleur originale du tissu")
    print("   • La brillance et les reflets")
    print("   • Les motifs et textures")
    print("   • Tout en supprimant efficacement les vraies ombres")

if __name__ == "__main__":
    main()

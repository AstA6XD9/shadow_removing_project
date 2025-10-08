#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration de la préservation des caractéristiques du tissu
Montre comment les nouvelles méthodes préservent couleur, brillance et motifs
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from shadow_removing import AdvancedShadowRemover

def analyze_fabric_characteristics(image):
    """Analyse les caractéristiques du tissu dans l'image"""
    print("🔍 Analyse des Caractéristiques du Tissu")
    print("-" * 40)
    
    # Convertir en LAB pour analyse
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Analyser la brillance
    brightness_mean = np.mean(l)
    brightness_std = np.std(l)
    print(f"💡 Brillance moyenne: {brightness_mean:.1f}")
    print(f"💡 Variation de brillance: {brightness_std:.1f}")
    
    # Analyser la saturation des couleurs
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    print(f"🎨 Saturation moyenne: {np.mean(saturation):.1f}")
    
    # Analyser la texture
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_complexity = np.std(cv2.Laplacian(gray, cv2.CV_64F))
    print(f"🧵 Complexité de texture: {texture_complexity:.1f}")
    
    # Déterminer le type de tissu
    if texture_complexity < 20:
        fabric_type = "Tissu lisse (velours, soie)"
    elif texture_complexity < 40:
        fabric_type = "Tissu mat (coton, lin)"
    else:
        fabric_type = "Tissu avec motifs"
    
    print(f"🏷️ Type de tissu détecté: {fabric_type}")
    
    return {
        'brightness_mean': brightness_mean,
        'brightness_std': brightness_std,
        'saturation_mean': np.mean(saturation),
        'texture_complexity': texture_complexity,
        'fabric_type': fabric_type
    }

def compare_fabric_preservation(original, processed, method_name):
    """Compare la préservation des caractéristiques du tissu"""
    print(f"\n📊 Comparaison - {method_name}")
    print("-" * 30)
    
    # Analyser l'original
    orig_chars = analyze_fabric_characteristics(original)
    
    # Analyser le résultat
    proc_chars = analyze_fabric_characteristics(processed)
    
    # Calculer les différences
    brightness_diff = abs(orig_chars['brightness_mean'] - proc_chars['brightness_mean'])
    saturation_diff = abs(orig_chars['saturation_mean'] - proc_chars['saturation_mean'])
    texture_diff = abs(orig_chars['texture_complexity'] - proc_chars['texture_complexity'])
    
    print(f"📈 Différence de brillance: {brightness_diff:.1f}")
    print(f"📈 Différence de saturation: {saturation_diff:.1f}")
    print(f"📈 Différence de texture: {texture_diff:.1f}")
    
    # Score de préservation (plus bas = mieux préservé)
    preservation_score = (brightness_diff + saturation_diff + texture_diff) / 3
    print(f"🏆 Score de préservation: {preservation_score:.1f} (plus bas = mieux)")
    
    return preservation_score

def demo_fabric_preservation():
    """Démonstration de la préservation des caractéristiques du tissu"""
    print("🎭 Démonstration de la Préservation des Caractéristiques du Tissu")
    print("=" * 70)
    print("Objectif: Montrer comment les nouvelles méthodes préservent:")
    print("• La couleur originale du tissu")
    print("• La brillance et les reflets")
    print("• Les motifs et textures")
    print("=" * 70)
    
    # Chemin de l'image
    image_path = "C:/Users/eloua/OneDrive/Images/IMG_6043.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée: {image_path}")
        return
    
    # Charger l'image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Impossible de charger l'image: {image_path}")
        return
    
    print(f"✅ Image chargée: {image.shape}")
    
    # Initialiser le shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Analyser l'image originale
    print("\n🔍 Analyse de l'image originale:")
    original_chars = analyze_fabric_characteristics(image)
    
    # Test des différentes méthodes
    methods = {
        'Méthode Classique (Retinex)': 'retinex',
        'Méthode Texture-Aware': 'texture_aware',
        'Méthode Intelligente': 'intelligent',
        'Méthode Fabric-Preserving': 'fabric_preserving'
    }
    
    results = {}
    preservation_scores = {}
    
    for method_name, method_key in methods.items():
        print(f"\n{'='*50}")
        print(f"🧪 Test: {method_name}")
        print(f"{'='*50}")
        
        try:
            # Appliquer la méthode
            result = shadow_remover.remove_shadows(image, method=method_key)
            
            if result is not None:
                # Sauvegarder le résultat
                filename = f"demo_{method_key}.jpg"
                cv2.imwrite(filename, result)
                print(f"✅ {method_name} réussie")
                print(f"💾 Sauvegardé: {filename}")
                
                # Comparer la préservation
                score = compare_fabric_preservation(image, result, method_name)
                preservation_scores[method_name] = score
                results[method_name] = result
                
                # Évaluer la qualité générale
                metrics = shadow_remover.assess_quality(image, result)
                print(f"📊 Qualité globale: {metrics['overall_quality']:.3f}")
                
            else:
                print(f"❌ {method_name} échouée")
                
        except Exception as e:
            print(f"❌ Erreur avec {method_name}: {e}")
    
    # Afficher le classement
    print(f"\n🏆 CLASSEMENT PAR PRÉSERVATION DES CARACTÉRISTIQUES")
    print("=" * 60)
    print("(Score plus bas = meilleure préservation)")
    print("-" * 60)
    
    sorted_scores = sorted(preservation_scores.items(), key=lambda x: x[1])
    for i, (method, score) in enumerate(sorted_scores, 1):
        print(f"{i}. {method}: {score:.1f}")
    
    # Créer une visualisation comparative
    create_comparison_visualization(image, results)
    
    print(f"\n✨ DÉMONSTRATION TERMINÉE")
    print("=" * 50)
    print("Les nouvelles méthodes 'intelligent' et 'fabric_preserving'")
    print("devraient mieux préserver les caractéristiques du tissu!")

def create_comparison_visualization(original, results):
    """Crée une visualisation comparative des résultats"""
    print(f"\n🎨 Création de la visualisation comparative...")
    
    try:
        n_methods = len(results)
        if n_methods == 0:
            print("❌ Aucun résultat à visualiser")
            return
        
        # Créer la figure
        fig, axes = plt.subplots(2, (n_methods + 1) // 2 + 1, figsize=(20, 10))
        axes = axes.flatten()
        
        # Image originale
        img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Image Originale", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Résultats
        for i, (method_name, result) in enumerate(results.items(), 1):
            if i < len(axes):
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                axes[i].imshow(result_rgb)
                axes[i].set_title(method_name, fontsize=10)
                axes[i].axis('off')
        
        # Masquer les axes inutilisés
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig("comparaison_preservation_tissu.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualisation sauvegardée: comparaison_preservation_tissu.png")
        
    except Exception as e:
        print(f"❌ Erreur lors de la création de la visualisation: {e}")

def test_shadow_vs_pattern_detection():
    """Test spécifique de la différenciation ombres vs motifs"""
    print(f"\n🔍 Test de Différenciation Ombres vs Motifs")
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
        
        # Analyser les zones détectées
        shadow_pixels = np.sum(shadow_mask)
        total_pixels = shadow_mask.size
        shadow_percentage = shadow_pixels / total_pixels * 100
        
        print(f"📊 Statistiques de détection:")
        print(f"  Pixels détectés comme ombres: {shadow_pixels:,}")
        print(f"  Total de pixels: {total_pixels:,}")
        print(f"  Pourcentage d'ombres: {shadow_percentage:.1f}%")
        
        # Analyser la distribution des probabilités
        prob_mean = np.mean(shadow_probability)
        prob_std = np.std(shadow_probability)
        print(f"  Probabilité moyenne: {prob_mean:.3f}")
        print(f"  Écart-type: {prob_std:.3f}")
        
        # Créer une visualisation détaillée
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Image originale
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title("Image Originale")
        axes[0, 0].axis('off')
        
        # Carte de probabilité
        im1 = axes[0, 1].imshow(shadow_probability, cmap='hot')
        axes[0, 1].set_title("Probabilité d'Ombres")
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Masque binaire
        axes[0, 2].imshow(shadow_mask, cmap='gray')
        axes[0, 2].set_title("Masque d'Ombres")
        axes[0, 2].axis('off')
        
        # Histogramme des probabilités
        axes[1, 0].hist(shadow_probability.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title("Distribution des Probabilités")
        axes[1, 0].set_xlabel("Probabilité")
        axes[1, 0].set_ylabel("Fréquence")
        
        # Image avec ombres en rouge
        overlay = img_rgb.copy()
        overlay[shadow_mask] = [255, 0, 0]
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Ombres Détectées (Rouge)")
        axes[1, 1].axis('off')
        
        # Image avec motifs préservés
        non_shadow_mask = ~shadow_mask
        preserved = img_rgb.copy()
        preserved[non_shadow_mask] = preserved[non_shadow_mask] * 0.7  # Assombrir légèrement
        axes[1, 2].imshow(preserved)
        axes[1, 2].set_title("Motifs Préservés")
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig("analyse_ombres_vs_motifs.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Analyse sauvegardée: analyse_ombres_vs_motifs.png")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'analyse: {e}")

def main():
    """Fonction principale"""
    print("🎭 Démonstration de la Préservation des Caractéristiques du Tissu")
    print("=" * 70)
    
    # Démonstration principale
    demo_fabric_preservation()
    
    # Test de différenciation ombres vs motifs
    test_shadow_vs_pattern_detection()
    
    print(f"\n🎯 RÉSUMÉ DE LA DÉMONSTRATION")
    print("=" * 50)
    print("✅ Analyse des caractéristiques du tissu")
    print("✅ Comparaison des méthodes de préservation")
    print("✅ Test de différenciation ombres vs motifs")
    print("✅ Visualisations détaillées")
    print("\n💡 Les nouvelles méthodes devraient montrer:")
    print("   • Meilleure préservation de la couleur")
    print("   • Conservation de la brillance")
    print("   • Maintien des motifs du tissu")
    print("   • Suppression efficace des vraies ombres")

if __name__ == "__main__":
    main()

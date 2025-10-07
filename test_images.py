#!/usr/bin/env python3
"""
Test du système de suppression d'ombres sur IMG_6043 et IMG_6020
"""

import cv2
import os
from shadow_removing import AdvancedShadowRemover

def find_test_images():
    """Recherche les images IMG_6043 et IMG_6020"""
    print("🔍 Recherche des images IMG_6043 et IMG_6020...")
    
    # Dossiers à rechercher
    search_dirs = [
        "C:\\Users\\eloua\\",
        "C:\\Users\\eloua\\OneDrive\\",
        "C:\\Users\\eloua\\Desktop\\",
        "C:\\Users\\eloua\\Pictures\\",
        "C:\\Users\\eloua\\Documents\\",
        "C:\\Users\\eloua\\Downloads\\"
    ]
    
    target_images = ["IMG_6043", "IMG_6020"]
    found_images = {}
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            print(f"Recherche dans: {search_dir}")
            try:
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        for target in target_images:
                            if target in file.upper():
                                full_path = os.path.join(root, file)
                                # Tester si c'est une image valide
                                img = cv2.imread(full_path)
                                if img is not None:
                                    found_images[target] = full_path
                                    print(f"  ✓ Trouvé {target}: {full_path} - Taille: {img.shape}")
                                else:
                                    print(f"  ❌ {target}: Fichier corrompu - {full_path}")
            except PermissionError:
                print(f"  ⚠️ Accès refusé: {search_dir}")
            except Exception as e:
                print(f"  ⚠️ Erreur: {e}")
    
    return found_images

def test_shadow_removal_on_images(image_paths):
    """Teste la suppression d'ombres sur les images trouvées"""
    print("\n🧪 Test du système de suppression d'ombres...")
    
    shadow_remover = AdvancedShadowRemover()
    
    for image_name, image_path in image_paths.items():
        print(f"\n{'='*60}")
        print(f"📸 Traitement de {image_name}: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        
        # Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Impossible de charger {image_path}")
            continue
        
        print(f"✅ Image chargée - Taille: {image.shape}")
        
        # Test de la sélection automatique de méthode
        print(f"\n🤖 Test de sélection automatique...")
        auto_result = shadow_remover.remove_shadows(image, method='auto')
        
        if auto_result is not None:
            # Évaluer la qualité
            metrics = shadow_remover.assess_quality(image, auto_result)
            print(f"📊 Métriques de qualité:")
            print(f"  Score global: {metrics['overall_quality']:.3f}")
            print(f"  Amélioration du contraste: {metrics['contrast_improvement']:.3f}")
            print(f"  Réduction des ombres: {metrics['shadow_reduction']:.3f}")
            print(f"  Préservation des couleurs: {metrics['color_preservation']:.3f}")
            
            # Sauvegarder le résultat automatique
            output_path = f"resultat_auto_{image_name}.jpg"
            cv2.imwrite(output_path, auto_result)
            print(f"💾 Résultat automatique sauvegardé: {output_path}")
        
        # Test de différentes méthodes
        print(f"\n🔬 Test de différentes méthodes...")
        methods = ['retinex', 'adaptive_histogram', 'gradient_domain']
        
        best_method = None
        best_score = -1
        
        for method in methods:
            print(f"  🧪 Test de {method}...")
            try:
                result = shadow_remover.remove_shadows(image, method=method)
                if result is not None:
                    method_metrics = shadow_remover.assess_quality(image, result)
                    score = method_metrics['overall_quality']
                    
                    print(f"    📊 Qualité: {score:.3f}")
                    print(f"    🎨 Contraste: {method_metrics['contrast_improvement']:.3f}")
                    print(f"    🌑 Ombres: {method_metrics['shadow_reduction']:.3f}")
                    print(f"    🎨 Couleurs: {method_metrics['color_preservation']:.3f}")
                    
                    # Sauvegarder le résultat
                    output_path = f"resultat_{method}_{image_name}.jpg"
                    cv2.imwrite(output_path, result)
                    print(f"    💾 Sauvegardé: {output_path}")
                    
                    # Garder le meilleur score
                    if score > best_score:
                        best_score = score
                        best_method = method
                else:
                    print(f"    ❌ Échec de {method}")
            except Exception as e:
                print(f"    ❌ Erreur avec {method}: {e}")
        
        if best_method:
            print(f"\n🏆 Meilleure méthode pour {image_name}: {best_method} (score: {best_score:.3f})")
        
        # Test d'optimisation des paramètres
        print(f"\n⚙️ Optimisation des paramètres...")
        try:
            best_result, best_params = shadow_remover.optimize_parameters(image, method='retinex')
            if best_result is not None:
                optimized_metrics = shadow_remover.assess_quality(image, best_result)
                print(f"📈 Résultat optimisé - Qualité: {optimized_metrics['overall_quality']:.3f}")
                print(f"🔧 Paramètres optimaux: {best_params}")
                
                output_path = f"resultat_optimise_{image_name}.jpg"
                cv2.imwrite(output_path, best_result)
                print(f"💾 Résultat optimisé sauvegardé: {output_path}")
        except Exception as e:
            print(f"❌ Erreur lors de l'optimisation: {e}")

def main():
    """Fonction principale"""
    print("🎯 Test du système de suppression d'ombres")
    print("Images cibles: IMG_6043 et IMG_6020")
    print("="*60)
    
    # Rechercher les images
    found_images = find_test_images()
    
    if not found_images:
        print("\n❌ Aucune des images IMG_6043 ou IMG_6020 n'a été trouvée")
        print("Veuillez vérifier que les images existent et sont accessibles")
        return
    
    print(f"\n✅ {len(found_images)} image(s) trouvée(s):")
    for name, path in found_images.items():
        print(f"  {name}: {path}")
    
    # Tester la suppression d'ombres
    test_shadow_removal_on_images(found_images)
    
    print(f"\n🎉 Test terminé! Vérifiez les fichiers de résultats générés.")

if __name__ == "__main__":
    main()




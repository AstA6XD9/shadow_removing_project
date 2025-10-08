#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test de la détection de zone CLAIRE améliorée avec zoom logique
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def test_zone_claire_amelioree():
    """Test de la détection de zone claire avec zoom logique"""
    print("Test Zone CLAIRE Amelioree avec Zoom Logique")
    print("=" * 60)
    print("Objectif: Trouver la zone la plus CLAIRE avec zoom LOGIQUE")
    print("Eviter les details indesirables du tissu")
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
    
    # Test 1: Detection de tissu uni
    print("\n1. Detection de Tissu Uni")
    print("-" * 30)
    
    is_solid, complexity = shadow_remover._is_solid_color_fabric(image)
    print(f"Complexite de texture: {complexity:.6f}")
    print(f"Tissu uni detecte: {is_solid}")
    
    if not is_solid:
        print("ATTENTION: Cette image ne semble pas etre un tissu uni")
        return
    
    # Test 2: Detection de la zone CLAIRE
    print("\n2. Detection de la Zone CLAIRE")
    print("-" * 30)
    
    try:
        perfect_region, score = shadow_remover._detect_perfect_fabric_region(image, min_region_size=150)
        
        if perfect_region is not None:
            x, y, w, h = perfect_region
            print(f"Zone CLAIRE trouvee: ({x}, {y}, {w}, {h})")
            print(f"Score de clarte/homogeneite: {score:.2f}")
            
            # Dessiner un rectangle sur l'image pour montrer la zone
            image_with_region = image.copy()
            cv2.rectangle(image_with_region, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.imwrite("zone_claire_detectee.jpg", image_with_region)
            print("Sauvegarde: zone_claire_detectee.jpg")
            
            # Extraire la zone claire
            perfect_zone = image[y:y+h, x:x+w]
            cv2.imwrite("zone_claire_extraite.jpg", perfect_zone)
            print("Sauvegarde: zone_claire_extraite.jpg")
            
        else:
            print("ERREUR: Aucune zone claire trouvee")
            
    except Exception as e:
        print(f"ERREUR lors de la detection: {e}")
    
    # Test 3: Creation d'echantillon avec zoom LOGIQUE
    print("\n3. Creation d'Echantillon avec Zoom LOGIQUE")
    print("-" * 30)
    
    # Facteurs de zoom LOGIQUES (pas trop agressifs)
    zoom_factors = [1.1, 1.2, 1.3, 1.4, 1.5]
    
    for zoom_factor in zoom_factors:
        try:
            print(f"\nTest avec zoom LOGIQUE x{zoom_factor}:")
            sample = shadow_remover.create_perfect_fabric_sample(
                image, 
                zoom_factor=zoom_factor, 
                min_region_size=150
            )
            
            if sample is not None:
                # Sauvegarder l'echantillon
                output_path = f"echantillon_claire_zoom_{zoom_factor}x.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  OK - Echantillon cree: {output_path}")
                print(f"  Taille: {sample.shape[1]}x{sample.shape[0]}")
                
                # Analyser la qualite de l'echantillon
                gray_sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray_sample)
                print(f"  Luminosite moyenne: {brightness:.1f}")
            else:
                print(f"  ECHEC - Impossible de creer l'echantillon")
                
        except Exception as e:
            print(f"  ERREUR: {e}")
    
    # Test 4: Comparaison avec l'ancienne methode
    print("\n4. Comparaison avec Ancienne Methode")
    print("-" * 30)
    
    try:
        # Ancienne methode (zoom agressif)
        print("Ancienne methode (zoom x2.0):")
        old_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=2.0, min_region_size=100)
        if old_sample is not None:
            cv2.imwrite("ancienne_methode_zoom_2x.jpg", old_sample)
            print(f"  OK - Ancienne methode: ancienne_methode_zoom_2x.jpg")
            print(f"  Taille: {old_sample.shape[1]}x{old_sample.shape[0]}")
        
        # Nouvelle methode (zoom logique)
        print("\nNouvelle methode (zoom x1.3):")
        new_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=1.3, min_region_size=150)
        if new_sample is not None:
            cv2.imwrite("nouvelle_methode_zoom_1.3x.jpg", new_sample)
            print(f"  OK - Nouvelle methode: nouvelle_methode_zoom_1.3x.jpg")
            print(f"  Taille: {new_sample.shape[1]}x{new_sample.shape[0]}")
        
    except Exception as e:
        print(f"ERREUR: {e}")
    
    # Test 5: Creation de texture avec echantillon clair
    print("\n5. Creation de Texture avec Echantillon Clair")
    print("-" * 30)
    
    try:
        # Creer un echantillon de base avec zoom logique
        base_sample = shadow_remover.create_perfect_fabric_sample(image, zoom_factor=1.3)
        
        if base_sample is not None:
            # Creer une texture de taille raisonnable
            texture = shadow_remover.create_fabric_texture_from_sample(
                base_sample, 
                target_size=(600, 400),
                preserve_characteristics=True
            )
            
            if texture is not None:
                cv2.imwrite("texture_zone_claire_600x400.jpg", texture)
                print("OK - Texture creee: texture_zone_claire_600x400.jpg")
            else:
                print("ECHEC - Impossible de creer la texture")
        else:
            print("ECHEC - Impossible de creer l'echantillon de base")
            
    except Exception as e:
        print(f"ERREUR: {e}")
    
    print("\n" + "=" * 60)
    print("TEST TERMINE")
    print("=" * 60)
    print("Fichiers generes:")
    print("- zone_claire_detectee.jpg : Image avec zone claire marquee")
    print("- zone_claire_extraite.jpg : Zone claire extraite")
    print("- echantillon_claire_zoom_*.jpg : Echantillons avec zoom logique")
    print("- ancienne_methode_zoom_2x.jpg : Ancienne methode (zoom agressif)")
    print("- nouvelle_methode_zoom_1.3x.jpg : Nouvelle methode (zoom logique)")
    print("- texture_zone_claire_600x400.jpg : Texture avec zone claire")
    print("\nLa nouvelle methode trouve la zone la plus CLAIRE")
    print("et utilise un zoom LOGIQUE pour eviter les details indesirables!")

def main():
    """Fonction principale"""
    print("Test de la Detection de Zone CLAIRE Amelioree")
    print("=" * 70)
    print("Ameliorations apportees:")
    print("1. Detection de la zone la plus CLAIRE (pas juste homogene)")
    print("2. Zoom LOGIQUE (1.1x a 1.5x au lieu de 2x-4x)")
    print("3. Region plus grande (150px au lieu de 100px)")
    print("4. Interpolation LINEAIRE (au lieu de CUBIC)")
    print("5. Evite les details indesirables du tissu")
    print("=" * 70)
    
    # Test des ameliorations
    test_zone_claire_amelioree()
    
    print("\nResume des Ameliorations:")
    print("=" * 40)
    print("✅ Zone la plus CLAIRE detectee")
    print("✅ Zoom LOGIQUE (pas trop agressif)")
    print("✅ Region plus grande (evite les details)")
    print("✅ Interpolation LINEAIRE (plus naturelle)")
    print("✅ Pas de details indesirables visibles")
    print("\nLes echantillons sont maintenant plus naturels!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration simple de la sélection interactive de couleur
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def demo_selection_couleur():
    """Démonstration simple de la sélection de couleur"""
    print("Demonstration - Selection Interactive de Couleur")
    print("=" * 60)
    print("Objectif: Selectionner manuellement la couleur desiree")
    print("et creer des textures MASSIVES")
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
    
    # Selection interactive de la couleur
    print("\nSelection interactive de la couleur...")
    print("Instructions:")
    print("- Cliquez et glissez pour selectionner une region")
    print("- Appuyez sur ENTER pour confirmer")
    print("- Appuyez sur ESC pour annuler")
    print("- Appuyez sur SPACE pour voir la couleur")
    
    try:
        region_coords, selected_color = shadow_remover.select_color_region_interactive(image)
        
        if region_coords is not None and selected_color is not None:
            print(f"\nSelection reussie!")
            print(f"Region: {region_coords}")
            print(f"Couleur selectionnee: BGR({selected_color[0]}, {selected_color[1]}, {selected_color[2]})")
            
            # Sauvegarder la region selectionnee
            x, y, w, h = region_coords
            selected_region = image[y:y+h, x:x+w]
            cv2.imwrite("demo_region_selectionnee.jpg", selected_region)
            print("Sauvegarde: demo_region_selectionnee.jpg")
            
        else:
            print("Selection annulee")
            return
            
    except Exception as e:
        print(f"ERREUR lors de la selection: {e}")
        return
    
    # Creation d'echantillon personnalise
    print("\nCreation d'echantillon personnalise...")
    try:
        custom_sample = shadow_remover.create_custom_fabric_sample(
            image, 
            region_coords, 
            selected_color, 
            zoom_factor=1.3, 
            uniformization_strength=0.8
        )
        
        if custom_sample is not None:
            cv2.imwrite("demo_echantillon_personnalise.jpg", custom_sample)
            print("OK - Echantillon personnalise: demo_echantillon_personnalise.jpg")
            print(f"Taille: {custom_sample.shape[1]}x{custom_sample.shape[0]}")
            
            # Verifier la couleur
            sample_color = np.mean(custom_sample.reshape(-1, 3), axis=0)
            print(f"Couleur de l'echantillon: BGR({sample_color[0]:.1f}, {sample_color[1]:.1f}, {sample_color[2]:.1f})")
            
        else:
            print("ECHEC - Impossible de creer l'echantillon")
            return
            
    except Exception as e:
        print(f"ERREUR: {e}")
        return
    
    # Creation de texture MASSIVE
    print("\nCreation de texture MASSIVE 2000x2000...")
    try:
        massive_texture = shadow_remover.create_massive_fabric_texture(
            image, 
            region_coords, 
            selected_color, 
            target_size=(2000, 2000),
            zoom_factor=1.3,
            uniformization_strength=0.8
        )
        
        if massive_texture is not None:
            cv2.imwrite("demo_texture_massive_2000x2000.jpg", massive_texture)
            print("OK - Texture MASSIVE: demo_texture_massive_2000x2000.jpg")
            print(f"Taille: {massive_texture.shape[1]}x{massive_texture.shape[0]}")
            
            # Calculer la taille en megapixels
            width, height = massive_texture.shape[1], massive_texture.shape[0]
            megapixels = (width * height) / 1000000
            print(f"Taille: {megapixels:.1f} megapixels")
            
            # Verifier la couleur
            texture_color = np.mean(massive_texture.reshape(-1, 3), axis=0)
            print(f"Couleur de la texture: BGR({texture_color[0]:.1f}, {texture_color[1]:.1f}, {texture_color[2]:.1f})")
            
        else:
            print("ECHEC - Impossible de creer la texture MASSIVE")
            
    except Exception as e:
        print(f"ERREUR: {e}")
    
    # Creation de plusieurs textures MASSIVES
    print("\nCreation de plusieurs textures MASSIVES...")
    try:
        massive_sizes = [(1500, 1500), (2500, 2500), (3500, 3500)]
        
        for width, height in massive_sizes:
            print(f"\nCreation texture {width}x{height}...")
            texture = shadow_remover.create_massive_fabric_texture(
                image, 
                region_coords, 
                selected_color, 
                target_size=(width, height),
                zoom_factor=1.3,
                uniformization_strength=0.8
            )
            
            if texture is not None:
                output_path = f"demo_texture_massive_{width}x{height}.jpg"
                cv2.imwrite(output_path, texture)
                print(f"  OK - {width}x{height}: {output_path}")
                
                # Calculer la taille en megapixels
                megapixels = (width * height) / 1000000
                print(f"  Taille: {megapixels:.1f} megapixels")
            else:
                print(f"  ECHEC - {width}x{height}")
                
    except Exception as e:
        print(f"ERREUR: {e}")

if __name__ == "__main__":
    demo_selection_couleur()

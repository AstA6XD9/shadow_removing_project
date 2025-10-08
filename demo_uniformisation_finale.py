#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démonstration finale de l'uniformisation de la zone zoomée
"""

import cv2
import numpy as np
import os
from shadow_removing import AdvancedShadowRemover

def demo_uniformisation_finale():
    """Démonstration finale de l'uniformisation"""
    print("Demonstration - Uniformisation de la Zone Zoomee")
    print("=" * 60)
    print("Objectif: Uniformiser la zone zoomee et ajuster la couleur")
    print("vers la couleur la plus claire de l'image originale")
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
    
    # Vérifier si c'est un tissu uni
    is_solid, complexity = shadow_remover._is_solid_color_fabric(image)
    print(f"Tissu uni detecte: {is_solid} (complexite: {complexity:.6f})")
    
    if not is_solid:
        print("ATTENTION: Cette image ne semble pas etre un tissu uni")
        return
    
    # Trouver la couleur la plus claire
    print("\nDetection de la couleur la plus claire...")
    lightest_color = shadow_remover._find_lightest_color_in_image(image)
    print(f"Couleur la plus claire: BGR({lightest_color[0]}, {lightest_color[1]}, {lightest_color[2]})")
    
    # Créer un échantillon uniformisé
    print("\nCreation d'un echantillon uniformise...")
    uniformized_sample = shadow_remover.create_uniformized_fabric_sample(
        image, 
        zoom_factor=1.3, 
        uniformization_strength=0.8
    )
    
    if uniformized_sample is not None:
        cv2.imwrite("demo_echantillon_uniformise.jpg", uniformized_sample)
        print(f"OK - Echantillon uniformise: demo_echantillon_uniformise.jpg")
        print(f"Taille: {uniformized_sample.shape[1]}x{uniformized_sample.shape[0]}")
        
        # Analyser la couleur de l'échantillon uniformisé
        sample_mean = np.mean(uniformized_sample.reshape(-1, 3), axis=0)
        print(f"Couleur moyenne de l'echantillon: BGR({sample_mean[0]:.1f}, {sample_mean[1]:.1f}, {sample_mean[2]:.1f})")
        
    else:
        print("ERREUR: Impossible de creer l'echantillon uniformise")
        return
    
    # Créer une texture uniformisée 1200x900 (comme votre exemple)
    print("\nCreation d'une texture uniformisee 1200x900...")
    texture_1200x900 = shadow_remover.create_uniformized_fabric_texture(
        image, 
        target_size=(1200, 900),
        zoom_factor=1.3,
        uniformization_strength=0.8
    )
    
    if texture_1200x900 is not None:
        cv2.imwrite("demo_texture_uniformisee_1200x900.jpg", texture_1200x900)
        print("OK - Texture uniformisee: demo_texture_uniformisee_1200x900.jpg")
        print(f"Taille: {texture_1200x900.shape[1]}x{texture_1200x900.shape[0]}")
        
        # Analyser la couleur de la texture
        texture_mean = np.mean(texture_1200x900.reshape(-1, 3), axis=0)
        print(f"Couleur moyenne de la texture: BGR({texture_mean[0]:.1f}, {texture_mean[1]:.1f}, {texture_mean[2]:.1f})")
        
    else:
        print("ERREUR: Impossible de creer la texture uniformisee")
    
    # Test avec différentes forces d'uniformisation
    print("\nTest avec differentes forces d'uniformisation:")
    forces = [0.5, 0.7, 0.8, 0.9]
    
    for force in forces:
        try:
            sample = shadow_remover.create_uniformized_fabric_sample(
                image, 
                zoom_factor=1.3, 
                uniformization_strength=force
            )
            
            if sample is not None:
                output_path = f"demo_uniformise_force_{force}.jpg"
                cv2.imwrite(output_path, sample)
                print(f"  Force {force}: {output_path}")
        except Exception as e:
            print(f"  Force {force}: ERREUR - {e}")
    
    # Créer des textures de différentes tailles
    print("\nCreation de textures de differentes tailles:")
    sizes = [(600, 400), (800, 600), (1000, 750), (1200, 900)]
    
    for width, height in sizes:
        try:
            texture = shadow_remover.create_uniformized_fabric_texture(
                image, 
                target_size=(width, height),
                zoom_factor=1.3,
                uniformization_strength=0.8
            )
            
            if texture is not None:
                output_path = f"demo_texture_uniformisee_{width}x{height}.jpg"
                cv2.imwrite(output_path, texture)
                print(f"  {width}x{height}: {output_path}")
        except Exception as e:
            print(f"  {width}x{height}: ERREUR - {e}")

if __name__ == "__main__":
    demo_uniformisation_finale()

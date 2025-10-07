#!/usr/bin/env python3
"""
Example usage of the Advanced Shadow Removal system
"""

import cv2
import numpy as np
from shadow_removing import AdvancedShadowRemover

def simple_example():
    """Simple example of shadow removal"""
    print("=== Simple Shadow Removal Example ===")
    
    # Initialize shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Load image (modify path as needed)
    image_path = "photo.jpeg"  # Change this to your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        print("Please ensure the image path is correct")
        return
    
    print(f"Image loaded: {image.shape}")
    
    # Remove shadows automatically
    result = shadow_remover.remove_shadows(image, method='auto')
    
    if result is not None:
        # Save result
        cv2.imwrite("simple_result.jpg", result)
        print("Result saved as 'simple_result.jpg'")
        
        # Assess quality
        metrics = shadow_remover.assess_quality(image, result)
        print(f"Quality score: {metrics['overall_quality']:.3f}")

def compare_methods_example():
    """Example comparing different methods"""
    print("\n=== Method Comparison Example ===")
    
    shadow_remover = AdvancedShadowRemover()
    
    # Load image
    image = cv2.imread("photo.jpeg")
    if image is None:
        print("Error: Could not load image")
        return
    
    # Test different methods
    methods = ['retinex', 'texture_aware', 'adaptive_histogram']
    
    for method in methods:
        print(f"\nTesting {method}...")
        result = shadow_remover.remove_shadows(image, method=method)
        
        if result is not None:
            # Save result
            cv2.imwrite(f"result_{method}.jpg", result)
            
            # Assess quality
            metrics = shadow_remover.assess_quality(image, result)
            print(f"  Quality: {metrics['overall_quality']:.3f}")
            print(f"  Contrast improvement: {metrics['contrast_improvement']:.3f}")
            print(f"  Shadow reduction: {metrics['shadow_reduction']:.3f}")

def batch_processing_example():
    """Example of batch processing multiple images"""
    print("\n=== Batch Processing Example ===")
    
    shadow_remover = AdvancedShadowRemover()
    
    # List of image paths (modify as needed)
    image_paths = [
        "photo1.jpg",
        "photo2.jpg", 
        "photo3.jpg"
    ]
    
    # Process all images
    results = shadow_remover.batch_process(image_paths, method='auto')
    
    # Print summary
    print("Batch processing results:")
    for path, data in results.items():
        print(f"  {path}: Quality = {data['metrics']['overall_quality']:.3f}")

def optimization_example():
    """Example of parameter optimization"""
    print("\n=== Parameter Optimization Example ===")
    
    shadow_remover = AdvancedShadowRemover()
    
    # Load image
    image = cv2.imread("photo.jpeg")
    if image is None:
        print("Error: Could not load image")
        return
    
    # Optimize retinex parameters
    best_result, best_params = shadow_remover.optimize_parameters(image, method='retinex')
    
    if best_result is not None:
        cv2.imwrite("optimized_result.jpg", best_result)
        print(f"Optimized result saved with parameters: {best_params}")

def fabric_specific_example():
    """Example for different fabric types"""
    print("\n=== Fabric-Specific Processing Example ===")
    
    shadow_remover = AdvancedShadowRemover()
    
    # Load image
    image = cv2.imread("photo.jpeg")
    if image is None:
        print("Error: Could not load image")
        return
    
    # Process for different fabric types
    fabric_types = {
        'plain_fabric': 'adaptive_histogram',
        'patterned_fabric': 'texture_aware',
        'mixed_textures': 'multi_scale',
        'strong_shadows': 'retinex'
    }
    
    for fabric_type, method in fabric_types.items():
        print(f"\nProcessing for {fabric_type} using {method}...")
        result = shadow_remover.remove_shadows(image, method=method)
        
        if result is not None:
            cv2.imwrite(f"fabric_{fabric_type}.jpg", result)
            metrics = shadow_remover.assess_quality(image, result)
            print(f"  Quality: {metrics['overall_quality']:.3f}")

if __name__ == "__main__":
    print("Advanced Shadow Removal - Example Usage")
    print("=" * 50)
    
    # Run examples
    simple_example()
    compare_methods_example()
    # batch_processing_example()  # Uncomment if you have multiple images
    optimization_example()
    fabric_specific_example()
    
    print("\n" + "=" * 50)
    print("Examples completed! Check the generated image files.")


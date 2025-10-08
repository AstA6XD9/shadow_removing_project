#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Shadow Removal for Moroccan Living Room Photos
Robust solution for various fabric textures and patterns

@author: Mohammed Amine EL Ouardini
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, filters, restoration, segmentation
from skimage.color import rgb2lab, lab2rgb
from scipy import ndimage
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import warnings
warnings.filterwarnings('ignore')

class AdvancedShadowRemover:
    """
    Advanced shadow removal class for Moroccan living room fabrics
    Handles various textures: plain, patterned, single/multi-color fabrics
    """
    
    def __init__(self):
        self.methods = {
            'retinex': self._retinex_shadow_removal,
            'gradient_domain': self._gradient_domain_shadow_removal,
            'adaptive_histogram': self._adaptive_histogram_matching,
            'texture_aware': self._texture_aware_shadow_removal,
            'multi_scale': self._multi_scale_shadow_removal,
            'intelligent': self._intelligent_shadow_removal,
            'fabric_preserving': self._fabric_preserving_shadow_removal
        }
    
    def remove_shadows(self, image, method='auto', **kwargs):
        """
        Main shadow removal function
        
        Args:
            image: Input image (BGR format)
            method: 'auto', 'retinex', 'gradient_domain', 'adaptive_histogram', 
                   'texture_aware', 'multi_scale', 'intelligent', 'fabric_preserving'
            **kwargs: Additional parameters for specific methods
        
        Returns:
            Shadow-removed image
        """
        if method == 'auto':
            # Automatically select best method based on image characteristics
            method = self._select_best_method(image)
        
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"[INFO] Using {method} method for shadow removal...")
        return self.methods[method](image, **kwargs)
    
    def _select_best_method(self, image):
        """Automatically select the best shadow removal method based on image characteristics"""
        # Analyze image characteristics
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture complexity
        texture_complexity = self._calculate_texture_complexity(gray)
        
        # Calculate shadow intensity
        shadow_intensity = self._calculate_shadow_intensity(gray)
        
        # Select method based on characteristics
        if texture_complexity > 0.3:  # High texture complexity
            return 'fabric_preserving'  # Use new intelligent method for complex textures
        elif shadow_intensity > 0.4:  # Strong shadows
            return 'intelligent'  # Use intelligent method for strong shadows
        else:
            return 'fabric_preserving'  # Default to fabric-preserving method
    
    def _calculate_texture_complexity(self, gray):
        """Calculate texture complexity using local binary patterns"""
        # Simplified texture complexity measure
        edges = cv2.Canny(gray, 50, 150)
        return np.mean(edges) / 255.0
    
    def _calculate_shadow_intensity(self, gray):
        """Calculate shadow intensity in the image"""
        # Use Otsu's method to find optimal threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate ratio of dark pixels
        dark_pixels = np.sum(binary == 0)
        total_pixels = binary.size
        
        return dark_pixels / total_pixels
    
    def _retinex_shadow_removal(self, image, sigma=15, alpha=125, beta=46):
        """
        Retinex-based shadow removal
        Good for general shadow removal with natural color preservation
        """
        # Convert to LAB color space for better color preservation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply Gaussian blur to estimate illumination
        illumination = cv2.GaussianBlur(l, (0, 0), sigma)
        illumination = np.maximum(illumination, 1)  # Avoid division by zero
        
        # Retinex formula: R = log(I) - log(I * F)
        reflectance = cv2.subtract(cv2.add(l, alpha), illumination)
        reflectance = np.clip(reflectance, 0, 255).astype(np.uint8)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        reflectance = clahe.apply(reflectance)
        
        # Reconstruct LAB image
        enhanced_lab = cv2.merge([reflectance, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _gradient_domain_shadow_removal(self, image, alpha=0.1):
        """
        Gradient domain shadow removal
        Excellent for preserving fine details and textures
        """
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Create gradient magnitude mask
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Solve Poisson equation for shadow removal
        h, w = gray.shape
        result = self._solve_poisson_equation(gray, grad_x, grad_y, alpha)
        
        # Apply to each color channel
        result_bgr = np.zeros_like(image, dtype=np.float64)
        for i in range(3):
            channel = image[:, :, i].astype(np.float64)
            channel_result = self._solve_poisson_equation(channel, grad_x, grad_y, alpha)
            result_bgr[:, :, i] = channel_result
        
        return np.clip(result_bgr, 0, 255).astype(np.uint8)
    
    def _solve_poisson_equation(self, image, grad_x, grad_y, alpha):
        """Solve Poisson equation for gradient domain processing"""
        h, w = image.shape
        n = h * w
        
        # Create sparse matrix for Poisson equation
        # Simplified implementation - in practice, use more efficient solvers
        result = image.copy()
        
        # Apply iterative refinement
        for _ in range(10):
            # Calculate divergence of gradients
            div_grad = np.zeros_like(image)
            div_grad[1:-1, 1:-1] = (grad_x[1:-1, 1:-1] - grad_x[1:-1, :-2] + 
                                   grad_y[1:-1, 1:-1] - grad_y[:-2, 1:-1])
            
            # Update result
            result = result + alpha * div_grad
            result = np.clip(result, 0, 255)
        
        return result
    
    def _adaptive_histogram_matching(self, image, clip_limit=3.0, tile_size=8):
        """
        Adaptive histogram matching for shadow removal
        Good for maintaining local contrast while removing shadows
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l_enhanced = clahe.apply(l)
        
        # Reconstruct image
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Additional gamma correction for better results
        gamma = 1.2
        result = np.power(result / 255.0, 1/gamma) * 255
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _texture_aware_shadow_removal(self, image, texture_threshold=0.1):
        """
        Texture-aware shadow removal
        Specifically designed for fabrics with complex patterns
        """
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Detect texture regions using local standard deviation
        kernel = np.ones((15, 15), np.float32) / 225
        mean_l = cv2.filter2D(l.astype(np.float32), -1, kernel)
        sqr_mean_l = cv2.filter2D((l.astype(np.float32))**2, -1, kernel)
        texture_map = np.sqrt(sqr_mean_l - mean_l**2)
        
        # Normalize texture map
        texture_map = texture_map / np.max(texture_map)
        
        # Create adaptive processing based on texture
        result_l = l.copy().astype(np.float32)
        
        # For high texture areas, use gentle processing
        high_texture_mask = texture_map > texture_threshold
        low_texture_mask = ~high_texture_mask
        
        # Process low texture areas (likely shadows) more aggressively
        if np.any(low_texture_mask):
            # Apply stronger enhancement to low texture areas
            clahe_strong = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            # Create a copy for processing
            low_texture_region = l.copy()
            low_texture_region[~low_texture_mask] = 0  # Zero out non-low-texture areas
            enhanced_low = clahe_strong.apply(low_texture_region)
            result_l[low_texture_mask] = enhanced_low[low_texture_mask]
        
        # Process high texture areas more gently
        if np.any(high_texture_mask):
            clahe_gentle = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
            # Create a copy for processing
            high_texture_region = l.copy()
            high_texture_region[~high_texture_mask] = 0  # Zero out non-high-texture areas
            enhanced_high = clahe_gentle.apply(high_texture_region)
            result_l[high_texture_mask] = enhanced_high[high_texture_mask]
        
        # Reconstruct image
        enhanced_lab = cv2.merge([result_l.astype(np.uint8), a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _advanced_shadow_pattern_detection(self, image):
        """
        Advanced method to differentiate real shadows from fabric patterns
        Uses multiple criteria to identify true shadows vs fabric texture
        """
        # Convert to LAB for better analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # 1. Gradient analysis - shadows have smooth gradients, patterns have sharp edges
        grad_x = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 2. Local contrast analysis - patterns have high local contrast
        kernel = np.ones((9, 9), np.float32) / 81
        local_mean = cv2.filter2D(l.astype(np.float32), -1, kernel)
        local_contrast = np.abs(l.astype(np.float32) - local_mean)
        
        # 3. Frequency analysis - shadows are low frequency, patterns are high frequency
        # Use Laplacian to detect high frequency components
        laplacian = cv2.Laplacian(l, cv2.CV_64F)
        frequency_map = np.abs(laplacian)
        
        # 4. Color consistency - shadows maintain color ratios, patterns may not
        # Calculate color ratio consistency
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        b, g, r = cv2.split(bgr)
        
        # Avoid division by zero
        g_safe = np.maximum(g, 1)
        r_safe = np.maximum(r, 1)
        
        br_ratio = b.astype(np.float32) / r_safe
        bg_ratio = b.astype(np.float32) / g_safe
        gr_ratio = g.astype(np.float32) / r_safe
        
        # Calculate ratio consistency (low variance = shadow, high variance = pattern)
        kernel_ratio = np.ones((15, 15), np.float32) / 225
        br_consistency = cv2.filter2D(br_ratio, -1, kernel_ratio)
        bg_consistency = cv2.filter2D(bg_ratio, -1, kernel_ratio)
        gr_consistency = cv2.filter2D(gr_ratio, -1, kernel_ratio)
        
        # Combine all criteria
        # Normalize each criterion
        gradient_norm = gradient_magnitude / np.max(gradient_magnitude)
        contrast_norm = local_contrast / np.max(local_contrast)
        frequency_norm = frequency_map / np.max(frequency_map)
        
        # Calculate ratio variance (higher = more pattern-like)
        ratio_variance = np.var([br_consistency, bg_consistency, gr_consistency], axis=0)
        ratio_variance_norm = ratio_variance / np.max(ratio_variance)
        
        # Shadow probability map (higher = more likely to be shadow)
        # Shadows: low gradient, low contrast, low frequency, high color consistency
        shadow_probability = (
            1.0 - gradient_norm +      # Low gradient = shadow
            1.0 - contrast_norm +      # Low contrast = shadow  
            1.0 - frequency_norm +     # Low frequency = shadow
            1.0 - ratio_variance_norm  # High consistency = shadow
        ) / 4.0
        
        # Create shadow mask (threshold can be adjusted)
        # Use adaptive threshold based on image characteristics
        threshold = np.percentile(shadow_probability, 75)  # Top 25% most shadow-like pixels
        shadow_mask = shadow_probability > threshold
        
        return shadow_mask, shadow_probability
    
    def _intelligent_shadow_removal(self, image, preserve_texture=True):
        """
        Intelligent shadow removal that differentiates shadows from fabric patterns
        Preserves fabric characteristics: color, shine, and patterns
        """
        # Detect shadow regions
        shadow_mask, shadow_probability = self._advanced_shadow_pattern_detection(image)
        
        # Convert to LAB for processing
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Create result
        result_l = l.copy().astype(np.float32)
        
        if np.any(shadow_mask):
            # Process only shadow regions
            shadow_regions = shadow_mask.astype(np.uint8)
            
            # Apply adaptive enhancement to shadow regions only
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            
            # Create a copy for processing
            l_copy = l.copy()
            l_copy[~shadow_mask] = 0  # Zero out non-shadow areas
            
            # Apply CLAHE to shadow regions
            enhanced_shadows = clahe.apply(l_copy)
            
            # Blend the enhanced shadows back
            # Use shadow probability for smooth blending
            blend_factor = shadow_probability[shadow_mask]
            result_l[shadow_mask] = (
                (1 - blend_factor) * l[shadow_mask] + 
                blend_factor * enhanced_shadows[shadow_mask]
            )
        
        # Preserve original texture in non-shadow areas
        if preserve_texture:
            # Extract original texture pattern
            original_texture = l.astype(np.float32) / 255.0
            
            # Apply texture preservation to non-shadow areas
            non_shadow_mask = ~shadow_mask
            if np.any(non_shadow_mask):
                # Preserve original brightness variations in non-shadow areas
                texture_factor = 0.3  # How much original texture to preserve
                result_l[non_shadow_mask] = (
                    (1 - texture_factor) * result_l[non_shadow_mask] +
                    texture_factor * l[non_shadow_mask]
                )
        
        # Reconstruct image
        enhanced_lab = cv2.merge([result_l.astype(np.uint8), a, b])
        result = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _fabric_characteristic_preservation(self, original, processed):
        """
        Enhance processed image to better preserve fabric characteristics
        """
        # Convert both to LAB
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        proc_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        
        orig_l, orig_a, orig_b = cv2.split(orig_lab)
        proc_l, proc_a, proc_b = cv2.split(proc_lab)
        
        # Preserve original color characteristics
        # Calculate color ratios from original
        orig_bgr = cv2.cvtColor(orig_lab, cv2.COLOR_LAB2BGR)
        orig_b, orig_g, orig_r = cv2.split(orig_bgr)
        
        # Avoid division by zero
        orig_g_safe = np.maximum(orig_g, 1)
        orig_r_safe = np.maximum(orig_r, 1)
        
        # Calculate original color ratios
        orig_br_ratio = orig_b.astype(np.float32) / orig_r_safe
        orig_bg_ratio = orig_b.astype(np.float32) / orig_g_safe
        orig_gr_ratio = orig_g.astype(np.float32) / orig_r_safe
        
        # Apply color ratio preservation
        proc_bgr = cv2.cvtColor(proc_lab, cv2.COLOR_LAB2BGR)
        proc_b, proc_g, proc_r = cv2.split(proc_bgr)
        
        # Preserve original color relationships
        color_preservation_factor = 0.4  # How much to preserve original colors
        
        # Adjust processed colors to maintain original ratios
        proc_b_adjusted = (
            (1 - color_preservation_factor) * proc_b +
            color_preservation_factor * (orig_br_ratio * proc_r + orig_bg_ratio * proc_g) / 2
        )
        
        proc_g_adjusted = (
            (1 - color_preservation_factor) * proc_g +
            color_preservation_factor * (proc_b / orig_bg_ratio + orig_gr_ratio * proc_r) / 2
        )
        
        proc_r_adjusted = (
            (1 - color_preservation_factor) * proc_r +
            color_preservation_factor * (proc_b / orig_br_ratio + proc_g / orig_gr_ratio) / 2
        )
        
        # Reconstruct BGR
        proc_bgr_adjusted = cv2.merge([
            np.clip(proc_b_adjusted, 0, 255).astype(np.uint8),
            np.clip(proc_g_adjusted, 0, 255).astype(np.uint8),
            np.clip(proc_r_adjusted, 0, 255).astype(np.uint8)
        ])
        
        # Convert back to LAB
        result_lab = cv2.cvtColor(proc_bgr_adjusted, cv2.COLOR_BGR2LAB)
        
        # Preserve original brightness patterns for shine
        orig_brightness_pattern = orig_l.astype(np.float32) / 255.0
        result_l, result_a, result_b = cv2.split(result_lab)
        
        # Apply original brightness pattern to preserve shine
        shine_preservation_factor = 0.2
        result_l = (
            (1 - shine_preservation_factor) * result_l.astype(np.float32) +
            shine_preservation_factor * (orig_brightness_pattern * 255)
        )
        
        # Reconstruct final image
        final_lab = cv2.merge([
            np.clip(result_l, 0, 255).astype(np.uint8),
            result_a,
            result_b
        ])
        
        result = cv2.cvtColor(final_lab, cv2.COLOR_LAB2BGR)
        
        return result
    
    def _fabric_preserving_shadow_removal(self, image, preserve_texture=True, preserve_color=True, preserve_shine=True):
        """
        Advanced fabric-preserving shadow removal
        Combines intelligent shadow detection with fabric characteristic preservation
        """
        # Step 1: Intelligent shadow removal
        shadow_removed = self._intelligent_shadow_removal(image, preserve_texture=preserve_texture)
        
        # Step 2: Preserve fabric characteristics
        if preserve_color or preserve_shine:
            result = self._fabric_characteristic_preservation(image, shadow_removed)
        else:
            result = shadow_removed
        
        return result
    
    def _multi_scale_shadow_removal(self, image, scales=[1, 2, 4]):
        """
        Multi-scale shadow removal
        Combines different scales for comprehensive shadow removal
        """
        results = []
        
        for scale in scales:
            # Resize image
            h, w = image.shape[:2]
            new_h, new_w = h // scale, w // scale
            
            if new_h < 10 or new_w < 10:
                continue
                
            resized = cv2.resize(image, (new_w, new_h))
            
            # Apply shadow removal at this scale
            if scale == 1:
                # Use retinex for original scale
                processed = self._retinex_shadow_removal(resized)
            else:
                # Use adaptive histogram for smaller scales
                processed = self._adaptive_histogram_matching(resized)
            
            # Resize back to original size
            processed = cv2.resize(processed, (w, h))
            results.append(processed)
        
        # Combine results using weighted average
        if len(results) == 1:
            return results[0]
        
        # Weight smaller scales more heavily for detail preservation
        weights = [1.0 / scale for scale in scales[:len(results)]]
        weights = np.array(weights) / np.sum(weights)
        
        result = np.zeros_like(image, dtype=np.float32)
        for i, (res, weight) in enumerate(zip(results, weights)):
            result += weight * res.astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def compare_methods(self, image, save_results=True):
        """
        Compare all shadow removal methods on the same image
        """
        results = {}
        
        for method_name in self.methods.keys():
            print(f"[INFO] Testing {method_name}...")
            try:
                results[method_name] = self.remove_shadows(image, method=method_name)
            except Exception as e:
                print(f"[WARNING] {method_name} failed: {e}")
                continue
        
        # Display results
        n_methods = len(results)
        if n_methods == 0:
            print("[ERROR] No methods succeeded")
            return None
        
        fig, axes = plt.subplots(2, (n_methods + 1) // 2, figsize=(20, 10))
        axes = axes.flatten() if n_methods > 1 else [axes]
        
        # Original image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Results
        for i, (method, result) in enumerate(results.items(), 1):
            if i < len(axes):
                result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                axes[i].imshow(result_rgb)
                axes[i].set_title(f"{method}")
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(results) + 1, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def assess_quality(self, original, processed):
        """
        Assess the quality of shadow removal results
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            Dictionary with quality metrics
        """
        # Convert to grayscale for analysis
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Calculate quality metrics
        metrics = {}
        
        # 1. Contrast improvement
        orig_contrast = np.std(orig_gray)
        proc_contrast = np.std(proc_gray)
        metrics['contrast_improvement'] = (proc_contrast - orig_contrast) / orig_contrast
        
        # 2. Shadow reduction (using local standard deviation)
        orig_local_std = self._calculate_local_std(orig_gray)
        proc_local_std = self._calculate_local_std(proc_gray)
        metrics['shadow_reduction'] = np.mean(orig_local_std - proc_local_std)
        
        # 3. Color preservation (using LAB color space)
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
        proc_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        
        # Calculate color difference
        color_diff = np.mean(np.abs(orig_lab.astype(np.float32) - proc_lab.astype(np.float32)))
        metrics['color_preservation'] = 1.0 / (1.0 + color_diff / 100.0)  # Normalize
        
        # 4. Overall quality score
        metrics['overall_quality'] = (
            0.3 * metrics['contrast_improvement'] +
            0.3 * metrics['shadow_reduction'] / 50.0 +  # Normalize
            0.4 * metrics['color_preservation']
        )
        
        return metrics
    
    def _calculate_local_std(self, image, kernel_size=15):
        """Calculate local standard deviation"""
        try:
            # Ensure image is 2D
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Convert to float32
            img_float = image.astype(np.float32)
            
            # Create kernel
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            
            # Calculate mean and squared mean
            mean = cv2.filter2D(img_float, -1, kernel)
            sqr_mean = cv2.filter2D(img_float**2, -1, kernel)
            
            # Calculate standard deviation
            local_std = np.sqrt(np.maximum(sqr_mean - mean**2, 0))
            
            return local_std
        except Exception as e:
            print(f"Warning: Error in _calculate_local_std: {e}")
            # Return a simple standard deviation as fallback
            return np.full_like(image, np.std(image), dtype=np.float32)
    
    def optimize_parameters(self, image, method='auto', max_iterations=5):
        """
        Optimize parameters for the best shadow removal results
        
        Args:
            image: Input image
            method: Shadow removal method
            max_iterations: Maximum optimization iterations
            
        Returns:
            Best result and optimized parameters
        """
        print(f"[INFO] Optimizing parameters for {method} method...")
        
        best_result = None
        best_score = -float('inf')
        best_params = {}
        
        # Define parameter ranges for optimization
        if method == 'retinex':
            param_ranges = {
                'sigma': [10, 15, 20, 25],
                'alpha': [100, 125, 150],
                'beta': [40, 46, 50]
            }
        elif method == 'adaptive_histogram':
            param_ranges = {
                'clip_limit': [2.0, 3.0, 4.0, 5.0],
                'tile_size': [4, 6, 8, 10]
            }
        elif method == 'texture_aware':
            param_ranges = {
                'texture_threshold': [0.05, 0.1, 0.15, 0.2]
            }
        else:
            # For other methods, use default parameters
            result = self.remove_shadows(image, method=method)
            return result, {}
        
        # Grid search optimization
        from itertools import product
        
        param_combinations = list(product(*param_ranges.values()))
        param_names = list(param_ranges.keys())
        
        for i, param_combo in enumerate(param_combinations):
            params = dict(zip(param_names, param_combo))
            
            try:
                result = self.remove_shadows(image, method=method, **params)
                if result is not None:
                    metrics = self.assess_quality(image, result)
                    score = metrics['overall_quality']
                    
                    if score > best_score:
                        best_score = score
                        best_result = result
                        best_params = params
                        
                    print(f"[INFO] Iteration {i+1}/{len(param_combinations)}: Score = {score:.3f}")
                else:
                    print(f"[WARNING] Iteration {i+1}/{len(param_combinations)}: No result")
                
            except Exception as e:
                print(f"[WARNING] Parameter combination failed: {e}")
                continue
        
        print(f"[INFO] Best score: {best_score:.3f} with parameters: {best_params}")
        return best_result, best_params
    
    def batch_process(self, image_paths, output_dir="results", method='auto'):
        """
        Process multiple images in batch
        
        Args:
            image_paths: List of image paths
            output_dir: Output directory for results
            method: Shadow removal method
            
        Returns:
            Dictionary with results for each image
        """
        import os
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"[INFO] Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Could not load {image_path}")
                continue
            
            # Process image
            try:
                result = self.remove_shadows(image, method=method)
                
                # Save result
                filename = os.path.basename(image_path)
                name, ext = os.path.splitext(filename)
                output_path = os.path.join(output_dir, f"{name}_shadow_removed{ext}")
                cv2.imwrite(output_path, result)
                
                # Assess quality
                metrics = self.assess_quality(image, result)
                
                results[image_path] = {
                    'output_path': output_path,
                    'metrics': metrics
                }
                
                print(f"[INFO] Saved: {output_path}")
                print(f"[INFO] Quality score: {metrics['overall_quality']:.3f}")
                
            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {e}")
                continue
        
        return results
    
    def uniformize_color(self, image, reference_region=None, method='smart_fabric'):
        """
        Smart fabric color uniformization that respects fabric characteristics
        
        Args:
            image: Input image (BGR format)
            reference_region: Tuple (x, y, w, h) for reference region, or None for auto-detection
            method: 'smart_fabric', 'brightest', 'most_saturated', or 'center'
        
        Returns:
            Smart color-uniformized image respecting fabric nature
        """
        if reference_region is None:
            # Auto-detect best reference region
            reference_region = self._detect_best_color_region(image, method)
        
        x, y, w, h = reference_region
        
        # Extract reference color from the selected region
        ref_region = image[y:y+h, x:x+w]
        
        # Get the base color from the center pixel
        center_x, center_y = w//2, h//2
        base_color = ref_region[center_y, center_x].astype(np.float32)
        
        # Analyze fabric type and create appropriate texture
        fabric_type = self._analyze_fabric_type(image)
        result = self._create_smart_fabric_texture(image, base_color, fabric_type)
        
        return result
    
    def _uniformize_color_smart_patch(self, image, reference_region=None, threshold=0.3):
        """Smart patch replacement - replace problematic zones with good zones"""
        if reference_region is not None:
            x, y, w, h = reference_region
            ref_region = image[y:y+h, x:x+w]
            
            # Get the base color from the center pixel
            center_x, center_y = w//2, h//2
            base_color = ref_region[center_y, center_x].astype(np.float32)
        else:
            # Use center region as reference
            h_img, w_img = image.shape[:2]
            center_x, center_y = w_img//2, h_img//2
            patch_size = 20
            x = center_x - patch_size//2
            y = center_y - patch_size//2
            ref_region = image[y:y+patch_size, x:x+patch_size]
            base_color = ref_region[patch_size//2, patch_size//2].astype(np.float32)
        
        # Apply smart patch replacement
        result = self._smart_patch_replacement(image, threshold)
        
        return result
    
    def _create_realistic_texture(self, image, base_color):
        """Create highly realistic texture with natural variations and depth"""
        h, w = image.shape[:2]
        
        # Create base color layer
        result = np.full_like(image, base_color, dtype=np.float32)
        
        # Add strong fabric texture variations
        fabric_texture = self._generate_realistic_fabric_texture(h, w)
        result = result * fabric_texture
        
        # Add realistic color temperature variations
        color_variation = self._generate_realistic_color_variations(h, w)
        result = result * color_variation
        
        # Add realistic lighting and shadows
        lighting = self._generate_realistic_lighting(h, w)
        result = result * lighting
        
        # Add fabric weave pattern
        weave_pattern = self._generate_fabric_weave(h, w)
        result = result * weave_pattern
        
        # Add surface irregularities
        surface_irregularities = self._generate_surface_irregularities(h, w)
        result = result + surface_irregularities
        
        # Ensure values stay within valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _generate_realistic_fabric_texture(self, h, w):
        """Generate highly realistic fabric texture with strong variations"""
        # Create multiple layers of noise for realistic texture
        texture = np.ones((h, w, 3), dtype=np.float32)
        
        # Layer 1: Fine grain noise (fabric fibers)
        fine_noise = np.random.normal(1.0, 0.15, (h, w, 1))
        fine_noise = np.repeat(fine_noise, 3, axis=2)
        texture *= fine_noise
        
        # Layer 2: Medium grain noise (fabric weave)
        medium_noise = np.random.normal(1.0, 0.08, (h, w, 1))
        medium_noise = np.repeat(medium_noise, 3, axis=2)
        texture *= medium_noise
        
        # Layer 3: Coarse grain noise (fabric irregularities)
        coarse_noise = np.random.normal(1.0, 0.12, (h, w, 1))
        coarse_noise = np.repeat(coarse_noise, 3, axis=2)
        texture *= coarse_noise
        
        return texture
    
    def _generate_realistic_color_variations(self, h, w):
        """Generate realistic color temperature and saturation variations"""
        color_variation = np.ones((h, w, 3), dtype=np.float32)
        
        # Create multiple frequency variations
        x = np.linspace(0, 8*np.pi, w)
        y = np.linspace(0, 8*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Blue channel - strong variations
        color_variation[:, :, 0] = 1.0 + 0.2 * (np.sin(X) * np.cos(Y) + np.sin(2*X) * np.cos(2*Y))
        
        # Green channel - medium variations
        color_variation[:, :, 1] = 1.0 + 0.15 * (np.cos(X) * np.sin(Y) + np.cos(3*X) * np.sin(3*Y))
        
        # Red channel - subtle variations
        color_variation[:, :, 2] = 1.0 + 0.1 * (np.sin(4*X) * np.cos(4*Y) + np.sin(5*X) * np.cos(5*Y))
        
        return color_variation
    
    def _generate_realistic_lighting(self, h, w):
        """Generate realistic lighting with shadows and highlights"""
        lighting = np.ones((h, w, 3), dtype=np.float32)
        
        # Create directional lighting from top-left
        x = np.linspace(-2, 2, w)
        y = np.linspace(-2, 2, h)
        X, Y = np.meshgrid(x, y)
        
        # Main light source
        main_light = 0.4 * np.exp(-(X**2 + Y**2) / 3)
        
        # Secondary light source (fill light)
        fill_light = 0.2 * np.exp(-((X-1)**2 + (Y+1)**2) / 4)
        
        # Add random highlights (fabric sheen)
        highlights = np.random.random((h, w))
        highlights = np.where(highlights > 0.98, 0.4, 0)
        highlights = np.where(highlights > 0.95, 0.2, highlights)
        
        # Combine all lighting
        total_lighting = main_light + fill_light + highlights
        lighting = lighting * (1.0 + total_lighting[:, :, np.newaxis])
        
        return lighting
    
    def _generate_fabric_weave(self, h, w):
        """Generate realistic fabric weave pattern"""
        weave = np.ones((h, w, 3), dtype=np.float32)
        
        # Create weave pattern
        x = np.linspace(0, 20*np.pi, w)
        y = np.linspace(0, 20*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Warp and weft pattern
        warp_pattern = 0.3 * np.sin(X)
        weft_pattern = 0.3 * np.sin(Y)
        
        # Combine patterns
        weave_pattern = warp_pattern + weft_pattern
        weave = weave * (1.0 + weave_pattern[:, :, np.newaxis])
        
        return weave
    
    def _generate_surface_irregularities(self, h, w):
        """Generate surface irregularities and depth"""
        irregularities = np.zeros((h, w, 3), dtype=np.float32)
        
        # Create depth variations
        depth_noise = np.random.normal(0, 15, (h, w, 1))
        depth_noise = np.repeat(depth_noise, 3, axis=2)
        
        # Add some creases and folds
        x = np.linspace(0, 4*np.pi, w)
        y = np.linspace(0, 4*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        creases = 10 * np.sin(X) * np.cos(Y)
        creases = np.repeat(creases[:, :, np.newaxis], 3, axis=2)
        
        irregularities = depth_noise + creases
        
        return irregularities
    
    def _analyze_fabric_type(self, image):
        """Analyze fabric type to determine appropriate texture treatment"""
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture complexity
        texture_complexity = np.std(cv2.Laplacian(gray, cv2.CV_64F))
        
        # Calculate brightness variation
        brightness_variation = np.std(gray)
        
        # Calculate color uniformity
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        color_uniformity = np.std(l) + np.std(a) + np.std(b)
        
        # Determine fabric type
        if texture_complexity < 20 and color_uniformity < 30:
            return 'smooth_solid'  # Velours, soie lisse
        elif texture_complexity < 40 and brightness_variation < 50:
            return 'matte_solid'   # Coton, lin mat
        elif texture_complexity > 60:
            return 'textured'      # Tissu avec motifs
        else:
            return 'standard'      # Tissu standard
    
    def _create_smart_fabric_texture(self, image, base_color, fabric_type):
        """Create smart fabric texture preserving original shine and characteristics"""
        h, w = image.shape[:2]
        
        # Create base color layer
        result = np.full_like(image, base_color, dtype=np.float32)
        
        # Extract original shine/brightness pattern
        original_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        original_gray = cv2.resize(original_gray, (w, h))
        
        # Normalize original brightness pattern
        original_brightness = original_gray / 255.0
        
        # Apply original shine pattern to base color
        result = result * original_brightness[:, :, np.newaxis]
        
        # Add subtle texture variations based on fabric type
        if fabric_type == 'smooth_solid':
            # Velours, soie - très subtile
            result = self._add_subtle_texture(result, h, w, intensity=0.02)
        elif fabric_type == 'matte_solid':
            # Coton, lin - texture mate subtile
            result = self._add_subtle_texture(result, h, w, intensity=0.03)
        elif fabric_type == 'textured':
            # Tissu avec motifs - préserver les motifs originaux
            result = self._preserve_original_texture(result, image)
        else:
            # Tissu standard - texture équilibrée
            result = self._add_subtle_texture(result, h, w, intensity=0.04)
        
        # Ensure values stay within valid range
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _create_smooth_solid_texture(self, base_image, h, w):
        """Create smooth solid fabric texture (velours, soie)"""
        # Subtle shine effect
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Soft directional lighting
        shine = 0.15 * np.exp(-(X**2 + Y**2) / 1.5)
        
        # Very fine texture
        fine_noise = np.random.normal(1.0, 0.03, (h, w, 1))
        fine_noise = np.repeat(fine_noise, 3, axis=2)
        
        # Apply effects
        result = base_image * fine_noise
        result = result * (1.0 + shine[:, :, np.newaxis])
        
        return result
    
    def _create_matte_solid_texture(self, base_image, h, w):
        """Create matte solid fabric texture (coton, lin)"""
        # Matte finish with subtle variations
        matte_variation = np.random.normal(1.0, 0.05, (h, w, 1))
        matte_variation = np.repeat(matte_variation, 3, axis=2)
        
        # Subtle weave pattern
        x = np.linspace(0, 10*np.pi, w)
        y = np.linspace(0, 10*np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        weave = 0.05 * (np.sin(X) + np.cos(Y))
        weave = np.repeat(weave[:, :, np.newaxis], 3, axis=2)
        
        # Apply effects
        result = base_image * matte_variation
        result = result * (1.0 + weave)
        
        return result
    
    def _create_textured_fabric(self, base_image, original_image):
        """Create textured fabric preserving original patterns"""
        # Preserve original texture but uniformize color
        h, w = base_image.shape[:2]
        
        # Extract texture from original image
        gray_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        gray_original = cv2.resize(gray_original, (w, h))
        
        # Normalize texture
        texture_normalized = gray_original.astype(np.float32) / 255.0
        
        # Apply texture to base color
        result = base_image * texture_normalized[:, :, np.newaxis]
        
        return result
    
    def _create_standard_fabric_texture(self, base_image, h, w):
        """Create standard fabric texture with balanced shine and texture"""
        # Balanced shine
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        X, Y = np.meshgrid(x, y)
        
        shine = 0.08 * np.exp(-(X**2 + Y**2) / 2)
        
        # Moderate texture
        texture = np.random.normal(1.0, 0.06, (h, w, 1))
        texture = np.repeat(texture, 3, axis=2)
        
        # Subtle weave
        x_weave = np.linspace(0, 8*np.pi, w)
        y_weave = np.linspace(0, 8*np.pi, h)
        X_weave, Y_weave = np.meshgrid(x_weave, y_weave)
        
        weave = 0.03 * (np.sin(X_weave) + np.cos(Y_weave))
        weave = np.repeat(weave[:, :, np.newaxis], 3, axis=2)
        
        # Apply effects
        result = base_image * texture
        result = result * (1.0 + shine[:, :, np.newaxis])
        result = result * (1.0 + weave)
        
        return result
    
    def _detect_best_color_region(self, image, method='perfect_patch'):
        """Detect the best small perfect patch for color reference"""
        h, w = image.shape[:2]
        
        # Use a small square size (much smaller than before)
        patch_size = min(20, min(h, w) // 20)  # Very small square
        
        if method == 'center':
            # Use center region
            x = (w - patch_size) // 2
            y = (h - patch_size) // 2
            return (x, y, patch_size, patch_size)
        
        elif method == 'brightest':
            # Find brightest small patch
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            max_brightness = 0
            best_region = (w//2, h//2, patch_size, patch_size)
            
            # Search with smaller steps for more precision
            step = max(1, patch_size // 2)
            for y in range(0, h - patch_size, step):
                for x in range(0, w - patch_size, step):
                    patch = gray[y:y+patch_size, x:x+patch_size]
                    brightness = np.mean(patch)
                    if brightness > max_brightness:
                        max_brightness = brightness
                        best_region = (x, y, patch_size, patch_size)
            
            return best_region
        
        elif method == 'most_saturated':
            # Find most saturated small patch
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            s = hsv[:, :, 1]
            
            max_saturation = 0
            best_region = (w//2, h//2, patch_size, patch_size)
            
            step = max(1, patch_size // 2)
            for y in range(0, h - patch_size, step):
                for x in range(0, w - patch_size, step):
                    patch = s[y:y+patch_size, x:x+patch_size]
                    saturation = np.mean(patch)
                    if saturation > max_saturation:
                        max_saturation = saturation
                        best_region = (x, y, patch_size, patch_size)
            
            return best_region
        
        else:  # perfect_patch
            # Find the most uniform small patch (lowest color variance)
            region_size = patch_size
            
            min_variance = float('inf')
            best_region = (w//2, h//2, region_size, region_size)
            
            step = max(1, region_size // 2)
            for y in range(0, h - region_size, step):
                for x in range(0, w - region_size, step):
                    patch = image[y:y+region_size, x:x+region_size]
                    
                    # Calculate color variance in BGR
                    b_var = np.var(patch[:, :, 0])
                    g_var = np.var(patch[:, :, 1])
                    r_var = np.var(patch[:, :, 2])
                    total_variance = b_var + g_var + r_var
                    
                    if total_variance < min_variance:
                        min_variance = total_variance
                        best_region = (x, y, region_size, region_size)
            
            return best_region
    
    def remove_shadows_with_color_uniformization(self, image, shadow_method='auto', color_method='dominant'):
        """
        Remove shadows and uniformize color in one step
        
        Args:
            image: Input image
            shadow_method: Shadow removal method
            color_method: Color uniformization method
        
        Returns:
            Processed image with shadows removed and color uniformized
        """
        # First remove shadows
        shadow_removed = self.remove_shadows(image, method=shadow_method)
        
        # Then uniformize color
        result = self.uniformize_color(shadow_removed, method=color_method)
        
        return result
    
    def _add_subtle_texture(self, base_image, h, w, intensity=0.03):
        """Add very subtle texture variations"""
        # Very fine noise
        fine_noise = np.random.normal(1.0, intensity, (h, w, 1))
        fine_noise = np.repeat(fine_noise, 3, axis=2)
        
        # Apply subtle texture
        result = base_image * fine_noise
        
        return result
    
    def _preserve_original_texture(self, base_image, original_image):
        """Preserve original texture patterns while uniformizing color"""
        h, w = base_image.shape[:2]
        
        # Extract original texture
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        original_gray = cv2.resize(original_gray, (w, h))
        
        # Normalize texture
        texture_normalized = original_gray.astype(np.float32) / 255.0
        
        # Apply original texture to base color
        result = base_image * texture_normalized[:, :, np.newaxis]
        
        return result
    
    def _smart_patch_replacement(self, image, threshold=0.3):
        """Replace problematic zones with good zones using color difference coefficient"""
        h, w = image.shape[:2]
        result = image.copy()
        
        # Convert to LAB for better color difference calculation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Calculate color variation coefficient for each pixel
        patch_size = 15
        step = 5
        
        # Create coefficient map
        coefficient_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(0, h - patch_size, step):
            for x in range(0, w - patch_size, step):
                # Extract patch
                patch = lab[y:y+patch_size, x:x+patch_size]
                
                # Calculate color variation coefficient
                mean_color = np.mean(patch, axis=(0, 1))
                color_diff = np.sqrt(np.sum((patch - mean_color) ** 2, axis=2))
                coefficient = np.mean(color_diff)
                
                # Store coefficient for the patch area
                coefficient_map[y:y+patch_size, x:x+patch_size] = coefficient
        
        # Find good zones (low coefficient) and bad zones (high coefficient)
        good_zones = coefficient_map < threshold
        bad_zones = coefficient_map >= threshold
        
        if np.any(bad_zones) and np.any(good_zones):
            # For each bad zone, find the best matching good zone
            bad_coords = np.where(bad_zones)
            good_coords = np.where(good_zones)
            
            for i in range(len(bad_coords[0])):
                by, bx = bad_coords[0][i], bad_coords[1][i]
                
                # Find closest good zone with similar color
                bad_color = lab[by, bx]
                distances = []
                
                for j in range(len(good_coords[0])):
                    gy, gx = good_coords[0][j], good_coords[1][j]
                    good_color = lab[gy, gx]
                    distance = np.sqrt(np.sum((bad_color - good_color) ** 2))
                    distances.append(distance)
                
                if distances:
                    best_idx = np.argmin(distances)
                    best_gy, best_gx = good_coords[0][best_idx], good_coords[1][best_idx]
                    
                    # Replace bad zone with good zone
                    result[by, bx] = image[best_gy, best_gx]
        
        return result
    
    def _detect_perfect_fabric_region(self, image, min_region_size=150):
        """
        Détecte la zone la plus CLAIRE et homogène de l'image
        Utile pour les tissus unis sans motifs
        """
        # Convertir en LAB pour une meilleure analyse
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        h, w = image.shape[:2]
        
        # Taille de la région à analyser (plus grande pour éviter les détails)
        region_size = min_region_size
        
        best_region = None
        best_score = float('inf')
        
        # Analyser différentes régions de l'image
        step = max(1, region_size // 3)  # Pas plus petit pour éviter trop de détails
        
        for y in range(0, h - region_size, step):
            for x in range(0, w - region_size, step):
                # Extraire la région
                region = image[y:y+region_size, x:x+region_size]
                region_lab = lab[y:y+region_size, x:x+region_size]
                l_region = region_lab[:, :, 0]
                
                # 1. PRIORITÉ: Zone la plus CLAIRE (luminosité moyenne élevée)
                brightness_mean = np.mean(l_region)
                
                # 2. Homogénéité (variance faible)
                color_variance = np.var(region.reshape(-1, 3), axis=0)
                color_score = np.sum(color_variance)
                
                brightness_variance = np.var(l_region)
                
                # Score combiné: PRIORITÉ à la clarté, puis homogénéité
                # Plus la zone est claire ET homogène, plus le score est bas
                combined_score = (255 - brightness_mean) * 0.7 + (color_score + brightness_variance) * 0.3
                
                # Garder la meilleure région (score le plus bas)
                if combined_score < best_score:
                    best_score = combined_score
                    best_region = (x, y, region_size, region_size)
        
        return best_region, best_score
    
    def _is_solid_color_fabric(self, image, threshold=0.1):
        """
        Détermine si l'image représente un tissu uni (sans motifs)
        """
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculer la complexité de texture
        # Utiliser le Laplacien pour détecter les variations
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_complexity = np.std(laplacian)
        
        # Normaliser par la taille de l'image
        normalized_complexity = texture_complexity / (image.shape[0] * image.shape[1])
        
        # Si la complexité est faible, c'est probablement un tissu uni
        is_solid = normalized_complexity < threshold
        
        return is_solid, normalized_complexity
    
    def create_perfect_fabric_sample(self, image, zoom_factor=1.3, min_region_size=150):
        """
        Crée un échantillon parfait de tissu en zoomant sur la zone la plus CLAIRE
        
        Args:
            image: Image d'entrée
            zoom_factor: Facteur de zoom LOGIQUE (1.3 = zoom léger, évite les détails)
            min_region_size: Taille minimale de la région à analyser
        
        Returns:
            Échantillon parfait du tissu ou None si pas de tissu uni
        """
        print("[INFO] Analyse de l'image pour détecter un tissu uni...")
        
        # Vérifier si c'est un tissu uni
        is_solid, complexity = self._is_solid_color_fabric(image)
        
        print(f"[INFO] Complexité de texture: {complexity:.6f}")
        print(f"[INFO] Tissu uni détecté: {is_solid}")
        
        if not is_solid:
            print("[WARNING] L'image ne semble pas être un tissu uni avec motifs")
            print("[INFO] Cette fonctionnalité est optimisée pour les tissus unis")
            return None
        
        # Détecter la zone la plus CLAIRE
        print("[INFO] Recherche de la zone la plus CLAIRE et homogène...")
        perfect_region, score = self._detect_perfect_fabric_region(image, min_region_size)
        
        if perfect_region is None:
            print("[ERROR] Aucune zone parfaite trouvée")
            return None
        
        x, y, w, h = perfect_region
        print(f"[INFO] Zone la plus CLAIRE trouvée: ({x}, {y}, {w}, {h})")
        print(f"[INFO] Score de clarté/homogénéité: {score:.2f}")
        
        # Extraire la région parfaite
        perfect_region_img = image[y:y+h, x:x+w]
        
        # Appliquer un zoom LOGIQUE (pas trop agressif)
        new_width = int(w * zoom_factor)
        new_height = int(h * zoom_factor)
        
        # Utiliser INTER_LINEAR pour éviter les artefacts de zoom excessif
        zoomed_sample = cv2.resize(perfect_region_img, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        print(f"[INFO] Échantillon créé: {new_width}x{new_height} (zoom LOGIQUE x{zoom_factor})")
        
        return zoomed_sample
    
    def create_fabric_texture_from_sample(self, sample, target_size=None, preserve_characteristics=True):
        """
        Crée une texture de tissu à partir d'un échantillon parfait
        
        Args:
            sample: Échantillon parfait du tissu
            target_size: Taille cible (width, height) ou None pour garder la taille originale
            preserve_characteristics: Préserver les caractéristiques du tissu
        
        Returns:
            Texture de tissu générée
        """
        if sample is None:
            return None
        
        if target_size is None:
            target_size = (sample.shape[1], sample.shape[0])
        
        target_width, target_height = target_size
        
        # Si la taille cible est la même que l'échantillon, retourner l'échantillon
        if target_width == sample.shape[1] and target_height == sample.shape[0]:
            return sample.copy()
        
        # Créer une texture en répétant l'échantillon
        # Calculer combien de fois répéter l'échantillon
        sample_h, sample_w = sample.shape[:2]
        
        repeat_x = int(np.ceil(target_width / sample_w))
        repeat_y = int(np.ceil(target_height / sample_h))
        
        # Répéter l'échantillon
        repeated = np.tile(sample, (repeat_y, repeat_x, 1))
        
        # Découper à la taille cible
        result = repeated[:target_height, :target_width]
        
        if preserve_characteristics:
            # Ajouter des variations subtiles pour simuler un tissu plus grand
            result = self._add_subtle_fabric_variations(result, sample)
        
        return result
    
    def _add_subtle_fabric_variations(self, texture, original_sample):
        """
        Ajoute des variations subtiles à la texture pour la rendre plus réaliste
        """
        h, w = texture.shape[:2]
        
        # Analyser les caractéristiques de l'échantillon original
        sample_mean = np.mean(original_sample, axis=(0, 1))
        sample_std = np.std(original_sample, axis=(0, 1))
        
        # Créer des variations subtiles
        # 1. Variations de luminosité très subtiles
        brightness_variation = np.random.normal(0, sample_std[0] * 0.1, (h, w, 1))
        brightness_variation = np.repeat(brightness_variation, 3, axis=2)
        
        # 2. Variations de couleur très subtiles
        color_variation = np.random.normal(0, sample_std * 0.05, (h, w, 3))
        
        # 3. Variations de texture (bruit très fin)
        texture_noise = np.random.normal(0, 2, (h, w, 3))
        
        # Appliquer les variations
        result = texture.astype(np.float32)
        result = result + brightness_variation + color_variation + texture_noise
        
        # S'assurer que les valeurs restent dans la plage valide
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def _find_lightest_color_in_image(self, image):
        """
        Trouve la couleur la plus claire dans l'image originale
        """
        # Convertir en LAB pour une meilleure analyse de la luminosité
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Trouver les pixels les plus clairs (top 5%)
        threshold = np.percentile(l, 95)
        lightest_mask = l >= threshold
        
        # Extraire les couleurs des pixels les plus clairs
        lightest_pixels = image[lightest_mask]
        
        if len(lightest_pixels) > 0:
            # Calculer la couleur moyenne des pixels les plus clairs
            lightest_color = np.mean(lightest_pixels, axis=0)
            return lightest_color.astype(np.uint8)
        else:
            # Fallback: couleur moyenne de l'image
            return np.mean(image.reshape(-1, 3), axis=0).astype(np.uint8)
    
    def _uniformize_zoomed_region(self, zoomed_sample, target_color, uniformization_strength=0.8):
        """
        Uniformise la zone zoomée et ajuste la couleur vers la couleur cible
        Version améliorée pour capturer le maximum de défauts
        
        Args:
            zoomed_sample: Échantillon zoomé à uniformiser
            target_color: Couleur cible (couleur la plus claire de l'original)
            uniformization_strength: Force d'uniformisation (0.0 = pas d'uniformisation, 1.0 = complètement uniforme)
        """
        print(f"[INFO] Uniformisation MAXIMALE avec force: {uniformization_strength}")
        
        h, w = zoomed_sample.shape[:2]
        
        # ÉTAPE 1: Détection et élimination des défauts
        print(f"[INFO] Detection et elimination des defauts...")
        
        # Convertir en LAB pour une meilleure analyse
        lab = cv2.cvtColor(zoomed_sample, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Détecter les variations de luminosité (défauts potentiels)
        l_blur = cv2.GaussianBlur(l_channel, (15, 15), 0)
        l_diff = np.abs(l_channel.astype(np.float32) - l_blur.astype(np.float32))
        
        # Seuil adaptatif pour détecter les défauts
        defect_threshold = np.percentile(l_diff, 85)  # Top 15% des variations
        defect_mask = l_diff > defect_threshold
        
        print(f"[INFO] Defauts detectes: {np.sum(defect_mask)} pixels ({np.sum(defect_mask)/defect_mask.size*100:.1f}%)")
        
        # ÉTAPE 2: Uniformisation multi-échelle
        print(f"[INFO] Uniformisation multi-echelle...")
        
        # Flous de différentes tailles pour capturer tous les défauts
        blur_sizes = [
            max(3, int(min(h, w) * 0.02)),   # Petit flou
            max(5, int(min(h, w) * 0.03)),   # Moyen flou
            max(7, int(min(h, w) * 0.05)),   # Grand flou
            max(9, int(min(h, w) * 0.08))    # Très grand flou
        ]
        
        # S'assurer que les tailles sont impaires
        blur_sizes = [size if size % 2 == 1 else size + 1 for size in blur_sizes]
        
        # Appliquer les flous multi-échelle
        uniformized = zoomed_sample.copy().astype(np.float32)
        
        for i, blur_kernel_size in enumerate(blur_sizes):
            print(f"[INFO] Application flou {i+1}/{len(blur_sizes)}: {blur_kernel_size}x{blur_kernel_size}")
            blurred = cv2.GaussianBlur(zoomed_sample, (blur_kernel_size, blur_kernel_size), 0)
            
            # Mélanger avec un poids décroissant
            weight = uniformization_strength * (1.0 - i * 0.2)
            weight = max(0.1, weight)  # Garder au moins 10%
            
            uniformized = (1 - weight) * uniformized + weight * blurred.astype(np.float32)
        
        # ÉTAPE 3: Correction spécifique des défauts détectés
        print(f"[INFO] Correction specifique des defauts...")
        
        if np.sum(defect_mask) > 0:
            # Flou très fort sur les zones de défauts
            strong_blur = cv2.GaussianBlur(zoomed_sample, (21, 21), 0)
            
            # Remplacer les zones de défauts par le flou fort
            for c in range(3):  # Pour chaque canal BGR
                uniformized[:, :, c] = np.where(
                    defect_mask, 
                    strong_blur[:, :, c].astype(np.float32), 
                    uniformized[:, :, c]
                )
        
        # ÉTAPE 4: Uniformisation finale avec filtre bilatéral
        print(f"[INFO] Uniformisation finale avec filtre bilateral...")
        
        # Convertir en uint8 pour le filtre bilatéral
        temp = np.clip(uniformized, 0, 255).astype(np.uint8)
        
        # Filtre bilatéral pour uniformiser tout en préservant les contours
        bilateral = cv2.bilateralFilter(temp, 15, 80, 80)
        
        # Mélanger avec l'uniformisé précédent
        bilateral_float = bilateral.astype(np.float32)
        uniformized = 0.7 * uniformized + 0.3 * bilateral_float
        
        # ÉTAPE 5: Ajustement de couleur vers la cible
        print(f"[INFO] Ajustement de couleur vers la cible...")
        
        current_mean = np.mean(uniformized.reshape(-1, 3), axis=0)
        color_adjustment = target_color.astype(np.float32) - current_mean
        
        # Appliquer l'ajustement de couleur
        uniformized = uniformized + color_adjustment
        
        # ÉTAPE 6: Réduction finale des variations
        print(f"[INFO] Reduction finale des variations...")
        
        # Convertir en LAB pour ajuster la luminosité
        lab = cv2.cvtColor(uniformized.astype(np.uint8), cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Réduire la variance de luminosité de manière plus agressive
        l_mean = np.mean(l)
        l_adjusted = l_mean + (l - l_mean) * (1 - uniformization_strength * 0.8)  # Plus agressif
        
        # Reconstruire l'image LAB
        lab_uniformized = cv2.merge([l_adjusted.astype(np.uint8), a, b])
        uniformized = cv2.cvtColor(lab_uniformized, cv2.COLOR_LAB2BGR).astype(np.float32)
        
        # ÉTAPE 7: Lissage final pour éliminer les derniers défauts
        print(f"[INFO] Lissage final pour eliminer les derniers defauts...")
        
        # Flou final très léger
        final_blur = cv2.GaussianBlur(uniformized.astype(np.uint8), (5, 5), 0)
        uniformized = 0.9 * uniformized + 0.1 * final_blur.astype(np.float32)
        
        # S'assurer que les valeurs restent dans la plage valide
        uniformized = np.clip(uniformized, 0, 255).astype(np.uint8)
        
        print(f"[INFO] Uniformisation MAXIMALE terminee")
        
        return uniformized
    
    def create_uniformized_fabric_sample(self, image, zoom_factor=1.3, min_region_size=150, 
                                       uniformization_strength=0.8):
        """
        Crée un échantillon de tissu uniformisé avec couleur ajustée
        
        Args:
            image: Image d'entrée
            zoom_factor: Facteur de zoom logique
            min_region_size: Taille minimale de la région à analyser
            uniformization_strength: Force d'uniformisation (0.0-1.0)
        
        Returns:
            Échantillon uniformisé du tissu
        """
        print("[INFO] Création d'échantillon uniformisé...")
        
        # Vérifier si c'est un tissu uni
        is_solid, complexity = self._is_solid_color_fabric(image)
        
        if not is_solid:
            print("[WARNING] L'image ne semble pas être un tissu uni")
            return None
        
        # Trouver la couleur la plus claire dans l'image originale
        print("[INFO] Recherche de la couleur la plus claire...")
        lightest_color = self._find_lightest_color_in_image(image)
        print(f"[INFO] Couleur la plus claire trouvée: BGR({lightest_color[0]}, {lightest_color[1]}, {lightest_color[2]})")
        
        # Détecter la zone la plus claire
        print("[INFO] Recherche de la zone la plus claire...")
        perfect_region, score = self._detect_perfect_fabric_region(image, min_region_size)
        
        if perfect_region is None:
            print("[ERROR] Aucune zone parfaite trouvée")
            return None
        
        x, y, w, h = perfect_region
        print(f"[INFO] Zone claire trouvée: ({x}, {y}, {w}, {h})")
        
        # Extraire et zoomer la région
        perfect_region_img = image[y:y+h, x:x+w]
        new_width = int(w * zoom_factor)
        new_height = int(h * zoom_factor)
        
        zoomed_sample = cv2.resize(perfect_region_img, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        print(f"[INFO] Échantillon zoomé créé: {new_width}x{new_height}")
        
        # Uniformiser l'échantillon zoomé
        print(f"[INFO] Uniformisation avec force: {uniformization_strength}")
        uniformized_sample = self._uniformize_zoomed_region(
            zoomed_sample, 
            lightest_color, 
            uniformization_strength
        )
        
        print("[INFO] Échantillon uniformisé créé avec succès")
        
        return uniformized_sample
    
    def create_uniformized_fabric_texture(self, image, target_size=(800, 600), 
                                        zoom_factor=1.3, uniformization_strength=0.8):
        """
        Crée une texture de tissu uniformisée à partir de l'image
        
        Args:
            image: Image d'entrée
            target_size: Taille cible de la texture (width, height)
            zoom_factor: Facteur de zoom pour l'échantillon
            uniformization_strength: Force d'uniformisation
        
        Returns:
            Texture de tissu uniformisée
        """
        print(f"[INFO] Création de texture uniformisée {target_size[0]}x{target_size[1]}...")
        
        # Créer l'échantillon uniformisé
        sample = self.create_uniformized_fabric_sample(
            image, 
            zoom_factor=zoom_factor, 
            uniformization_strength=uniformization_strength
        )
        
        if sample is None:
            print("[ERROR] Impossible de créer l'échantillon uniformisé")
            return None
        
        # Créer la texture à partir de l'échantillon uniformisé
        texture = self.create_fabric_texture_from_sample(
            sample, 
            target_size=target_size,
            preserve_characteristics=False  # Pas besoin de préserver, déjà uniformisé
        )
        
        if texture is not None:
            print(f"[INFO] Texture uniformisée créée: {target_size[0]}x{target_size[1]}")
        
        return texture
    
    def select_color_region_interactive(self, image, window_name="Selection de Couleur"):
        """
        Permet à l'utilisateur de sélectionner interactivement une région de couleur
        
        Args:
            image: Image d'entrée
            window_name: Nom de la fenêtre
        
        Returns:
            Tuple (region_coords, selected_color) ou None si annulé
        """
        import cv2
        
        # Variables pour la sélection
        drawing = False
        start_point = None
        end_point = None
        region_coords = None
        selected_color = None
        
        # Copie de l'image pour dessiner
        display_image = image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drawing, start_point, end_point, display_image
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                start_point = (x, y)
                end_point = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    end_point = (x, y)
                    # Redessiner l'image
                    display_image = image.copy()
                    cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                end_point = (x, y)
                # Redessiner le rectangle final
                display_image = image.copy()
                cv2.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
        
        # Créer la fenêtre et définir le callback
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print(f"[INFO] Instructions pour la selection:")
        print(f"1. Cliquez et glissez pour selectionner une region")
        print(f"2. Appuyez sur 'ENTER' pour confirmer la selection")
        print(f"3. Appuyez sur 'ESC' pour annuler")
        print(f"4. Appuyez sur 'SPACE' pour voir la couleur selectionnee")
        
        while True:
            cv2.imshow(window_name, display_image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                if start_point and end_point:
                    # Calculer les coordonnées de la région
                    x1, y1 = start_point
                    x2, y2 = end_point
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Vérifier que la région est valide
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        region_coords = (x1, y1, x2 - x1, y2 - y1)
                        
                        # Extraire la couleur moyenne de la région sélectionnée
                        region = image[y1:y2, x1:x2]
                        selected_color = np.mean(region.reshape(-1, 3), axis=0).astype(np.uint8)
                        
                        print(f"[INFO] Region selectionnee: ({x1}, {y1}, {x2-x1}, {y2-y1})")
                        print(f"[INFO] Couleur selectionnee: BGR({selected_color[0]}, {selected_color[1]}, {selected_color[2]})")
                        break
                    else:
                        print("[WARNING] Region trop petite, selectionnez une region plus grande")
                else:
                    print("[WARNING] Aucune region selectionnee")
            
            elif key == 27:  # ESC
                print("[INFO] Selection annulee")
                region_coords = None
                selected_color = None
                break
            
            elif key == 32:  # SPACE
                if start_point and end_point:
                    x1, y1 = start_point
                    x2, y2 = end_point
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    if x2 - x1 > 10 and y2 - y1 > 10:
                        region = image[y1:y2, x1:x2]
                        color = np.mean(region.reshape(-1, 3), axis=0).astype(np.uint8)
                        print(f"[INFO] Couleur de la region: BGR({color[0]}, {color[1]}, {color[2]})")
                        
                        # Afficher un aperçu de la couleur
                        color_preview = np.full((100, 200, 3), color, dtype=np.uint8)
                        cv2.imshow("Couleur Selectionnee", color_preview)
        
        cv2.destroyAllWindows()
        
        if region_coords and selected_color is not None:
            return region_coords, selected_color
        else:
            return None, None
    
    def create_custom_fabric_sample(self, image, region_coords, target_color, zoom_factor=1.3, 
                                  uniformization_strength=0.8):
        """
        Crée un échantillon de tissu à partir d'une région sélectionnée manuellement
        
        Args:
            image: Image d'entrée
            region_coords: Coordonnées de la région (x, y, w, h)
            target_color: Couleur cible à préserver
            zoom_factor: Facteur de zoom
            uniformization_strength: Force d'uniformisation
        
        Returns:
            Échantillon uniformisé avec la couleur préservée
        """
        print(f"[INFO] Creation d'echantillon personnalise...")
        
        x, y, w, h = region_coords
        print(f"[INFO] Region utilisee: ({x}, {y}, {w}, {h})")
        print(f"[INFO] Couleur cible: BGR({target_color[0]}, {target_color[1]}, {target_color[2]})")
        
        # Extraire la région sélectionnée
        selected_region = image[y:y+h, x:x+w]
        
        # Appliquer le zoom
        new_width = int(w * zoom_factor)
        new_height = int(h * zoom_factor)
        
        zoomed_sample = cv2.resize(selected_region, (new_width, new_height), 
                                 interpolation=cv2.INTER_LINEAR)
        
        print(f"[INFO] Echantillon zoome cree: {new_width}x{new_height}")
        
        # Uniformiser en préservant la couleur cible
        uniformized_sample = self._uniformize_zoomed_region(
            zoomed_sample, 
            target_color, 
            uniformization_strength
        )
        
        print(f"[INFO] Echantillon uniformise cree avec couleur preservee")
        
        return uniformized_sample
    
    def create_massive_fabric_texture(self, image, region_coords, target_color, 
                                    target_size=(5000, 5000), zoom_factor=1.3, 
                                    uniformization_strength=0.8):
        """
        Crée une texture de tissu MASSIVE avec multiplication énorme
        
        Args:
            image: Image d'entrée
            region_coords: Coordonnées de la région sélectionnée
            target_color: Couleur cible à préserver
            target_size: Taille MASSIVE de la texture (width, height)
            zoom_factor: Facteur de zoom
            uniformization_strength: Force d'uniformisation
        
        Returns:
            Texture MASSIVE uniformisée
        """
        print(f"[INFO] Creation de texture MASSIVE {target_size[0]}x{target_size[1]}...")
        print(f"[INFO] Multiplication enorme en cours...")
        
        # Créer l'échantillon personnalisé
        sample = self.create_custom_fabric_sample(
            image, 
            region_coords, 
            target_color, 
            zoom_factor, 
            uniformization_strength
        )
        
        if sample is None:
            print("[ERROR] Impossible de creer l'echantillon personnalise")
            return None
        
        # Créer la texture MASSIVE
        texture = self.create_fabric_texture_from_sample(
            sample, 
            target_size=target_size,
            preserve_characteristics=False  # Déjà uniformisé
        )
        
        if texture is not None:
            print(f"[INFO] Texture MASSIVE creee: {target_size[0]}x{target_size[1]}")
            print(f"[INFO] Multiplication de {sample.shape[1]}x{sample.shape[0]} vers {target_size[0]}x{target_size[1]}")
        
        return texture
    
    def create_multiple_massive_textures(self, image, region_coords, target_color, 
                                       sizes=[(2000, 2000), (3000, 3000), (4000, 4000), (5000, 5000)],
                                       zoom_factor=1.3, uniformization_strength=0.8):
        """
        Crée plusieurs textures MASSIVES de différentes tailles
        
        Args:
            image: Image d'entrée
            region_coords: Coordonnées de la région sélectionnée
            target_color: Couleur cible à préserver
            sizes: Liste des tailles MASSIVES à créer
            zoom_factor: Facteur de zoom
            uniformization_strength: Force d'uniformisation
        
        Returns:
            Dictionnaire avec les textures créées
        """
        print(f"[INFO] Creation de {len(sizes)} textures MASSIVES...")
        
        textures = {}
        
        for i, (width, height) in enumerate(sizes, 1):
            print(f"\n[INFO] Creation texture {i}/{len(sizes)}: {width}x{height}")
            
            try:
                texture = self.create_massive_fabric_texture(
                    image, 
                    region_coords, 
                    target_color, 
                    target_size=(width, height),
                    zoom_factor=zoom_factor,
                    uniformization_strength=uniformization_strength
                )
                
                if texture is not None:
                    textures[f"{width}x{height}"] = texture
                    print(f"[INFO] Texture {width}x{height} creee avec succes")
                else:
                    print(f"[ERROR] Echec creation texture {width}x{height}")
                    
            except Exception as e:
                print(f"[ERROR] Erreur lors de la creation de {width}x{height}: {e}")
        
        print(f"[INFO] {len(textures)} textures MASSIVES creees avec succes")
        return textures

def main():
    """
    Main function to demonstrate shadow removal capabilities
    """
    # Initialize shadow remover
    shadow_remover = AdvancedShadowRemover()
    
    # Load image (modify path as needed)
    # For Google Colab:
    # from google.colab import drive
    # drive.mount("/content/drive")
    # image_path = "/content/drive/MyDrive/koko/photo.jpeg"
    
    # For local use:
    image_path = "C:/Users/eloua/OneDrive/Images/photo-velours.png"  # Found image path
    
    print("[INFO] Loading image...")
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"[ERROR] Could not load image from {image_path}")
        print("Please ensure the image path is correct")
        return
    
    print(f"[INFO] Image loaded: {image.shape}")
    
    # Test automatic method selection
    print("\n[INFO] Testing automatic method selection...")
    auto_result = shadow_remover.remove_shadows(image, method='auto')
    
    # Assess quality of automatic result
    if auto_result is not None:
        metrics = shadow_remover.assess_quality(image, auto_result)
        print(f"[INFO] Automatic method quality score: {metrics['overall_quality']:.3f}")
        print(f"[INFO] Contrast improvement: {metrics['contrast_improvement']:.3f}")
        print(f"[INFO] Shadow reduction: {metrics['shadow_reduction']:.3f}")
        print(f"[INFO] Color preservation: {metrics['color_preservation']:.3f}")
    
    # Compare all methods
    print("\n[INFO] Comparing all methods...")
    results = shadow_remover.compare_methods(image)
    
    # Save results
    if results:
        print("\n[INFO] Saving results...")
        for method, result in results.items():
            output_path = f"shadow_removed_{method}.jpg"
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")
    
    # Demonstrate parameter optimization
    print("\n[INFO] Demonstrating parameter optimization...")
    best_result, best_params = shadow_remover.optimize_parameters(image, method='retinex')
    if best_result is not None:
        cv2.imwrite("optimized_result.jpg", best_result)
        print(f"Optimized result saved with parameters: {best_params}")

def demo_batch_processing():
    """
    Demonstrate batch processing capabilities
    """
    shadow_remover = AdvancedShadowRemover()
    
    # Example image paths (modify as needed)
    image_paths = [
        "photo1.jpeg",
        "photo2.jpeg", 
        "photo3.jpeg"
    ]
    
    # Process all images
    results = shadow_remover.batch_process(image_paths, method='auto')
    
    # Print summary
    print("\n[INFO] Batch processing summary:")
    for path, data in results.items():
        print(f"  {path}: Quality = {data['metrics']['overall_quality']:.3f}")

def demo_specific_methods():
    """
    Demonstrate specific shadow removal methods
    """
    shadow_remover = AdvancedShadowRemover()
    
    # Load image
    image = cv2.imread("photo.jpeg")
    if image is None:
        print("[ERROR] Could not load image")
        return
    
    methods_to_test = [
        'retinex',
        'texture_aware', 
        'adaptive_histogram',
        'multi_scale'
    ]
    
    print("[INFO] Testing specific methods...")
    for method in methods_to_test:
        print(f"\n[INFO] Testing {method}...")
        result = shadow_remover.remove_shadows(image, method=method)
        
        if result is not None:
            metrics = shadow_remover.assess_quality(image, result)
            print(f"  Quality score: {metrics['overall_quality']:.3f}")
            
            # Save result
            cv2.imwrite(f"demo_{method}.jpg", result)
            print(f"  Saved: demo_{method}.jpg")

if __name__ == "__main__":
    # Run main demonstration
    main()
    
    # Uncomment to run additional demonstrations:
    # demo_batch_processing()
    # demo_specific_methods()

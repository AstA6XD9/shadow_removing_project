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
            'multi_scale': self._multi_scale_shadow_removal
        }
    
    def remove_shadows(self, image, method='auto', **kwargs):
        """
        Main shadow removal function
        
        Args:
            image: Input image (BGR format)
            method: 'auto', 'retinex', 'gradient_domain', 'adaptive_histogram', 
                   'texture_aware', 'multi_scale'
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
            return 'texture_aware'
        elif shadow_intensity > 0.4:  # Strong shadows
            return 'multi_scale'
        else:
            return 'retinex'
    
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

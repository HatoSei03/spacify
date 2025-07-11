"""
Experimental Framework for View Selection Strategy Comparison
Research purpose: Compare different view selection methods for 3D scene reconstruction
"""

import cv2
import os
import numpy as np
from PIL import Image
import torch
import json
import time
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any
from torch.cuda.amp import autocast

# Import existing modules (same as pro_inpaint.py)
from utils.option import args
from models.egformer import EGDepthModel
from inpaint import inpaint
from evaluate.depest import depth_est
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import importlib
import argparse

class ViewSelectionExperiment:
    def __init__(self, base_models_setup=True):
        """Initialize experimental framework"""
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"experiments/exp_{self.experiment_id}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Setup models (same as original)
        if base_models_setup:
            self.setup_models()
        
        # Experiment configuration
        self.strategies = {
            'random': self.random_selection,
            'uniform': self.uniform_selection, 
            'quality_based': self.quality_based_selection,
            'diversity_based': self.diversity_based_selection,
            'quality_diversity': self.quality_diversity_selection,
            'phase_balanced': self.phase_balanced_selection,
            'early_focus': self.early_focus_selection,
            'late_focus': self.late_focus_selection,
            'edge_enhanced': self.edge_enhanced_selection
        }
        
        self.metrics_functions = {
            'psnr': self.calculate_psnr,
            'ssim': self.calculate_ssim,
            'lpips': self.calculate_lpips,
            'entropy': self.calculate_entropy,
            'sharpness': self.calculate_sharpness,
            'diversity_angular': self.calculate_angular_diversity,
            'diversity_feature': self.calculate_feature_diversity,
            'coverage': self.calculate_coverage
        }
        
        # Global parameters
        self.mov = 0.02
        self.step = 12
        self.min = 6
        self.num_views = 12
        
        # Initialize cache
        self._spherical_cache = None
        
    def setup_models(self):
        """Setup all required models (depth, inpainting, SR)"""
        # Same setup as pro_inpaint.py
        self.device = 'cuda:0'
        
        # Depth estimation model
        parser = argparse.ArgumentParser()
        parser.add_argument("--checkpoint_path", default='pretrained_models/EGformer_pretrained.pkl')
        parser.add_argument('--world_size', type=int, default=1)
        parser.add_argument('--rank', type=int, default=0)
        parser.add_argument('--multiprocessing_distributed', default=True)
        parser.add_argument('--dist-backend', type=str, default="nccl")
        parser.add_argument('--dist-url', type=str, default="tcp://127.0.0.1:7777")
        
        config = parser.parse_args()
        torch.distributed.init_process_group(backend=config.dist_backend, 
                                           init_method=config.dist_url,
                                           world_size=config.world_size, 
                                           rank=config.rank)
        
        self.depth_net = EGDepthModel(hybrid=False).to(self.device)
        self.depth_net = torch.nn.parallel.DistributedDataParallel(
            self.depth_net, device_ids=[0], find_unused_parameters=True)
        self.depth_net.load_state_dict(torch.load(config.checkpoint_path), strict=False)
        self.depth_net.eval()
        
        # Inpainting model
        net_inpa = importlib.import_module('model.' + args.model)
        self.inpaint_model = net_inpa.InpaintGenerator(args).to(self.device)
        args.pre_train = 'pretrained_models/G0185000.pt'
        self.inpaint_model.load_state_dict(torch.load(args.pre_train, map_location=self.device))
        self.inpaint_model.eval()
        
        # SR model
        sr_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=2)
        self.upsampler = RealESRGANer(scale=2, model_path='models/RealESRGAN_x2plus.pth',
                                     model=sr_model, tile=384, tile_pad=20, 
                                     pre_pad=20, half=False, device=self.device)

    def generate_all_views(self, ori_rgb: np.ndarray, ori_depth: np.ndarray) -> List[Dict]:
        """Generate all possible views with metadata for selection"""
        all_views = []
        directions = ['x', 'z', 'xz', '-xz']
        
        print("Generating all candidate views...")
        
        for dir_idx, direction in enumerate(directions):
            input_rgb = ori_rgb
            depth = ori_depth
            flag = 1
            
            for i in range(self.step):
                if i == self.min:
                    flag = -1
                    input_rgb = ori_rgb
                    depth = ori_depth
                
                # Generate view (using existing functions)
                mask, img, depth, mask_index = self.generate_optimized(
                    input_rgb, depth, flag, direction, (i == 0 or i == self.min)
                )
                
                inpainted = self.inpaint_image(mask, img)
                depth = self.depth_completion_optimized(inpainted)
                input_rgb = inpainted
                
                # Store view with metadata
                view_data = {
                    'image': inpainted,
                    'step': i,
                    'direction': direction,
                    'dir_idx': dir_idx,
                    'phase': 1 if i < self.min else 2,
                    'camera_position': self.get_camera_position(flag, direction),
                    'timestamp': time.time()
                }
                
                all_views.append(view_data)
        
        print(f"Generated {len(all_views)} candidate views")
        return all_views

    # =================== VIEW SELECTION STRATEGIES ===================
    
    def random_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Random selection baseline"""
        np.random.seed(42)  # For reproducibility
        selected_indices = np.random.choice(len(all_views), self.num_views, replace=False)
        return [all_views[i] for i in selected_indices]
    
    def uniform_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Uniform temporal sampling"""
        indices = np.linspace(0, len(all_views)-1, self.num_views, dtype=int)
        return [all_views[i] for i in indices]
    
    def quality_based_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Select based on image quality metrics"""
        quality_scores = []
        for view in all_views:
            score = (self.calculate_sharpness(view['image']) * 0.4 +
                    self.calculate_entropy(view['image']) * 0.3 +
                    self.calculate_edge_density(view['image']) * 0.3)
            quality_scores.append(score)
        
        top_indices = np.argsort(quality_scores)[-self.num_views:]
        return [all_views[i] for i in top_indices]
    
    def diversity_based_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Select for maximum diversity"""
        selected_indices = set()
        
        # Start with most central view
        central_idx = len(all_views) // 2
        selected_indices.add(central_idx)
        
        while len(selected_indices) < self.num_views and len(selected_indices) < len(all_views):
            best_idx = None
            max_diversity = -1
            
            for i in range(len(all_views)):
                if i in selected_indices:
                    continue
                    
                view = all_views[i]
                selected_views = [all_views[j] for j in selected_indices]
                diversity = self.calculate_diversity_score(view, selected_views)
                
                if diversity > max_diversity:
                    max_diversity = diversity
                    best_idx = i
            
            if best_idx is not None:
                selected_indices.add(best_idx)
        
        return [all_views[i] for i in selected_indices]
    
    def quality_diversity_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Combined quality + diversity approach"""
        selected_indices = set()
        
        # Calculate quality scores for all views
        quality_scores = [self.calculate_quality_score(view) for view in all_views]
        
        # Start with highest quality view
        best_quality_idx = np.argmax(quality_scores)
        selected_indices.add(best_quality_idx)
        
        while len(selected_indices) < self.num_views and len(selected_indices) < len(all_views):
            best_idx = None
            max_combined_score = -1
            
            for i in range(len(all_views)):
                if i in selected_indices:
                    continue
                    
                view = all_views[i]
                selected_views = [all_views[j] for j in selected_indices]
                
                quality = quality_scores[i]
                diversity = self.calculate_diversity_score(view, selected_views)
                combined_score = quality * 0.6 + diversity * 0.4
                
                if combined_score > max_combined_score:
                    max_combined_score = combined_score
                    best_idx = i
            
            if best_idx is not None:
                selected_indices.add(best_idx)
        
        return [all_views[i] for i in selected_indices]
    
    def phase_balanced_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Ensure balanced representation from both phases"""
        phase1_views = [v for v in all_views if v['phase'] == 1]
        phase2_views = [v for v in all_views if v['phase'] == 2]
        
        # Select 6 from each phase
        phase1_selected = self.quality_based_selection(phase1_views)[:6]
        phase2_selected = self.quality_based_selection(phase2_views)[:6]
        
        return phase1_selected + phase2_selected
    
    def early_focus_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Focus on early steps (less inpainting artifacts)"""
        early_views = [v for v in all_views if v['step'] <= 4]
        return self.quality_based_selection(early_views)
    
    def late_focus_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Focus on late steps (more diverse viewpoints)"""
        late_views = [v for v in all_views if v['step'] >= 8]
        return self.quality_based_selection(late_views)
    
    def edge_enhanced_selection(self, all_views: List[Dict]) -> List[Dict]:
        """Select views with rich edge information"""
        edge_scores = []
        for view in all_views:
            edge_score = self.calculate_edge_density(view['image'])
            edge_scores.append(edge_score)
        
        top_indices = np.argsort(edge_scores)[-self.num_views:]
        return [all_views[i] for i in top_indices]

    # =================== QUALITY METRICS ===================
    
    def calculate_psnr(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate PSNR between two images"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two images"""
        if len(img1.shape) == 3:
            return np.mean([ssim(img1[:,:,i], img2[:,:,i], data_range=255) 
                           for i in range(3)])
        return ssim(img1, img2, data_range=255)
    
    def calculate_lpips(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate LPIPS (placeholder - requires lpips library)"""
        # Simplified version - replace with actual LPIPS calculation
        return np.mean(np.abs(img1.astype(float) - img2.astype(float))) / 255.0
    
    def calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate image entropy"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return shannon_entropy(gray)
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_edge_density(self, image: np.ndarray) -> float:
        """Calculate edge density using Canny edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size

    # =================== DIVERSITY METRICS ===================
    
    def calculate_angular_diversity(self, views: List[Dict]) -> float:
        """Calculate angular diversity between camera positions"""
        positions = np.array([v['camera_position'] for v in views])
        if len(positions) < 2:
            return 0.0
        
        # Calculate pairwise angular distances
        distances = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                angle = np.arccos(np.clip(np.dot(positions[i], positions[j]), -1, 1))
                distances.append(angle)
        
        return np.mean(distances)
    
    def calculate_feature_diversity(self, views: List[Dict]) -> float:
        """Calculate feature diversity using image histograms"""
        features = []
        for view in views:
            # Calculate color histogram as feature
            hist = cv2.calcHist([view['image']], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
            features.append(hist.flatten())
        
        if len(features) < 2:
            return 0.0
        
        # Calculate pairwise cosine distances
        similarity_matrix = cosine_similarity(features)
        diversity = 1 - np.mean(similarity_matrix)
        return diversity
    
    def calculate_coverage(self, views: List[Dict]) -> float:
        """Calculate spatial coverage"""
        directions = set(v['direction'] for v in views)
        steps = set(v['step'] for v in views)
        phases = set(v['phase'] for v in views)
        
        # Normalized coverage score
        dir_coverage = len(directions) / 4.0  # 4 total directions
        step_coverage = len(steps) / float(self.step)  # total steps
        phase_coverage = len(phases) / 2.0  # 2 phases
        
        return (dir_coverage + step_coverage + phase_coverage) / 3.0

    # =================== EXPERIMENTAL RUNNER ===================
    
    def run_experiment(self, input_image_path: str, num_runs: int = 3) -> Dict[str, Any]:
        """Run complete experiment comparing all strategies"""
        print(f"Starting experiment {self.experiment_id}")
        print(f"Input image: {input_image_path}")
        print(f"Number of runs per strategy: {num_runs}")
        
        # Load input image and generate depth
        rgb = np.array(Image.open(input_image_path).convert('RGB'))
        depth = depth_est(rgb, self.depth_net)
        
        # Generate all candidate views
        all_views = self.generate_all_views(rgb, depth)
        
        # Run experiments for each strategy
        results = {}
        
        for strategy_name, strategy_func in self.strategies.items():
            print(f"\nTesting strategy: {strategy_name}")
            strategy_results = []
            
            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}")
                
                # Select views using this strategy
                selected_views = strategy_func(all_views)
                
                # Save selected views
                run_dir = f"{self.results_dir}/{strategy_name}/run_{run}"
                os.makedirs(run_dir, exist_ok=True)
                
                for i, view in enumerate(selected_views):
                    Image.fromarray(view['image']).save(f"{run_dir}/view_{i:03d}.jpg")
                
                # Calculate detailed metrics
                detailed_metrics = self.calculate_all_metrics(selected_views, rgb)
                
                # Create flattened metrics for compatibility with existing analysis
                run_metrics = detailed_metrics['aggregate_metrics'].copy()
                run_metrics['strategy'] = strategy_name
                run_metrics['run'] = run
                run_metrics['num_views'] = len(selected_views)
                run_metrics['detailed_data'] = detailed_metrics  # Store full details
                
                strategy_results.append(run_metrics)
            
            results[strategy_name] = strategy_results
        
        # Save raw results
        with open(f"{self.results_dir}/raw_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Analyze and visualize results
        analysis = self.analyze_results(results)
        self.visualize_results(analysis)
        
        print(f"\nExperiment complete! Results saved to {self.results_dir}")
        return analysis

    def calculate_all_metrics(self, selected_views: List[Dict], reference_rgb: np.ndarray) -> Dict[str, Any]:
        """Calculate detailed metrics for each selected view and aggregates"""
        detailed_metrics = {
            'individual_views': [],
            'aggregate_metrics': {},
            'view_metadata': []
        }
        
        # Calculate metrics for each individual view
        psnr_scores = []
        ssim_scores = []
        sharpness_scores = []
        entropy_scores = []
        edge_density_scores = []
        
        for i, view in enumerate(selected_views):
            view_metrics = {
                'view_id': i,
                'step': view['step'],
                'direction': view['direction'],
                'phase': view['phase'],
                'dir_idx': view['dir_idx']
            }
            
            # Quality metrics for this view
            psnr = self.calculate_psnr(view['image'], reference_rgb)
            psnr = psnr if psnr != float('inf') else 50.0
            
            ssim_val = self.calculate_ssim(view['image'], reference_rgb)
            sharpness = self.calculate_sharpness(view['image'])
            entropy = self.calculate_entropy(view['image'])
            edge_density = self.calculate_edge_density(view['image'])
            
            # Store individual metrics
            view_metrics.update({
                'psnr': psnr,
                'ssim': ssim_val,
                'sharpness': sharpness,
                'entropy': entropy,
                'edge_density': edge_density,
                'quality_score': self.calculate_quality_score(view)
            })
            
            detailed_metrics['individual_views'].append(view_metrics)
            detailed_metrics['view_metadata'].append({
                'step': view['step'],
                'direction': view['direction'],
                'phase': view['phase'],
                'camera_position': view['camera_position'].tolist()
            })
            
            # Collect for aggregates
            psnr_scores.append(psnr)
            ssim_scores.append(ssim_val)
            sharpness_scores.append(sharpness)
            entropy_scores.append(entropy)
            edge_density_scores.append(edge_density)
        
        # Aggregate metrics
        detailed_metrics['aggregate_metrics'] = {
            'psnr_mean': np.mean(psnr_scores),
            'psnr_std': np.std(psnr_scores),
            'psnr_min': np.min(psnr_scores),
            'psnr_max': np.max(psnr_scores),
            'psnr_median': np.median(psnr_scores),
            
            'ssim_mean': np.mean(ssim_scores),
            'ssim_std': np.std(ssim_scores),
            'ssim_min': np.min(ssim_scores),
            'ssim_max': np.max(ssim_scores),
            'ssim_median': np.median(ssim_scores),
            
            'sharpness_mean': np.mean(sharpness_scores),
            'sharpness_std': np.std(sharpness_scores),
            'sharpness_min': np.min(sharpness_scores),
            'sharpness_max': np.max(sharpness_scores),
            
            'entropy_mean': np.mean(entropy_scores),
            'entropy_std': np.std(entropy_scores),
            'entropy_min': np.min(entropy_scores),
            'entropy_max': np.max(entropy_scores),
            
            'edge_density_mean': np.mean(edge_density_scores),
            'edge_density_std': np.std(edge_density_scores),
            'edge_density_min': np.min(edge_density_scores),
            'edge_density_max': np.max(edge_density_scores),
            
            # Diversity metrics (aggregate by nature)
            'angular_diversity': self.calculate_angular_diversity(selected_views),
            'feature_diversity': self.calculate_feature_diversity(selected_views),
            'coverage': self.calculate_coverage(selected_views),
            
            # Temporal distribution
            'step_std': np.std([v['step'] for v in selected_views]),
            'step_range': max([v['step'] for v in selected_views]) - min([v['step'] for v in selected_views]),
            'step_distribution': [v['step'] for v in selected_views],
            'direction_distribution': [v['direction'] for v in selected_views],
            'phase_distribution': [v['phase'] for v in selected_views]
        }
        
        return detailed_metrics

    def analyze_results(self, results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Statistical analysis of experimental results"""
        # Convert to DataFrame for analysis
        all_data = []
        for strategy, runs in results.items():
            all_data.extend(runs)
        
        df = pd.DataFrame(all_data)
        
        # Calculate summary statistics
        summary = df.groupby('strategy').agg({
            'psnr_mean': ['mean', 'std'],
            'ssim_mean': ['mean', 'std'],
            'sharpness_mean': ['mean', 'std'],
            'entropy_mean': ['mean', 'std'],
            'angular_diversity': ['mean', 'std'],
            'feature_diversity': ['mean', 'std'],
            'coverage': ['mean', 'std']
        }).round(4)
        
        # Statistical significance testing
        strategies = list(results.keys())
        significance_tests = {}
        
        for metric in ['psnr_mean', 'ssim_mean', 'angular_diversity', 'coverage']:
            significance_tests[metric] = {}
            
            for i, strategy1 in enumerate(strategies):
                for strategy2 in strategies[i+1:]:
                    data1 = [run[metric] for run in results[strategy1]]
                    data2 = [run[metric] for run in results[strategy2]]
                    
                    # Perform t-test
                    statistic, p_value = stats.ttest_ind(data1, data2)
                    significance_tests[metric][f"{strategy1}_vs_{strategy2}"] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        # Rankings
        rankings = {}
        for metric in ['psnr_mean', 'ssim_mean', 'angular_diversity', 'coverage']:
            metric_means = df.groupby('strategy')[metric].mean().sort_values(ascending=False)
            rankings[metric] = metric_means.to_dict()
        
        analysis = {
            'summary_statistics': summary.to_dict(),
            'significance_tests': significance_tests,
            'rankings': rankings,
            'raw_dataframe': df
        }
        
        # Save analysis
        with open(f"{self.results_dir}/analysis.json", 'w') as f:
            json.dump({k: v for k, v in analysis.items() if k != 'raw_dataframe'}, 
                     f, indent=2, default=str)
        
        df.to_csv(f"{self.results_dir}/results.csv", index=False)
        
        return analysis

    def visualize_results(self, analysis: Dict[str, Any]):
        """Generate visualizations for the results"""
        df = analysis['raw_dataframe']
        
        # Set up plotting style  
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                pass  # Use default style
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'View Selection Strategy Comparison - Experiment {self.experiment_id}', fontsize=16)
        
        metrics_to_plot = [
            ('psnr_mean', 'PSNR (Higher is Better)'),
            ('ssim_mean', 'SSIM (Higher is Better)'),
            ('angular_diversity', 'Angular Diversity (Higher is Better)'),
            ('feature_diversity', 'Feature Diversity (Higher is Better)'),
            ('coverage', 'Coverage (Higher is Better)'),
            ('sharpness_mean', 'Sharpness (Higher is Better)')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_plot):
            ax = axes[idx // 3, idx % 3]
            
            # Box plot
            df.boxplot(column=metric, by='strategy', ax=ax)
            ax.set_title(title)
            ax.set_xlabel('Strategy')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comparison_plots.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create radar chart for overall comparison
        self.create_radar_chart(analysis)
        
        # Create detailed ranking plot
        self.create_ranking_plot(analysis)

    def create_radar_chart(self, analysis: Dict[str, Any]):
        """Create radar chart comparing strategies"""
        df = analysis['raw_dataframe']
        strategies = df['strategy'].unique()
        
        # Normalize metrics for radar chart (0-1 scale)
        metrics = ['psnr_mean', 'ssim_mean', 'angular_diversity', 'feature_diversity', 'coverage']
        
        # Calculate normalized means
        normalized_data = {}
        for strategy in strategies:
            strategy_data = df[df['strategy'] == strategy]
            normalized_values = []
            
            for metric in metrics:
                value = strategy_data[metric].mean()
                # Normalize to 0-1 scale
                metric_values = df[metric].values
                normalized = (value - metric_values.min()) / (metric_values.max() - metric_values.min())
                normalized_values.append(normalized)
            
            normalized_data[strategy] = normalized_values
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for strategy, values in normalized_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=strategy)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Strategy Comparison - Normalized Metrics', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.savefig(f"{self.results_dir}/radar_chart.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    def create_ranking_plot(self, analysis: Dict[str, Any]):
        """Create ranking visualization"""
        rankings = analysis['rankings']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Strategy Rankings by Metric', fontsize=16)
        
        metrics_to_rank = [
            ('psnr_mean', 'PSNR Ranking'),
            ('ssim_mean', 'SSIM Ranking'),
            ('angular_diversity', 'Angular Diversity Ranking'),
            ('coverage', 'Coverage Ranking')
        ]
        
        for idx, (metric, title) in enumerate(metrics_to_rank):
            ax = axes[idx // 2, idx % 2]
            
            ranking_data = rankings[metric]
            strategies = list(ranking_data.keys())
            values = list(ranking_data.values())
            
            # Create horizontal bar chart
            bars = ax.barh(strategies, values)
            ax.set_title(title)
            ax.set_xlabel('Score')
            
            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/rankings.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    def create_detailed_visualization(self, detailed_metrics: Dict[str, Any], strategy_name: str, output_dir: str):
        """Create detailed visualizations for individual view metrics"""
        individual_views = detailed_metrics['individual_views']
        aggregate_metrics = detailed_metrics['aggregate_metrics']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Detailed View Analysis - {strategy_name.upper()}', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        view_ids = [v['view_id'] for v in individual_views]
        steps = [v['step'] for v in individual_views]
        directions = [v['direction'] for v in individual_views]
        psnr_values = [v['psnr'] for v in individual_views]
        ssim_values = [v['ssim'] for v in individual_views]
        sharpness_values = [v['sharpness'] for v in individual_views]
        entropy_values = [v['entropy'] for v in individual_views]
        quality_values = [v['quality_score'] for v in individual_views]
        
        # 1. PSNR per view
        ax1 = axes[0, 0]
        bars1 = ax1.bar(view_ids, psnr_values, color='skyblue', alpha=0.7)
        ax1.axhline(y=aggregate_metrics['psnr_mean'], color='red', linestyle='--', label=f'Mean: {aggregate_metrics["psnr_mean"]:.2f}')
        ax1.set_title('PSNR per View')
        ax1.set_xlabel('View ID')
        ax1.set_ylabel('PSNR (dB)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars1, psnr_values):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, 
                    f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 2. SSIM per view
        ax2 = axes[0, 1]
        bars2 = ax2.bar(view_ids, ssim_values, color='lightgreen', alpha=0.7)
        ax2.axhline(y=aggregate_metrics['ssim_mean'], color='red', linestyle='--', label=f'Mean: {aggregate_metrics["ssim_mean"]:.3f}')
        ax2.set_title('SSIM per View')
        ax2.set_xlabel('View ID')
        ax2.set_ylabel('SSIM')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, ssim_values):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Quality Score per view
        ax3 = axes[0, 2]
        bars3 = ax3.bar(view_ids, quality_values, color='orange', alpha=0.7)
        ax3.axhline(y=np.mean(quality_values), color='red', linestyle='--', label=f'Mean: {np.mean(quality_values):.3f}')
        ax3.set_title('Quality Score per View')
        ax3.set_xlabel('View ID')
        ax3.set_ylabel('Quality Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Step distribution
        ax4 = axes[1, 0]
        step_counts = {}
        for step in steps:
            step_counts[step] = step_counts.get(step, 0) + 1
        ax4.bar(step_counts.keys(), step_counts.values(), color='purple', alpha=0.7)
        ax4.set_title('Step Distribution')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Count')
        ax4.grid(True, alpha=0.3)
        
        # 5. Direction distribution
        ax5 = axes[1, 1]
        direction_counts = {}
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        ax5.bar(direction_counts.keys(), direction_counts.values(), color='teal', alpha=0.7)
        ax5.set_title('Direction Distribution')
        ax5.set_xlabel('Direction')
        ax5.set_ylabel('Count')
        ax5.grid(True, alpha=0.3)
        
        # 6. Metrics correlation scatter
        ax6 = axes[1, 2]
        scatter = ax6.scatter(psnr_values, ssim_values, c=quality_values, 
                             cmap='viridis', alpha=0.7, s=100)
        ax6.set_title('PSNR vs SSIM (colored by Quality)')
        ax6.set_xlabel('PSNR')
        ax6.set_ylabel('SSIM')
        plt.colorbar(scatter, ax=ax6, label='Quality Score')
        ax6.grid(True, alpha=0.3)
        
        # Add view ID annotations
        for i, (x, y) in enumerate(zip(psnr_values, ssim_values)):
            ax6.annotate(f'V{i}', (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/detailed_analysis.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create a summary table image
        self.create_metrics_table(individual_views, aggregate_metrics, strategy_name, output_dir)
    
    def create_metrics_table(self, individual_views: List[Dict], aggregate_metrics: Dict, strategy_name: str, output_dir: str):
        """Create a detailed metrics table as image"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Metrics Summary - {strategy_name.upper()}', fontsize=16, fontweight='bold')
        
        # Individual metrics table
        ax1.axis('tight')
        ax1.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['View', 'Step', 'Dir', 'PSNR', 'SSIM', 'Sharpness', 'Entropy', 'Quality']
        
        for view in individual_views:
            row = [
                view['view_id'],
                view['step'],
                view['direction'],
                f"{view['psnr']:.2f}",
                f"{view['ssim']:.3f}",
                f"{view['sharpness']:.1f}",
                f"{view['entropy']:.2f}",
                f"{view['quality_score']:.3f}"
            ]
            table_data.append(row)
        
        table1 = ax1.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1.2, 1.5)
        
        # Color code the header
        for i in range(len(headers)):
            table1[(0, i)].set_facecolor('#4CAF50')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('Individual View Metrics', pad=20, fontweight='bold')
        
        # Aggregate statistics table
        ax2.axis('tight')
        ax2.axis('off')
        
        agg_data = [
            ['PSNR', f"{aggregate_metrics['psnr_mean']:.2f}", f"{aggregate_metrics['psnr_std']:.2f}", 
             f"{aggregate_metrics['psnr_min']:.2f}", f"{aggregate_metrics['psnr_max']:.2f}"],
            ['SSIM', f"{aggregate_metrics['ssim_mean']:.3f}", f"{aggregate_metrics['ssim_std']:.3f}", 
             f"{aggregate_metrics['ssim_min']:.3f}", f"{aggregate_metrics['ssim_max']:.3f}"],
            ['Sharpness', f"{aggregate_metrics['sharpness_mean']:.1f}", f"{aggregate_metrics['sharpness_std']:.1f}", 
             f"{aggregate_metrics['sharpness_min']:.1f}", f"{aggregate_metrics['sharpness_max']:.1f}"],
            ['Entropy', f"{aggregate_metrics['entropy_mean']:.2f}", f"{aggregate_metrics['entropy_std']:.2f}", 
             f"{aggregate_metrics['entropy_min']:.2f}", f"{aggregate_metrics['entropy_max']:.2f}"],
            ['Angular Div', f"{aggregate_metrics['angular_diversity']:.3f}", '-', '-', '-'],
            ['Coverage', f"{aggregate_metrics['coverage']:.3f}", '-', '-', '-']
        ]
        
        agg_headers = ['Metric', 'Mean', 'Std', 'Min', 'Max']
        
        table2 = ax2.table(cellText=agg_data, colLabels=agg_headers, loc='center', cellLoc='center')
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.8)
        
        # Color code the header
        for i in range(len(agg_headers)):
            table2[(0, i)].set_facecolor('#2196F3')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        ax2.set_title('Aggregate Statistics', pad=20, fontweight='bold')
        
        plt.savefig(f"{output_dir}/metrics_table.jpg", dpi=300, bbox_inches='tight')
        plt.close()

    def create_strategy_comparison_report(self, results_dir: str):
        """Create comprehensive comparison report across all strategies"""
        print("\n📊 Creating strategy comparison report...")
        
        # Read all strategy results
        strategy_results = {}
        for strategy_name in ['random', 'quality_based', 'quality_diversity']:
            strategy_dir = f"{results_dir}/{strategy_name}"
            if os.path.exists(f"{strategy_dir}/detailed_metrics.json"):
                with open(f"{strategy_dir}/detailed_metrics.json", 'r') as f:
                    strategy_results[strategy_name] = json.load(f)
        
        if not strategy_results:
            print("  No strategy results found for comparison")
            return
        
        # Create comparison DataFrame
        comparison_data = []
        
        for strategy_name, metrics in strategy_results.items():
            # Add aggregate row
            agg = metrics['aggregate_metrics']
            comparison_data.append({
                'strategy': strategy_name,
                'view_type': 'aggregate',
                'view_id': 'ALL',
                'psnr': agg['psnr_mean'],
                'ssim': agg['ssim_mean'], 
                'sharpness': agg['sharpness_mean'],
                'entropy': agg['entropy_mean'],
                'angular_diversity': agg['angular_diversity'],
                'coverage': agg['coverage'],
                'psnr_std': agg['psnr_std'],
                'ssim_std': agg['ssim_std']
            })
            
            # Add individual view rows
            for view in metrics['individual_views']:
                comparison_data.append({
                    'strategy': strategy_name,
                    'view_type': 'individual',
                    'view_id': view['view_id'],
                    'step': view['step'],
                    'direction': view['direction'],
                    'phase': view['phase'],
                    'psnr': view['psnr'],
                    'ssim': view['ssim'],
                    'sharpness': view['sharpness'],
                    'entropy': view['entropy'],
                    'quality_score': view['quality_score'],
                    'angular_diversity': None,  # Individual views don't have this
                    'coverage': None
                })
        
        # Save comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv(f"{results_dir}/strategy_comparison.csv", index=False)
        
        # Create summary statistics
        summary_stats = {}
        for strategy in strategy_results.keys():
            strategy_data = comparison_df[
                (comparison_df['strategy'] == strategy) & 
                (comparison_df['view_type'] == 'individual')
            ]
            
            if len(strategy_data) > 0:
                summary_stats[strategy] = {
                    'psnr_mean': strategy_data['psnr'].mean(),
                    'psnr_std': strategy_data['psnr'].std(),
                    'ssim_mean': strategy_data['ssim'].mean(),
                    'ssim_std': strategy_data['ssim'].std(),
                    'sharpness_mean': strategy_data['sharpness'].mean(),
                    'entropy_mean': strategy_data['entropy'].mean(),
                    'quality_mean': strategy_data['quality_score'].mean(),
                    'step_diversity': strategy_data['step'].std(),
                    'view_count': len(strategy_data)
                }
        
        # Save summary
        summary_df = pd.DataFrame(summary_stats).T
        summary_df.to_csv(f"{results_dir}/strategy_summary.csv")
        
        # Create visual comparison
        self.create_cross_strategy_visualization(comparison_df, results_dir)
        
        print(f"  ✅ Comparison report saved:")
        print(f"     📄 {results_dir}/strategy_comparison.csv")
        print(f"     📄 {results_dir}/strategy_summary.csv")
        print(f"     📊 {results_dir}/cross_strategy_comparison.jpg")
    
    def create_cross_strategy_visualization(self, comparison_df: pd.DataFrame, results_dir: str):
        """Create cross-strategy comparison visualizations"""
        individual_data = comparison_df[comparison_df['view_type'] == 'individual']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Cross-Strategy Comparison', fontsize=16, fontweight='bold')
        
        # 1. PSNR distribution by strategy
        ax1 = axes[0, 0]
        strategies = individual_data['strategy'].unique()
        psnr_data = [individual_data[individual_data['strategy'] == s]['psnr'].values for s in strategies]
        ax1.boxplot(psnr_data, labels=strategies)
        ax1.set_title('PSNR Distribution by Strategy')
        ax1.set_ylabel('PSNR (dB)')
        ax1.grid(True, alpha=0.3)
        
        # 2. SSIM distribution by strategy  
        ax2 = axes[0, 1]
        ssim_data = [individual_data[individual_data['strategy'] == s]['ssim'].values for s in strategies]
        ax2.boxplot(ssim_data, labels=strategies)
        ax2.set_title('SSIM Distribution by Strategy')
        ax2.set_ylabel('SSIM')
        ax2.grid(True, alpha=0.3)
        
        # 3. Quality score comparison
        ax3 = axes[1, 0] 
        quality_means = [individual_data[individual_data['strategy'] == s]['quality_score'].mean() for s in strategies]
        bars = ax3.bar(strategies, quality_means, color=['skyblue', 'lightgreen', 'orange'])
        ax3.set_title('Average Quality Score by Strategy')
        ax3.set_ylabel('Quality Score')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, quality_means):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        
        # 4. Step distribution by strategy
        ax4 = axes[1, 1]
        for strategy in strategies:
            strategy_data = individual_data[individual_data['strategy'] == strategy]
            ax4.hist(strategy_data['step'], alpha=0.6, label=strategy, bins=range(0, 13))
        ax4.set_title('Step Distribution by Strategy')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Count')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/cross_strategy_comparison.jpg", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  📊 Cross-strategy visualization created")

    # =================== HELPER FUNCTIONS ===================
    
    def get_camera_position(self, flag: int, direction: str) -> np.ndarray:
        """Get camera position based on flag and direction"""
        cam_pos_map = {
            'x': np.array([self.mov * flag, 0, 0]),
            'z': np.array([0, 0, self.mov * flag]),
            'xz': np.array([self.mov * flag, 0, self.mov * flag]),
            '-xz': np.array([self.mov * flag, 0, -self.mov * flag])
        }
        return cam_pos_map[direction]
    
    def calculate_quality_score(self, view: Dict) -> float:
        """Calculate overall quality score for a view"""
        image = view['image']
        sharpness = self.calculate_sharpness(image) / 1000.0  # Normalize
        entropy = self.calculate_entropy(image) / 8.0  # Normalize
        edge_density = self.calculate_edge_density(image)
        
        return sharpness * 0.4 + entropy * 0.3 + edge_density * 0.3
    
    def calculate_diversity_score(self, view: Dict, selected_views: List[Dict]) -> float:
        """Calculate diversity score relative to already selected views"""
        if not selected_views:
            return 1.0
        
        diversity_score = 1.0
        
        for selected in selected_views:
            # Temporal diversity
            step_diff = abs(view['step'] - selected['step'])
            temporal_penalty = max(0, 1 - step_diff / 6.0)  # Penalty for similar steps
            
            # Direction diversity
            direction_penalty = 0.5 if view['direction'] == selected['direction'] else 0
            
            # Phase diversity
            phase_penalty = 0.3 if view['phase'] == selected['phase'] else 0
            
            diversity_score *= (1 - temporal_penalty - direction_penalty - phase_penalty)
        
        return max(diversity_score, 0.1)  # Minimum diversity score

    # =================== CORE GENERATION FUNCTIONS ===================
    
    def translate(self, crd, rgb, d, cam=[]):
        """Translate coordinates for view synthesis"""
        H, W = rgb.shape[0], rgb.shape[1]
        d = np.where(d == 0, -1, d)

        tmp_coord = crd - cam
        new_d = np.sqrt(np.sum(np.square(tmp_coord), axis=2))
        new_coord = tmp_coord / new_d.reshape(H, W, 1)

        new_depth = np.zeros(new_d.shape)

        [x, y, z] = new_coord[..., 0], new_coord[..., 1], new_coord[..., 2]

        idx = np.where(new_d > 0)

        theta = np.zeros(y.shape)
        phi = np.zeros(y.shape)
        x1 = np.zeros(z.shape)
        y1 = np.zeros(z.shape)
        
        # chuyển đổi tọa độ 3D sang tọa độ góc (spherical coordinates)
        theta[idx] = np.arctan2(y[idx], np.sqrt(np.square(x[idx]) + np.square(z[idx])))
        phi[idx] = np.arctan2(-z[idx], x[idx])

        # chuyển đổi tọa độ góc sang tọa độ pixel trong ảnh panorama
        x1[idx] = (0.5 - theta[idx] / np.pi) * H
        y1[idx] = (0.5 - phi[idx] / (2 * np.pi)) * W

        x, y = np.floor(x1).astype('int'), np.floor(y1).astype('int')

        img = np.zeros(rgb.shape)

        mask = (new_d > 0) & (H > x) & (x > 0) & (W > y) & (y > 0)

        x = x[mask]
        y = y[mask]
        new_d = new_d[mask]
        rgb = rgb[mask]

        # sắp xếp lại các điểm dựa trên độ sâu giảm dần, để xử lý che khuất
        reorder = np.argsort(-new_d)
        x = x[reorder]
        y = y[reorder]
        new_d = new_d[reorder]
        rgb = rgb[reorder]

        # Assign
        new_depth[x, y] = new_d
        img[x, y] = rgb

        mask = (new_depth != 0).astype(int)
        mask_index = np.argwhere(mask == 0)
        
        return img, new_depth.reshape(H, W, 1), tmp_coord, cam.reshape(1, 1, 3), mask, mask_index
    
    def get_spherical_coords(self, H=512, W=1024):
        """Cache spherical coordinates để tránh tính toán lặp lại"""
        if not hasattr(self, '_spherical_cache') or self._spherical_cache is None:
            _y = np.repeat(np.array(range(W)).reshape(1, W), H, axis=0)
            _x = np.repeat(np.array(range(H)).reshape(1, H), W, axis=0).T
            
            _theta = (1 - 2 * (_x) / H) * np.pi / 2
            _phi = 2 * np.pi * (0.5 - (_y) / W)
            
            axis0 = (np.cos(_theta) * np.cos(_phi)).reshape(H, W, 1)
            axis1 = np.sin(_theta).reshape(H, W, 1)
            axis2 = (-np.cos(_theta) * np.sin(_phi)).reshape(H, W, 1)
            
            self._spherical_cache = np.concatenate((axis0, axis1, axis2), axis=2)
        
        return self._spherical_cache

    def generate_optimized(self, input_rgb, input_depth, flag, dir, first):
        """Phiên bản tối ưu của hàm generate"""
        H, W = 512, 1024
        
        rgb = input_rgb
        d = input_depth
        
        d_max = np.max(d)
        d = d / d_max
        d = d.reshape(rgb.shape[0], rgb.shape[1], 1)
        d = np.where(d == 0, 1, d)
        
        # Sử dụng cache thay vì tính toán lại
        coord = self.get_spherical_coords(H, W) * d
        
        # Tối ưu cam_pos calculation
        cam_pos_map = {
            'x': np.array([self.mov * flag, 0, 0]),
            'z': np.array([0, 0, self.mov * flag]),
            'xz': np.array([self.mov * flag, 0, self.mov * flag]),
            '-xz': np.array([self.mov * flag, 0, -self.mov * flag])
        }
        cam_pos = cam_pos_map[dir]
        
        img1, d1, _, _, mask1, mask_index = self.translate(coord, rgb, d, cam_pos)
        d1 = np.squeeze(d1, axis=-1)
        d1 = np.stack((d1, d1, d1), axis=-1)
        
        mask = np.uint8(mask1 * 255)
        img = np.uint8(img1)
        img[mask == 0] = 255
        mask = cv2.bitwise_not(mask)
        
        return mask, img, d1[:, :, 0], mask_index
    
    def inpaint_image(self, mask, img):
        """Use the inpaint function from pro_inpaint.py"""
        return inpaint(mask, img, self.inpaint_model, self.upsampler)
    
    def depth_completion_optimized(self, input_rgb):
        """Use the optimized depth completion from pro_inpaint.py"""
        with autocast():
            return depth_est(input_rgb, self.depth_net)

def run_quick_test():
    """Run a quick test with minimal strategies"""
    print("Starting View Selection Experiment")
    
    # Initialize experiment with minimal setup
    experiment = ViewSelectionExperiment()
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"experiments/quick_{experiment_id}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Test input
    input_image = '/content/SD-T2I-360PanoImage/result.png'
    print(f"Quick experiment {experiment_id}")
    print(f"Input: {input_image}")
    
    # Load input image and generate depth
    rgb = np.array(Image.open(input_image).convert('RGB'))
    depth = depth_est(rgb, experiment.depth_net)
    
    # Generate all candidate views
    print("Generating all candidate views...")
    all_views = experiment.generate_all_views(rgb, depth)
    
    # Test only 3 strategies
    strategies_to_test = {
        'random': experiment.random_selection,
        'quality_based': experiment.quality_based_selection,
        'quality_diversity': experiment.quality_diversity_selection
    }
    
    for strategy_name, strategy_func in strategies_to_test.items():
        print(f"\nTesting: {strategy_name}")
        print("=" * 40)
        try:
            selected_views = strategy_func(all_views)
            print(f"  Generated {len(selected_views)} views")
            
            # Calculate detailed metrics
            detailed_metrics = experiment.calculate_all_metrics(selected_views, rgb)
            
            # Print individual view metrics
            print(f"\n  📊 INDIVIDUAL VIEW METRICS:")
            print(f"  {'View':<4} {'Step':<4} {'Dir':<4} {'PSNR':<6} {'SSIM':<6} {'Sharp':<8} {'Entropy':<7} {'Quality':<7}")
            print(f"  {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*7}")
            
            for view_data in detailed_metrics['individual_views']:
                print(f"  {view_data['view_id']:<4} "
                     f"{view_data['step']:<4} "
                     f"{view_data['direction']:<4} "
                     f"{view_data['psnr']:<6.2f} "
                     f"{view_data['ssim']:<6.3f} "
                     f"{view_data['sharpness']:<8.1f} "
                     f"{view_data['entropy']:<7.2f} "
                     f"{view_data['quality_score']:<7.3f}")
            
            # Print aggregate statistics
            agg = detailed_metrics['aggregate_metrics']
            print(f"\n  📈 AGGREGATE STATISTICS:")
            print(f"     PSNR: μ={agg['psnr_mean']:.2f} σ={agg['psnr_std']:.2f} [{agg['psnr_min']:.1f}-{agg['psnr_max']:.1f}]")
            print(f"     SSIM: μ={agg['ssim_mean']:.3f} σ={agg['ssim_std']:.3f} [{agg['ssim_min']:.3f}-{agg['ssim_max']:.3f}]")
            print(f"     Sharp: μ={agg['sharpness_mean']:.1f} σ={agg['sharpness_std']:.1f}")
            print(f"     Entropy: μ={agg['entropy_mean']:.2f} σ={agg['entropy_std']:.2f}")
            print(f"     Angular Diversity: {agg['angular_diversity']:.3f}")
            print(f"     Coverage: {agg['coverage']:.3f}")
            
            # Print selection distribution
            print(f"  📍 SELECTION DISTRIBUTION:")
            print(f"     Steps: {agg['step_distribution']}")
            print(f"     Directions: {agg['direction_distribution']}")
            print(f"     Phases: {agg['phase_distribution']}")
            
            # Save detailed results
            strategy_dir = f"{results_dir}/{strategy_name}"
            os.makedirs(strategy_dir, exist_ok=True)
            
            # Save all views with detailed info
            for i, (view, view_data) in enumerate(zip(selected_views, detailed_metrics['individual_views'])):
                # Save image with detailed filename
                img_filename = f"view_{i:03d}_step{view_data['step']:02d}_{view_data['direction']}_psnr{view_data['psnr']:.1f}.jpg"
                Image.fromarray(view['image']).save(f"{strategy_dir}/{img_filename}")
            
            # Save detailed metrics as JSON
            with open(f"{strategy_dir}/detailed_metrics.json", 'w') as f:
                json.dump(detailed_metrics, f, indent=2, default=str)
            
            # Save CSV for easy analysis
            view_df = pd.DataFrame(detailed_metrics['individual_views'])
            view_df.to_csv(f"{strategy_dir}/individual_metrics.csv", index=False)
            
            # Create individual view visualization
            experiment.create_detailed_visualization(detailed_metrics, strategy_name, strategy_dir)
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
        print(f"\nQuick test complete! Results saved to {results_dir}")
    
    # Create comparison summary
    experiment.create_strategy_comparison_report(results_dir)
    
    return results_dir

def main():
    """Main experimental runner"""
    # Run quick test first
    quick_results = run_quick_test()
    return quick_results
    
    # Uncomment below for full experiment
    # # Initialize experiment
    # experiment = ViewSelectionExperiment()
    # 
    # # Run experiment
    # input_image = '/content/SD-T2I-360PanoImage/result.png'
    # results = experiment.run_experiment(input_image, num_runs=1)
    # 
    # # Print summary
    # print("\n" + "="*50)
    # print("EXPERIMENT SUMMARY")
    # print("="*50)
    # 
    # # Print top strategies for each metric
    # for metric, ranking in results['rankings'].items():
    #     print(f"\n{metric.upper()} - Top 3 strategies:")
    #     for i, (strategy, score) in enumerate(list(ranking.items())[:3]):
    #         print(f"  {i+1}. {strategy}: {score:.4f}")
    # 
    # # Print statistical significance
    # print(f"\nStatistical significance tests saved to: {experiment.results_dir}/analysis.json")
    # print(f"Visualizations saved to: {experiment.results_dir}/")
    # 
    # return results

if __name__ == "__main__":
    results = main()
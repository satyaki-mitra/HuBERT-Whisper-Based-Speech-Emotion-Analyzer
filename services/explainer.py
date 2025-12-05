# DEPENDENCIES
import torch
import matplotlib
import numpy as np
import seaborn as sns
from typing import Any
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
from config.settings import EXPORTS_DIR
from utils.logging_util import setup_logger


# Non-interactive backend
matplotlib.use('Agg')  


# SETUP LOGGING
logger = setup_logger(__name__)


class ExplainabilityService:
    """
    Service for generating explainability visualizations
    """
    
    def __init__(self):
        self.viz_dir = EXPORTS_DIR / 'visualizations'
        self.viz_dir.mkdir(parents = True, exist_ok = True)

    
    def generate_attention_visualization(self, analysis_id: str, attention_weights: torch.Tensor, layer_idx: int = -1) -> Path:
        """
        Generate attention heatmap visualization
        """
        try:
            # Extract attention
            if (len(attention_weights.shape) == 4):
                attention = attention_weights[0, :, :, :].mean(dim=0).detach().cpu().numpy()

            else:
                attention = attention_weights.detach().cpu().numpy()
            
            # Create heatmap
            plt.figure(figsize = (12, 10))
            sns.heatmap(data   = attention, 
                        cmap   = 'viridis', 
                        cbar   = True, 
                        square = True,
                       )

            # Decorate plot
            plt.title(label    = f'Attention Weights - Layer {layer_idx}', 
                      fontsize = 16,
                     )
            
            plt.xlabel(xlabel   = 'Key Position', 
                       fontsize = 12,
                      )

            plt.ylabel(ylabel   = 'Query Position', 
                       fontsize = 12,
                      )

            plt.tight_layout()
            
            # Save the plot
            filepath = self.viz_dir / f"{analysis_id}_attention_layer_{layer_idx}.png"

            plt.savefig(fname       = filepath, 
                        dpi         = 150, 
                        bbox_inches = 'tight',
                       )

            plt.close()
            
            logger.info(f"Saved attention visualization: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate attention viz: {repr(e)}")
            raise
    

    def generate_emotion_distribution(self, analysis_id: str, emotion_scores: Dict[str, float]) -> Path:
        """
        Generate emotion distribution bar chart
        """
        try:
            emotions         = list(emotion_scores.keys())
            scores           = list(emotion_scores.values())
            
            # Sort by score
            sorted_pairs     = sorted(zip(emotions, scores), key = lambda x: x[1], reverse = True)
            emotions, scores = zip(*sorted_pairs)
            
            # Create bar chart
            fig, ax          = plt.subplots(figsize = (10, 6))
            colors           = plt.cm.viridis(np.linspace(0.3, 0.9, len(emotions)))
            
            # Plot Horizontal bars
            bars             = ax.barh(y      = emotions, 
                                       width  = scores, 
                                       height = 0.8, 
                                       color  = colors,
                                      )

            # Decorate plot
            ax.set_xlabel(xlabel  = 'Confidence Score', 
                          fontsize = 12,
                         )

            ax.set_title(label      = 'Emotion Distribution', 
                         fontsize   = 16, 
                         fontweight = 'bold',
                        )

            ax.set_xlim(left  = 0, 
                        right = 1.0,
                       )
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()

                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                        f'{width:.2%}', 
                        ha       = 'left', 
                        va       = 'center', 
                        fontsize = 10,
                       )
            
            plt.tight_layout()
            
            # Save
            filepath = self.viz_dir / f"{analysis_id}_emotion_distribution.png"
            plt.savefig(fname       = filepath, 
                        dpi         = 150, 
                        bbox_inches = 'tight',
                       )

            plt.close()
            
            logger.info(f"Saved emotion distribution: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to generate emotion distribution: {repr(e)}")
            raise
    

    def get_explanation(self, analysis_id: str) -> Dict[str, Any]:
        """
        Get complete explainability results
        """
        try:
            visualization_paths = list(self.viz_dir.glob(f"{analysis_id}_*.png"))
            
            result              = {'analysis_id'    : analysis_id,
                                   'visualizations' : {path.stem: str(path) for path in visualization_paths},
                                   'available'      : len(visualization_paths) > 0,
                                  }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get explanation: {repr(e)}")
            return None
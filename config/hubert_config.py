# DEPENDENCIES
from typing import Literal
from dataclasses import dataclass


@dataclass
class HubertConfig:
    """
    HuBERT model configuration
    """
    # Model architecture
    hidden_size              : int                           = 768
    num_hidden_layers        : int                           = 12
    num_attention_heads      : int                           = 12
    intermediate_size        : int                           = 3072
    
    # Pooling strategy
    pooling_mode             : Literal['mean', 'sum', 'max'] = 'mean'
    
    # Classification
    num_labels               : int                           = 6
    final_dropout            : float                         = 0.1
    
    # Processing
    max_length_seconds       : int                           = 30
    stride_length_seconds    : int                           = 5
    
    # Batch processing
    batch_size               : int                           = 8
    use_gpu                  : bool                          = False
    use_mps                  : bool                          = True
    
    # Thresholds
    min_confidence_threshold : float                         = 0.3
    calibration_enabled      : bool                          = True
    
    # Feature extraction
    feature_extractor_config : dict                          = None

    
    def __post_init__(self):
        if self.feature_extractor_config is None:
            self.feature_extractor_config = {'feature_size'          : 1,
                                             'sampling_rate'         : 16000,
                                             'padding_value'         : 0.0,
                                             'do_normalize'          : True,
                                             'return_attention_mask' : True,
                                            }


# Default configuration
DEFAULT_HUBERT_CONFIG = HubertConfig()


# Optimized configurations for different scenarios
STREAMING_CONFIG      = HubertConfig(pooling_mode        = 'mean',
                                     batch_size          = 1,
                                     max_length_seconds  = 10,
                                     calibration_enabled = False,
                                    )

BATCH_CONFIG          = HubertConfig(pooling_mode        = 'mean',
                                     batch_size          = 16,
                                     max_length_seconds  = 30,
                                     calibration_enabled = True,
                                    )

GPU_CONFIG            = HubertConfig(pooling_mode       = 'mean',
                                     batch_size         = 32,
                                     use_gpu            = True,
                                     max_length_seconds = 30,
                                    )

MPS_CONFIG            = HubertConfig(pooling_mode       = 'mean',
                                     batch_size         = 16,
                                     use_mps            = True,
                                     max_length_seconds = 30,
                                    )
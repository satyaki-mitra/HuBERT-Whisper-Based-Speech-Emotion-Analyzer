# DEPENDENCIES
import torch
import torch.nn as nn
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from torch.nn import MSELoss
import torch.nn.functional as F
from dataclasses import dataclass
from config.settings import DEVICE
from transformers import AutoConfig
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss
from config.settings import BATCH_SIZE
from config.settings import SAMPLE_RATE
from config.settings import BASE_EMOTIONS
from utils.logging_util import setup_logger
from config.settings import COMPLEX_EMOTIONS
from config.settings import HUBERT_MODEL_PATH
from transformers.file_utils import ModelOutput
from config.settings import GRANULAR_EMOTION_MAP
from transformers import Wav2Vec2FeatureExtractor
from utils.audio_utils import load_audio_as_tensor
from config.hubert_config import DEFAULT_HUBERT_CONFIG
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.models.hubert.modeling_hubert import HubertPreTrainedModel


# SETUP LOGGING
logger = setup_logger(__name__)

# MODEL CLASSES
class HubertClassificationHead(nn.Module):
    """
    Head for HuBERT classification task
    """
    def __init__(self, config):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        
        return x


@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss          : Optional[torch.FloatTensor]        = None
    logits        : torch.FloatTensor                  = None
    hidden_states : Optional[Tuple[torch.FloatTensor]] = None
    attentions    : Optional[Tuple[torch.FloatTensor]] = None


class HubertForSpeechClassification(HubertPreTrainedModel):
    """
    HuBERT model for speech emotion classification
    """
    def __init__(self, config):
        super().__init__(config)

        self.num_labels   = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config       = config
        self.hubert       = HubertModel(config)
        self.classifier   = HubertClassificationHead(config)

        self.init_weights()
    

    def freeze_feature_extractor(self):
        self.hubert.feature_extractor._freeze_parameters()
    

    def merged_strategy(self, hidden_states, mode = "mean"):
        """
        
        """
        if (mode == "mean"):
            return torch.mean(hidden_states, dim=1)
        
        elif (mode == "sum"):
            return torch.sum(hidden_states, dim=1)

        elif (mode == "max"):
            return torch.max(hidden_states, dim=1)[0]

        else:
            raise ValueError(f"Invalid pooling mode: {mode}")
    

    def forward(self, input_values, attention_mask = None, output_attentions = None, output_hidden_states = None, return_dict = None, labels = None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs       = self.hubert(input_values         = input_values,
                                    attention_mask       = attention_mask,
                                    output_attentions    = output_attentions,
                                    output_hidden_states = output_hidden_states,
                                    return_dict          = return_dict,
                                   )
        
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states = hidden_states, 
                                             mode          = self.pooling_mode,
                                            )

        logits        = self.classifier(hidden_states)
        
        # Loss Calculation
        loss          = None

        if labels is not None:
            if self.config.problem_type is None:
                if (self.num_labels == 1):
                    self.config.problem_type = "regression"

                elif ((self.num_labels > 1) and ((labels.dtype == torch.long) or (labels.dtype == torch.int))):
                    self.config.problem_type = "single_label_classification"

                else:
                    self.config.problem_type = "multi_label_classification"
            
            
            if (self.config.problem_type == "regression"):
                loss_fct = MSELoss()
                loss     = loss_fct(logits.view(-1, self.num_labels), labels)

            elif (self.config.problem_type == "single_label_classification"):
                loss_fct = CrossEntropyLoss()
                loss     = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            elif (self.config.problem_type == "multi_label_classification"):
                loss_fct = BCEWithLogitsLoss()
                loss     = loss_fct(logits, labels)
        
        
        if not return_dict:
            output = (logits,) + outputs[2:]

            return ((loss,) + output) if loss is not None else output
        
        return SpeechClassifierOutput(loss          = loss,
                                      logits        = logits,
                                      hidden_states = outputs.hidden_states,
                                      attentions    = outputs.attentions,
                                     )



# EMOTION PREDICTOR
class EmotionPredictor:
    """
    Emotion prediction service with batch processing
    """
    def __init__(self):
        self.device            = torch.device(DEVICE)
        self.batch_size        = BATCH_SIZE
        
        # Load config
        self.config            = AutoConfig.from_pretrained(pretrained_model_name_or_path = HUBERT_MODEL_PATH)
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path = HUBERT_MODEL_PATH,
                                                                          local_files_only              = True,
                                                                         )
        
        # Load model
        self.model             = HubertForSpeechClassification.from_pretrained(pretrained_model_name_or_path = HUBERT_MODEL_PATH,
                                                                               config                        = self.config,
                                                                               ignore_mismatched_sizes       = True,
                                                                               local_files_only              = True,
                                                                               use_safetensors               = True,
                                                                              )
        
        # Move model to device
        self.model.to(self.device)

        # Set model to eval mode
        self.model.eval()
        
        logger.info(f"Emotion predictor loaded on {self.device}")
    

    def predict_base_emotions(self, audio_path: str) -> Dict[str, float]:
        """
        Predict base emotions from audio
        """
        try:
            # Load audio
            speech = load_audio_as_tensor(audio_path, SAMPLE_RATE)
            
            # Extract features
            inputs = self.feature_extractor(speech,
                                            sampling_rate  = SAMPLE_RATE,
                                            return_tensors = "pt",
                                            padding        = True,
                                           )

            # Move to device
            inputs = {key: inputs[key].to(self.device) for key in inputs}
            
            # Predict
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get probabilities (raw)
            scores_raw  = F.softmax(logits, dim = 1).detach().cpu().numpy()[0]
            
            # Convert to percentages and round
            percentages = list()
            total       = 0
            
            for score in scores_raw:
                percentage = round(float(score) * 100, 4)

                percentages.append(percentage)
                total     += percentage
            
            # Adjust for rounding errors to ensure sum = 100%
            if (abs(total - 100.00) > 0.01):
                max_idx              = np.argmax(scores_raw)
                adjustment           = 100.00 - total
                percentages[max_idx] = round(percentages[max_idx] + adjustment, 2)
            
            # Create dictionaries : For internal calculations (0-1) &  for frontend display (0-100%)
            raw_emotions     = {BASE_EMOTIONS[i]: float(scores_raw[i]) for i in range(len(scores_raw))}
            rounded_emotions = {BASE_EMOTIONS[i]: percentages[i] for i in range(len(percentages))}
            
            return {'raw'     : raw_emotions,      
                    'display' : rounded_emotions,
                   }
            
        except Exception as e:
            logger.error(f"Emotion prediction error: {repr(e)}")
            raise
    

    def predict_granular_emotions(self, base_emotions: Dict[str, dict]) -> Dict:
        """
        Map base emotions to granular emotions : Uses raw scores for calculations, returns display values
        """
        # Extract raw emotions for calculations
        base_emotions_raw      = base_emotions['raw']
        base_emotions_display  = base_emotions['display']

        result                 = {'base'      : base_emotions_display,
                                  'primary'   : None,
                                  'secondary' : None,
                                  'complex'   : [],
                                 }
        
        dominant_name          = max(base_emotions_raw.items(), key = lambda x: x[1])[0]
        dominant_score_raw     = base_emotions_raw[dominant_name]
        
        # Display percentage
        dominant_score_display = base_emotions_display[dominant_name]
        
        # Map to granular using RAW scores
        if dominant_name in GRANULAR_EMOTION_MAP:
            mapping   = GRANULAR_EMOTION_MAP[dominant_name]
            threshold = mapping['threshold']
            
            # Use raw score for threshold comparison
            if (dominant_score_raw >= threshold):
                result['primary'] = {'emotions'   : mapping['primary'],
                                     'confidence' : dominant_score_display,  
                                    }
                
                # For secondary, still use raw score
                if (0.3 <= dominant_score_raw < 0.7):
                    result['secondary'] = {'emotions'   : mapping['secondary'],
                                           'confidence' : round(dominant_score_display * 0.7, 2),
                                          }
        
        # Detect complex emotions using RAW scores
        sorted_emotions_raw = sorted(base_emotions_raw.items(), 
                                     key     = lambda x: x[1], 
                                     reverse = True,
                                    )

        if (len(sorted_emotions_raw) >= 2):
            top_two = tuple(sorted([sorted_emotions_raw[0][0], sorted_emotions_raw[1][0],]))
            
            for emotion_pair, complex_name in COMPLEX_EMOTIONS.items():
                if (set(top_two) == set(emotion_pair)):
                    # Get display values for confidence
                    score1_display    = base_emotions_display[top_two[0]]
                    score2_display    = base_emotions_display[top_two[1]]
                    avg_score_display = (score1_display + score2_display) / 2
                    
                    # Use raw scores for threshold
                    score1_raw        = base_emotions_raw[top_two[0]]
                    score2_raw        = base_emotions_raw[top_two[1]]
                    avg_score_raw     = (score1_raw + score2_raw) / 2
                    
                    if (avg_score_raw > 0.3):
                        result['complex'].append({'name'       : complex_name,
                                                  'components' : list(top_two),
                                                  'confidence' : round(avg_score_display, 2),
                                                })
        
        return result


    def predict(self, audio_path: str, mode: str = 'both') -> Dict:
        """
        Predict emotions based on mode
        """
        base_emotions_result = self.predict_base_emotions(audio_path = audio_path)
        
        if (mode == 'base'):
            return {'base' : base_emotions_result['display']}
        
        elif (mode in ['granular', 'both']):
            granular_result = self.predict_granular_emotions(base_emotions = base_emotions_result)
            
            return granular_result
        
        else:
            raise ValueError(f"Invalid mode: {mode}")
    

    def get_attention_weights(self, audio_path: str):
        """
        Extract attention weights for explainability
        """
        try:
            speech = load_audio_as_tensor(audio_path, SAMPLE_RATE)
            
            inputs = self.feature_extractor(speech,
                                            sampling_rate  = SAMPLE_RATE,
                                            return_tensors = "pt",
                                            padding        = True,
                                           )

            inputs = {key: inputs[key].to(self.device) for key in inputs}
            
            with torch.no_grad():
                outputs = self.model(**inputs,
                                     output_attentions    = True,
                                     output_hidden_states = True,
                                    )
            
            return outputs.attentions
            
        except Exception as e:
            logger.error(f"Failed to extract attention: {repr(e)}")
            raise



# Global instance
_predictor_instance = None



def get_emotion_predictor() -> EmotionPredictor:
    """
    Get or create emotion predictor singleton
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = EmotionPredictor()
    
    return _predictor_instance
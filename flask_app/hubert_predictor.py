# DEPENDENCIES
import sys
import torch
import warnings
import torchaudio
import torch.nn as nn
from typing import Tuple
from typing import Optional
from torch.nn import MSELoss
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoConfig
from torch.nn import CrossEntropyLoss
from torch.nn import BCEWithLogitsLoss 
from transformers.file_utils import ModelOutput
from transformers import Wav2Vec2FeatureExtractor
from transformers.models.hubert.modeling_hubert import HubertModel
from transformers.models.hubert.modeling_hubert import HubertPreTrainedModel

sys.path.append('../')
from config import HUBERT_MODEL_PATH


# IGNORING ALL KIND OF WARNINGS RAISED AT RUN TIME
warnings.filterwarnings('ignore')


class HubertClassificationHead(nn.Module):
    """
    Head for hubert classification task
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
    loss          : Optional[torch.FloatTensor] = None
    logits        : torch.FloatTensor = None
    hidden_states : Optional[Tuple[torch.FloatTensor]] = None
    attentions    : Optional[Tuple[torch.FloatTensor]] = None

    

class HubertForSpeechClassification(HubertPreTrainedModel):
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

    def merged_strategy(self, hidden_states, mode="mean"):
        if (mode == "mean"):
            outputs = torch.mean(hidden_states, dim=1)
        elif (mode == "sum"):
            outputs = torch.sum(hidden_states, dim=1)
        elif (mode == "max"):
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception("The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(self, input_values, attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None):
        return_dict   = return_dict if return_dict is not None else self.config.use_return_dict
        outputs       = self.hubert(input_values         = input_values,
                                    attention_mask       = attention_mask,
                                    output_attentions    = output_attentions,
                                    output_hidden_states = output_hidden_states,
                                    return_dict          = return_dict)
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states = hidden_states, 
                                             mode          = self.pooling_mode)
        logits        = self.classifier(hidden_states)
        loss          = None
        
        if (labels is not None):
            if (self.config.problem_type is None):
                if (self.num_labels == 1):
                    self.config.problem_type = "regression"
                elif ((self.num_labels > 1) and (labels.dtype == torch.long or labels.dtype == torch.int)):
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

        if (not return_dict):
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        speech_classifier_output = SpeechClassifierOutput(loss          = loss,
                                                          logits        = logits,
                                                          hidden_states = outputs.hidden_states,
                                                          attentions    = outputs.attentions) 
        return speech_classifier_output



# CONFIGURATIONS
# Using CPU for inference
device                    = torch.device("cpu")

# Load the model configuration
config                    = AutoConfig.from_pretrained(pretrained_model_name_or_path = HUBERT_MODEL_PATH)

# Load the feature extractor model
feature_extractor         = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path = HUBERT_MODEL_PATH,
                                                                     local_files_only              = True,
                                                                     token                         = True)
# Expected sampling rate for the model
sampling_rate             = feature_extractor.sampling_rate

# Load the HuBERT model
hubert_model              = HubertForSpeechClassification.from_pretrained(pretrained_model_name_or_path = HUBERT_MODEL_PATH,
                                                                          config                        = config,
                                                                          ignore_mismatched_sizes       = True,
                                                                          local_files_only              = True,
                                                                          use_safetensors               = True).to(device)


def speech_to_tensor(audio_path: str, sampling_rate: int) -> torch.Tensor:
    """
    Load an audio file and resample it to the given sampling rate, finally returns a torch Tensor

    Arguments:
    ----------
        audio_path    {str} : Path to the audio file to be loaded
        
        sampling_rate {int} : Desired sampling rate for resampling the audio

    Errors:
    -------
        SpeechToTensorError : If the audio file cannot be loaded as torch tensor

    Returns:
    --------
         {torch.Tensor}     : The resampled speech as a one-dimensional tensor    
    """
    try:
        # Load audio data and extract the original sampling rate
        speech_array, _sampling_rate = torchaudio.load(uri = audio_path)
        
        # Resample the audio to the desired sampling rate
        resampler                    = torchaudio.transforms.Resample(orig_freq = _sampling_rate, 
                                                                      new_freq  = sampling_rate)
        
        # Resample and convert to a one-dimensional tensor
        speech_tensor                = resampler(speech_array).squeeze().numpy()
        
        return speech_tensor
    
    except Exception as SpeechToTensorError:
        # Return exceptions related to loading or processing audio
        return (f"SpeechToTensorError: Error while converting audio to Tensor: '{audio_path}': {repr(SpeechToTensorError)}")



def predict_emotion(audio_path: str, sampling_rate: int) -> list:
    """
    Predicts the emotions in percentage from an audio file using a pre-trained HuBERT model

    Arguments:
    ----------
        audio_path      {str}  : Path to the audio file

        sampling_rate   {int}  : The sampling rate to which the audio will be resampled

    Raises:
    -------
        EmotionPredictionError : If the emotion prediction process fails at any stage

    Returns:
    --------
               {list}          : A list of dictionaries containing emotion labels and their 
                                 respective scores in percentage format
    """
    try:
        # Convert the audio file to an array
        speech = speech_to_tensor(audio_path    = audio_path, 
                                  sampling_rate = sampling_rate)
        
        # Preprocess with feature extractor
        inputs = feature_extractor(speech, 
                                   sampling_rate  = sampling_rate, 
                                   return_tensors = "pt", 
                                   padding        = True)
        
        # Move inputs to the specified device
        inputs = {key: inputs[key].to(device) for key in inputs}

        # Perform emotion prediction with the pre-trained HuBERT model
        with torch.no_grad():
            logits = hubert_model(**inputs).logits
  
        # Convert logits to probabilities using softmax
        scores  = F.softmax(input = logits, 
                            dim   = 1).detach().cpu().numpy()[0]
        
        # Convert probabilities to a list of emotion-label dictionaries
        outputs = [{config.id2label[i] : f"{round(score * 100, 5):.2f}%"} for i, score in enumerate(scores)]
        
        return outputs
    
    except Exception as EmotionPredictionError:
        # Return exceptions related to emotion prediction
        return (f"EmotionPredictionError : While predicting emotion from the input audio, got : {repr(EmotionPredictionError)}")



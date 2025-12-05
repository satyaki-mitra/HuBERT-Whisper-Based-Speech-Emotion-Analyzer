# DEPENDENCIES
import os
import json
import time
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from collections import Counter
from dataclasses import dataclass
from utils.logging_util import setup_logger
from config.settings import BENCHMARK_DATASETS
from config.settings import BASE_EMOTIONS
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


# SETUP LOGGING
logger = setup_logger(__name__)


# DATASET METADATA
@dataclass
class DatasetInfo:
    """
    Metadata for benchmark dataset
    """
    name           : str
    path           : Path
    num_samples    : int
    num_classes    : int
    emotion_labels : List[str]
    sample_rate    : int
    description    : str
    citation       : Optional[str] = None


# BASE DATASET CLASS
class BaseDataset:
    """
    Base class for all datasets
    """
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.samples      = list()
        self.labels       = list()
        self.EMOTION_MAP  = dict()
    

    def load(self) -> Tuple[List[Path], List[str]]:
        """
        Load dataset - to be implemented by subclasses
        """
        raise NotImplementedError

    
    def _filter_by_base_emotions(self) -> None:
        """
        Filter samples to only include BASE_EMOTIONS if specified
        """
        if not BASE_EMOTIONS:
            return
        
        filtered_samples = list()
        filtered_labels  = list()
        
        for sample, label in zip(self.samples, self.labels):
            if label in BASE_EMOTIONS:
                filtered_samples.append(sample)
                filtered_labels.append(label)
            
            else:
                logger.debug(f"Filtered out {sample}: emotion '{label}' not in BASE_EMOTIONS")
        
        self.samples = filtered_samples
        self.labels  = filtered_labels
        
        logger.info(f"Filtered to {len(self.samples)} samples with BASE_EMOTIONS")


# IEMOCAP DATASET
class IEMOCAPDataset(BaseDataset):
    """
    Interactive Emotional Dyadic Motion Capture (IEMOCAP) Database
    
    Citation : Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database. Language Resources and Evaluation, 42(4), 335-359.
    """
    EMOTION_MAP = {'ang' : 'Anger',
                   'hap' : 'Happiness',
                   'exc' : 'Happiness',  # Map excitement to happiness
                   'sad' : 'Sadness',
                   'neu' : 'Neutral',
                   'fru' : 'Anger',      # Map frustration to anger
                   'fea' : 'Fear',
                   'sur' : 'Surprise',
                   'dis' : 'Sadness',    # Map disgust to sadness
                   'oth' : None,         # Skip other emotions
                   'xxx' : None,         # Skip undefined
                  }

                
    def __init__(self, dataset_path: Path):
        super().__init__(dataset_path)
    

    def load(self) -> Tuple[List[Path], List[str]]:
        """
        Load IEMOCAP dataset
        
        Returns:
        --------
            { tuple }    : Tuple of (audio_paths, emotion_labels)
        """
        logger.info(f"Loading IEMOCAP dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.warning(f"IEMOCAP path not found: {self.dataset_path}")
            return [], []
        
        # IEMOCAP structure : Session{1-5}/sentences/wav/{speaker}/{file}.wav
        # Labels in         : Session{1-5}/dialog/EmoEvaluation/*.txt
        for session_dir in self.dataset_path.glob('Session*'):
            if not session_dir.is_dir():
                continue
            
            logger.debug(f"Processing {session_dir.name}")
            
            # Load emotion labels from EmoEvaluation files
            emotion_labels = self._load_emotion_labels(session_dir)
            
            # Load audio files
            wav_dir        = session_dir / 'sentences' / 'wav'
            
            if wav_dir.exists():
                for wav_file in wav_dir.rglob('*.wav'):
                    file_id = wav_file.stem
                    
                    # Get emotion label
                    emotion = emotion_labels.get(file_id)
                    
                    if emotion and emotion in self.EMOTION_MAP:
                        mapped_emotion = self.EMOTION_MAP[emotion]
                        
                        if mapped_emotion:
                            self.samples.append(wav_file)
                            self.labels.append(mapped_emotion)
        
        logger.info(f"Loaded {len(self.samples)} IEMOCAP samples")
        
        # Filter by BASE_EMOTIONS if specified
        self._filter_by_base_emotions()
        
        return self.samples, self.labels
    

    def _load_emotion_labels(self, session_dir: Path) -> Dict[str, str]:
        """
        Load emotion labels from EmoEvaluation files
        """
        labels   = dict()
        eval_dir = session_dir / 'dialog' / 'EmoEvaluation'
        
        if not eval_dir.exists():
            return labels
        
        for eval_file in eval_dir.glob('*.txt'):
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        
                        if line.startswith('['):
                            # Format: [START - END]	TURN_NAME	EMOTION	[V, A, D]
                            parts = line.split('\t')
                            
                            if (len(parts) >= 3):
                                file_id         = parts[1].strip()
                                emotion         = parts[2].strip().lower()
                                labels[file_id] = emotion

            except Exception as e:
                logger.warning(f"Error reading {eval_file}: {e}")
                continue
        
        return labels
    

    @staticmethod
    def get_info() -> DatasetInfo:
        """
        Get dataset metadata
        """
        return DatasetInfo(name           = 'IEMOCAP',
                           path           = BENCHMARK_DATASETS.get('IEMOCAP', {}).get('path', Path('benchmarks/data/iemocap')),
                           num_samples    = 10039,
                           num_classes    = 6,
                           emotion_labels = ['Anger', 'Happiness', 'Sadness', 'Neutral', 'Fear', 'Surprise'],
                           sample_rate    = 16000,
                           description    = 'Acted emotional speech from 10 actors in dyadic sessions',
                           citation       = 'Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database.',
                          )


# RAVDESS DATASET
class RAVDESSDataset(BaseDataset):
    """
    Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
    
    Citation : Livingstone, S.R., & Russo, F.A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). PLOS ONE, 13(5), e0196391.
    """
    EMOTION_MAP = {1 : 'Neutral',
                   2 : 'Neutral',    # Calm -> Neutral
                   3 : 'Happiness',
                   4 : 'Sadness',
                   5 : 'Anger',
                   6 : 'Fear',
                   7 : 'Sadness',    # Disgust -> Sadness
                   8 : 'Surprise',
                 }

    
    def __init__(self, dataset_path: Path):
        super().__init__(dataset_path)
    

    def load(self) -> Tuple[List[Path], List[str]]:
        """
        Load RAVDESS dataset
        
        Returns:
        --------
            { tuple }    : Tuple of (audio_paths, emotion_labels)
        """
        logger.info(f"Loading RAVDESS dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.warning(f"RAVDESS path not found: {self.dataset_path}")
            return [], []
        
        # RAVDESS structure : Actor_{01-24}/*.wav
        # Filename format   : {modality}-{vocal_channel}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav
        for actor_dir in self.dataset_path.glob('Actor_*'):
            if not actor_dir.is_dir():
                continue
            
            for wav_file in actor_dir.glob('*.wav'):
                emotion_code = self._parse_emotion_code(wav_file.name)
                
                if emotion_code and emotion_code in self.EMOTION_MAP:
                    emotion = self.EMOTION_MAP[emotion_code]
                    
                    self.samples.append(wav_file)
                    self.labels.append(emotion)
        
        logger.info(f"Loaded {len(self.samples)} RAVDESS samples")
        
        # Filter by BASE_EMOTIONS if specified
        self._filter_by_base_emotions()
        
        return self.samples, self.labels
    

    def _parse_emotion_code(self, filename: str) -> Optional[int]:
        """
        Parse emotion code from RAVDESS filename
        """
        try:
            # Remove extension and split by '-'
            base_name = os.path.splitext(filename)[0]
            parts     = base_name.split('-')
            
            if (len(parts) >= 3):
                return int(parts[2])
                
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse emotion code from {filename}: {e}")
        
        return None
    

    @staticmethod
    def get_info() -> DatasetInfo:
        """
        Get dataset metadata
        """
        return DatasetInfo(name           = 'RAVDESS',
                           path           = BENCHMARK_DATASETS.get('RAVDESS', {}).get('path', Path('benchmarks/data/ravdess')),
                           num_samples    = 1440,
                           num_classes    = 8,
                           emotion_labels =['Neutral', 'Calm', 'Happiness', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise'],
                           sample_rate    = 48000,
                           description    = 'Emotional speech and song from 24 professional actors',
                           citation       = 'Livingstone, S.R., & Russo, F.A. (2018). The Ryerson Audio-Visual Database.',
                          )


# CREMA-D DATASET
class CREMADDataset(BaseDataset):
    """
    Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D)
    
    Citation : Cao, H., et al. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. IEEE Transactions on Affective Computing, 5(4), 377-390.
    """
    EMOTION_MAP = {'ANG' : 'Anger',
                   'DIS' : 'Sadness',  # Disgust -> Sadness
                   'FEA' : 'Fear',
                   'HAP' : 'Happiness',
                   'NEU' : 'Neutral',
                   'SAD' : 'Sadness',
                  }

                
    def __init__(self, dataset_path: Path):
        super().__init__(dataset_path)
    

    def load(self) -> Tuple[List[Path], List[str]]:
        """
        Load CREMA-D dataset
        """
        logger.info(f"Loading CREMA-D dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            logger.warning(f"CREMA-D path not found: {self.dataset_path}")
            return [], []
        
        # CREMA-D filename format : {ActorID}_{Sentence}_{Emotion}_{Intensity}.wav
        # Example                 : 1001_DFA_ANG_XX.wav 
        for wav_file in self.dataset_path.glob('*.wav'):
            emotion_code = self._parse_emotion_code(wav_file.name)
            
            if (emotion_code and (emotion_code in self.EMOTION_MAP)):
                emotion = self.EMOTION_MAP[emotion_code]

                self.samples.append(wav_file)
                self.labels.append(emotion)
        
        logger.info(f"Loaded {len(self.samples)} CREMA-D samples")
        
        # Filter by BASE_EMOTIONS if specified
        self._filter_by_base_emotions()
        
        return self.samples, self.labels
    

    def _parse_emotion_code(self, filename: str) -> Optional[str]:
        """
        Parse emotion code from CREMA-D filename
        """
        try:
            # Remove extension and split by '_'
            base_name = os.path.splitext(filename)[0]
            parts     = base_name.split('_')
            
            if (len(parts) >= 3):
                return parts[2]

        except IndexError as e:
            logger.debug(f"Failed to parse emotion code from {filename}: {e}")
        
        return None
    

    @staticmethod
    def get_info() -> DatasetInfo:
        """
        Get dataset metadata
        """
        return DatasetInfo(name           = 'CREMA-D',
                           path           = BENCHMARK_DATASETS.get('CREMA-D', {}).get('path', Path('benchmarks/data/crema-d')),
                           num_samples    = 7442,
                           num_classes    = 6,
                           emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness'],
                           sample_rate    = 16000,
                           description    = 'Multimodal emotional expressions from 91 actors',
                           citation       = 'Cao, H., et al. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.',
                          )


# DATASET FACTORY
class DatasetFactory:
    """
    Factory for loading different datasets
    """
    DATASETS = {'IEMOCAP' : IEMOCAPDataset,
                'RAVDESS' : RAVDESSDataset,
                'CREMA-D' : CREMADDataset,
               }
    

    @classmethod
    def load_dataset(cls, name: str) -> Tuple[List[Path], List[str]]:
        """
        Load dataset by name
        
        Arguments:
        ----------
            name { str } : Dataset name (IEMOCAP, RAVDESS, CREMA-D)
        
        Returns:
        --------
            { tuple }    : Tuple of (audio_paths, labels)
        """
        if name not in cls.DATASETS:
            available = list(cls.DATASETS.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        # Get dataset path from config or default
        if name in BENCHMARK_DATASETS:
            dataset_path = BENCHMARK_DATASETS[name]['path']
        
        else:
            dataset_path = Path(f'benchmarks/data/{name.lower()}')
        
        # Create and load dataset
        dataset_class = cls.DATASETS[name]
        dataset       = dataset_class(dataset_path)
        
        return dataset.load()
    

    @classmethod
    def get_dataset_info(cls, name: str) -> DatasetInfo:
        """
        Get dataset metadata
        """
        if name not in cls.DATASETS:
            available = list(cls.DATASETS.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        dataset_class = cls.DATASETS[name]
        
        return dataset_class.get_info()
    

    @classmethod
    def list_datasets(cls) -> List[str]:
        """
        List available datasets
        """
        return list(cls.DATASETS.keys())
    

    @classmethod
    def get_all_dataset_info(cls) -> Dict[str, DatasetInfo]:
        """
        Get metadata for all datasets
        """
        return {name: cls.get_dataset_info(name) for name in cls.DATASETS}


# BENCHMARK DATASET LOADER
class BenchmarkDatasetLoader:
    """
    Unified loader for benchmark runner
    """
    @staticmethod
    def load_for_benchmark(dataset_name: str) -> Tuple[List[Path], List[str]]:
        """
        Load dataset specifically for benchmarking
        
        Arguments:
        ----------
            dataset_name { str } : Dataset name
        
        Returns:
        --------
                { tuple }        : (audio_paths, emotion_labels)
        """
        try:
            # Load using DatasetFactory
            audio_files, labels = DatasetFactory.load_dataset(name = dataset_name)
            
            if not audio_files:
                logger.warning(f"No samples loaded for {dataset_name}")
                return [], []
            
            logger.info(f"Loaded {len(audio_files)} samples from {dataset_name}")
            logger.info(f"Emotion distribution: {Counter(labels)}")
            
            return audio_files, labels
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {repr(e)}", exc_info=True)
            return [], []
    

    @staticmethod
    def validate_dataset(dataset_name: str) -> Dict[str, Any]:
        """
        Validate dataset structure and contents
        """
        validation_result = {'dataset'    : dataset_name,
                             'valid'      : False,
                             'issues'     : [],
                             'statistics' : {},
                            }
        
        try:
            # Check if dataset exists in factory
            if dataset_name not in DatasetFactory.list_datasets():
                validation_result['issues'].append(f"Dataset not registered in DatasetFactory")
                
                return validation_result
            
            # Get dataset info
            dataset_info                      = DatasetFactory.get_dataset_info(name = dataset_name)
            validation_result['dataset_info'] = {'name'             : dataset_info.name,
                                                 'expected_path'    : str(dataset_info.path),
                                                 'expected_samples' : dataset_info.num_samples,
                                                 'expected_classes' : dataset_info.num_classes,
                                                }
            
            # Check path exists
            if not dataset_info.path.exists():
                validation_result['issues'].append(f"Dataset path not found: {dataset_info.path}")

                return validation_result
            
            # Try to load samples
            audio_files, labels = DatasetFactory.load_dataset(name = dataset_name)
            
            if not audio_files:
                validation_result['issues'].append("No samples loaded")

                return validation_result
            
            # Compute statistics
            stats                           = compute_dataset_statistics(labels)
            validation_result['statistics'] = stats
            validation_result['valid']      = True
            
            # Check for issues
            if (stats['total_samples'] == 0):
                validation_result['issues'].append("No valid samples found")
                validation_result['valid'] = False
            
            if BASE_EMOTIONS:
                missing_emotions = [e for e in BASE_EMOTIONS if e not in stats['class_distribution']]
                
                if missing_emotions:
                    validation_result['issues'].append(f"Missing BASE_EMOTIONS: {missing_emotions}")
            
        except Exception as e:
            validation_result['issues'].append(f"Validation error: {repr(e)}")
        
        return validation_result



# DATA SPLITS
def create_train_test_split(samples: List[Path], labels: List[str], test_size: float = 0.2, random_seed: int = 42) -> Tuple:
    """
    Create train/test split
    
    Arguments:
    ----------
        samples     { list }  : List of audio paths

        labels      { list }  : List of emotion labels
        
        test_size   { float } : Proportion for test set
        
        random_seed { int }   : Random seed for reproducibility
    
    Returns:
    --------
             { tuple }        : (X_train, X_test, y_train, y_test)
    """
    return train_test_split(samples,
                            labels,
                            test_size    = test_size,
                            random_state = random_seed,
                            stratify     = labels,     # Maintain class balance
                           )


def create_k_folds(samples: List[Path], labels: List[str], n_folds: int = 5, random_seed: int = 42):
    """
    Create k-fold cross-validation splits
    
    Arguments:
    ----------
        samples     { list } : List of audio paths

        labels      { list } : List of emotion labels
        
        n_folds     { int }  : Number of folds
        
        random_seed { int }  : Random seed
    
    Yields:
    -------
        (train_indices, test_indices) for each fold
    """
    skf           = StratifiedKFold(n_splits     = n_folds, 
                                    shuffle      = True, 
                                    random_state = random_seed,
                                   )

    samples_array = np.array(samples)
    labels_array  = np.array(labels)
    
    for train_idx, test_idx in skf.split(samples_array, labels_array):
        yield (samples_array[train_idx],
               samples_array[test_idx],
               labels_array[train_idx],
               labels_array[test_idx]
              )


# DATASET STATISTICS
def compute_dataset_statistics(labels: List[str]) -> Dict[str, any]:
    """
    Compute dataset statistics
    
    Arguments:
    ----------
        labels { list } : List of emotion labels
    
    Returns:
    --------
          { dict }      : Dictionary with statistics
    """
    label_counts = Counter(labels)
    total        = len(labels)
    
    if (total == 0):
        return {'total_samples'      : 0,
                'num_classes'        : 0,
                'class_distribution' : {},
                'class_percentages'  : {},
                'is_balanced'        : False,
                'balance_ratio'      : 0.0,
               }
    
    return {'total_samples'      : total,
            'num_classes'        : len(label_counts),
            'class_distribution' : dict(label_counts),
            'class_percentages'  : {label: (count / total) * 100 for label, count in label_counts.items()},
            'most_common'        : label_counts.most_common(1)[0] if label_counts else None,
            'least_common'       : label_counts.most_common()[-1] if label_counts else None,
            'is_balanced'        : max(label_counts.values()) / min(label_counts.values()) < 1.5 if label_counts else False,
            'balance_ratio'      : min(label_counts.values()) / max(label_counts.values()) if label_counts else 0.0,
           }


# UTILITY FUNCTIONS
def save_dataset_cache(name: str, samples: List[Path], labels: List[str], cache_dir: Path):
    """
    Save dataset to cache for faster loading
    """
    cache_dir.mkdir(parents = True, exist_ok = True)
    
    cache_file = cache_dir / f"{name}_cache.json"
    
    data       = {'samples'   : [str(s) for s in samples],
                  'labels'    : labels,
                  'timestamp' : time.strftime("%Y-%m-%d %H:%M:%S") if 'time' in globals() else "unknown",
                 }
    
    try:
        with open(cache_file, 'w') as f:
            json.dump(obj    = data, 
                      fp     = f, 
                      indent = 4,
                     )
        
        logger.info(f"Saved dataset cache: {cache_file}")
        
        return cache_file
    
    except Exception as e:
        logger.error(f"Failed to save cache: {e}")
        return None


def load_dataset_cache(name: str, cache_dir: Path) -> Optional[Tuple[List[Path], List[str]]]:
    """
    Load dataset from cache
    """
    cache_file = cache_dir / f"{name}_cache.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r') as f:
            data = json.load(fp=f)
        
        samples = [Path(s) for s in data['samples']]
        labels  = data['labels']
        
        logger.info(f"Loaded dataset from cache: {cache_file}")
        logger.info(f"Cached samples: {len(samples)}")
        
        return samples, labels
    
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None

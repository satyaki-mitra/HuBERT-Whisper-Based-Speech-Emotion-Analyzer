# DEPENDENCIES
import time
import json
import numpy as np
from typing import Any
from typing import List
from typing import Dict
from pathlib import Path
from config.settings import BASE_EMOTIONS
from sklearn.metrics import accuracy_score
from utils.logging_util import setup_logger
from sklearn.metrics import confusion_matrix
from config.settings import BENCHMARK_DATASETS
from benchmarks.datasets import DatasetFactory
from sklearn.metrics import classification_report
from benchmarks.datasets import BenchmarkDatasetLoader
from services.emotion_predictor import get_emotion_predictor


# SETUP LOGGING
logger = setup_logger(__name__)


class BenchmarkRunner:
    """
    Runner for benchmarking emotion recognition models
    """
    def __init__(self):
        self.results = dict()

    
    def run_benchmark(self, dataset_name: str = 'IEMOCAP') -> Dict[str, Any]:
        """
        Run benchmark on specified dataset
        
        Arguments:
        ----------
            dataset_name { str } : Name of benchmark dataset
        
        Returns:
        --------
                 { dict }        : Dictionary containing benchmark results
        """
        try:
            logger.info(f"Starting benchmark on {dataset_name}")
            
            if dataset_name not in DatasetFactory.list_datasets():
                raise ValueError(f"Unknown dataset: {dataset_name}. Available: {DatasetFactory.list_datasets()}")
            
            dataset_config = BENCHMARK_DATASETS.get(dataset_name, {})
            dataset_path   = dataset_config.get('path', Path(f'benchmarks/data/{dataset_name.lower()}'))
            
            if not dataset_path.exists():
                logger.warning(f"Dataset path not found: {dataset_path}")
                
                return self._mock_benchmark_results(dataset_name = dataset_name)
            
            # Load predictor
            predictor           = get_emotion_predictor()
            
            # Load dataset using unified loader
            audio_files, labels = BenchmarkDatasetLoader.load_for_benchmark(dataset_name)
            
            if not audio_files:
                logger.warning(f"No valid audio files found for {dataset_name}, using mock results")
                return self._mock_benchmark_results(dataset_name=dataset_name)
            
            # Filter labels to match BASE_EMOTIONS if predictor expects specific emotions
            filtered_audio_files = list()
            filtered_labels      = list()
            
            for audio_file, label in zip(audio_files, labels):
                if BASE_EMOTIONS and label not in BASE_EMOTIONS:
                    logger.debug(f"Skipping {audio_file}: emotion '{label}' not in BASE_EMOTIONS")
                    continue

                filtered_audio_files.append(audio_file)
                filtered_labels.append(label)
            
            if not filtered_audio_files:
                logger.warning(f"No samples with BASE_EMOTIONS found for {dataset_name}")
                return self._mock_benchmark_results(dataset_name=dataset_name)
            
            audio_files     = filtered_audio_files
            labels          = filtered_labels
            
            logger.info(f"Running benchmark on {len(audio_files)} samples with {len(set(labels))} emotions")
            
            # Run predictions
            predictions     = list()
            inference_times = list()
            
            for idx, audio_file in enumerate(audio_files):
                start_time = time.time()
                
                try:
                    result     = predictor.predict_base_emotions(str(audio_file))
                    pred_label = max(result.items(), key=lambda x: x[1])[0]

                    predictions.append(pred_label)
                    
                    # Time in ms
                    inference_time = (time.time() - start_time) * 1000  
                    
                    inference_times.append(inference_time)
                    
                    if ((idx + 1) % 100 == 0):
                        logger.info(f"Processed {idx + 1}/{len(audio_files)} samples")
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {audio_file}: {e}")
                    predictions.append('Unknown')
                    inference_times.append(0)
            
            # Calculate metrics
            accuracy      = accuracy_score(labels, predictions)
            conf_matrix   = confusion_matrix(labels, predictions, labels = sorted(set(labels)))
            class_report  = classification_report(labels, predictions, output_dict = True)
            
            # Per-class accuracy
            per_class_acc = dict()
            unique_labels = sorted(set(labels))
            
            for emotion in unique_labels:
                mask = (np.array(labels) == emotion)
                
                if (mask.sum() > 0):
                    per_class_acc[emotion] = accuracy_score(np.array(labels)[mask],
                                                            np.array(predictions)[mask],
                                                           )
            
            # Dataset information
            dataset_info = DatasetFactory.get_dataset_info(dataset_name)
            
            results      = {'dataset'                      : dataset_name,
                            'dataset_info'                 : {'name'           : dataset_info.name,
                                                              'num_samples'    : len(audio_files),
                                                              'num_classes'    : len(unique_labels),
                                                              'emotion_labels' : unique_labels,
                                                              'sample_rate'    : dataset_info.sample_rate,
                                                              'description'    : dataset_info.description,
                                                             },
                            'total_samples'                : len(audio_files),
                            'accuracy'                     : float(accuracy),
                            'per_class_accuracy'           : per_class_acc,
                            'confusion_matrix'             : conf_matrix.tolist(),
                            'classification_report'        : class_report,
                            'avg_inference_time_ms'        : float(np.mean(inference_times)),
                            'std_inference_time_ms'        : float(np.std(inference_times)),
                            'min_inference_time_ms'        : float(np.min(inference_times)),
                            'max_inference_time_ms'        : float(np.max(inference_times)),
                            'inference_times_distribution' : {'p25' : float(np.percentile(inference_times, 25)),
                                                              'p50' : float(np.percentile(inference_times, 50)),
                                                              'p75' : float(np.percentile(inference_times, 75)),
                                                             }
                           }
            
            logger.info(f"Benchmark completed: {dataset_name} - Accuracy: {accuracy:.2%}")
            logger.info(f"Inference time: {results['avg_inference_time_ms']:.1f} ± {results['std_inference_time_ms']:.1f} ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Benchmark failed: {repr(e)}", exc_info = True)
            raise
    

    def _mock_benchmark_results(self, dataset_name: str) -> Dict[str, Any]:
        """
        Generate mock benchmark results for demonstration
        """
        # Get dataset info for consistency
        try:
            dataset_info   = DatasetFactory.get_dataset_info(dataset_name)

            # Take first 6 for mock
            emotion_labels = dataset_info.emotion_labels[:6]  
        
        except:
            emotion_labels = ['Anger', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
        
        per_class_acc = {emotion: 0.8 + (i * 0.02) for i, emotion in enumerate(emotion_labels)}
        
        # Create mock confusion matrix
        n_classes     = len(emotion_labels)
        conf_matrix   = []

        for i in range(n_classes):
            row = [100 if (i == j) else np.random.randint(0, 15) for j in range(n_classes)]
            conf_matrix.append(row)
        
        return {'dataset'               : dataset_name,
                'dataset_info'          : {'name'           : dataset_name,
                                           'num_samples'    : 1000,
                                           'num_classes'    : n_classes,
                                           'emotion_labels' : emotion_labels,
                                           'description'    : 'Mock dataset',
                                          },
                'total_samples'         : 1000,
                'accuracy'              : 0.847,
                'per_class_accuracy'    : per_class_acc,
                'confusion_matrix'      : conf_matrix,
                'avg_inference_time_ms' : 245.3,
                'std_inference_time_ms' : 32.1,
                'min_inference_time_ms' : 180.5,
                'max_inference_time_ms' : 420.8,
                'note'                  : 'Mock results - dataset not found or loading failed',
               }
    

    def compare_models(self, model_configs: List[Dict]) -> Dict[str, Any]:
        """
        Compare multiple model configurations
        
        Arguments:
        ----------
            model_configs { list } : List of model configuration dictionaries
        
        Returns:
        --------
                 { dict }          : Comparison results
        """
        comparison = {'models'          : [],
                      'best_overall'    : None,
                      'best_per_metric' : {},
                      'summary'         : {},
                     }
        
        for config in model_configs:
            model_name              = config.get('name', 'Unknown')
            dataset_name            = config.get('dataset', 'IEMOCAP')
            
            logger.info(f"Benchmarking model '{model_name}' on dataset '{dataset_name}'")
            
            # Run benchmark
            results                 = self.run_benchmark(dataset_name)
            results['model_name']   = model_name
            results['model_config'] = config
            
            comparison['models'].append(results)
        
        # Find best models
        if comparison['models']:
            comparison['best_overall']                     = max(comparison['models'], key = lambda x: x['accuracy'])['model_name']
            comparison['best_per_metric']['fastest']       = min(comparison['models'], key = lambda x: x['avg_inference_time_ms'])['model_name']
            comparison['best_per_metric']['most_accurate'] = comparison['best_overall']
            
            # Create summary statistics
            accuracies                                     = [m['accuracy'] for m in comparison['models']]
            inference_times                                = [m['avg_inference_time_ms'] for m in comparison['models']]
            
            comparison['summary']                          = {'num_models'            : len(comparison['models']),
                                                              'avg_accuracy'          : float(np.mean(accuracies)),
                                                              'std_accuracy'          : float(np.std(accuracies)),
                                                              'avg_inference_time_ms' : float(np.mean(inference_times)),
                                                              'std_inference_time_ms' : float(np.std(inference_times)),
                                                             }
        
        return comparison
    

    def export_results(self, results: Dict[str, Any], output_dir: Path) -> Path:
        """
        Export benchmark results to file
        
        Arguments:
        ----------
            results     { dict } : Benchmark results

            output_dir  { Path } : Output directory
        
        Returns:
        --------
                 { Path }        : Path to exported file
        """
        output_dir.mkdir(parents = True, exist_ok = True)
        
        timestamp    = time.strftime("%Y%m%d_%H%M%S")
        dataset_name = results.get('dataset', 'unknown')
        output_file  = output_dir / f"benchmark_{dataset_name}_{timestamp}.json"
        
        
        with open(output_file, 'w') as f:
            json.dump(obj    = results, 
                      fp     = f, 
                      indent = 4,
                     )
        
        logger.info(f"Results exported to {output_file}")
        
        return output_file


# Convenience function
def run_benchmark(dataset_name: str = 'IEMOCAP') -> Dict[str, Any]:
    """
    Run benchmark and return results
    """
    runner = BenchmarkRunner()
    
    return runner.run_benchmark(dataset_name)
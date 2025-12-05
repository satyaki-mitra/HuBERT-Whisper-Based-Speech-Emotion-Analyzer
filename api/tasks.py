# DEPENDENCIES
import time
from celery import Task
from celery import Celery
import concurrent.futures
from celery.result import AsyncResult
from utils.logging_util import setup_logger
from services.exporters import ExportService
from config.settings import CELERY_BROKER_URL
from config.settings import CELERY_RESULT_BACKEND
from config.settings import CELERY_TASK_TIME_LIMIT
from services.audio_analyzer import get_audio_analyzer
        

# SETUP LOGGING
logger = setup_logger(__name__)


# Initialize Celery
celery_app = Celery('emotivoice',
                    broker  = CELERY_BROKER_URL,
                    backend = CELERY_RESULT_BACKEND,
                   )


celery_app.conf.update(task_serializer            = 'json',
                       accept_content             = ['json'],
                       result_serializer          = 'json',
                       timezone                   = 'UTC',
                       enable_utc                 = True,
                       task_track_started         = True,
                       task_time_limit            = CELERY_TASK_TIME_LIMIT,
                       worker_prefetch_multiplier = 1,
                       worker_max_tasks_per_child = 50,
                      )


# CUSTOM TASK BASE CLASS
class CallbackTask(Task):
    """
    Base task with callbacks
    """
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} completed successfully")
    

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")
    

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task {task_id} retrying: {exc}")



# ANALYSIS TASKS
@celery_app.task(base = CallbackTask, bind = True, max_retries = 3)
def analyze_audio_task(self, analysis_id: str, filepath: str, emotion_mode: str = 'both', language: str = None):
    """
    Async task for audio analysis
    """
    try:
        logger.info(f"Starting analysis: {analysis_id}")
        
        # Update task state
        self.update_state(state = 'PROCESSING',
                          meta  = {'progress': 0, 'message': 'Loading models...'},
                         )
        
        analyzer = get_audio_analyzer()
        
        # Update progress
        self.update_state(state = 'PROCESSING',
                          meta  = {'progress': 25, 'message': 'Analyzing audio...'},
                         )
        
        start_time                            = time.time()
        
        # Perform analysis
        result                                = analyzer.analyze_complete(audio_path   = filepath,
                                                                          emotion_mode = emotion_mode,
                                                                         )
        
        processing_time                       = time.time() - start_time
        result['metadata']['processing_time'] = processing_time
        result['metadata']['analysis_id']     = analysis_id
        
        # Update progress
        self.update_state(state = 'PROCESSING',
                          meta  = {'progress': 90, 'message': 'Formatting results...'},
                         )
        
        formatted_result                      = analyzer.format_results_for_display(result)
        
        logger.info(f"Analysis completed: {analysis_id} in {processing_time:.2f}s")
        
        return {'status'          : 'completed',
                'analysis_id'     : analysis_id,
                'result'          : formatted_result,
                'processing_time' : processing_time,
               }
        
    except Exception as e:
        logger.error(f"Analysis task failed: {repr(e)}")
        

        # Retry on certain errors
        if isinstance(e, (ConnectionError, TimeoutError)):
            raise self.retry(exc = e, countdown = 5)
        
        return {'status'      : 'failed',
                'analysis_id' : analysis_id,
                'error'       : str(e),
               }


@celery_app.task(base = CallbackTask, bind = True, max_retries = 3)
def batch_analyze_task(self, batch_id: str, filepaths: list, emotion_mode: str = 'both', parallel: bool = True):
    """
    Async task for batch analysis
    """
    try:
        logger.info(f"Starting batch analysis: {batch_id}, {len(filepaths)} files")
        
        analyzer    = get_audio_analyzer()
        results     = list()
        
        total_files = len(filepaths)
        
        if parallel:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers = 4) as executor:
                futures = {executor.submit(analyzer.analyze_complete, filepath, emotion_mode): filepath for filepath in filepaths}
                
                for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    filepath = futures[future]
                    
                    try:
                        result    = future.result()
                        formatted = analyzer.format_results_for_display(result)
                        
                        results.append({'filepath' : filepath,
                                        'result'   : formatted,
                                        'status'   : 'success',
                                      })

                    except Exception as e:
                        logger.error(f"Failed to analyze {filepath}: {e}")
                        results.append({'filepath' : filepath,
                                        'error'    : str(e),
                                        'status'   : 'failed',
                                      })
                    
                    # Update progress
                    progress = int(((idx + 1) / total_files) * 100)
                    
                    self.update_state(state = 'PROCESSING',
                                      meta  = {'progress'  : progress,
                                               'message'   : f'Processed {idx + 1}/{total_files} files',
                                               'completed' : idx + 1,
                                               'total'     : total_files,
                                              }
                                     )
        
        else:
            # Sequential processing
            for idx, filepath in enumerate(filepaths):
                
                try:
                    result    = analyzer.analyze_complete(filepath, emotion_mode)
                    formatted = analyzer.format_results_for_display(result)
                    
                    results.append({'filepath' : filepath,
                                    'result'   : formatted,
                                    'status'   : 'success',
                                  })

                except Exception as e:
                    logger.error(f"Failed to analyze {filepath}: {e}")
                    results.append({'filepath' : filepath,
                                    'error'    : str(e),
                                    'status'   : 'failed',
                                  })
                
                # Update progress
                progress = int(((idx + 1) / total_files) * 100)
                
                self.update_state(state = 'PROCESSING',
                                  meta  = {'progress'  : progress,
                                           'message'   : f'Processed {idx + 1}/{total_files} files',
                                           'completed' : idx + 1,
                                           'total'     : total_files,
                                          }
                                 )
        
        logger.info(f"Batch analysis completed: {batch_id}")
        
        return {'status'      : 'completed',
                'batch_id'    : batch_id,
                'results'     : results,
                'total_files' : total_files,
                'successful'  : sum(1 for r in results if r['status'] == 'success'),
                'failed'      : sum(1 for r in results if r['status'] == 'failed'),
               }
        
    except Exception as e:
        logger.error(f"Batch analysis task failed: {repr(e)}")
        return {'status'   : 'failed',
                'batch_id' : batch_id,
                'error'    : str(e),
               }



# EXPORT TASKS
@celery_app.task(base = CallbackTask, bind = True)
def export_results_task(self, analysis_id: str, format: str = 'json', include_visualizations: bool = False):
    """
    Async task for exporting results
    """
    try:
        logger.info(f"Starting export: {analysis_id} as {format}")
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'message': 'Preparing export...'}
        )
        
        export_service = ExportService()
        
        # Get analysis result (mock - in production, retrieve from database)
        # For now, return export path
        
        self.update_state(
            state='PROCESSING',
            meta={'progress': 50, 'message': f'Generating {format} export...'}
        )
        
        export_path = export_service.export(
            analysis_id=analysis_id,
            format=format,
            include_visualizations=include_visualizations
        )
        
        logger.info(f"Export completed: {export_path}")
        
        return {
            'status': 'completed',
            'export_id': analysis_id,
            'export_path': str(export_path),
            'format': format
        }
        
    except Exception as e:
        logger.error(f"Export task failed: {repr(e)}")
        return {
            'status': 'failed',
            'export_id': analysis_id,
            'error': str(e)
        }

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_task_status(task_id: str) -> dict:
    """Get status of a Celery task"""
    try:
        task_result = AsyncResult(task_id, app=celery_app)
        
        if task_result.state == 'PENDING':
            return {
                'status': 'pending',
                'progress': 0,
                'message': 'Task is pending...'
            }
        
        elif task_result.state == 'PROCESSING':
            return {
                'status': 'processing',
                'progress': task_result.info.get('progress', 0),
                'message': task_result.info.get('message', 'Processing...')
            }
        
        elif task_result.state == 'SUCCESS':
            result = task_result.result
            return {
                'status': 'completed',
                'progress': 100,
                'message': 'Task completed',
                'result': result
            }
        
        elif task_result.state == 'FAILURE':
            return {
                'status': 'failed',
                'progress': 0,
                'message': 'Task failed',
                'error': str(task_result.info)
            }
        
        else:
            return {
                'status': task_result.state.lower(),
                'progress': 0,
                'message': f'Task state: {task_result.state}'
            }
    
    except Exception as e:
        logger.error(f"Failed to get task status: {repr(e)}")
        return None
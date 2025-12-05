# DEPENDENCIES
import csv
import json
import zipfile
from typing import Any
from typing import Dict
from pathlib import Path
from datetime import datetime
from reportlab.platypus import Spacer
from config.settings import EXPORTS_DIR
from reportlab.platypus import Paragraph
from reportlab.lib.pagesizes import letter
from utils.logging_util import setup_logger
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet


# SETUP LOGGING
logger = setup_logger(__name__)


class ExportService:
    """
    Service for exporting analysis results
    """
    def __init__(self):
        EXPORTS_DIR.mkdir(parents  = True, 
                          exist_ok = True,
                         )

    
    def export(self, analysis_id: str, format: str, result_data: Dict[str, Any] = None, include_visualizations: bool = False) -> Path:
        """
        Export analysis results with optional visualizations
    
        Arguments:
        ----------
            include_visualizations { bool } : If True, bundle visualization images with export
        """
        if include_visualizations:
            # Get visualization paths
            viz_dir   = EXPORTS_DIR / 'visualizations'
            viz_files = list(viz_dir.glob(f"{analysis_id}_*.png"))
            
            if (viz_files and (format == 'png')):
                # Create ZIP archive with all visualizations
                zip_path = EXPORTS_DIR / f"{analysis_id}_visualizations.zip"
                
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for viz_file in viz_files:
                        zipf.write(viz_file, viz_file.name)
                
                logger.info(f"Created visualization bundle: {zip_path}")
                
                return zip_path

        if (format == 'json'):
            return self.export_json(analysis_id = analysis_id, 
                                    result_data = result_data,
                                   )

        elif (format == 'csv'):
            return self.export_csv(analysis_id = analysis_id, 
                                   result_data = result_data,
                                  )

        elif (format == 'pdf'):
            return self.export_pdf(analysis_id = analysis_id, 
                                   result_data = result_data,
                                  )

        else:
            raise ValueError(f"Unsupported format: {format}")

    
    def export_json(self, analysis_id: str, result_data: Dict) -> Path:
        """
        Export as JSON
        """
        filepath    = EXPORTS_DIR / f"{analysis_id}.json"
        
        export_data = {'analysis_id' : analysis_id,
                       'exported_at' : datetime.utcnow().isoformat(),
                       'version'     : '1.0.0',
                       'results'     : result_data or {},
                      }
        
        with open(filepath, 'w', encoding = 'utf-8') as f:
            json.dump(obj          = export_data, 
                      fp           = f, 
                      indent       = 4, 
                      ensure_ascii = False,
                     )
        
        logger.info(f"Exported JSON: {filepath}")

        return filepath

    
    def export_csv(self, analysis_id: str, result_data: Dict) -> Path:
        """
        Export as CSV
        """
        filepath = EXPORTS_DIR / f"{analysis_id}.csv"
        
        with open(filepath, 'w', newline = '', encoding = 'utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['Analysis ID', analysis_id])
            writer.writerow(['Exported At', datetime.utcnow().isoformat()])
            writer.writerow([])
            
            # Transcription
            if result_data and 'transcription' in result_data:
                writer.writerow(['Transcription'])
                writer.writerow(['Text', result_data['transcription']])
                writer.writerow(['Language', result_data.get('language', 'N/A')])
                writer.writerow([])
            
            # Emotions
            if result_data and 'emotions' in result_data:
                if ('base' in result_data['emotions']):
                    writer.writerow(['Base Emotions'])
                    writer.writerow(['Emotion', 'Score'])
                    
                    for emotion in result_data['emotions']['base']:
                        writer.writerow([emotion.get('label', ''),
                                         emotion.get('percentage', '')
                                       ])
        
        logger.info(f"Exported CSV: {filepath}")

        return filepath
    

    def export_pdf(self, analysis_id: str, result_data: Dict) -> Path:
        """
        Export as PDF (requires reportlab)
        """
        try:
            filepath = EXPORTS_DIR / f"{analysis_id}.pdf"
            
            doc      = SimpleDocTemplate(str(filepath), pagesize=letter)
            story    = list()
            styles   = getSampleStyleSheet()
            
            # Title
            story.append(Paragraph("EmotiVoice Analysis Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            # Content
            story.append(Paragraph(f"Analysis ID: {analysis_id}", styles['Normal']))
            story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", styles['Normal']))
            
            if result_data:
                story.append(Spacer(1, 12))
                if 'transcription' in result_data:
                    story.append(Paragraph(f"Transcription: {result_data['transcription']}", styles['Normal']))
            
            doc.build(story)
            
            logger.info(f"Exported PDF: {filepath}")
            
            return filepath
            
        except ImportError:
            logger.error("reportlab not installed. Install with: pip install reportlab")
            # Fallback to JSON
            return self.export_json(analysis_id = analysis_id, 
                                    result_data = result_data,
                                   )
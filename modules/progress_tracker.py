"""
Progress Tracking Module
Provides real-time progress indicators for long-running operations
"""

import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ProgressTracker:
    """Tracks and reports progress of long-running operations"""
    
    def __init__(self, operation_name="Analysis"):
        self.operation_name = operation_name
        self.start_time = None
        self.steps = []
        self.current_step = 0
        self.total_steps = 0
    
    def start(self, total_steps=None):
        """Start tracking progress"""
        self.start_time = time.time()
        self.current_step = 0
        self.steps = []
        self.total_steps = total_steps or 1
        logger.info(f"Starting {self.operation_name}")
    
    def add_step(self, step_name, status="pending"):
        """Add a new step to track"""
        step = {
            'name': step_name,
            'status': status,  # pending, running, completed, error
            'start_time': None,
            'end_time': None,
            'duration': None,
            'message': ''
        }
        self.steps.append(step)
        self.total_steps = len(self.steps)
        return len(self.steps) - 1
    
    def start_step(self, step_index, message=''):
        """Mark a step as running"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'running'
            self.steps[step_index]['start_time'] = time.time()
            self.steps[step_index]['message'] = message
            self.current_step = step_index
            logger.info(f"  → {self.steps[step_index]['name']}: {message}")
    
    def complete_step(self, step_index, message=''):
        """Mark a step as completed"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'completed'
            self.steps[step_index]['end_time'] = time.time()
            self.steps[step_index]['duration'] = (
                self.steps[step_index]['end_time'] - self.steps[step_index]['start_time']
            )
            self.steps[step_index]['message'] = message
            self.current_step = step_index + 1
            logger.info(f"  ✓ {self.steps[step_index]['name']} ({self.steps[step_index]['duration']:.2f}s)")
    
    def error_step(self, step_index, error_message=''):
        """Mark a step as errored"""
        if 0 <= step_index < len(self.steps):
            self.steps[step_index]['status'] = 'error'
            self.steps[step_index]['message'] = error_message
            logger.error(f"  ✗ {self.steps[step_index]['name']}: {error_message}")
    
    def get_progress_percentage(self):
        """Get current progress as percentage"""
        if self.total_steps == 0:
            return 0
        completed = sum(1 for step in self.steps if step['status'] == 'completed')
        return int((completed / self.total_steps) * 100)
    
    def get_elapsed_time(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def get_estimated_time_remaining(self):
        """Estimate remaining time based on progress"""
        elapsed = self.get_elapsed_time()
        progress = self.get_progress_percentage()
        
        if progress == 0 or elapsed == 0:
            return None
        
        total_estimated = (elapsed / progress) * 100
        remaining = total_estimated - elapsed
        return max(0, remaining)
    
    def get_status_html(self):
        """Generate HTML status display"""
        progress_pct = self.get_progress_percentage()
        elapsed = self.get_elapsed_time()
        
        html_parts = [
            f'<div style="margin-bottom: 12px;">',
            f'  <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">',
            f'    <span style="font-weight: 600; font-size: 12px;">{self.operation_name}</span>',
            f'    <span style="font-size: 11px; color: #666;">{progress_pct}% • {elapsed:.1f}s</span>',
            f'  </div>',
            f'  <div style="background: #e0e0e0; border-radius: 4px; height: 6px; overflow: hidden;">',
            f'    <div style="background: linear-gradient(90deg, #4CAF50, #45a049); width: {progress_pct}%; height: 100%; transition: width 0.3s;"></div>',
            f'  </div>',
        ]
        
        # Add step details
        if self.steps:
            html_parts.append('<div style="margin-top: 8px; font-size: 11px;">')
            for i, step in enumerate(self.steps[-5:]):  # Show last 5 steps
                status_icon = {
                    'pending': '◯',
                    'running': '⊙',
                    'completed': '✓',
                    'error': '✗'
                }.get(step['status'], '?')
                
                color = {
                    'pending': '#ccc',
                    'running': '#FF9800',
                    'completed': '#4CAF50',
                    'error': '#f44336'
                }.get(step['status'], '#999')
                
                duration_text = ''
                if step.get('duration'):
                    duration_text = f" ({step['duration']:.2f}s)"

                html_parts.append(
                    f'<div style="margin: 2px 0; color: {color};">'
                    f'{status_icon} {step["name"]}'
                    f'{duration_text}'
                    f'</div>'
                )
            html_parts.append('</div>')
        
        html_parts.append('</div>')
        return ''.join(html_parts)
    
    def get_summary(self):
        """Get complete progress summary"""
        elapsed = self.get_elapsed_time()
        estimated_remaining = self.get_estimated_time_remaining()
        
        return {
            'operation': self.operation_name,
            'progress_percentage': self.get_progress_percentage(),
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'elapsed_seconds': elapsed,
            'estimated_remaining_seconds': estimated_remaining,
            'steps': self.steps,
            'timestamp': datetime.now().isoformat()
        }
    
    def finish(self, success=True):
        """Mark operation as finished"""
        elapsed = self.get_elapsed_time()
        if success:
            logger.info(f"Completed {self.operation_name} in {elapsed:.2f}s")
        else:
            logger.error(f"Failed {self.operation_name} after {elapsed:.2f}s")


# Global progress trackers for different operations
_progress_trackers = {}


def get_progress_tracker(operation_name):
    """Get or create a progress tracker"""
    if operation_name not in _progress_trackers:
        _progress_trackers[operation_name] = ProgressTracker(operation_name)
    return _progress_trackers[operation_name]


def clear_progress_tracker(operation_name):
    """Clear a progress tracker"""
    if operation_name in _progress_trackers:
        del _progress_trackers[operation_name]


# Common operation names
OPERATION_FILE_UPLOAD = "File Upload & Validation"
OPERATION_ENRICHMENT = "Pathway Enrichment Analysis"
OPERATION_PLOT_GENERATION = "Visualization Generation"
OPERATION_PCA = "PCA Analysis"
OPERATION_NETWORK = "Network Analysis"
OPERATION_EXPORT = "Data Export"


def create_upload_progress_tracker():
    """Create and setup upload progress tracker"""
    tracker = get_progress_tracker(OPERATION_FILE_UPLOAD)
    tracker.start(total_steps=3)
    tracker.add_step("Parsing CSV file", "pending")
    tracker.add_step("Validating data", "pending")
    tracker.add_step("Preparing analysis", "pending")
    return tracker


def create_enrichment_progress_tracker():
    """Create and setup enrichment analysis progress tracker"""
    tracker = get_progress_tracker(OPERATION_ENRICHMENT)
    tracker.start(total_steps=4)
    tracker.add_step("Querying Enrichr API", "pending")
    tracker.add_step("Processing results", "pending")
    tracker.add_step("Computing significance", "pending")
    tracker.add_step("Finalizing enrichment", "pending")
    return tracker


def create_visualization_progress_tracker():
    """Create and setup visualization progress tracker"""
    tracker = get_progress_tracker(OPERATION_PLOT_GENERATION)
    tracker.start(total_steps=5)
    tracker.add_step("Generating volcano plot", "pending")
    tracker.add_step("Creating heatmap", "pending")
    tracker.add_step("Building pathway bubbles", "pending")
    tracker.add_step("Computing PCA", "pending")
    tracker.add_step("Rendering network", "pending")
    return tracker

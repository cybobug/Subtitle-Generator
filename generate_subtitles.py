import sys
import os
from pathlib import Path
import traceback
import tempfile
import logging
import json
import time
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import whisper
import ffmpeg
import torch
from tqdm import tqdm
from datetime import timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QComboBox, QProgressBar, 
    QTextEdit, QMessageBox, QCheckBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QPalette, QColor

class EnhancedVideoProcessor:
    def __init__(
        self, 
        batch_size: int = 5, 
        max_workers: int = None,
        output_dir: str = None,
        log_file: str = None,
        precision: str = 'float32',
        language: str = 'en'  # Added language parameter
    ):
        """
        Enhanced video processor with more configuration options.
        
        Args:
            batch_size (int): Number of videos to process in parallel
            max_workers (int): Maximum number of worker processes
            output_dir (str): Custom output directory for subtitles
            log_file (str): Path to log file
            precision (str): Computation precision ('float16' or 'float32')
            language (str): Language for transcription
        """
        self.batch_size = batch_size
        self.max_workers = max_workers or os.cpu_count()
        self.whisper_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.precision = torch.float16 if precision == 'float16' else torch.float32
        self.language = language  # Store language setting
        self.is_cancelled = False  # Flag for cancellation

        # Create output directories
        self.temp_dir = Path(tempfile.mkdtemp(prefix='temp_audio_'))
        self.output_dir = Path(output_dir or 'subtitle_output')
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Enhanced logging
        self.setup_logging(log_file)
        self.error_log_path = self.output_dir / 'processing_errors.json'
        self.error_log = []

    def cancel_processing(self):
        """Set cancellation flag to stop processing."""
        self.is_cancelled = True

    def reset_cancellation(self):
        """Reset cancellation flag."""
        self.is_cancelled = False

    # ... [Previous methods remain the same] ...

    def process_video_with_retry(
        self, 
        video_path: str, 
        max_retries: int = 2, 
        retry_delay: int = 1,
        progress_callback = None  # Added progress callback
    ):
        """
        Process a single video with retry mechanism and progress tracking.
        
        Args:
            video_path (str): Path to video file
            max_retries (int): Number of retry attempts
            retry_delay (int): Delay between retries in seconds
            progress_callback (callable): Optional callback to report progress
        
        Returns:
            Optional path to generated SRT or None if failed
        """
        if self.is_cancelled:
            return None

        for attempt in range(max_retries + 1):
            try:
                # Progress update
                if progress_callback:
                    progress_callback(f"Processing {Path(video_path).name}")

                # Process video
                audio_path = self.extract_audio(video_path)
                
                if self.is_cancelled:
                    return None

                segments = self.transcribe_audio(audio_path)
                
                if self.is_cancelled:
                    return None

                output_path = self.output_dir / f"{Path(video_path).stem}.srt"
                
                self.write_srt(segments, str(output_path))
                
                # Optional: Clean up audio file
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                
                return str(output_path)
            
            except Exception as e:
                if self.is_cancelled:
                    return None

                if attempt < max_retries:
                    self.logger.warning(f"Retry attempt {attempt + 1} for {video_path}")
                    time.sleep(retry_delay)
                else:
                    error_details = {
                        'video_path': video_path,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                        'traceback': traceback.format_exc()
                    }
                    self.error_log.append(error_details)
                    self.logger.error(f"Failed to process {video_path}: {str(e)}")
                    return None

    def transcribe_audio(self, audio_path: str):
        """Transcribe audio using the loaded Whisper model with language support."""
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                temperature=0.0,  # Disable sampling for faster, more consistent results
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
                language=self.language  # Use specified language
            )
            return result['segments']
        except Exception as e:
            self.logger.error(f"Error transcribing {audio_path}: {str(e)}")
            raise
    def extract_audio(self, video_path: str) -> str:
        """Extract audio from video using ffmpeg.Args:video_path (str): Path to the input video file Returns:str: Path to the extracted audio file"""
        # Generate a unique audio filename based on the video
        audio_path = self.temp_dir / f"{Path(video_path).stem}_audio.wav"
        
        try:
            # Use ffmpeg to extract audio
            stream = ffmpeg.input(video_path)
            stream = ffmpeg.output(
                stream, 
                str(audio_path), 
                acodec='pcm_s16le',  # 16-bit PCM audio codec
                ac=1,  # Convert to mono
                ar='16k'  # Resample to 16kHz
            )
            
            # Run ffmpeg with error capturing
            ffmpeg.run(
                stream, 
                capture_stdout=True, 
                capture_stderr=True, 
                overwrite_output=True  # Overwrite existing audio file
            )
            
            # Log successful audio extraction
            self.logger.info(f"Audio extracted from {video_path} to {audio_path}")
            
            return str(audio_path)
        
        except ffmpeg.Error as e:
            # Log FFmpeg specific errors
            error_msg = f"FFmpeg error processing {video_path}: {e.stderr.decode()}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        except Exception as e:
            # Log any other unexpected errors
            error_msg = f"Error extracting audio from {video_path}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    def setup_logging(self, log_file: str = None):
        """Configure advanced logging for the video processor.Args:log_file (str, optional): Path to the log file. If None, a default log file will be created in the output directory."""
        # Determine log file path
        if log_file is None:
            log_file = self.output_dir / 'video_processing.log'
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(processName)s - %(threadName)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler(log_file, mode='a')  # Append to log file
            ]
        )
        
        # Create logger instance
        self.logger = logging.getLogger(__name__)
    def write_srt(self, segments, output_path):
        """Convert Whisper transcription segments to SRT subtitle format.Args:segments (list): Transcription segments from Whisperoutput_path (str): Path to save the SRT file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as srt_file:
                for index, segment in enumerate(segments, 1):
                    # Convert start and end times to SRT time format
                    start_time = str(timedelta(seconds=segment['start'])).replace('.', ',')[:12]
                    end_time = str(timedelta(seconds=segment['end'])).replace('.', ',')[:12]
                    
                    # Write SRT entry
                    srt_file.write(f"{index}\n")
                    srt_file.write(f"{start_time} --> {end_time}\n")
                    srt_file.write(f"{segment['text'].strip()}\n\n")
            
            # Log successful SRT generation
            self.logger.info(f"SRT file generated: {output_path}")
        
        except Exception as e:
            # Log error during SRT generation
            error_msg = f"Error writing SRT file {output_path}: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    def process_batch(self, video_paths, progress_callback=None):
        """Enhanced batch processing with multiprocessing and progress tracking."""
        self.reset_cancellation()  # Reset cancellation flag
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Create futures for each video
            futures = {
                executor.submit(
                    self.process_video_with_retry, 
                    video_path, 
                    progress_callback=progress_callback
                ): video_path for video_path in video_paths
            }
            
            # Track progress and results
            for future in as_completed(futures):
                if self.is_cancelled:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break
                
                video_path = futures[future]
                try:
                    result = future.result()
                    results.append((video_path, result))
                except Exception as e:
                    self.logger.error(f"Error processing {video_path}: {str(e)}")
                    results.append((video_path, None))
        
        return results

class SubtitleProcessorThread(QThread):
    """Background thread for processing videos with enhanced progress tracking"""
    progress_update = pyqtSignal(str)  # Changed to emit detailed progress messages
    processing_progress = pyqtSignal(int, int)  # Total videos, processed videos
    processing_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)

    def __init__(self, processor, input_path, model_name, language='en'):
        self.whisper_model = None
        super().__init__()
        self.processor = processor
        self.input_path = input_path
        self.model_name = model_name
        self.language = language

    def run(self):
        try:
            # Load Whisper model before processing
            if not self.processor.whisper_model:
                self.processor.whisper_model = whisper.load_model(
                    self.model_name, 
                    device=self.processor.device,
                    download_root=None  # Optional: specify a download path if needed
                )
            # Determine if input is a file or directory
            input_path = Path(self.input_path)
            
            # Set language for processor
            self.processor.language = self.language
            
            def progress_callback(message):
                self.progress_update.emit(message)
            
            if input_path.is_dir():
                # Get video files
                video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
                video_files = [
                    str(f) for f in Path(input_path).glob('**/*') 
                    if f.suffix.lower() in video_extensions
                ]
                
                total_videos = len(video_files)
                processed_videos = 0
                
                # Process in batches
                results = []
                for i in range(0, total_videos, self.processor.batch_size):
                    if self.processor.is_cancelled:
                        break
                    
                    batch = video_files[i:i + self.processor.batch_size]
                    batch_results = self.processor.process_batch(
                        batch, 
                        progress_callback=progress_callback
                    )
                    
                    results.extend(batch_results)
                    processed_videos += len(batch)
                    self.processing_progress.emit(total_videos, processed_videos)
                
                self.processing_complete.emit(results)
            
            elif input_path.is_file():
                def single_file_progress(message):
                    progress_callback(message)
                    self.processing_progress.emit(1, 1)
                
                srt_path = self.processor.process_video_with_retry(
                    str(input_path), 
                    progress_callback=single_file_progress
                )
                results = [(str(input_path), srt_path)]
                self.processing_complete.emit(results)
            
            else:
                self.error_occurred.emit("Invalid input path")
        
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)

class SubtitleGeneratorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Subtitle Generator")
        self.setGeometry(100, 100, 700, 600)
        self.setup_ui()
    def select_input(self):
        """Open file/directory selection dialog for input videos. Allows selecting either a single video file or a directory."""
        # Create a dialog to choose between file and directory
        dialog_choice = QMessageBox.question(
            self, 
            "Select Input", 
            "Do you want to select a single video file or a directory?",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )

        # Video file formats
        video_formats = "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)"

        if dialog_choice == QMessageBox.Yes:
            # Select single video file
            input_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Select Input Video", 
                "", 
                video_formats
            )
        elif dialog_choice == QMessageBox.No:
            # Select directory
            input_path = QFileDialog.getExistingDirectory(
                self, 
                "Select Input Directory"
            )
        else:
            # User cancelled
            return

        if input_path:
            self.input_path_label.setText(input_path)

    def setup_ui(self):
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Input selection section
        input_layout = QHBoxLayout()
        self.input_path_label = QLabel("No file/directory selected")
        input_select_btn = QPushButton("Select Input")
        input_select_btn.clicked.connect(self.select_input)
        input_layout.addWidget(self.input_path_label)
        input_layout.addWidget(input_select_btn)
        main_layout.addLayout(input_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Whisper Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        main_layout.addLayout(model_layout)

        # Advanced options
        advanced_layout = QHBoxLayout()
        self.low_precision_check = QCheckBox("Use Low Precision (float16)")
        advanced_layout.addWidget(self.low_precision_check)
        main_layout.addLayout(advanced_layout)

        # Language selection
        language_layout = QHBoxLayout()
        language_label = QLabel("Transcription Language:")
        self.language_combo = QComboBox()
        # Common language codes, can be expanded
        languages = [
            'en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar'
        ]
        self.language_combo.addItems(languages)
        self.language_combo.setCurrentText('en')
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        main_layout.addLayout(language_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Processing button
        self.process_btn = QPushButton("Generate Subtitles")
        self.process_btn.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_btn)

        # Cancel button
        self.cancel_btn = QPushButton("Cancel Processing")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_processing)
        main_layout.addWidget(self.cancel_btn)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        main_layout.addWidget(self.log_display)

        # Set main layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Dark mode styling
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 12px;
            }
            QLabel, QCheckBox {
                color: #ffffff;
            }
            QPushButton {
                background-color: #4a4a4a;
                color: #ffffff;
                border: 1px solid #666;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #e0e0e0;
                font-family: 'Courier New', monospace;
            }
            QComboBox {
                background-color: #4a4a4a;
                color: #ffffff;
            }
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)

        # Initialize processor
        self.processor = None
        self.processing_thread = None

    def start_processing(self):
        # Validate input
        input_path = self.input_path_label.text()
        if input_path == "No file/directory selected":
            QMessageBox.warning(self, "Error", "Please select an input file or directory")
            return

        # Disable processing button, enable cancel button
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.log_display.clear()

        # Configure processor
        self.processor = EnhancedVideoProcessor(
            precision='float16' if self.low_precision_check.isChecked() else 'float32',
            output_dir='subtitle_output'
        )

        # Start processing thread
        model_name = self.model_combo.currentText()
        language = self.language_combo.currentText()
        
        self.processing_thread = SubtitleProcessorThread(
            self.processor, input_path, model_name, language
        )
        self.processing_thread.processing_complete.connect(self.on_processing_complete)
        self.processing_thread.error_occurred.connect(self.on_processing_error)
        self.processing_thread.progress_update.connect(self.update_log)
        self.processing_thread.processing_progress.connect(self.update_progress)
        
        self.processing_thread.start()

        # Log start of processing
        self.log_display.append(f"Started processing: {input_path}")
        self.log_display.append(f"Using model: {model_name}")
        self.log_display.append(f"Language: {language}")

    def cancel_processing(self):
        """Cancel ongoing video processing"""
        if self.processor:
            self.processor.cancel_processing()
        
        # Disable cancel button, re-enable process button
        self.cancel_btn.setEnabled(False)
        self.process_btn.setEnabled(True)
        
        self.log_display.append("\n--- PROCESSING CANCELLED BY USER ---")

    def update_log(self, message):
        """Update log display with real-time messages"""
        self.log_display.append(message)

    def update_progress(self, total_videos, processed_videos):
        """Update progress bar"""
        progress = int((processed_videos / total_videos) * 100) if total_videos > 0 else 0
        self.progress_bar.setValue(progress)
        self.progress_bar.setFormat(f"{processed_videos}/{total_videos} videos processed")

    def on_processing_complete(self, results):
        # ... [Previous implementation remains the same] ...
        
        # Reset UI state
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        self.process_btn.setEnabled(True)
        
        # Display results
        successful = sum(1 for _, srt in results if srt is not None)
        self.log_display.append("\nProcessing Summary:")
        self.log_display.append(f"Total videos processed: {len(results)}")
        self.log_display.append(f"Successful: {successful}")
        self.log_display.append(f"Failed: {len(results) - successful}")

        # Detailed results
        self.log_display.append("\nGenerated Subtitles:")
        for video, srt in results:
            if srt:
                self.log_display.append(f"{video} → {srt}")

        # Show completion message
        QMessageBox.information(self, "Processing Complete", 
                                f"Processed {len(results)} videos\n{successful} successful")

        # Open output directory
        if results:
            output_dir = os.path.dirname(results[0][1]) if results[0][1] else None
            if output_dir and os.path.exists(output_dir):
                reply = QMessageBox.question(
                    self, 
                    "Open Output Directory", 
                    "Do you want to open the output directory?", 
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    import subprocess
                    import platform

                    try:
                        if platform.system() == "Windows":
                            os.startfile(output_dir)
                        elif platform.system() == "Darwin":  # macOS
                            subprocess.Popen(["open", output_dir])
                        else:  # linux variants
                            subprocess.Popen(["xdg-open", output_dir])
                    except Exception as e:
                        QMessageBox.warning(self, "Error", f"Could not open directory: {str(e)}")

    def on_processing_error(self, error_msg):
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.log_display.append(f"Error: {error_msg}")

def main():
    app = QApplication(sys.argv)
    window = SubtitleGeneratorApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
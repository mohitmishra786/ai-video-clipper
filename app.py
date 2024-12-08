import os  
import subprocess  
import whisper  
import yt_dlp  
import torch  
import re  
import logging  
from flask import Flask, request, render_template, jsonify, send_from_directory  
from nltk.tokenize import sent_tokenize  
import nltk  
from urllib.parse import urlparse, parse_qs  
import datetime  
import shutil  
import time  

nltk.download('punkt')  

# Initialize Flask app  
app = Flask(__name__)  

# Directories for logs, clips, videos, and audio  
LOGS_DIR = "logs"  
CLIPS_DIR = "clips"  
VIDEOS_DIR = "videos"  
AUDIO_DIR = "audio"  

# Create directories if they don't exist  
for directory in [LOGS_DIR, CLIPS_DIR, VIDEOS_DIR, AUDIO_DIR]:  
    os.makedirs(directory, exist_ok=True)  

# Load Whisper model  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = whisper.load_model("base", device=device)  

# Define technical keywords  
technical_keywords = {  
    'gpu_computing': [  
        'gpu', 'cuda', 'parallel computing', 'graphics processing', 'shader',  
        'compute shader', 'opencl', 'vulkan', 'graphics pipeline', 'rendering',  
        'texture', 'buffer', 'compute unit', 'thread block', 'warp', 'kernel'  
    ],  
    'computer_architecture': [  
        'instruction set', 'isa', 'pipeline', 'branch prediction', 'cache hierarchy',  
        'memory hierarchy', 'von neumann', 'harvard architecture', 'superscalar',  
        'out of order execution', 'speculative execution', 'microarchitecture',  
        'fetch', 'decode', 'execute', 'writeback', 'forwarding', 'hazard',  
        'stall', 'microcode', 'microinstruction'  
    ],  
    'assembly_programming': [  
        'assembly', 'assembler', 'mnemonic', 'opcode', 'operand', 'risc',  
        'cisc', 'arm assembly', 'x86 assembly', 'riscv', 'risc-v', 'instruction set',  
        'register file', 'immediate value', 'addressing mode', 'branch instruction',  
        'jump instruction', 'load store', 'arithmetic instruction', 'logical instruction'  
    ],  
    'low_level': [  
        'memory', 'pointer', 'address', 'register', 'cache', 'assembly',  
        'instruction', 'binary', 'bit', 'byte', 'stack', 'heap', 'allocation',  
        'memory mapping', 'virtual memory', 'physical memory', 'page table',  
        'segmentation', 'protection ring', 'privilege level'  
    ],  
    'system_programming': [  
        'operating system', 'driver', 'interrupt', 'system call', 'process',  
        'thread', 'scheduling', 'synchronization', 'mutex', 'semaphore',  
        'context switch', 'privilege level', 'kernel mode', 'user mode'  
    ]  
}  

# Flatten the list of keywords  
all_keywords = set()  
for keywords in technical_keywords.values():  
    all_keywords.update(keywords)  

def setup_logger(youtube_id):  
    """Set up logging configuration for a specific YouTube ID."""  
    log_file = os.path.join(LOGS_DIR, f"{youtube_id}.log")  
    logger = logging.getLogger(youtube_id)  

    if not logger.handlers:  
        logger.setLevel(logging.DEBUG)  
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  

        # File handler  
        file_handler = logging.FileHandler(log_file)  
        file_handler.setFormatter(formatter)  
        logger.addHandler(file_handler)  

        # Console handler  
        console_handler = logging.StreamHandler()  
        console_handler.setFormatter(formatter)  
        logger.addHandler(console_handler)  

    return logger  

def extract_youtube_id(url):  
    """Extract YouTube video ID from various URL formats."""  
    try:  
        parsed_url = urlparse(url)  

        if parsed_url.hostname == 'youtu.be':  
            return parsed_url.path[1:]  

        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):  
            if parsed_url.path == '/watch':  
                return parse_qs(parsed_url.query)['v'][0]  
            elif parsed_url.path.startswith('/embed/'):  
                return parsed_url.path.split('/')[2]  
            elif parsed_url.path.startswith('/v/'):  
                return parsed_url.path.split('/')[2]  
    except Exception:  
        return None  

    return None  

def download_youtube_video(url, youtube_id, logger):  
    """Download YouTube video with enhanced error handling."""  
    logger.info(f"Starting download for YouTube URL: {url}")  
    video_path = os.path.join(VIDEOS_DIR, f"{youtube_id}.mp4")  

    ydl_opts = {  
        'format': 'best[ext=mp4]/best',  
        'outtmpl': video_path,  
        'quiet': True,  
        'no_warnings': True,  
        'extract_audio': False,  
        'postprocessors': [],  
        'merge_output_format': 'mp4',  
        'ignoreerrors': False,  
        'noplaylist': True,  
    }  

    try:  
        # Remove existing file if it exists  
        if os.path.exists(video_path):  
            os.remove(video_path)  
            logger.info(f"Removed existing video file: {video_path}")  

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:  
            try:  
                ydl.download([url])  
            except Exception as e:  
                logger.error(f"yt-dlp download failed: {str(e)}")  
                raise Exception(f"Failed to download video: {str(e)}")  

        if not os.path.exists(video_path):  
            raise Exception(f"Video file not created at expected path: {video_path}")  

        # Verify file size  
        file_size = os.path.getsize(video_path)  
        if file_size == 0:  
            raise Exception("Downloaded video file is empty")  

        logger.info(f"Video downloaded successfully: {video_path} ({file_size} bytes)")  
        return video_path  

    except Exception as e:  
        logger.error(f"Error in download_youtube_video: {str(e)}")  
        if os.path.exists(video_path):  
            os.remove(video_path)  
        raise  

def extract_audio(video_path, youtube_id, logger):  
    """Extract audio from video with enhanced error handling."""  
    logger.info(f"Starting audio extraction from: {video_path}")  
    audio_path = os.path.join(AUDIO_DIR, f"{youtube_id}.wav")  

    try:  
        # Remove existing audio file if it exists  
        if os.path.exists(audio_path):  
            os.remove(audio_path)  

        command = [  
            'ffmpeg',  
            '-y',  
            '-i', video_path,  
            '-vn',  
            '-acodec', 'pcm_s16le',  
            '-ar', '16000',  
            '-ac', '1',  
            audio_path  
        ]  

        process = subprocess.run(  
            command,  
            capture_output=True,  
            text=True,  
            check=True  
        )  

        if not os.path.exists(audio_path):  
            raise Exception("Audio file was not created")  

        # Verify file size  
        file_size = os.path.getsize(audio_path)  
        if file_size == 0:  
            raise Exception("Extracted audio file is empty")  

        logger.info(f"Audio extracted successfully: {audio_path} ({file_size} bytes)")  
        return audio_path  

    except subprocess.CalledProcessError as e:  
        logger.error(f"FFmpeg error: {e.stderr}")  
        if os.path.exists(audio_path):  
            os.remove(audio_path)  
        raise Exception(f"FFmpeg failed to extract audio: {e.stderr}")  
    except Exception as e:  
        logger.error(f"Error in extract_audio: {str(e)}")  
        if os.path.exists(audio_path):  
            os.remove(audio_path)  
        raise  
def transcribe_audio(audio_path, logger):  
    """Transcribe audio using Whisper with enhanced error handling."""  
    logger.info(f"Starting transcription of: {audio_path}")  

    try:  
        if not os.path.exists(audio_path):  
            raise Exception(f"Audio file not found: {audio_path}")  

        result = model.transcribe(  
            audio_path,  
            word_timestamps=True,  
            language=None,  # Auto-detect language  
            fp16=torch.cuda.is_available()  
        )  

        if not result or 'segments' not in result:  
            raise Exception("Transcription failed to produce valid output")  

        logger.info(f"Transcription completed successfully with {len(result['segments'])} segments")  
        return result  

    except Exception as e:  
        logger.error(f"Error in transcribe_audio: {str(e)}")  
        raise  

def process_transcription(result, logger):  
    """Process transcription to extract technical sentences and timestamps."""  
    logger.info("Processing transcription to extract technical sentences.")  
    segments = result['segments']  
    sentence_times = []  
    current_sentence = ''  
    current_start = None  
    sentence_endings = re.compile(r'[.!?]')  

    for segment in segments:  
        words = segment['words']  
        for word_info in words:  
            word = word_info['word']  
            word_start = word_info['start']  
            word_end = word_info['end']  

            if current_start is None:  
                current_start = word_start  

            current_sentence += word  

            if sentence_endings.search(word):  
                current_end = word_end  
                sentence_times.append((current_start, current_end, current_sentence.strip()))  
                current_sentence = ''  
                current_start = None  
            else:  
                current_sentence += ' '  

    if current_sentence:  
        current_end = word_end  
        sentence_times.append((current_start, current_end, current_sentence.strip()))  

    # Filter technical sentences  
    technical_sentences = [  
        (start, end, sentence)  
        for start, end, sentence in sentence_times  
        if any(keyword in sentence.lower() for keyword in all_keywords)  
    ]  

    logger.info(f"Extracted {len(technical_sentences)} technical sentences.")  
    return technical_sentences  

def extract_clips(video_path, technical_sentences, youtube_id, logger):  
    """Extract clips based on technical sentences with minimum 30s and maximum 75s duration."""  
    logger.info("Starting clip extraction.")  
    min_clip_duration = 30  # minimum clip duration in seconds  
    max_clip_duration = 75  # maximum clip duration in seconds  
    clips = []  
    current_clip_sentences = []  
    current_clip_start = None  
    current_clip_end = None  

    for start, end, sentence in technical_sentences:  
        if current_clip_start is None:  
            current_clip_start = start  
            current_clip_end = end  
            current_clip_sentences.append((start, end, sentence))  
        else:  
            # Check if adding this sentence would exceed max duration  
            if end - current_clip_start > max_clip_duration:  
                # If current clip meets minimum duration, save it  
                if current_clip_end - current_clip_start >= min_clip_duration:  
                    clips.append((current_clip_start, current_clip_end, current_clip_sentences.copy()))  
                # Start new clip with current sentence  
                current_clip_sentences = [(start, end, sentence)]  
                current_clip_start = start  
                current_clip_end = end  
            else:  
                # Add sentence to current clip  
                current_clip_sentences.append((start, end, sentence))  
                current_clip_end = end  

        # Check if current clip duration meets minimum and save if it does  
        if current_clip_end - current_clip_start >= min_clip_duration:  
            clips.append((current_clip_start, current_clip_end, current_clip_sentences.copy()))  
            current_clip_sentences = []  
            current_clip_start = None  
            current_clip_end = None  

    # Handle any remaining sentences  
    if current_clip_sentences and current_clip_end - current_clip_start >= min_clip_duration:  
        clips.append((current_clip_start, current_clip_end, current_clip_sentences))  

    # Create clips using FFmpeg  
    clip_folder = os.path.join(CLIPS_DIR, youtube_id)  
    os.makedirs(clip_folder, exist_ok=True)  

    clip_filenames = []  
    for idx, (start, end, sentences) in enumerate(clips):  
        # Ensure clip duration is within bounds  
        duration = end - start  
        if duration > max_clip_duration:  
            end = start + max_clip_duration  

        output_path = os.path.join(clip_folder, f"{youtube_id}_{idx+1}.mp4")  
        try:  
            command = [  
                'ffmpeg',  
                '-y',  # Overwrite output files  
                '-i', video_path,  
                '-ss', str(start),  
                '-to', str(end),  
                '-c:v', 'libx264',  # Use H.264 codec  
                '-c:a', 'aac',      # Use AAC audio codec  
                '-strict', 'experimental',  
                output_path  
            ]  

            subprocess.run(command, check=True, capture_output=True, text=True)  

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:  
                clip_filenames.append(output_path)  
                logger.info(f"Created clip {idx+1}: {output_path} ({end-start:.2f}s)")  
            else:  
                logger.error(f"Failed to create clip {idx+1}: Empty or missing file")  

        except subprocess.CalledProcessError as e:  
            logger.error(f"FFmpeg failed for clip {idx+1}: {e.stderr}")  
            if os.path.exists(output_path):  
                os.remove(output_path)  
            continue  

    logger.info(f"Successfully created {len(clip_filenames)} clips")  
    return clip_filenames  

def cleanup_files(files_list, logger):  
    """Clean up temporary files."""  
    for file_path in files_list:  
        try:  
            if os.path.exists(file_path):  
                os.remove(file_path)  
                logger.info(f"Cleaned up file: {file_path}")  
        except Exception as e:  
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")  

def validate_video_length(video_path, max_duration=3600):  
    """Validate video length is within acceptable limits."""  
    try:  
        cmd = [  
            'ffprobe',  
            '-v', 'error',  
            '-show_entries', 'format=duration',  
            '-of', 'default=noprint_wrappers=1:nokey=1',  
            video_path  
        ]  
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  
        duration = float(result.stdout)  

        return duration <= max_duration  
    except Exception:  
        return False  

def check_disk_space(required_space_mb=1000):  
    """Check if there's enough disk space available."""  
    try:  
        total, used, free = shutil.disk_usage(".")  
        free_mb = free // (2**20)  # Convert to MB  
        return free_mb >= required_space_mb  
    except Exception:  
        return False  

@app.route('/')  
def index():  
    """Render the main page."""  
    return render_template('index.html')  

@app.route('/process', methods=['POST'])  
def process():  
    """Process the YouTube video URL."""  
    start_time = time.time()  
    temp_files = []  

    try:  
        data = request.get_json()  
        url = data.get('url')  

        if not url:  
            return jsonify({'error': 'No URL provided'}), 400  

        youtube_id = extract_youtube_id(url)  
        if not youtube_id:  
            return jsonify({'error': 'Invalid YouTube URL'}), 400  

        logger = setup_logger(youtube_id)  
        logger.info(f"Processing started for YouTube ID: {youtube_id}")  

        # Check disk space  
        if not check_disk_space():  
            raise Exception("Insufficient disk space")  

        # Create a specific folder for this video's clips  
        video_clip_dir = os.path.join(CLIPS_DIR, youtube_id)  
        os.makedirs(video_clip_dir, exist_ok=True)  

        # Process the video  
        video_path = download_youtube_video(url, youtube_id, logger)  
        temp_files.append(video_path)  

        # Validate video length  
        if not validate_video_length(video_path):  
            raise Exception("Video is too long (max 1 hour)")  

        audio_path = extract_audio(video_path, youtube_id, logger)  
        temp_files.append(audio_path)  

        transcription_result = transcribe_audio(audio_path, logger)  
        technical_sentences = process_transcription(transcription_result, logger)  

        if not technical_sentences:  
            return jsonify({'error': 'No technical content found in video'}), 404  

        clip_filenames = extract_clips(video_path, technical_sentences, youtube_id, logger)  

        # Clean up temporary files  
        cleanup_files(temp_files, logger)  

        # Prepare response with clip information  
        clips_info = []  
        for clip_path in clip_filenames:  
            clip_name = os.path.basename(clip_path)  
            clips_info.append({  
                'filename': clip_name,  
                'url': f'/clips/{youtube_id}/{clip_name}'  
            })  

        processing_time = time.time() - start_time  
        logger.info(f"Processing completed in {processing_time:.2f} seconds")  

        return jsonify({  
            'status': 'success',  
            'youtube_id': youtube_id,  
            'clips': clips_info,  
            'total_clips': len(clips_info),  
            'processing_time': processing_time  
        })  

    except Exception as e:  
        logger.error(f"Error occurred: {str(e)}")  
        # Clean up any temporary files in case of error  
        cleanup_files(temp_files, logger)  
        return jsonify({'error': str(e)}), 500  

@app.route('/clips/<youtube_id>/<filename>')  
def serve_clip(youtube_id, filename):  
    """Serve a generated clip file."""  
    try:  
        clip_dir = os.path.join(CLIPS_DIR, youtube_id)  
        return send_from_directory(clip_dir, filename)  
    except Exception as e:  
        return jsonify({'error': str(e)}), 404  

@app.route('/health')  
def health_check():  
    """Health check endpoint."""  
    return jsonify({  
        'status': 'healthy',  
        'timestamp': datetime.datetime.now().isoformat(),  
        'disk_space_available': check_disk_space(),  
        'version': '1.0.0'  
    })  

if __name__ == '__main__':  
    app.run(debug=True)  
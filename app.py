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

nltk.download('punkt')  

# Initialize Flask app  
app = Flask(__name__)  

# Directories for logs and clips  
LOGS_DIR = "logs"  
CLIPS_DIR = "clips"  
os.makedirs(LOGS_DIR, exist_ok=True)  
os.makedirs(CLIPS_DIR, exist_ok=True)  

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


def extract_youtube_id(url):  
    """Extract the YouTube video ID from the URL."""  
    parsed_url = urlparse(url)  
    if parsed_url.hostname in ['youtu.be']:  
        return parsed_url.path[1:]  # Remove leading '/'  
    elif parsed_url.hostname in ['www.youtube.com', 'youtube.com']:  
        query_params = parse_qs(parsed_url.query)  
        return query_params.get('v', [None])[0]  
    return None  


def setup_logger(youtube_id):  
    """Set up a logger for the given YouTube ID."""  
    log_file = os.path.join(LOGS_DIR, f"{youtube_id}.log")  
    logger = logging.getLogger(youtube_id)  
    logger.setLevel(logging.DEBUG)  
    file_handler = logging.FileHandler(log_file)  
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  
    file_handler.setFormatter(formatter)  
    logger.addHandler(file_handler)  
    return logger  


def download_youtube_video(url, youtube_id, logger):  
    """Download YouTube video using yt-dlp."""  
    logger.info(f"Starting download for YouTube URL: {url}")  
    ydl_opts = {  
        'format': 'bestvideo+bestaudio/best',  
        'outtmpl': f"{youtube_id}.%(ext)s",  
        'quiet': True,  
    }  
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  
        info = ydl.extract_info(url, download=True)  
        video_path = ydl.prepare_filename(info)  
        logger.info(f"Video downloaded successfully: {video_path}")  
        return video_path  


def extract_audio(video_path, youtube_id, logger):  
    """Extract audio from the video."""  
    audio_path = f"{youtube_id}.wav"  
    logger.info(f"Extracting audio from video: {video_path}")  
    command = [  
        'ffmpeg',  
        '-i', video_path,  
        '-f', 'wav',  
        '-ar', '16000',  
        '-ac', '1',  
        audio_path  
    ]  
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)  
    logger.info(f"Audio extracted successfully: {audio_path}")  
    return audio_path  


def transcribe_audio(audio_path, logger):  
    """Transcribe audio using Whisper."""  
    logger.info(f"Starting transcription for audio: {audio_path}")  
    result = model.transcribe(audio_path, word_timestamps=True)  
    logger.info("Transcription completed successfully.")  
    return result  


def sentence_contains_keyword(sentence, keywords):  
    """Check if a sentence contains any of the technical keywords."""  
    sentence_lower = sentence.lower()  
    for keyword in keywords:  
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', sentence_lower):  
            return True  
    return False  


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
        if sentence_contains_keyword(sentence, all_keywords)  
    ]  

    logger.info(f"Extracted {len(technical_sentences)} technical sentences.")  
    return technical_sentences  


def extract_clips(video_path, technical_sentences, youtube_id, logger):  
    """Extract clips based on technical sentences."""  
    logger.info("Starting clip extraction.")  
    min_clip_duration = 30  
    max_clip_duration = 75  
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
            current_clip_end = end  
            current_clip_sentences.append((start, end, sentence))  

        clip_duration = current_clip_end - current_clip_start  

        if clip_duration >= min_clip_duration:  
            if clip_duration > max_clip_duration:  
                current_clip_end = current_clip_start + max_clip_duration  
                clip_duration = max_clip_duration  

            clips.append((current_clip_start, current_clip_end, current_clip_sentences.copy()))  
            current_clip_sentences = []  
            current_clip_start = None  
            current_clip_end = None  

    # Create a folder for the clips  
    clip_folder = os.path.join(CLIPS_DIR, youtube_id)  
    os.makedirs(clip_folder, exist_ok=True)  

    # Extract video clips using FFmpeg  
    clip_filenames = []  
    for idx, (start, end, _) in enumerate(clips):  
        output_path = os.path.join(clip_folder, f"{youtube_id}_{idx+1}.mp4")  
        command = [  
            'ffmpeg',  
            '-y',  
            '-i', video_path,  
            '-ss', str(start),  
            '-to', str(end),  
            '-c', 'copy',  
            output_path  
        ]  
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)  
        clip_filenames.append(output_path)  

    logger.info(f"Extracted {len(clip_filenames)} clips.")  
    return clip_filenames  


@app.route('/')  
def index():  
    """Render the main page."""  
    return render_template('index.html')  


@app.route('/process', methods=['POST'])  
def process():  
    """Process the YouTube video URL."""  
    data = request.get_json()  # Get JSON data from the request  
    url = data.get('url')  # Extract the URL from the JSON payload  
    if not url:  
        return jsonify({'error': 'No URL provided'}), 400  

    youtube_id = extract_youtube_id(url)  
    if not youtube_id:  
        return jsonify({'error': 'Invalid YouTube URL'}), 400  

    logger = setup_logger(youtube_id)  

    try:  
        logger.info(f"Processing started for YouTube ID: {youtube_id}")  

        # Download YouTube video  
        video_path = download_youtube_video(url, youtube_id, logger)  

        # Extract audio  
        audio_path = extract_audio(video_path, youtube_id, logger)  

        # Transcribe audio  
        transcription_result = transcribe_audio(audio_path, logger)  

        # Process transcription  
        technical_sentences = process_transcription(transcription_result, logger)  

        # Extract clips  
        clip_filenames = extract_clips(video_path, technical_sentences, youtube_id, logger)  

        logger.info("Processing completed successfully.")  
        return jsonify({'clips': [os.path.basename(clip) for clip in clip_filenames]})  
    except Exception as e:  
        logger.error(f"Error occurred: {str(e)}")  
        return jsonify({'error': str(e)}), 500  


@app.route('/clips/<youtube_id>/<filename>')  
def download_clip(youtube_id, filename):  
    """Serve a clip for download."""  
    clip_folder = os.path.join(CLIPS_DIR, youtube_id)  
    return send_from_directory(clip_folder, filename)  


if __name__ == '__main__':  
    app.run(debug=True)  
import os
import subprocess
import yt_dlp
import re
import logging
import zipfile
from flask import Flask, request, render_template, jsonify, send_from_directory, send_file
from nltk.tokenize import sent_tokenize
import nltk
from urllib.parse import urlparse, parse_qs
import datetime
import shutil
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import threading
import socket

# Suppress warnings from faster_whisper/numpy (divide by zero, overflow in matmul)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="faster_whisper")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

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

# Initialize PO token server on app startup (for production mode)
@app.before_request
def init_po_token():
    """Initialize PO token server on first request."""
    if not hasattr(app, 'po_token_initialized'):
        start_po_token_server()
        app.po_token_initialized = True

# Global whisper model (lazy loaded)
whisper_model = None
po_token_server = None

def get_whisper_model():
    """Lazy load faster-whisper model for CPU."""
    global whisper_model
    if whisper_model is None:
        from faster_whisper import WhisperModel
        # Use 'tiny' model with int8 for fastest CPU inference on Apple Silicon
        # tiny is ~4x faster than base with acceptable accuracy for clip detection
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return whisper_model


def start_po_token_server():
    """Start the PO token provider server in background."""
    global po_token_server
    if po_token_server is not None:
        return  # Already running
    
    try:
        import bgutil_ytdlp_pot_provider
        
        # Start PO token server on a free port
        port = 8050
        
        def run_server():
            try:
                # Run the PO token provider server
                subprocess.Popen(
                    ['python', '-m', 'bgutil_ytdlp_pot_provider', '--bind', f'127.0.0.1:{port}'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                print(f"Failed to start PO token server: {e}")
        
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        po_token_server = True
        time.sleep(2)  # Give it time to start
        print(f"PO Token server started on port {port}")
    except ImportError:
        print("bgutil-ytdlp-pot-provider not installed, skipping PO token server")
    except Exception as e:
        print(f"Failed to initialize PO token server: {e}")


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

    # Start PO token server if not already running (optional, helps with some videos)
    start_po_token_server()
    
    logger.info("Using tv_embedded client (no authentication required)")

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
        # Use multiple client fallbacks to bypass bot detection
        'extractor_args': {
            'youtube': {
                # Try tv_embedded first (no sign-in required), then android, then ios
                'player_client': ['tv_embedded', 'android', 'ios', 'mweb'],
                'player_skip': ['webpage', 'configs', 'js'],
                'skip': ['hls', 'dash', 'translated_subs'],
            }
        },
        'http_headers': {
            'User-Agent': 'com.google.android.youtube/17.36.4 (Linux; U; Android 12; GB) gzip',
            'Accept-Language': 'en-US,en;q=0.9',
        }
    }
    
    # Add PO token provider if available (optional enhancement)
    if po_token_server:
        try:
            ydl_opts['extractor_args']['youtube']['po_token'] = ['http://127.0.0.1:8050']
            logger.info("PO token server available")
        except Exception:
            pass

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
    """Transcribe audio using faster-whisper."""
    logger.info(f"Starting transcription of: {audio_path}")

    try:
        if not os.path.exists(audio_path):
            raise Exception(f"Audio file not found: {audio_path}")

        model = get_whisper_model()
        segments, info = model.transcribe(
            audio_path,
            word_timestamps=True,
            language=None  # Auto-detect language
        )

        # Convert generator to list and build result structure
        segments_list = list(segments)
        
        if not segments_list:
            raise Exception("Transcription failed to produce valid output")

        logger.info(f"Transcription completed successfully with {len(segments_list)} segments")
        logger.info(f"Detected language: {info.language} with probability {info.language_probability:.2f}")
        
        return {
            'segments': segments_list,
            'language': info.language
        }

    except Exception as e:
        logger.error(f"Error in transcribe_audio: {str(e)}")
        raise


def generate_clip_title(sentences, keyword=None, max_length=60):
    """
    Generate a short, engaging title for a clip based on its transcript content.
    
    Args:
        sentences: List of sentences in the clip
        keyword: Optional keyword that was searched for
        max_length: Maximum length of the title
    
    Returns:
        A concise, descriptive title string
    """
    if not sentences:
        return "Untitled Clip"
    
    # Common filler words to remove from titles
    filler_words = {
        'um', 'uh', 'like', 'you know', 'basically', 'actually', 'literally',
        'so', 'well', 'right', 'okay', 'ok', 'and', 'but', 'the', 'a', 'an',
        'i', 'we', 'they', 'he', 'she', 'it', 'is', 'are', 'was', 'were',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'
    }
    
    # Take the first 2-3 sentences for title extraction
    source_text = ' '.join(sentences[:3])
    
    # Clean up the text
    source_text = re.sub(r'[^\w\s]', ' ', source_text)  # Remove punctuation
    source_text = re.sub(r'\s+', ' ', source_text).strip()  # Normalize whitespace
    
    words = source_text.split()
    
    # If keyword is provided, try to include it in the title
    if keyword and keyword.lower() in source_text.lower():
        # Find a phrase containing the keyword
        keyword_lower = keyword.lower()
        for i, word in enumerate(words):
            if keyword_lower in word.lower():
                # Extract surrounding context (3 words before and after)
                start = max(0, i - 2)
                end = min(len(words), i + 4)
                title_words = words[start:end]
                title = ' '.join(title_words).title()
                if len(title) <= max_length:
                    return title[:max_length]
    
    # Extract key content words (nouns, verbs, adjectives)
    content_words = [w for w in words if w.lower() not in filler_words and len(w) > 2]
    
    if not content_words:
        content_words = words[:6]
    
    # Create title from first meaningful words
    title_words = []
    for word in content_words[:8]:
        title_words.append(word)
        current_title = ' '.join(title_words)
        if len(current_title) >= max_length - 10:
            break
    
    title = ' '.join(title_words).title()
    
    # Ensure title is not too long
    if len(title) > max_length:
        title = title[:max_length-3].rsplit(' ', 1)[0] + '...'
    
    return title if title else "Untitled Clip"


def analyze_transcript(segments, keyword=None, num_clips=5, logger=None):
    """
    Analyze transcript to find the most engaging clips using TF-IDF scoring.
    
    Args:
        segments: List of transcription segments from faster-whisper
        keyword: Optional keyword/theme to boost matching segments
        num_clips: Number of clips to extract
        logger: Logger instance
    
    Returns:
        List of (start, end, description, score) tuples for best clips
    """
    if logger:
        logger.info(f"Analyzing transcript for {num_clips} clips" + 
                   (f" with keyword '{keyword}'" if keyword else ""))
    
    # Extract sentences with timestamps
    sentence_data = []
    for segment in segments:
        text = segment.text.strip()
        if len(text) > 10:  # Skip very short segments
            sentence_data.append({
                'text': text,
                'start': segment.start,
                'end': segment.end,
                'words': list(segment.words) if segment.words else []
            })
    
    if not sentence_data:
        if logger:
            logger.warning("No valid sentences found in transcript")
        return []
    
    # Get all text for TF-IDF
    texts = [s['text'] for s in sentence_data]
    
    # Calculate TF-IDF scores
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Score each sentence by its TF-IDF importance (sum of non-zero values)
        sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
        
        # Normalize scores
        if sentence_scores.max() > 0:
            sentence_scores = sentence_scores / sentence_scores.max()
    except Exception as e:
        if logger:
            logger.warning(f"TF-IDF scoring failed: {e}, using uniform scores")
        sentence_scores = np.ones(len(sentence_data))
    
    # Apply keyword boosting if provided
    if keyword:
        keyword_lower = keyword.lower()
        keyword_terms = keyword_lower.split()
        for i, sent in enumerate(sentence_data):
            text_lower = sent['text'].lower()
            # Boost if any keyword term is found
            for term in keyword_terms:
                if term in text_lower:
                    sentence_scores[i] *= 2.0  # Double the score for keyword matches
                    break
    
    # Engagement heuristics - boost sentences that indicate important content
    engagement_patterns = [
        (r'\?', 1.3),  # Questions often introduce explanations
        (r'important|key|crucial|essential|must|should', 1.4),
        (r'example|for instance|such as|like', 1.3),
        (r'first|second|third|step|finally', 1.2),
        (r'how to|why|what is|let me', 1.3),
        (r'tip|trick|secret|hack', 1.4),
        (r'problem|solution|issue|fix', 1.3),
    ]
    
    for i, sent in enumerate(sentence_data):
        text_lower = sent['text'].lower()
        for pattern, boost in engagement_patterns:
            if re.search(pattern, text_lower):
                sentence_scores[i] *= boost
    
    # Add scores to sentence data
    for i, sent in enumerate(sentence_data):
        sent['score'] = sentence_scores[i]
    
    # Sort by score but also consider temporal clustering
    # We want clips that are coherent chunks, not scattered sentences
    scored_data = sorted(enumerate(sentence_data), key=lambda x: x[1]['score'], reverse=True)
    
    # Select top candidates and cluster them into clips
    clips = []
    used_indices = set()
    min_clip_duration = 15  # seconds
    max_clip_duration = 60  # seconds
    
    for idx, sent in scored_data:
        if idx in used_indices:
            continue
            
        if len(clips) >= num_clips:
            break
        
        # Start a clip from this high-scoring sentence
        clip_start = sent['start']
        clip_end = sent['end']
        clip_sentences = [sent['text']]
        clip_score = sent['score']
        used_indices.add(idx)
        
        # Expand clip by including adjacent sentences
        # Look backwards
        for j in range(idx - 1, -1, -1):
            if j in used_indices:
                break
            prev_sent = sentence_data[j]
            new_duration = clip_end - prev_sent['start']
            if new_duration > max_clip_duration:
                break
            if prev_sent['end'] < clip_start - 2:  # Gap too large
                break
            clip_start = prev_sent['start']
            clip_sentences.insert(0, prev_sent['text'])
            clip_score += prev_sent['score'] * 0.5  # Weighted less
            used_indices.add(j)
        
        # Look forwards
        for j in range(idx + 1, len(sentence_data)):
            if j in used_indices:
                break
            next_sent = sentence_data[j]
            new_duration = next_sent['end'] - clip_start
            if new_duration > max_clip_duration:
                break
            if next_sent['start'] > clip_end + 2:  # Gap too large
                break
            clip_end = next_sent['end']
            clip_sentences.append(next_sent['text'])
            clip_score += next_sent['score'] * 0.5
            used_indices.add(j)
        
        # Check minimum duration
        if clip_end - clip_start >= min_clip_duration:
            # Ensure we end on a complete sentence
            # Check if the last sentence ends with proper punctuation
            last_sentence = clip_sentences[-1] if clip_sentences else ""
            if not last_sentence.rstrip().endswith(('.', '!', '?', ':')):
                # Try to extend to next sentence for clean ending
                next_idx = max(used_indices) + 1 if used_indices else idx + 1
                if next_idx < len(sentence_data):
                    next_sent = sentence_data[next_idx]
                    # Only extend if it won't make clip too long
                    if next_sent['end'] - clip_start <= max_clip_duration + 10:
                        clip_end = next_sent['end']
                        clip_sentences.append(next_sent['text'])
                        used_indices.add(next_idx)
            
            description = ' '.join(clip_sentences)[:200]  # Truncate description
            if len(' '.join(clip_sentences)) > 200:
                description += '...'
            
            # Generate a short, engaging title from the transcript
            title = generate_clip_title(clip_sentences, keyword)
            
            clips.append({
                'start': clip_start,
                'end': clip_end,
                'title': title,
                'description': description,
                'score': clip_score
            })
    
    # Sort clips by start time for consistent ordering
    clips.sort(key=lambda x: x['start'])
    
    if logger:
        logger.info(f"Found {len(clips)} clip candidates")
    
    return clips


def detect_silences(audio_path, min_silence_duration=0.3, silence_threshold=-35, logger=None):
    """
    Detect silence points in audio using FFmpeg's silencedetect filter.
    Returns list of silence end timestamps (natural break points).
    """
    if logger:
        logger.info("Detecting silence points for clean clip boundaries...")
    
    try:
        command = [
            'ffmpeg',
            '-i', audio_path,
            '-af', f'silencedetect=noise={silence_threshold}dB:d={min_silence_duration}',
            '-f', 'null',
            '-'
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        stderr = result.stderr
        
        # Parse silence_end timestamps from FFmpeg output
        silence_points = []
        for line in stderr.split('\n'):
            if 'silence_end' in line:
                match = re.search(r'silence_end: ([\d.]+)', line)
                if match:
                    silence_points.append(float(match.group(1)))
        
        if logger:
            logger.info(f"Found {len(silence_points)} silence points")
        
        return sorted(silence_points)
    
    except Exception as e:
        if logger:
            logger.warning(f"Silence detection failed: {e}, using fallback")
        return []


def find_nearest_silence(target_time, silence_points, search_range=5.0, prefer_after=True):
    """
    Find the nearest silence point to the target time.
    
    Args:
        target_time: The timestamp we want to end near
        silence_points: List of detected silence timestamps
        search_range: How far (seconds) to search for a silence point
        prefer_after: If True, prefer silence points after the target
    
    Returns:
        Best silence point timestamp, or target_time if none found
    """
    if not silence_points:
        return target_time
    
    candidates = []
    for sp in silence_points:
        distance = sp - target_time
        if -search_range <= distance <= search_range:
            # Score: prefer points after target, but not too far after
            if prefer_after and distance >= 0:
                score = 1.0 / (1.0 + distance)  # Closer is better
            elif not prefer_after and distance <= 0:
                score = 1.0 / (1.0 + abs(distance))
            else:
                score = 0.5 / (1.0 + abs(distance))
            candidates.append((sp, score))
    
    if candidates:
        # Return the one with highest score
        best = max(candidates, key=lambda x: x[1])
        return best[0]
    
    return target_time


def extract_clips(video_path, clip_data, youtube_id, logger, audio_path=None):
    """Extract video clips with silence-based boundary detection for clean endings."""
    logger.info(f"Starting clip extraction for {len(clip_data)} clips")
    
    clip_folder = os.path.join(CLIPS_DIR, youtube_id)
    os.makedirs(clip_folder, exist_ok=True)
    
    # Detect silence points for clean cut boundaries
    silence_points = []
    if audio_path and os.path.exists(audio_path):
        silence_points = detect_silences(audio_path, logger=logger)
    
    extracted_clips = []
    
    for idx, clip in enumerate(clip_data):
        start = clip['start']
        end = clip['end']
        
        # Find the nearest silence point for a clean ending
        if silence_points:
            # Look for a silence point within 4 seconds after our intended end
            clean_end = find_nearest_silence(end, silence_points, search_range=4.0, prefer_after=True)
            # Also try to start at a clean point
            clean_start = find_nearest_silence(start, silence_points, search_range=2.0, prefer_after=False)
            
            # Use clean boundaries if they're reasonable
            if clean_start < start + 2:
                start = max(0, clean_start - 0.3)  # Small buffer before silence ends
            else:
                start = max(0, start - 0.5)
            
            if clean_end > end - 1:  # Don't shorten the clip
                end = clean_end + 0.3  # Small buffer after silence
            else:
                end = end + 1.5  # Fallback buffer
        else:
            # No silence detection - use buffers
            start = max(0, start - 0.5)
            end = end + 2.0  # Larger buffer without silence detection
        
        logger.info(f"Clip {idx+1}: adjusted {clip['start']:.1f}-{clip['end']:.1f} -> {start:.1f}-{end:.1f}")
        
        output_path = os.path.join(clip_folder, f"clip_{idx+1:02d}.mp4")
        
        try:
            # Use ultrafast preset for speed
            command = [
                'ffmpeg',
                '-y',
                '-ss', str(start),
                '-i', video_path,
                '-t', str(end - start),
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-crf', '23',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-avoid_negative_ts', 'make_zero',
                output_path
            ]
            
            subprocess.run(command, check=True, capture_output=True, text=True)
            
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                extracted_clips.append({
                    'filename': f"clip_{idx+1:02d}.mp4",
                    'path': output_path,
                    'start': start,
                    'end': end,
                    'duration': end - start,
                    'title': clip.get('title', f'Clip {idx+1}'),
                    'description': clip['description']
                })
                logger.info(f"Created clip {idx+1}: {output_path} ({end-start:.2f}s)")
            else:
                logger.error(f"Failed to create clip {idx+1}: Empty or missing file")
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg failed for clip {idx+1}: {e.stderr}")
            if os.path.exists(output_path):
                os.remove(output_path)
            continue
    
    logger.info(f"Successfully created {len(extracted_clips)} clips")
    return extracted_clips


def cleanup_files(files_list, logger):
    """Clean up temporary files."""
    for file_path in files_list:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")


def validate_video_length(video_path, max_duration=1800):
    """Validate video length is within acceptable limits (default 30 min)."""
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
        return duration <= max_duration, duration
    except Exception:
        return False, 0


def check_disk_space(required_space_mb=1000):
    """Check if there's enough disk space available."""
    try:
        total, used, free = shutil.disk_usage(".")
        free_mb = free // (2**20)
        return free_mb >= required_space_mb
    except Exception:
        return False


def create_zip(youtube_id, clips_info, logger):
    """Create a ZIP file containing all clips."""
    clip_folder = os.path.join(CLIPS_DIR, youtube_id)
    zip_path = os.path.join(clip_folder, f"{youtube_id}_all_clips.zip")
    
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for clip in clips_info:
                clip_path = os.path.join(clip_folder, clip['filename'])
                if os.path.exists(clip_path):
                    zipf.write(clip_path, clip['filename'])
        
        logger.info(f"Created ZIP file: {zip_path}")
        return zip_path
    except Exception as e:
        logger.error(f"Failed to create ZIP: {str(e)}")
        return None


# ============== Flask Routes ==============

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Process the YouTube video URL."""
    start_time = time.time()
    temp_files = []
    logger = None

    try:
        data = request.get_json()
        url = data.get('url')
        keyword = data.get('keyword', '').strip() or None
        num_clips = int(data.get('num_clips', 5))
        
        # Validate inputs
        if not url:
            return jsonify({'error': 'No URL provided'}), 400
        
        num_clips = max(1, min(10, num_clips))  # Clamp between 1-10

        youtube_id = extract_youtube_id(url)
        if not youtube_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400

        logger = setup_logger(youtube_id)
        logger.info(f"Processing started for YouTube ID: {youtube_id}")
        logger.info(f"Parameters: keyword='{keyword}', num_clips={num_clips}")

        # Check disk space
        if not check_disk_space():
            raise Exception("Insufficient disk space")

        # Create a specific folder for this video's clips
        video_clip_dir = os.path.join(CLIPS_DIR, youtube_id)
        
        # Clean up old clips if they exist
        if os.path.exists(video_clip_dir):
            shutil.rmtree(video_clip_dir)
        os.makedirs(video_clip_dir, exist_ok=True)

        # Download video
        video_path = download_youtube_video(url, youtube_id, logger)
        temp_files.append(video_path)

        # Validate video length (max 30 minutes)
        valid, duration = validate_video_length(video_path)
        if not valid:
            raise Exception(f"Video is too long ({duration/60:.1f} min). Maximum is 30 minutes.")

        # Extract audio
        audio_path = extract_audio(video_path, youtube_id, logger)
        temp_files.append(audio_path)

        # Transcribe
        transcription_result = transcribe_audio(audio_path, logger)
        
        # Analyze transcript and find best clips
        clip_data = analyze_transcript(
            transcription_result['segments'],
            keyword=keyword,
            num_clips=num_clips,
            logger=logger
        )

        if not clip_data:
            return jsonify({'error': 'No suitable clips found in video. Try a different keyword or video.'}), 404

        # Extract clips (pass audio_path for silence-based boundary detection)
        clips_info = extract_clips(video_path, clip_data, youtube_id, logger, audio_path=audio_path)

        # Clean up temporary files after clips are created
        cleanup_files([audio_path, video_path], logger)

        processing_time = time.time() - start_time
        logger.info(f"Processing completed in {processing_time:.2f} seconds")

        return jsonify({
            'status': 'success',
            'youtube_id': youtube_id,
            'clips': clips_info,
            'total_clips': len(clips_info),
            'processing_time': round(processing_time, 2),
            'keyword': keyword
        })

    except Exception as e:
        if logger:
            logger.error(f"Error occurred: {str(e)}")
        # Clean up any temporary files in case of error
        if logger:
            cleanup_files(temp_files, logger)
        return jsonify({'error': str(e)}), 500


@app.route('/clips/<youtube_id>/<filename>')
def serve_clip(youtube_id, filename):
    """Serve a generated clip file for preview."""
    try:
        clip_dir = os.path.join(CLIPS_DIR, youtube_id)
        return send_from_directory(clip_dir, filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/download/<youtube_id>/<filename>')
def download_clip(youtube_id, filename):
    """Download a single clip file."""
    try:
        clip_dir = os.path.join(CLIPS_DIR, youtube_id)
        return send_from_directory(
            clip_dir, 
            filename, 
            as_attachment=True,
            download_name=f"{youtube_id}_{filename}"
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/download_all/<youtube_id>')
def download_all(youtube_id):
    """Download all clips as a ZIP file."""
    try:
        logger = setup_logger(youtube_id)
        clip_dir = os.path.join(CLIPS_DIR, youtube_id)
        
        if not os.path.exists(clip_dir):
            return jsonify({'error': 'No clips found for this video'}), 404
        
        # Get all clip files
        clips = [f for f in os.listdir(clip_dir) if f.endswith('.mp4') and f.startswith('clip_')]
        
        if not clips:
            return jsonify({'error': 'No clips found for this video'}), 404
        
        clips_info = [{'filename': f} for f in clips]
        zip_path = create_zip(youtube_id, clips_info, logger)
        
        if zip_path and os.path.exists(zip_path):
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f"{youtube_id}_all_clips.zip",
                mimetype='application/zip'
            )
        else:
            return jsonify({'error': 'Failed to create ZIP file'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'disk_space_available': check_disk_space(),
        'version': '2.0.0'
    })


if __name__ == '__main__':
    print("\n" + "="*50)
    print("AI Video Clipper - Starting server...")
    print("="*50)
    
    # Start PO token server for YouTube authentication
    print("Initializing YouTube authentication...")
    start_po_token_server()
    
    print("\nOpen http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop the server\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
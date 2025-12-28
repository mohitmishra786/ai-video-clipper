# AI Video Clipper

> **Automatically extract the most engaging moments from any YouTube video using AI**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)

AI Video Clipper is an open-source tool that uses AI-powered transcription and NLP analysis to automatically identify and extract the most engaging segments from YouTube videos. No API keys required!

## Features

- **Keyword-based filtering** - Find clips related to specific themes or topics
- **AI-powered clip detection** - TF-IDF scoring + engagement heuristics
- **Smart boundary detection** - Silence detection for clean clip endings
- **Video previews** - Watch clips directly in the browser
- **Easy downloads** - Single clip or batch ZIP download
- **Auto-generated titles** - Descriptive titles from transcript content
- **Fast processing** - Optimized for Apple Silicon (M1/M4)

## Quick Start

### Prerequisites

- Python 3.11+
- FFmpeg (`brew install ffmpeg` on macOS)

### Installation

```bash
# Clone the repository
git clone https://github.com/mohitmishra786/ai-video-clipper.git
cd ai-video-clipper

# Create virtual environment (Python 3.11)
python3.11 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

Open http://localhost:5000 in your browser.

## Usage

1. **Paste a YouTube URL** - Any video up to 30 minutes
2. **Optional: Add a keyword** - e.g., "tutorial", "funny moments", "tips"
3. **Select number of clips** - 1 to 10
4. **Click Extract Clips** - Wait for processing
5. **Preview and Download** - Watch in browser, download individually or as ZIP

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

### Core Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Web Framework | Flask | REST API and UI serving |
| Video Download | yt-dlp | YouTube video extraction |
| Transcription | faster-whisper | Speech-to-text (CPU optimized) |
| Clip Analysis | scikit-learn | TF-IDF scoring + NLP |
| Video Processing | FFmpeg | Audio extraction, clip creation |

### Processing Pipeline

```
YouTube URL -> Download -> Extract Audio -> Transcribe -> Analyze -> Extract Clips -> Preview/Download
```

## Configuration

### Environment Variables (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_DEBUG` | `true` | Enable debug mode |
| `MAX_VIDEO_LENGTH` | `1800` | Max video length in seconds (30 min) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main UI |
| `/process` | POST | Process video URL |
| `/clips/<id>/<file>` | GET | Serve clip for preview |
| `/download/<id>/<file>` | GET | Download single clip |
| `/download_all/<id>` | GET | Download all clips as ZIP |
| `/health` | GET | Health check |

### POST /process

```json
{
  "url": "https://youtube.com/watch?v=...",
  "keyword": "optional keyword",
  "num_clips": 5
}
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run with auto-reload
python app.py
```

## Testing

```bash
# Manual testing
# 1. Start the server
python app.py

# 2. Open browser to http://localhost:5000
# 3. Paste a short YouTube video URL
# 4. Verify clips are generated
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - Optimized Whisper implementation
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube downloading
- [FFmpeg](https://ffmpeg.org/) - Video processing

## Contact

- **Author**: Mohit Mishra
- **GitHub**: [@mohitmishra786](https://github.com/mohitmishra786)

---

**Star this repo** if you find it useful!
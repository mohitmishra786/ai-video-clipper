import sys
print("Python executable:", sys.executable)
print("Python path:", sys.path)
from moviepy import VideoFileClip
import os

def extract_audio(video_path, output_audio_path="audio.wav"):
    # Load the video file
    video = VideoFileClip(video_path)
    # Extract and save the audio
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Example usage
if __name__ == "__main__":
    video_path = "downloads/A New Era for C and C++？ Goodbye, Rust？.webm"
    audio_path = extract_audio(video_path)
    print(f"Audio extracted to: {audio_path}")
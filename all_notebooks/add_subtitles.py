import os
import subprocess

def add_subtitles_to_clips(clips_dir, srt_path, output_dir="downloads/clips_with_subtitles"):
    os.makedirs(output_dir, exist_ok=True)
    for clip in os.listdir(clips_dir):
        if clip.endswith(".mp4"):
            input_clip = os.path.join(clips_dir, clip)
            output_clip = os.path.join(output_dir, clip)
            # Use ffmpeg to add subtitles
            subprocess.run([
                "ffmpeg", "-i", input_clip, "-vf", f"subtitles={srt_path}", output_clip
            ])
    print(f"Subtitled clips saved to: {output_dir}")

# Example usage
if __name__ == "__main__":
    clips_dir = "downloads/clips"
    srt_path = "downloads/subtitles.srt"
    add_subtitles_to_clips(clips_dir, srt_path)
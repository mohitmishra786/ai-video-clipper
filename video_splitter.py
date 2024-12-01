from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def split_video(video_path, clip_length=60, output_dir="downloads/clips"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Load the video file
    video = VideoFileClip(video_path)
    duration = int(video.duration)  # Total duration of the video in seconds
    clips = []
    # Split the video into clips
    for start_time in range(0, duration, clip_length):
        end_time = min(start_time + clip_length, duration)
        clip = video.subclipped(start_time, end_time)
        clip_path = os.path.join(output_dir, f"clip_{start_time}_{end_time}.mp4")
        clip.write_videofile(clip_path, codec="libx264")
        clips.append(clip_path)
    return clips

# Example usage
if __name__ == "__main__":
    video_path = "downloads/A New Era for C and C++？ Goodbye, Rust？.webm"
    clips = split_video(video_path, clip_length=120)  # 2-minute clips
    print("Clips created:")
    print(clips)
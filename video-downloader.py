import yt_dlp

def download_video(youtube_url, output_path="downloads"):
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',  # Download best quality
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',  # Save with video title
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return ydl.prepare_filename(info)

# Example usage
video_path = download_video("https://youtu.be/V_QAJAhbH9A?si=-aOd7l3-_yOS3Fv-")
print(f"Video downloaded to: {video_path}")
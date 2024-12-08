def create_srt(transcription, output_srt_path="downloads/subtitles.srt"):
    lines = transcription.split(". ")
    with open(output_srt_path, "w") as srt_file:
        for i, line in enumerate(lines):
            start_time = f"00:00:{i*2:02d},000"
            end_time = f"00:00:{(i+1)*2:02d},000"
            srt_file.write(f"{i+1}\n{start_time} --> {end_time}\n{line.strip()}\n\n")
    return output_srt_path

# Example usage
if __name__ == "__main__":
    with open("downloads/transcription.txt", "r") as f:
        transcription = f.read()
    srt_path = create_srt(transcription)
    print(f"SRT file created: {srt_path}")
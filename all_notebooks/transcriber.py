import whisper

def transcribe_audio(audio_path):
    # Load the Whisper model
    model = whisper.load_model("base")  # Use "base" model for free
    # Transcribe the audio
    result = model.transcribe(audio_path)
    return result["text"]

# Example usage
if __name__ == "__main__":
    audio_path = "downloads/A New Era for C and C++？ Goodbye, Rust？.wav"
    transcription = transcribe_audio(audio_path)
    print("Transcription:")
    print(transcription)
    # Save transcription to a file
    with open("downloads/transcription.txt", "w") as f:
        f.write(transcription)
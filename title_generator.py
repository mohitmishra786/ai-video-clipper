from transformers import pipeline

def generate_title(text):
    summarizer = pipeline("summarization")
    max_input_length = 1024  # Maximum token limit for distilbart-cnn-12-6
    chunk_size = 1000  # Slightly less than the max limit to avoid errors
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Summarize each chunk and combine the results
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=10, min_length=5, do_sample=False)
        summaries.append(summary[0]["summary_text"])

    # Combine all summaries into a single title
    combined_summary = " ".join(summaries)
    return combined_summary

# Example usage
if __name__ == "__main__":
    with open("downloads/transcription.txt", "r") as f:
        transcription = f.read()
    title = generate_title(transcription)
    print(f"Generated Title: {title}")
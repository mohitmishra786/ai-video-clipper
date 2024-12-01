import whisper
from collections import Counter
from nltk.util import ngrams
import nltk
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
import re
import os
from moviepy.config import change_settings

# Configure MoviePy to use ImageMagick
# For Windows, replace with your ImageMagick installation path
# change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.0.10-Q16\magick.exe"})

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def transcribe_video(video_path):
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        return result["text"], result["segments"]
    except Exception as e:
        print(f"Error in transcription: {str(e)}")
        return None, None

def create_technical_keywords():
    return {
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

def is_sentence_boundary(text):
    return bool(re.search(r'[.!?]\s*$', text.strip()))

def find_technical_segments(segments, keywords_dict):
    if not segments:
        return []

    technical_clips = []
    all_keywords = [keyword.lower() for sublist in keywords_dict.values() 
                   for keyword in sublist]

    current_segment = None

    for segment in segments:
        text = segment["text"].lower()

        if any(keyword in text for keyword in all_keywords):
            if current_segment is None:
                current_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "keywords": [],
                    "subtitles": []
                }

            current_segment["end"] = segment["end"]
            current_segment["text"] += " " + segment["text"]
            current_segment["subtitles"].append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
            current_segment["keywords"].extend([k for k in all_keywords if k in text])

        elif current_segment is not None:
            if not is_sentence_boundary(current_segment["text"]):
                current_segment["end"] = segment["end"]
                current_segment["text"] += " " + segment["text"]
                current_segment["subtitles"].append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                })

            if current_segment["end"] - current_segment["start"] >= 15:
                technical_clips.append(current_segment)
            current_segment = None

    if current_segment is not None and current_segment["end"] - current_segment["start"] >= 15:
        technical_clips.append(current_segment)

    return technical_clips

def create_subtitle_clip(text, video_width, start_time, duration):
    try:
        txt_clip = TextClip(
            text, 
            font='Arial', 
            fontsize=30, 
            color='white',
            bg_color='black',
            size=(video_width-20, None),  # Slight padding on sides
            method='caption'
        )
        txt_clip = txt_clip.set_position(('center', 'bottom'))
        txt_clip = txt_clip.set_start(start_time)
        txt_clip = txt_clip.set_duration(duration)
        return txt_clip
    except Exception as e:
        print(f"Error creating subtitle: {str(e)}")
        return None

def adjust_clip_duration(start_time, end_time, min_duration=20, max_duration=90):
    duration = end_time - start_time
    if duration < min_duration:
        end_time = start_time + min_duration
    elif duration > max_duration:
        end_time = start_time + max_duration
    return start_time, end_time

def clip_video(video_path, clips, output_prefix):
    try:
        video = VideoFileClip(video_path)
        video_width = video.size[0]

        for i, clip_info in enumerate(clips, 1):
            try:
                start_time, end_time = adjust_clip_duration(clip_info["start"], clip_info["end"])

                # Create unique keywords string
                keywords_set = set(clip_info["keywords"])
                keywords_str = "_".join(list(keywords_set)[:2])
                output_path = f"{output_prefix}_{i}_{keywords_str}.mp4"

                # Create main video clip
                main_clip = video.subclip(start_time, end_time)

                # Create subtitle clips
                subtitle_clips = []
                for sub in clip_info["subtitles"]:
                    if sub["start"] >= start_time and sub["start"] < end_time:
                        sub_start = sub["start"] - start_time
                        sub_duration = min(sub["end"] - sub["start"], 
                                         end_time - sub["start"])

                        subtitle_clip = create_subtitle_clip(
                            sub["text"],
                            video_width,
                            sub_start,
                            sub_duration
                        )
                        if subtitle_clip:
                            subtitle_clips.append(subtitle_clip)

                # Combine video with subtitles
                if subtitle_clips:
                    final_clip = CompositeVideoClip([main_clip] + subtitle_clips)
                else:
                    final_clip = main_clip

                # Write the final clip
                final_clip.write_videofile(output_path, 
                                         codec="libx264",
                                         audio_codec="aac",
                                         temp_audiofile="temp-audio.m4a",
                                         remove_temp=True)

                print(f"Successfully saved clip {i} to {output_path}")
                print(f"Content: {clip_info['text'][:100]}...")
                print(f"Keywords found: {', '.join(keywords_set)}\n")

                # Clean up
                final_clip.close()
                for clip in subtitle_clips:
                    clip.close()

            except Exception as e:
                print(f"Error processing clip {i}: {str(e)}")
                continue

        video.close()

    except Exception as e:
        print(f"Error in video processing: {str(e)}")

def extract_technical_clips(video_path, output_prefix="technical_clip"):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print("Transcribing video...")
    transcription, segments = transcribe_video(video_path)

    if not segments:
        print("Error: Failed to transcribe video")
        return

    keywords_dict = create_technical_keywords()

    print("Analyzing content for technical segments...")
    technical_clips = find_technical_segments(segments, keywords_dict)

    if not technical_clips:
        print("No significant technical content found.")
        return

    print(f"Found {len(technical_clips)} technical segments")

    print("Creating video clips...")
    clip_video(video_path, technical_clips, output_prefix)

if __name__ == "__main__":
    video_path = "downloads/video.mp4"
    extract_technical_clips(video_path, "gpu_technical_clip")

# Created/Modified files during execution:
# - gpu_technical_clip_1_*.mp4 (multiple files based on found segments)
# - temp-audio.m4a (temporary file, automatically removed)
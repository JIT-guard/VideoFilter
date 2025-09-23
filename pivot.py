import whisper
import torch
import ollama
import librosa
import soundfile as sf
import numpy as np
import random
from detoxify import Detoxify

def DEX(sentence: str):
    '''
    Judge if the sentence should be removed using detoxify

    Args:
        sentence: string

    Returns:
        bool value
    '''
    model = Detoxify('original')
    scores = model.predict(sentence)
    if max(scores.values()) > 0.5:
        return True
    else:
        return False

def guard(sentence: str):
    '''
    Judge if the sentence should be removed using llamaguard

    Args:
        sentence: string

    Returns:
        bool value
    '''
    print(sentence)
    response = ollama.chat(
        model="llama-guard3:8b",
        messages=[
            {"role": "user", "content": sentence}
            ],
    )
    judge = response["message"]["content"].split()
    print(judge[0])
    if judge[0] == "safe":
        return False # False means should not remove
    else:
        return True # True means should remove

def transcribe_with_timestamps(audio_file):
    model = whisper.load_model("base", device="cuda")
    result = model.transcribe(audio_file, word_timestamps=True)
    return result

def filter_sentences_from_audio(result, should_remove_function, audio_file, output_file="filtered_audio.wav", method="silence"):
    """
    Remove sentences based on a custom filtering function
    
    Args:
        result: Whisper transcription result with timestamps
        should_remove_function: Function that takes a sentence (string) and returns True if it should be removed
        audio_file: Original audio file path
        output_file: Output filtered audio file path
        method: "silence" (replace with silence) or "cut" (remove completely)
    
    Returns:
        List of removed sentences with their info
    """
    audio, sr = librosa.load(audio_file, sr=None)
    removed_sentences = []
    kept_segments = []
    
    print("Analyzing sentences...")
    
    for i, segment in enumerate(result['segments']):
        sentence = segment['text'].strip()
        start_time = segment['start']
        end_time = segment['end']
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        should_remove = DEX(sentence) # use detoxify instead of llamaguard
        # should_remove = should_remove_function(sentence)

        # Call your custom function to decide if sentence should be removed
        
        if should_remove:
            removed_sentences.append({
                'text': sentence,
                'start': start_time,
                'end': end_time,
                'index': i
            })
            print(f"❌ REMOVE [{start_time:.1f}s-{end_time:.1f}s]: {sentence[:60]}...")
            
            if method == "silence":
                # Replace with silence
                audio[start_sample:end_sample] = 0
        else:
            kept_segments.append({
                'audio': audio[start_sample:end_sample],
                'start': start_time,
                'end': end_time
            })
            print(f"✅ KEEP   [{start_time:.1f}s-{end_time:.1f}s]: {sentence[:60]}...")
    
    if method == "cut":
        # Concatenate only kept segments
        if kept_segments:
            audio_pieces = [seg['audio'] for seg in kept_segments]
            final_audio = np.concatenate(audio_pieces)
            sf.write(output_file, final_audio, sr)
            original_duration = len(audio) / sr
            new_duration = len(final_audio) / sr
            print(f"Saved cut audio: {original_duration:.1f}s → {new_duration:.1f}s ({original_duration-new_duration:.1f}s removed)")
        else:
            print("No sentences to keep!")
            return removed_sentences
    else:
        # Save silenced audio (preserves timing)
        sf.write(output_file, audio, sr)
        print(f"Saved silenced audio to: {output_file}")
    
    print(f"Removed {len(removed_sentences)} sentences, kept {len(kept_segments)}")
    return removed_sentences

def SilenceWarning():
    '''remove warnings for demos'''
    import warnings
    warnings.filterwarnings("ignore")

SilenceWarning()
audio_file_name = "hate.mp4"
result = transcribe_with_timestamps(audio_file_name)
final_audio = filter_sentences_from_audio(
    result=result,
    should_remove_function=guard,
    audio_file=audio_file_name
)
print(final_audio)

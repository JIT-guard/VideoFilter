import whisper
import librosa
import soundfile as sf
import numpy as np
import random

def guard(sentence: str):
    '''
    Judge if the sentence should be removed

    Args:
        sentence: string

    Returns:
        bool value
    '''
    return (random.randint(1, 10) % 2 == 1)

def transcribe_with_timestamps(audio_file):
    model = whisper.load_model("base")
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
        
        # Call your custom function to decide if sentence should be removed
        should_remove = should_remove_function(sentence)
        
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

audio_file_name = "t1.m4a"
result = transcribe_with_timestamps(audio_file_name)
final_audio = filter_sentences_from_audio(
    result=result,
    should_remove_function=guard,
    audio_file=audio_file_name
)
print(final_audio)

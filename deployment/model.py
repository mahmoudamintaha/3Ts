import numpy as np
import os
import torch
import librosa
import noisereduce as nr
from pydub import AudioSegment
from datasets import Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import streamlit as st


#__model_path = "whisper_finetuned_V2"

#__model = WhisperForConditionalGeneration.from_pretrained(__model_path)
#__processor = WhisperProcessor.from_pretrained(__model_path)

# Get the absolute path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
__model_path = os.path.join(current_dir, "whisper_finetuned_V2")


# Load model with explicit local path settings
__model = WhisperForConditionalGeneration.from_pretrained(
    __model_path)

__processor = WhisperProcessor.from_pretrained(
    __model_path)


def __load_custom_dataset_pip(data):
    audio_paths = []

    for sample in data:
        audio_path = sample["audio"]

        # Load audio using librosa
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000)  # Resample to 16kHz
        audio_paths.append({"array": audio_array, "sampling_rate": sampling_rate})

    return Dataset.from_dict({"audio": audio_paths})


def __preprocess_function_pip(batch):
    # To hold the expanded input features and labels
    all_input_features = []

    # Process each example in the batch
    for i in range(len(batch["audio"])):
        # Access the audio array and corresponding text for the single example
        audio = batch["audio"][i]["array"]

        # Noise reduction, trimming, and normalization
        audio = nr.reduce_noise(y=audio, sr=16000)
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.util.normalize(audio)

        # Set maximum length for a 30-second chunk
        sample_rate = 16000
        max_length = sample_rate * 30  # 30 seconds in samples
        overlap_duration = 1  # seconds
        overlap_samples = overlap_duration * sample_rate  # Convert overlap duration to samples

        # Create overlapping audio chunks
        audio_chunks = []
        for j in range(0, len(audio), max_length - 0):
            chunk = audio[j:j + max_length]
            if len(chunk) < max_length:  # If chunk is shorter than 30 seconds, pad it
                padding = max_length - len(chunk)
                chunk = np.pad(chunk, (0, padding), mode='constant')
            audio_chunks.append(chunk)

        # Process each audio-text chunk pair
        for audio_chunk in zip(audio_chunks):
            inputs = __processor(audio_chunk, sampling_rate=sample_rate, padding=True,truncation=True, return_tensors="pt")  # Extract audio features

            # Append each chunk's features and labels to the expanded lists
            all_input_features.append(inputs.input_features[0].numpy())

    # Check if any features were collected
    print(f"Preprocessed {len(all_input_features)} Audio Chunk")
    # Return the expanded dictionary
    return {"input_features": all_input_features,}


def __generate_transcription_pip(batch):
    # Extract input features from the batch and create attention mask
    input_features = torch.tensor(batch["input_features"]).unsqueeze(0)  # Add batch dimension

    # Create attention mask to ensure model distinguishes padding from meaningful input
    attention_mask = torch.ones(input_features.shape)  # Initialize attention mask with 1s
    attention_mask[input_features == __processor.feature_extractor.padding_value] = 0  # Set to 0 where there's padding

    # Generate predicted token IDs with the input features and attention mask
    predicted_ids = __model.generate(
        input_features,
        attention_mask=attention_mask,  # Pass the attention mask
        language="en",  # Specify language directly here to avoid conflict
        max_new_tokens=444,  # Set a reasonable limit on tokens to generate
        no_repeat_ngram_size=3,  # To avoid repeated phrases in transcription
        num_beams=5,  # For better decoding (optional)
    )

    # Decode the predicted token IDs back into text
    transcription = __processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    return transcription


def __T3_pipeline(input_path):
    audio = AudioSegment.from_wav(input_path)
    wav_path = input_path
    audio.export(wav_path, format="wav")
    data_in = [
      {"audio": wav_path},
    ]
    input = __load_custom_dataset_pip(data_in)
    input_prep = input.map(__preprocess_function_pip, batched=True, remove_columns=input.column_names)
    results = input_prep.map(lambda batch: {"prediction": __generate_transcription_pip(batch)})
    text = results.to_pandas()
    predictions = text['prediction'].tolist()
    result_str = "\n".join(predictions)

    return result_str


def get_transcript(audio):
    return __T3_pipeline(audio)


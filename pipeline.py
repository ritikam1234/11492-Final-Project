import torch
import librosa
import espnet
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import time
import torch
import soundfile as sf
import json

import argparse
import numpy as np
import wave
import gradio as gr
from huggingface_hub import HfApi

import gradio as gr
import torch
from espnet2.bin.s2t_inference import Speech2Text # Core ESPnet module for pre-trained models
import numpy as np

# global variables (can be passed into function below)
FINETUNE_MODEL="espnet/owsm_v3.1_ebf_base"
owsm_language="eng" # language code in ISO3

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1" # "google/gemma-1.1-7b-it"  # mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, meta-llama/Llama-2-7b-chat-hf


def transcribe_audio(audio, model_name: str = "espnet/owsm_v3.1_ebf_base", language: str = "eng"):
    """
    Perform ASR on a given audio input (wav format)

    Args:
        audio: Audio input as a NumPy array.
        model_name (str): Name of the pretrained model.
        language (str): Language code in ISO3.

    Returns:
        str: Transcribed text from the audio.
    """
    
    speech2text = Speech2Text.from_pretrained(
        model_name,
        lang_sym=f"<{language}>",
        beam_size=1,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


    results = speech2text(audio)

    # Extract transcription
    transcription = results[0][0] if results else ""

    return transcription

import sounddevice as sd
# Parameters
fs = 16000  # Sample rate
duration = 5  # Duration in seconds
output_file = "output.wav"

print("Recording...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait() 
print("Done recording.")

# Transcribe
transcription = transcribe_audio(audio)
print(f" Transcription:", transcription)


# from espnet_model_zoo.downloader import ModelDownloader
# from espnet2.bin.tts_inference import Text2Speech
# from espnet2.utils.types import str_or_none
# import time
# PRETRAIN_MODEL = "espnet/kan-bayashi_libritts_xvector_vits"

# d = ModelDownloader()
# pretrain_downloaded = d.download_and_unpack(PRETRAIN_MODEL)

# lang = 'English'
# tag = "kan-bayashi/ljspeech_tacotron2" #@param ["kan-bayashi/ljspeech_tacotron2"]
# vocoder_tag = "parallel_wavegan/ljspeech_parallel_wavegan.v1" #@param ["parallel_wavegan/ljspeech_parallel_wavegan.v1"]

# import nltk
# import ssl

# # Fix SSL certificate issues (use only if you're getting CERTIFICATE_VERIFY_FAILED)
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass  # Older Python versions may not have this
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# # Now attempt to download cmudict
# # nltk.download('cmudict')
# nltk.download('averaged_perceptron_tagger_eng')

# def load_tts_model(tag: str = "kan-bayashi/ljspeech_tacotron2", vocoder_tag: str = "parallel_wavegan/ljspeech_parallel_wavegan.v1", device="cpu"):
#     """
#     Load the Text-to-Speech model from ESPnet.

#     Args:
#         tag (str): Model tag for TTS.
#         vocoder_tag (str): Vocoder model tag.

#     Returns:
#         Text2Speech: Loaded TTS model.
#     """

#     text2speech2 = Text2Speech.from_pretrained(
#             model_tag=tag,
#             vocoder_tag=str_or_none(vocoder_tag),
#             device=device,
#             # Only for Tacotron 2 & Transformer
#             threshold=0.5,
#             # Only for Tacotron 2
#             minlenratio=0.0,
#             maxlenratio=10.0,
#             use_att_constraint=False,
#             backward_window=1,
#             forward_window=3,
#             # Only for FastSpeech & FastSpeech2 & VITS (not used in this block)
#             speed_control_alpha=1.0,
#             # Only for VITS (not used in this block)
#             noise_scale=0.333,
#             noise_scale_dur=0.333,
#         )
#     return text2speech2

# def synthesize_speech(text, text2speechmodel, output_path):

#     with torch.no_grad():
#         start = time.time()
#         wav = text2speechmodel(text)["wav"]

#     rtf = (time.time() - start) / (len(wav) / text2speechmodel.fs)
#     print(f"RTF = {rtf:.5f}")

#     return wav

# # Synthesize TTS

# torch.cuda.empty_cache()
# text2speech_model = load_tts_model()
# audio = synthesize_speech("example text", text2speech_model, "/content/tts_result/tts_output.wav")
# from IPython.display import display, Audio
# display(Audio(audio.view(-1).cpu().numpy(), rate=text2speech_model.fs))
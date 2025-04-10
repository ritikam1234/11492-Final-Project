import torch
import librosa
import espnet
from espnet2 import Speech2Text # Core ESPnet module for pre-trained models
import json

import argparse
import numpy as np
import wave
import gradio as gr
from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# TTS imports 
from espnet_model_zoo.downloader import ModelDownloader


# global variables (can be passed into function below)
# FINETUNE_MODEL="espnet/owsm_v3.1_ebf_base"
# owsm_language="eng" # language code in ISO3

# MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.1" # "google/gemma-1.1-7b-it"  # mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, meta-llama/Llama-2-7b-chat-hf


# def transcribe_audio(audio, model_name: str = "espnet/owsm_v3.1_ebf_base", language: str = "eng"):
#     """
#     Perform ASR on a given audio input (wav format)

#     Args:
#         audio: Audio input as a NumPy array.
#         model_name (str): Name of the pretrained model.
#         language (str): Language code in ISO3.

#     Returns:
#         str: Transcribed text from the audio.
#     """
    
#     speech2text = Speech2Text.from_pretrained(
#         model_name,
#         lang_sym=f"<{language}>",
#         beam_size=1,
#         device='cuda' if torch.cuda.is_available() else 'cpu'
#     )


#     results = speech2text(audio)

#     # Extract transcription
#     transcription = results[0][0] if results else ""

#     return transcription

# # Transcribe
# transcription = transcribe_audio(audio)
# print(f" Transcription:", transcription)


from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import time
PRETRAIN_MODEL = "espnet/kan-bayashi_libritts_xvector_vits"

d = ModelDownloader()
pretrain_downloaded = d.download_and_unpack(PRETRAIN_MODEL)

lang = 'English'
tag = "kan-bayashi/ljspeech_tacotron2" #@param ["kan-bayashi/ljspeech_tacotron2"]
vocoder_tag = "parallel_wavegan/ljspeech_parallel_wavegan.v1" #@param ["parallel_wavegan/ljspeech_parallel_wavegan.v1"]

def load_tts_model(tag: str = "kan-bayashi/ljspeech_tacotron2", vocoder_tag: str = "parallel_wavegan/ljspeech_parallel_wavegan.v1", device="cpu"):
    """
    Load the Text-to-Speech model from ESPnet.

    Args:
        tag (str): Model tag for TTS.
        vocoder_tag (str): Vocoder model tag.

    Returns:
        Text2Speech: Loaded TTS model.
    """

    text2speech1 = Text2Speech.from_pretrained(
    model_tag=tag,
    vocoder_tag=str_or_none(vocoder_tag),
    device=device,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1.0,
)
    text2speech2 = Text2Speech.from_pretrained(
            model_tag=tag,
            vocoder_tag=str_or_none(vocoder_tag),
            device=device,
            # Only for Tacotron 2 & Transformer
            threshold=0.5,
            # Only for Tacotron 2
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=False,
            backward_window=1,
            forward_window=3,
            # Only for FastSpeech & FastSpeech2 & VITS (not used in this block)
            speed_control_alpha=1.0,
            # Only for VITS (not used in this block)
            noise_scale=0.333,
            noise_scale_dur=0.333,
        )
    return text2speech2

def synthesize_speech(text, text2speechmodel, output_path):

    with torch.no_grad():
        start = time.time()
        wav = text2speechmodel(text)["wav"]

    rtf = (time.time() - start) / (len(wav) / text2speechmodel.fs)
    print(f"RTF = {rtf:.5f}")

    return text2speechmodel.fs
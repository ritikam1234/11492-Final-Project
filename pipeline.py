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

# Transcribe
transcription = transcribe_audio(audio)
print(f" Transcription:", transcription)


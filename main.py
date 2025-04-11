import gradio as gr
import torch
from espnet2.bin.s2t_inference import Speech2Text # Core ESPnet module for pre-trained models
import numpy as np
import librosa
from dotenv import load_dotenv
load_dotenv()
import os

#ASR PRocessing
# global variables (can make it a function later)
FINETUNE_MODEL="espnet/owsm_v3.1_ebf_base"
owsm_language="eng" # language code in ISO3

access_token = os.environ.get("HF_TOKEN")

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



import torch

from huggingface_hub import InferenceClient

hf_token = access_token

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    token=hf_token
)

def format_prompt(question: str) -> str:
    return (
        "<start_of_turn>user\n"
        "Answer concisely in one to two sentences:\n"
        f"{question}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

def run_llm_remote(question: str) -> str:
    prompt = format_prompt(question)
    response = client.text_generation(prompt, max_new_tokens=200)
    return response.strip()



#TTS Code
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

import nltk
import ssl

# Fix SSL certificate issues (use only if you're getting CERTIFICATE_VERIFY_FAILED)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass  # Older Python versions may not have this
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger_eng')

def load_tts_model(tag: str = "kan-bayashi/ljspeech_tacotron2", vocoder_tag: str = "parallel_wavegan/ljspeech_parallel_wavegan.v1", device="cpu"):
    """
    Load the Text-to-Speech model from ESPnet.

    Args:
        tag (str): Model tag for TTS.
        vocoder_tag (str): Vocoder model tag.

    Returns:
        Text2Speech: Loaded TTS model.
    """

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

    return (text2speechmodel.fs, wav.numpy())



def process_audio(audio):

    sr, speech = audio
    speech = speech.astype(np.float32)
    if len(speech.shape) > 1 and speech.shape[1] == 2:
        speech = np.mean(speech, axis=1)
    if np.max(np.abs(speech)) > 1.0:
        speech = speech / np.max(np.abs(speech))
    transcription = transcribe_audio(speech)
    print(transcription)


    return transcription


def tts_response(text):
    text2speech_model = load_tts_model()
    audio = synthesize_speech(text, text2speech_model, "/content/tts_result/tts_output.wav")
    return audio

with gr.Blocks(title="RAG Cascade Model") as demo:
    gr.Markdown("### Record Audio and View Transcription, LLM Output, and TTS Synthesis")

    with gr.Column():
        input_audio = gr.Audio(
            sources=["microphone"],
            streaming=False,
            label="Record Audio",
            waveform_options=gr.WaveformOptions(sample_rate=16000)
        )

        transcription_output = gr.Textbox(label="Transcription", interactive=False)
        llm_output = gr.Textbox(label="LLM Response", interactive=False)
        tts_output = gr.Audio(label="TTS Output", interactive=False)

    with gr.Row():
        clear_btn = gr.Button("Clear")

    clear_btn.click(
        lambda: (None, "", "", None),
        outputs=[input_audio, transcription_output, llm_output, tts_output]
    )

    input_audio.input(
        process_audio,
        inputs=input_audio,
        outputs=transcription_output
    ).then(
        run_llm_remote,
        inputs=transcription_output,
        outputs=llm_output
    ).then(
        tts_response,
        inputs=llm_output,
        outputs=tts_output
    )

demo.launch(share=True)

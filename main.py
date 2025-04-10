import gradio as gr
import torch
from espnet2.bin.s2t_inference import Speech2Text # Core ESPnet module for pre-trained models
import numpy as np
import argparse

#ASR PRocessing
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



def process_audio(audio):
    # In this example, we just return the received audio back as output.
    sr, speech = audio
    # speech = speech.astype(np.float32)
    # if len(speech.shape) > 1 and speech.shape[1] == 2:
    #     speech = np.mean(speech, axis=1)
    transcription = transcribe_audio(speech)
    print(transcription)

    return transcription

# Simplified interface
with gr.Blocks(title="RAG Cascade System") as demo:
    with gr.Row():
        gr.Markdown("### Record and Listen to Your Audio")

    with gr.Column(scale=1):
        # Record audio from the microphone
        input_audio = gr.Audio(
            sources=["microphone"],
            streaming=False,  # Disable streaming to make it process when recording is done
            waveform_options=gr.WaveformOptions(sample_rate=16000)
        )

    with gr.Row():
        # Output area for the transcription
        output_text = gr.Textbox(label="Transcription Output", interactive=False)

    # Define behavior: when user records audio, process it and return transcription result
    input_audio.change(
        process_audio, 
        inputs=[input_audio], 
        outputs=[output_text]
    )

demo.launch(share=True)

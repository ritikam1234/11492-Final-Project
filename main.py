import gradio as gr
import torch
from espnet2.bin.s2t_inference import Speech2Text # Core ESPnet module for pre-trained models
import numpy as np
import argparse

from dotenv import load_dotenv
load_dotenv()
import os

#ASR PRocessing
# global variables (can be passed into function below)
FINETUNE_MODEL="espnet/owsm_v3.1_ebf_base"
owsm_language="eng" # language code in ISO3
MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"  # "mistralai/Mistral-7B-Instruct-v0.1" # "google/gemma-1.1-7b-it"  # mistralai/Mistral-7B-Instruct-v0.1, mistralai/Mixtral-8x7B-Instruct-v0.1, meta-llama/Llama-2-7b-chat-hf

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


from huggingface_hub import HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=access_token)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    token=access_token
)

# Create a pipeline for text generation
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=False,
    return_full_text=False
)

# prompt formatting function
def format_prompt(user_input: str) -> str:
    return (
        "<start_of_turn>user\n"
        "Answer concisely in one to two sentences:\n"
        f"{user_input}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

# run LLM with question
def run_llm(question: str) -> str:
    prompt = format_prompt(question)
    output = llm(prompt)[0]["generated_text"]
    return output.strip()

#TTS Code
from espnet_model_zoo.downloader import ModelDownloader
PRETRAIN_MODEL = "espnet/kan-bayashi_libritts_xvector_vits"
d = ModelDownloader()
pretrain_downloaded = d.download_and_unpack(PRETRAIN_MODEL)

import nltk
nltk.download('averaged_perceptron_tagger_eng')


def process_audio(audio):
    # In this example, we just return the received audio back as output.
    sr, speech = audio
    # speech = speech.astype(np.float32)
    # if len(speech.shape) > 1 and speech.shape[1] == 2:
    #     speech = np.mean(speech, axis=1)
    transcription = transcribe_audio(speech)
    print(transcription)

    response = run_llm("What are the benefits of solar energy?")
    print("🤖", response)

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

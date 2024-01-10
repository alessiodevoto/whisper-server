import time
import torch
import numpy as np
import logging
from typing import Union, Literal, Tuple, Union, Any
from utils import format_timestamps
from multiprocessing import Pool
import math
import gradio as gr
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def identity(batch):
    return batch


class WhisperInference:
    """
    Perform Speech to Text inference on an audio track. 
    """

    def __init__(
        self, 
        logger : Any, 
        cuda_device_index : int, 
        model_path : str ,
        hf_token : Union[str, None] = None,
        chunk_len: int = 30,
        batch_size: int = 16
        ):

        # Init default fields       
        self.logger = logger or logging.getLogger()
        self.cuda_device = torch.device(f'cuda:{cuda_device_index}')
        self.batch_size = batch_size
        self.model_path = model_path
        self.sample_rate = 16_000
        self.model_path = model_path
        self.chunk_len = chunk_len
        self.hf_token = hf_token

        self.setup_pipeline()


    def setup_pipeline(self):
        """
        Set up the pipeline for speech-to-text inference using the Whisper model.

        Returns:
            None
        """
        # load Whisper
        self.logger.info(f'Loading Whisper model from {self.model_path}')
        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_path, 
                torch_dtype=torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True,
                token=self.hf_token
            )
            self.model = model.to(self.cuda_device)

            self.processor = AutoProcessor.from_pretrained(self.model_path)

            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                chunk_length_s=self.chunk_len,
                batch_size=self.batch_size,
                return_timestamps=True,
                torch_dtype=torch_dtype,
                device=self.cuda_device,
            )
        except Exception as e:
            self.logger.info(f'Unable to load pipeline! {e}')
            raise Exception(e)
        self.logger.info(f'Pipeline ready!')

            
    @torch.inference_mode()
    def generate(self, 
                       audio_inp: str, 
                       additional_prompt: str = None,
                       language: str = 'Italian',
                       task: str = 'transcribe',
                       temperature : float = 1.0, 
                       repetition_penalty: float = 1.0,
                       num_beams: int = 1,
                       do_sample: bool = True,
                       ):
        """
        Perform safe inference for speech-to-text using the Fast Whisper pipeline.

        Args:
            audio_inp str: Path to audio file.
            stereo_to_mono (Literal['left', 'right', 'average'], optional): Method for converting stereo audio to mono. Defaults to None.
            additional_prompt (str, optional): Additional prompt for the speech-to-text generation. Defaults to None.
            language (str, optional): Language for speech-to-text generation. Defaults to 'Italian'.
            task (str, optional): Task for speech-to-text generation. Defaults to 'transcribe'.
            temperature (float, optional): Temperature for controlling the randomness of generation. Defaults to 1.0.
            repetition_penalty (float, optional): Repetition penalty for controlling the likelihood of repeated tokens. Defaults to 1.0.
            num_beams (int, optional): Number of beams for beam search generation. Defaults to 1.

        Returns:
            Tuple[str, Dict[str, str]], Dict[str, float]]: Tuple containing the transcription, timestamps, and processing times.
        """
        
        if audio_inp is None or audio_inp[0] is None:
            raise gr.Error("ERROR: Provided audio file is empty.")
        
        # generation arguments
        # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        generate_kwargs = {
                        "language": language, 
                        "task": task, 
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "do_sample": do_sample,
                        "num_beams": num_beams
                        }
        if additional_prompt is not None:
            generate_kwargs["prompt_ids"] = self.processor.get_prompt_ids(additional_prompt)
        
        self.logger.info(f"Transcribing {audio_inp} with additional prompt: {additional_prompt} and settings {generate_kwargs}...")

        # whisper inference
        
        inf_start = time.time()
        # https://huggingface.co/docs/transformers/v4.36.1/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline
        results = self.pipeline(
            audio_inp, 
            return_timestamps=True,
            generate_kwargs=generate_kwargs)
        inf_time = time.time()-inf_start

        transcription = results['text']
        timestamps = results['chunks']

        if additional_prompt is not None:
            # we have to remove the additional prompt from the transcription 
            transcription = transcription.replace(additional_prompt, '', 1)
            # the initial prompt is always in the first chunk
            timestamps[0]['text'] = timestamps[0]['text'].replace(additional_prompt, '', 1)
            
            

            

        processing_times = {
            'whisper_inference': round(inf_time,2),
        }

        self.logger.info(transcription)
        self.logger.info(timestamps)

        return transcription, timestamps, processing_times

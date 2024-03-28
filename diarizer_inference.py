import time
import torch
import numpy as np
import logging
from typing import Union, Literal, Tuple, Union, Any, List
from utils import format_timestamps
from multiprocessing import Pool

import gradio as gr
from pyannote.audio import Pipeline

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

class PyannoteDiarizer:
    """
        Performs the diarization on a .wav file path.
        Reference at: 
            - https://github.com/pyannote/pyannote-audio

    """

    def __init__(
        self, 
        logger : Any, 
        accelerator_device_id : Union[str, int], # Either cpu, cuda, cuda:0, cuda:1, ...
        diarizer_model_path : str ,
        hf_token : Union[str, None] = None,
        ):

        # Init default fields       
        self.logger = logger or logging.getLogger()
        self.accelerator_device = torch.device(f'cuda:{accelerator_device_id}') if isinstance(accelerator_device_id, int) else accelerator_device_id
        
        self.diarizer_model_path = diarizer_model_path
        self.hf_token = hf_token

        self.setup_pipeline()


    def setup_pipeline(self,):
        """
        Set up the pipeline for the pyannote.audio diarizer.

        Returns:
            None
        """
        self.logger.info(f'Loading PyAnnote Diarizer model from {self.diarizer_model_path}')
        try:
            pipeline = Pipeline.from_pretrained(self.diarizer_model_path, use_auth_token = self.hf_token,)
            self.pipeline = pipeline.to(torch.device(self.accelerator_device))

        
        except Exception as e:
            self.logger.info(f'Unable to load pipeline! {e}')
            raise Exception(e)
        
        self.logger.info(f'Pipeline ready!')

            
    @torch.inference_mode()
    def generate(self, 
                 audio_inp: str,) -> List[dict]:

        if audio_inp is None or audio_inp[0] is None:
            raise gr.Error("ERROR: Provided audio file is empty.")
        

        result = self.pipeline(audio_inp)

        diarization_result = list()  #  [ (start_t, end_t, speaker_id) ] # start_t, end_t are delta time relative to the beginning of the file

        for segment_t, _, speaker_id in result.itertracks(yield_label=True):
            diarization_result.append( { 'start_t' : segment_t.start, 'end_t': segment_t.end, 'speaker_id': speaker_id } )

        return diarization_result
        


"""
# Example usage 
if __name__ == '__main__':
    file_path = "/path/to/file.wav"  
    HF_TOKEN = "<your token here>"
    diarizer = PyannoteDiarizer(None, accelerator_device_id = 'cpu', diarizer_model_path = "pyannote/speaker-diarization@2.1", hf_token = HF_TOKEN)
    res = diarizer.generate(file_path)

    print(res)
"""

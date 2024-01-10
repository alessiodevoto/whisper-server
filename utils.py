from typing import List, Union
import torch
import os
import logging
import sys
from datetime import datetime
from typing import Union, Any, List
from torch import nn

logger = logging.getLogger()


# Copied from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


def format_timestamps(timestamps, last_ts):
    formatted_ts = {}
    for chunk in timestamps:
        interval = '[' + str(chunk['timestamp'][0]) + ' - ' + str(chunk['timestamp'][1] or str(last_ts)) + ']'
        t = chunk['text']
        formatted_ts[interval] = t
    return formatted_ts


def legacy_format_wrapper(vad_dict: Union[dict, None]):
    if vad_dict is None:
        return None
    
    results = []
    vad_ts = []
    d = {}

    for k, v in vad_dict.items():
        ts =  'start: ' + k.replace('-', ', end:')[1:-1]
        new_str = str(k) + ':' + str(v)
        results.append(new_str)
        vad_ts.append(ts)
    

    d[f'chunk_0'] = {'results': results, 'vad_ts': vad_ts, 'diarizer_ts': None}
    
    return d





def get_duration(audio: torch.Tensor , sample_rate: int):
    """
    Get duration in seconds of provided audio track
    Args:
        audio :  torch.Tensor -> audio track
        sample_rate :  (int): sample rate of provided audio track

    Returns:
        torch.Tesor -> audio duration in seconds
    """
    return audio.shape[0]/float(sample_rate)


def get_length(seconds: float , sample_rate: int):
    """
    Get tensor length of provided audio track
    Args:
        audio :  float
        sample_rate :  (int): sample rate of provided audio track

    Returns:
        flost -> audio duration in seconds
    """
    return float(int(seconds * sample_rate + 1))


@torch.no_grad()
def get_max_gpu_capacity(models : Union[nn.ModuleList, List, Any], cuda_device: int, initial_size:int=10000, increment:float=1.5, logger=logger):
    """
    Test the maximum capacity of a cuda device by exponentially increasing the forwarded tensor size.
    Args:
        - model : torch.nn.Module -> model to be tested
        - cuda_index : int -> index of GPU where to test model
        - initial_size : int -> start fowarding tensors of this size
        - increment : float -> increment forwarded tensor by this factor at every step
        - logger -> use this logger to print info, if provided    
    Return:
        - max_gpu_capacity : Union[int, None] -> max gpu capacity or None if calculation was not possible
    """ 
    to_be_tested_max_size = tested_max_size = initial_size 

    if isinstance(cuda_device, int):
        cuda_device = torch.device(f'cuda:{cuda_device}')
    
    if not isinstance(models, nn.ModuleList) and not isinstance(models, list):
        models = [models]

    try:
        for model in models:
            model = model.to(cuda_device)
        while(True):
            x = torch.randn((1, int(tested_max_size)), dtype=torch.float32, device=cuda_device)
            for model in models:
                x = model(x)
                if isinstance(x, torch.Tensor):
                    x = x.view(1, -1)
                else:
                    x = x.logits
            logger.info(f"Forwarded tensor of size :{int(tested_max_size)}")
            tested_max_size = int(to_be_tested_max_size)
            to_be_tested_max_size *= increment 
    except RuntimeError as e:
        if 'out of memory' in str(e):
            logger.info("Reached out of memory.")
            # Free GPU cache
            del models
            del x
            torch.cuda.empty_cache()
            max_size =  int(tested_max_size / increment) # cuda error is async, so we have to divide
            return max_size 
        else:
            logger.warning(f"Unable to compute maximum capacity. The following error has occurred: {e}")
            return None


def make_logger(logs_dir: str, verbose:bool):
    """
    Make logger 
    Args:
        logs_dir : str -> log files will be saved to this directory
        verbose : bool -> whether we should also log to stdout
    Returns:
        logger -> the logger 
    """
    logger = logging.getLogger('app')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir,f'{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def escape_html(s):
    """
    Escape unicode chars to html entities for italian vowels.
    """
    s = s.replace('è', '&egrave')
    s = s.replace('é', '&eacute')
    s = s.replace('à', '&agrave')
    s = s.replace('ì', '&igrave')
    s = s.replace('ò', '&ograve')
    s = s.replace('ù', '&ugrave')
    return s


def format_dict(dic: dict):
    """
    Flatten a dictionary only until keys of depth = 2
    """
    new_dict = {}
    for key in dic:
        new_dict[key] = f"{dic[key]:.2f}"
    return new_dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
import argparse
import os
import gradio as gr
import time
import json
import torch
from utils import make_logger
from utils import format_timestamps
import datetime

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

# Paths.
workspace_dir = Path('/workspace')
default_logs_dir = workspace_dir / 'logs/'
default_models_dir = workspace_dir / 'models/'
server_crash_log = workspace_dir / 'server_crashes_log.txt'

app_dir = workspace_dir / 'fast-whisper/'
whisper_service_description = app_dir / 'fast_whisper_service_description.md'

hf_token = '<secret-token-goes-here>'   
whisper_path = 'openai/whisper-{whisper_size}'


# Max beam size for each model size.
max_beam_size = {
    'tiny': 50,
    'small': 30,
    'medium': 20,
    'large-v2': 10,
    'large-v3': 10,
    }

#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################

SUPPORTED_LANGUAGES = ['Afrikaans', 'Arabic', 'Armenian', 'Azerbaijani', 'Belarusian', 'Bosnian', 'Bulgarian', 'Catalan', 'Chinese', 'Croatian', 'Czech', 'Danish', 'Dutch', 'English', 'Estonian', 'Finnish', 'French', 'Galician', 'German', 'Greek', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic', 'Indonesian', 'Italian', 'Japanese', 'Kannada', 'Kazakh', 'Korean', 'Latvian', 'Lithuanian', 'Macedonian', 'Malay', 'Marathi', 'Maori', 'Nepali', 'Norwegian', 'Persian', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Serbian', 'Slovak', 'Slovenian', 'Spanish', 'Swahili', 'Swedish', 'Tagalog', 'Tamil', 'Thai', 'Turkish', 'Ukrainian', 'Urdu', 'Vietnamese', 'Welsh']



             
if __name__ == '__main__':

    # Parse args.
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=4020, help='port to run the server on')
    parser.add_argument('--gpu_index', type=int, default=0, help='GPU index to use')
    parser.add_argument('--max_threads', type=int, default=40, help='max threads to use for a single request')
    parser.add_argument('--logs_dir', type=str, default=default_logs_dir, help='path directory to save logs')
    parser.add_argument('--enable_corrections', action='store_true', help='enable flagging, i.e. correction of transcriptions')
    parser.add_argument('--verbose', action='store_true', help='enable verbose logging')
    parser.add_argument('--live', action='store_true', help='enable live mode')
    parser.add_argument('--model_size', type=str, default='tiny', help='model size. See README.md for more info')
    parser.add_argument('--interval', type=int, default=2, help='interval between live inference chunks')
    parser.add_argument('--online', action='store_true', help='enable online mode. This will download models from the web') 
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for inference')
    parser.add_argument('--chunk_len', type=int, default=30, help='chunk length for inference')
    parser.add_argument('--hf_token', type=str, default=hf_token, help='huggingface token for downloading models')

    args = parser.parse_args()

    if args.live:
        raise NotImplementedError("Live mode is not supported yet.")

    # Default logging configuration.
    logger = make_logger(args.logs_dir, args.verbose) 

    # Set directories to download models and offline mode.
    torch.hub.set_dir(default_models_dir)
    os.environ['TRANSFORMERS_OFFLINE'] = str(int(not args.online)) #hf_offline_mode
    os.environ['HF_HOME'] = str(default_models_dir)
    os.environ['HF_HUB_CACHE'] = str(default_models_dir)
    os.environ['TRANSFORMERS_CACHE'] = str(default_models_dir)
    server_port = args.port
    whisper_path = whisper_path.format(whisper_size=args.model_size)


    logger.info("*** Speech to Text application ***")
    logger.info(f"Offline mode {'deactivated' if args.online else 'activated'}")
    
    
    from fast_whisper_pipeline_inference import WhisperInference
    inference = WhisperInference(
        logger=logger,
        cuda_device_index=args.gpu_index, 
        model_path=whisper_path,
        batch_size=args.batch_size,
        hf_token=hf_token,
        chunk_len=args.chunk_len,
    )
  
    
    # load markdown file with service description
    service_description = None
    try:
        with open(whisper_service_description, 'r') as f:
            service_description = f.read()
    except Exception as e:
        logger.warning("Unable to read service description. The service will run anyway.")
    
    # wrapper to perform live inference
    def live_wrapper(audio_inp, state=''):
        """
        Wrapper function for live speech-to-text transcription.

        Args:
            audio_inp (str): Input audio file path.
            state (str): Current state of transcription.

        Returns:
            tuple: A tuple containing the updated state of transcription.
        """
        time.sleep(args.interval)
        transcription, _, _ = inference.generate(audio_inp)
        state = state + transcription
        return state, state
    
    # wrapper to perform async inference
    def wrapper(
              audio_inp, 
              additional_prompt, 
              language, 
              task, 
              temperature, 
              repetition_penalty, 
              num_beams,
              do_sample):
            """
            Wrapper function for the fast_whisper_app module.

            Args:
                audio_inp (str): Path to the input audio file
                additional_prompt (str): Additional prompt for the model.
                language (str): Language code for the model.
                task (str): Task type for the model.
                temperature (float): Temperature value for sampling from the model.
                repetition_penalty (float): Repetition penalty value for the model.
                num_beams (int): Number of beams for beam search decoding.

            Returns:
                tuple: A tuple containing the transcription, nice timestamps, processing times, and transcription again.
            """
            transcription, nice_timestamps, processing_times, transcription = None, None, None, None
            try:
                transcription, timestamps, processing_times = inference.generate(
                    audio_inp, 
                    additional_prompt,
                    language,
                    task,
                    temperature,
                    repetition_penalty,
                    num_beams,
                    do_sample
                    )
                nice_timestamps = json.dumps(format_timestamps(timestamps, last_ts=0.0)) if timestamps else None
                processing_times = json.dumps(processing_times) if processing_times else None
            except RuntimeError as e:
                if 'out of memory' in str(e):
                        gr.Warning('''Out of Memory Error. This can be due to memory consuming settings for inference (e.g. too large number of beams) 
                                   or an external process that is occupying the GPU. Restarting Server. Reload the page in a few seconds.''')
                        # in case of out of memory error, restart server
                        # we write the error to a file to keep track of 
                        with open(server_crash_log, 'a') as f:
                            f.write(str(datetime.datetime.now()) + '\n' + str(e) + '\n')
                else:
                    raise 
            
            return transcription, nice_timestamps, processing_times, transcription


    # Gradio interface
    live = args.live
    callback = gr.CSVLogger()
    with gr.Blocks(mode="Automatic Speech Recognition", title="Automatic Speech Recognition") as demo:
        
        # keep track for live transcription
        state = gr.State('') 
        
        if service_description is not None:
                descr = gr.Markdown(service_description)

        with gr.Row():
            with gr.Row():
                audio_file = gr.Audio(
                     label='audio', 
                     interactive=True, 
                     sources=['upload', 'microphone'] if not live else 'microphone', 
                     type='filepath', 
                     streaming=args.live)
            with gr.Column():
                language = gr.Dropdown(choices=SUPPORTED_LANGUAGES, value='Italian', label='language')
                task = gr.Radio(choices=['transcribe', 'translate'], value='transcribe', label='task')
                additional_prompt = gr.Textbox(visible=not live, label='custom prompt')
                do_sample = gr.Checkbox(value=True, label='sample')
                temperature = gr.Slider(minimum=0.1, maximum=2, value=0.1, label='temperature')
                repetition_penalty = gr.Slider(minimum=0.4, maximum=1.0, value=1.0, label='repetition penalty')
                num_beams = gr.Slider(minimum=1, maximum=max_beam_size[args.model_size], step=1, value=1, label='number of beams')
                clear_btn = gr.Button("Clear", visible=live)
                transcribe_btn = gr.Button("Start inference", visible=not live, variant='primary')

            with gr.Column():
                prediction = gr.Textbox(label='Transcription', interactive=False)
                legacy_format = gr.JSON(label='legacy_format', visible=False)
                timestamps = gr.JSON(label='Timestamps')
                processing_times = gr.JSON(label='Processing times') 

                with gr.Accordion("Correct", open=False, visible=args.enable_corrections and not live):
                    correction = gr.Textbox(label='Correction', lines=10, interactive=True)
                    correct_btn = gr.Button("Correct")
        

        # this is for flagging
        callback.setup([audio_file, prediction, correction], "corrected_transcriptions")

        # async transcription 
        transcribe_btn.click(
            wrapper, 
            inputs=[audio_file, additional_prompt, language, task, temperature, repetition_penalty, num_beams, do_sample], 
            outputs=[prediction, timestamps, processing_times, correction],
            api_name='predict',
            queue=True
            )
        correct_btn.click(lambda *args: callback.flag(args), [audio_file, prediction, correction], preprocess=False)

        # sync (live) transcription starts automatically  
        audio_file.stream(
            live_wrapper,
            inputs=[audio_file, state], 
            outputs=[prediction, state],
            api_name='live_predict' 
            )

        # TODO fix part below here with a nice reset state function
        def reset_states(audio, text, state):
            return None, '', ''
        clear_btn.click(reset_states, [audio_file, prediction, state], [audio_file, prediction, state], api_name='live_clear')

        # finally launch demo
        demo.queue(max_size=5)
        demo.launch(server_name='0.0.0.0', server_port=server_port, show_api=True, show_error=True, max_threads=args.max_threads)

    
    
        

    

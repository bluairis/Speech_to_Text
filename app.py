# Models
import torch
from transformers import Wav2Vec2Processor, HubertForCTC, Wav2Vec2Tokenizer

# Audio Manipulation
import audioread
import librosa
from pydub import AudioSegment, silence
import youtube_dl
from youtube_dl import DownloadError

# Others
from datetime import timedelta
import os
import streamlit as st
import time

def transcribe_audio_part(filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end, index):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        with torch.no_grad():
            new_audio = myaudio[sub_start:sub_end]  # Works in milliseconds
            path = filename[:-3] + "audio_" + str(index) + ".mp3"
            new_audio.export(path)  # Exports to a mp3 file in the current path

            # Load audio file with librosa, set sound rate to 16000 Hz because the model we use was trained on 16000 Hz data
            input_audio, _ = librosa.load(path, sr=16000)

            # return PyTorch torch.Tensor instead of a list of python integers thanks to return_tensors = ‘pt’
            input_values = stt_tokenizer(input_audio, return_tensors="pt").to(device).input_values

            # Get logits from the data structure containing all the information returned by the model and get our prediction
            logits = stt_model.to(device)(input_values).logits
            prediction = torch.argmax(logits, dim=-1)
           
            # Decode & lower our string (model's output is only uppercase)
            if isinstance(stt_tokenizer, Wav2Vec2Tokenizer):
                transcription = stt_tokenizer.batch_decode(prediction)[0]
            elif isinstance(stt_tokenizer, Wav2Vec2Processor):
                transcription = stt_tokenizer.decode(prediction[0])

            # return transcription
            return transcription.lower()

    except audioread.NoBackendError:
        # Means we have a chunk with a [value1 : value2] case with value1>value2
        st.error("Sorry, seems we have a problem on our side. Please change start & end values.")
        time.sleep(3)
        st.stop()

def detect_silences(audio):

    # Get Decibels (dB) so silences detection depends on the audio instead of a fixed value
    dbfs = audio.dBFS

    # Get silences timestamps > 750ms
    silence_list = silence.detect_silence(audio, min_silence_len=750, silence_thresh=dbfs-14)

    return silence_list

def get_middle_silence_time(silence_list):

    length = len(silence_list)
    index = 0
    while index < length:
        diff = (silence_list[index][1] - silence_list[index][0])
        if diff < 3500:
            silence_list[index] = silence_list[index][0] + diff/2
            index += 1
        else:

            adapted_diff = 1500
            silence_list.insert(index+1, silence_list[index][1] - adapted_diff)
            silence_list[index] = silence_list[index][0] + adapted_diff

            length += 1
            index += 2

    return silence_list


def silences_distribution(silence_list, min_space, max_space, start, end, srt_token=False):

    # If starts != 0, we need to adjust end value since silences detection is performed on the trimmed/cut audio
    # (and not on the original audio) (ex: trim audio from 20s to 2m will be 0s to 1m40 = 2m-20s)

    # Shift the end according to the start value
    end -= start
    start = 0
    end *= 1000

    # Step 1 - Add start value
    newsilence = [start]

    # Step 2 - Create a regular distribution between start and the first element of silence_list to don't have a gap > max_space and run out of memory
    # example newsilence = [0] and silence_list starts with 100000 => It will create a massive gap [0, 100000]

    if silence_list[0] - max_space > newsilence[0]:
        for i in range(int(newsilence[0]), int(silence_list[0]), max_space):  # int bc float can't be in a range loop
            value = i + max_space
            if value < silence_list[0]:
                newsilence.append(value)

    # Step 3 - Create a regular distribution until the last value of the silence_list
    min_desired_value = newsilence[-1]
    max_desired_value = newsilence[-1]
    nb_values = len(silence_list)

    while nb_values != 0:
        max_desired_value += max_space

        # Get a window of the values greater than min_desired_value and lower than max_desired_value
        silence_window = list(filter(lambda x: min_desired_value < x <= max_desired_value, silence_list))

        if silence_window != []:
            # Get the nearest value we can to min_desired_value or max_desired_value depending on srt_token
            if srt_token:
                nearest_value = min(silence_window, key=lambda x: abs(x - min_desired_value))
                nb_values -= silence_window.index(nearest_value) + 1  # (index begins at 0, so we add 1)
            else:
                nearest_value = min(silence_window, key=lambda x: abs(x - max_desired_value))
                # Max value index = len of the list
                nb_values -= len(silence_window)

            # Append the nearest value to our list
            newsilence.append(nearest_value)

        # If silence_window is empty we add the max_space value to the last one to create an automatic cut and avoid multiple audio cutting
        else:
            newsilence.append(newsilence[-1] + max_space)

        min_desired_value = newsilence[-1]
        max_desired_value = newsilence[-1]

    # Step 4 - Add the final value (end)

    if end - newsilence[-1] > min_space:
        # Gap > Min Space
        if end - newsilence[-1] < max_space:
            newsilence.append(end)
        else:
            # Gap too important between the last list value and the end value
            # We need to create automatic max_space cut till the end
            newsilence = generate_regular_split_till_end(newsilence, end, min_space, max_space)
    else:
        # Gap < Min Space <=> Final value and last value of new silence are too close, need to merge
        if len(newsilence) >= 2:
            if end - newsilence[-2] <= max_space:
                # Replace if gap is not too important
                newsilence[-1] = end
            else:
                newsilence.append(end)

        else:
            if end - newsilence[-1] <= max_space:
                # Replace if gap is not too important
                newsilence[-1] = end
            else:
                newsilence.append(end)

    return newsilence

def generate_regular_split_till_end(time_list, end, min_space, max_space):

    # In range loop can't handle float values so we convert to int
    int_last_value = int(time_list[-1])
    int_end = int(end)

    # Add maxspace to the last list value and add this value to the list
    for i in range(int_last_value, int_end, max_space):
        value = i + max_space
        if value < end:
            time_list.append(value)

    # Fix last automatic cut
    # If small gap (ex: 395 000, with end = 400 000)
    if end - time_list[-1] < min_space:
        time_list[-1] = end
    else:
        # If important gap (ex: 311 000 then 356 000, with end = 400 000, can't replace and then have 311k to 400k)
        time_list.append(end)
    return time_list

def clean_directory(path):

    for file in os.listdir(path):
        os.remove(os.path.join(path, file))

def config():

    st.set_page_config(page_title="Speech to Text", page_icon="📝")
    
    # Create a data directory to store our audio files
    # Will not be executed with AI Deploy because it is indicated in the DockerFile of the app
    if not os.path.exists("../data"):
        os.makedirs("../data")
    
    # Display Text and CSS
    st.title("Speech to Text App 📝")

    st.subheader("You want to extract text from an audio/video? You are in the right place!")

@st.cache(allow_output_mutation=True)
def load_models():

    # Load Wav2Vec2 (Transcriber model)
    stt_model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")
    stt_tokenizer = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")

    return stt_tokenizer, stt_model

def transcript_from_file(stt_tokenizer, stt_model):

    uploaded_file = st.file_uploader("Upload your file! It can be a .mp3, .mp4 or .wav", type=["mp3", "mp4", "wav"])

    if uploaded_file is not None:
        # get name and launch transcription function
        filename = uploaded_file.name
        transcription(stt_tokenizer, stt_model, filename, uploaded_file)

def extract_audio_from_yt_video(url):
    
    filename = "yt_download_" + url[-11:] + ".mp3"
    try:

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': filename,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }
        with st.spinner("We are extracting the audio from the video"):
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

    # Handle DownloadError: ERROR: unable to download video data: HTTP Error 403: Forbidden / happens sometimes
    except DownloadError:
        filename = None

    return filename

def transcript_from_url(stt_tokenizer, stt_model):
    
    url = st.text_input("Enter the YouTube video URL then press Enter to confirm!")
    
    # If link seems correct, we try to transcribe
    if "youtu" in url:
        filename = extract_audio_from_yt_video(url)
        if filename is not None:
            transcription(stt_tokenizer, stt_model, filename)
        else:
            st.error("We were unable to extract the audio. Please verify your link, retry or choose another video")

def init_transcription(start, end):
    
    st.write("Transcription between", start, "and", end, "seconds in process.\n\n")
    txt_text = ""
    srt_text = ""
    save_result = []
    return txt_text, srt_text, save_result

def transcription_non_diarization(filename, myaudio, start, end, srt_token, stt_model, stt_tokenizer, min_space, max_space, save_result, txt_text, srt_text):
    
    # get silences
    silence_list = detect_silences(myaudio)
    if silence_list != []:
        silence_list = get_middle_silence_time(silence_list)
        silence_list = silences_distribution(silence_list, min_space, max_space, start, end, srt_token)
    else:
        silence_list = generate_regular_split_till_end(silence_list, int(end), min_space, max_space)

    # Transcribe each audio chunk (from timestamp to timestamp) and display transcript
    for i in range(0, len(silence_list) - 1):
        sub_start = silence_list[i]
        sub_end = silence_list[i + 1]

        transcription = transcribe_audio_part(filename, stt_model, stt_tokenizer, myaudio, sub_start, sub_end, i)
        
        if transcription != "":
            save_result, txt_text, srt_text = display_transcription(transcription, save_result, txt_text, srt_text, sub_start, sub_end)

    return save_result, txt_text, srt_text      

def display_transcription(transcription, save_result, txt_text, srt_text, sub_start, sub_end):

    temp_timestamps = str(timedelta(milliseconds=sub_start)).split(".")[0] + " --> " + str(timedelta(milliseconds=sub_end)).split(".")[0] + "\n"        
    temp_list = [temp_timestamps, transcription, int(sub_start / 1000)]
    save_result.append(temp_list)
    st.write(temp_timestamps)    
    st.write(transcription + "\n\n")
    txt_text += transcription + " "  # So x seconds sentences are separated

    return save_result, txt_text, srt_text 

def transcription(stt_tokenizer, stt_model, filename, uploaded_file=None):

    # If the audio comes from the YouTube extracting mode, the audio is downloaded so the uploaded_file is
    # the same as the filename. We need to change the uploaded_file which is currently set to None
    if uploaded_file is None:
        uploaded_file = filename

    # Get audio length of the file(s)
    myaudio = AudioSegment.from_file(uploaded_file)
    audio_length = myaudio.duration_seconds
    
    # Display audio file
    st.audio(uploaded_file)

    # Is transcription possible
    if audio_length > 0:
        
        # display a button so the user can launch the transcribe process
        transcript_btn = st.button("Transcribe")

        # if button is clicked
        if transcript_btn:

            # Transcribe process is running
            with st.spinner("We are transcribing your audio. Please wait"):

                # Init variables
                start = 0
                end = audio_length
                txt_text, srt_text, save_result = init_transcription(start, int(end))
                srt_token = False
                min_space = 25000
                max_space = 45000


                # Non Diarization Mode
                filename = "../data/" + filename
                
                # Transcribe process with Non Diarization Mode
                save_result, txt_text, srt_text = transcription_non_diarization(filename, myaudio, start, end, srt_token, stt_model, stt_tokenizer, min_space, max_space, save_result, txt_text, srt_text)

                # Delete files
                clean_directory("../data")  # clean folder that contains generated files

                # Display the final transcript
                if txt_text != "":
                    st.subheader("Final text is")
                    st.write(txt_text)

                else:
                    st.write("Transcription impossible, a problem occurred with your audio or your parameters, we apologize :(")

    else:
        st.error("Seems your audio is 0 s long, please change your file")
        time.sleep(3)
        st.stop()

if __name__ == '__main__':
    config()
    choice = st.radio("Features", ["By a video URL", "By uploading a file"]) 

    stt_tokenizer, stt_model = load_models()
    if choice == "By a video URL":
        transcript_from_url(stt_tokenizer, stt_model)

    elif choice == "By uploading a file":
        transcript_from_file(stt_tokenizer, stt_model)
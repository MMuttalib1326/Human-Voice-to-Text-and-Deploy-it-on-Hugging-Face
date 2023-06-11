import whisper
import gradio as gr 
import time
model = whisper.load_model("base")
#from transformers import pipeline
#es_en_translator = pipeline("translation_es_to_en")

def transcribe(audio):
    
    #time.sleep(3)
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    #lang = LANGUAGES[language]
    #lang=(f"Detected language: {lang}")


    # decode the audio
    options = whisper.DecodingOptions(fp16 = False)#,task= "translate")
    result = whisper.decode(model, mel, options)
    #word= result.text
    #trans = es_en_translator(word)
    #Trans = trans[0]['translation_text']
    #result=f"{lang}\n{word}\n\nEnglish translation: {Trans}"
    return result.text
    
    
 
gr.Interface(
    title = 'SPEECH TO TEXT', 
    fn=transcribe, 
    inputs=[
        gr.inputs.Audio(source="microphone", type="filepath")
    ],
    outputs=[
        "textbox"
    ],
    live=True).launch()
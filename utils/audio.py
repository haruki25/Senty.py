import speech_recognition as sr


# Function to transcribe audio file
def transcribe_audio(audio_file):
    r = sr.Recognizer()

    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)

        text = r.recognize_google(audio_data)
        return text, None
    except sr.UnknownValueError:
        return (
            None,
            "Speech recognition could not understand the audio. Please ensure the audio is clear and try again.",
        )
    except sr.RequestError as e:
        return (
            None,
            f"Could not request results from the speech recognition service. Error: {str(e)}",
        )
    except Exception as e:
        return None, f"An unexpected error occurred: {str(e)}"

import model
import streamlit as st
import os


# Set the theme to dark
st.set_page_config(page_title="Talk to text", page_icon=":guardsman:", initial_sidebar_state="expanded")

# Optional: Add this to change the theme to dark mode
st.markdown(
    """
    <style>
    .css-1y4k23f {
        background-color: #121212; /* Dark background */
        color: #ffffff; /* Light text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your Streamlit app code here
st.title("Talk to text")



def save_audio_file(audio_bytes, file_extension):
    file_name = f"audio.{file_extension}"

    with open(file_name, "wb") as f:
        f.write(audio_bytes)

    return file_name


# Get the absolute path to the model directory
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, 'img2.png')


st.image(image_path, use_column_width=True)

tab1, tab2 = st.tabs(["Record Audio", "Upload Audio"])

# Record Audio tab
with tab1:
    recorded_audio = st.experimental_audio_input("Record a voice message")
    if recorded_audio is not None:
        try:
            audio_bytes = recorded_audio.getvalue()
            save_audio_file(audio_bytes, "wav")
        except:
            st.write("Re-record the audio.")

# Upload Audio tab
with tab2:
    uploaded_audio = st.file_uploader("Upload a file", type=["mp3", "wav"])
    if uploaded_audio is not None:
        try:
            audio_bytes = uploaded_audio.getvalue()
            st.audio(audio_bytes, format="audio/wav")
            save_audio_file(audio_bytes, "wav")
        except:
            st.write("File is corrupted.")
        

# Transcribe button action
if st.button("Transcribe"):
    transcript_text = ''
    # Transcribe the audio file
    with st.spinner('In progress...'):
        try:
            transcript_text = model.get_transcript('audio.wav')
        except:
            st.write("An error accurred during transcription.")

    # Display the transcript
    st.header("Transcript")
    st.write(transcript_text)

    # Save the transcript to a text file
    with open("transcript.txt", "w") as f:
        f.write(transcript_text)

    # Provide a download button for the transcript
    st.download_button("Download Transcript", transcript_text)
    
    
    
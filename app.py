
import streamlit as st
from util.html_blocks import  css
import util.chatflow as chat
from util.voice import Record, Whisper


        
LLAMA2_7B_PATH = "models/TheBloke-Llama-2-7b-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"
    



#LLM Components
#documents = data.Document_Loader('https://ridebeep.com/','/home/jramos/Documents/Beep Folder/Streamlit App/test')


vector_storage = chat.get_vectorstore(load_from_s disk=True)
llm = chat.get_llm(LLAMA2_7B_PATH)
memory = chat.get_memory(llm)

agent = chat.get_agent(llm,vector_storage ,memory)



whisper = Whisper()

#UI Interactivity
st.session_state.conversation = agent 
                    
st.write(css, unsafe_allow_html=True)
                
if "conversation" not in st.session_state:
        st.session_state.conversation = None

if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
                
st.header('Beep Assistant')
    
if 'clicked' not in st.session_state:
        st.session_state.clicked = False

def click_button():
        st.session_state.clicked = True

        
question = st.text_input("Type in question: ")

st.button('Record',on_click=click_button)

if st.session_state.clicked:
        recording = Record(16000)
        speech = recording.output()[:,0]
        sample_rate = recording.freq
        transcription = whisper.process(speech,sample_rate)
        print(transcription)
        question = transcription
       
if question:
    chat.handle_user_input(question)
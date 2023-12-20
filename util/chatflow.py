# https://www.pinecone.io/learn/llama-2/#Building-a-Llama-2-Conversational-Agent

# Create chatbot
import datetime
import streamlit as st
from util.html_blocks import bot_template, user_template

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool, tool
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp


def get_vectorstore(load_from_disk=True, docs=None):

    print(f"[get_vectorstore] ENTER")
    
    # - if you save the FAISS db using a certain embedding, the same one has to be used
    #   when loading it
    # - so just use sentence-transformers/all-MiniLM-L6-v2 all the time
    # - sentence-transformers/all-MiniLM-L6-v2 is fast. 3 mins for the entire dataset
    # - thenlper/gte-small is too slow
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if load_from_disk:
        vectorstore = FAISS.load_local("vectorstore.faiss", embeddings)
    else:
        # [x] todo: chroma vs pinecone vs FAISS
        # FAISS is good just save it to a file
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        now_str = datetime.datetime.utcnow().isoformat().replace(":", "_")
        vectorstore.save_local(f"vectorstore.faiss.{now_str}")
    
    print(f"[get_vectorstore] EXIT")
    
    return vectorstore



def get_llm(path=None):
    '''

                        HUGGINGFACE OPEN SOURCE MODELS

    '''

    #LLAMA2_13B_PATH="/home/ubuntu/langchain/models/TheBloke-Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q3_K_S.gguf"
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        
    llm = LlamaCpp(
    # model_path=CODELLAMA_7B_PATH,
    # model_path=LLAMA2_7B_PATH,
        model_path=path,#LLAMA2_13B_PATH,
        temperature=0.0,
        max_tokens=4000, # 2000
        n_batch=4000, # 1024
        top_p=1,
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
        n_ctx=4000 # 1024
)
    return llm

def get_memory(llm):
    memory = ConversationBufferMemory(llm=llm,memory_key="chat_history",return_messages=True)
    
    return memory

def get_agent(llm,vectorstore,memory):
        
    from langchain.tools import tool

    @tool
    def search_api(query: str) -> str:
        """Searches the API for the query."""
        return f"fjeukjcks is a type of camel"
        

    @tool
    def emergency(emergency_information: str) -> bool:
        """
        Notifies the appropriate parties about an emergency with information provided
        in emergency_information. Returns a boolean indicating whether or not the
        appropriate parties were successfully notified.
        """
        print(f"emergency_information: {emergency_information}")
        return True


    from langchain.chains import ConversationalRetrievalChain
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
    from langchain.tools import BaseTool, StructuredTool, Tool, tool
    from langchain.vectorstores import FAISS
    retriever = vectorstore.as_retriever()
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    _qa_tool = Tool(
        name="_qa_tool",
        func=conversation_chain.run,
        description="Use this tool for any all questions about . Except emergencies. If there's an emergency, use a different tool. Use this for any question.",
        #return_direct=True
    )

    tools = [
        # search_api,
        _qa_tool,
        emergency
    ]

    # initialize conversational agent

    from langchain.agents import initialize_agent

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        early_stopping_method="generate",
        memory=memory
    )

    # Setup prompt for Llama2

    # special tokens used by llama 2 chat
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    # create the system message
    sys_msg = "<s>" + B_SYS + """Assistant is a expert JSON builder designed to assist with a wide range of tasks.

    Assistant is able to respond to the User and use tools using JSON strings that contain "action" and "action_input" parameters.

    All of Assistant's communication is performed using this JSON format.

    Assistant can also use tools by responding to the user with tool use instructions in the same "action" and "action_input" JSON format. Tools available to Assistant are:

    - "emergency": Notifies the appropriate parties about an emergency with information provided.
        Returns a boolean indicating whether or not the appropriate parties were successfully notified.
    - To use the emergency tool, Assistant should write like so:
        ```json
        {{"action": "emergency",
        "action_input": "The passenger is not breathing"}}
        ```
    - "_qa_tool": Use this tool for any all questions about . Except emergencies. If there's an emergency, use a different tool. Use this for any question.
        Returns the answer to the question.
    - To use the emergency tool, Assistant should write like so:
        ```json
        {{"action": "_qa_tool",
        "action_input": "What is ?"}}
        ```

    Here are some previous conversations between the Assistant and User:

    User: Hey how are you today?
    Assistant: ```json
    {{"action": "Final Answer",
    "action_input": "I'm good thanks, how are you?"}}
    ```
    User: The car's tire is flat. What should we do?
    Assistant: ```json
    {{"action": "emergency",
    "action_input": "The car's tire is flat."}}
    ```
    User: True
    Assistant: ```json
    {{"action": "Final Answer",
    "action_input": "The appropriate parties have been notified of your emergency."}}
    ```
    User: Does  sell robotic vacuums?
    Assistant: ```json
    {{"action": "_qa_tool",
    "action_input": "Does  sell robotic vacuums?"}}
    ```
    User: Based on the provided context, there is no information indicating that  sells robotic vacuums.
    Assistant: ```json
    {{"action": "Final Answer",
    "action_input": "Based on the provided context, there is no information indicating that  sells robotic vacuums."}}
    ```

    Here is the latest conversation between Assistant and User.""" + E_SYS
    new_prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = new_prompt

    # Address the forgetfulness problem

    instruction = B_INST + " Respond to the following in JSON with 'action' and 'action_input' values " + E_INST
    human_msg = instruction + "\nUser: {input}"

    agent.agent.llm_chain.prompt.messages[2].prompt.template = human_msg

    return agent



def handle_user_input(question):
    response = st.session_state.conversation(question)
    st.session_state.chat_history = response['chat_history']
    
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




import datetime
import json
import os

from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.document_loaders import DirectoryLoader, WebBaseLoader, PyPDFLoader, TextLoader, BSHTMLLoader, CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
import requests
from tqdm import tqdm


def get_urls(url):
    
    dict_href_urls = {}
    reqs = requests.get(url)
    html_data = reqs.text
    soup = BeautifulSoup(html_data, "html.parser")
    list_urls = []
    for link in soup.find_all("a", href=True):
        
        # Append to list if new url contains original url
        if str(link["href"]).startswith((str(url))):
            list_urls.append(link["href"])
            
        # Include all href that do not start with website url but with "/"
        if str(link["href"]).startswith("/"):
            if link["href"] not in dict_href_urls:
                #print(link["href"])
                dict_href_urls[link["href"]] = None
                url_with_www = url + link["href"][1:]
                #print("adjusted url =", url_with_www)
                list_urls.append(url_with_www)
                
    # Convert list of urls to dictionary and define keys as the urls and the values as "Not-checked"
    dict_urls = dict.fromkeys(list_urls, "Not-checked")

    return dict_urls


def get_subpage_urls(parent_url):
    
    for url in tqdm(parent_url):
        # If not crawled through this page start crawling and get urls
        if parent_url[url] == "Not-checked":
            dict_urls_subpages = get_urls(url) 
            # Change the dictionary value of the url to "Checked"
            parent_url[url] = "Checked"
        else:
            # Create an empty dictionary in case every url is checked
            dict_urls_subpages = {}
        # Add new dictionary to old dictionary
        parent_url = {**dict_urls_subpages, **parent_url}
    return parent_url


def get_all_urls(root_url):

    print(f"[get_all_urls]: ENTER")
    
    # create dictionary of website
    dict_urls = {root_url:"Not-checked"}
    
    counter, counter2 = None, 0
    while counter != 0:
        counter2 += 1
        dict_urls2 = get_subpage_urls(dict_urls)
        # Count number of non-values and set counter to 0 if there are no values within the dictionary equal to the string "Not-checked"
        # https://stackoverflow.com/questions/48371856/count-the-number-of-occurrences-of-a-certain-value-in-a-dictionary-in-python
        counter = sum(value == "Not-checked" for value in dict_urls2.values())
        # Print some statements
        #print("")
        #print("THIS IS LOOP ITERATION NUMBER", counter2)
        #print("LENGTH OF DICTIONARY WITH urlS =", len(dict_urls2))
        #print("NUMBER OF 'Not-checked' urlS = ", counter)
        #print("")
        dict_urls = dict_urls2

    urls = list(dict_urls.keys())

    print(f"[get_all_urls]: EXIT")
    return urls


def get_documents(path, urls=None):

    print(f"[get_documents] ENTER")
    
    web_loader = WebBaseLoader(urls)
    csv_loader = DirectoryLoader(path, glob='**/*.csv', loader_cls=CSVLoader)
    html_loader = DirectoryLoader(path, glob='**/*.html', loader_cls=BSHTMLLoader)
    pdf_loader = DirectoryLoader(path, glob='**/*.pdf', loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(path, glob='**/*.txt', loader_cls=TextLoader)
    
    loaders = []
    loaders.append(web_loader)
    loaders.append(csv_loader)
    loaders.append(html_loader)
    loaders.append(pdf_loader)
    loaders.append(txt_loader)
    
    data = []
    
    # load data from files
    for loader in loaders:
        data.extend(loader.load())

    # text split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    
    docs = text_splitter.split_documents(data)

    print(f"[get_documents] EXIT")
    return docs


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


def get_conv_chain(vectorstore):

    LLAMA2_13B_PATH="models/TheBloke-Llama-2-13B-chat-GGUF/llama-2-13b-chat.Q3_K_S.gguf"
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    
    llm = LlamaCpp(
        model_path=LLAMA2_13B_PATH,
        temperature=0.0,
        max_tokens=2000,
        n_batch=1024,
        top_p=1,
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
        n_ctx=1024
    )

    # todo: ConversationSummaryMemory (no good) vs ConversationBufferMemory vs ConversationSUmmaryBefferMemory
    #memory = ConversationSummaryMemory(llm=llm,memory_key="chat_history",return_messages=True)
    memory = ConversationBufferMemory(llm=llm,memory_key="chat_history",return_messages=True)
    retriever = vectorstore.as_retriever()
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    return conv_chain
    
    


def main():
    # add website WITH slash at the end
    website = 'Enter URL'
    path = './docs'

    USE_EXISTING_VECTORSTORE = False
    print(f"USE_EXISTING_VECTORSTORE: {USE_EXISTING_VECTORSTORE}")

    if USE_EXISTING_VECTORSTORE:
        vs = get_vectorstore(load_from_disk=True)
    else:
        urls = get_all_urls(website)
        docs = get_documents(path, urls)
        vs = get_vectorstore(load_from_disk=False, docs=docs)
        
    conv_chain = get_conv_chain(vs)

   
    answer = conv_chain("Enter Question")
    print(f"answer: {answer}")



if __name__ == '__main__':
    main()
import argparse
import os
from typing import Optional
from pydantic import Field
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from fastapi import FastAPI, Query, File, UploadFile

from load_data import load_split_pdf_file, load_split_html_file, initialize_splitter
# from load_llm import load_lamma_cpp
from vector_db import create_vector_db, load_local_db
from prompts import create_prompt
from utils import read_file
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

# curl http://localhost:11434/api/chat -d '{"model": "llama3.1", "messages": [{ "role": "user", "content": "why is the sky blue?" }]}'
st.set_page_config(page_title="Chatbot")
# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TheBloke/Llama-2-7b-Chat-AWQ",
#     task="text-generation",
#     pipeline_kwargs=dict(
#         max_new_tokens=512,
#         do_sample=False,
#         repetition_penalty=1.03,
#     ),
# )


# # chat_model = ChatHuggingFace(llm=llm)
# from awq import AutoAWQForCausalLM
# from transformers import AutoTokenizer

# model_name_or_path = "TheBloke/neural-chat-7B-v3-1-AWQ"

# # Load model
# model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=False,
#                                           trust_remote_code=False, safetensors=True, device_map='auto')
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)


def fake_output(x: float):
    return "Answer to this query is 42"

ml_models = {}
db_name = {}
text_splitter = initialize_splitter(chunk_size = 1000, chunk_overlap = 100)
vector_db_model_name = "test_db"

app = FastAPI(
    title="RAG_APP",
    description="Retrival Augmented Generation APP which let's user get the answer for the question using LLMs",
)

@app.get("/health")
def index():
    return {"message": "The server is up and running"}



# # the model initialized when the app gets loaded but we can configure it if we want
# @app.get("/init_llm")
# def init_llama_llm(n_gpu_layers: int = Query(500, description="Number of layers to load in GPU"),
#                 n_batch: int = Query(32, description="Number of tokens to process in parallel. Should be a number between 1 and n_ctx."),
#                 max_tokens: int = Query(300, description="The maximum number of tokens to generate."),
#                 n_ctx: int = Query(4096, description="Token context window."),
#                 temperature: int = Query(0, description="Temperature for sampling. Higher values means more random samples.")):
#     model_path = model_args["model_path"]
#     model_args = {'model_path' : model_path,
#                   'n_gpu_layers': n_gpu_layers,
#                   'n_batch': n_batch,
#                   'max_tokens': max_tokens,
#                   'n_ctx': n_ctx,
#                   'temperature': temperature,
#                   'device': device}
#     llm = load_lamma_cpp(model_args)
#     ml_models["answer_to_query"] = llm
#     return {"message": "LLM initialized"}

@app.post("/upload")
def upload_file(file: UploadFile = File(...), collection_name : Optional[str] = "test_collection"):
    try:
        contents = file.file.read()
        with open(f'../data/{file.filename}', 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
    if file.filename.endswith('.pdf'):
        data = load_split_pdf_file(f'../data/{file.filename}', text_splitter)
    elif file.filename.endswith('.html'):
        data = load_split_html_file(f'../data/{file.filename}', text_splitter)
    else:
        return {"message": "Only pdf and html files are supported"}
    
    db = create_vector_db(data, vector_db_model_name, collection_name)


    return {"message": f"Successfully uploaded {file.filename}", 
            "num_splits" : len(data)}


@app.get("/query")
def query(query : str, n_results : Optional[int] = 2, collection_name : Optional[str] = "test_collection"):
    SYSTEM_MSG = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Alsways provide the answer in the saame language in which the question is asked.\n\n"
    
    # try:
    #     collection_list = read_file('COLLECTIONS.txt')
    #     collection_list = collection_list.split("\n")[:-1]
    # except Exception:
    #     return {"message": "No collections found uplaod some documents first"}

    # if collection_name not in collection_list:
    #     return {"message": f"There is no collection with name {collection_name}",
    #             "available_collections" : collection_list}
    collection = load_local_db(collection_name)
    retriever = collection.as_retriever(
    search_type="mmr", search_kwargs={"k": 1, "fetch_k": 1}
    )
    results = retriever.invoke(query)
    SYSTEM_MSG = SYSTEM_MSG + results[0].page_content
    # messages = [
    #     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    #     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    #     {"role": "user", "content": SYSTEM_MSG + '\n\n' + query},
    # ]
    prompt_template=f'''<|im_start|>system
    {SYSTEM_MSG}<|im_end|>
    <|im_start|>user
    {query}<|im_end|>
    <|im_start|>assistant

    '''
    print("\n\n*** Generate:")

    tokens = tokenizer(
        prompt_template,
        return_tensors='pt'
    ).input_ids.cuda()

    # Generate output
    generation_output = model.generate(
        tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512
    )

    return tokenizer.decode(generation_output[0]).split('assistant')[1]
    # results = collection.query(query_texts=[query], n_results = n_results)
    # print("results: ", results)
    # prompt = create_prompt(query, results)
    # print("=================================")
    # print(prompt)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=collection.as_retriever()
    # )

    # response = qa_chain.invoke({"query": query})
    # return response["result"] 

    # output = ml_models["answer_to_query"](prompt)
    # return {"message": f"Query is {query}",
    #         "relavent_docs" : results,
    #         "llm_output" : output}


collection = load_local_db("test_collection")

def get_response(query, chat_history):

    retriever = collection.as_retriever(
    search_type="mmr", search_kwargs={"k": 5, "score_threshold": 0.5}
    )

    context = ""
    results = retriever.invoke(query)
    for res in results:
        # print("=================")

        # print(res.page_content)
        context += res.page_content
    template = """
    You are a helpful assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer the question in the same language asked:
    Q: what all is prohibited during the use of the car?
    A: According to the provided text, the following are prohibited during the use of the car: Smoking (smoking) Eating and drinking inside the vehicle Additionally, private rides are generally not allowed, except in certain circumstances , such as when traveling between home and work to attend an out-of-town meeting.
    Q: was ist bei der Nutzung des Autos alles verboten?
    A: Gemäß dem bereitgestellten Text ist während der Nutzung des Autos Folgendes verboten:
    Rauchen
    Der Verzehr von Speisen und Getränken im Fahrzeug
    Darüber hinaus sind private Fahrten grundsätzlich nicht gestattet, außer unter bestimmten Umständen, beispielsweise bei Fahrten zwischen Wohnung und Arbeit, um an einem auswärtigen Meeting teilzunehmen.


    context: {context}

    User question: {query}

    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = Ollama(model="llama3.1")
    chain = prompt | llm | StrOutputParser()

    # return chain.stream({
    #     "context": context,
    #     "chat_history": chat_history,
    #     "query": query
    # })
    return chain.stream({
    "context": context,
    "query": query
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, How can I help you?")
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


query = st.chat_input("Type your message here...")
if query is not None and query != "":
    st.session_state.chat_history.append(HumanMessage(content=query))

    with st.chat_message("human"):
        st.markdown(query)

    with st.chat_message("AI"):
        response = st.write_stream(get_response(query, st.session_state.chat_history))
    st.session_state.chat_history.append(AIMessage(content=response))
    
            



# if __name__ == "__main__":
#     pass


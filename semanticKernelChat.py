# coding: shift_jis
import os
import asyncio
import streamlit as st
import uuid
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextCompletion, OpenAITextEmbedding, OpenAIChatCompletion
from streamlit_chat import message

DBA = {}  # DBとして使う

kernel = sk.Kernel()

os.environ['OPENAI_API_KEY'] = "APIキーを入力してください"
api_key = os.getenv("OPENAI_API_KEY")

kernel.add_chat_service("chat-gpt", OpenAIChatCompletion("gpt-3.5-turbo", api_key))
kernel.add_text_embedding_generation_service("ada", OpenAITextEmbedding("text-embedding-ada-002", api_key))

kernel.register_memory_store(memory_store=sk.memory.VolatileMemoryStore())

prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
    max_tokens=2000, temperature=0.7, top_p=0.8
)

prompt_template = sk.ChatPromptTemplate("""Chat:
        {{$chat_history}}
        User: {{$user_input}}
        ChatBot:""", kernel.prompt_template_engine, prompt_config)

prompt_template.add_system_message("""
        あなたはケロちゃんです。あなたはフレンドリーな AI チャットボットで、ユーザーと積極的に話します。
        ユーザーの質問にフレンドリーに関西弁で答えてください。短く答えてください。
        """)

function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template) 


chat_function = kernel.register_semantic_function("KeroChan", "Chat", function_config)


async def chat(user_id, message):
    
    context_vars = DBA.get(user_id, sk.ContextVariables(variables=dict(
            user_input="",
            chat_history="",
        )))
        
    context_vars["user_input"] = message
    talk = f'{user_id}: ' + context_vars["user_input"]
    print(talk)
    answer = await kernel.run_async(chat_function, input_vars=context_vars)
    print('Assistaunt: ' + answer.result)
    await kernel.memory.save_information_async(user_id, talk, uuid.uuid4())
    context_vars["chat_history"] += f"\n{talk}\nAssistaunt:> {answer.result}\n"
    DBA[user_id] = context_vars

    return answer.result

async def conversational_chat(user_id, message):

    result = await chat(user_id, message)
    
    st.session_state['history'].append((message, result))
    return result

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["ワイはケロちゃんやで。よろしくな！"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["こんにちは！"]

response_container = st.container()
container = st.container()



async def main():
  with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Query:", placeholder="チャットしましょう！", key='input')
        submit_button = st.form_submit_button(label='送る')
        
    if submit_button and user_input:
        output = await conversational_chat('user-1', user_input)

        print(DBA)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="fun-emoji")
                message(st.session_state["generated"][i], key=str(i), avatar_style="adventurer-neutral")
                


asyncio.run(main())

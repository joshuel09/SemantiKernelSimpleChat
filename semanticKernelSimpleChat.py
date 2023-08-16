# coding: shift_jis
import os
import asyncio
import uuid
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding, OpenAIChatCompletion

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

async def main():

    async def chat(user_id, message):
        context_vars = DBA.get(user_id, sk.ContextVariables(variables=dict(
            user_input="",
            chat_history="",
        )))
        context_vars["user_input"] = message
        talk = f'{user_id}: ' + context_vars["user_input"]
        print(talk)
        answer = await kernel.run_async(chat_function, input_vars=context_vars)
        print('ケロちゃん: ' + answer.result)
        await kernel.memory.save_information_async(user_id, talk, uuid.uuid4())
        context_vars["chat_history"] += f"\n{talk}\nケロちゃん:> {answer.result}\n"
        DBA[user_id] = context_vars
        return "\n".join([talk, 'ケロちゃん: ' + answer.result])
    
    await chat('ユーザー1', 'こんにちはあなたは誰ですか?')

    await chat('ユーザー1', '僕はジョシュです！よろしく！')
    await chat('ユーザー2', '僕はれんげといいます！よろしくにゃ！')

    await chat('ユーザー1', 'あれ僕の名前は何だっけ？')
    await chat('ユーザー2', '僕の名前は何ですか？')

    await chat('ユーザー1', '僕は今日フライドチキンを食べたよ')
    await chat('ユーザー2', '僕は今日ゲームをしました。')

    await chat('ユーザー1', '僕は何を食べたか?')
    await chat('ユーザー2', '僕は何を食べましたか?')
    memory = await kernel.memory.search_async('ユーザー1', 'ユーザーの今日の活動')
    print("メモリー: " + memory[0].text)

    memory = await kernel.memory.search_async('ユーザー2', 'ユーザーの今日の活動')
    print("メモリー: " + memory[0].text)


if __name__ == "__main__":
    asyncio.run(main())
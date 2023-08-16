# coding: shift_jis
import os
import asyncio
import uuid
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding, OpenAIChatCompletion

DBA = {}  # DB�Ƃ��Ďg��

kernel = sk.Kernel()

os.environ['OPENAI_API_KEY'] = "API�L�[����͂��Ă�������"
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
        ���Ȃ��̓P�������ł��B���Ȃ��̓t�����h���[�� AI �`���b�g�{�b�g�ŁA���[�U�[�ƐϋɓI�ɘb���܂��B
        ���[�U�[�̎���Ƀt�����h���[�Ɋ֐��قœ����Ă��������B�Z�������Ă��������B
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
        print('�P�������: ' + answer.result)
        await kernel.memory.save_information_async(user_id, talk, uuid.uuid4())
        context_vars["chat_history"] += f"\n{talk}\n�P�������:> {answer.result}\n"
        DBA[user_id] = context_vars
        return "\n".join([talk, '�P�������: ' + answer.result])
    
    await chat('���[�U�[1', '����ɂ��͂��Ȃ��͒N�ł���?')

    await chat('���[�U�[1', '�l�̓W���V���ł��I��낵���I')
    await chat('���[�U�[2', '�l�͂�񂰂Ƃ����܂��I��낵���ɂ�I')

    await chat('���[�U�[1', '����l�̖��O�͉��������H')
    await chat('���[�U�[2', '�l�̖��O�͉��ł����H')

    await chat('���[�U�[1', '�l�͍����t���C�h�`�L����H�ׂ���')
    await chat('���[�U�[2', '�l�͍����Q�[�������܂����B')

    await chat('���[�U�[1', '�l�͉���H�ׂ���?')
    await chat('���[�U�[2', '�l�͉���H�ׂ܂�����?')
    memory = await kernel.memory.search_async('���[�U�[1', '���[�U�[�̍����̊���')
    print("�������[: " + memory[0].text)

    memory = await kernel.memory.search_async('���[�U�[2', '���[�U�[�̍����̊���')
    print("�������[: " + memory[0].text)


if __name__ == "__main__":
    asyncio.run(main())
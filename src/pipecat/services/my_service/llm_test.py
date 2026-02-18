import asyncio
import dotenv
import os
from dotenv import load_dotenv

load_dotenv()

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.my_service.llm import MyLLMService
from pipecat.services.my_service.tts import MyTTSService



async def main():
    conversation = [{
        "content": "You are a helpful assistant. Answer to everthing in single sentence",
        "role": "system"
    }]
    llm = MyLLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")


    messages = [{
        "content": "Hello Im thilak",
        "role": "user"
    }]
    conversation.append({"content": messages[0].get('content'), "role": messages[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    
    second_message = [{
        "content": "Im a software engineer by profession, cricketer by passion",
        "role": "user"
    }]
    conversation.append({"content": second_message[0].get('content'), "role": second_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})


    third_message = [{
        "content": "I'm playing proper red ball cricket and weekend league's both",
        "role": "user"
    }]
    # print("after third message")
    conversation.append({"content": third_message[0].get('content'), "role": third_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    fourth_message = [{
        "content": "Im an all rounder who can bat some, bowl some, and field some.",
        "role": "user"
    }]
    # print("after fourth message")
    conversation.append({"content": fourth_message[0].get('content'), "role": fourth_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation 



    fifth_message = [{
        "content": "Im a medium pace bowler and right arm batsman",
        "role": "user"
    }]
    # print("after fifth message")
    conversation.append({"content": fifth_message[0].get('content'), "role": fifth_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    sixth_message = [{
        "content": "I'm a new ball bowler some times when it comes to bowling.",
        "role": "user"
    }]
    conversation.append({"content": sixth_message[0].get('content'), "role": sixth_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    seventh_message = [{
        "content": "My role is I need to bowl in correct line and lengths and I try to do that",
        "role": "user"
    }]
    conversation.append({"content": seventh_message[0].get('content'), "role": seventh_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    eighth_message = [{
        "content": "Im a middle order batsman who is capable of rotating the strike not hitting the ball",
        "role": "user"
    }]
    conversation.append({"content": eighth_message[0].get('content'), "role": eighth_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    ninth_message = [{
        "content": "I'm a backend developer by profession who is working on python as my primary language",
        "role": "user"
    }]
    conversation.append({"content": ninth_message[0].get('content'), "role": ninth_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    tenth_message = [{
        "content": "I just want to balance both in good way and be good in the fields Im on",
        "role": "user"
    }]
    conversation.append({"content": tenth_message[0].get('content'), "role": tenth_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation



    eleventh_message = [{
        "content": "Summarize about me in a single sentence",
        "role": "user"
    }]
    conversation.append({"content": eleventh_message[0].get('content'), "role": eleventh_message[0].get('role')})
    response = await llm.generate(conversation)
    conversation.append({"content": response.text, "role": "assistant"})
    conversation = await check_conversation_length(conversation) if len(conversation) > 10 else conversation

    print("conversation", conversation)



async def check_conversation_length(conversation):
    SUMMARY_TAG = "[SUMMARY]"
    count = 0
    existing_summary = []

    if len(conversation) > 10:
        # print("into if len of convo greater than 5")
        system_message = []
        for convo in conversation:
            if convo.get("role") == "system":
                if convo["content"].startswith(SUMMARY_TAG):
                    existing_summary = convo
                    count+= 1
                    if count > 2:
                        break
                else:
                    system_message.append(convo)
                    count+= 1
                    if count > 2:
                        break
                    

        NON_SYSTEM = [m for m in conversation if m.get("role") != "system"]

        old_messages = NON_SYSTEM[:-5]

        new_messages = NON_SYSTEM[-5:]

        old_text = messages_to_text(old_messages, existing_summary if existing_summary else None)
        print("================================================")
        print("existing_summary", existing_summary)


        llm = MyLLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")

        summary = await llm.summarise_text(old_text)
        # print("================================================")
        # print("summary", summary)

        summary = [{"role": "system",  "content": f"{SUMMARY_TAG}\n{summary}"}]

        conversation = system_message + summary + new_messages
            
        return conversation


def messages_to_text(messages, existing_summary=None):
    parts = []

    if existing_summary:
        parts.append(f"Summary so far:\n{existing_summary}")

    for m in messages:
        parts.append(f"{m.get('role')}: {m.get('content')}")

    return "\n".join(parts)



async def test_tool_calls():
    llm = MyLLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1")
    messages = [{
        "role": "system", "content": "You are a helpful assistant."
    }]

    messages.append({
        "role": "user", "content": "What is the current time now?"
    })

    response = await llm.generate(messages)
    final_text = response.text

    tts_service = MyTTSService(api_key=os.getenv("OPENAI_API_KEY"))
    result = await tts_service.run_tts(final_text)
    
    with open("output.wav", "wb") as f:
        f.write(result)



if __name__ == "__main__":
    # asyncio.run(main())
    asyncio.run(test_tool_calls())
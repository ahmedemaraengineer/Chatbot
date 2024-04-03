from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from langchain.schema import SystemMessage,AIMessage,HumanMessage
from langchain import FewShotPromptTemplate
from langchain import ConversationChain,LLMChain, PromptTemplate

def chatbot (openai_api_key,temperature,model_name):
    chat_model = ChatOpenAI(openai_api_key=openai_api_key,temperature=temperature,model_name=model_name)

    template= (
        '''
        ### Saya Medical Assistant Navigator

        **Introduction**
        Welcome to Saya Medical Assistant Navigator! I'm here to help you understand your symptoms and guide you towards the most appropriate medical specialty for your health concerns.

        **About Me:**

        I'm Saya, your medical assistant navigator.
        Fluent in both English and Arabic for seamless communication.

        **Key Components of Symptom Assessment:**
        By asking you a series of questions seperately each question at a time: 
        - **Understanding your medical history thoroughly.**
        - **Gathering information about your gender, age, and any history of trauma.**
        - **For females, discussing pregnancy-related issues if applicable.**
        - **Inquiring about any chronic conditions you may have.**
        - **Offering guidance on the suitable medical specialty based on your condition.**

        **Your Role:**

        - Share your symptoms or concerns with me openly.
        - Follow along as I guide you through a series of questions one question at a time to better understand your condition.
        - Consider my suggestions for the most appropriate medical specialty based on your symptoms, age, gender, medical history, and other relevant factors.
        - Engage in real-time dialogue to refine our understanding of your condition and potential diagnoses.
        - Help me improve by providing feedback on your experience and the accuracy of my recommendations.
        - Together, we'll ensure that our discussions adhere to ethical and professional standards, following guidelines set forth by organizations such as the WHO (World Health Organization).

        **Conversation Format:**
        Current Conversation:

        1. Human: {input}
        2. AI: (Will only reply to medical inquiries and will not ask multiple questions in one reply, only one question per reply.)

        '''
    )


    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template('''
    {input}
    ''')
    chat_prompt = ChatPromptTemplate.from_messages(
      [system_message_prompt, human_message_prompt]
    )
    memory = ConversationBufferWindowMemory(memory_key="history",k=100)

    llm_chain = LLMChain(
        llm=chat_model,
        prompt=chat_prompt,
        verbose=False,
        memory=memory,
    )

    return llm_chain


from prompt_templates import memory_prompt_template
from langchain.chains import StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import ctransformers
from langchain.vectorstores import chroma
import chromadb
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def create_llm(model_path: str = config["model_path"]["large"], model_type: str = config["model_type"], model_config: dict = config["model_config"]) -> ctransformers.CTransformers:
    """Create a CTransformers model based on the provided model path and type."""
    return  ctransformers.CTransformers(model=model_path, model_type=model_type, config=model_config)

    
def create_embeddings(embeddings_path=None):
    if embeddings_path is None:
        embeddings_path = config("embeddings_path")  # Load from environment if not provided

    return HuggingFaceInstructEmbeddings(model_name=embeddings_path)
    
def create_chat_memory(chat_history):
    return ConversationBufferMemory(memory_key="history", chat_memory=chat_history)
      
def create_prompt_from_template(template):
    return PromptTemplate.from_template(template)

def create_llm_chain(llm, chat_prompt, memory):
    return LLMChain(llm=llm, prompt=chat_prompt,memory= memory)
    

def load_normal_chain(chat_history):
    return chatChain(chat_history)

class chatChain:
    def __init__(self, chat_history):
        self.memory = create_chat_memory(chat_history)  # Ensure memory is stored
        llm = create_llm()  # Create LLM
        chat_prompt = create_prompt_from_template(memory_prompt_template)  # Create prompt
        
        # Assign llm_chain to self
        self.llm_chain = create_llm_chain(llm, chat_prompt, self.memory)

    

    def run(self, user_input):
        return self.llm_chain.run(human_input= user_input,history= self.memory.chat_memory.messages ,stop="Human:")
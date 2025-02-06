import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(prompt_file, model_dir="/model-weights", temp_dir="temp", device="cuda"):
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    tools_dict = {
        "ChestXRayClassifierTool": ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": ImageVisualizerTool(),
        "DicomProcessorTool": DicomProcessorTool(temp_dir=temp_dir),
    }

    checkpointer = MemorySaver()
    model = ChatOpenAI(model="gpt-4o", temperature=0.7, top_p=0.95)
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    print("Starting server...")

    # Setup model_dir to where you want to download the weights
    # Some tools needs you to download the weights beforehand from Hugging Face
    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt", model_dir="/model-weights"
    )
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8585, share=True)

import accelerator
from huggingface_hub import InferenceClient
from request_templates import HugginfaceInferenceRequest
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import json

class HuggingfaceModel:
    """Currently only supports bitsandbytes quantization."""
    def __init__(self):

        self.model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", device_map="auto", cache_dir="./model_weights")
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", cache_dir="./model_weights", padding_side="right")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.config.use_cache = False

        self.rule = """You are a helpful, respectful and honest assistant."""
        self.prompt_template = """
        <s>[INST] 
        {rules}\n
        {prompt}
        [/INST] </s>
        """

        self.chat_history = {}

    def generate(self, prompt: str, generate_args: dict, client_host: str) -> str:

        if client_host in list(self.chat_history.keys()):
            basic_prompt = self.chat_history[client_host][-1000:]
        else:
            self.chat_history[client_host] = ""
            basic_prompt = ""

        input_prompt = self.prompt_template.format(rules=self.rule, prompt=basic_prompt+"user: "+prompt)
        ids = self.tokenizer.encode(f'{input_prompt}', return_tensors='pt', truncation=True).to("cuda")

        final_outputs = self.model.generate(
            ids,
            **generate_args
        )
        
        decoded_output = self.tokenizer.decode(final_outputs[0], skip_special_tokens=True)[len(input_prompt):]
        self.chat_history[client_host] += "user: "+prompt+"\n" + "answer: "+decoded_output+"\n"
        print(self.chat_history[client_host])
        return decoded_output

    def get_json(self, payload: HugginfaceInferenceRequest, client_host: str):
        output = self.generate(payload.prompt, payload.get_generation_arguments_as_dict(), client_host)
        response = {
            "model": payload.model_name,
            "answer": [
            {
                "content": [
                {
                    "type": "text",
                    "user_prompt": payload.prompt
                },
                {
                    "type": "text",
                    "model_output": output
                },
                ]
            }
            ],
        } 
        return json.dumps(response, indent=4)
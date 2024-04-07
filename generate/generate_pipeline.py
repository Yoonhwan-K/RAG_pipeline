import torch

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

class LLMGenerate :

    def __init__(self, config) :

        self.config = config

    def load_model(self) :

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_path, device_map = "auto", torch_dtype = torch.float16)

        return tokenizer, model

    def  tuning_prompt(self, documents) :

        messages = [{"role" : "system", "content" : "아래 나용을 바탕으로 질문에 답변합니다."}]
        for i in range(len(documents)) :

            messages.append({"role" : "system", "content" : documents[i]})

        return messages

    def generate(self, tokenizer, model, messages) :

        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)

        inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)
        outputs = model.generate(**inputs, use_cache = True, max_length = self.config.max_length)

        result = tokenizer.decode(outputs[0])

        return result




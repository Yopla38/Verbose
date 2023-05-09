import json
import os
import paramiko
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class stablelmServer:
    def __init__(self, host, username, password, chosen_model: str = "stablelm-tuned-alpha-7b",
                 path: str = 'stabilityai/',
                 device: str = 'auto', tuned_mode: bool = False, local_cache: str = '/media/yoann/Deep_Disk/LLM/'):
        # @param ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]

        self.host = host
        self.username = username
        self.password = password

        self.chosen_model = chosen_model  # name of the model
        self.path = path  # path from huggingface
        self.path_model = os.path.join(path, chosen_model)
        self.local_cache = local_cache  # local cache path for preexisting model
        self.model = None  # model tensor
        self.tokenizer = None

        # tuned mode is like chatGPT: system, user, assistant
        self.tuned_mode = tuned_mode  # use preprompt for tuned model
        self.tm_system = """<|SYSTEM|># StableLM Tuned (Alpha version)
          - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
          - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
          - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
          - StableLM will refuse to participate in anything that could harm a human.
          """

        # Sampling args
        self.max_tokens = 128  # @param {type:"slider", min:32.0, max:3072.0, step:32}
        self.temperature = 0.7  # @param {type:"slider", min:0.0, max:1.25, step:0.05}
        self.top_k = 0  # @param {type:"slider", min:0.0, max:1.0, step:0.05}
        self.top_p = 0.9  # @param {type:"slider", min:0.0, max:1.0, step:0.05}
        self.do_sample = True  # @param {type:"boolean"}

        # Select "big model inference" parameters
        self.torch_dtype = "float16"  # @param ["float16", "bfloat16", "float"]
        self.load_in_8bit = False  # @param {type:"boolean"}
        self.device_map = device
        self.load_model()

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                           offline=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                              offline=True,
                                                              torch_dtype=getattr(torch,
                                                                                  self.torch_dtype),
                                                              load_in_8bit=self.load_in_8bit,
                                                              device_map=self.device_map,
                                                              offload_folder="./offload", )
        except OSError:
            # Télécharger le modèle en ligne si le fichier n'existe pas
            print("Download model from huggingface...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                           offline=False)
            self.model = AutoModelForCausalLM.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                              offline=False,
                                                              torch_dtype=getattr(torch,
                                                                                  self.torch_dtype),
                                                              load_in_8bit=self.load_in_8bit,
                                                              device_map=self.device_map,
                                                              offload_folder="./offload", )

    def generate_text(self, prompt, temperature, max_token):
        # Create `generate` inputs
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs.to(self.model.device)

        # Generate
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_token,
            temperature=temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()])
        )

        # Extract out only the completion tokens
        completion_tokens = tokens[0][inputs['input_ids'].size(1):]
        completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        return completion

    def serve(self):
        client = paramiko.Transport(self.host)
        client.connect(username=self.username, password=self.password)

        while True:
            try:
                channel = client.accept(20)
                if channel is None:
                    continue
                message = channel.recv(2048).decode("utf-8")
                if message == "quit":
                    channel.close()
                    break
                prompt = json.loads(message)["prompt"]
                temperature = json.loads(message)["temperature"]
                max_token = json.loads(message)["max_token"]

                response = self.generate_text(prompt, temperature, max_token)
                channel.send(json.dumps({"response": response}))
            except Exception as e:
                print(e)

        client.close()


server = stablelmServer("192.168.1.100", "username", "password")
server.serve()

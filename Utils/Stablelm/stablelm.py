# requiere : !pip install -U pip
# !pip install accelerate bitsandbytes torch transformers
import os.path
import json
from Utils.Utils import cprint
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StopOnSyntax(StoppingCriteria):
    # pas sure qu'elle fonctionne
    # Generate sequences up to a maximum length of 20 tokens, or until the "```end" token is generated
    # stopping_criteria = StopOnSyntax(end_token="```end")
    # output_ids = model.generate(input_ids, stopping_criteria=stopping_criteria)
    def __init__(self, end_token):
        super().__init__()
        self.end_token = end_token

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        last_token = input_ids[0][-1]
        last_token_str = stablelm.tokenizer.decode(last_token)

        if last_token_str == self.end_token:
            return True

        return False


class stablelm:
    from IPython.display import Markdown, display

    def __init__(self, chosen_model: str = "stablelm-tuned-alpha-7b", path: str = 'stabilityai/', local: bool = True,
                 device: str = 'auto', tuned_mode: bool = False, local_cache: str = '/media/yoann/Deep_Disk/LLM/'):
        # @param ["stabilityai/stablelm-tuned-alpha-7b", "stabilityai/stablelm-base-alpha-7b", "stabilityai/stablelm-tuned-alpha-3b", "stabilityai/stablelm-base-alpha-3b"]

        self.chosen_model = chosen_model  # name of the model
        self.path = path  # path from huggingface
        self.path_model = os.path.join(path, chosen_model)
        self.local_cache = local_cache  # local cache path for preexisting model
        self.model = None  # model tensor
        self.tokeniser = None
        self.local = local  # local load tensor. not local is not implemented
        # SSH client
        if local:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.torch = torch
            self.AutoModelForCausalLM = AutoModelForCausalLM
            self.AutoTokenizer = AutoTokenizer
        else:
            import paramiko
            self.paramiko = paramiko
            self.host = "192.168.1.100"
            self.username = "username"
            self.password = "password"

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
        if self.local:
            try:
                self.tokenizer = self.AutoTokenizer.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                                    offline=True)
                self.model = self.AutoModelForCausalLM.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                                       offline=True,
                                                                       torch_dtype=getattr(self.torch,
                                                                                           self.torch_dtype),
                                                                       load_in_8bit=self.load_in_8bit,
                                                                       device_map=self.device_map,
                                                                       offload_folder="./offload", )
            except OSError:
                # Télécharger le modèle en ligne si le fichier n'existe pas
                cprint("Download model from huggingface...", "blue")
                self.tokenizer = self.AutoTokenizer.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                                    offline=False)
                self.model = self.AutoModelForCausalLM.from_pretrained(self.path_model, cache_dir=self.local_cache,
                                                                       offline=False,
                                                                       torch_dtype=getattr(self.torch,
                                                                                           self.torch_dtype),
                                                                       load_in_8bit=self.load_in_8bit,
                                                                       device_map=self.device_map,
                                                                       offload_folder="./offload", )
        else:
            self.client = self.paramiko.Transport(self.host)
            self.client.connect(username=self.username, password=self.password)
            self.channel = self.client.open_channel(kind="session")

    def chat_completion(self, user_prompt):
        if self.tuned_mode:
            # Add system prompt for chat tuned models
            system_prompt = self.tm_system
            prompt = f"{system_prompt}<|USER|>{user_prompt}<|ASSISTANT|>"
        else:
            prompt = user_prompt

        cprint(
            f"Sampling with: `{self.max_tokens=}, {self.temperature=}, {self.top_k=}, {self.top_p=}, {self.do_sample=}`")
        self.hr()

        if self.local:
            # Create `generate` inputs
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs.to(self.model.device)

            # Generate
            tokens = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()])
            )

            # Extract out only the completion tokens
            completion_tokens = tokens[0][inputs['input_ids'].size(1):]
            completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True)
        else:
            self.channel.send(json.dumps({"prompt": prompt, "temperature": str(self.temperature),
                                          "max_token": str(self.max_tokens)}))
            response = self.channel.recv(2048).decode("utf-8")
            completion = json.loads(response)["response"]
        return completion

    def __del__(self):
        self.client.close()

    def hr(self):
        stablelm.display(stablelm.Markdown('---'))

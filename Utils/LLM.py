import os
import time


class llm:
    def __init__(self, type_name:str="openai", local:bool=False, debug_mode:bool = False):
        self.type_name = type_name
        self.local = local
        self.init_ok = False
        self.print = []
        self.debug_mode = False
        # Method
        self.chat = None
        self.completion = None
        self.tokenizer = None
        self.used_form = {'temperature': 0.7, 'max_token': 4000, 'model': 'gpt3-5'}
        #Init
        self.available_models = self.init()
        self.init_ok = True if self.available_models else False

    def _print(self, text, role: str = "system_print"):
        if self.debug_mode:
            print(text)
        self.print.append({role: text})

    def init(self):

        if self.type_name == "openai":
            self.module = __import__('openai')
            self.module2 = __import__('tiktoken')
            self.error_class = self.module.OpenAIError
            api_key_path = "openAI_key.txt"
            # Vérifier si le fichier existe
            if not os.path.isfile(api_key_path):
                self._print(f"Le fichier {api_key_path} n'existe pas.")
                return None

            # Lire la clé d'API depuis le fichier
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()

            # Vérifier si la clé d'API est valide
            self.module.openai.api_key = api_key
            try:
                models = self.module.openai.Model.list()
                self.chat = self.openai_chat
                #self.completion = self.openai_completion
                self.tokenizer = self.openai_tokinizer
                return models
            except Exception as e:
                self._print(f"La clé d'API OpenAI n'est pas valide : {e}")
                return None

    def openai_chat(self, message_log, *args):

        #define in role.json
        exemple_role = args[0]
        user_parameter_role = args[1]

        # count token
        tokens = self.tokenizer(message_log, exemple_role[user_parameter_role]['engine'])
        print("Tokens numbers = " + str(tokens))
        # Use OpenAI's ChatCompletion API to get the chatbot's response
        try:
            response = self.module.openai.ChatCompletion.create(
                model=exemple_role[user_parameter_role]['engine'],  # The name of the OpenAI chatbot model to use
                messages=message_log,  # The conversation history up to this point, as a list of dictionaries
                max_tokens=self.used_form['max_token'] - tokens,
                # The maximum number of tokens (words or subwords) in the generated response
                stop=None,  # The stopping sequence for the generated response, if any (not used here)
                temperature=self.used_form['temperature'],
                # The "creativity" of the generated response (higher temperature = more creative)
            )

            # Find the first response from the chatbot that has text in it (some responses may not have text)
            for choice in response.choices:
                if "text" in choice:
                    return choice.text

            # If no response with text is found, return the first response's content (which may be empty)
            return response.choices[0].message.content
        except self.error_class.OpenAIError as error:
            if error.__class__.__name__ == 'AuthenticationError':
                self._print("Erreur d'authentification: vérifiez votre clé API.")
            elif error.__class__.__name__ == 'RateLimitError':
                self._print("Erreur de taux de requête: Attente de 5 minutes.")
                # Attendre 5 minutes (300 secondes) avant de réessayer
                time.sleep(300)
            elif error.__class__.__name__ == 'APIError':
                self._print("Erreur de l'API OpenAI: {}".format(error))
            else:
                self._print("Une erreur s'est produite: {}".format(error))

        return None

    def openai_tokinizer(self, messages, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = self.module2.tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = self.module2.tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            return self.openai_tokinizer(messages, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            return self.openai_tokinizer(messages, model="gpt-4-0314")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

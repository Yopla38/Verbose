import ast
import json
import os
import re
import time
import openai
from openai import OpenAIError
from Chat import Chat
from Jupyter_page import JupyterNotebook
import tiktoken
from Utils import LLM
def get_openai_api_key():
    '''
    This function retrieves the OpenAI API key from a text file named "openAI_key.txt". It then checks if the file exists, reads the API key from the file, sets it as the current OpenAI API key, and finally validates the key by attempting to list the available models using the OpenAI API.

    If the file does not exist, it returns None.

    If the API key is successfully validated, it returns the API key, otherwise it also returns None and prints an error message indicating that the API key is not valid, along with the error message describing the issue.

    Args:
        None

    Returns:
        api_key (str): The OpenAI API key, may be None if the key is invalid or the file does not exist.
    '''
    api_key_path = "openAI_key.txt"
    # Vérifier si le fichier existe
    if not os.path.isfile(api_key_path):
        print(f"Le fichier {api_key_path} n'existe pas.")
        return None

    # Lire la clé d'API depuis le fichier
    with open(api_key_path, "r") as f:
        api_key = f.read().strip()

    # Vérifier si la clé d'API est valide
    openai.api_key = api_key
    try:
        models = openai.Model.list()
        return models
    except Exception as e:
        print(f"La clé d'API OpenAI n'est pas valide : {e}")
        return None




class verbose:
    def __init__(self, role: str = "auto_programming", library: str = "", model: str = "gpt-4-0314", printing: bool = True, notebook:str = "conversation.ipynb"):

        self.model_api = get_openai_api_key()
        self.notebook = None
        self.exemple_role = self.ouvrir_dico("role.json")
        #print(self.exemple_role[role])
        self.user_parameter_role = role
        self.model = model
        self.first_request = True
        self.max_token = 4095
        self.temperature = 0.0
        self.printing = printing
        self.print = []
        self.auto_programming = False
        self.set_role(role)
        self.notebook_file = notebook
        if self.auto_programming:
            self.library_code = library
            self.init_auto_programming(file=self.notebook_file)

    def set_role(self, role):

        if role == "auto_programming":
            self.auto_programming = True
            self.model = self.exemple_role[role]['engine']
        elif role == "auto_prog":
            self.auto_programming = True
            self.model = self.exemple_role[role]['engine']
            self.temperature = float(self.exemple_role[role]['temperature'])
        elif role == 'string_to_listdict':
            self.model = self.exemple_role[role]['engine']
            self.temperature = 0.0
            self.stop_sequence = self.exemple_role[role]['stop_sequence']
            self.role = self.exemple_role[role]['prompt']
            self.end_prompt = self.exemple_role[role]['end_prompt']
            self.end_response = self.exemple_role[role]['end_response']
        elif role == 'string_to_dict':
            self.model = self.exemple_role[role]['engine']
            self.temperature = 0.0
            self.stop_sequence = self.exemple_role[role]['stop_sequence']
            self.end_prompt = self.exemple_role[role]['end_prompt']
            self.end_response = self.exemple_role[role]['end_response']
            self.role = self.exemple_role[role]['prompt']
        else:
            self.auto_programming = True
            self.model = self.exemple_role[role]['engine']
            self.temperature = float(self.exemple_role[role]['temperature'])
        self.message_log = [
            {"role": "system", "content": self.exemple_role[role]['prompt']}
        ]

    def chat_mode(self, user_input: str = "", me: str = "AI: ", you: str = "You: "):
        # si user_input vide, pose la question
        # si self.printing: ecrit avec print
        if user_input == "":
            return None

        if self.first_request:
            # If this is the first request, get the user's input and add it to the conversation history
            if user_input == "":
                user_input = input(you)

            self.message_log.append({"role": "user", "content": self.exemple_role[self.user_parameter_role]["pre_prompt"]+user_input})

            if self.function_system(user_input, me):
                return None

            # Send the conversation history to the chatbot and get its response
            response = self.send_message(self.message_log)

            # Add the chatbot's response to the conversation history and print it to the console
            self.message_log.append({"role": "assistant", "content": response})
            # Set the flag to False so that this branch is not executed again
            self.first_request = False

            if self.auto_programming:
                self.call_auto_programming(response)
            if self.printing:
                self._print(me + response, "IA_print")
            else:
                return response

        else:

            # If this is not the first request, get the user's input and add it to the conversation history
            if user_input == "":
                user_input = input(you)

            if self.function_system(user_input, me):
                return None

            #self.message_log.append({"role": "user", "content": user_input})
            self.reformat_prompt_session(user_input)
            # Send the conversation history to the chatbot and get its response
            response = self.send_message(self.message_log)

            # Add the chatbot's response to the conversation history and print it to the console
            self.message_log.append({"role": "assistant", "content": response})

            if self.auto_programming:
                self.call_auto_programming(response)

            if self.printing:
                self._print(me + response, "IA_print")
            else:
                return response

    def reset(self):
        self.set_role(self.user_parameter_role)
        if self.printing:
            self._print("Reset", "system_print")

    def function_system(self, user_input, me):
        # If the user types "quit", end the loop and print a goodbye message
        if user_input.lower() == "quit" or user_input.lower() == "exit":
            if self.printing:
                self._print(me + "Goode bye!", "IA_print")

            return True
        if user_input.lower() == "restart" or user_input.lower() == "reset":
            self.reset()
            return True

        if user_input.lower() == "restore":
            self.restore_session()
            if self.printing:
                self._print(me + "restore chat from jupyter..."), "IA_print"
            return True
        if user_input.lower() == "prompt?":
            index = 0
            for prompt in self.message_log:
                self._print(f"-----------INDEX {index}-----------", "IA_print")
                self._print(str(prompt), "IA_print")
                index +=1
            return True
        if user_input.lower() == "prompt-":
            index = input("System remove role index no :")
            role = self.message_log.pop(int(index))
            self._print(f"{role} deleted !", "IA_print")
        return False

    def restore_session(self):
        self.reformat_prompt_session("")

    def _old_restore_session(self):
        self._print("Restore chat from jupyter..."), "system_print"
        code_block = self.notebook.get_code_cells()
        self.set_role(self.user_parameter_role)
        if code_block is not None:
            for code in code_block:
                self.message_log.append({"role": "user", "content": code})
                self.first_request = False

    def reformat_prompt_session(self, question):
        code_block = self.notebook.get_code_cells()
        self.set_role(self.user_parameter_role)
        codes = ''
        if code_block is not None:
            for code in code_block:
                codes += code+'\n'
            if 'pre_restore_prompt' in self.exemple_role[self.user_parameter_role]:
                prompt = self.exemple_role[self.user_parameter_role]['pre_restore_prompt']\
                         + "\n" + codes +  self.exemple_role[self.user_parameter_role]['post_restore_prompt'] + '\n' + question
            else:
                prompt = codes + '\n' + question

            self.message_log.append({"role": "user", "content": prompt})
            self.first_request = False

    # Function to send a message to the OpenAI chatbot model and return its response
    def send_message(self, message_log):
        # count token
        tokens = self.num_tokens_from_messages(message_log, self.exemple_role[self.user_parameter_role]['engine'])
        print("Tokens numbers = " + str(tokens))
        # Use OpenAI's ChatCompletion API to get the chatbot's response
        try:
            response = openai.ChatCompletion.create(
                model=self.model,  # The name of the OpenAI chatbot model to use
                messages=message_log,  # The conversation history up to this point, as a list of dictionaries
                max_tokens=self.max_token-tokens,  # The maximum number of tokens (words or subwords) in the generated response
                stop=None,  # The stopping sequence for the generated response, if any (not used here)
                temperature=self.temperature,
                # The "creativity" of the generated response (higher temperature = more creative)
            )

            # Find the first response from the chatbot that has text in it (some responses may not have text)
            for choice in response.choices:
                if "text" in choice:
                    return choice.text

            # If no response with text is found, return the first response's content (which may be empty)
            return response.choices[0].message.content
        except OpenAIError as error:
            if error.__class__.__name__ == 'AuthenticationError':
                print("Erreur d'authentification: vérifiez votre clé API.")
            elif error.__class__.__name__ == 'RateLimitError':
                print("Erreur de taux de requête: Attente de 5 minutes.")
                # Attendre 5 minutes (300 secondes) avant de réessayer
                time.sleep(300)
            elif error.__class__.__name__ == 'APIError':
                print("Erreur de l'API OpenAI: {}".format(error))
            else:
                print("Une erreur s'est produite: {}".format(error))

        return None

    def init_auto_programming(self, file):
        self.notebook = JupyterNotebook(file)
        if len(self.notebook.nb.cells) == 0:
            self.notebook.add_markdown_cell("Bienvenue chez verbose !\n Veuillez executer la cellule ci dessous.")
            self.notebook.add_code_cell("%autosave 2")
            self.notebook.add_code_cell(self.library_code)
        else:
            self.restore_session()
            self._print("Session restored !")

        self.notebook.save_notebook()
        time.sleep(1)
        self.notebook.open_notebook()


    def call_auto_programming(self, string):
        if self.auto_programming:
            fonctions, code, comment_up, comment_down = self.extraire_fonctions_et_code(string)
            self.notebook.update_notebook()
            if comment_up != '': self.notebook.add_markdown_cell(comment_up)
            if fonctions != '': self.notebook.add_code_cell(fonctions)
            if code != '' : self.notebook.add_code_cell(code)
            if comment_down != '': self.notebook.add_markdown_cell(comment_down)
            self.notebook.save_notebook()
            self.notebook.refresh_browser()

    def extraire_fonctions_et_code(self, text):
        if self.model == "gpt-4-0314":
            if text:
                delim_dict = self.exemple_role[self.user_parameter_role]

                pattern = r'?```(?:python)?\n(.*?)\n?```'
                code_pattern = re.escape(delim_dict['s_code_str']) + "\n" + pattern

                code_block = re.search(code_pattern, text, re.DOTALL)
                if code_block:
                    code_blocks = code_block.group(1)
                else:
                    code_blocks = ''

                function_pattern = re.escape(delim_dict['s_fonction_str']) + "\n" + pattern
                function_block = re.search(function_pattern, text, re.DOTALL)
                if function_block:
                    functions = function_block.group(1)
                else:
                    functions = ''

                if function_block:
                    comment_up = text[:function_block.start()]
                    comment_down = text[code_block.end():] if code_block else text[function_block.end():]
                elif code_block:
                    comment_up = text[:code_block.start()]
                    comment_down = text[code_block.end():]
                else:
                    comment_up, comment_down = text, ''

                if functions == '' and code_block == '':
                    comment_up, code_blocks, comment_down = self.unformated_code(text)
                self._print("commentaire up : " + comment_up + '\n' + "commentaire down: " + '\n' + comment_down)
                return functions, code_blocks, comment_up, comment_down
        else:
            if text:
                delim_dict = self.exemple_role[self.user_parameter_role]

                pattern = r'?```(?:python)?\n(.*?)\n?```'
                code_pattern = re.escape(delim_dict['s_code_str']) + pattern

                code_block = re.search(code_pattern, text, re.DOTALL)
                if code_block:
                    code_blocks = code_block.group(1)
                else:
                    code_blocks = ''

                function_pattern = re.escape(delim_dict['s_fonction_str']) + pattern
                function_block = re.search(function_pattern, text, re.DOTALL)
                if function_block:
                    functions = function_block.group(1)
                else:
                    functions = ''

                if function_block:
                    comment_up = text[:function_block.start()]
                    comment_down = text[code_block.end():] if code_block else text[function_block.end():]
                elif code_block:
                    comment_up = text[:code_block.start()]
                    comment_down = text[code_block.end():]
                else:
                    comment_up, comment_down = text, ''

                if functions == '' and code_block == '':
                    comment_up, code_blocks, comment_down = self.unformated_code(text)
                self._print("commentaire up : " + comment_up + '\n' + "commentaire down: " + '\n' + comment_down)
                return functions, code_blocks, comment_up, comment_down
        return "", "", "No function or code, verify your connection", ""

    def unformated_code(self, text):
        code_block = ""
        above_text = ""
        below_text = ""

        start_index = text.find("```python")
        end_index = text.find("```", start_index + 1)

        if start_index != -1 and end_index != -1:
            code_block = text[start_index + 8:end_index]
            above_text = text[:start_index].strip()
            below_text = text[end_index + 3:].strip()

        return code_block, above_text, below_text

    def _gpt4_extraire_fonctions_et_code(self, texte: str):
        fonctions_debut = texte.find("FONCTIONS CREES:") \
                          + len("FONCTIONS CREES:")
        fonctions_fin = texte.find("CODE:")
        code_debut = fonctions_fin + len("CODE:")

        fonctions = texte[fonctions_debut:fonctions_fin].strip()
        code = texte[code_debut:].strip()

        return fonctions, code


    def format_string(self, text):
        prompt = self.role + text + self.end_prompt
        try:
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=1500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=self.stop_sequence
            )
            list = self.end_prompt + response['choices'][0]['text'] + self.end_response
            list = self.replace_html_codes(list)
            # transform to list
            try:
                list_dict = json.loads(list)
                return list_dict
            except json.JSONDecodeError:
                print(list)
                print("La chaîne de caractères fournie n'est pas un objet JSON valide.")
                return None
        except OpenAIError as error:
            if error.__class__.__name__ == 'AuthenticationError':
                print("Erreur d'authentification: vérifiez votre clé API.")
            elif error.__class__.__name__ == 'RateLimitError':
                print("Erreur de taux de requête: Attente de 5 minutes.")
                # Attendre 5 minutes (300 secondes) avant de réessayer
                time.sleep(300)
            elif error.__class__.__name__ == 'APIError':
                print("Erreur de l'API OpenAI: {}".format(error))
            else:
                print("Une erreur s'est produite: {}".format(error))

        return None


    def replace_html_codes(self, string):
        """
        Remplace les codes HTML '&#x27E8;' et '&#x27E9;' par des guillemets simples dans une chaîne de caractères.

        Args:
            string (str): La chaîne de caractères à modifier.

        Returns:
            str: La chaîne de caractères modifiée.
        """
        string = string.replace('\'', '"')
        string = string.replace('&#x27E8;', "\'")
        string = string.replace('&#x27E9;', "\'")

        return string

    def get_avalaible_model(self):
        model = self.model_api["data"]
        mod = []
        for m in model:
            mod.append(m["id"])
        return mod

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo-0301"):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
            return self.num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
        elif model == "gpt-4":
            print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
            return self.num_tokens_from_messages(messages, model="gpt-4-0314")
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

    def enregistrer_dico(self, dico, nom_fichier, indent=4):
        with open(nom_fichier, 'w') as f:
            json.dump(dico, f, indent=indent)

    def ouvrir_dico(self, nom_fichier):
        with open(nom_fichier, 'r') as f:
            dico = json.load(f)
        return dico

    def _print(self, text, role:str = "system_print"):
        self.print.append({role:text})

    def printing_verification(self):
        if self.notebook:
            if self.notebook.print:
                for dico in self.notebook.print:
                    self.print.append(dico)
                self.notebook.print = []

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        if callable(attr) and name != "__init__" and name != "printing_verification":
            def newfunc(*args, **kwargs):
                result = attr(*args, **kwargs)
                self.printing_verification()
                return result
            return newfunc
        else:
            return attr


class user:
    def __init__(self, team: str = "NPSC", library: str ="", role: str = 'auto_prog_gpt3', fenetre: bool = False):
        if library == "HAL":
            library_code = "from HAL import Equipe\nNPSC = Equipe('"+team+"')"
        else:
            library_code = ""
        #self.UI = verbose(role="auto_prog", library=library_code)
        self.UI = verbose(role=role, library=library_code)
        self.fenetre = fenetre
        if fenetre:
            self.chat = Chat()
            self.chat.run()

    def start(self):
        if self.fenetre:
            while True:
                self.chat.print(self.UI.chat_mode(self.chat.input("You:")))
        else:
            while True:
                self.UI.chat_mode()


if __name__ == '__main__':
    user = user(library="", role='auto-prog-scientifique')
    user.start()
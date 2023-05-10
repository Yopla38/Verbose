# Copyright CEA Grenoble 2023
# Autheur : Yoann CURE
import json
import os
import re
import time

GUI = True
GTK = False
QT = True

from PyQt6.QtCore import QRunnable, QThreadPool, QMutexLocker, QMutex
from PyQt6.QtGui import QTextCursor
from pyqt6_plugins.examplebutton import QtWidgets
from Jupyter_page import JupyterNotebook
from Utils.LLM import llm


class verbose:
    def __init__(self, role: str = "auto_programming", library: str = "", model: str = "gpt-4-0314",
                 printing: bool = True, notebook: str = "conversation.ipynb", llm_name: str = "openai"):
        self.message_log = []
        self.stream = True
        self.llm = llm(type_name=llm_name, local=False, debug_mode=False, max_token=4000)  # Large model langage
        self.model_api = self.llm.available_models
        self.notebook = None
        self.exemple_role = self.ouvrir_dico("role.json")
        # print(self.exemple_role[role])
        self.user_parameter_role = role
        self.model = model
        self.first_request = True
        self.max_token = self.llm.max_token
        self.temperature = 0.0
        self.printing = printing
        self.print = []
        self.auto_programming = False
        self.set_role(role)
        if self.auto_programming:
            self.library_code = library
            self.init_auto_programming(file=notebook)
        if self.stream:
            self.stream_first_pass = True
        self.notebook_file = notebook
        self.completions = ""

    @property
    def notebook_file(self):
        return self.notebook_file

    @notebook_file.setter
    def notebook_file(self, value):
        #self.notebook_file = value
        self.notebook.notebook_name = os.path.basename(value)
        self.notebook.notebook_path = os.path.abspath(value)

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

        if not self.message_log:
            self.message_log = [{"role": "system", "content": self.exemple_role[role]['prompt']}]
        else:
            self.message_log[0] = {"role": "system", "content": self.exemple_role[role]['prompt']}

    def chat_mode(self, user_input: str = "", me: str = "AI: ", you: str = "You: ", gtk_area_text=None, done_event=None):
        # si user_input vide, pose la question
        # si self.printing: ecrit avec print
        self.user_input = user_input
        if user_input == "":
            return None

        if self.first_request:
            # If this is the first request, get the user's input and add it to the conversation history
            self.message_log.append({"role": "user", "content": self.exemple_role[self.user_parameter_role]["pre_prompt"]+user_input})
            if self.function_system(user_input, me):
                return None
            # Send the conversation history to the chatbot and get its response
            response = self.send_message(self.message_log, gtk_area_text=gtk_area_text, done_event=done_event)
            self.first_request = False
            return response
        else:

            # If this is not the first request, get the user's input and add it to the conversation history

            if self.function_system(user_input, me):
                return None

            self.message_log.append({"role": "user", "content": self.exemple_role[self.user_parameter_role]["pre_prompt"]+user_input})

            #self.reformat_prompt_session(user_input)
            # Send the conversation history to the chatbot and get its response
            response = self.send_message(self.message_log, gtk_area_text=gtk_area_text, done_event=done_event)
            return response

    def post_traitment(self):
        if self.stream:
            self.message_log.append({"role": "assistant", "content": self.completions})
            if self.auto_programming:
                self.call_auto_programming(self.completions)
            return self.completions

    def reset(self):
        self.message_log = []
        self.set_role(self.user_parameter_role)
        self.first_request = True
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
        self._print("Restore chat from jupyter...", "system_print")
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
                         + "\n" + codes + self.exemple_role[self.user_parameter_role]['post_restore_prompt'] + '\n' + question
            else:
                prompt = codes + '\n' + question

            self.message_log.append({"role": "user", "content": prompt})
            self.first_request = False

    # Function to send a message to the OpenAI chatbot model and return its response
    def send_message(self, message_log, gtk_area_text=None, done_event=None):
        message = self.llm.chat(message_log, self.exemple_role, self.user_parameter_role)
        return message
        #return self.call_stream_programming(message, gtk_area_text=gtk_area_text, done_event=done_event)

    def init_auto_programming(self, file, first: bool = True):
        if first : self.notebook = JupyterNotebook(file)
        if len(self.notebook.nb.cells) == 0:
            self.notebook.add_markdown_cell("Bienvenue chez verbose !\n Veuillez executer la cellule ci dessous.")
            self.notebook.add_code_cell("%autosave 2")
            self.notebook.add_code_cell(self.library_code)
        else:
            self.restore_session()
            self._print("Session restored !")

        self.notebook.save_notebook()
        time.sleep(1)
        if first:
            self.notebook.open_notebook()
        else:
            self.notebook.update_notebook()

    def streaming_decompose_text(self, completion):
        text = completion['choices'][0]['delta']
        if "content" in text:
            return text["content"]
        return ""

    def call_stream_programming(self, completions_word, gtk_area_text=None, done_event=None):

        if GUI:

            if QT:
                self.task = StreamProgrammingTask(completions_word, gtk_area_text, done_event)
                #self.task.setAutoDelete(True)
                self.threadPool = QThreadPool()
                self.threadPool.globalInstance().start(self.task)
                #self.task.finished.connect(self.handle_stream_programming_result)  # connecter le signal

    def call_auto_programming(self, string):
        if self.auto_programming:
            fonctions, code, comment_up, comment_down = self.extraire_fonctions_et_code(string)
            print("FONCTIONS : " + fonctions)
            print("CODE : " + code)
            print("COMMENT UP : " + comment_up)
            print("COMMENT DOWN : " + comment_down)
            self.notebook.update_notebook()
            if comment_up != '': self.notebook.add_markdown_cell(self.user_input+"\n"+comment_up)
            if fonctions != '': self.notebook.add_code_cell(fonctions)
            if code != '': self.notebook.add_code_cell(code)
            if comment_down != '': self.notebook.add_markdown_cell(comment_down)
            self.notebook.save_notebook()
            self.notebook.refresh_browser()

    def extraire_fonctions_et_code(self, text):
        if text:
            delim_dict = self.exemple_role[self.user_parameter_role]
            avant, apres = extract(text, delim_dict['s_code_str'])
            comment_up, suite = extract(avant, delim_dict['s_fonction_str'])
            functions, _ = extraire_contenu_python(suite)
            code, comment_down = extraire_contenu_python(apres)
            return functions, code, comment_up, comment_down
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

    def format_string(self, text):

        prompt = self.role + text + self.end_prompt
        response = self.llm.completion(prompt, self.exemple_role, self.role)

        list = self.end_prompt + response + self.end_response
        list = self.replace_html_codes(list)
        # transform to list
        try:
            list_dict = json.loads(list)
            return list_dict
        except json.JSONDecodeError:
            print(list)
            print("La chaîne de caractères fournie n'est pas un objet JSON valide.")
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

    def enregistrer_dico(self, dico, nom_fichier, indent=4):
        with open(nom_fichier, 'w') as f:
            json.dump(dico, f, indent=indent)

    def ouvrir_dico(self, nom_fichier):
        with open(nom_fichier, 'r') as f:
            dico = json.load(f)
        return dico

    def _print(self, text, role: str = "system_print"):
        self.print.append({role: text})

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

class StreamProgrammingTask(QRunnable):

    def __init__(self, completions_word, qt_area_text=None, done_event=None):
        super(StreamProgrammingTask, self).__init__()
        self.completions_word = completions_word
        self.qt_area_text = qt_area_text
        self.done_event = done_event
        self.mutex = QMutex()
        self.completions = ""

    def run(self):
        self.completions = ""
        cursor = self.qt_area_text.textCursor()
        for completion in self.completions_word:
            text = completion['choices'][0]['delta']
            if "content" in text:
                mot = text["content"]
                # Append the completion text to the window
                if self.qt_area_text:
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.insertText(mot)
                    self.qt_area_text.ensureCursorVisible()
                    QtWidgets.QApplication.processEvents()
                self.completions += mot
        if self.qt_area_text:
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText("\n")
            self.qt_area_text.ensureCursorVisible()
            QtWidgets.QApplication.processEvents()
        self.done_event.set()

    def get_completions(self):
        with QMutexLocker(self.mutex):
            return self.completions

    def finished(self):
        self.done_event.set()


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



def unformated_code(text):
    code_block = ""
    above_text = ""
    below_text = ""

    start_index = text.find("```python")
    end_index = text.find("```", start_index + 1)

    if start_index != -1 and end_index != -1:
        code_block = text[start_index + 8:end_index]
        above_text = text[:start_index].strip()
        below_text = text[end_index + 3:].strip()

    return "", code_block, above_text, below_text


def extract(texte, separateur):
    # Diviser le texte en deux parties
    parties = texte.split(separateur, 1)

    # Extraire les deux parties
    partie_avant_code = parties[0].strip()
    partie_code = parties[1].strip() if len(parties) > 1 else ""
    return partie_avant_code, partie_code
def extraire_contenu_python(texte: str):
    # Utiliser une expression régulière pour chercher le contenu entre ```python et ```
    pattern = re.compile(r'```(?:python)?(.*?)```', re.DOTALL)
    resultat = pattern.search(texte)

    # Si le contenu est trouvé, extraire le contenu et le reste, sinon retourner une chaîne de caractères vide pour le contenu et le texte original pour le reste
    if resultat:
        contenu_python = resultat.group(1).strip()
        reste = pattern.sub("", texte).strip()
        return contenu_python, reste
    else:
        return "", texte


if __name__ == '__main__':
    with open('1', 'r') as file:
        text = file.read()


    avant, apres = extract(text, "CODE:")
    comment_up, suite = extract(avant, "FONCTIONS CREES:")
    functions, _ = extraire_contenu_python(suite)
    code, comment_down = extraire_contenu_python(apres)
    print(comment_down)

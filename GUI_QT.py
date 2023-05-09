# Copyright CEA Grenoble 2023
# Auteur : Yoann CURE

import time
from typing import Optional, Dict

from Utils import highlighting
from Utils.highlighting import PythonHighlighter


from PyQt6 import QtCore, QtWidgets, uic
from PyQt6.QtCore import Qt, QStringListModel, QThreadPool, pyqtSlot, QMutexLocker, QMutex, QRunnable, \
    QRegularExpression
from PyQt6.QtWidgets import QApplication, QMainWindow, QListView, QListWidgetItem, QDialog, QLabel, QLineEdit, \
    QVBoxLayout, QDialogButtonBox, QMessageBox, QPlainTextEdit, QComboBox, QSlider, QWidget, QPushButton, QFileDialog
from PyQt6.QtGui import QClipboard, QStandardItem, QStandardItemModel, QTextCursor, QSyntaxHighlighter, QTextCharFormat, \
    QColor, QFont, QTextOption, QTextBlockUserData, QGuiApplication, QBrush

from PyQt6 import uic
import ast
import json
import os
import pickle
import threading
from Utils.LLM import llm as llmtemp
from pyqt6_plugins.examplebuttonplugin import QtGui

from Verbose_core_V2 import verbose, StreamProgrammingTask

def dialog_box(message):
    # Créer une boîte de dialogue
    msg_box = QMessageBox()
    msg_box.setStyleSheet('''
                    QDialog {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                        border: 2px solid black;
                    }
                    QPlainTextEdit {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                    QComboBox {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                    QComboBox::drop-down {
                        background-color: #444e5e;
                    }
                    QComboBox QAbstractItemView {
                        background-color: #444e5e;
                        color: white;
                    }
                    QLabel {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                    QPushButton {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                ''')

    # Définir le texte du message
    msg_box.setText(message)

    # Ajouter les boutons "Oui" et "Non"
    msg_box.addButton(QMessageBox.StandardButton.Yes)
    msg_box.addButton(QMessageBox.StandardButton.No)

    # Afficher la boîte de dialogue et récupérer le bouton cliqué
    button = msg_box.exec()

    # Vérifier quel bouton a été cliqué
    if button == QMessageBox.StandardButton.Yes:
        return True

    return False

def show_error_dialog(message):
    dialog = QMessageBox()
    dialog.setStyleSheet('''
                    QDialog {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                        border: 2px solid black;
                    }
                    QPlainTextEdit {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                    QComboBox {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                    QComboBox::drop-down {
                        background-color: #444e5e;
                    }
                    QComboBox QAbstractItemView {
                        background-color: #444e5e;
                        color: white;
                    }
                    QLabel {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                    QPushButton {
                        font: 12pt "Chilanka";
                        background-color: #444e5e;
                        color: white;
                    }
                ''')
    dialog.setIcon(QMessageBox.critical)
    dialog.setText(message)
    dialog.setWindowTitle("Error")
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.setModal(True)
    dialog.exec()


class ChatWindow(QMainWindow):
    def __init__(self, role: str = 'auto-prog-scientifique'):

        self.app = QtWidgets.QApplication([])
        super().__init__()
        # define all widget to None
        self.edit_role = None
        self.modified_prompt = None
        self.role = None
        self.tokens = None
        self.tree = None
        self.chat_area = None
        self.message_entry = None
        self.slider = None
        self.quit = None
        self.modify_button = None
        self.delete_button = None
        self.add_button = None
        self.save = None
        self.restore = None
        self.reset = None
        self.new_2 = None
        self.load = None
        self.send_button = None
        # construct GUI

        self.done_event = None

        self.classe = user(library="", role=role)
        self.init_GUI()
        x, y, width, height = self.set_half_screen_size()
        self.classe.UI.notebook.position_screen(x, y, width, height)


        self.temperature = self.classe.get_temperature()

        # copy clipboard jupyter token
        clipboard = QApplication.clipboard()
        clipboard.setText(self.classe.token())

        self.list = []
        self.gestion_liststore('init')
        self.populate_role()
        self.role.currentIndexChanged.connect(self.change_role)
        self.print_action()

    def init_GUI(self):
        # Chargement de l'interface graphique créée avec Qt Designer
        uic.loadUi("./Utils/window.ui", self)
        # Initialisation des éléments de l'interface
        self.send_button.clicked.connect(self.send_message)
        self.load.clicked.connect(self.load_file)
        self.new_2.clicked.connect(self.new_button)
        self.reset.clicked.connect(self.reset_tree)
        self.restore.clicked.connect(self.restore_tree)
        self.help.clicked.connect(self.helper)
        self.delete_button.clicked.connect(self.delete_selected_button)
        self.modify_button.clicked.connect(self.modify_selected_button)
        self.quit.clicked.connect(self.close)
        self.slider.valueChanged.connect(self.slider_value_changed)
        self.highlighter = PythonHighlighter(self.chat_area.document())
        self.edit_role.clicked.connect(self.edit_role_dialog)

    def populate_role(self):
        for key in self.classe.UI.exemple_role:
            self.role.addItem(str(key))

    def change_role(self):
        key = self.role.currentText()
        self.classe.role = key
        self.classe.UI.role = self.classe.role
        self.classe.UI.set_role(self.classe.role)

    def send_message(self):
        message = self.message_entry.toPlainText()

        if message:
            self.chat_area.insertPlainText("You: " + message + "\n")
            self.message_entry.setPlainText("")
            self.classe.UI.completions = ""
            stream = True
            if not stream:
                self.classe.send_message(text=message, history_prompt=self.modified_prompt)
            else:
                response = self.classe.send_message(text=message, history_prompt=self.modified_prompt,
                                                    gtk_area_text=self.chat_area, done_event=self.done_event)

                self.state_buttons(False)
                for completion in response:
                    word = self.classe.UI.streaming_decompose_text(completion)
                    cursor = self.chat_area.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    self.chat_area.setTextCursor(cursor)
                    self.chat_area.insertPlainText(word)
                    self.classe.UI.completions += word
                    QApplication.processEvents()

                self.classe.post_traitment()

            self.print_action()
            self.gestion_liststore('store')
            self.state_buttons(True)

    def state_buttons(self, state: bool = True):
        self.send_button.setEnabled(state)
        self.modify_button.setEnabled(state)
        self.delete_button.setEnabled(state)
        self.reset.setEnabled(state)
        self.restore.setEnabled(state)

    def load_file(self):
        filename = self.classe.load()
        if filename:
            self.gestion_liststore("store")
            synchronize = dialog_box("Voulez-vous analyser la page jupyter avec Verbose ?")
            if synchronize:
                self.restore_tree()

    def new_button(self):
        filename = self.classe.new()
        self.reset_tree()

    def reset_tree(self):
        self.chat_area.clear()
        self.classe.reset()
        self.gestion_liststore('store')

    def restore_tree(self):
        #self.classe.synchronize()
        self.classe.UI.notebook.save_page_auto()
        self.resume_session()
        self.gestion_liststore('store')

    def calculate_token(self):
        #print("max_token = " + self.classe.llm.max_token)
        print(self.classe.UI.exemple_role[self.classe.role]['engine'])
        token = self.classe.UI.llm.tokenizer(self.modified_prompt,
                                             self.classe.UI.exemple_role[self.classe.role]['engine'])
        self.tokens.display(token)
        return token

    def helper(self):
        # Code pour sauvegarder la liste dans un fichier...

        pass

    def add_new_button(self):
        # Code pour ajouter un nouveau bouton dans la liste...
        pass

    def delete_selected_button(self):
        selected_indexes = self.tree.selectedIndexes()
        if selected_indexes:
            # Récupère l'indice de l'élément sélectionné
            selected_index = selected_indexes[0]
            if selected_index.row() is not None:
                del self.modified_prompt[selected_index.row()]
                self.classe.UI.message_log = self.modified_prompt
                self.gestion_liststore(action="store")

    def modify_selected_button(self):
        selected_indexes = self.tree.selectedIndexes()
        if selected_indexes:
            # Récupère l'indice de l'élément sélectionné
            selected_index = selected_indexes[0]
            # Récupère le modèle associé à l'affichage
            model = self.tree.model()
            # Récupère les données de l'élément sélectionné sous forme de QVariant
            text = self.modified_prompt[selected_index.row()]
            response = create_textview_dialog("Modify", text)

            if response:
                try:
                    dico = ast.literal_eval(str(response))
                    # self.tree.setData(selected_index, dico)
                    # self.modified_prompt = self.convert_list_to_dict(self.get_liststore_as_list(model))
                    self.classe.actual_prompt()[selected_index.row()] = response
                    self.gestion_liststore(action="store")

                except ValueError as e:
                    show_error_dialog(f"Error: not a dictionnary. Ignoring modification")
                except SyntaxError as se:
                    show_error_dialog(f"Error: not a dictionnary. Ignoring modification")

    def get_liststore_as_list(self, model):
        lst = []
        for row in model:
            item = []
            for i in range(model.get_n_columns()):
                item.append(model.get_value(row.iter, i))
            lst.append(item)
        return lst

    def convert_list_to_dict(self, lst):
        result = []
        for item in lst:
            result.append(eval(item[0]))
            '''
            try:
                d = ast.literal_eval(item)
                result.append(d)
            except ValueError as e:
                print(f"Error: {e}. Ignoring item {item}")
            '''
        return result

    def convert_str_to_dict(self, data_str):
        # Remplacer les guillemets simples par des guillemets doubles
        data_str = data_str.replace("'", "\"")
        # Charger la chaîne en tant que dictionnaire JSON
        data_dict = json.loads(data_str)
        # Retourner le dictionnaire
        return data_dict

    def slider_value_changed(self, value):
        # Code pour gérer le changement de valeur de la réglette...
        pass

    def on_tree_selection_changed(self):
        selected_indexes = self.tree.selectedIndexes()
        if selected_indexes:
            # Récupère l'indice de l'élément sélectionné
            selected_index = selected_indexes[0]
            # Récupère le modèle associé à l'affichage
            model = self.tree.model()
            # Récupère les données de l'élément sélectionné sous forme de QVariant
            selected_data = model.data(selected_index)
            return selected_data
        return ""

    def gestion_liststore(self, action: str = 'init'):
        if action == 'init':
            self.tree.clicked.connect(self.on_tree_selection_changed)
            self.gestion_liststore('store')

        elif action == "store":
            self.modified_prompt = self.classe.actual_prompt()
            self.update_list_area(self.modified_prompt)
            model = QStringListModel()
            model.setStringList(self.list)
            self.tree.setModel(model)
            self.calculate_token()

        elif action == "item":
            self.update_list_area(self.modified_prompt)
            model = QStringListModel()
            model.setStringList(self.list)
            self.tree.setModel(model)
            self.calculate_token()

    def update_list_area(self, new_list):
        # Clear the existing list
        self.list.clear()
        # Add items from the new list
        for item in new_list:
            self.list.append(str(item))

    def print_action(self):
        if self.classe.print:
            for dico in reversed(self.classe.print):
                if "system_print" in dico:
                    self.chat_area.insertPlainText("System: " + dico["system_print"] + "\n")
                if "IA_print" in dico:
                    self.chat_area.insertPlainText("Verbose: " + dico["IA_print"] + "\n")

            self.classe.print = []

    def set_half_screen_size(self):
        # Get primary screen
        primary_screen = QGuiApplication.primaryScreen()

        # Get available geometry of primary screen
        available_geometry = primary_screen.availableGeometry()

        # Calculate width and height for the window
        width = available_geometry.width() / 2
        height = available_geometry.height()

        # Calculate x and y positions for the window
        x = available_geometry.width() / 2
        y = 0

        # Set geometry of the window
        self.setGeometry(x, y, width, height)
        return 0, 0, width, height

    def edit_role_dialog(self):
        self.classe.UI.exemple_role = self.edit_config_dialog(self.classe.UI.exemple_role, self.classe.role)
        self.classe.UI.set_role(self.classe.role)
        self.gestion_liststore("store")

    def edit_config_dialog(self, config_dict, role):
        dialog = QDialog()
        dialog.setWindowTitle("System prompt " + str(role))
        # Get primary screen
        primary_screen = QGuiApplication.primaryScreen()
        # Get available geometry of primary screen
        available_geometry = primary_screen.availableGeometry()
        # Calculate width and height for the window
        width = available_geometry.width() / 2
        height = available_geometry.height()

        dialog.setGeometry(width/3, 0, width, height)
        dialog.setStyleSheet('''
                        QDialog {
                            font: 12pt "Chilanka";
                            background-color: #444e5e;
                            color: white;
                            border: 2px solid black;
                        }
                        QPlainTextEdit {
                            font: 12pt "Chilanka";
                            background-color: #444e5e;
                            color: white;
                        }
                        QComboBox {
                            font: 12pt "Chilanka";
                            background-color: #444e5e;
                            color: white;
                        }
                        QComboBox::drop-down {
                            background-color: #444e5e;
                        }
                        QComboBox QAbstractItemView {
                            background-color: #444e5e;
                            color: white;
                        }
                        QLabel {
                            font: 12pt "Chilanka";
                            background-color: #444e5e;
                            color: white;
                        }
                        QPushButton {
                            font: 12pt "Chilanka";
                            background-color: #444e5e;
                            color: white;
                        }
                        QSlider {
                            font: 12pt "Chilanka";
                            background-color: #444e5e;
                            color: white;
                            selection-background-color: rgb(173, 173, 173);
                        }
                    ''')
        layout = QVBoxLayout()
        dialog.setLayout(layout)

        # prompt
        prompt_label = QLabel("Prompt:")
        prompt_edit = QPlainTextEdit(config_dict[role]["prompt"])
        highlighter = PythonHighlighter(prompt_edit.document())
        prompt_edit.setTabStopDistance(20)
        prompt_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        prompt_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        prompt_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(prompt_label)
        layout.addWidget(prompt_edit)

        # engine
        engine_label = QLabel("Engine:")
        engine_combo = QComboBox()
        font = QFont()
        font.setBold(True)
        font.setUnderline(True)
        # find available engine from LLM
        if self.classe.UI.llm.available_engines:
            i = 0
            for engine in self.classe.UI.llm.available_engines:
                engine_combo.addItem(engine)
                if engine.find("gpt") >= 0:
                    item = engine_combo.model().item(i)
                    item.setFont(font)
                i += 1

        engine_combo.setCurrentText(config_dict[role]["engine"])
        layout.addWidget(engine_label)
        layout.addWidget(engine_combo)

        # temperature
        temp_label = QLabel("Temperature:")
        temp_slider = QSlider()
        temp_slider.setOrientation(Qt.Orientation.Horizontal)
        temp_slider.setMinimum(0)
        temp_slider.setMaximum(100)
        temp_slider.setSingleStep(1)
        temp_slider.setValue(int(float(config_dict[role]["temperature"]) * 100))
        layout.addWidget(temp_label)
        layout.addWidget(temp_slider)

        # s_fonction_str
        s_fonc_label = QLabel("s_fonction_str:")
        s_fonc_edit = QPlainTextEdit(config_dict[role]["s_fonction_str"])

        layout.addWidget(s_fonc_label)
        layout.addWidget(s_fonc_edit)

        # f_fonction_str
        f_fonc_label = QLabel("f_fonction_str:")
        f_fonc_edit = QPlainTextEdit(config_dict[role]["f_fonction_str"])
        layout.addWidget(f_fonc_label)
        layout.addWidget(f_fonc_edit)

        # s_code_str
        s_code_label = QLabel("s_code_str:")
        s_code_edit = QPlainTextEdit(config_dict[role]["s_code_str"])
        layout.addWidget(s_code_label)
        layout.addWidget(s_code_edit)

        # f_code_str
        f_code_label = QLabel("f_code_str:")
        f_code_edit = QPlainTextEdit(config_dict[role]["f_code_str"])
        layout.addWidget(f_code_label)
        layout.addWidget(f_code_edit)

        # pre_prompt
        pre_prompt_label = QLabel("pre_prompt:")
        pre_prompt_edit = QPlainTextEdit(config_dict[role]["pre_prompt"])
        layout.addWidget(pre_prompt_label)
        layout.addWidget(pre_prompt_edit)

        # pre_restore_prompt
        pre_restore_prompt_label = QLabel("pre_restore_prompt:")
        pre_restore_prompt_edit = QPlainTextEdit(config_dict[role]["pre_restore_prompt"])
        layout.addWidget(pre_restore_prompt_label)
        layout.addWidget(pre_restore_prompt_edit)

        # post_restore_prompt
        post_restore_prompt_label = QLabel("post_restore_prompt:")
        post_restore_prompt_edit = QPlainTextEdit(config_dict[role]["post_restore_prompt"])
        layout.addWidget(post_restore_prompt_label)
        layout.addWidget(post_restore_prompt_edit)

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            modified_dict = config_dict.copy()
            modified_dict["prompt"] = prompt_edit.toPlainText()
            modified_dict["engine"] = engine_combo.currentText()
            modified_dict["temperature"] = str(temp_slider.value() / 100)
            modified_dict["s_fonction_str"] = s_fonc_edit.toPlainText()
            modified_dict["f_fonction_str"] = f_fonc_edit.toPlainText()
            modified_dict["s_code_str"] = s_code_edit.toPlainText()
            modified_dict["f_code_str"] = f_code_edit.toPlainText()
            modified_dict["pre_prompt"] = pre_prompt_edit.toPlainText()
            modified_dict["pre_restore_prompt"] = pre_restore_prompt_edit.toPlainText()
            modified_dict["post_restore_prompt"] = post_restore_prompt_edit.toPlainText()
            return modified_dict
        else:
            return config_dict

    def resume_session(self, llm_model_name: str = "openai"):
        llm_temp = llmtemp(type_name=llm_model_name)
        #llm_temp.stream = True
        code_block = self.classe.UI.notebook.get_code_cells()
        ex_role = self.classe.UI.exemple_role
        role = "resume_session"
        sum_code = ""
        prompt = ""


        if code_block is not None:
            self.chat_area.insertPlainText("Résumé de session : \n")
            QApplication.processEvents()
            for code in code_block:
                message_log = [{"role": "system", "content": ex_role[role]['prompt']}]
                message_log.append({"role": "user", "content": code})
                completions = ""
                response = llm_temp.chat(message_log, ex_role, "resume_session")
                self.chat_area.insertPlainText("Nouvelle fonction trouvée : ")
                for completion in response:
                    word = self.classe.UI.streaming_decompose_text(completion)
                    cursor = self.chat_area.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    self.chat_area.setTextCursor(cursor)
                    self.chat_area.insertPlainText(word)
                    completions += word
                    QApplication.processEvents()

                sum_code += completions
                self.chat_area.insertPlainText("\n")

            if 'pre_restore_prompt' in self.classe.UI.exemple_role[self.classe.UI.user_parameter_role]:
                prompt = self.classe.UI.exemple_role[self.classe.UI.user_parameter_role]['prompt'] \
                         + "\n" + self.classe.UI.exemple_role[self.classe.UI.user_parameter_role]['pre_restore_prompt'] \
                         + "\n" + sum_code

                         #+ self.classe.UI.exemple_role[self.classe.UI.user_parameter_role]['post_restore_prompt'] + '\n'
            self.classe.UI.set_role(self.classe.UI.user_parameter_role)
            self.classe.UI.message_log = [{"role": "system", "content": prompt}]
            #self.classe.UI.message_log.append({"role": "user", "content": prompt})
            self.classe.UI.first_request = False
        return prompt

def create_textview_dialog(title: str, initial_text: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    dialog = QDialog()
    dialog.setWindowTitle(title)
    dialog.setStyleSheet('''
                QDialog {
                    font: 12pt "Chilanka";
                    background-color: #444e5e;
                    color: white;
                    border: 2px solid black;
                }
                QPlainTextEdit {
                    font: 12pt "Chilanka";
                    background-color: #444e5e;
                    color: white;
                }
                QComboBox {
                    font: 12pt "Chilanka";
                    background-color: #444e5e;
                    color: white;
                }
                QComboBox::drop-down {
                    background-color: #444e5e;
                }
                QComboBox QAbstractItemView {
                    background-color: #444e5e;
                    color: white;
                }
                QLabel {
                    font: 12pt "Chilanka";
                    background-color: #444e5e;
                    color: white;
                }
                QPushButton {
                    font: 12pt "Chilanka";
                    background-color: #444e5e;
                    color: white;
                }
            ''')

    # Check if the initial text is a dictionary with a "role" key
    if isinstance(initial_text, dict) and "role" in initial_text:
        # Create label and combo box widgets
        label = QLabel("Role:")
        combo_box = QComboBox()
        combo_box.addItems(["system", "user", "assistant"])

        # Create label and line edit widgets for content
        content_label = QLabel("Content:")
        content_edit = QPlainTextEdit(initial_text.get("content", ""))
        # highlighter = PythonHighlighter(content_edit.document())
        content_edit.setTabStopDistance(20)
        content_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        content_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        content_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Add widgets to layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(combo_box)
        layout.addWidget(content_label)
        layout.addWidget(content_edit)

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Set initial values of combo box and content edit
        combo_box.setCurrentText(initial_text.get("role"))
        content_edit.setPlainText(initial_text.get("content"))
        highlighter = PythonHighlighter(content_edit.document())
        # content_edit.textChanged.connect(highlighter.rehighlight)
        # Set layout and run dialog
        dialog.setLayout(layout)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            modified_dict = initial_text.copy()
            modified_dict["role"] = combo_box.currentText()
            modified_dict["content"] = content_edit.toPlainText()
            return modified_dict
        else:
            return initial_text

    else:
        # Create label and line edit widgets
        label = QLabel("Text:")
        text_edit = QPlainTextEdit(initial_text)
        text_edit.setTabStopDistance(20)
        text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.WidgetWidth)
        text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Add widgets to layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(text_edit)

        # Create OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Set layout and run dialog
        dialog.setLayout(layout)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return text_edit.toPlainText()
        else:
            return initial_text


class user:
    def __init__(self, team: str = "NPSC", library: str = "", role: str = 'auto_prog_gpt3', fenetre: bool = False,
                 notebook: str = "conversation.ipynb"):
        if library == "HAL":
            library_code = "from HAL import Equipe\nNPSC = Equipe('" + team + "')"
        else:
            library_code = ""
        self.role = role
        self.library = library_code
        self.notebook = notebook
        self.UI = verbose(role=role, library=library_code, notebook=notebook)

        self.task = None
        self.print = []

    def new(self):
        kwargs = {}
        if 'SNAP' in os.environ:
            kwargs['options'] = QFileDialog.Option.DontUseNativeDialog
        # Ouvrir la boîte de dialogue pour choisir le répertoire et le nom du fichier

        options = QFileDialog.Option.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(None, "Créer un nouveau fichier Notebook Jupyter", os.path.abspath(os.getcwd()),
                                                   "Notebook Jupyter (*.ipynb)", options=options)

        # Si le nom du fichier est vide, on ne fait rien
        if not file_path:
            return
        file_path = file_path + ".ipynb"
        # Vérifier que le fichier n'existe pas déjà
        if os.path.isfile(file_path):
            QMessageBox.warning(None, "Attention", "Le fichier existe déjà.")
            return

        self.notebook = file_path
        self.UI.notebook_file = file_path
        #print(file_path)
        self.UI.notebook.new_notebook()
        self.UI.notebook.save_notebook()
        #time.sleep(1)
        self.UI.init_auto_programming(file_path, first=False)
        self.UI.notebook.browser.get(self.UI.notebook.url())
        self.UI.notebook.refresh_browser()
        return file_path

    def load(self):
        # Ouvrez une boîte de dialogue de sélection de fichier pour les fichiers .ipynb
        filename, _ = QFileDialog.getOpenFileName(None, 'Ouvrir un fichier notebook Jupyter', '.',
                                                  'Fichiers notebook (*.ipynb)')
        if filename:
            self.notebook = filename
            self.UI.notebook_file = filename
            self.UI.notebook.update_notebook()
            self.UI.notebook.browser.get(self.UI.notebook.url())
            self.UI.notebook.refresh_browser()
            return filename
        return None
    def send_message(self, text, history_prompt: str = "", gtk_area_text=None, done_event=None):
        self.UI.notebook.save_page_auto()
        self.UI.message_log = history_prompt
        response = self.UI.chat_mode(text, gtk_area_text=gtk_area_text, done_event=done_event)
        return response
        # self.print.append({"IA_print": self.UI.chat_mode(text)})

    def post_traitment(self):
        self.UI.post_traitment()

    def actual_prompt(self):
        return self.UI.message_log

    def token(self):
        return self.UI.notebook.token

    def synchronize(self):
        self.UI.restore_session()

    def reset(self):
        self.UI.reset()

    def start(self):
        while True:
            self.UI.chat_mode()

    def cell_content(self):
        return self.UI.notebook.code_from_selected_cell()

    def get_temperature(self):
        return self.UI.temperature

    def set_temperature(self, value):
        self.UI.temperature = value

    def printing_verification(self):
        if self.UI.print:
            for dico in self.UI.print:
                self.print.append(dico)
            self.UI.print = []

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


if __name__ == "__main__":
    test = ChatWindow(role="auto-prog-general")
    test.show()
    test.app.exec()

# test = Test()

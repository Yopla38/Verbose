import ast
import json
import os
import pickle

import gi
# to install : sudo apt install libgirepository1.0-dev gcc libcairo2-dev pkg-config python3-dev gir1.2-gtk-3.0
# pip3 install pycairo
# pip3 install opencv-python==4.5.1.48
#conda install -c conda-forge pygobject gtk3
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk
from gi.repository import Gdk

css_provider = Gtk.CssProvider()
css_provider.load_from_path("./Utils/Verbose.css")

screen = Gdk.Screen.get_default()
context = Gtk.StyleContext()
context.add_provider_for_screen(screen, css_provider,
                                 Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

from Verbose import verbose

class user:
    def __init__(self, team: str = "NPSC", library: str ="", role: str = 'auto_prog_gpt3', fenetre: bool = False, notebook: str = "conversation.ipynb"):
        if library == "HAL":
            library_code = "from HAL import Equipe\nNPSC = Equipe('"+team+"')"
        else:
            library_code = ""
        #self.UI = verbose(role="auto_prog", library=library_code)
        self.role = role
        self.library = library_code
        self.notebook = notebook
        self.UI = verbose(role=role, library=library_code, notebook=notebook)
        self.print = []

    def send_message(self, text, history_prompt:str = ""):
        self.UI.message_log = history_prompt
        self.UI.chat_mode(text)
        #self.print.append({"IA_print": self.UI.chat_mode(text)})


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



class ChatWindow:
    def __init__(self, classe=user):
        self.classe = user(library="", role='auto-prog-scientifique')
        self.temperature = self.classe.get_temperature()
        self.builder = Gtk.Builder()
        self.builder.add_from_file("./Utils/Window.glade")

        self.window = self.builder.get_object("main_window")
        self.chat_area = self.builder.get_object("chat_area")
        self.input_area = self.builder.get_object("input_area")
        self.message_entry = self.builder.get_object("message_entry")
        self.send_button = self.builder.get_object("send_button")
        self.file_label = self.builder.get_object("file_label")
        self.slider = self.builder.get_object("slider")
        self.modify_button = self.builder.get_object("modify_button")
        self.delete_button = self.builder.get_object("delete_button")
        self.add_button = self.builder.get_object("add_button")
        self.zone_list = self.builder.get_object("zone_list")
        self.reset = self.builder.get_object("reset")
        self.sync = self.builder.get_object("restore")


        clipboard = Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD)
        clipboard.set_text(self.classe.token(), -1)
        # TreeView
        #self.list = self.builder.get_object("liststore")
        self.gestion_liststore('init')
        #self.create_dialog_box()
        self.print_action()
        self.connect_signals()
        #self.builder.connect_signals(self)

        #self.update_list_area(self.classe.actual_prompt())

        self.window.show_all()
        self.window.present()

    def gestion_liststore(self, action:str = 'init'):
        if action == 'init':
            self.list = Gtk.ListStore(str)
            self.tree = self.builder.get_object('tree')
            self.tree.set_model(self.list)
            self.renderer = Gtk.CellRendererText()
            self.column = Gtk.TreeViewColumn("Prompt", self.renderer, text=0, weight=1)
            self.tree.append_column(self.column)
            # Ajouter la sélection au TreeView
            self.selection = self.tree.get_selection()
            self.selection.set_mode(Gtk.SelectionMode.SINGLE)
            # Activer la surbrillance des éléments sélectionnés
            self.tree.set_activate_on_single_click(True)
            self.selection.connect("changed", self.on_tree_selection_changed)
            self.gestion_liststore('store')

        elif action == 'store':
            self.modified_prompt = self.classe.actual_prompt()

            self.update_list_area(self.modified_prompt)

    def create_textview_dialog(self, title, initial_text):
        dialog = Gtk.Dialog(title=title, modal=True)
        dialog.set_default_size(300, 200)

        box = dialog.get_content_area()
        scrolledwindow = Gtk.ScrolledWindow()
        scrolledwindow.set_hexpand(True)
        scrolledwindow.set_vexpand(True)
        box.pack_start(scrolledwindow, True, True, 0)

        textview = Gtk.TextView()
        textview.set_wrap_mode(Gtk.WrapMode.WORD)
        scrolledwindow.add(textview)

        buffer = textview.get_buffer()
        buffer.set_text(initial_text)

        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("OK", Gtk.ResponseType.OK)
        # Add a margin of 1 pixels between the buttons and the parent window
        dialog.get_content_area().set_property("margin", 1)
        dialog.set_keep_above(True)


        dialog.show_all()
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            text = buffer.get_text(buffer.get_start_iter(), buffer.get_end_iter(), True)
            dialog.destroy()
            print(text)
            return text
        else:
            dialog.destroy()
            return None


    def on_tree_selection_changed(self, selection):
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            print("You selected", model[treeiter][0])

    def connect_signals(self):
        self.builder.connect_signals({
            "on_send_button_clicked": self.on_send_button_clicked,
            "on_modify_button_clicked": self.on_modify_button_clicked,
            "on_delete_button_clicked": self.on_button_delete_clicked,
            "on_add_button_clicked": self.on_button_add_clicked,
            "delete_event": self.quit,
            "t_value": self.change_temperature,
            "sync_clicked": self.synchronize,
            "reset_clicked": self.reset_env,
            "load_clicked": self.load,
            "save_clicked": self.save,
            "new_clicked": self.new
        })

    def add_f(self, button):
        code = self.classe.cell_content()
        return code
    def synchronize(self, button):
        self.classe.synchronize()
        self.gestion_liststore('store')

    def reset_env(self, button):
        self.classe.reset()
        self.gestion_liststore('store')

    def change_temperature(self, button, n):
        self.temperature = self.slider.get_value()
        self.classe.set_temperature(self.temperature)

    def on_send_button_clicked(self, button):
        message = self.message_entry.get_buffer().get_text(self.message_entry.get_buffer().get_start_iter(),
            self.message_entry.get_buffer().get_end_iter(),
            True)
        if message:
            self.chat_area.get_buffer().insert_at_cursor("You: " + message + "\n")
            self.message_entry.get_buffer().set_text("")

            #history_prompt = [self.list_area.get_row_text(i) for i in range(self.list_area.get_n_items())]
            history_prompt =""
            self.classe.send_message(text=message, history_prompt=self.modified_prompt)
            self.print_action()

            self.gestion_liststore('store')


    def add_item_to_list(self, item):
        self.list.append([item])

    def update_list_area(self, new_list):
        # Clear the existing list
        self.list.clear()
        # Add items from the new list

        for item in new_list:
            self.list.append([str(item)])

    def print_action(self):
        if self.classe.print:
            for dico in reversed(self.classe.print):
                if "system_print" in dico:
                    buffer = self.chat_area.get_buffer()
                    buffer.insert_at_cursor("System: " + dico["system_print"] + "\n")
                if "IA_print" in dico:
                    buffer = self.chat_area.get_buffer()
                    buffer.insert_at_cursor("Verbose: " + dico["IA_print"] + "\n")
            self.classe.print = []

    def show_error_dialog(self, message):
        dialog = Gtk.MessageDialog(
            transient_for=None,
            flags=0,
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.OK,
            text=message,
        )
        # Add a margin of 4 pixels between the buttons and the parent window
        dialog.get_content_area().set_property("margin", 4)
        dialog.set_modal(True)
        dialog.set_keep_above(True)

        dialog.run()
        dialog.destroy()

    def on_modify_button_clicked(self, button):

        selection = self.tree.get_selection()
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            text = model[treeiter][0]
            response = self.create_textview_dialog("Modify", text)
            if response:
                try:
                    dico = ast.literal_eval(response)
                    model.set(treeiter, 0, response)
                    self.modified_prompt = self.convert_list_to_dict(self.get_liststore_as_list(model))
                except ValueError as e:
                    self.show_error_dialog(f"Error: not a dictionnary. Ignoring modification")
                except SyntaxError as se:
                    self.show_error_dialog(f"Error: not a dictionnary. Ignoring modification")

    def on_button_delete_clicked(self, widget):
        selection = self.tree.get_selection()
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            self.list.remove(treeiter)
            self.modified_prompt = self.convert_list_to_dict(self.get_liststore_as_list(model))
            print(self.modified_prompt)

    def on_button_add_clicked(self, widget):
        pass

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

    def load(self, button):

        dialog = Gtk.FileChooserDialog(title="Choisir un fichier", parent=None, action=Gtk.FileChooserAction.OPEN)

        # Ajouter un bouton Annuler à la boîte de dialogue
        dialog.add_button("_Annuler", Gtk.ResponseType.CANCEL)

        # Ajouter un bouton Ouvrir à la boîte de dialogue
        dialog.add_button("_Ouvrir", Gtk.ResponseType.OK)

        # Ajouter un filtre pour les fichiers .ipynb
        filter_ipynb = Gtk.FileFilter()
        filter_ipynb.set_name("Notebook IPython")
        filter_ipynb.add_pattern("*.ipynb")
        dialog.add_filter(filter_ipynb)

        # Exécuter la boîte de dialogue et récupérer le résultat
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            dialog.destroy()
            path_without_extension = os.path.splitext(filepath)[0]
            path=path_without_extension + ".pickle"
            # Recharger l'objet depuis le fichier
            if os.path.exists(path):
                with open(path, "rb") as fichier:
                    self.classe = pickle.load(fichier)
                    notebook = self.classe.notebook
                    library = self.classe.library
                    role=self.classe.role
            else:
                notebook = ""
                library = ""
                role = self.create_dialog_select_key()

            self.classe.__init__(library=library, role=role, notebook=notebook)
        else:
            dialog.destroy()
            return None


    def save(self, button):
        dialog = Gtk.FileChooserDialog(title="Enregistrer sous...", parent=None, action=Gtk.FileChooserAction.SAVE)

        # Ajouter un bouton Annuler à la boîte de dialogue
        dialog.add_button("_Annuler", Gtk.ResponseType.CANCEL)

        # Ajouter un bouton Enregistrer à la boîte de dialogue
        dialog.add_button("_Enregistrer", Gtk.ResponseType.OK)

        # Ajouter un filtre pour les fichiers .ipynb
        filter_ipynb = Gtk.FileFilter()
        filter_ipynb.set_name("Notebook IPython")
        filter_ipynb.add_pattern("*.ipynb")
        dialog.add_filter(filter_ipynb)

        # Exécuter la boîte de dialogue et récupérer le résultat
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            dialog.destroy()
            path_without_extension = os.path.splitext(filepath)[0]
            path = path_without_extension + ".pickle"
            # Enregistrer l'objet dans un fichier
            with open(path, "wb") as fichier:
                pickle.dump(self.classe, fichier)
            return filepath
        else:
            dialog.destroy()
            return None

    def new(self, button):
        # Créer une boîte de dialogue pour créer un nouveau fichier IPython Notebook
        dialog = Gtk.FileChooserDialog(
            title="Créer un nouveau fichier IPython Notebook",
            parent=None,
            action=Gtk.FileChooserAction.SAVE
        )

        # Ajouter les filtres pour les fichiers .ipynb
        filter_ipynb = Gtk.FileFilter()
        filter_ipynb.set_name("Fichiers IPython Notebook (*.ipynb)")
        filter_ipynb.add_pattern("*.ipynb")
        dialog.add_filter(filter_ipynb)

        # Ajouter un bouton pour créer le fichier
        dialog.add_button("Créer", Gtk.ResponseType.OK)

        # Ajouter un bouton pour annuler
        dialog.add_button("Annuler", Gtk.ResponseType.CANCEL)

        # Afficher la boîte de dialogue
        response = dialog.run()

        # Récupérer le chemin du fichier sélectionné
        if response == Gtk.ResponseType.OK:
            filepath = dialog.get_filename()
            if not filepath.endswith('.ipynb'):
                filepath += '.ipynb'
        else:
            filepath = None

        # Fermer la boîte de dialogue
        dialog.destroy()
        role = self.create_dialog_select_key()
        if role:
            self.classe = user(library="", role=role, notebook=filepath)
            self.gestion_liststore('init')

        return filepath

    def create_dialog_select_key(self, title: str = "Choose role"):

        with open("role.json", 'r') as f:
            dictionary = json.load(f)


        dialog = Gtk.Dialog(title=title, modal=True, transient_for=None, flags=0)
        dialog.set_default_size(300, 200)

        box = dialog.get_content_area()
        listbox = Gtk.ListBox()
        box.pack_start(listbox, True, True, 0)

        for key in dictionary.keys():
            row = Gtk.ListBoxRow()
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
            label = Gtk.Label(key)
            hbox.pack_start(label, True, True, 0)
            row.add(hbox)
            listbox.add(row)

        dialog.add_button("Cancel", Gtk.ResponseType.CANCEL)
        dialog.add_button("OK", Gtk.ResponseType.OK)

        dialog.show_all()
        response = dialog.run()

        if response == Gtk.ResponseType.OK:
            selected_row = listbox.get_selected_row()
            if selected_row is not None:
                keys_list = list(dictionary.keys())
                index = listbox.get_children().index(selected_row)
                key = keys_list[index]
                #key = selected_row.get_children()[0].get_text()
                dialog.destroy()
                return key
            else:
                dialog.destroy()
                return None
        else:
            dialog.destroy()
            return None

    def quit(self, button):
        # Créer une boîte de dialogue de confirmation
        dialog = Gtk.MessageDialog(transient_for=None, flags=0, message_type=Gtk.MessageType.QUESTION,
                                   buttons=Gtk.ButtonsType.OK_CANCEL, text="Voulez-vous vraiment quitter ?")
        response = dialog.run()
        dialog.destroy()

        # Arrêter le programme si l'utilisateur a cliqué sur le bouton "OK"
        if response == Gtk.ResponseType.OK:
            Gtk.main_quit()
            exit()
        else:
            return True  # Empêcher la fermeture de la fenêtre si l'utilisateur a cliqué sur le bouton "Annuler"


if __name__ == "__main__":

    chat_window = ChatWindow()
    Gtk.main()

    '''
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()
    '''
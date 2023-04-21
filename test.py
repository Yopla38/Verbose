import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gdk

css_provider = Gtk.CssProvider()
css_provider.load_from_path("./Utils/Verbose.css")

screen = Gdk.Screen.get_default()
context = Gtk.StyleContext()
context.add_provider_for_screen(screen, css_provider,
                                 Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

class TreeViewWindow(Gtk.Window):

    def __init__(self):
        Gtk.Window.__init__(self, title="TreeView Window")

        # Créer le TreeView
        self.treeview = Gtk.TreeView()

        # Créer les colonnes
        column_names = ['Nom', 'Âge']
        for i, column_name in enumerate(column_names):
            renderer = Gtk.CellRendererText()
            column = Gtk.TreeViewColumn(column_name, renderer, text=i)
            self.treeview.append_column(column)

        # Créer le modèle de données
        self.data_list = [['Alice', '25'], ['Bob', '30'], ['Charlie', '35']]
        self.store = Gtk.ListStore(str, str)
        for data in self.data_list:
            self.store.append(data)
        self.treeview.set_model(self.store)

        # Créer les boutons
        self.button_modify = Gtk.Button(label="Modifier")
        self.button_modify.connect("clicked", self.on_button_modify_clicked)

        self.button_delete = Gtk.Button(label="Supprimer")
        self.button_delete.connect("clicked", self.on_button_delete_clicked)

        self.button_add = Gtk.Button(label="Ajouter")
        self.button_add.connect("clicked", self.on_button_add_clicked)

        # Créer la boîte de dialogue pour modifier ou ajouter des éléments
        self.dialog = Gtk.Dialog(title="Modifier / Ajouter", parent=self, modal=True)
        self.dialog.add_buttons("Annuler", Gtk.ResponseType.CANCEL, "Ok", Gtk.ResponseType.OK)

        self.dialog_entry = Gtk.Entry()
        self.dialog_entry.set_text("Texte en exemple")
        self.dialog.get_content_area().add(self.dialog_entry)

        # Créer le ScrolledWindow et ajouter le TreeView
        self.scrolled_window = Gtk.ScrolledWindow()
        self.scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self.scrolled_window.add(self.treeview)

        # Créer la boîte horizontale pour les boutons
        self.box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.box.pack_start(self.button_modify, True, True, 0)
        self.box.pack_start(self.button_delete, True, True, 0)
        self.box.pack_start(self.button_add, True, True, 0)

        # Créer la boîte verticale pour le ScrolledWindow et les boutons
        self.vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.vbox.pack_start(self.scrolled_window, True, True, 0)
        self.vbox.pack_start(self.box, False, False, 0)

        # Ajouter la boîte verticale dans la fenêtre
        self.add(self.vbox)

    def on_button_modify_clicked(self, widget):
        selection = self.treeview.get_selection()
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            text = model[treeiter][0]
            self.dialog_entry.set_text(text)
            response = self.dialog.run()
            if response == Gtk.ResponseType.OK:
                new_text = self.dialog_entry.get_text()
                model[treeiter][0] = new_text
                self.store.set(self.store.get_path(treeiter), [new_text, model[treeiter][1]])
            self.dialog.hide()

    def on_button_delete_clicked(self, widget):
        selection = self.treeview.get_selection()
        model, treeiter = selection.get_selected()
        if treeiter is not None:
            self.store.remove(treeiter)

    def on_button_add_clicked(self, widget):
        self.dialog_entry.set_text("Texte en exemple")
        response = self.dialog.run()
        if response == Gtk.ResponseType.OK:
            text = self.dialog_entry.get_text()
            self.store.append([text, ''])
        self.dialog.hide()

win = TreeViewWindow()
win.connect("destroy", Gtk.main_quit)
win.show_all()
Gtk.main()

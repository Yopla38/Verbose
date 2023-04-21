import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

def select_libraries_dialog(parent_window, libraries):
    # Créer une boîte de dialogue pour choisir des bibliothèques
    dialog = Gtk.Dialog(
        "Choisir des bibliothèques",
        parent_window,
        Gtk.DialogFlags.MODAL | Gtk.DialogFlags.DESTROY_WITH_PARENT,
        (
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
            Gtk.STOCK_OK,
            Gtk.ResponseType.OK,
        ),
    )

    # Créer une liste pour afficher les bibliothèques
    listbox = Gtk.ListBox()
    listbox.set_selection_mode(Gtk.SelectionMode.MULTIPLE)

    # Ajouter les bibliothèques à la liste
    for library in libraries:
        library_name = list(library.keys())[0]
        library_prompt = library[library_name]
        row = Gtk.ListBoxRow()
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=50)
        label_name = Gtk.Label(label=library_name)
        label_prompt = Gtk.Label(label=library_prompt)
        hbox.pack_start(label_name, True, True, 0)
        hbox.pack_start(label_prompt, True, True, 0)
        row.add(hbox)
        listbox.add(row)

    # Ajouter la liste à la boîte de dialogue
    dialog.get_content_area().add(listbox)
    dialog.show_all()

    # Afficher la boîte de dialogue et récupérer la réponse
    response = dialog.run()
    selected_libraries = []
    if response == Gtk.ResponseType.OK:
        # Récupérer les bibliothèques sélectionnées
        selected_rows = listbox.get_selected_rows()
        for selected_row in selected_rows:
            selected_library_name = selected_row.get_children()[0].get_text()
            selected_library = next(
                (library for library in libraries if library_name in library), None
            )
            selected_libraries.append(selected_library)

    # Fermer la boîte de dialogue
    dialog.destroy()

    return selected_libraries

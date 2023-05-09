import tkinter as tk
from tkinter import messagebox
import tkinter.simpledialog as simpledialog


class ChatWindow:
    def __init__(self, master):
        self.master = master
        master.title("Verbose V_0.1")
        # Définir le gestionnaire de géométrie "grid"
        master.grid_columnconfigure(0, pad=10, weight=1)
        master.grid_columnconfigure(1, pad=10, weight=1)
        master.grid_columnconfigure(2, pad=10, weight=1)
        master.grid_columnconfigure(3, pad=10, weight=1)
        master.grid_columnconfigure(4, pad=10, weight=1)
        #master.grid_columnconfigure(1, weight=2)
        master.grid_rowconfigure(0, pad=10, weight=1)
        self.items = []
        self.font = ("Courier", 14)
        self.bg = "#444e5e"
        self.hlbg = '#000000'
        self.fg = "white"
        # Créer la zone de texte pour afficher les messages height=18, width=50
        self.chat_history = tk.Text(master, state=tk.DISABLED, font=self.font, bg=self.bg, fg=self.fg, highlightbackground=self.hlbg)
        self.chat_history.grid(row=0, column=0, rowspan=3, columnspan=4, padx=5, pady=5)

        # Créer la zone de liste pour afficher les éléments height=17, width=35
        self.listbox = tk.Listbox(master, width=35, font=self.font, bg=self.bg, highlightbackground=self.hlbg, fg=self.fg)
        self.listbox.grid(row=3, column=3, rowspan=3, columnspan=5, padx=5, pady=5)


        self.temperature = tk.Scale(master, from_=1, to=0, resolution=0.1, orient=tk.VERTICAL, length=100, label="Temp", font=self.font, bg=self.bg, highlightbackground=self.hlbg, troughcolor="#4d4d4d", fg=self.fg)
        self.temperature.grid(row=0, column=5, padx=5, pady=5)

        self.filename_entry = tk.Entry(master)
        self.filename_entry.grid(row=2, column=5, padx=5, pady=5)

        # Créer la zone de texte pour afficher le nom de fichier
        self.filename_label = tk.Label(master, text="Jupyter NB", font=self.font, bg=self.bg,
                                       highlightbackground=self.hlbg, fg=self.fg)
        self.filename_label.grid(row=1, column=5, padx=5, pady=5)

        # Créer la zone de texte pour saisir les messages
        self.chat_entry = tk.Text(master, height=5, width=50, font=self.font, bg=self.bg, highlightbackground=self.hlbg, fg=self.fg)
        self.chat_entry.grid(row=4, column=0, rowspan=1, padx=5, pady=5)

        # Créer le bouton "Envoyer"
        self.send_button = tk.Button(master, text="Envoyer", command=self.send, font=self.font, bg=self.bg, highlightbackground=self.hlbg, fg=self.fg)
        self.send_button.grid(row=5, column=0, padx=5, pady=5)



        # Créer les boutons "Modifier", "Supprimer" et "Ajouter"
        self.edit_button = tk.Button(master, text="Modifier", command=self.modify_item, font=self.font, bg=self.bg, highlightbackground=self.hlbg, fg=self.fg)
        self.edit_button.grid(row=3, column=1, padx=5, pady=5)

        self.delete_button = tk.Button(master, text="Supprimer", command=self.delete_item, font=self.font, bg=self.bg, highlightbackground=self.hlbg, fg=self.fg)
        self.delete_button.grid(row=4, column=1, padx=5, pady=5)

        self.add_button = tk.Button(master, text="Ajouter", command=self.add_list, font=self.font, bg=self.bg, highlightbackground=self.hlbg, fg=self.fg)
        self.add_button.grid(row=5, column=1, padx=5, pady=5)

    

   


    def send(self):
        message = self.chat_entry.get("1.0", tk.END).strip()
        if message:
            self.chat_entry.delete("1.0", tk.END)
            self.chat_history.configure(state=tk.NORMAL)
            self.chat_history.insert(tk.END, f"Vous : {message}\n")
            self.chat_history.configure(state=tk.DISABLED)

    # Ajoute un message au chat
    def add_message(self, message):
        self.chat_history.configure(state=tk.NORMAL)
        self.chat_history.insert(tk.END, message + "\n")
        self.chat_history.configure(state=tk.DISABLED)

    # Ajoute une liste à la listebox
    def add_list(self, items):
        self.items = items
        self.listbox.delete(0, tk.END)
        for item in self.items:
            self.listbox.insert(tk.END, item)

    # Méthode appelée lorsqu'on clique sur le bouton "Modifier"
    def modify_item(self):
        # Récupère l'indice de l'élément sélectionné dans la listebox
        selection = self.listbox.curselection()
        if len(selection) == 1:
            # Affiche une boîte de dialogue pour modifier l'élément sélectionné
            item = self.listbox.get(selection[0])
            new_item = simpledialog.askstring("Modifier l'élément", "Nouvelle valeur :", initialvalue=item)
            if new_item is not None:
                # Modifie l'élément sélectionné dans la listebox et dans la liste
                self.listbox.delete(selection[0])
                self.listbox.insert(selection[0], new_item)
                self.items[selection[0]] = new_item

    # Méthode appelée lorsqu'on clique sur le bouton "Supprimer"
    def delete_item(self):
        # Récupère l'indice de l'élément sélectionné dans la listebox
        selection = self.listbox.curselection()
        if len(selection) == 1:
            # Supprime l'élément sélectionné de la listebox et de la liste
            self.listbox.delete(selection[0])
            del self.items[selection[0]]


if __name__ == "__main__":
    root = tk.Tk()
    # Récupération des dimensions de l'écran
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # Calcul de la moitié des dimensions de l'écran
    half_screen_width = screen_width // 2
    half_screen_height = screen_height // 2
    # Redimensionnement de la fenêtre
    root.geometry(f"{half_screen_width}x{half_screen_height}+0+0")
    root.configure(bg="#444e5e")

    chat = ChatWindow(root)
    root.mainloop()

import tkinter as tk

class Chat:
    def __init__(self):
        # Créer la fenêtre graphique
        self.fenetre = tk.Tk()
        self.fenetre.title("Chat")

        # Ajouter un widget de zone de texte pour le chat
        self.chat = tk.Text(self.fenetre, state=tk.DISABLED)
        self.chat.pack(fill=tk.BOTH, expand=True)

        # Ajouter un widget d'entrée pour taper le texte
        self.entree = tk.Entry(self.fenetre)
        self.entree.pack(fill=tk.X)

        # Lier la touche Entrée pour ajouter du texte au chat
        def envoyer_message(event):
            texte = self.entree.get()
            self.print("Vous : " + texte)
            self.entree.delete(0, tk.END)
        self.entree.bind("<Return>", envoyer_message)

        # Ajouter un premier message au chat
        self.print("Bienvenue dans le chat !")

    def print(self, texte):
        self.chat.config(state=tk.NORMAL)
        self.chat.insert(tk.END, texte + '\n')
        self.chat.config(state=tk.DISABLED)
        self.chat.yview(tk.END)

    def input(self, prompt=''):
        self.entree.delete(0, tk.END)
        self.entree.insert(0, prompt)
        self.entree.focus()
        self.fenetre.wait_window()

        return self.entree.get()

    def run(self):
        # Lancer la boucle d'événements Tkinter
        self.fenetre.mainloop()

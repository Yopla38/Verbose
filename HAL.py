import ast
import json
import os
import time
from typing import List

import requests
import xml.etree.ElementTree as ET
import scholarly
import datetime
import difflib
import xlwt
# from Jupyter_page import JupyterNotebook
import Verbose
from Verbose import verbose


class Chercheur:
    def __init__(self, nom, prenom, affiliation):
        self.base_url = "https://api.archives-ouvertes.fr/search/"
        self.nom = nom
        self.prenom = prenom
        self.full_name = self.prenom + " " + self.nom

        # _________________________HAL___________________________
        self.affiliation = affiliation
        self.auth_id_hal = None
        self.publication_HAL = None

        # ______________________GOOGLE_____________________________
        self.all_google_information = None
        self.last_update_google = None
        self.publication_google = None

    def chercheur_existe_hal(self, nom_chercheur):
        # URL de l'API HAL pour rechercher un chercheur par nom
        url = "https://api.archives-ouvertes.fr/ref/author/?q={nom_chercheur}&wt=json".format(
            nom_chercheur=nom_chercheur)

        # Envoi de la requête à l'API HAL
        response = requests.get(url)

        # Traitement de la réponse JSON
        if response.status_code == 200:
            data = response.json()
            # Vérification si le chercheur existe
            if data["response"]["numFound"] > 0:
                return True
            else:
                return False
        else:
            print("Une erreur est survenue lors de la requête : {code}".format(code=response.status_code))
            return False

    def extract_address_link(self, dictionary):
        address = ""
        link = ""
        if "address" in dictionary and "addrLine" in dictionary["address"]:
            address = dictionary["address"]["addrLine"]
        if "ref" in dictionary:
            link = dictionary["ref"]
        return address, link

    def extract_title_HAL_and_google(self):
        title_HAL = []
        title_google = []
        if self.publication_HAL is not None:
            for publication in self.publication_HAL:
                title_HAL.append(publication["title"])

        publications_google = self.get_publication_google()
        if publications_google is not None:
            for publication in publications_google:
                title_google.append(publication['title'])

        return title_HAL, title_google

    def update_HAL(self):
        url = "https://api.archives-ouvertes.fr/search/?q=NEEL-NPSC%20AND%20authFullName_t:" + self.full_name
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            publications = []
            # print(data)
            nombre = len(data["response"]["docs"])
            print("Wait...")
            for doc in data["response"]["docs"]:
                print("Nombre de publication dans HAL à traiter: " + str(nombre))
                if "title_s" in doc:
                    publications.append(self.nettoyer_publication_HAL(doc["title_s"]))
                elif "label_s" in doc:
                    publications.append(self.nettoyer_publication_HAL(doc["label_s"]))
                nombre -= 1
            #self.publication_HAL = self.nettoyer_publication_HAL_one_shoot(publications)
            self.publication_HAL = publications
        else:
            print(f"Error {response.status_code} while fetching data for {self.nom}")
            return []

    def get_publications_HAL(self) -> List[str]:
        return self.publication_HAL

    def get_auth_id_hal(self):
        if self.auth_id_hal is None:
            params = {
                "q": f"authFullName_s:\"{self.full_name}\"",
                "wt": "json"
            }
            response = requests.get("https://api.archives-ouvertes.fr/search", params=params)
            if response.status_code == 200:
                data = response.json()
                if "response" in data and "docs" in data["response"] and len(data["response"]["docs"]) > 0:
                    self.auth_id_hal = data["response"]["docs"][0]["authIdHal_i"]
        return self.auth_id_hal

    def nettoyer_publication_HAL_one_shoot(self, publi):
        publicat = '\n'.join(repr(e) for e in publi)
        publications = "[" + publicat + "]"
        v = verbose(role='string_to_listdict')
        return v.format_string(publications)

    def nettoyer_publication_HAL(self, publication):
        v = verbose(role='string_to_dict')
        return v.format_string(publication)


    def add_publication_to_hal(self, title, authors, abstract):
        # Remplir les champs nécessaires pour l'API HAL
        data = {
            'title_s': title,
            'authFullName_s': authors,
            'abstract_t': abstract,
            'halId_s': 'monidentifianthal',  # remplacer par votre identifiant HAL
            'submit_s': 'Valider'
        }

        # Envoyer une requête POST à l'API HAL pour ajouter la publication
        r = requests.post('https://api.archives-ouvertes.fr/ref/submit/', data=data)

        # Vérifier si la publication a été ajoutée avec succès
        if r.status_code == 200:
            print("Publication ajoutée avec succès dans HAL")
        else:
            print("Erreur lors de l'ajout de la publication dans HAL")

    # ___________________GOOGLE___________________________
    def update_google(self):
        self.get_information_google()
        self.last_update_google = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.publication_google = self.get_publication_google()

    def get_information_google(self):
        # Get an iterator for the author results
        search_query = scholarly.scholarly.search_author(self.full_name)
        # Retrieve the first result from the iterator
        try:
            first_author_result = next(search_query, None)
        # scholarly.scholarly.pprint(first_author_result)
            # Retrieve all the details for the author
            author = scholarly.scholarly.fill(first_author_result)
            self.all_google_information = author
        except ValueError as e:
            print("Encountered a ValueError:", e)
            print(search_query)
            return
        except Exception as e:
            print("Encountered an error:", e)
            print(search_query)
            return


    def _get_publication_google(self):
        publication_titles = [pub['bib']['title'] for pub in self.all_google_information['publications']]
        return publication_titles

    def get_publication_google(self):
        publications = []
        for pub in self.all_google_information['publications']:
            publication_dict = {}
            for key, value in pub['bib'].items():
                publication_dict[key] = value
            publications.append(publication_dict)
        return publications

    def cited_by_google(self, no_publication: int = 0):
        # Which papers cited that publication?
        publication = self.all_google_information['publications'][no_publication]
        publication_filled = scholarly.scholarly.fill(publication)
        cited_by = scholarly.scholarly.citedby(publication_filled)
        print(cited_by)
        return cited_by

    def get_affiliations_google(self):
        affiliations = []
        for affiliation in self.all_google_information['affiliations']:
            affiliations.append(affiliation['name'])
        return affiliations

    def compare_lists_publications(self, listA, listB, n, cutoff):
        '''Cette fonction utilise la méthode "get_close_matches" de la bibliothèque "difflib" pour trouver les titres
        d'articles scientifiques similaires dans les deux listes, même s'ils ne sont pas formatés exactement de la même
         manière. Les résultats sont stockés dans deux listes de tuples. Chaque tuple contient le titre d'article
         scientifique et le titre similaire dans l'autre liste. Si le titre n'a pas de correspondance, le deuxième
         élément du tuple est None.'''

        # Trouver les titres d'articles scientifiques de listB qui ne sont pas dans listA
        not_in_A = []
        for title in listB:
            if title not in listA:
                matches = difflib.get_close_matches(title, listA, n=n, cutoff=cutoff)
                if len(matches) > 0:
                    not_in_A.append((title, matches[0]))
                else:
                    not_in_A.append((title, None))

        # Trouver les titres d'articles scientifiques de listA qui ne sont pas dans listB
        not_in_B = []
        for title in listA:
            if title not in listB:
                matches = difflib.get_close_matches(title, listB, n=n, cutoff=cutoff)
                if len(matches) > 0:
                    not_in_B.append((title, matches[0]))
                else:
                    not_in_B.append((title, None))

        return not_in_A, not_in_B


class Equipe:
    def __init__(self, nom: str):
        self.nom = nom
        # Définir le nom du répertoire
        self.repertoire = os.path.join(os.path.abspath(''), "Equipes", nom)
        # Vérifier si le répertoire existe, sinon le créer
        if not os.path.exists(self.repertoire):
            os.makedirs(self.repertoire)
        self.membres = []
        self.database_path = os.path.join(self.repertoire, f"{self.nom}.json")
        self.charger_base_de_donnees()


    def ajouter_chercheur(self, nom: str, prenom: str, affiliation: str = "NPSC"):
        membre = Chercheur(nom, prenom, affiliation)
        self.membres.append(membre)
        return membre

    def supprimer_chercheur(self, nom: str, prenom: str):
        for i in range(len(self.membres)):
            if self.membres[i].nom == nom and self.membres[i].prenom == prenom:
                del self.membres[i]
                break

    def lister_membres(self):
        for membre in self.membres:
            print(f"{membre.nom}, {membre.prenom}, {membre.affiliation}")
        return self.membres

    def update_tous_les_chercheurs(self):
        for membre in self.membres:
            print(f"Traitement des publications de {membre.prenom} {membre.nom}")
            self.update_chercheur(membre.nom, membre.prenom)

    def update_chercheur(self, nom: str, prenom: str):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                membre.update_google()
                membre.update_HAL()
                return

    def update_chercheur_HAL(self, nom: str, prenom: str):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                membre.update_HAL()
                return

    def get_publications_entre_dates(self, debut: datetime.datetime, fin: datetime.datetime) -> List[str]:
        publications = []
        for membre in self.membres:
            for publication in membre.get_publication_google():
                publication_date = datetime.datetime.strptime(publication['bib']['pub_year'], "%Y")
                if debut <= publication_date <= fin:
                    publications.append(publication['bib']['title'])
        return publications

    def enregistrer_base_de_donnees(self):
        data = {
            "nom": self.nom,
            "membres": [m.__dict__ for m in self.membres]
        }
        with open(self.database_path, "w") as f:
            json.dump(data, f)

    def charger_base_de_donnees(self):
        if os.path.exists(self.database_path):
            with open(self.database_path, "r") as f:
                data = json.load(f)
                for m in data["membres"]:
                    membre = Chercheur(m["nom"], m["prenom"], m["affiliation"])
                    membre.last_update_google = m["last_update_google"]
                    membre.all_google_information = m["all_google_information"]
                    membre.publication_HAL = m["publication_HAL"]

                    self.membres.append(membre)

    def consulter_publications_chercheur_google(self, nom: str, prenom: str, affichage:bool = True):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                publications = membre.get_publication_google()
                if affichage:
                    if publications is not None:
                        print(f"Publications de {membre.full_name}:")
                        for publication in publications:
                            print(publication)
                        break
                    else:
                        print("Il n'y a pas de publication présente")
                else:
                    return publications
            else:
                print(f"Le chercheur {prenom} {nom} n'est pas dans l'équipe.")

    def consulter_publications_chercheur_HAL(self, nom: str, prenom: str, affichage:bool = True):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                publications_HAL = membre.get_publications_HAL()
                if affichage:
                    print(f"Publications de {membre.full_name}:")
                    if publications_HAL is not None:
                        for publication in publications_HAL:
                            print(publication)
                        break
                    else:
                        print("Il n'y a pas de publication présente")
                else:
                    return publications_HAL
            else:
                print(f"Le chercheur {prenom} {nom} n'est pas dans l'équipe.")

    def get_citations_publication(self, publication_titre: str):
        citations = []
        for membre in self.membres:
            membre.update_google()
            publications = membre.all_google_information['publications']
            for publication in publications:
                if publication['bib']['title'] == publication_titre:
                    citation_result = membre.cited_by_google(publications.index(publication))
                    for citation in citation_result:
                        citations.append(citation.bib['title'])
                    break
        return citations

    def chercher_publication(self, titre: str):
        for membre in self.membres:
            for publication in membre.get_publication_google():
                if titre.lower() in publication['bib']['title'].lower():
                    print(
                        f"Le titre '{titre}' est présent dans la publication '{publication['bib']['title']}' de {membre.full_name}")

    def pas_present_dans_HAL(self, nom: str, prenom: str, n: float = 1.0, cutoff: float = 0.7):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                A, B = membre.extract_title_HAL_and_google()
                not_in_A, not_in_B = membre.compare_lists_publications(A, B, n, cutoff)
                if len(not_in_A) > 0:
                    print(f"Publications de {membre.full_name} non présentes dans HAL :")
                    for title in not_in_A:
                        print(title)
                return not_in_A
            else:
                print(f"Le chercheur {prenom} {nom} n'est pas dans l'équipe.")

    def pas_present_dans_google(self, nom: str, prenom: str, n: float = 1.0, cutoff: float = 0.7):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                A, B = membre.extract_title_HAL_and_google()
                not_in_A, not_in_B = membre.compare_lists_publications(A, B, n, cutoff)
                if len(not_in_B) > 0:
                    print(f"Publications de {membre.full_name} non présentes dans google scholar :")
                    for title in not_in_B:
                        print(title)
                return not_in_B
            else:
                print(f"Le chercheur {prenom} {nom} n'est pas dans l'équipe.")

    def creer_fichier_excel_equipe(self):
        for membre in self.membres:
            self.creer_fichier_excel_chercheur(membre.nom, membre.prenom)

    def creer_fichier_excel_chercheur(self, nom, prenom):
        for membre in self.membres:
            if membre.nom == nom and membre.prenom == prenom:
                publi_HAL, publi_google = membre.extract_title_HAL_and_google()
                pas_dans_HAL, pas_dans_google = membre.compare_lists_publications(publi_HAL, publi_google, 1, 0.6)

                # Créer un nouveau fichier Excel
                wb = xlwt.Workbook()

                # Ajouter une nouvelle feuille à ce fichier
                sheet = wb.add_sheet('Publications')

                # Définir la largeur des colonnes
                col_width = 14  # en centimètres
                col_width_pixels = int(col_width * 28.3465)  # conversion en points
                for i in range(4):
                    sheet.col(i).width = col_width_pixels * 40

                # Ajouter les en-têtes de colonnes
                sheet.write(0, 0, 'Publications GoogleScholar')
                sheet.write(0, 1, 'Publications HAL')
                sheet.write(0, 2, 'Publications non présentes dans HAL')
                sheet.write(0, 3, 'Publications non présentes dans Google')

                # Ajouter les données
                for i in range(max(len(publi_google), len(publi_HAL), len(pas_dans_HAL), len(pas_dans_google))):
                    if i < len(publi_google):
                        sheet.write(i + 1, 0, publi_google[i])
                    if i < len(publi_HAL):
                        sheet.write(i + 1, 1, publi_HAL[i])
                    if i < len(pas_dans_HAL):
                        print(pas_dans_HAL[i][0])
                        sheet.write(i + 1, 2, pas_dans_HAL[i][0])
                    if i < len(pas_dans_google):
                        sheet.write(i + 1, 3, pas_dans_google[i])

                # Sauvegarder le fichier
                excel_path = os.path.join(self.repertoire, f"{nom}_{prenom}.xls")
                wb.save(excel_path)
                print(f"Fichier écrit dans {excel_path} !")
            else:
                print(f"Le chercheur {prenom} {nom} n'est pas dans l'équipe.")


'''
Vous disposez de plusieurs fonctions. USER va exprimer ce qu\'il souhaite et vous devez écrire le code complet python permettant d'obtenir la réponse à la question demandée. Votre réponse doit toujours être au format: EXECUTE :\n code à executer multiligne\nAFFICHAGE DE LA REPONSE :\ncode python et phrase pour repondre à la question\n.N\'écrivez jamais d\'explications. Liste des fonctions : ajouter_chercheur(nom, prenom, affiliation); supprimer_chercheur(nom, prenom); lister_membres(); update_tous_les_chercheurs(); update_chercheur(nom, prenom) va chercher sur HAL et google scholar toute ses publications; update_chercheur_HAL(nom, prenom) va chercher sur HAL toute ses publications; enregistrer_base_de_donnees(); consulter_publications_chercheur_google(nom, prenom) retourne toute les publications de google scholar du chercheur; consulter_publication_chercheur_HAL(nom, prenom) retourne toute les publications de HAL du chercheur; pas_present_dans_HAL(nom, prenom) retourne la liste des publications présente dans google scholar mais pas dans HAL; pas_present_dans_google(nom, prenom) retourne la liste des publications présente dans HAL mais pas dans google scholar; creer_fichier_excel_chercheur(nom, prenom). Toute ces fonctions sont présente dans une classe Equipe où chaque chercheur est dans self.membres. L'objet Equipe est déjà existante sous le nom NPSC.
Cas 1 : il manque, pour répondre, un nom,un prenom ou une affiliation dans le texte de [USER] pour répondre correctement alors posez la bonne question avec input dans le champ EXECUTE. Cas 2 : vous avez toute les informations alors vous résolvez le problème.
[USER]: Vérifier que Bruno a toutes ses publications à jours


Vous devez implementer une fonction python en utilisant les fonctions ci dessous(tout les arguments seront impérativement renseignés) pour trouver la reponse de USER. Le code doit etre directement executable.Ne faites pas de commentaire, la réponse doit être de la forme : F: fonction executable multiligne\nCODE: code executant la fonction. Vous devez poser une question avec la fonction input dans F si un argument de fonction est manquant pour résoudre le problème. Les membres de l'équipe sont disponibles dans self.membres. La base de données est NPSC. A la question d'après, vous devez toujours écrire la réponse de la même manière.
Format des fonctions: function_name(arg,...)->type:description


Vous devez implementer une fonction python répondant à USER en utilisant les fonctions ci dessous(tout les arguments seront impérativement renseignés) et en déduisant les arguments des fonctions grace à la question de USER. Le code doit etre directement executable.Ne faites pas de commentaire, la réponse doit être de la forme : F: fonction executable multiligne\nCODE: code executant la fonction. Toute question posée doit être executable en python avec la fonction input dans F si un argument de fonction est manquant pour résoudre le problème. A la question d'après, vous devez toujours écrire la réponse de la même manière. Vous devez utiliser le plus possible les fonctions fournies. Les membres de l'équipe sont disponibles dans self.membres. La base de données est NPSC. 
Format des fonctions: function_name(arg,...)->type:description
Fonctions python de la classe equipe disponibles:ajouter_chercheur(nom, prenom, affiliation)->None: Ajoute un chercheur à la base;supprimer_chercheur(nom, prenom)->None: Supprime un chercheur de la base;lister_membres()->List string:Liste tous les chercheurs dans la base;update_tous_les_chercheurs()->None:Met à jour toute les publications des chercheurs sur HAL ET GoogleScholar
update_chercheur(nom, prenom)->None: met a jour les publications de HAL et google scholar;update_chercheur_HAL(nom, prenom)->None: met a jour les publications de HAL;enregistrer_base_de_donnees()->None:Enregistre la base de donnée;consulter_publications_chercheur_google(nom, prenom)->List string:retourne toute les publications de google scholar du chercheur;consulter_publication_chercheur_HAL(nom, prenom)->List string: retourne toute les publications de HAL du chercheur;pas_present_dans_HAL(nom, prenom)->List string:retourne la liste des publications présente uniquement dans google;pas_present_dans_google(nom, prenom)->List string:retourne la liste des publications présente uniquement dans HAL


Classe `Equipe` :
- `__init__(self, nom: str)`: Initialise une nouvelle instance d'équipe avec un nom donné et crée un répertoire pour l'équipe.
- `ajouter_chercheur(self, nom: str, prenom: str, affiliation: str)`: crée un objet chercheur et ajoute un chercheur à l'équipe.
- `supprimer_chercheur(self, nom: str, prenom: str)`: Supprime un chercheur de l'équipe.
- `lister_membres(self)`: Affiche les membres de l'équipe.
- `update_tous_les_chercheurs(self)`: Met à jour les informations de tous les chercheurs de l'équipe.
- `update_chercheur(self, nom: str, prenom: str)`: Met à jour un chercheur spécifique de l'équipe.
- `update_chercheur_HAL(self, nom: str, prenom: str)`: Met à jour les informations HAL d'un chercheur spécifique.
- `get_publications_entre_dates(self, debut: datetime.datetime, fin: datetime.datetime) -> List[str]`: Récupère les publications entre deux dates.
- `enregistrer_base_de_donnees(self)`: Enregistre la base de données de l'équipe.
- `charger_base_de_donnees(self)`: Charge la base de données de l'équipe.
- `consulter_publications_chercheur_google(self, nom: str, prenom: str)`: Affiche les publications Google Scholar d'un chercheur.
- `consulter_publications_chercheur_HAL(self, nom: str, prenom: str)`: Affiche les publications HAL d'un chercheur.
- `get_citations_publication(self, publication_titre: str)`: Récupère les citations d'une publication.
- `chercher_publication(self, titre: str)`: Recherche une publication par titre.
- `pas_present_dans_HAL(self, nom: str, prenom: str, n: float = 1.0, cutoff: float = 0.7)`: Recherche les publications non présentes dans HAL.
- `pas_present_dans_google(self, nom: str, prenom: str, n: float = 1.0, cutoff: float = 0.7)`: Recherche les publications non présentes dans Google Scholar.
- `creer_fichier_excel_equipe(self)`: Crée un fichier Excel pour l'équipe.
- `creer_fichier_excel_chercheur(self, nom, prenom)`: Crée un fichier Excel pour un chercheur spécifique.

'''
if __name__ == '__main__':
    v = verbose("NPSC")
    v.write_instruction()

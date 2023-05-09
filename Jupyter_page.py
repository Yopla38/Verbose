import time

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
import os
import webbrowser
import subprocess
from jupyter_client import KernelManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from IPython.display import Javascript, display
import platform


class JupyterNotebook:
    def __init__(self, filename="conversation.ipynb", port: int = 8888):

        self.process = None
        self.notebook_name = os.path.basename(filename)
        self.notebook_path = os.path.abspath(filename)
        self.browser = None
        self.chromedriver_path = self.get_chromedriver_path()
        self.token = ''
        # crée une pile de dictionnaire printable dans l'interface.
        # "system_print"
        # "system_input"
        # "IA_print"
        # "IA_input"
        self.print = []
        if os.path.exists(self.notebook_path):
            self.load_notebook()
        else:
            self.nb = new_notebook()

        self.port_number = port
        if self.process is None:
            self.launch_jupyter_server_and_get_token(port)

        time.sleep(3)

    def new_notebook(self):
        self.nb = new_notebook()

    def read_output(self, jupyter_process, output_queue):
        while True:
            output = jupyter_process.stdout.readline()
            if "http://localhost:" in output:
                output_queue.put(output)
                break

    def launch_jupyter_server_and_get_token(self, port: int = 8888):
        #"--no-browser"
        process = subprocess.Popen(
            ["jupyter", "notebook", f"--port={port}", "--no-browser"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.process = process

    def wait_webbrowser(self, element):
        element = WebDriverWait(self.browser, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "div.input_area"))
        )

    def add_code_cell(self, code):
        cell = new_code_cell(source=code)
        self.nb.cells.append(cell)
        return cell

    def select_cell(self, index):
        """Select a cell by index"""
        cell = self.browser.find_element_by_css_selector('div.cell:nth-child({})'.format(index))
        self.browser.switch_to.frame(cell)
        return cell

    def find_last_cell(self):
        # Sélectionner tous les éléments correspondant à la cellule
        cell_elements = self.browser.find_elements_by_xpath('//div[contains(@class, "cell")]')
        cell = cell_elements[-1]
        #self.browser.switch_to.frame(cell)
        # Récupérer le dernier élément de la liste (la dernière cellule)
        return cell

    def streaming_code(self, completions, cell_index=None):
        # not working !
        '''
        # Select the new cell
        if cell_index is None:
            selected_cell = self.select_cell(len(self.nb))
        else:
            selected_cell = self.select_cell(cell_index)
        '''

        # Send the completion results to the new cell
        for completion in completions:
            selected_cell = self.find_last_cell()
            #self.wait_webbrowser(selected_cell)
            text = completion['choices'][0]['delta']
            if "content" in text:
                mot = text["content"]
                print(mot)
                # Append the completion text to the cell source
                cell_text = selected_cell.text
                print("cell_text :" +cell_text)
                cell_text += mot
                # Send the completion text to the cell
                selected_cell.send_keys(cell_text)
                self.browser.refresh()
        return selected_cell.text

    def add_markdown_cell(self, markdown):
        cell = new_markdown_cell(source=markdown)
        self.nb.cells.append(cell)
        return cell

    def delete_cell(self, index):
        if 0 <= index < len(self.nb.cells):
            del self.nb.cells[index]
        else:
            self._print("Invalid cell index")

    def save_notebook(self):
        with open(self.notebook_name, "w") as f:
            nbf.write(self.nb, f)

    def url(self):
        return f"http://localhost:{self.port_number}/notebooks/{self.notebook_name}"
    def open_notebook(self):
        url = f"http://localhost:{self.port_number}/notebooks/{self.notebook_name}"
        options = Options()
        options.add_argument("--no-sandbox")
        self.browser = webdriver.Chrome(self.chromedriver_path, options=options) # ou autre webdriver selon le navigateur
        self.browser.get(url)
        self.launch_jupyter_notebook_list()

    def save_page_auto(self):
        # Envoyer la combinaison de touches "Ctrl + S" pour enregistrer le notebook
        self.browser.find_element_by_tag_name('body').send_keys(Keys.CONTROL + 's')

    def _open_notebook(self):
        url = f"http://localhost:{self.port_number}/notebooks/{self.notebook_name}"
        webbrowser.open(url)

    def update_notebook(self):
        self.load_notebook()

    def load_notebook(self):
        with open(self.notebook_path, "r") as f:
            self.nb = nbf.read(f, as_version=nbf.NO_CONVERT)
            #print("Loading : " + self.notebook_path)

    def get_code_cells(self):
        # récupère tout les blocs de code présent dans la page jupyter
        self.load_notebook()
        code_cells = []
        if self.nb is not None:
            for cell in self.nb.cells:
                if cell.cell_type == 'code':
                    code_cells.append(cell.source)
        return code_cells

    def refresh_browser(self):
        if self.browser is None:
            return

        self.browser.refresh()
        print("Page refreshed.")
        #self._print("Page refreshed.")


    def find_if_code_exist(self, multiline_string):
        for index in self.nb.cells:
            if self.is_code_cell_equal(index, multiline_string):
                return index
        return False

    def is_code_cell_equal(self, index, multiline_string):
        if 0 <= index < len(self.nb.cells):
            cell = self.nb.cells[index]
            if cell.cell_type == "code":
                return cell.source.strip() == multiline_string.strip()
            else:
                #print("The cell at the given index is not a code cell")
                return False
        else:
            #print("Invalid cell index")
            return False

    def get_chromedriver_path(self):
        os_name = platform.system()
        if os_name == "Windows":
            return "./Utils/chromedriver_win32/chromedriver.exe"
        elif os_name == "Darwin":
            return "./Utils/chromedriver_mac64/chromedriver"
        elif os_name == "Linux":
            return "./Utils/chromedriver_linux64/chromedriver"
        else:
            self._print("Chromedriver, unsupported operating system.")
            raise Exception("Chromedriver, unsupported operating system.")


    def launch_jupyter_notebook_list(self):
        process = subprocess.Popen(
            ["jupyter", "notebook", "list"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        stdout, stderr = process.communicate()

        urls = []
        for line in stdout.splitlines():
            if "http://" in line or "https://" in line:
                urls.append(line)

        for url in urls:
            if url.find("localhost:8888") >= 0:
                self.token = url.split("token=")[-1].split(" ::")[0]
                self._print("Token copied in your clipboard. Just paste into your webbrowser.")

    def copy2clip(self, txt):
        cmd = 'echo ' + txt.strip() + '|clip'
        return subprocess.check_call(cmd, shell=True)

    def _print(self, text, role:str = "system_print"):
        self.print.append({role: text})

    def code_from_selected_cell(self):
        cells = self.browser.find_elements_by_css_selector(".cell")
        for i, cell in enumerate(cells):
            if cell.get_attribute("class").find("selected") >= 0:
                self._print(str(i), "system_print")
                return i
        '''
        # Trouver la cellule sélectionnée
        selected_cell = self.browser.find_element_by_css_selector(".jp-Notebook:focus .jp-InputArea-editor")
        # Obtenir le contenu de la cellule sélectionnée
        content = selected_cell.get_attribute('value')
        '''

        return i

    # Define function to execute code in a Jupyter cell
    def execute_cell(self, cell_text):
        js_code = f"var cell = Jupyter.notebook.get_selected_cell();cell.set_text('{cell_text}');"
        display(Javascript(js_code))
        js_code = "Jupyter.notebook.execute_cell()"
        display(Javascript(js_code))

    def position_screen(self, x, y, width, height):
        self.browser.set_window_position(x, y)
        self.browser.set_window_size(width, height)

class JupyterKernel:
    def __init__(self, notebook_file):
        self.km = KernelManager()
        self.km.start_kernel()

        self.kc = self.km.client()
        self.kc.start_channels()

        self.notebook_file = notebook_file

    def restart_kernel(self):
        self.km.restart_kernel()

    def shutdown_kernel(self):
        self.kc.stop_channels()
        self.km.shutdown_kernel()


if __name__ == '__main__':

    '''
    notebook = JupyterNotebook()
    notebook.add_code_cell("print('Hello, World!')")
    notebook.add_markdown_cell("This is a markdown cell")
    notebook.save_notebook()
    notebook.open_notebook()
    '''
    notebook = JupyterNotebook("conversation.ipynb")
    notebook.load_notebook()
    notebook.open_notebook()
    input()
    notebook.refresh_browser()
    #notebook.add_code_cell("print('Hello, world!')")
    #notebook.add_markdown_cell("# This is a markdown cell")
    #notebook.save_notebook()
    #notebook.open_notebook()

    #kernel = JupyterKernel("conversation.ipynb")
    #kernel.restart_kernel()

    # Assurez-vous de fermer le kernel à la fin de son utilisation
    #time.sleep(5)
    #kernel.shutdown_kernel()
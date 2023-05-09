

class gptj:
    from transformers import GPTJForCausalLM, AutoTokenizer
    import torch

    def __int__(self, chosen_model: str="EleutherAI/gpt-j-6B", path:str = './checkpoints/', local: bool = True,
                 device: str = 'cuda:0'):

        self.chosen_model = chosen_model
        self.model = None
        self.local = local

        if local:
            if not self.gpu_vs_cpu(f'{path}{chosen_model}') and device == 'cuda:0':
                print('Le modèle ne peut-être charger sur le gpu.\nChargement sur le cpu...')
                device = 'cpu'


    def gpu_vs_cpu(self, dossier):
        import pynvml
        import shutil
        # Récupération de la taille totale du dossier (en bytes)
        taille_dossier = shutil.disk_usage(dossier).total
        print("Taille du modèle : "+taille_dossier)
        # Initialisation de pynvml
        pynvml.nvmlInit()

        # Récupération de l'identifiant de la carte graphique (GPU)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # Récupération de la mémoire totale de la carte graphique (en bytes)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        memoire_disponible = memory_info.free
        print('Mémoire disponible dans le GPU : '+memoire_disponible)
        # Fermeture de pynvml
        pynvml.nvmlShutdown()

        # Vérification que la taille du dossier est inférieure ou égale à la mémoire disponible sur la carte graphique
        if taille_dossier <= memoire_disponible:
            return True
        else:
            return False



model = gptj.GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=gptj.torch.float16, low_cpu_mem_usage=True)
tokenizer = gptj.AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
context = """In a shocking finding, scientists discovered a herd of unicorns living in a remote, 
            previously unexplored valley, in the Andes Mountains. Even more surprising to the 
            researchers was the fact that the unicorns spoke perfect English."""

input_ids = tokenizer(context, return_tensors="pt").input_ids
gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
gen_text = tokenizer.batch_decode(gen_tokens)[0]
print(gen_text)


# from transformers import GPTJForCausalLM
# import tensorflow

# model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=tf.float16, low_cpu_mem_usage=True)
# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# context = """In a shocking finding, scientists discovered a herd of unicorns living in a remote,
#           previously unexplored valley, in the Andes Mountains. Even more surprising to the
#           researchers was the fact that the unicorns spoke perfect English."""

# input_ids = tokenizer(context, return_tensors="pt").input_ids
# gen_tokens = model.generate(input_ids, do_sample=True, temperature=0.9, max_length=100,)
# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print(gen_text)
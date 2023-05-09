#@markdown # Load model and tokenizer
import torch

from Utils.Codegen.codegen import set_env, print_time, create_model, create_custom_gpt2_tokenizer, create_tokenizer

chosen_model = "codegen-350M-multi" #@param ["codegen-350M-nl", "codegen-350M-multi", "codegen-350M-mono", "codegen-2B-nl", "codegen-2B-multi", "codegen-2B-mono", "codegen-6B-nl", "codegen-6B-multi", "codegen-6B-mono", "codegen-16B-nl", "codegen-16B-multi", "codegen-16B-mono"]
fp16 = True #@param {type:"boolean"}

import os
import urllib.request
import tarfile

if not os.path.exists(f'./checkpoints/{chosen_model}'):
    # model :
    # codegen-350M-nl
    # codegen-350M-multi
    # codegen-350M-mono
    # same with 2B, 6B, 16B
    # bool : fp16

    url = f"https://storage.googleapis.com/codegen-anonymized/checkpoints/{chosen_model}.tar.gz"
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, f"{chosen_model}.tar.gz")

    # Téléchargement du fichier checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    urllib.request.urlretrieve(url, checkpoint_path)

    # Extraction des fichiers checkpoint
    with tarfile.open(checkpoint_path, "r:gz") as tar:
        tar.extractall(checkpoint_dir)

    #wget -P checkpoints https://storage.googleapis.com/codegen-anonymized/checkpoints/{chosen_model}.tar.gz && tar -xvf checkpoints/{chosen_model}.tar.gz -C checkpoints/

# (0) constants

models_nl = ['codegen-350M-nl', 'codegen-2B-nl', 'codegen-6B-nl', 'codegen-16B-nl']
models_pl = ['codegen-350M-multi', 'codegen-2B-multi', 'codegen-6B-multi', 'codegen-16B-multi', 'codegen-350M-mono', 'codegen-2B-mono', 'codegen-6B-mono', 'codegen-16B-mono']
models = models_nl + models_pl


# (2) preamble

set_env()

pad = 50256
device = torch.device('cuda:0')
ckpt = f'./checkpoints/{chosen_model}'

if device.type == "cpu":
  print()
  print("force full precision for cpu!!")
  print()
  fp16 = False


# (3) load

with print_time('loading parameters'):
  model = create_model(ckpt=ckpt, fp16=fp16).to(device)


with print_time('loading tokenizer'):
  if chosen_model in models_pl:
    tokenizer = create_custom_gpt2_tokenizer()
  else:
    tokenizer = create_tokenizer()
  tokenizer.padding_side = 'left'
  tokenizer.pad_token = pad
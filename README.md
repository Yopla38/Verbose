# Verbose
A GPT helper for jupyter notebook.
Just say what you want !
Paste the token code in your Jupyter page before asking question

Example : "I have an image cross-05.tif and I would like to extract the intensity profile at x=200 and plot it with the image in the background. I also have a file Osc_2.txt with 2 columns. The first column represents time in ms and the second column represents the intensity of RHEED. I would like to plot this data in a separate graph." then click Send

![image](https://github.com/Yopla38/Verbose/assets/70442829/cad5a198-763b-4d6f-bebf-07480e21afd2)


Linux:

Create an environnement conda and clone repository:
```
conda create --name verbose python=3.9
git clone https://github.com/Yopla38/Verbose.git
cd Verbose
```
I use chrome Version 111.0.5563.110. If not the same, download the good chromedriver version on https://chromedriver.chromium.org/downloads and push it on /Utils/chromedriver_linux64/ 


Active environnement:
```
conda activate verbose
```
Install jupyter notebook OR jupyter-lab
```
pip install notebook
OR
pip install jupyterlab
```
Install all dependency:
```
pip install -r requirements.txt
```

OpenAI:
You must have openaiKey to use this program. 
Open openAI_key.txt file and paste your openAIkey string, save it

Run:
```
python GUI_QT.py
```

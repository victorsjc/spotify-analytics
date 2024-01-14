# spotify-analytics

#**Objetivo**
Projeto para iniciantes em data science utilizando o Python e aplicando ferramentas para ML

#**Pré-requisitos**
- Jupyter, no caso utilizei o plugin para VSCode
- [Miniconda] (https://docs.conda.io/projects/miniconda/en/latest/)

#**Configuração**

1. Comece criando o ambiente Anaconda para o projeto de data science. Segue o comando para ser executado no terminal de comando.

```console
conda create -n data-science-env python=3.10 pandas jupyter seaborn scikit-learn keras tensorflow
```

Ao final da execução, deverá ser exibido similar a mensagem abaixo:

```console
# To activate this environment, use                                             
#                                                                               
#     $ conda activate data-science-env                                         
#                                                                               
# To deactivate an active environment, use                                      
#                                                                               
#     $ conda deactivate
```
2. Para remover o ambiente basta executar o seguinte comando:

```console
conda env remove --name data-science-env
```
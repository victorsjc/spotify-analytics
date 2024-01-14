# spotify-analytics

# **Objetivo**
Projeto para iniciantes em data science utilizando o Python e aplicando ferramentas para ML

# **Pré-requisitos**
- Dataset de músicas do Spotify disponível no [Kaggle](https://empresas.alura.com.br/e3t/Ctc/I8+113/d2z6gD04/VVDmqL7Wyty9W9grYBt4-p3wTW1MKd8X585FvGM1--kM3qgyTW95jsWP6lZ3m0W5RbJcS5zBJvJVCQ4pH4csY92W8zCX9f70N1mpW8NJzHt8PwMz0W61L8Rx7CdyHGVXY8PM99nhxcW4RBh3Y36RKSzW4gH-9y8vBCBLW5NnnqC7ZpMHHW4Psm7D4c1fx-W6K7rkn1qt7VmW6NPzHq64DGjdN2tjjm-yHQ1_W9hTrqc3pjWC8W339SjH724Z5CW4vqRfL4LxnVTW5_d3k14XNth1W8tT4lx4hdFJwW1B7Bz16YfH2NW48YWqg7Yd4Y7W7kZ6Mx3nwQBhW7yHzNn3QKdDXVjwM6z5Sd-vvW8bsyP983pdZBW2bzbzY5hckglW8yhvFM59MYFjV2M-Fd8T8tXRW5zB7MG6CLzWwW8SWy-q1GMq81W7wJrt93brg9_f946hlM04)
- Jupyter Notebook, no caso utilizei o plugin para VSCode
- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/)

# **Configuração**

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
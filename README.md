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

# **Coleta de dados e Análise Exploratória**

Para iniciar a análise exploratória, precisamos importar o dataset com os dados do Spotify mencionado na preparação.

No caso, estou utilizando o VSCode como IDE de desenvolvimento com o plugin Jupyter da Microsoft. Sinta-se a vontade em utilizar qualquer ferramenta de desenvolvimento, o importante é ter um ambiente de desenvolvimento adequado as suas condições.

Para a análise exploratória, recomendo fortemente iniciar o uso do Pandas para fazermos uma primeira leitura desse dataset, conforme o código abaixo.

```python
import pandas as pd
df = pd.read_csv("./datasets/dataset.csv")
df.head()
```
Depois de entender melhor as colunas e as informações que são apresentadas, realize uma investigação dos dados e como podemos extrair estatísticas baseado nesses dados. Por exemplo, você consegue ver quais são as músicas mais populares? Quais os artistas mais populares? 

# **Pré-processamento dos Dados**

Iniciar qualquer análise de dados pode exigir um certo grau de pré-processamento dos dados obtidos, muitos casos os dados estarão não-estruturados ou mesmo com dados inconsistentes. Normalmente, temos que garantir uma "triagem" desses dados de forma que exista um padrão de qualidade dos dados para que as próximas etapas sejam eficientes.

Para facilitar no entendimento, podemos antecipar cenários que irão prejudicar nossa análise, neste caso dados que estejam nulos possivelmente poderiam ser descartados. No código abaixo, podemos identificar quantos registros da base estão nessa condição.

```python
# Missing values in each row
missing_values_per_row = df.isnull().sum(axis=1)
count_per_missing_value = missing_values_per_row.value_counts().sort_index()

# Print the results
for missing, rows in count_per_missing_value.items():
    print(f'{rows} row(s) have {missing} missing values')

total_rows_with_missing_values = (df.isnull().any(axis=1)).sum()
print(f'Total number of rows with missing values: {total_rows_with_missing_values}')
```

Continuando nossa análise, podemos extrair dados numéricos de forma que possamos gerar gráficos para agilizar o reconhecimento de padrões e possivelmente responder nossos questionamentos iniciais.

```python
import matplotlib.pyplot as plt
import seaborn as sns

numerical_cols = df[df.columns[(df.dtypes == 'float64') | (df.dtypes == 'int64')]]
numerical_cols.shape
numerical_cols.sample(5)

sns.set_style('darkgrid')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
numerical_cols.hist(figsize=(20,15), bins=30, xlabelsize=8, ylabelsize=8)
plt.tight_layout()
plt.show()
```
Para efeitos de análise, podemos trabalhar com essas mesmas informações categorizando-as sob diferentes aspectos como: 
- ranking dos artistas mais popluares
- ranking dos álbuns mais populares
- ranking das músicas mais populares
- ranking dos gêneros de músicas mais populares

```python
top_n = 10
sns.set_style('darkgrid')
sns.set(rc={"axes.facecolor":"#F2EAC5","figure.facecolor":"#F2EAC5"})
# Get the top N most frequent artists, albums, tracks, and genres
top_artists = df['artists'].value_counts().head(top_n)
top_albums = df['album_name'].value_counts().head(top_n)
top_tracks = df['track_name'].value_counts().head(top_n)
top_genres = df['track_genre'].value_counts().head(top_n)

# Disable FutureWarning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)

    # Plotting
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    # Top N Artists
    sns.barplot(x=top_artists.values, y=top_artists.index, palette="crest", ax=axes[0, 0], orient='h',  zorder=3, width=0.5)
    axes[0, 0].set_title(f'Top {top_n} Artists')
    axes[0, 0].set_xlabel('Frequency')
    axes[0, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    # Top N Albums
    sns.barplot(x=top_albums.values, y=top_albums.index, palette="crest", ax=axes[0, 1], orient='h', zorder=3, width=0.5)
    axes[0, 1].set_title(f'Top {top_n} Albums')
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    # Top N Tracks
    sns.barplot(x=top_tracks.values, y=top_tracks.index, palette="crest", ax=axes[1, 0], orient='h', zorder=3, width=0.5)
    axes[1, 0].set_title(f'Top {top_n} Tracks')
    axes[1, 0].set_xlabel('Frequency')
    axes[1, 0].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    # Top N Genres
    sns.barplot(x=top_genres.values, y=top_genres.index, palette="crest", ax=axes[1, 1], orient='h', zorder=3, width=0.5)
    axes[1, 1].set_title(f'Top {top_n} Genres')
    axes[1, 1].set_xlabel('Frequency')
    axes[1, 1].xaxis.grid(linestyle='-', linewidth=0.5, alpha=1, zorder=0)

    plt.tight_layout()
    plt.show()
```
![Ranking Popularidade Spotify](/assets/img/ranking-popularidade.png "Ranking de Popularidade Spotify") {align=center}
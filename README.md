# Transfer Learning: Classificacao de Gatos e Cachorros

Projeto desenvolvido para o desafio da DIO sobre **Transfer Learning com Deep Learning**, utilizando Python, TensorFlow e Keras no Google Colab.

A proposta e construir um classificador de imagens capaz de diferenciar gatos e cachorros a partir do dataset `cats_vs_dogs`, reaproveitando uma rede neural ja treinada em um grande conjunto de imagens. Em vez de treinar uma CNN profunda do zero, o projeto usa a MobileNetV2 como extratora de caracteristicas e adiciona uma pequena cabeca de classificacao para adaptar o modelo ao problema.

## Objetivo

Criar um pipeline completo, organizado e didatico para classificacao binaria de imagens, cobrindo desde o carregamento do dataset ate a avaliacao final do modelo.

O projeto foi pensado para ser executado no Google Colab, mas tambem pode ser adaptado para um ambiente local com Python configurado corretamente.

## Tecnologias utilizadas

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Google Colab

## Conceitos aplicados

- Transfer Learning
- Deep Learning
- Redes neurais convolucionais
- MobileNetV2
- Classificacao binaria de imagens
- Pre-processamento de imagens
- Normalizacao
- Resize de imagens
- Data augmentation
- Congelamento de camadas
- Treinamento supervisionado
- Avaliacao com metricas de acuracia e loss

## O que e Transfer Learning?

Transfer Learning e uma tecnica em que aproveitamos o conhecimento aprendido por um modelo em uma tarefa anterior e o reutilizamos em uma nova tarefa relacionada.

No contexto de imagens, redes como MobileNetV2, ResNet e EfficientNet ja foram treinadas em bases enormes, como a ImageNet, e aprenderam a identificar padroes visuais genericos: bordas, texturas, formas, partes de objetos e composicoes mais complexas.

Para um problema como gatos versus cachorros, esse conhecimento e muito util. O modelo base ja entende varios elementos visuais importantes, entao precisamos treinar apenas uma parte menor da arquitetura para especializar a rede no nosso conjunto de dados.

Essa abordagem costuma trazer tres vantagens importantes:

- reduz o tempo de treinamento;
- exige menos dados do que treinar uma rede profunda do zero;
- geralmente melhora a qualidade do modelo em projetos de classificacao de imagens.

## Por que MobileNetV2?

A MobileNetV2 e uma arquitetura de rede neural convolucional criada para ser leve, eficiente e adequada para cenarios com restricao de recursos.

Ela foi escolhida neste projeto porque oferece um bom equilibrio entre desempenho e custo computacional. Isso faz bastante sentido para um notebook educacional em Google Colab, onde queremos um modelo forte, mas sem tornar o treinamento pesado demais.

Neste projeto, a MobileNetV2 e utilizada com pesos pre-treinados na ImageNet. A parte convolucional da rede fica congelada durante o treinamento inicial, funcionando como extratora de caracteristicas. Em seguida, uma cabeca de classificacao e adicionada para prever se a imagem representa um gato ou um cachorro.

## Dataset

O projeto utiliza o dataset `cats_vs_dogs`, disponivel pelo `tensorflow_datasets`.

Esse conjunto contem imagens reais de gatos e cachorros em diferentes poses, iluminacoes, tamanhos e contextos. Por isso, ele e um bom exemplo para praticar classificacao de imagens com redes neurais convolucionais.

Durante o notebook, o dataset e dividido em:

- treino;
- validacao;
- teste.

As imagens passam por resize, normalizacao e data augmentation antes de serem usadas no treinamento.

## Estrutura do projeto

```text
transfer-learning-cats-vs-dogs-dio/
├── README.md
├── requirements.txt
├── transfer_learning_cats_vs_dogs.ipynb
├── .gitignore
└── images/
    └── .gitkeep
```

## Etapas do desenvolvimento

1. Preparacao do ambiente no Google Colab.
2. Importacao das bibliotecas necessarias.
3. Carregamento do dataset `cats_vs_dogs` com TensorFlow Datasets.
4. Divisao dos dados em treino, validacao e teste.
5. Pre-processamento das imagens.
6. Aplicacao de resize e normalizacao.
7. Criacao de uma camada de data augmentation.
8. Visualizacao de amostras do dataset.
9. Carregamento da MobileNetV2 com pesos da ImageNet.
10. Congelamento do modelo base.
11. Criacao da cabeca de classificacao.
12. Compilacao do modelo.
13. Treinamento.
14. Avaliacao final.
15. Visualizacao dos graficos de acuracia e loss.
16. Predicao em imagens de exemplo.

## Resultados obtidos

Os resultados podem variar conforme o ambiente de execucao, a GPU disponivel e a quantidade de epocas usada no treinamento.

Mesmo com poucas epocas, o uso de Transfer Learning costuma gerar uma acuracia de validacao bastante competitiva para o problema de gatos e cachorros. Isso acontece porque a MobileNetV2 ja aprendeu representacoes visuais poderosas durante o treinamento na ImageNet.

O notebook inclui graficos de acuracia e loss para acompanhar a evolucao do treinamento e identificar sinais de overfitting ou underfitting.

## Como executar

1. Abra o arquivo `transfer_learning_cats_vs_dogs.ipynb` no Google Colab.
2. Selecione um ambiente com GPU em `Ambiente de execucao > Alterar tipo de ambiente de execucao`.
3. Execute as celulas em ordem.
4. Acompanhe os graficos e a avaliacao final do modelo.

Para executar localmente, instale as dependencias:

```bash
pip install -r requirements.txt
```

Depois, abra o notebook em um ambiente Jupyter.

## Conclusao

Este projeto mostra como aplicar Transfer Learning de forma pratica e organizada em um problema classico de classificacao de imagens.

A combinacao de TensorFlow, Keras, TensorFlow Datasets e MobileNetV2 permite construir um modelo eficiente sem precisar treinar uma arquitetura profunda do zero. Alem disso, o notebook foi estruturado para facilitar o entendimento de cada etapa, servindo tanto como entrega para o desafio quanto como material de portfolio.

## Melhorias futuras

- Aplicar fine-tuning nas camadas finais da MobileNetV2.
- Comparar a MobileNetV2 com outras arquiteturas, como EfficientNet ou ResNet.
- Adicionar matriz de confusao e relatorio de classificacao.
- Salvar o modelo treinado em formato `.keras` ou `SavedModel`.
- Criar uma interface simples com Streamlit ou Gradio.
- Testar imagens externas enviadas pelo usuario.
- Automatizar experimentos com diferentes taxas de aprendizado e tamanhos de batch.

## Sugestoes de commits semanticos

```bash
git add .
git commit -m "chore: create project structure"

git add README.md
git commit -m "docs: add project documentation"

git add transfer_learning_cats_vs_dogs.ipynb
git commit -m "feat: add transfer learning notebook"

git add requirements.txt .gitignore images/.gitkeep
git commit -m "chore: add dependencies and repository config"
```

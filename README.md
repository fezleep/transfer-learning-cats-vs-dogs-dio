# Transfer learning com gatos e cachorros

Projeto de classificação de imagens desenvolvido para o desafio da DIO, usando **transfer learning** com MobileNetV2, TensorFlow e o dataset `cats_vs_dogs`.

A ideia aqui foi construir um notebook simples de acompanhar, mas completo o suficiente para mostrar o fluxo real de um projeto de visão computacional: carregar os dados, preparar as imagens, reaproveitar um modelo pré-treinado, treinar a camada final e avaliar os resultados.

## objetivo

Classificar imagens de gatos e cachorros usando uma rede neural pré-treinada.

Em vez de treinar uma CNN do zero, o projeto usa a MobileNetV2 como base para extração de características. A partir dela, foi adicionada uma cabeça de classificação para adaptar o modelo ao problema binário.

O notebook foi montado para rodar bem no Google Colab, mas também pode ser executado localmente com Python e Jupyter.

## tecnologias

- Python
- TensorFlow
- Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Jupyter Notebook
- Google Colab

## transfer learning

Transfer learning é uma forma de reaproveitar o que um modelo já aprendeu em uma tarefa maior e aplicar esse conhecimento em outro problema.

Neste projeto, a MobileNetV2 entra com pesos pré-treinados na ImageNet. Isso significa que ela já aprendeu padrões visuais úteis, como bordas, texturas, formas e combinações mais complexas de objetos.

Para o problema de gatos e cachorros, esse reaproveitamento faz sentido porque o modelo não começa do zero. Ele já sabe extrair boas características das imagens, e o treinamento fica concentrado em ajustar a parte final para separar as duas classes.

## dataset

O dataset usado foi o `cats_vs_dogs`, disponível pelo `tensorflow_datasets`.

Ele contém imagens reais de gatos e cachorros em diferentes poses, tamanhos, iluminações e cenários. Essa variação torna o conjunto interessante para testar um modelo de classificação de imagens de forma um pouco mais próxima de um caso real.

No notebook, os dados são divididos em treino, validação e teste. Antes do treinamento, as imagens passam por redimensionamento, normalização e data augmentation.

## estrutura

```text
transfer-learning-cats-vs-dogs-dio/
├── README.md
├── requirements.txt
├── transfer_learning_cats_vs_dogs.ipynb
├── .gitignore
└── images/
    ├── accuracy_loss.png
    ├── final_accuracy.png
    └── predictions.png
```

## etapas

1. Preparação do ambiente no Google Colab.
2. Importação das bibliotecas.
3. Carregamento do `cats_vs_dogs` com TensorFlow Datasets.
4. Separação dos dados em treino, validação e teste.
5. Redimensionamento e normalização das imagens.
6. Aplicação de data augmentation.
7. Visualização de amostras do dataset.
8. Carregamento da MobileNetV2 com pesos da ImageNet.
9. Congelamento da base convolucional.
10. Criação da cabeça de classificação.
11. Compilação e treinamento do modelo.
12. Avaliação no conjunto de teste.
13. Visualização das curvas de acurácia e loss.
14. Teste com predições em imagens de exemplo.

## resultados

O modelo teve um bom desempenho para a tarefa proposta. A MobileNetV2 conseguiu extrair características visuais fortes, e a camada final aprendeu a separar as classes com boa precisão.

Resultados aproximados no conjunto de teste:

- acurácia: **98,15%**
- loss: **0,0476**

### acurácia e loss

![Gráfico de acurácia e loss](images/accuracy_loss.png)

### acurácia final

![Acurácia final do modelo](images/final_accuracy.png)

### exemplos de predição

![Predições do modelo](images/predictions.png)

## como executar

Abra o notebook `transfer_learning_cats_vs_dogs.ipynb` no Google Colab e execute as células em ordem.

Para melhor desempenho, use GPU:

```text
Ambiente de execução > Alterar tipo de ambiente de execução > GPU
```

Para rodar localmente, instale as dependências:

```bash
pip install -r requirements.txt
```

Depois disso, abra o notebook em um ambiente Jupyter.

## conclusão

Este projeto mostra como transfer learning pode acelerar bastante o desenvolvimento de um classificador de imagens.

Com a MobileNetV2 pré-treinada, foi possível chegar a uma boa acurácia sem precisar treinar uma rede profunda do zero. O resultado é um notebook direto, reproduzível e útil tanto como entrega para a DIO quanto como projeto de portfólio.

## melhorias futuras

- Fazer fine-tuning nas últimas camadas da MobileNetV2.
- Comparar o resultado com arquiteturas como EfficientNet e ResNet.
- Adicionar matriz de confusão.
- Incluir relatório de classificação com precision, recall e F1-score.
- Salvar o modelo treinado em `.keras` ou `SavedModel`.
- Criar uma interface simples com Streamlit ou Gradio.
- Testar imagens externas enviadas pelo usuário.
- Automatizar experimentos com diferentes taxas de aprendizado e tamanhos de batch.

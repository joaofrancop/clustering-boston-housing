# Clusterização de Imóveis - Boston Housing Dataset 🏠

Este repositório contém a solução do exercício prático de Modelos Não Supervisionados (K-Means) desenvolvido para a disciplina de Sistemas Inteligentes Avançados.

## 🎯 Objetivo do Projeto
O objetivo é treinar um modelo de clusterização capaz de agrupar imóveis da região de Boston com base em suas características (como taxa de criminalidade, número de quartos, impostos, poluição, etc.), criar perfis descritivos para esses grupos e realizar a inferência de novos imóveis.

## 🏗️ Estrutura do Código
O projeto foi desenvolvido de forma modular e orientada a objetos:
* `modelo_clustering.py`: Contém a classe `HousingClustering`, responsável por todo o pipeline de Machine Learning (pré-processamento, hiperparametrização para encontrar o K ótimo, treinamento e inferência).
* `main.py`: O script de execução que instancia o modelo, treina os dados, exibe o perfil dos segmentos e testa a inferência com um imóvel fictício.
* `data/HousingData.csv`: O dataset original utilizado para o treinamento.

## 🚀 Como Executar

1. Certifique-se de ter o Python instalado, além das bibliotecas `pandas`, `numpy`, `scikit-learn` e `scipy`.
2. Clone este repositório ou baixe os arquivos.
3. No terminal, navegue até a pasta do projeto e execute:
   ```bash
   python main.py

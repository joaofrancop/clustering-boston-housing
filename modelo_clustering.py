import pandas as pd
import numpy as np
import math
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist

class HousingClustering:
    def __init__(self):
        # Nomes dos arquivos que serão salvos pelo pickle
        self.scaler_file = 'normalizador_boston.pkl'
        self.model_file = 'cluster_boston.pkl'

    def train(self, filepath):
        """Módulo de Treinamento baseado no cluster_iris.py"""
        print("1. Carregando e limpando os dados...")
        dados = pd.read_csv(filepath)
        
        # Preenche valores nulos com a mediana para não quebrar o algoritmo
        dados = dados.fillna(dados.median())

        print("2. Normalizando os dados (MinMaxScaler)...")
        scaler = MinMaxScaler()
        normalizador = scaler.fit(dados)
        
        # Salva o normalizador para uso posterior
        pickle.dump(normalizador, open(self.scaler_file, 'wb'))
        
        # Normaliza os dados numéricos
        dados_norm = normalizador.transform(dados)
        dados_norm_df = pd.DataFrame(dados_norm, columns=dados.columns)

        print("3. Hiperparametrizando (Descobrindo K ótimo)...")
        distortions = []
        K = range(1, 15) # Limitado a 15 para processamento rápido
        
        for i in K:
            cluster_model = KMeans(n_clusters=i, random_state=42, n_init='auto').fit(dados_norm_df)
            distortions.append(
                sum(np.min(cdist(dados_norm_df, cluster_model.cluster_centers_, 'euclidean'), axis=1)) / dados_norm_df.shape[0]
            )

        # Determinar o número ótimo de clusters usando a matemática da aula
        x0, y0 = K[0], distortions[0]
        xn, yn = K[-1], distortions[-1]
        distances = []
        
        for i in range(len(distortions)):
            x, y = K[i], distortions[i]
            numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
            denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
            distances.append(numerador/denominador)

        numero_clusters_otimo = K[distances.index(np.max(distances))]
        print(f"-> Número ótimo de clusters encontrado: {numero_clusters_otimo}")

        print("4. Treinando o modelo final e salvando...")
        modelo_final = KMeans(n_clusters=numero_clusters_otimo, random_state=42, n_init='auto').fit(dados_norm_df)
        pickle.dump(modelo_final, open(self.model_file, 'wb'))
        
        # Salvamos também as features para usar na inferência depois
        pickle.dump(dados.columns.tolist(), open('features_boston.pkl', 'wb'))
        print("Treinamento concluído com sucesso!\n")

    def describe_segments(self):
        """Módulo de Descrição baseado no descritor_cluster.py"""
        print("=== Módulo de Descrição dos Segmentos ===")
        
        # Abrir os arquivos salvos
        cluster_model = pickle.load(open(self.model_file, 'rb'))
        normalizador = pickle.load(open(self.scaler_file, 'rb'))
        features = pickle.load(open('features_boston.pkl', 'rb'))

        # Converter os centroides em dataframe
        df_centroides_norm = pd.DataFrame(cluster_model.cluster_centers_, columns=features)

        # Desnormalizar (voltar para a escala real em Dólares, Taxas, etc)
        centroides_reais = normalizador.inverse_transform(df_centroides_norm)
        
        # Formatando para exibição elegante
        df_descritivo = pd.DataFrame(centroides_reais, columns=features)
        df_descritivo.index.name = 'Cluster'
        
        # Exibe a tabela transposta (como no exemplo anterior)
        print(df_descritivo.T.round(2))
        print("=========================================\n")

    def infer(self, new_data_dict):
        """Módulo de Inferência baseado no iris_inferencia.py"""
        # Abrir normalizador, modelo e colunas
        normalizador = pickle.load(open(self.scaler_file, 'rb'))
        cluster_model = pickle.load(open(self.model_file, 'rb'))
        features = pickle.load(open('features_boston.pkl', 'rb'))

        # Organiza o dado novo no mesmo formato que o modelo espera
        nova_casa = pd.DataFrame([new_data_dict])
        
        # Garante que a ordem das colunas seja igual ao treino
        nova_casa = nova_casa[features] 

        # Normaliza a nova instância (isso retorna um array sem nomes)
        nova_casa_norm_array = normalizador.transform(nova_casa)

        # CONSERTO: Transforma de volta em DataFrame com os nomes das colunas
        nova_casa_norm = pd.DataFrame(nova_casa_norm_array, columns=features)

        # Faz a predição (agora sem reclamar!)
        cluster_previsto = cluster_model.predict(nova_casa_norm)[0]
        
        return cluster_previsto
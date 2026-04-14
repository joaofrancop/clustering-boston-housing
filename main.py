# Importa a classe do outro arquivo
from modelo_clustering import HousingClustering

def main():
    # Caminho exato da sua máquina
    caminho_dados = r"G:\Outros computadores\JGFP NOTEBOOK\Documents\Faculdade\sistemas inteligentes avançados\Atividade Boston Dataset\data\HousingData.csv"
    
    # 1. Instancia o modelo
    modelo = HousingClustering()

    # 2. Executa o Treinamento (que cria os arquivos .pkl)
    modelo.train(caminho_dados)

    # 3. Executa a Descrição dos clusters encontrados
    modelo.describe_segments()

    # 4. Executa a Inferência com dados de um imóvel qualquer
    # (Usando as 14 colunas padrão do Dataset)
    imovel_desconhecido = {
        'CRIM': 0.05,    'ZN': 18.0,    'INDUS': 2.31,   'CHAS': 0.0,
        'NOX': 0.53,     'RM': 6.57,    'AGE': 65.2,     'DIS': 4.09,
        'RAD': 1.0,      'TAX': 296.0,  'PTRATIO': 15.3, 'B': 396.9,
        'LSTAT': 4.98,   'MEDV': 24.0
    }

    cluster = modelo.infer(imovel_desconhecido)
    
    print("=== Módulo de Inferência ===")
    print(f"Os dados informados foram classificados no Cluster: {cluster}")

if __name__ == "__main__":
    main()
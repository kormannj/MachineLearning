import pandas as pd

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes):
    modelo.fit(treino_dados, treino_marcacoes)

    resultado = modelo.predict(teste_dados)
    # diferencas = resultado - teste_marcacoes
    # acertos = [d for d in diferencas if d == 0]

    acertos = (resultado == teste_marcacoes)
    total_de_acertos = sum(acertos)  # len(acertos)
    total_de_elementos = len(teste_dados)
    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    print("A taxa de acerto do algoritmo {0} é {1}".format(nome, taxa_de_acerto))

#O Pandas gera um conjunto de dados mais elaborado, chamado normalmente de data_frame (df)
df = pd.read_csv(r"C:\Users\Usuario\Desktop\Machine_Learning\buscas.csv")

#Quando uma variável for data_frame, o ideal é identificar com df, para não confundir com array.
X_df = df[['home', 'busca', 'logado']] #Quando estou pedindo mais de um item, tenho que passar como um array, com []
Y_df = df['comprou']

Xdummies_df = pd.get_dummies(X_df) #Dummy é o nome que se dá quando se transforma um dado em outro. Nesse exercício
#pegamos uma coluna que continha valores caracter (pesquisas feitas em um site) e convertendo para binário.
Ydummies_df = Y_df

#É necessário converter os dataframes em arrays para passar para o MultinomialNB
X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.9
porcentagem_teste = 0.1

tamanho_de_treino = int(porcentagem_treino * len(Y))
tamanho_de_teste = int(porcentagem_teste * len(Y))
tamanho_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste

treino_dados = X[0:tamanho_de_treino] #Deixando explícitas as posições de início e fim, pois pegaremos três faixas
treino_marcacoes = Y[0:tamanho_de_treino]

fim_de_teste = tamanho_de_teste + tamanho_de_treino
teste_dados = X[tamanho_de_treino:fim_de_teste]
teste_marcacoes = Y[tamanho_de_treino:fim_de_teste]

validacao_dados = X[fim_de_teste:]
validacao_marcacoes = Y[fim_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
fit_and_predict("MultinomialNB", modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
fit_and_predict("AdaBoostClassifier", modelo, treino_dados, treino_marcacoes, teste_dados, teste_marcacoes)

print("Total de elementos de teste:", len(teste_dados))

#eficacia do algoritmo que chuta um único valor
#acerto_de_um = len(Y[Y=="sim"]) #forma de fazer com array
#acerto_de_um = list(Y).count("sim") # forma de fazer com lista
#acerto_de_zero = len(Y[Y=="nao"])
#acerto_de_zero = list(Y).count("nao")
#taxa_de_acerto_melhor = 100.0 * max(acerto_de_um, acerto_de_zero) / len(Y)

#o código acima pode ser substituído pelo uso da biblioteca abaixo
from collections import Counter
acerto_base = max(Counter(teste_marcacoes).values())
taxa_de_acerto_melhor = 100.0 * acerto_base / len(teste_marcacoes)

print("Taxa de acerto base: %f" % taxa_de_acerto_melhor)


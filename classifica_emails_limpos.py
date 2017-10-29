import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import cross_val_score

def vetorizar_texto(texto, tradutor):
    vetor = [0] * len(tradutor) #Criei um vetor com várias posições, todas iniciadas com 0

    for palavra in texto:
        if palavra in tradutor:
            posicao = tradutor[palavra]
            vetor[posicao] += 1

    return vetor

def fit_and_predict(nome, modelo, treino_dados, treino_marcacoes):
    k = 10 #Quantidade de 'folds' (dobras/quebras) nos dados para fazer o treino
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k) #Essa classe faz os cálculos dos 'kfolds'
    taxa_de_acerto = np.mean(scores) #Mean devolve a média desses resultados que é usada a partir de então
    msg = "Taxa de acerto do {0}: {1}".format(nome, taxa_de_acerto)
    print(msg)
    return taxa_de_acerto

def teste_real(modelo, validacao_dados, validacao_marcacoes):
    resultado = modelo.predict(validacao_dados)

    acertos = resultado == validacao_marcacoes

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto do vencedor entre os dois algoritmos no mundo real: {0}".format(taxa_de_acerto)
    print(msg)

conteudo_arquivo = pd.read_csv(r"C:\Users\Usuario\Desktop\Machine_Learning\entrada_emails.csv")
textos_emails = conteudo_arquivo['email'] #Função do Pandas que pega o dataframe 'email'
textos_quebrados = textos_emails.str.lower().str.split(' ') #Função do Pandas que, a cada ' ' da string, faz uma quebra com ','. Também usamos o lower para transformar tudo para minúsculo, evitando duplicidade de palavras.

dicionario = set() #Dicionário é um set, ou seja, um conjunto que não permite itens repetidos

for lista in textos_quebrados: #Percorro o array completo
    dicionario.update(lista)

numero_de_palavras = len(dicionario)
tuplas = zip(dicionario, range(numero_de_palavras)) #Estou associando cada palavra ao número de sua posição na lista
tradutor = {palavra:indice for palavra, indice in tuplas} #Criei um dicionário (representado pelos {}). Com isso consigo pesquisar por uma string e saber em qual posição está.

vetores_de_texto = [vetorizar_texto(texto, tradutor) for texto in textos_quebrados] #Estou lendo todos os e-mails e transformando em vetores, com as quantidades de palavras contadas.
marcas = conteudo_arquivo['classificacao'] #Função do Pandas que pega o dataframe 'classificacao'

X = vetores_de_texto
Y = marcas

porcentagem_de_treino = 0.8

tamanho_do_treino = int(len(Y) * porcentagem_de_treino)
tamanho_da_validacao = len(Y) - tamanho_do_treino

treino_dados = X[:tamanho_do_treino]
treino_marcacoes = Y[:tamanho_do_treino]

validacao_dados = X[tamanho_do_treino:]
validacao_marcacoes = Y[tamanho_do_treino:]

resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state = 0))
resultadoOneVsRest = fit_and_predict("OneVsRest", modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state = 0))
resultadoOneVsOne = fit_and_predict("OneVsOne", modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne

from sklearn.naive_bayes import MultinomialNB
modeloMultinomial = MultinomialNB()
resultadoMultinomial = fit_and_predict("MultinomialNB", modeloMultinomial, treino_dados, treino_marcacoes)
resultados[resultadoMultinomial] = modeloMultinomial

from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoost = AdaBoostClassifier()
resultadoAdaBoost = fit_and_predict("AdaBoostClassifier", modeloAdaBoost, treino_dados, treino_marcacoes)
resultados[resultadoAdaBoost] = modeloAdaBoost

maximo = max(resultados)
vencedor = resultados[maximo]

print("Vencedor: ")
print(vencedor)

vencedor.fit(treino_dados, treino_marcacoes)
teste_real(vencedor, validacao_dados, validacao_marcacoes)

acerto_base = max(Counter(validacao_marcacoes).values())
taxa_de_acerto_base = 100.0 * acerto_base / len(validacao_marcacoes)
print("Taxa de acerto base: %f" % taxa_de_acerto_base)

total_de_elementos = len(validacao_dados)
print("Total de teste: %d" % total_de_elementos)
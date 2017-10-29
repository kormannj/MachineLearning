from dados import carregar_acessos

X, Y = carregar_acessos()

dados_de_treino = X[:90]
marcacoes_de_treino = Y[:90]
dados_de_teste = X[-10:]
marcacoes_de_teste = Y[-10:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(dados_de_treino,marcacoes_de_treino)

resultado = modelo.predict(dados_de_teste)
diferencas = resultado - marcacoes_de_teste
acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(dados_de_teste)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print(taxa_de_acerto)
print(total_de_elementos)
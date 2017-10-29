#é gordinho, tem perninha curta, faz auau?
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]

marcacoes = ["porco", "porco", "porco", "cachorro", "cachorro", "cachorro"]
misterioso = [1, 1, 1]

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()
modelo.fit(dados, marcacoes)
print(modelo.predict(misterioso))
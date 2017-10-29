import csv

def carregar_acessos():
    X = []
    Y = []

    #arquivo = open(r"C:\Users\jean.kormann\Desktop\acesso.csv")
    arquivo = open("C:/Users/Usuario/Desktop/Machine_Learning/acesso.csv")
    leitor = csv.reader(arquivo)
    next(leitor)

    for home, como_funciona, contato, comprou in leitor:

        dado = [int(home), int(como_funciona), int(contato)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y

def carregar_buscas():
    X = []
    Y = []

    arquivo = open("C:/Users/Usuario/Desktop/Machine_Learning/buscas.csv")
    leitor = csv.reader(arquivo)
    next(leitor)

    for home,busca,logado,comprou in leitor:
        dado = [int(home),busca,int(logado)]
        X.append(dado)
        Y.append(int(comprou))

    return X, Y

if(__name__ == "__main__"):
    carregar_acessos()
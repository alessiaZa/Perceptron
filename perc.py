#importo le librerie necessarie
import random
import math

class Perceptron:
    def __init__(self, n, lRate=0.01):
        self._learningRate = lRate       #inizializzo la learning rate
        self._bias = 0.0                 #inizializzo il bias a 0
        self._n_inputs = n               #numero di input del perceptron

        #inizializzo i pesi (weights) con muneri random compresi tra 0.0 e 1.0 (escluso)
        self._weights = [random.random() for _ in range(self._n_inputs)]

    #getters and setters

    @property
    def learningRate(self):
        return self._learningRate
    
    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias
    
    @learningRate.setter
    def learningRate(self,rate):
        if rate > 0:
            self._learningRate = rate
        else:
            raise ValueError("La learnimg rate deve essere maggiore di 0")
        

    #implementazione della funzione di costo (loss)
    #utilizzeremo la MSE (mean squared error)
    #questo metodo restituisce:
    #il quadrato della differenza tra il valore atteso e il valore predetto   
    def mse(self, y, y_p):
        return (y - y_p)**2
    
    #derivata della funzione di costo MSE
    #utilizzata per eseguire l'aggiornamento dei pesi durante il gradient descent
    def d_mse(self, y, y_p):
        return -2 * (y - y_p)
        
    #funzione sigmoid
    #f(z) = 1 / (1 + exp(-z))
    #output range (0.0, 1.0)
    #utilizzata per calcolare a
    def sigmoid(self, z):
        return (1.0 / (1.0 + math.exp(-z)))
    

    #derivata della funzione sigmoid
    #σ'(z) = σ(z) * (1 - σ(z)) ma sappiamo che σ(z) = a
    def d_sigmoid(self, a):
        return a * (1 - a)


    #funzione di trasferimento (transfer function)
    #restituisce x0*w0 + x1*w1 + ... xn*wn + b
    def z(self, inputs):
        f_z = 0.0

        for i in range(self._n_inputs):
            f_z = f_z + inputs[i] * self.weights[i]

        f_z = f_z + self.bias
        return f_z
    
    #funzione di attivazione 
    #applica la funzione di attivazione su z
    #a = f(z)
    def a(self, z):
        return self.sigmoid(z)
    

    #restituisce la classe associata alla predizione
    def get_class_id(self, y_p):
        return 0 if y_p < 0.5 else 1
    

    #nel seguito vengono implementate le API (Application Programming Interface) disponibili all'utente per:
    #effettuare il training della rete utilizzando il dataset in input 
    #effettuare le previsioni sui nuovi dati (mai visti dalla rete)

    #funzione di training
    #la funzione prende in input:
    #x_train, la matrice di input che continene i dati di training (dataset)
    #formata da tante righe quante sono i dati di training e da tante colonne quante sono i valori di input (n_inputs)
    #labels, vettore che contiene per ogni riga di training il risultato atteso
    #epochs, indica il numero di volte che dovremo far vedere tutti i dati alla rete
    def train(self, x_train, labels, epochs):
        #la lista conterrà la media dei loss per singola epoca
        loss_per_epoch = []

        #per il numero di epoche passato dall'utente
        for e in range(epochs):
            num_of_errors = 0       #numero totale di previsioni errate
            total_loss = 0          #somma di tutte le loss rilevate nell' epoca

            #scorro tutta la matrice di training
            for j in range(len(x_train)):
                #forward pass
                z = self.z(x_train[j])      #applico la funzione di trasferimento ai dati di input
                a = y_p = self.a(z)         #applico la funzione di attivazione a z


                #calcolo la perdita corrente usando la funzione costo
                loss = self.mse(labels[j], y_p)

                #calcolo il delta
                delta = self.d_mse(labels[j], y_p) * self.d_sigmoid(a)
                #gradient descent e backpropagation: calcolo i nuovi pesi e il nuovo bias
                for i in range(self._n_inputs):
                    delta_w = delta * x_train[j][i]
                    self._weights[i] -=  (delta_w * self.learningRate)    #aggiorno i pesi

                delta_b = delta
                self._bias -= (delta_b * self.learningRate)               #aggiorno il bias 

                #verifichiamo se la rete ha classificato correttamente
                if self.get_class_id(y_p) != labels[j]:
                    num_of_errors += 1

                total_loss += loss

            avg_loss = total_loss / len(x_train)    #calcolo la perdita media per l'epoch corrente
            loss_per_epoch.append(avg_loss)
            print(f"epoch {e+1}, errori totali: {num_of_errors} perdita media: {round(avg_loss, 4)}")

        return num_of_errors, loss_per_epoch


    #predice un risultato a partire dai dati in input 
    def predict(self, x_test):
        predictions = []        #conterrà i valori predetti
        for _input in x_test:
            y_p = self.a(self.z(_input))                #applico in seguenza z e poi a e ottengo la predizione
            predictions.append(self.get_class_id(y_p))  #aggiungo la predizione alla lista (predictions)

        return predictions      #restituisco la lista dei valori predetti dalla rete
    
    #fine della classe Perceptron 


if __name__ == "__main__":
    p = Perceptron(4, 0.1)

    print(p._learningRate)
    print(p._weights)
    print(p._bias)




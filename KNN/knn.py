import qiskit_quantum_knn
from qiskit_quantum_knn.qknn import QKNeighborsClassifier
from qiskit_quantum_knn.encoding import analog
from qiskit import aqua
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import qiskit as qk
def predict(custom=[]):
    # initialising the quantum instance
    

    backend = qk.BasicAer.get_backend('qasm_simulator')
    instance = aqua.QuantumInstance(backend, shots=10000,seed_simulator=8923)

    # initialising the qknn model
    qknn = QKNeighborsClassifier(
    n_neighbors=3,
    quantum_instance=instance
    )

    n_variables = 4        # should be positive power of 2
    n_train_points =32     # can be any positive integer
    n_test_points = 2    # can be any positive integer


    iris = datasets.load_iris()
    labels = iris.target
    data_raw = iris.data
    X_train, X_test,y_train, y_test = train_test_split(data_raw,labels ,
                                    test_size=0.5, 
                                    shuffle=True,random_state=7)
    print(X_train)
    print(y_train)
    print("over")

    ##encoded_data = analog.encode(data_raw[:, :n_variables])
    enc=analog.encode(X_train)
    encTest=analog.encode(X_test)


    #train_data = encoded_data[:n_train_points]
    train_data = enc[:n_train_points]
    train_labels = y_train[:n_train_points]



    #test_data = encoded_data[n_train_points:(n_train_points+n_test_points), :n_variables]
    #test_labels = labels[n_train_points:(n_train_points+n_test_points)]


    test_data = encTest[n_train_points:(n_train_points+n_test_points), :n_variables]
    test_labels = y_test[n_train_points:(n_train_points+n_test_points)]

    print(train_data)
    print(train_labels)
    label_mapping = {
        0: "Setosa",
        1: "Versicolor",
        2: "Virginica"
        }
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_data,train_labels)
   
    qknn.fit(train_data,train_labels)
    if not bool(custom):
        qknn_prediction = qknn.predict(test_data)
      
        print(qknn_prediction)
        print(test_labels)

        qknn_prediction_text = [label_mapping[i] for i in qknn_prediction]
        test_labels_text = [label_mapping[i] for i in test_labels]
        print(qknn_prediction_text)
        print(test_labels_text)
        
        classicalPred=neigh.predict(test_data)
        classicalPred_text=[label_mapping[i] for i in classicalPred]
        return qknn_prediction_text,test_labels_text,classicalPred_text
    else:
        test_data[1]=(analog.encode(custom))[0]
        qknn_prediction = qknn.predict(test_data)
        qknn_prediction_text = [label_mapping[i] for i in qknn_prediction]
        classicalPred=neigh.predict(test_data)
        classicalPred_text=[label_mapping[i] for i in classicalPred]
        
        return qknn_prediction_text[1:],classicalPred_text[1:]
        


    

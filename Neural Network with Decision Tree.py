# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 08:57:01 2018

@author: tamnich
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 10:00:00 2018

@author: tamnich
"""
# IMPORT PACKAGES
import Metadata
import Question_Set_Cleaning
import re
import csv
import string
import time
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer as TV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import text
from sklearn.model_selection import train_test_split


##### STEP 0: GENERATE TRAINING AND TESTING SETS, AND OTHER REQUIRED VARIABLES
def generate_sets (questionset):
    # OVERALL TRAIN / TEST
    set_B = []
    set_E = []
    set_F = []
    
    # BREAKDOWN BY B/E/F
    for i in range(0, len(questionset)):
        if questionset[i][1] == 'B':
            set_B.append(questionset[i])
        elif questionset[i][1] == 'E':
            set_E.append(questionset[i])
        elif questionset[i][1] == 'F':
            set_F.append(questionset[i])
            
    return set_B, set_E, set_F

# GETS A LIST OF CONCEPTS FOR ALL QUESTIONS
def get_level2 (original_set):
    level2_lst = []
    for i in range(0, len(original_set)):
        if original_set[i][2] not in level2_lst:
            level2_lst.append(original_set[i][2])
        
    return level2_lst

# GETS A LIST OF CLASS WORD FOR ALL QUESTIONS
def get_classwords (original_set):
    classwords_lst = []
    for i in range(0, len(original_set)):
        the_classword = original_set[i][3]
        if the_classword not in classwords_lst:
            classwords_lst.append(the_classword)
            
    return classwords_lst

# GETS A LIST OF FIRST2WORDS FOR ALL QUESTIONS (COMBINATION OF THE ABOVE TWO FUNCTIONS)
def get_first2words (original_set):
    first2words_lst = []
    for i in range(0, len(original_set)):
        the_first2words = original_set[i][2] + " " + original_set[i][3]
        if the_first2words not in first2words_lst:
            first2words_lst.append(the_first2words)
        
    return first2words_lst


###### STEP 1: GENERATE TF-IDF REPRESENTATION OF THE QUESTION SET
# EXTRACTS THE QUESTION TEXT
def get_questions (original_set):
    questionset = []
    for i in range(0, len(original_set)):
        questionset.append(original_set[i][0])
    
    return questionset

def fit_tfidf(questionset, get_model = False):
    tfidf = TV(min_df = 2, analyzer = 'word', stop_words = 'english')
    tfidf.fit(questionset)
    vectors = tfidf.transform(questionset)
    
    if get_model:    
        return tfidf
    else:
        return vectors

###### OPTION TO CLASSIFY B/E/F USING MULTINOMIAL NAÏVE BAYES CLASSIFIER #####
### NAÏVE BAYES - B/E/F CLASSIFICATION
def classify(training_set, testing_set):
    
    # GET THE RESPONSE VARIABLES
    original_set = training_set + testing_set
    question_text = []
    response_var = []
    for i in range(0, len(original_set)):
        question_text.append(original_set[i][0])
        response_var.append(original_set[i][1])
    
    # WORD2VEC / TF-IDF REPRESENTATION
    tfidf = TV(min_df = 2, analyzer='word', stop_words='english')
    tfidf.fit(question_text)
    vectors = tfidf.transform(question_text)

    model = MultinomialNB(alpha = 0.5, fit_prior = True, class_prior = [0.3, 0.5, 0.2])
    model.fit(vectors[:len(training_set)], response_var[:len(training_set)])
    
    #prob = model.predict(vectors[len(training_set):])
    score = model.score(vectors[len(training_set):], response_var[len(training_set):])
    print ("Performance of the Naïve Bayes Classifier", score)
    
    return model, tfidf


##### STEP 1: TRAIN NEURAL NETWORKS

# USE THE SIGMOID FUNCTION TO FIT PROBABILITIES (HAS A RANGE OF [0,1])   
def sigmoid (x):
    return 1/(1+np.exp(-x))

# DERIVATIVE/GRADIENT OF THE SIGMOID FUNCTION (ASSUMING X IS A SIGMOID FUNCTION)
def sigmoid_derivative(x):
    #return sigmoid(x) * (1-sigmoid(x))
    #return np.exp(-x)/((1+np.exp(-x)) * (1+np.exp(-x)))
    return x*(1-x)

# SHUFFLES THE OBSERVATION MATRICES 
def shuffle_matrices (X, y):
    assert X.shape[0] == y.shape[0]
    p = np.random.permutation(X.shape[0])
    new_X = X[p]
    new_Y = y[p]
    
    return new_X, new_Y


#def find_gradient (layer, weight, bias, results):   
#    X = weight.dot(layer) + bias # SHAPE 40052*12
#    
#    empty_gradient_weight = 1/sqrt(F) * 
#    
#    F_matrix = (sigmoid(X) - results) ** 2
#    F = 0
#    
#    for p in range(0, weight.shape[0]):
#        for q in range(0, layer.shape[1]):
#            F += F_matrix[p][q]
    
# TRAINS A NEURAL NETWORK
def train_network (X, y, neurons = 60, alpha = 0.001, batch_size = 50, epoch = 10000):
    #np.random.seed(20437867)
    
    # CONVERT RESPONSE VARIABLES TO MATRIX
    y = np.array(y)
    
    # GENERATE INITIAL SYNAPSE WEIGHTS (TRANSFORMATION MATRIX) RANDOMLY - FROM -1 TO 1, MEAN 0
    synapse_0 = 2*np.random.random((X.shape[1], neurons)) - 1
    synapse_1 = 2*np.random.random((neurons, y.shape[1])) - 1
    
    #layer_2_delta = np.zeros((batch_size, y.shape[1]))
        
    start = 0
    end = start + batch_size
    epoch_count = 0
    
    while epoch_count <= epoch:
        
        ### BEGIN THE FORWARD PROPAGATION PROCESS ###
        
        layer_0 = X[start:end]  # INPUT LAYER
        layer_1 = sigmoid(layer_0.dot(synapse_0))  # HIDDEN LAYER
        layer_2 = sigmoid(layer_1.dot(synapse_1))  # OUTPUT LAYER
        
        target_resp = y[start:end]  # ANSWERS
        output_error = target_resp - layer_2
        
        if epoch_count % 50 == 0 and start == 0:
            print ("Epoch", epoch_count, "- Error: ", str(np.mean(np.abs(output_error))))

        ### INITIATE BACK PROPAGATION PROCESS ###
        
        # FIND THE GRADIENT OF THE OUTPUT LAYER
#        F = 0
#        for p in range(0, layer_2.shape[0]):
#            for q in range(0, layer_2.shape[1]):
#                F += (layer_2[p][q] - float(target_resp[p][q])) ** 2
#        
#        for p in range(0, layer_2.shape[0]):
#            for q in range(0, layer_2.shape[1]):
#                layer_2_delta[p][q] = sigmoid_derivative(layer_2[p][q])/np.sqrt(F)
#                
        #print (layer_2_delta.shape)
        
        layer_2_delta = (target_resp - layer_2) * sigmoid_derivative(layer_2)
        gradient_layer_2 = (layer_1.T).dot(layer_2_delta)

        # FIND WHAT THE FIRST SYNAPSE HAS TO CHANGE BASED ON CHANGES OF THE SECOND LAYER
        layer_1_delta = layer_2_delta.dot(synapse_1.T)

        gradient_layer_1 = (layer_0.T).dot(layer_1_delta)
        
        # ADJUST SYNAPSES
        synapse_1 = synapse_1 + alpha * gradient_layer_2
        synapse_0 = synapse_0 + alpha * gradient_layer_1
        
        
        ### MOVE ON TO NEXT BATCH ###
        start = end
        end = end + batch_size
        
        # RESET START WHEN REACHED THE END OF THE MATRIX
        if start == X.shape[0]:
            start = 0
            end = start+batch_size
            epoch_count += 1
            X, y = shuffle_matrices(X, y)
        
        # IF GETTING TO THE END OF THE MATRIX, FIX THE END OF THE BATCH TO THE END
        if end > X.shape[0]:
            start = X.shape[0] - batch_size
            end = X.shape[0]
    
    return [synapse_0, synapse_1]

# CREATES THE NEURAL NETWORK
def get_responses (original_set, response_index, responses):
    response_var = []
    for i in range(0, len(original_set)):
        the_response = original_set[i][response_index] # 1 = B/E/F, 2 = CONCEPT, 3 = CLASS WORD
        response_vector = [0] * len(responses)
        
        if the_response in responses:
            response_vector[responses.index(the_response)] = 1
        
        response_var.append(response_vector)
    
    return response_var

##### STEP 3: GENERATE PREDICTION FOR A PARTICULAR QUESTION TEXT

# PREDICT THE CLASS OF A SPECIFIC QUESTION
def predict_specific(tfidf, neural, responses, sentence):
    sentence_lst = []
    
    sentence = Question_Set_Cleaning.data_clean(sentence)
    sentence_lst.append(sentence)
    sentence_vector = tfidf.transform(sentence_lst)
#    bef_vector = tfidf_BEF.transform(sentence_lst)
#    
#    print(model_BEF.predict(bef_vector))
    
    synapse_0_BEF = neural[0][0]
    synapse_1_BEF = neural[0][1]
    
    # PASS THROUGH BEF NEURAL NETWORK
    first_layer = sigmoid(sentence_vector.dot(synapse_0_BEF))
    second_layer = sigmoid(first_layer.dot(synapse_1_BEF))
    
    max_prob = responses[0][second_layer.argmax()]
    nli = second_layer.argmax() + 1 # NLI = NEXT LAYER INDEX
    print (max_prob)
    
    first_layer = sigmoid(sentence_vector.dot(neural[nli][0]))
    second_layer = sigmoid(sentence_vector.dot(neural[nli][1]))
        
    max_prob = responses[nli][second_layer.argmax()]
    print (max_prob)
    
    return max_prob

# APPLY NEURAL NETWORK ON THE TESTING SET AND EVALUATE ACCURACY
def batch_predict_score (tfidf, neural, responses, batch):
#    sentence_lst = []
    real_response = []
    num_of_corrects = 0
    out_of = len(batch)
    
    for i in range(0, len(batch)):
        prediction = predict_specific(tfidf, neural, responses, batch[i][0])
        real_response = real_response.append(batch[i][2])
        
        if prediction == real_response:
            num_of_corrects += 1
    
    score = num_of_corrects / out_of
    
    print (score)
    

##### PERIPHERIES: EXPORT OPTIONS
def import_network ():
    synapse_0 = np.genfromtxt('F:\Harmonization Project\Synapse_0.csv', delimiter=',')
    print (synapse_0.shape)
    synapse_1 = np.genfromtxt('F:\Harmonization Project\Synapse_1.csv', delimiter=',')
    print (synapse_1.shape)
    return [synapse_0, synapse_1]
    
def export_network(network, specification):
    synapse_0 = network[0]
    synapse_1 = network[1]
    
    np.savetxt("F:\Harmonization Project\Synapse_0_" + specification + ".csv", synapse_0, delimiter=',')
    np.savetxt("F:\Harmonization Project\Synapse_1_" + specification + ".csv", synapse_1, delimiter=',')


def export_all(questionset):
    filename = "F:\Harmonization Project\Testing Set.csv"
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(questionset)
    
def main():
#    dataset = Metadata.importfile()
#    questionset = get_questionpairs(dataset)

    print("Re-import data and perform data cleaning once again? (Y/N)")
    reimport = input()
    
    # FOR FIRST TIME IMPORT OR WISH TO REFRESH DATA SET - PERFORM DATA CLEANING AGAIN
    if reimport == 'Y' or reimport == 'y':
        print ("Please wait. Data cleaning will take > 5 mins.")
        questionset = Question_Set_Cleaning.get_questionset()
    
    # USE PREVIOUSLY MADE DATA SET
    else:
        questionset = Question_Set_Cleaning.import_all()
    
    # GENERATE TRAINING AND TESTING SETS, ALONG WITH THE LIST OF POSSIBLE RESPONSES
    X_train, X_test = train_test_split(questionset, test_size=0.15, random_state=int(time.time()))
    tfidf = fit_tfidf(get_questions(X_train), True)
    
    X_train_B, X_train_E, X_train_F = generate_sets(X_train)
    X_test_B, X_test_E, X_test_F = generate_sets(X_test)
    
    BEF_resp = ['B', 'E', 'F']
    B_resp = get_level2(X_train_B + X_test_B)
    E_resp = get_level2(X_train_E + X_test_E)
    F_resp = get_level2(X_train_F + X_test_F)
#    responses = get_first2words (questionset)
    
    # IMPORT EXISTING NEURAL NETWORK TO SAVE TIME
#    print("Import an existing neural network? (Y/N)")
#    import_neural = input()
#    
#    if import_neural == 'Y' or import_neural == 'y':
#        neural = import_network()
#        tfidf = create_network(X_train, responses, False)   # GENERATE TF-IDF REPRESENTATIONS
#    
#    # OTHERWISE, CREATE A NEW NEURAL NETWORK AND TF-IDF
#    else:
#        print ("Building a neural network")
#        tfidf, neural = create_network (X_train, responses)
        
    
    print ("Building B/E/F Network")
    BEF_neural = train_network(tfidf.transform(get_questions(X_train)), get_responses(X_train, 1, BEF_resp), 20, 0.0005)
    export_network(BEF_neural, "BEF")
    
    print ("Building Network B")
    network_B = train_network(tfidf.transform(get_questions(X_train_B)), get_responses(X_train_B, 2, B_resp))
    export_network(network_B, 'B')
    
    print ("Building Network E")
    network_E = train_network(tfidf.transform(get_questions(X_train_E)), get_responses(X_train_E, 2, E_resp))
    export_network(network_E, 'E')
    
    print ("Building Network F")
    network_F = train_network(tfidf.transform(get_questions(X_train_F)), get_responses(X_train_F, 2, F_resp))
    export_network(network_F, 'F')

    neural_ensemble = []
    response_ensemble = []
    neural_ensemble.extend([BEF_neural, network_B, network_E, network_F])
    response_ensemble.extend([BEF_resp, B_resp, E_resp, F_resp])
        #print (len(X_train_B), len(X_train_E), len(X_train_F))
    
#    # B/E/F CLASSIFICATION
#    classify(X_train, X_test)
    


    batch_predict_score (tfidf, neural_ensemble, response_ensemble, X_test)
#    
#    # TRAIN NEURAL NETWORKS FOR B, E, F, RESPECTIVELY
    #tfidf_B, neural_B = create_network (X_train_B, 'B')
#    neural_E = create_network (X_train, 'E')
#    neural_F = create_network (X_train, 'F')
#    
#    total_score = 0
#    for i in range(0, 100):
   # BEF_model, BEF_tfidf = classify(X_train, X_test)
#    total_score += score
    
#    print(total_score/100)
    
#    print (X_test[0:20])
#    print (probabilities[0:20])
    
    print (len(X_train), " ", len(X_test))
    print (X_train[1926])
    
    #export_all(X_test_B)
    
    again = "Y"
    
    while again == "Y":
        print ("Input a sentence for prediction")
        sentence = input()
        predict_specific(tfidf, neural_ensemble, response_ensemble, sentence)
        print ("Again? (Y/N)")
        again = input()
    
#    print ("Export Neural? (Y/N)")
#    export = input()
#    if export == "y" or export == "Y":
#        export_network(neural)

    
if __name__ == '__main__':
    main()

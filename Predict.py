import os
import numpy as np
import pandas as pd
import pickle
from sklearn.externals import joblib
import sys

def fastaToNgram(fastaSequenceFile, ngram):
    f=open(fastaSequenceFile,"r")
    lines=f.readlines()
    f.close()
    fasttext_input_sequence_dic={}
#    threshold=float(lines[0][:-1])
    fastaSequence=""
    temp=lines[0]
    temp=temp.replace(">sp|","").replace(">","")
    proteinID=temp[:temp.find("|")]
    for line in lines:
        if line.find(">")<0:
            if line[-1]=="\n":
                fastaSequence+=line[:-1]
            else:
                fastaSequence+=line
        else:
            fasttext_input_sequence_dic[proteinID]=fastaSequence
            temp=line
            temp=temp.replace(">sp|","").replace(">","")
            proteinID=temp[:temp.find("|")]
            fastaSequence=""
    fasttext_input_sequence_dic[proteinID]=fastaSequence
    #cho nay ko gan ngram=3 vi voi bai tnf, feature tot nhat la combine ngram2 va ngram3
    for  key in fasttext_input_sequence_dic.keys():
        fastaSequence=fasttext_input_sequence_dic.get(key)
        fasttext_input_sequence=""
        i=0
        while i<len(fastaSequence)-ngram+1:
            for j in range(ngram):
                fasttext_input_sequence+=fastaSequence[i+j]
            fasttext_input_sequence+=" "
            i=i+1
#        print(fasttext_input_sequence)
        fasttext_input_sequence_dic[key]=fasttext_input_sequence[:-1]
        
    return fasttext_input_sequence_dic

def create_word_vector(embedding_file):
    word_int={}
    f=open(embedding_file,"r")
    lines=f.readlines()
    for line in lines[1:]:
        word_int[line.split()[0]]=[float(line.split()[1])] #dim=1
    del word_int["</s>"]
    return word_int

def init(word_int):
    input_word_int={}
    for key in word_int.keys():
        input_word_int[key]=[0,0]#dim=2
    return input_word_int

def count_word(aList, word):
    count=0
    for l in aList:
        if l==word:
            count+=1
    return count

def create_svm_input_from_dict(embedding_file, fasttext_input_sequence_dic, svm_input_file):
    word_int = create_word_vector(embedding_file)
    input_word_int = init(word_int)
    f=open(svm_input_file,"w") #w: allow overwrite
    for fasttext_input_sequence in fasttext_input_sequence_dic.values():
        for ngram in fasttext_input_sequence.split():
            count=count_word(fasttext_input_sequence[:-1].split(),ngram)
            if word_int.get(ngram)!=None:
                input_word_int[ngram]=[count*word_int.get(ngram)[0]]#dim1
        for key in input_word_int.keys():
            f.write(str('{:.3f}'.format(input_word_int.get(key)[0]))+", ")#dim 1, lay 3 chinh xac den 3 chu so
        f.write("\n")    
    f.close()
    

def create_svm_input_from_one_seq(embedding_file, fasttext_input_sequence, svm_input_file):
    f=open(svm_input_file,"w") #w: allow overwrite
    word_int = create_word_vector(embedding_file)
    input_word_int = init(word_int)
    # f=open(svm_input_file,"w") #w: allow overwrite --> sai cho nay vi da open file o line 87
    for ngram in fasttext_input_sequence.split():
        count=count_word(fasttext_input_sequence[:-1].split(),ngram)
        if word_int.get(ngram)!=None:
            input_word_int[ngram]=[count*word_int.get(ngram)[0]]#dim1
    for key in input_word_int.keys():
        f.write(str('{:.3f}'.format(input_word_int.get(key)[0]))+", ")#dim 1, lay 3 chinh xac den 3 chu so
    
    f.write("\n")    
    f.close()

def labelToOneHot(label):# 0--> [1 0], 1 --> [0 1]
    label = label.reshape(len(label), 1)
    label = np.append(label, label, axis = 1)
    label[:,0] = label[:,0] == 0;
    return label


def run(svm_input_file, model_file):
    dataset = pd.read_csv(svm_input_file, header=None)
    X_test = dataset.iloc[:, 0:-1].values
    try:
        classifier=joblib.load(model_file)
    except (IOError, pickle.UnpicklingError, AssertionError):
        print(pickle.UnpicklingError)
        return True

    y_pred = classifier.predict_proba(X_test)
    return y_pred[0][1] #all the second values


inputFile= sys.argv[1]
outputFile="Result.csv" #w: allow overwrite
print("input file ",inputFile)
#ngram3
fasttext_input_sequence_dic=fastaToNgram("your_fasta_file.fasta",3)
f=open(outputFile,"w")
f.write("ProteinID,ComplexI,ComplexII,ComplexIII,ComplexIV,ComplexV\n")
answerDict={}

for proteinID in fasttext_input_sequence_dic.keys():
    f.write(proteinID+",")
    fasttext_input_sequence=fasttext_input_sequence_dic.get(proteinID)
    for cla in ["A","B","C","D","E"]:
            svm_input_file="tmp//"+cla+"_"+proteinID+".csv"
            embedding_file="fastText embedding vectors//fastText embedding vectors//dfSubword.embedding.train."+cla+".vec"
            create_svm_input_from_one_seq(embedding_file, fasttext_input_sequence, svm_input_file)
            model_file="Models//Models//"+cla+".pickle_model.pkl"
            answerForOneClass = run(svm_input_file,model_file)
            f.write(str("{:.3f}".format(answerForOneClass))+",")
    f.write("\n")

#delete temporary files    
for filename in os.listdir("tmp"):
    os.remove("tmp//"+filename)
    
f.close()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:08:40 2022

@author: hrithvikatluri1
"""
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder
from flask import Flask,redirect,url_for,request,render_template
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
global New_Patient_symptoms
global df
import networkx as nx
global FG
global source

h = pd.read_csv('trail_data.csv')
df=h
df

def clean(df):
    df = df.drop(df.index[[38,39,40]])
    new=[]
    for st in df['nxt-node']:
        if(st=='n5'):
            new.append('safe')
        else:
            new.append(st)
    df['nxt-node']=new
    label_encoder = LabelEncoder()
    cols=['temp','Dry_Cough','Fatigue','Loss_Of_Appetite','Sore_Throat','Headache','Loss_Of_Smell/Taste','Vomiting']
    for i in cols:
        df[i] = label_encoder.fit_transform(df[i])
    l=[]
    for no in df['Body_Aches']:
        if(no=='extreme' or no=='extreme '):
            x=3
            l.append(x)
        elif(no=='high'):
            y=2
            l.append(y)
        elif(no=='normal'):
            z=1
            l.append(z)
        else:
            z1=0
            l.append(z1)
    df['Body_Aches']=l
    df['Shortness_Of_Breath']=df['Shortness_Of_Breath'].replace(['no','normal','high','extreme','extreme '],[0,1,2,3,3])
    return df
df=clean(df)

features=df[['temp','Dry_Cough','Fatigue','Loss_Of_Appetite','Body_Aches','Shortness_Of_Breath','Sore_Throat','Headache','Loss_Of_Smell/Taste','Vomiting']]
label=df['src-node']
label1=df[['src-node','nxt-node']]
##ACCURACY for algos
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.3)
def accuracy():
    gnb = GaussianNB()
#Train the model using the training sets
    gnb.fit(X_train, y_train)
#Predict the response for test dataset
    y_pred = gnb.predict(X_test)
    print("Accuracy for gaussian naive bayes:",metrics.accuracy_score(y_test, y_pred))
    kn = KNeighborsClassifier(metric='manhattan',n_neighbors=1)
# kn= KNeighborsClassifier(metric='cosine',n_neighbors=1)
#Train the model using the training sets
    kn.fit(X_train, y_train)
#Predict the response for test dataset
    y_pred = kn.predict(X_test)
    print("Accuracy for Knn:",metrics.accuracy_score(y_test, y_pred))
accuracy()

#Create a Gaussian Classifier

#global New_Patient_symptoms
New_Patient_symptoms=[]
from flask import Flask,render_template,redirect,url_for,request
app=Flask(__name__,template_folder='.')

@app.route('/result',methods=['POST','GET'])
def result():
    FG = nx.from_pandas_edgelist(df, source='src-node', target='nxt-node', edge_attr=True,create_using=nx.DiGraph())
    req=request.form['sel']
    New_Patient_symptoms.append(req)
    #print('name:',name,'Maths:',maths)
    Listnxtnode=list(df['nxt-node'])
    Listsrcnode=list(df['src-node'])
    if(req=='c'):
        cost_path= nx.astar_path(FG,source=source, target='safe',weight='cost_treatment')
        print("cost-optimized path",cost_path)
        New_Patient_symptoms.append(cost_path[1])
        Listplan=list(df['Plan'])
        wxyz=cost_path[0]
        for i in range(0,len(Listsrcnode)):
            if(Listsrcnode[i]==cost_path[0]):
                if(Listnxtnode[i]==cost_path[1]):
                    xcost=Listplan[i]
                print("Recommended medication for cost-optimized plan:",xcost)
                return render_template('final_page.html',plan=req,med=xcost)
    
    elif(req=='t'):
        FG = nx.from_pandas_edgelist(df, source='src-node', target='nxt-node', edge_attr=True,create_using=nx.DiGraph())
        time_path= nx.astar_path(FG,source=source,target='safe',weight='treatment_time(days)')
        print("time-optimized path",time_path)
        New_Patient_symptoms.append(time_path[1])
        Listplan=list(df['Plan'])
        for i in range(0,len(Listsrcnode)):
            if(Listsrcnode[i]==time_path[0]):
                if(Listnxtnode[i]==time_path[1]):
                    xtime=Listplan[i]
                print("Recommended medication for days-optimized plan:",xtime)
                return render_template('final_page.html',plan=req,med=xtime)
    print(New_Patient_symptoms)
    return render_template('result.html',res_list=New_Patient_symptoms)
@app.route('/req',methods=['POST','GET'])
def required():
    global req
    global acc_node
    if request.method=='POST':
        temp=int(request.form['temp'])
        New_Patient_symptoms.append(temp)
        cough=int(request.form['cough'])
        New_Patient_symptoms.append(cough)
        fatigue=int(request.form['fatigue'])
        New_Patient_symptoms.append(fatigue)
        appetite=int(request.form['appetite'])
        New_Patient_symptoms.append(appetite)
        body=int(request.form['body'])
        New_Patient_symptoms.append(body)
        breath=int(request.form['breath'])
        New_Patient_symptoms.append(breath)
        throat=int(request.form['throat'])
        New_Patient_symptoms.append(throat)
        head=int(request.form['head'])
        New_Patient_symptoms.append(head)
        smell=int(request.form['smell'])
        New_Patient_symptoms.append(smell)
        vomit=int(request.form['vomit'])
        New_Patient_symptoms.append(vomit)
        model = GaussianNB()
        x=[]
        # Train the model using the training sets
        model.fit(features,label)
        #Predict Output
        predicted= model.predict([New_Patient_symptoms]) # 0:Overcast, 2:Mild
        print("Predicted Value:", predicted)
        x.append(predicted)
        acc_node=predicted
        print(acc_node)
        global FG
        global PS
        FG = nx.from_pandas_edgelist(df, source='src-node', target='nxt-node', edge_attr=True,create_using=nx.DiGraph())
        ps=[]
        global source
        source=''.join(map(str, acc_node))
        for path in nx.all_simple_paths(FG, source=''.join(map(str, acc_node)), target='safe'):
            print(path)
 
        return render_template('required.html',predicted=acc_node)
    

@app.route('/')
def ipynb_trail():
    return render_template('ipynb_trail.html')

if __name__=='__main__':
    app.run(debug=True)

    

    

    
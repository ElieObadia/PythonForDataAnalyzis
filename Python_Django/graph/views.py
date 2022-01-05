from django.shortcuts import render
from .forms import ModelForm
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



def predict_model(request):
    f = open("ml_model/acc_logreg.txt", "r")
    acc_logreg = (f.read())
    f.close()
    f = open("ml_model/acc_tree.txt", "r")
    acc_tree = (f.read())
    f.close()

    fig = plt.figure(figsize = (10,5))
    ax = plt.subplot()
    models = ['Logistic Regression', 'Decision Tree']
    vals = [acc_logreg, acc_tree]
    plt.bar([0,1], vals, align='center', width = 0.2,label = 'model comparaison')
    plt.xticks([0,1],models)
    return render(request, 'home.html', {'acc_logreg':acc_logreg, 'acc_tree':acc_tree})
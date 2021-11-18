from django.shortcuts import render, HttpResponse
import numpy as np
import pandas as pd


# Create your views here.
    
global v,model
def predict(request):
    df = pd.read_csv("../spam.csv")
    df.groupby('Category')
    df['spam']=df['Category'].apply(lambda x: 1 if x=='spam' else 0)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.Message,df.spam,test_size=0.2)
    from sklearn.feature_extraction.text import CountVectorizer
    v = CountVectorizer()
    X_train_count = v.fit_transform(X_train.values)
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(X_train_count,y_train)
    
    # emails = [
    #     'Hey mohan, can we get together to watch footbal game tomorrow?',
    #     'Upto 20 percent discount on parking, exclusive offer just for you. Dont miss this reward!'
    # ]
    # emails_count = v.transform(emails)
    # model.predict(emails_count)
    #X_test_count = v.transform(X_test)
    # model.score(X_test_count, y_test)
    if request.method=="POST":
        
        email=request.POST.get('email_text')
        email_text=[email]
        emails_count=v.transform(email_text)
        #emails = ['Hey mohan, can we get together to watch footbal game tomorrow?']
        emails_count = v.transform(email_text)
        #model.predict(emails_count)
        result=model.predict(emails_count)
        if result[0]==1:
            res='SPAM'
        else:
            res='HAM(notSpam)'
        context={'res': res,'email_text':email}    
        return render(request,'sp_pred.html',{'context':context} )
    return render(request,'sp_pred.html')

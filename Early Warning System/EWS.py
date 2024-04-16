import graphics
import os
import pandas as pd
from button import *
from typing import Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Models.RandomForest import random_forest_result
from Models.RNN import rnn_result
import seaborn as sns
import math
import graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
import numpy as np


class Patient:
    def __init__(self, age, gender, diagnosis, risk, icu, ethnicity, test_results={}):
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.risk = risk
        self.icu = icu
        self.ethnicity = ethnicity
        self.test_results = test_results
        

    def switchGender(self):
        if(self.gender == "M"):
            self.gender = "F"
        else:
            self.gender = "M"

    def getAge(self):
        return self.age
    def getGender(self):
        return self.gender
    def getDiagnosis(self):
        return self.diagnosis
    def getTestResults(self):
        return self.test_results
    def getRisk(self):
        return self.risk
    
    def gender_to_numeric(self):
        if self.gender =="M":
            return 10
        else:
            return 0
        
    def ethnicity_to_numeric(self):
        if 'WHITE' in self.ethnicity:
            return 10
        elif 'AFRICAN AMERICAN' in self.ethnicity or 'BLACK' in self.ethnicity:
            return 30
        elif 'ASIAN' in self.ethnicity:
            return 40
        elif 'HISPANIC' in self.ethnicity or 'LATINO' in self.ethnicity:
            return 20
        else:
            return 0
        


class UI:
    def Start(self):
        # Create a window for EWS
        win = GraphWin("Early Warning System", 1400, 800)
        win.setBackground("black")

        # text displays
        title = Text(Point(700,100),"Early Warning System")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text1 = Text(Point(300,500),"Choose a model:")
        text1.setFill("Cyan")
        text1.setSize(20)
        text1.draw(win)

        text2 = Text(Point(270,180),"Patient Info:")
        text2.setFill("Cyan")
        text2.setSize(20)
        text2.draw(win)

        text3 = Text(Point(450,250),"Diagnosis:")
        text3.setFill("Cyan")
        text3.setSize(15)
        text3.draw(win)

        text4 = Text(Point(680,250),"Age:")
        text4.setFill("Cyan")
        text4.setSize(15)
        text4.draw(win)

        text5 = Text(Point(850,250),"Ethnicity:")
        text5.setFill("Cyan")
        text5.setSize(15)
        text5.draw(win)

        text = Text(Point(900,375),"Hours in ICU:")
        text.setFill("Cyan")
        text.setSize(15)
        text.draw(win)

        text5 = Text(Point(500,375),"Test Result Directory:")
        text5.setFill("Cyan")
        text5.setSize(15)
        text5.draw(win)

        selectedText = Text(Point(225,395),"Selected")
        selectedText.setFill("Cyan")
        selectedText.setSize(10)
        selectedText.draw(win)

        text7 = Text(Point(270,240),"Gender")
        text7.setFill("Cyan")
        text7.setSize(15)
        text7.draw(win)

        # Draw Buttons
        rnnButton = Button(win, Point(400,600),100,75,"RNN")
        rfButton = Button(win, Point(575,600),175,75,"Random Forest")
        lrButton = Button(win, Point(810,600),200,75,"Linear Regression")
        lstmButton = Button(win, Point(1000,600),100,75,"LSTM")
    
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")


        # Patient Info Buttons & Entries
        maleButton = Image(Point(220, 320), "./images/male.png")
        maleButton.draw(win)
        selectionBox = Rectangle(Point(195,260),Point(245,380))
        selectionBox.setOutline('Cyan')
        selectionBox.draw(win)

        femaleButton = Image(Point(300, 320), "./images/female.png")
        femaleButton.draw(win)

        diagnosisEntry = Entry(Point(580, 250), 15)
        diagnosisEntry .setText("Cardiac Arrest")
        diagnosisEntry .draw(win)

        ageEntry = Entry(Point(730, 250), 3)
        ageEntry .setText("75")
        ageEntry .draw(win)

        ethnicityEntry = Entry(Point(930, 250), 5)
        ethnicityEntry .setText("Asian")
        ethnicityEntry .draw(win)

        directoryEntry = Entry(Point(700, 375), 20)
        directoryEntry .setText('./example1.csv')
        directoryEntry .draw(win)

        icuEntry = Entry(Point(990, 375), 5)
        icuEntry.setText("3")
        icuEntry.draw(win)

        p = Patient(age = ageEntry.getText(), gender = "M", diagnosis= diagnosisEntry.getText(), risk = 56, icu = icuEntry.getText(), ethnicity= ethnicityEntry.getText().upper(),test_results= [])
        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if ( 195 <= pt.getX() and pt.getX() <= 250 and 270 <= pt.getY() and pt.getY() <= 370):
                selectedText.undraw()
                selectedText = Text(Point(220,395),"Selected")
                selectedText.setFill("Cyan")
                selectedText.setSize(10)
                selectedText.draw(win)
                selectionBox.undraw()
                selectionBox = Rectangle(Point(195,260),Point(245,380))
                selectionBox.setOutline('Cyan')
                selectionBox.draw(win)

                # Set gender type 
                if(p.getGender() != "M"):
                    p.switchGender()

            if ( 275 <= pt.getX() and pt.getX() <= 325 and 270 <= pt.getY() and pt.getY() <= 370):
                selectedText.undraw()
                selectedText = Text(Point(300,395),"Selected")
                selectedText.setFill("Cyan")
                selectedText.setSize(10)
                selectedText.draw(win)
                selectionBox.undraw()
                selectionBox = Rectangle(Point(275,260),Point(325,380))
                selectionBox.setOutline('Cyan')
                selectionBox.draw(win)
                # Set gender type 
                if(p.getGender() != "F"):
                    p.switchGender()


            if rnnButton.isClicked(pt) or lstmButton.isClicked(pt) or rfButton.isClicked(pt) or lrButton.isClicked(pt):
                df = pd.read_csv(directoryEntry.getText(),header = None, skiprows = 1, nrows=1)
                tests = {"Chloride" : df[0].iloc[0], "Creatinine" : df[1].iloc[0], "Potassium" : df[2].iloc[0], "Sodium" : df[3].iloc[0],  "Hematocrit" : df[4].iloc[0]}
                test_results = list(tests.values())
                p = Patient(age = ageEntry.getText(), gender = "M", diagnosis= diagnosisEntry.getText(), risk = 56, icu = icuEntry.getText(), ethnicity= ethnicityEntry.getText().upper(),test_results= tests)
                features = [p.getAge(), p.gender_to_numeric(), p.ethnicity_to_numeric()]+ test_results
                features = [float(item) for item in features]

            if rnnButton.isClicked(pt):
                win.close()
                self.rnnPage(p,features)
                
            elif lstmButton.isClicked(pt):
                win.close()
                self.lstmPage(p)

            elif rfButton.isClicked(pt):
                win.close()
                self.rfPage(p,features)
            
            elif lrButton.isClicked(pt):
                win.close()
                self.lrPage(p)
        
            try:
                pt = win.getMouse()
            except:
                sys.exit()
        win.close()


    # RNN page
    def rnnPage(self,p,features):
        win = GraphWin("RNN Results", 1400, 800)
        win.setBackground("black")
        # title = Text(Point(700,100),"Risk of "+ p.getDiagnosis() + ": " + str(p.getRisk()) + "%")
        title = Text(Point(700,100),"RNN model")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text1 = Text(Point(300,530), "Age: "+ p.getAge())
        text1.setFill("Cyan")
        text1.setSize(20)
        text1.draw(win)


        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        goBackButton = Button(win, Point(100, 750), 100, 50, "Go Back")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")

        text = Text(Point(600,230), "Classification Report")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)
        
        text = Text(Point(900,230), "Test Results")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)

        if (p.getGender() == "M"):
            text = Text(Point(300,180), "Male")
            text.setFill("Cyan")
            text.setSize(20)
            text.draw(win)
        else:
            text = Text(Point(300,180), "Female")
            text.setFill("Cyan")
            text.setSize(20)
            text.draw(win)


        loading_text = Text(Point(750,350), "Calculating...")
        loading_text.setFill("Cyan")
        loading_text.setSize(24)
        loading_text.draw(win)

        loading_text1 = Text(Point(750,400), "◌")
        loading_text1.setFill("Cyan")
        loading_text1.setSize(36)
        loading_text1.draw(win)

        model, y_true, y_pred_classes, test_acc, test_loss = rnn_result()

        loading_text.undraw()
        loading_text1.undraw()
        
        y = 300
        tests = p.getTestResults()
        for key, value in tests.items():
            text = Text(Point(900,y), str(key)+ ": "+str(value) )
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50
        
        text = Text(Point(580,380),classification_report(y_true, y_pred_classes)  )
        text.setFill("Cyan")
        text.setSize(15)
        text.draw(win)

        
        features_array = np.array(features).reshape(1, len(features), 1)
        prediction = model.predict(features_array)

        if(prediction[0] > prediction[1]):
            person = Image(Point(300, 350), "./images/p1.png")
            person.draw(win)
            text = Text(Point(700,530), "Your patient has low risk")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
        # elif(p.getRisk() <= 50):
        #     person = Image(Point(300, 350), "./images/p2.png")
        #     person.draw(win)
        #     text = Text(Point(700,630), "Your patient is at risk")
        #     text.setFill("Cyan")
        #     text.setSize(15)
        #     text.draw(win)
        # elif(p.getRisk() <= 75):
        #     person = Image(Point(300, 350), "./images/p3.png")
        #     person.draw(win)
        #     text = Text(Point(700,630), "Your patient is at high risk")
        #     text.setFill("Cyan")
        #     text.setSize(15)
        #     text.draw(win)
        else:
            person = Image(Point(300, 350), "./images/p4.png")
            person.draw(win)
            text = Text(Point(700,630), "Your patient has high risk")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)







        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                cm = confusion_matrix(y_true, y_pred_classes)
                self.modelPage("RNN", cm , None , None, None )
    
            elif goBackButton.isClicked(pt):
                win.close()
                self.Start()


            try:
                pt = win.getMouse()
            except:
                sys.exit()
        
        win.close()


    # LSTM page
    def lstmPage(self,p):

        def read_lstm_output(filename):
            with open(filename, 'rb') as file:
                filesize = os.stat(filename).st_size
                if filesize == 0:
                    return None
                else:
                    offset = -2
                    while -offset <= filesize:  # Go back until the beginning of the file
                        file.seek(offset, os.SEEK_END)
                        if file.read(1) == b'\n':
                            return file.readline().decode()
                        offset -= 1
                    file.seek(0)
                    return file.readline().decode()

        result_string = read_lstm_output('./LSTM/train_test_avg.out')
        result_string = result_string.split(": ", 1)[1]

        # Now, let's create a dictionary by splitting the remaining string correctly
        metrics = result_string.split(", ")
        results = {metric.split(": ")[0]: float(metric.split(": ")[1]) for metric in metrics}

        

        win = GraphWin("LSTM Results", 1400, 800)
        win.setBackground("black")

        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        goBackButton = Button(win, Point(100, 750), 100, 50, "Go Back")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")

        title = Text(Point(700,100),"LSTM Model")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text = Text(Point(300,530), "stuff")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)

        text = Text(Point(600,250), "Model Statistics")
        text.setFill("Cyan")
        text.setSize(25)
        text.draw(win)

        y = 325
        for key, value in results.items():
            text = Text(Point(600,y), key+" : " + str(math.ceil(value * 1000)/1000) + "%")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50

        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                self.modelPage("RNN")

            elif goBackButton.isClicked(pt):
                win.close()
                self.Start()

            try:
                pt = win.getMouse()
            except:
                sys.exit()
            
        

    # Random Forest Page
    def rfPage(self,p, features):

        win = GraphWin("Random Forest Results", 1400, 800)
        win.setBackground("black")

        rfImage = Image(Point(300, 350), "./images/rf.png")
        rfImage.draw(win)
        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        goBackButton = Button(win, Point(100, 750), 100, 50, "Go Back")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")
        
        title = Text(Point(700,100),"Random Forest Model")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text1 = Text(Point(600,250), "Model Statistics ")
        text1.setFill("Cyan")
        text1.setSize(24)
        text1.draw(win)
        text1 = Text(Point(900,250), "Predicted class: ")
        text1.setFill("Cyan")
        text1.setSize(24)
        text1.draw(win)
        loading_text = Text(Point(750,350), "Calculating...")
        loading_text.setFill("Cyan")
        loading_text.setSize(24)
        loading_text.draw(win)

        loading_text1 = Text(Point(750,400), "◌")
        loading_text1.setFill("Cyan")
        loading_text1.setSize(36)
        loading_text1.draw(win)

        rf, X_train, y_test, y_pred, results = random_forest_result()
        loading_text.undraw()
        loading_text1.undraw()
        # Display Mode Stats

        prediction = rf.predict(features)
        text1.undraw()

        if prediction[0] == 0:
            text1 = Text(Point(900,250), "Predicted class: "+ "0")
            text1.setFill("Cyan")
            text1.setSize(24)
            text1.draw(win)

            text1 = Text(Point(900,350), "The patient has no-risk of")
            text1.setFill("Cyan")
            text1.setSize(15)
            text1.draw(win)

            text1 = Text(Point(900,375), "mortality based on our model")
            text1.setFill("Cyan")
            text1.setSize(15)
            text1.draw(win)
            
        else:
            text1 = Text(Point(900,250), "Predicted class: "+ "1")
            text1.setFill("Cyan")
            text1.setSize(24)
            text1.draw(win)

            text1 = Text(Point(900,350), "The patient has high-risk of")
            text1.setFill("Cyan")
            text1.setSize(15)
            text1.draw(win)

            text1 = Text(Point(900,375), "mortality based on our model")
            text1.setFill("Cyan")
            text1.setSize(15)
            text1.draw(win)
            


        y = 325
        for key, value in results.items():
            text = Text(Point(600,y), key+" : " + str(math.ceil(value * 1000)/1000) + "%")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50

        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                self.modelPage("Random Forest", rf, X_train, y_test, y_pred)
            elif goBackButton.isClicked(pt):
                win.close()
                self.Start()
            try:
                pt = win.getMouse()
            except:
                sys.exit()
            

        win.close()


    # Logistic Regression Page
    def lrPage(self,p):
        win = GraphWin("Logistic Regression model", 1400, 800)
        win.setBackground("black")

        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        goBackButton = Button(win, Point(100, 750), 100, 50, "Go Back")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")
        
        title = Text(Point(700,100),"Logistic Regression model")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text = Text(Point(300,530), "Accuracy: ")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)



        

        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                self.modelPage("LR")
            
            elif goBackButton.isClicked(pt):
                win.close()
                self.Start()

            try:
                pt = win.getMouse()
            except:
                sys.exit()
            


    # When View model button is clicked return different results
    def modelPage(self, model_type, model, X_train, y_test, y_pred ):
        if(model_type == "RNN"):
            plt.figure(figsize=(10,7))
            sns.heatmap(model, annot=True, fmt='d')
            plt.title('Confusion Matrix')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.savefig('rnn_confusion_matrix.png', bbox_inches='tight')
            plt.close()
            
            win = GraphWin("RNN output", 800, 800)
            win.setBackground("white")
             
            rnn_cm = Image(Point(400,400), './rnn_confusion_matrix.png')
            rnn_cm.draw(win)
            
            

        elif(model_type == "Random Forest"):
            for i, tree in enumerate(model.estimators_[:3]):
                dot_data = export_graphviz(tree,
                                        out_file=None,
                                        feature_names=X_train.columns,
                                        filled=True,
                                        max_depth=2,
                                        impurity=False,
                                        proportion=True)
                graph = graphviz.Source(dot_data)
                graph.render(f'rf_tree_{i}', format='png', cleanup=True)
            

            cm = confusion_matrix(y_test, y_pred)
            ConfusionMatrixDisplay(confusion_matrix=cm).plot()
            for i in range(3):
                img = mpimg.imread(f'rf_tree_{i}.png')
                plt.figure(figsize=(10,10))
                plt.imshow(img)
                plt.axis('off')  # Do not show axes to keep it tidy
                plt.show()


        elif(model_type == "LSTM"):
            win = GraphWin("LSTM output", 800, 800)
            win.setBackground("white")

        elif(model_type == "LR"):
            win = GraphWin("Linear Regression output", 800, 800)
            win.setBackground("white")


def main():
    ui = UI()
    ui.Start()

main()







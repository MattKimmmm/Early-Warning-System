import graphics
from button import *
from typing import Dict
# from ..Models.RandomForest import RandomForest


class Patient:
    def __init__(self, age, gender, diagnosis, risk, icu, test_results={} ):
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.risk = risk
        self.icu = icu
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
        rnnButton = Button(win, Point(300,600),100,75,"RNN")
        rfButton = Button(win, Point(500,600),175,75,"Random Forest")
        lstmButton = Button(win, Point(700,600),100,75,"LSTM")
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
        directoryEntry .setText("\example.csv")
        directoryEntry .draw(win)

        icuEntry = Entry(Point(990, 375), 5)
        icuEntry.setText("3")
        icuEntry.draw(win)

        tests = {"pO2" : 12, "Potassium" : 10, "pH" : 11, "Lactate" : 10}
        p = Patient(age = ageEntry.getText(), gender = "M", diagnosis= diagnosisEntry.getText(), risk = 56, icu = icuEntry.getText() ,test_results= tests)

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


            if rnnButton.isClicked(pt):
                win.close()
                self.rnnPage(p)
                
            elif lstmButton.isClicked(pt):
                win.close()
                self.lstmPage()

            elif rfButton.isClicked(pt):
                win.close()
                self.rfPage()
                
        
            try:
                pt = win.getMouse()
            except:
                sys.exit()
        win.close()


    # RRN page
    def rnnPage(self,p):
        win = GraphWin("RNN Results", 1400, 800)
        win.setBackground("black")
        title = Text(Point(700,100),"Risk of "+ p.getDiagnosis() + ": " + str(p.getRisk()) + "%")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text1 = Text(Point(300,530), "Age: "+ p.getAge())
        text1.setFill("Cyan")
        text1.setSize(20)
        text1.draw(win)


        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")
        
        if(p.getRisk() <= 25):
            person = Image(Point(300, 350), "./images/p1.png")
            person.draw(win)
            text = Text(Point(700,530), "Your patient is at low risk")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)

        elif(p.getRisk() <= 50):
            person = Image(Point(300, 350), "./images/p2.png")
            person.draw(win)
            text = Text(Point(700,630), "Your patient is at risk")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
        elif(p.getRisk() <= 75):
            person = Image(Point(300, 350), "./images/p3.png")
            person.draw(win)
            text = Text(Point(700,630), "Your patient is at high risk")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
        else:
            person = Image(Point(300, 350), "./images/p4.png")
            person.draw(win)
            text = Text(Point(700,630), "Your patient is in very high risk")
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)

        text = Text(Point(900,230), "Abnormal!")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)

        y = 300
        tests = p.getTestResults()
        for key, value in tests.items():
            text = Text(Point(900,y), str(key)+ ": "+str(value) )
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50

        text = Text(Point(600,230), "Model")
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


        y = 300
        for i in range(4):
            text = Text(Point(600,y), "Model Stat " + str(i))
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50




        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                self.modelPage("RNN")
            try:
                pt = win.getMouse()
            except:
                sys.exit()
        
        win.close()


    # LSTM page
    def lstmPage(self,p):
        win = GraphWin("LSTM Results", 1400, 800)
        win.setBackground("black")

        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")

        title = Text(Point(700,100),"Random Forest Model")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text = Text(Point(300,530), "Accuracy: ")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)

        y = 300
        for i in range(4):
            text = Text(Point(600,y), "Model Stat " + str(i))
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50

        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                self.modelPage("RNN")
            try:
                pt = win.getMouse()
            except:
                sys.exit()
            pt = win.getMouse()
        win.close()

    # Random Forest Page
    def rfPage(self,p):
        win = GraphWin("Random Forest Results", 1400, 800)
        win.setBackground("black")

        rf = Image(Point(300, 350), "./images/rf.png")
        rf.draw(win)
        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")
        
        title = Text(Point(700,100),"Random Forest Model")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)

        text = Text(Point(300,530), "Accuracy: ")
        text.setFill("Cyan")
        text.setSize(20)
        text.draw(win)


        
        # Display Mode Stats
        y = 300
        for i in range(4):
            text = Text(Point(600,y), "Model Stat " + str(i))
            text.setFill("Cyan")
            text.setSize(15)
            text.draw(win)
            y += 50

        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if viewButton.isClicked(pt):
                self.modelPage("RNN")
            try:
                pt = win.getMouse()
            except:
                sys.exit()
            pt = win.getMouse()

        win.close()


    # Linear Regression Page
    def lrPage(self,p):
        win = GraphWin("Linear Regression model", 1400, 800)
        win.setBackground("black")

        viewButton = Button(win, Point(700, 720), 150, 50, "View Model")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")
        
        title = Text(Point(700,100),"Random Forest Model")
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
            try:
                pt = win.getMouse()
            except:
                sys.exit()
            pt = win.getMouse()


    # decision tree page
    def dtPage(self, p):
        win = GraphWin("Decision Tree", 1400, 800)
        win.setBackground("black")



    # When View model button is clicked return different results
    def modelPage(self, model):
        if(model == "RNN"):
            win = GraphWin("RNN output", 800, 800)
            win.setBackground("white")
            

        elif(model == "Random Forest"):
            win = GraphWin("Random Forest output", 800, 800)
            win.setBackground("white")


        elif(model == "LSTM"):
            win = GraphWin("LSTM output", 800, 800)
            win.setBackground("white")

        elif(model == "LR"):
            win = GraphWin("Linear Regression output", 800, 800)
            win.setBackground("white")

        elif(model == "Decision Tree"):
            win = GraphWin("Decision Tree output", 800, 800)
            win.setBackground("white")



def main():
    ui = UI()
    ui.Start()

main()







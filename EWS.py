import graphics
from button import *
from typing import Dict



class Patient:
    def __init__(self, age, gender, diagnosis, risk,test_results=[] ):
        self.age = age
        self.gender = gender
        self.diagnosis = diagnosis
        self.risk = risk
        self.test_results = test_results

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
        win = GraphWin("Early Warning System", 1400, 800)
        win.setBackground("black")

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


        # Patient Info
        maleButton = Image(Point(220, 320), "male.png")
        maleButton.draw(win)

        femaleButton = Image(Point(300, 320), "female.png")
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

        
        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if ( 195 <= pt.getX() and pt.getX() <= 250 and 270 <= pt.getY() and pt.getY() <= 370):
                selectedText.undraw()
                selectedText = Text(Point(220,395),"Selected")
                selectedText.setFill("Cyan")
                selectedText.setSize(10)
                selectedText.draw(win)

            if ( 275 <= pt.getX() and pt.getX() <= 325 and 270 <= pt.getY() and pt.getY() <= 370):
                selectedText.undraw()
                selectedText = Text(Point(300,395),"Selected")
                selectedText.setFill("Cyan")
                selectedText.setSize(10)
                selectedText.draw(win)


            if rnnButton.isClicked(pt):
                win.close()
                p = Patient(age = ageEntry.getText(), gender = "M", diagnosis= diagnosisEntry.getText(), risk = 56, test_results= [])
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


    def rnnPage(self,p):
        win = GraphWin("RNN Results", 1400, 800)
        win.setBackground("black")
        title = Text(Point(700,100),"Risk of "+ p.getDiagnosis() + ": " + str(p.getRisk()) + "%")
        title.setFill("Cyan")
        title.setSize(36)
        title.setStyle("italic")
        title.draw(win)



        viewButton = Button(win, Point(700, 720), 100, 50, "View Model")
        exitButton = Button(win, Point(1325, 750), 100, 50, "Exit")

        pt = win.getMouse()
        while not exitButton.isClicked(pt):
            if(p.getRisk() <= 10):
                person = Image(Point(300, 320), "green.png")
                person.draw(win)
            elif(p.getRisk() <= 25):
                person = Image(Point(300, 320), "blue.png")
                person.draw(win)
            elif(p.getRisk() <= 50):
                person = Image(Point(300, 320), "yellow.png")
                person.draw(win)
            else:
                person = Image(Point(300, 320), "red.png")
                person.draw(win)


            try:
                pt = win.getMouse()
            except:
                sys.exit()
        
        win.close()


    def lstmPage(self):
        win = GraphWin("LSTM Results", 1400, 800)
        win.setBackground("black")

    def rfPage(self):
        win = GraphWin("Random Forest Results", 1400, 800)
        win.setBackground("black")




def main():
    ui = UI()
    ui.Start()

main()







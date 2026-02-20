using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System;
using System.Linq;

public class QuestionLayout : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI questionNumber;
    [SerializeField] private TextMeshProUGUI marksGained;
    [SerializeField] private TextMeshProUGUI solutions;
    [SerializeField] private TextMeshProUGUI answers;
    [SerializeField] private TextMeshProUGUI questionType;
    [SerializeField] private TextMeshProUGUI quesiotnBody;
    [SerializeField] private Image firstAttemptCorrect;
    [SerializeField] private Image firstAttemptWrong;
    [SerializeField] private TextMeshProUGUI timeElapsed;
    [SerializeField] private Image correct;
    [SerializeField] private Image wrong;
    [SerializeField] private Image wrongBar;
    [SerializeField] private Image correctBar;
    


    [SerializeField] private QuestionBuilder questionBuilder;

   
    public void AssignValues(int questionNumber, QuestionRecord questionRecord, Question question) {

        this.questionNumber.text = questionNumber.ToString();
        this.questionType.text = question.questionType;
        this.quesiotnBody.text = question.body.text;

        foreach (QuestionTimer questionTimer in questionBuilder.GetTimerList())
        {
            if (questionTimer.question == question)
            {
                int minutes = (int)questionTimer.timeTook / 60;
                int second = (int)questionTimer.timeTook - 60 * minutes;
                string seconds;
                if (second < 10) { seconds = "0" + second; }
                else { seconds = second.ToString(); }
                timeElapsed.text = "" + minutes + ":" + seconds;
            }
        }


        if (questionRecord.answers.Count == 1 && questionRecord.IsCorret)
        {
            firstAttemptCorrect.gameObject.SetActive(true);
            firstAttemptWrong.gameObject.SetActive(false);
        }
        else {
            firstAttemptCorrect.gameObject.SetActive(false);
            firstAttemptWrong.gameObject.SetActive(true);
        }

        if (questionRecord.IsCorret)
        {
            marksGained.text= (question.marks).ToString();
            correct.gameObject.SetActive(true);
            wrong.gameObject.SetActive(false);
        }
        else {
            marksGained.text = "0" ;
            correct.gameObject.SetActive(false);
            wrong.gameObject.SetActive(true);
        }

        string currentSolutions = "";
        if(question.questionType != "rangeAnswer") { 
        foreach(int answer in question.solutions) {
            if(question.solutions.Length - 1 == Array.IndexOf(question.solutions, answer))
                {
                    currentSolutions += answer;
                }
                else { currentSolutions += answer + ", " ; }
             }
        }
        else { currentSolutions = "Between " + question.solutions.Min() + " and " + question.solutions.Max(); }

        string currentAnswers = "";
        foreach(List<int> answer in questionRecord.answers)
        {
            if (questionRecord.answers.Count - 1 == questionRecord.answers.IndexOf(answer) && questionRecord.IsCorret) { 
            currentAnswers += "<color=green>"+"{" ;
            foreach(int x in answer) {
                if(answer.Count-1 == answer.IndexOf(x)) {
                    currentAnswers += x; } 
                else {
                    currentAnswers += x + ", "; }

            }
            currentAnswers += "}"+ "</color>";
            }
            else {

                currentAnswers += "{";
                foreach (int x in answer)
                {
                    if (answer.Count - 1 == answer.IndexOf(x))
                    {
                        currentAnswers += x;
                    }
                    else
                    {
                        currentAnswers += x + ", ";
                    }

                }
                currentAnswers += "} ";
            }
        }

        this.solutions.text = currentSolutions;
        this.answers.text = currentAnswers;

        if(!questionRecord.IsCorret)
        {
            correctBar.fillAmount = 0f;
        }
        else { correctBar.fillAmount = (float) 1 / questionRecord.answers.Count ; }

        if (questionRecord.answers.Count == 0 )
        {
            wrongBar.fillAmount = 0f;
        } else if (questionRecord.IsCorret) { wrongBar.fillAmount = (1f - (float) 1f / questionRecord.answers.Count); }
        else { wrongBar.fillAmount = 1f; }

        Debug.Log("wow1");
    }

    public void AssignValuesNoAnswer(int questionNumber, Question question) {

        this.questionNumber.text = questionNumber.ToString();
        this.questionType.text = question.questionType;
        this.quesiotnBody.text = question.body.text;
        this.marksGained.text = "0";
        this.answers.text = "";

       
        foreach (QuestionTimer questionTimer in questionBuilder.GetTimerList())
        {
            if (questionTimer.question == question)
            {
                int minutes = (int)questionTimer.timeTook / 60;
                int second = (int) questionTimer.timeTook - 60 * minutes;
                string seconds;
                if(second < 10) { seconds = "0" + second; }
                else { seconds = second.ToString(); }
                timeElapsed.text = "" + minutes + ":" + seconds;

            }

        }

        firstAttemptCorrect.gameObject.SetActive(false);
        correct.gameObject.SetActive(false);

        string currentSolutions = "";
        if (question.questionType != "rangeAnswer")
        {
            foreach (int answer in question.solutions)
            {
                if (question.solutions.Length - 1 == Array.IndexOf(question.solutions, answer))
                {
                    currentSolutions += answer;
                }
                else { currentSolutions += answer + ", "; }
            }
        }
        else { currentSolutions = "Between " + question.solutions.Min() + " and " + question.solutions.Max(); }
        this.solutions.text = currentSolutions;

        correctBar.fillAmount = 0f;
        wrongBar.fillAmount = 0f;
        this.answers.text ="<color=yellow>"+ "Unattempted"+ "</color>";

        //Debug.Log("wow"); 
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using UnityEngine.UI;

public class ContentAuto : MonoBehaviour
{
    [SerializeField] private TextMeshProUGUI timer;
    [SerializeField] private TextMeshProUGUI correctAnswers;
    [SerializeField] private TextMeshProUGUI wrongAnswers;
    [SerializeField] private TextMeshProUGUI recycledPoints;

    [SerializeField] private Transform QuestionLayout;
    [SerializeField] private ExitDoorOpen doorOpen;
    [SerializeField] private ContentAppear contentAppear;


    [SerializeField] private PrintingTable printingTable;
    [SerializeField] private RecycleBin recycleBin;
    [SerializeField] private EquationGenerator equationGenerator;

    private string examTimer = ""; 


    private List<QuestionAnswer> questionList = new List<QuestionAnswer>();
    private void Awake()
    {
        contentAppear.OnResultsActivated += ContentAppear_OnResultsActivated;
       
    }

    private void ContentAppear_OnResultsActivated(object sender, System.EventArgs e)
    {
        questionList = printingTable.GetQuestionAnswers();
        examTimer = equationGenerator.getExamTimer();
        HandleAnalysis();
        HandleResults();

    }


    private void HandleResults()
    {
        QuestionLayout.gameObject.SetActive(true);
        foreach (QuestionAnswer questionAnswer in questionList)
        {
            Transform questionAnswerTransform = Instantiate(QuestionLayout, this.transform);
            questionAnswerTransform.GetComponent<QuestionLayoutAuto>().AssignFields(questionAnswer);
        }
        QuestionLayout.gameObject.SetActive(false);
    }


    private void HandleAnalysis()
    {
        timer.text = examTimer;
        int correctCounter = 0;
        foreach(QuestionAnswer index in questionList)
        {
            if (index.isCorrect) { correctCounter++; }
        }
        correctAnswers.text = correctCounter+"/"+questionList.Count;

        int wrongCounter = 0;
        foreach (QuestionAnswer index in questionList)
        {
            if (!index.isCorrect) { wrongCounter++; }
        }
        wrongAnswers.text = wrongCounter + "/" + questionList.Count;


         recycledPoints.text = recycleBin.GetCurrentPoints().ToString();

    }



    
    
}

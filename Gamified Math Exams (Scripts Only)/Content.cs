using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

public class Content : MonoBehaviour
{
    [SerializeField] private PrintingTableImport printingTable;
    [SerializeField] private QuestionBuilder questionBuilder;
    [SerializeField] private ExitDoorOpen exitDoorOpen;
    [SerializeField] private Results results;
    [SerializeField] private JSONReader JsonReader;
    [SerializeField] private Transform overAlInfo;
    [SerializeField] private Transform studentGrade;
    [SerializeField] private Transform UploadFailed;


    [SerializeField] private GameObject ImageToPdf;

    [SerializeField] private Transform questionLayout;

    [SerializeField] private TMP_Dropdown dropdown;
    [SerializeField] private Button unityButton;
    [SerializeField] private TextMeshProUGUI marks;
    [SerializeField] private TextMeshProUGUI corrctAnswers;
    [SerializeField] private TextMeshProUGUI wrongAnswers;
    [SerializeField] private TextMeshProUGUI unattemptedQuestions;
    [SerializeField] private TextMeshProUGUI attemptedQuestions;
    [SerializeField] private TextMeshProUGUI correctFirstAttemptQuestions;
    [SerializeField] private TextMeshProUGUI timeElapsed;

    [SerializeField] private Image answeredQuestionsGraph;
    [SerializeField] private Image correctAnswersGraph;
    [SerializeField] private Image wrongAnswersGraph;
    [SerializeField] private Image correctFirstAttemptGraph;

    [SerializeField] private TextMeshProUGUI studentMarkText;
    [SerializeField] private TextMeshProUGUI studentGradetext;
    [SerializeField] private TextMeshProUGUI studentwrongAnswers;
    [SerializeField] private TextMeshProUGUI studentAttemptedQuestions;
    [SerializeField] private TextMeshProUGUI studentCorrectAnswers;
    [SerializeField] private TextMeshProUGUI studentTimeElapsed;

    private float attemptedQuestionsFraction;
    private float correctAnswersQuestionsFraction;
    private float wrongAnswersQuestionFraction;
    private float FirstattemptCorrectQuestionFraction;
    private int studentMarks;

    private List<QuestionRecord> questionAnswersList = new List<QuestionRecord>();
    private List<Question> questionList = new List<Question>();


    private void Awake()
    {
        results.OnGenrateData += Results_OnGenrateData;
        
    }

    private void Start()
    {
        dropdown.onValueChanged.AddListener(delegate {
            DropdownChanged(dropdown);
        });
        Debug.Log(JsonReader.questionsLoad);
        studentGradetext.text = "reset";
    }

    public void resetMark()
    {
        studentMarkText.text = "reset";
    }

    private void onButtonClick()
    {
        ImageToPdf.GetComponent<GeneratePdf>().GeneratePDF();
    }

    private void DropdownChanged(TMP_Dropdown dropdown) {
    if(dropdown.value == 0) {
            foreach (Transform transform in transform) {
                if (transform == studentGrade) { transform.gameObject.SetActive(true); }
                else { transform.gameObject.SetActive(false); }
                
            }
        }
    else {
            foreach (Transform transform in transform)
            {
                if(transform == studentGrade | transform ==  questionLayout) { transform.gameObject.SetActive(false); }
                else { transform.gameObject.SetActive(true); }
            }

        }
    
    }

    public TMP_Dropdown GetDropdown()
    {
        return dropdown;
    }
    private void Results_OnGenrateData(object sender, System.EventArgs e)
    {
        questionAnswersList = printingTable.GetQuestionRecords();
        questionList = JsonReader.GetQuestionsIntact();
        HandleAnalysis();
        //CheckFailure();
        
    }

    private void CheckFailure()
    {
        if (!JsonReader.questionsLoad)
        {
            UploadFailed.gameObject.SetActive(true);
        }
        else { UploadFailed.gameObject.SetActive(false); }

        if (!JsonReader.questionsLoad)
        {
            foreach (Transform child in transform)
            { child.gameObject.SetActive(false); }
        }

    }
    private void HandleAnalysis()
    {
        if (JsonReader.questionsLoad)
        {
            AnalyseMarks();
            AnalyseCorrectAnswers();
            AnalyseWrongAnswers();
            AnalyseUnattemptedQuestions();
            AnalyseattemptedQuestions();
            AnalyseCorrectFirstAttemptQuestions();
            AnalyseTimeElapsed();
            DrawPieCharts();
            HandleQuestionCreation();
            DropdownChanged(dropdown);
            AnalyseGrade();
            
        }
        CheckFailure();
    }


    public List<string> GetStudentGradeInfo()
    {
        if (questionBuilder.GetIsGameOver())
        {
            List<string> info = new List<string>();
            info.Add("Grade " + studentGradetext.text);
            info.Add("Marks " + studentMarkText.text);
            info.Add("Time Took " + timeElapsed.text);
            info.Add("Correct Answers " + corrctAnswers.text);
            info.Add("Wrong Answers " + wrongAnswers.text);
            info.Add("Attempted QUestions " + attemptedQuestions.text);
            Debug.Log("returnedthis");
            return info;
        }
        else return null;
    }


    private void DrawPieCharts()
    {
        answeredQuestionsGraph.fillAmount = attemptedQuestionsFraction;
        correctAnswersGraph.fillAmount = correctAnswersQuestionsFraction;
        wrongAnswersGraph.fillAmount = wrongAnswersQuestionFraction;
        correctFirstAttemptGraph.fillAmount = FirstattemptCorrectQuestionFraction;
        
    }


    private void AnalyseGrade()
    {
        foreach (gradeBoundaries gradeBoundary in JsonReader.GetGradeBoundaries())
        {
            Debug.Log(gradeBoundary.gradeBorders[0]+","+gradeBoundary.gradeBorders[1]);
        }
            foreach (gradeBoundaries gradeBoundary in JsonReader.GetGradeBoundaries())
        {
            if (studentMarks >= gradeBoundary.gradeBorders[0] && studentMarks <= gradeBoundary.gradeBorders[1])
            {
                Debug.Log(gradeBoundary.grade);
                studentGradetext.text = gradeBoundary.grade;
                if (JsonReader.GetGradeBoundaries().IndexOf(gradeBoundary) == 0) {
                    studentGradetext.color = Color.green;
                   }
                else if (JsonReader.GetGradeBoundaries().IndexOf(gradeBoundary) == 1)
                {
                    studentGradetext.color = Color.cyan;
                }
                else if (JsonReader.GetGradeBoundaries().IndexOf(gradeBoundary) == JsonReader.GetGradeBoundaries().Count - 1)
                {
                    studentGradetext.color = Color.red;
                }
                else { studentGradetext.color = Color.white; }

            }

        }

       
    }

    private void AnalyseTimeElapsed()
    {
        float examDuration = (JsonReader.GetExamTime() * 60 - questionBuilder.TimeTookForExam());
        string currentExamTimer;
        if (examDuration > 0)
        {
            int minutes = (int)examDuration / 60;
            int seconds = (int)examDuration % 60;
            string currentMinutes;
            string currentSeconds;
            if (minutes < 10) currentMinutes = "0" + minutes; else currentMinutes = minutes.ToString();
            if (seconds < 10) currentSeconds = "0" + seconds; else currentSeconds = seconds.ToString();
            currentExamTimer = currentMinutes + ":" + currentSeconds;
            timeElapsed.text = currentExamTimer;
            studentTimeElapsed.text = currentExamTimer;
        }
        else {
            examDuration = JsonReader.GetExamTime() * 60;
            int minutes = (int)examDuration / 60;
            int seconds = (int)examDuration % 60;
            string currentMinutes;
            string currentSeconds;
            if (minutes < 10) currentMinutes = "0" + minutes; else currentMinutes = minutes.ToString();
            if (seconds < 10) currentSeconds = "0" + seconds; else currentSeconds = seconds.ToString();
            currentExamTimer = currentMinutes + ":" + currentSeconds; ; }
            timeElapsed.text = currentExamTimer;
            studentTimeElapsed.text = currentExamTimer;
    }

    private void AnalyseCorrectFirstAttemptQuestions()
    {
        int correctQuestions = 0;
        int correctFirstAttemptQuestions = 0;

        foreach(QuestionRecord questionRecord in questionAnswersList)
        {
            if (questionRecord.IsCorret)
            {
                correctQuestions++;
            }
        }

        foreach (QuestionRecord questionRecord in questionAnswersList)
        {
            if (questionRecord.IsCorret && questionRecord.answers.Count == 1)
            {
                correctFirstAttemptQuestions++;
            }
        }
        if(correctQuestions != 0) { FirstattemptCorrectQuestionFraction = (float) correctFirstAttemptQuestions / correctQuestions; } else {  FirstattemptCorrectQuestionFraction = 0f; }
        this.correctFirstAttemptQuestions.text = correctFirstAttemptQuestions + "/" + correctQuestions;
    }


    private void AnalyseUnattemptedQuestions()
    {
        unattemptedQuestions.text = (questionList.Count - questionAnswersList.Count) + "/" + questionList.Count;
       

    }

    private void AnalyseattemptedQuestions()
    {
        attemptedQuestionsFraction =(float) questionAnswersList.Count / questionList.Count;
        attemptedQuestions.text =  questionAnswersList.Count + "/" + questionList.Count;
        studentAttemptedQuestions.text = questionAnswersList.Count + "/" + questionList.Count; 
    }


    private void AnalyseCorrectAnswers()
    {
        int CorrectQuestions = 0;

        foreach(QuestionRecord questionRecord in questionAnswersList)
        {
            if (questionRecord.IsCorret)
            {
                CorrectQuestions++;
            }

        }
        correctAnswersQuestionsFraction =(float) CorrectQuestions / questionList.Count;
        corrctAnswers.text = CorrectQuestions + "/" + questionList.Count;
        studentCorrectAnswers.text = CorrectQuestions + "/" + questionList.Count;
    }

    private void AnalyseWrongAnswers()
    {
        int WrongQuestions = 0;

        foreach (QuestionRecord questionRecord in questionAnswersList)
        {
            if (!questionRecord.IsCorret)
            {
                WrongQuestions++;
            }

        }
        wrongAnswersQuestionFraction = (float) WrongQuestions / questionList.Count;
        wrongAnswers.text = WrongQuestions + "/" + questionList.Count;
        studentwrongAnswers .text= WrongQuestions + "/" + questionList.Count; 
    }
    private void AnalyseMarks() {
        int totalMarks = 0;
        int studentMarks = 0;
        foreach (Question question in questionList)
        {
            totalMarks += question.marks;
        }

        foreach (QuestionRecord questionRecord in questionAnswersList)
        {
            if (questionRecord.IsCorret)
            {
                studentMarks += questionRecord.question.marks;
            }

        }

        marks.text = studentMarks + "/" + totalMarks;
         this.studentMarks = studentMarks;
        studentMarkText.text = studentMarks + "/" + totalMarks;
    }


    private void HandleQuestionCreation()
    {
        questionLayout.gameObject.SetActive(true);

        foreach (Transform child in transform)
        {
            if (child != overAlInfo)
            {
                if( child != questionLayout)
                {
                    if (child != studentGrade)
                    {
                        Destroy(child.gameObject);
                    }
                }
            }
        }

        foreach (Question question in questionList)
        {
            bool questionAnswered = false;
            foreach(QuestionRecord questionRecord in questionAnswersList)
            {
                if(questionRecord.question == question)
                {
                    questionAnswered = true;
                    Transform questionTransform = Instantiate(questionLayout, transform);
                    questionTransform.GetComponent<QuestionLayout>().AssignValues(questionList.IndexOf(question)+1, questionRecord, question);
                 
                }
            }
            if (!questionAnswered) { Transform questionTransform = Instantiate(questionLayout, transform);
                questionTransform.GetComponent<QuestionLayout>().AssignValuesNoAnswer((questionList.IndexOf(question)+1),question);
            }
        }

        questionLayout.gameObject.SetActive(false);
       // CheckFailure();
    }



    
}

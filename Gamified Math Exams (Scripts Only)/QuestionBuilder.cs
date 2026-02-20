using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using Unity.IO;
using System;
using UnityEngine.Windows.Speech;
using System.Speech.Synthesis;




public class QuestionBuilder: MonoBehaviour
{
    private Question currentQuestion;
    private List<QuestionTimer> questionTimerList = new List<QuestionTimer>();
    [SerializeField] private TextMeshProUGUI questionText;
    private int currentQuestionNumber;
    [SerializeField] private TextMeshProUGUI marksAndQuestionNumber;
    private List<Question> questionList =  new List<Question>();
    private List<Question> intactQuestionList = new List<Question>();
    [SerializeField] private JSONReader JsonReader;
    [SerializeField] private TextMeshProUGUI examTimer;
    private float timer = 0f;
    private float maxTimer = 5f;
    private int counter = 0;
    [SerializeField] private NextQuestionDesk next;
    [SerializeField] private PrevQuestionDesk prev;
    [SerializeField] private ImageGenerator imageGenerator;
    [SerializeField] private MovementSystem movementSystem;
    private bool isGameOver = false;
    private float currentTimer;
    private float examDuration;
    private bool isTimerPaused;
    [SerializeField] private ExitDoorOpen doorOpen;
    System.Speech.Synthesis.SpeechSynthesizer synthesizer = new System.Speech.Synthesis.SpeechSynthesizer();


    private void Start()
       {
        synthesizer = new SpeechSynthesizer();
        isGameOver = false;
        TriggerVisuals(false);
        next.OnGoToNextQuestion += Next_OnGoToNextQuestion; 
        prev.OnGoToPrevQuestion += Prev_OnGoToPrevQuestion;
        Player.Instance.OntrafficPaused += Instance_OntrafficPaused;
       
       }

    private void Instance_OntrafficPaused(object sender, Player.OnTrafficPausedEventArgs e)
    {
        this.isTimerPaused = e.isPaused;
    }

    
    public void DeleteQuestion()
    {
        
        questionList.RemoveAt(currentQuestionNumber);
        if (questionList.Count != 0 ) {
            if (currentQuestionNumber >= questionList.Count) {
                currentQuestionNumber = 0;
            }
            ShowQeustion(currentQuestionNumber);
        }
       else
        {
            currentQuestion = null;
            TriggerVisuals(false);
        }
    }

    private void Prev_OnGoToPrevQuestion(object sender, EventArgs e)
    {
        if (questionList.Count != 0) { 
        if (currentQuestionNumber == 0)
        {
            currentQuestionNumber = questionList.Count - 1;
        }
        else
        {
            currentQuestionNumber--;
        }
        ShowQeustion(currentQuestionNumber);
        }
    }

    private void Next_OnGoToNextQuestion(object sender, EventArgs e)
    {
        GoNextQuestion();
    }

    private void GoNextQuestion()
    {
    if (questionList.Count != 0)
        {
            if (currentQuestionNumber + 1 >= questionList.Count)
            {
                currentQuestionNumber = 0;
            }
            else
            {
                currentQuestionNumber++;
            }
            ShowQeustion(currentQuestionNumber);
        }
    }
    private void Update()
    {
        if(JsonReader.GetQuestions().Count == 0) {

            TriggerVisuals(false);
        }
        else if (timer > maxTimer && counter == 0 )
        {
            GetData();
            counter++;
            //Debug.Log("called");
        }
        else if( timer < maxTimer && counter == 0 )

        {
            timer += Time.deltaTime;
           // Debug.Log(timer);
        }
       
        if(currentQuestion != null && !isGameOver)
        {
            SetTimer(Time.deltaTime);
            
            HandleExamTime();
           
        }

    }

    private void ConvertExamTimer()
    {
        examDuration = JsonReader.GetExamTime() * 60; 
    }
    private void HandleExamTime()
    {
        if (!isTimerPaused)
        {
            examDuration -= Time.deltaTime;
            if (examDuration > 0)
            {
                int minutes = (int)examDuration / 60;
                int seconds = (int)examDuration % 60;
                string currentMinutes;
                string currentSeconds;
                string currentExamTimer;
                if (minutes < 10) currentMinutes = "0" + minutes; else currentMinutes = minutes.ToString();
                if (seconds < 10) currentSeconds = "0" + seconds; else currentSeconds = seconds.ToString();
                currentExamTimer = currentMinutes + ":" + currentSeconds;

                if (minutes < 5) examTimer.text = "<color=red>" + currentExamTimer + "</color>";
                else if (minutes > 40) examTimer.text = "<color=green>" + currentExamTimer + "</color>";
                else examTimer.text = "<color=white>" + currentExamTimer + "</color>";
            }
            else
            {

                doorOpen.OpenDoor();
            }
        }
        
    }


    
    private void TriggerVisuals(bool state)
    {

        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(state);
        }

    }

    public float TimeTookForExam()
    {
        return examDuration;
    }
    private void GetData()
    {
        Debug.Log("hello");
        questionList = JsonReader.GetQuestions();
        intactQuestionList = JsonReader.GetQuestions();
        if (questionList.Count != 0)
        {
            TriggerVisuals(true);
            this.transform.gameObject.SetActive(true);
            currentQuestionNumber = 0;
            currentQuestion = questionList[0];
            ShowQeustion(currentQuestionNumber);
            ConvertExamTimer();
        }
    }

    public void EndGame()
    {
        isGameOver = true;
    }
    public bool GetIsGameOver()
    {
        return isGameOver;
    }
    
    public Question GetCurrentQuestion()
    {
        /*foreach(int x in currentQuestion.solutions)
        {
            Debug.Log(x);
        }*/
        return currentQuestion;
    }

  
    public void SetTimer(float addedTime)
    {
        //Debug.Log("here");
        if (questionTimerList.Count != 0)
        {
            bool found = false;
            foreach (QuestionTimer questionTimer in questionTimerList)
            {
                if (questionTimer.question == currentQuestion)
                {
                    questionTimer.timeTook += addedTime;
                    //Debug.Log(questionTimer.timeTook);
                    found = true;
                }
            }
            if (found == false)
            {
                
                QuestionTimer questionTimer = new QuestionTimer();
                questionTimer.question = currentQuestion;
                questionTimer.timeTook = 0f;
                questionTimer.timeTook += addedTime;
                questionTimerList.Add(questionTimer);
            }

        }
        else
        {  
            QuestionTimer questionTimer = new QuestionTimer();
            questionTimer.question = currentQuestion;
            questionTimer.timeTook = 0f;
            questionTimer.timeTook += addedTime;
            questionTimerList.Add(questionTimer);
        }
       

    }
    private void ShowQeustion(int index)
    {
        currentQuestion = questionList[index];
        
        if (currentQuestion.body.text != null)
        {
            questionText.gameObject.SetActive(true);
            questionText.text = currentQuestion.body.text;
            marksAndQuestionNumber.text = currentQuestion.marks + " marks" + "\n" + "QuestionNo." + (currentQuestionNumber + 1) + "/" + questionList.Count;
        }
        if (currentQuestion.body.image.Length != 0)
        {
            List<string> imageUrls = new List<string>();
            foreach (string url in currentQuestion.body.image)
            {
                imageUrls.Add(url);
            }
            imageGenerator.gameObject.SetActive(true);
            imageGenerator.GenratePictures(imageUrls);
        }
        else { imageGenerator.gameObject.SetActive(false); }
    }

    public List<Question> GetIntactQuestions()
    { return intactQuestionList; }



    public List<QuestionTimer> GetTimerList()
    {
        return questionTimerList;
    }
}

public class QuestionTimer
{
    public Question question;
    public float timeTook;
}
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using TMPro;
public class EquationGenerator : MonoBehaviour
{
    private List<string> operators = new List<string> { "+", "*", "-", "/" };
    [SerializeField] private TextMeshProUGUI examTimer;
    private string currentExamTimer;
    float currenttimer=5f;
    float maxTimer = 8f;
    private int rangeMax = 20;
    int maxEquationNumber = 5;
    private List<string> questionList = new List<string>();
    private float examDuration = 210f;
    public event EventHandler<OnEquationAddedEventArgs> OnEquationAdded;
    [SerializeField] private PrintingTable printingTable;
    [SerializeField] private ExitDoorOpen doorOpen;
    public class OnEquationAddedEventArgs {
        public string addedString;
    }
    private void Start()
    {
        printingTable.OnAnswerSubmitted += PrintingTable_OnAnswerSubmitted;
        doorOpen.OnGameExitted += DoorOpen_OnGameExitted;
    }

    private void DoorOpen_OnGameExitted(object sender, EventArgs e)
    {
        examTimer.gameObject.SetActive(false);
    }

    private void PrintingTable_OnAnswerSubmitted(object sender, EventArgs e)
    {
        SetDifficulty();
    }

    private void Update()
    {
        GenerateQuestion();
        HandleExamTime();
    }

    public string getExamTimer()
    {
        return currentExamTimer;
    }
    private void SetDifficulty()
    {
        int count = 0;
        foreach(QuestionAnswer current in printingTable.GetQuestionAnswers())
        {
            if(current.isCorrect == true) { count++ ; }
        }

        switch (count){
            
            case (> 20):
                rangeMax = 100;
                break;
            case (> 15):
                rangeMax = 80;
                break;
            case (> 10):
                rangeMax = 60;
                break;
            case (> 5):
                rangeMax = 40;
                break;

        }
        float addFails = 0;
        float minusFails = 0;
        float divisionFails = 0;
        float multiplicationFails = 0;
        float failCount = 0;

        foreach (QuestionAnswer current in printingTable.GetQuestionAnswers())
        {
            if (current.isCorrect == false && current.question.Contains("+")) { addFails++; failCount++; }
            if (current.isCorrect == false && current.question.Contains("-")) { minusFails++; failCount++; }
            if (current.isCorrect == false && current.question.Contains("*")) { divisionFails++; failCount++; }
            if (current.isCorrect == false && current.question.Contains("/")) { multiplicationFails++; failCount++; }
        }
        
        if (failCount >= 3 && failCount / printingTable.GetQuestionAnswers().Count >= 1)
        {
            if ((addFails / failCount) >= 0.4f) {
                int n = operators.RemoveAll(item => item.Contains("+"));
                operators.Add("+"); operators.Add("+");
            }
            if ((addFails / failCount) >= 0.6f) {
                int n = operators.RemoveAll(item => item.Contains("+"));
                operators.Add("+"); operators.Add("+"); operators.Add("+");
            }
            if ((addFails / failCount) >= 0.8f)
            {
                int n = operators.RemoveAll(item => item.Contains("+"));
                operators.Add("+"); operators.Add("+"); operators.Add("+");
                operators.Add("+");
            }
            else
            {
                int n = operators.RemoveAll(item => item.Contains("+"));
                operators.Add("+");
            }


            if ((minusFails / failCount) >= 0.4f)
            {
                int n = operators.RemoveAll(item => item.Contains("-"));
                operators.Add("-"); operators.Add("-");
            }
            if ((minusFails / failCount) >= 0.6f)
            {
                int n = operators.RemoveAll(item => item.Contains("-"));
                operators.Add("-"); operators.Add("-"); operators.Add("-");
            }
            if ((minusFails / failCount) >= 0.8f)
            {
                int n = operators.RemoveAll(item => item.Contains("-"));
                operators.Add("-"); operators.Add("-"); operators.Add("-");
                operators.Add("-");
            }
            else
            {
                int n = operators.RemoveAll(item => item.Contains("-"));
                operators.Add("-");
            }

            if ((multiplicationFails / failCount) >= 0.4f)
            {
                int n = operators.RemoveAll(item => item.Contains("*"));
                operators.Add("*"); operators.Add("*");
            }
            if ((multiplicationFails / failCount) >= 0.6f)
            {
                int n = operators.RemoveAll(item => item.Contains("*"));
                operators.Add("*"); operators.Add("*"); operators.Add("*");
            }
            if ((multiplicationFails / failCount) >= 0.8f)
            {
                int n = operators.RemoveAll(item => item.Contains("*"));
                operators.Add("*"); operators.Add("*"); operators.Add("*");
                operators.Add("*");
            }
            else
            {
                int n = operators.RemoveAll(item => item.Contains("*"));
                operators.Add("*");
            }




            if ((divisionFails / failCount) >= 0.4f)
            {
                int n = operators.RemoveAll(item => item.Contains("/"));
                operators.Add("/"); operators.Add("/");
            }
            if ((divisionFails / failCount) >= 0.6f)
            {
                int n = operators.RemoveAll(item => item.Contains("/"));
                operators.Add("/"); operators.Add("/"); operators.Add("/");
            }
            if ((divisionFails / failCount) >= 0.8f)
            {
                int n = operators.RemoveAll(item => item.Contains("/"));
                operators.Add("/"); operators.Add("/"); operators.Add("/");
                operators.Add("/");
            }
            else
            {
                int n = operators.RemoveAll(item => item.Contains("/"));
                operators.Add("/");
            }

        }
        //Debug.Log(operators);
    }
    private void GenerateQuestion() {
        if (questionList.Count < maxEquationNumber)
        {

            if (currenttimer > maxTimer)
            {
                currenttimer = 0;
                System.Random random = new System.Random();
                string operation = operators[random.Next(0, operators.Count)];
                int FirstNumber = random.Next(2, rangeMax); // Generates a random integer between minValue (inclusive) and maxValue (exclusive)
                int SecondNumber;
                if (operation == "/" || operation == "-")
                {
                    if (operation == "/") { SecondNumber = random.Next(1, FirstNumber/2); } 
                    else {
                        SecondNumber = random.Next(1, FirstNumber);
                    } 
                }
                else
                {
                    SecondNumber = random.Next(1, rangeMax);
                }
                string currentquestion = FirstNumber + "  " + operation + "  " + SecondNumber;
                questionList.Add(currentquestion);
                OnEquationAdded?.Invoke(this, new OnEquationAddedEventArgs { addedString = currentquestion });
                //Debug.Log(FirstNumber + operation + SecondNumber);

            }
            else { currenttimer += Time.deltaTime; }
        }
    }

    private void HandleExamTime()
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
                this.currentExamTimer = currentExamTimer;
                if (minutes < 1) examTimer.text = "<color=red>" + currentExamTimer + "</color>";
                else if (minutes >= 2) examTimer.text = "<color=green>" + currentExamTimer + "</color>";
                else examTimer.text = "<color=white>" + currentExamTimer + "</color>";
            }
            else
            {
                doorOpen.OpenDoor();
            }
      }

    public void IncreaseTimer()
    {
        examDuration += 30f;
    }

    public void DecreaseTimer()
    {
        examDuration -= 30f;
    }


    public List<string> GetQuestionList()
    {
        return questionList;
    }
}

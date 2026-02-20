using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Data;
using System.Linq;

[Serializable]
public enum PrintingProgress
{
    Idle,
    Placed,
    Typing,
    Done
}
public class PrintingTable : StationaryObject, IItemOwner
{

    [SerializeField] private EquationGenerator equationGenerator;
    private List<QuestionAnswer> questionAnswerList;
    [SerializeField] private EquationUI equationUI;
    private int currentAnswer;
    private List<int> currentIntList;
    [SerializeField] private List<CarriableItemSO> digits; 
    [SerializeField] private MovementSystem movementSystem;
    private CarriableItem currentCarriableItem;
    [SerializeField] private Transform placementPoint;
    [SerializeField] private Transform printerOutputPoint;
    [SerializeField] private CrumpledPaper crumpledPaper;
    [SerializeField] private Transform spawnPoint;
    private PrintingProgress printingProgress = PrintingProgress.Idle;
    private bool previousAnswerCorrectness;
    private float currentSolution;
    public event EventHandler<PrintingProgressChangedEventArgs> OnPrintingProgressChanged;
    public event EventHandler OnAnswerSubmitted;
    [SerializeField] private SoundManager soundManager;
    public class PrintingProgressChangedEventArgs
    {
        
        public PrintingProgress currentPrintingState;
        public float progress;
    }

    public void Awake()
    {
        questionAnswerList = new List<QuestionAnswer>();
        currentIntList = new List<int>();
        movementSystem.OnTypingChanged += Instance_OnTypingChanged;
    }

    private void Instance_OnTypingChanged(object sender, MovementSystem.OnTypingChangedEventArgs e)
    {
        if(e.progress < 1f && e.progress > 0) { 
            printingProgress = PrintingProgress.Typing; //Debug.Log(printingProgress);
            OnPrintingProgressChanged?.Invoke(this, new PrintingProgressChangedEventArgs { currentPrintingState = printingProgress, progress = e.progress });
        }
        else if (e.progress >= 1f) { 
            printingProgress = PrintingProgress.Done; Typed(); //Debug.Log(printingProgress);
            OnPrintingProgressChanged?.Invoke(this, new PrintingProgressChangedEventArgs { currentPrintingState = printingProgress, progress = e.progress });
        }
        else if(e.progress == 0 && (printingProgress != PrintingProgress.Done)) { 
            printingProgress = PrintingProgress.Placed; //Debug.Log(printingProgress);
            OnPrintingProgressChanged?.Invoke(this, new PrintingProgressChangedEventArgs { currentPrintingState = printingProgress, progress = e.progress }) ;
        }
        
    }

    public override void Interact()
    {
        if (!HasCarriableItem())
        {
            if (Player.Instance.HasCarriableItem())
            {
                if (Player.Instance.GetCarriableItem() is NotePad)
                {
                    (Player.Instance.GetCarriableItem()).SetObjectOwner(this);
                    printingProgress = PrintingProgress.Placed;
                    soundManager.PlaySuccessPlacement();
                }
                else soundManager.PlayFailInteract();
            }
            else soundManager.PlayFailInteract();
        }
        else { if (HasCarriableItem())
            {
                if (!Player.Instance.HasCarriableItem())
                {
                    GetCarriableItem().SetObjectOwner(Player.Instance);
                    printingProgress = PrintingProgress.Idle;
                    OnPrintingProgressChanged?.Invoke(this, new PrintingProgressChangedEventArgs { currentPrintingState = printingProgress, progress = 0f });
                    soundManager.PlaySuccessPlacement();
                }
                else soundManager.PlayFailInteract();
            }
            else soundManager.PlayFailInteract();
        }
    }

    private void Typed()
    {
        CreateAnswer();
        previousAnswerCorrectness = CheckAnswer();
        QuestionAnswer currentquestionAnswer = new QuestionAnswer();
        if (previousAnswerCorrectness) 
        {

            currentquestionAnswer.CorrectAnswer = currentSolution.ToString();
            currentquestionAnswer.isCorrect = previousAnswerCorrectness;
            currentquestionAnswer.providedAnswer = currentAnswer.ToString();
            currentquestionAnswer.question = equationUI.GetCurrentQuestion();
            questionAnswerList.Add(currentquestionAnswer);
            OnAnswerSubmitted?.Invoke(this, EventArgs.Empty);
            equationGenerator.IncreaseTimer();
            //Debug.Log("correct");
            equationUI.DeleteEquation();
            soundManager.PlayUploadSuccessful();
        }
        else{
            currentquestionAnswer.CorrectAnswer = currentSolution.ToString();
            currentquestionAnswer.isCorrect = previousAnswerCorrectness;
            currentquestionAnswer.providedAnswer = currentAnswer.ToString();
            currentquestionAnswer.question = equationUI.GetCurrentQuestion();
            questionAnswerList.Add(currentquestionAnswer);
            equationGenerator.DecreaseTimer();
            OnAnswerSubmitted?.Invoke(this, EventArgs.Empty);
            if (ExamType.isSingleAttempt) { equationUI.DeleteEquation(); }
            soundManager.PlayUploadFail();
            //Debug.Log("False");
        }

        foreach(QuestionAnswer index in questionAnswerList)
        {
            //Debug.Log(index.question + "\n" + index.CorrectAnswer + "\n" + index.providedAnswer +  "\n" + index.isCorrect);   
        }

        GetCarriableItem().DeleteItem();
        crumpledPaper.SpawnItemAlternateLocation(this, spawnPoint, previousAnswerCorrectness);
    }


    private void CreateAnswer()
    {
        currentIntList.Clear();
        if ((GetCarriableItem() as NotePad).GetDigitList().Count != 0)
        {
            foreach (CarriableItem item in (GetCarriableItem() as NotePad).GetDigitList())
            {

                foreach (CarriableItemSO digit in digits)
                {
                    if (item.CompareTag(digit.carriableItem.tag))
                    {
                        currentIntList.Add(int.Parse(digit.Digit));
                    }
                }

            }
            string combinedNumber = string.Join("", currentIntList.Select(i => i.ToString()));
            currentAnswer = int.Parse(combinedNumber);
            Debug.Log(currentAnswer);
        }
        else { currentAnswer = 0; }     

    }


    private bool CheckAnswer()
    {
        /*if ((GetCarriableItem() as NotePad).GetDigitList().Count != 0)
        {*/
        DataTable calculator = new DataTable();
        float solution = 0f;
        if (equationUI.GetCurrentQuestion().Contains("=")) {
            int delimiterIndex = equationUI.GetCurrentQuestion().IndexOf('=');
            string cur = equationUI.GetCurrentQuestion().Substring(0, delimiterIndex);
            solution = Convert.ToInt32(Math.Floor(Convert.ToDouble(calculator.Compute(cur, ""))));
        }
        else
        {
            solution = Convert.ToInt32(Math.Floor(Convert.ToDouble(calculator.Compute(equationUI.GetCurrentQuestion(), ""))));
        }
            currentSolution = solution;
            return solution == currentAnswer;
        /*}
        else return false;  */
    }

    public string GetAnswer()
    {
        DataTable calculator = new DataTable();
        float solution = 0f;
        if (equationUI.GetCurrentQuestion().Contains("="))
        {
            int delimiterIndex = equationUI.GetCurrentQuestion().IndexOf('=');
            string cur = equationUI.GetCurrentQuestion().Substring(0, delimiterIndex);
            solution = Convert.ToInt32((calculator.Compute(cur, "")));
        }
        else
        {
            solution = Convert.ToInt32((calculator.Compute(equationUI.GetCurrentQuestion(), "")));
        }
        return solution.ToString();
    }
   
    public PrintingProgress GetDeskState()
    {
        return printingProgress;
    }

    public bool CanType()
    { return ((!Player.Instance.HasCarriableItem()) && HasCarriableItem() && (GetCarriableItem() is NotePad));   }
    public void ClearCarriableItem()
    {
        currentCarriableItem = null;
    }

    public CarriableItem GetCarriableItem()
    {
        return currentCarriableItem;
    }

    public Transform GetSpawnPoint()
    {
        return placementPoint;
    }

    public bool HasCarriableItem()
    {
        return currentCarriableItem != null;
    }

    public void SetCarriableItem(CarriableItem carriableItem)
    {
        currentCarriableItem = carriableItem;
    }
    public bool VerifyAnswer()
    {
        return previousAnswerCorrectness;
    }

    public List<QuestionAnswer> GetQuestionAnswers()
    {
        return questionAnswerList;
    }
}
[System.Serializable]
public class QuestionAnswer
{
    public string question;
    public bool isCorrect;
    public string CorrectAnswer;
    public string providedAnswer;
}
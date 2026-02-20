using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

[Serializable]
public class PrintingTableImport : StationaryObject, IItemOwner
{
    [SerializeField] private EquationUI equationUI;
    [SerializeField] private List<QuestionRecord> questionRecords = new List<QuestionRecord>();
    private int currentAnswer;
    private List<int> currentIntList;
    private List<String> currentStringList;
    [SerializeField] private List<CarriableItemSO> digits; 
    [SerializeField] private MovementSystem movementSystem;
    private CarriableItem currentCarriableItem;
    [SerializeField] private Transform placementPoint;
    [SerializeField] private Transform printerOutputPoint;
    [SerializeField] private CrumpledPaper crumpledPaper;
    [SerializeField] private Transform spawnPoint;
    [SerializeField] private QuestionBuilder questionBuiler;
    private Question currentQuestion;
    private PrintingProgress printingProgress = PrintingProgress.Idle;
    private bool previousAnswerCorrectness;
    public event EventHandler<PrintingProgressChangedEventArgs> OnPrintingProgressChanged;
    private bool singleAttempt;
    [SerializeField] private SoundManager soundManager;
    
    public class PrintingProgressChangedEventArgs
    {
        public PrintingProgress currentPrintingState;
        public float progress;
    }

    public void Awake()
    {
        currentIntList = new List<int>();
        currentStringList = new List<string>();
        movementSystem.OnTypingChanged += Instance_OnTypingChanged;
    }

    private void Start()
    {
        Scene currentScene = SceneManager.GetActiveScene();
        string x = "ClassRoomImportSingleAttempt";
        singleAttempt = (currentScene.name == x);
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
        else { if (HasCarriableItem()) {
                if(!Player.Instance.HasCarriableItem()) {
                    GetCarriableItem().SetObjectOwner(Player.Instance);
                    printingProgress = PrintingProgress.Idle;
                    OnPrintingProgressChanged?.Invoke(this, new PrintingProgressChangedEventArgs { currentPrintingState = printingProgress, progress = 0f });
                    soundManager.PlaySuccessPlacement();
                }
                else soundManager.PlayFailInteract();
            }
            else soundManager.PlayFailInteract();
        }
        Scene currentScene = SceneManager.GetActiveScene();
        
    }

    private void Typed()
    {
        currentQuestion = questionBuiler.GetCurrentQuestion();
        createAnswer();
        if (previousAnswerCorrectness)
        {
            Debug.Log("correct");
            questionBuiler.DeleteQuestion();
            soundManager.PlayUploadSuccessful();
            
        }
        else
        {
            Debug.Log("False");
            soundManager.PlayUploadFail();
            if (ExamType.isSingleAttempt) { questionBuiler.DeleteQuestion(); }
        }
        GetCarriableItem().DeleteItem();
        crumpledPaper.SpawnItemAlternateLocation(this, spawnPoint, previousAnswerCorrectness);
    }


    private void createAnswer()
    {
        if (currentQuestion != null)
        {
            switch (currentQuestion.questionType)
            {
                case "numberSequence":
                    //Debug.Log("1");
                    previousAnswerCorrectness = CheckAnswersRegularly(ProcessNotePadData(), currentQuestion.isOrderImportant);
                    SetRecord(ProcessNotePadData(),previousAnswerCorrectness);
                    break;
                case "rangeAnswer":
                    // Debug.Log("2");
                    previousAnswerCorrectness = CheckAnswerForRange(ProcessNotePadData(), currentQuestion.isOrderImportant);
                    SetRecord(ProcessNotePadData(), previousAnswerCorrectness);
                    break;
                case "multipleChoice":
                    //Debug.Log("3");
                    previousAnswerCorrectness = CheckAnswersRegularly(ProcessNotePadData(), currentQuestion.isOrderImportant);
                    SetRecord(ProcessNotePadData(), previousAnswerCorrectness);
                    break;
                case "singleInput":
                    //Debug.Log("4");
                    previousAnswerCorrectness = CheckAnswersRegularly(ProcessNotePadData(), currentQuestion.isOrderImportant);
                    SetRecord(ProcessNotePadData(), previousAnswerCorrectness);
                    break;
                default:
                    Debug.LogError("QuestionTypeNotSupported");
                    previousAnswerCorrectness = false;
                    break;
            }
        }
        else
        {
            previousAnswerCorrectness = false;
        }
    }

    private bool CheckAnswerForRange(List<int> answers, bool isOrdered) {
        int maxNumber = currentQuestion.solutions.Max();
        int minNumber = currentQuestion.solutions.Min();
        bool isCorrect = true;
        if (isOrdered == false) {
           
            foreach(int x in answers)
            {
                if (x < minNumber | x > maxNumber)
                {
                    isCorrect = false;
                }
            }
            
        } else {
            
            foreach (int x in answers)
            {
                if (x < minNumber | x > maxNumber)
                {
                    isCorrect = false;
                }
            }
            if (!IsAscending(answers)) { isCorrect = false; }
        }
        Debug.Log(isCorrect);
        return isCorrect;
    }
    private bool CheckAnswersRegularly(List<int> answers, bool isOrdered)
    {
        if(isOrdered == false)
        {
            HashSet<int> answerSet = new HashSet<int>();
            HashSet<int> solutionSet = new HashSet<int>();

            foreach (int x in answers) { answerSet.Add(x); }
            foreach (int x in currentQuestion.solutions) { solutionSet.Add(x); }

            foreach(int x in answerSet) { Debug.Log(x); }
            foreach(int x in solutionSet) { Debug.Log(x); }
            Debug.Log(answerSet.Count == solutionSet.Count && answerSet.All(solutionSet.Contains));
            return (answerSet.Count == solutionSet.Count && answerSet.All(solutionSet.Contains));
        }
        else
        {
           
            List<int> solutionList = new List<int>();
            foreach (int x in currentQuestion.solutions) { solutionList.Add(x); }
            bool areEqual = true;

            if (solutionList.Count == answers.Count)
             {
               
                for (int i = 0; i < answers.Count; i++)
            {
                if (answers[i] != solutionList[i])
                {
                    areEqual = false;
                    break;
                }
            }
            }
            else { areEqual = false; }
            Debug.Log(areEqual);
            return areEqual;
        }

    }


    public List<QuestionRecord> GetQuestionRecords()
    {
        return questionRecords;
    }

    private List<int> ProcessNotePadData()
    {
        currentStringList.Clear();
        List<string> NotePadNumbersList = new List<string>();
        List<int> IntNotePadNumbersList = new List<int>();
        if ((GetCarriableItem() as NotePad).GetDigitList().Count != 0)
        {
            foreach (CarriableItem item in (GetCarriableItem() as NotePad).GetDigitList())
            {

                foreach (CarriableItemSO digit in digits)
                {
                    if (item.CompareTag(digit.carriableItem.tag))
                    {
                        currentStringList.Add((digit.Digit));
                        /*Debug.Log(digit.Digit);*/
                    }
                }

            }

            string combinedNumber = string.Join("", currentStringList.Select(i => i.ToString()));
            string[] currentNumbers = combinedNumber.Split(new string[] { "And" }, StringSplitOptions.None);
            

            for (int i = 0; i < currentNumbers.Length; i++)
            {
                if (currentNumbers[i] != "")
                {
                    NotePadNumbersList.Add(currentNumbers[i]);
                }
            }

            foreach(string number in NotePadNumbersList)
            {
                IntNotePadNumbersList.Add(int.Parse(number));
               // Debug.Log(int.Parse(number));
            }
            
        }

        return IntNotePadNumbersList;
        

    }


    private bool IsAscending(List<int> numbers)
    {
        for (int i = 0; i < numbers.Count - 1; i++)
        {
            if (numbers[i] > numbers[i + 1])
            {
                return false;
            }
        }
        return true;
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

    private void SetRecord(List<int> currentAnswer, bool isright)
    {
        
        if(currentQuestion != null) {
            bool isAdded = false;
        if(questionRecords.Count != 0){
            foreach(QuestionRecord questionRecord in questionRecords)
            {
                if(questionRecord.question == currentQuestion && !isAdded)
                {
                    questionRecord.answers.Add(currentAnswer);
                    questionRecord.IsCorret = isright;
                    isAdded = true;
                    foreach(List<int> x in questionRecord.answers)
                        {
                            Debug.Log("-");
                            foreach(int y in x) { Debug.Log(y); }
                        }       
                }
            }
        }
            if (!isAdded)
            {
                QuestionRecord newQuestionRecord = new QuestionRecord();
                newQuestionRecord.answers = new List<List<int>>();
                newQuestionRecord.question = currentQuestion;
                newQuestionRecord.IsCorret = isright;
                newQuestionRecord.answers.Add(currentAnswer);
                questionRecords.Add(newQuestionRecord);
                

            }

        }
    }


}


[Serializable]
public class QuestionRecord
{
    public Question question;
    public List<List<int>> answers;
    public bool IsCorret;
}

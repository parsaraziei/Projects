using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System;
using UnityEngine.UI;

public class EquationUI : MonoBehaviour
{
   [SerializeField] private TextMeshProUGUI TMPequation;
   private string currentEquation;
   private int currentEquationIndex;
   [SerializeField] private EquationGenerator equationGenerator;
   [SerializeField] private TextMeshProUGUI TMPcurrentNumber;
   [SerializeField] private NextQuestionDesk nextQuestionDesk;
   [SerializeField] private PrevQuestionDesk prevQuestionDesk;
   [SerializeField] private ExitDoorOpen openDoor;
    [SerializeField] private PrintingTable printingTable;
    private bool isShown = true;

    private void Start()
    {
        openDoor.OnGameExitted += OpenDoor_OnGameExitted;
        nextQuestionDesk.OnGoToNextQuestion += NextQuestionDesk_OnGoToNextQuestion;
        transform.gameObject.SetActive(false);
        equationGenerator.OnEquationAdded += EquationGenerator_OnEquationAdded;
        prevQuestionDesk.OnGoToPrevQuestion += PrevQuestionDesk_OnGoToPrevQuestion;
    }

    private void OpenDoor_OnGameExitted(object sender, EventArgs e)
    {
        isShown = false;
    }

    private void Update()
    {
        if (isShown)
        {
            if (equationGenerator.GetQuestionList().Count == 0)
                transform.gameObject.SetActive(false);
        }
        else
        {
            transform.gameObject.SetActive(false);
        }
    }
    private void PrevQuestionDesk_OnGoToPrevQuestion(object sender, EventArgs e)
    {
        GotoPrevEquation();
    }

    private void GotoPrevEquation() {

        if (currentEquationIndex == 0)
        {
            currentEquationIndex = equationGenerator.GetQuestionList().Count - 1;
        }
        else
        {
            currentEquationIndex--;
        }
        UpdateUI();
    }

    private void NextQuestionDesk_OnGoToNextQuestion(object sender, EventArgs e)
    {
        GotoNextEquation();
    }
    private void GotoNextEquation()
    {
        if ((currentEquationIndex + 1) >= equationGenerator.GetQuestionList().Count)
        {
            currentEquationIndex = 0;
        }
        else
        {
            currentEquationIndex++;
        }
        UpdateUI();
    }
    private void EquationGenerator_OnEquationAdded(object sender, EquationGenerator.OnEquationAddedEventArgs e)
    {
        if(currentEquation == null) { 
            transform.gameObject.SetActive(true);
            currentEquation = e.addedString;
            currentEquationIndex = 0;
        }
        
        
        UpdateUI();
    }


    private void UpdateUI()
    {
        if (currentEquation != null)
        {
            currentEquation = equationGenerator.GetQuestionList()[currentEquationIndex];
            TMPequation.text = currentEquation;
            TMPcurrentNumber.text = currentEquationIndex + 1 + "/" + equationGenerator.GetQuestionList().Count;
        }
    }

    public string GetCurrentQuestion()
    {
        return currentEquation;
    }


    public void RevealDigit()
    {
        string answer = printingTable.GetAnswer();
        if (!currentEquation.Contains("="))
        {
            if (answer.Length == 1)
            {
                string question = (equationGenerator.GetQuestionList())[currentEquationIndex];
                Debug.Log(question);
                question = question + " = " + printingTable.GetAnswer();
                equationGenerator.GetQuestionList().RemoveAt(currentEquationIndex);
                equationGenerator.GetQuestionList().Insert(currentEquationIndex, question);
                Debug.Log(answer);
            }
            else if (answer.Length > 1)
            {
                string question = (equationGenerator.GetQuestionList())[currentEquationIndex];
                Debug.Log(question);
                int ans = UnityEngine.Random.Range(0, answer.Length);
                char[] charArray = answer.ToString().ToCharArray();
                Debug.Log(answer);
                Debug.Log(ans);
                for (int i = 0; i < charArray.Length; i++)
                {
                    if (i != ans)
                    {
                        charArray[i] = '_';
                    }
                }
                question = question + "  = ";
                foreach (char x in charArray) { question += x + " "; }

                equationGenerator.GetQuestionList().RemoveAt(currentEquationIndex);
                equationGenerator.GetQuestionList().Insert(currentEquationIndex, question);
            }
        }
        else
        {
            if (currentEquation.Contains("_"))
            {
                List<int> indexes = new List<int>();
                char[] charArray = currentEquation.ToString().ToCharArray();
                int delimiterIndex = GetCurrentQuestion().IndexOf('=');
                string cur = GetCurrentQuestion().Substring(delimiterIndex + 1).Replace(" ", "");
                Debug.Log(cur);
                char[] curcharArray= cur.ToCharArray();
                for (int i = 0; i < curcharArray.Length; i++)
                {
                    if (curcharArray[i] == '_') {
                        Debug.Log(":)");
                        indexes.Add(i); }
                }
                foreach(int x in indexes) { Debug.Log(x); }
                Debug.Log(indexes.Count);
                int num = indexes[UnityEngine.Random.Range(0, indexes.Count)];
                curcharArray[num] = answer.ToCharArray()[num];
                string question = GetCurrentQuestion().Substring(0, delimiterIndex) + " =  " ;
                foreach(char letter in curcharArray)
                {
                    question += letter + " ";
                }
                equationGenerator.GetQuestionList().RemoveAt(currentEquationIndex);
                equationGenerator.GetQuestionList().Insert(currentEquationIndex, question);
            }
        }
        UpdateUI();
    }

    public void DeleteEquation() {
        string deletedEquation = currentEquation;
        //GotoNextEquation();
        equationGenerator.GetQuestionList().Remove(deletedEquation);
        if (currentEquationIndex + 1 >= equationGenerator.GetQuestionList().Count)
        {
            currentEquationIndex = 0;
            currentEquation = equationGenerator.GetQuestionList()[currentEquationIndex];
        }
        else
        { 
            currentEquation = equationGenerator.GetQuestionList()[currentEquationIndex];
        }
        UpdateUI();
        Debug.Log(deletedEquation);
        
    }
}

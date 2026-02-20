using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class QuestionLayoutAuto : MonoBehaviour
{

    [SerializeField] private TextMeshProUGUI questionText;
    [SerializeField] private TextMeshProUGUI correctAnswer;
    [SerializeField] private TextMeshProUGUI providedANswer;
    [SerializeField] private Image checkMark;
    [SerializeField] private Image xMark;
    // Start is called before the first frame update
    public void AssignFields(QuestionAnswer questionAnswer) {
        if (questionAnswer.question.Contains('=')) {
           string[] split_values = questionAnswer.question.Split('=');
            questionText.text = split_values[0]+" =  "+ "<color=green>" + split_values[1] + "</color>";
        }
        else { questionText.text = questionAnswer.question; }
        correctAnswer.text = questionAnswer.CorrectAnswer;
        providedANswer.text = questionAnswer.providedAnswer;

        if (questionAnswer.isCorrect)
        {
            checkMark.gameObject.SetActive(true);
            xMark.gameObject.SetActive(false);
        }
        else
        {
            checkMark.gameObject.SetActive(false);
            xMark.gameObject.SetActive(true);
        }
    }
}

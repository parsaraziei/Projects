using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;


public class NextQuestionDesk : StationaryObject
{
    [SerializeField] private EquationUI equationUI;
    public event EventHandler OnGoToNextQuestion;
    [SerializeField] private SoundManager soundManager;


    public override void Interact()
    {
        OnGoToNextQuestion?.Invoke(this, EventArgs.Empty);
        soundManager.PlayChangeQuestion();
    }
}

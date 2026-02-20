using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;


public class PrevQuestionDesk : StationaryObject
{
    [SerializeField] private EquationUI equationUI;
    public event EventHandler OnGoToPrevQuestion;
    [SerializeField] private SoundManager soundManager;


    public override void Interact()
    {
        OnGoToPrevQuestion?.Invoke(this, EventArgs.Empty);
        soundManager.PlayChangeQuestion();
    }
}

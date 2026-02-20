using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ATM : StationaryObject
{
    [SerializeField] private RecycleBin recycleBin;
    [SerializeField] private EquationUI equationUI;
    [SerializeField] private SoundManager soundManager;

   public override void Interact() {
        if (recycleBin.GetCurrentPoints() >= 3 && equationUI.GetCurrentQuestion()!=null)
        {
            if (!equationUI.GetCurrentQuestion().Contains("="))
            {
                recycleBin.PointDeduction();
                equationUI.RevealDigit();
                soundManager.PayPoints();
                Debug.Log("points Subtarcted");
            }
            else
            {
                if (equationUI.GetCurrentQuestion().Contains("_")) {

                    recycleBin.PointDeduction();
                    equationUI.RevealDigit();
                    soundManager.PayPoints();
                    Debug.Log("points Subtarcted");
                }
                else { soundManager.PlayFailInteract(); }
            }
        }
        else { soundManager.PlayFailInteract(); }
    }



 



}

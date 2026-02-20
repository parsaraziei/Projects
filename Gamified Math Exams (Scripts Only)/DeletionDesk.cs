using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeletionDesk : StationaryObject
{
    [SerializeField] private SoundManager soundManager;
    public override void Interact()
    {
        if (Player.Instance.HasCarriableItem() && Player.Instance.GetCarriableItem() is NotePad) {
            if(((NotePad)Player.Instance.GetCarriableItem()).GetDigitList().Count != 0)
            {
                ((NotePad)Player.Instance.GetCarriableItem()).GetDigitList()[((NotePad)Player.Instance.GetCarriableItem()).GetDigitList().Count - 1].DeleteItem();
                soundManager.PlaySuccessDelete();
            }
            else { soundManager.PlayFailDelete(); }
        }
        else { soundManager.PlayFailDelete(); }
    }
}

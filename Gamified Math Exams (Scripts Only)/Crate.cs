using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class Crate : StationaryObject
{
    public event EventHandler OnBoxInteracted;
    [SerializeField] private CarriableItem carriableItem;
    [SerializeField] private PlacementTable tempPlacementTable;
    [SerializeField] private SoundManager soundManager;
    public override void Interact()
    {

        if (!Player.Instance.HasCarriableItem())
        {
            carriableItem.SpawnItem(Player.Instance);

            OnBoxInteracted?.Invoke(this, EventArgs.Empty);
            soundManager.PlaySuccessInteract();
        }
        else if (Player.Instance.HasCarriableItem())
        {
            if (Player.Instance.GetCarriableItem() is NotePad && Player.Instance.isMultiPickUpActive())
            {
                carriableItem.SpawnItem(tempPlacementTable);
                tempPlacementTable.GetCarriableItem().SetObjectOwner((Player.Instance.GetCarriableItem() as NotePad));

                OnBoxInteracted?.Invoke(this, EventArgs.Empty);
                soundManager.PlaySuccessInteract();
            }
            else { soundManager.PlayFailInteract(); }
        }
       
    }
}

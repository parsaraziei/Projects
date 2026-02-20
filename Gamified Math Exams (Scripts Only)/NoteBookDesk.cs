using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class NoteBookDesk : StationaryObject
{
    public event EventHandler OnBoxInteracted;
    [SerializeField] private CarriableItem carriableItem;
    [SerializeField] private SoundManager soundManager;
    public override void Interact()
    {
        if (!Player.Instance.HasCarriableItem())
        {
            carriableItem.SpawnItem(Player.Instance);

            OnBoxInteracted?.Invoke(this, EventArgs.Empty);
            soundManager.PlayNotebookInteract();
        }
        else { soundManager.PlayFailInteract(); }
    }
}

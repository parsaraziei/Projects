using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlacementTable : StationaryObject, IItemOwner
{
    public CarriableItem currentCarriableItem = null;
    [SerializeField] private Transform tableSpawnPoint;
    [SerializeField] private SoundManager soundManager;
    //private Transform itemCarryingInsastnce;
    public override void Interact()
    {
        if (!HasCarriableItem())
        {
            if (Player.Instance.GetCarriableItem())
            {
                Player.Instance.GetCarriableItem().SetObjectOwner(this);
                if (soundManager != null){
                    soundManager.PlaySuccessPlacement();
                }
            }
        }
        else
        {
            if (!Player.Instance.HasCarriableItem())
            {

                GetCarriableItem().SetObjectOwner(Player.Instance);
                soundManager.PlaySuccessPlacement();
            }
            else if (Player.Instance.GetCarriableItem() is NotePad && !(GetCarriableItem() is NotePad) && !(GetCarriableItem() is CrumpledPaper))
            {
                GetCarriableItem().SetObjectOwner((Player.Instance.GetCarriableItem() as NotePad));
                soundManager.PlaySuccessPlacement();

                //GetCarriableItem().DeleteItem();
            }
            else if (GetCarriableItem() is NotePad && !(Player.Instance.GetCarriableItem() is NotePad) && !(Player.Instance.GetCarriableItem() is CrumpledPaper))
            {
                (Player.Instance.GetCarriableItem()).SetObjectOwner(GetCarriableItem() as NotePad);
                soundManager.PlaySuccessPlacement();
            }
            else {
                if (soundManager != null)
                {
                    soundManager.PlayFailInteract();
                }
            }
        }

    }

    public void SetCarriableItem(CarriableItem carriableItem)
    {
        currentCarriableItem = carriableItem;
    }

    public void ClearCarriableItem()
    {
        currentCarriableItem = null;
    }
    public bool HasCarriableItem()
    {
        return (currentCarriableItem != null);
    }
    public CarriableItem GetCarriableItem()
    {
        return currentCarriableItem;
    }
    public Transform GetSpawnPoint()
    {
        return tableSpawnPoint;
    }

   /* public CarriableItem GetCarryingItemTransform() 
    { 
        return itemCarryingInsastnce; 
    }

    public void SetCarriableItemTranssform(Transform carriableItemTransform)
    {
        itemCarryingInsastnce = carriableItemTransform;
    }*/
}

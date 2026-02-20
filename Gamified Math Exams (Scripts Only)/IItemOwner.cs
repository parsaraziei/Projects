using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public interface IItemOwner 
{
   /* public CarriableItem GetCarryingItemTransform();
    public void SetCarriableItemTranssform(Transform carriableItemTransform);*/
    public Transform GetSpawnPoint();
    public void SetCarriableItem(CarriableItem carriableItem);
    public void ClearCarriableItem();
    public bool HasCarriableItem();
    public CarriableItem GetCarriableItem();
}

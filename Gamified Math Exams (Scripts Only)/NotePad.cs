using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class NotePad: CarriableItem, IItemOwner
{
    private List<CarriableItem> digitsPickedUp;
    [SerializeField] private Transform SpawnPoint;
    public event EventHandler OnItemAdded;


    private void Awake()
    {
        digitsPickedUp = new List<CarriableItem>();
    }
    public override void SetObjectOwner(IItemOwner itemOwner)
    {

        if (this.itemOwner != null)
        {
            this.itemOwner.ClearCarriableItem();
        }
        this.itemOwner = itemOwner;
        this.itemOwner.SetCarriableItem(this);
        
        this.transform.parent = itemOwner.GetSpawnPoint();

        
        if (this.itemOwner is Player)
        {
         
            this.transform.localPosition = new Vector3(1f, 0f, 0f);
            this.transform.localScale = new Vector3(6f, 4.5f, 5.5f);
            this.transform.localRotation = Quaternion.Euler(-40f, 0f, 0f);
        }
        else if (this.itemOwner is PrintingTable)
        {
            
                this.transform.parent.localRotation = Quaternion.Euler(-90f, 0f, 0f);

        }
        else if (this.itemOwner is PrintingTableImport)
        {

            this.transform.parent.localRotation = Quaternion.Euler(-90f, 0f, 0f);

        }
        else
        {
            this.transform.localPosition = new Vector3(.9f, 0.2f, -0.3f);
            this.transform.parent.localRotation = Quaternion.Euler(40f, 0f, 0f);
            this.transform.localScale = new Vector3(6f,7f,4f);
        }

    }

    public void AddDigit(CarriableItem carriableItem) {
        digitsPickedUp.Add(carriableItem);
        OnItemAdded?.Invoke(this, EventArgs.Empty);
        /*foreach (CarriableItem item in GetDigitList()) { 
            Debug.Log(item);
        }*/
    }


    public List<CarriableItem> GetDigitList()
    {
        return digitsPickedUp; 
    }
    public override void SpawnItem(IItemOwner itemOwner)
    {
        Transform itemTransform = Instantiate(this.transform, itemOwner.GetSpawnPoint());
        itemOwner.SetCarriableItem(this);
        itemTransform.GetComponent<CarriableItem>().SetObjectOwner(itemOwner);

    }

    public Transform GetSpawnPoint()
    {
        return SpawnPoint;
    }

    public void SetCarriableItem(CarriableItem carriableItem)
    {
        AddDigit(carriableItem);
    }

    public void ClearCarriableItem()
    {
        if(digitsPickedUp.Count != 0) {
            digitsPickedUp.RemoveAt(digitsPickedUp.Count - 1);
            OnItemAdded?.Invoke(this, EventArgs.Empty); 
        }

    }

    public bool HasCarriableItem()
    {
        return digitsPickedUp.Count == 0;
    }

    public CarriableItem GetCarriableItem()
    {
        return digitsPickedUp[0];
    }
}

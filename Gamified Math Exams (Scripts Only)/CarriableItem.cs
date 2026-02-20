using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarriableItem : MonoBehaviour
{
    protected IItemOwner itemOwner;
    

    public virtual void SetObjectOwner(IItemOwner itemOwner)
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

            this.transform.localPosition = new Vector3(0f, 0f, 0f);
            //Debug.Log("here1");
        }
        else if (this.itemOwner is NotePad)
        {
            this.transform.parent.localRotation = Quaternion.Euler(50f, 0f, 0f);
           // Debug.Log("here2");
        }
        
        else
        {
            this.transform.parent.localRotation = Quaternion.Euler(50f, 0f, 0f);
            //Debug.Log("here1");
        }  

    }


    
    public virtual void SpawnItem(IItemOwner itemOwner)
    {
        Transform itemTransform = Instantiate(this.transform, itemOwner.GetSpawnPoint());
        itemOwner.SetCarriableItem(this);
        itemTransform.GetComponent<CarriableItem>().SetObjectOwner(itemOwner);
        
    }

   

    public void DeleteItem() {
        
        itemOwner.ClearCarriableItem();
        Destroy(gameObject);
    }

    public IItemOwner GetItemOwner()
    {
        return itemOwner;
    }
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrumpledPaper : CarriableItem
{
    private bool isCorrect;
    public void SetCorrectness(bool correctness)
    {
        isCorrect = correctness;
        //Debug.Log(isCorrect);
    }
    public bool GetCorrectness() {
        return isCorrect;

    }
    public  void SpawnItemAlternateLocation(IItemOwner itemOwner, Transform spawnPoint,bool isCorrect)
    {
        Transform itemTransform = Instantiate(this.transform, spawnPoint);
        itemOwner.SetCarriableItem(this);
        itemTransform.GetComponent<CrumpledPaper>().SetCorrectness(isCorrect);
        itemTransform.GetComponent<CrumpledPaper>().SetObjectOwnerAlternateLocation(itemOwner, spawnPoint);
        

    }
    public void SetObjectOwnerAlternateLocation(IItemOwner itemOwner, Transform spawnPoint)
    {
       
        if (this.itemOwner != null)
        {
            this.itemOwner.ClearCarriableItem();
        }
        this.itemOwner = itemOwner;
        this.itemOwner.SetCarriableItem(this);

        this.transform.parent = spawnPoint;
        //this.transform.parent.localPosition =  new Vector3(0f, 0f, 0f);
        this.transform.localPosition = new Vector3(0f, 0f, 0f);
        this.transform.localRotation = Quaternion.Euler(0f, 0f, 0f);
        this.transform.parent.localRotation = Quaternion.Euler(0f, 0f, 0f);
        //Debug.Log("executed");
        

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


        if (this.itemOwner is Player )
        {

            this.transform.localPosition = new Vector3(0f, 0f, 0f);
            //this.transform.localScale = new Vector3(6f, 4.5f, 5.5f);
            this.transform.localRotation = Quaternion.Euler(-40f, 0f, 0f);
        }
        
        else
        {
            this.transform.localPosition = new Vector3(0f, 0f, 0f);
            this.transform.localRotation = Quaternion.Euler(0f, 0f, 0f);
            this.transform.parent.localRotation = Quaternion.Euler(0f, 0f, 0f);
            //this.transform.localScale = new Vector3(1f, 7f, 4f);
        }

    }

}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class RecycleBin : StationaryObject
{
    [SerializeField] private Transform floor;
    [SerializeField] private Transform spawnPoint;
    private bool ledOpen = false;
    public event EventHandler<OnOpenCloseLedEventArgs> OnOpenCloseLed;
    //private List<CrumpledPaper> dumpedPapers = new List<CrumpledPaper>();
    private List<bool> correctPapersDumped = new List<bool>();
    public event EventHandler OnInteractionVisualUpdate;
    public event EventHandler<OnPointsChangedEventArgs> OnPointsChanged;
    private bool hasDissapeared =false;
    private float respawnTimer = 0f;
    private float maxRespawnTimer = 5f;
    [SerializeField] private Transform binSmoke;
    [SerializeField] private SoundManager soundManager;
    
    public class OnPointsChangedEventArgs
    {
        public int currentRecyclePoints;
    }

    private void Start()
    {
    }
    public void Respawn()
    {

       
        binSmoke.GetComponent<ParticleSystem>().Play();
        hasDissapeared = true;
        GetComponent<BoxCollider>().enabled = false;
       foreach (Transform child in transform)
        {
            if(child.transform != binSmoke)
            child.gameObject.SetActive(false);
        }
        if (Player.Instance.IsCarrying())
        {
            Player.Instance.HandleCarrying();
        }
    }
    
    public class OnOpenCloseLedEventArgs
    {
        public bool Open;
    }
    
   private  void Update()
    {
        if (hasDissapeared)
        {
            if (maxRespawnTimer >= respawnTimer) { respawnTimer += Time.deltaTime; }
            else { 
                respawnTimer = 0;
                hasDissapeared = false;
                foreach(Transform child in transform)
                {
                    child.gameObject.SetActive(true);
                    transform.position = spawnPoint.position;
                    transform.rotation = new Quaternion(0, 0, 0, 0);
                }
                GetComponent<BoxCollider>().enabled = true;
            }
        }
    }
    public override void Interact()
    {
        if (!Player.Instance.IsCarrying() && !Player.Instance.HasCarriableItem())
        {
            ledOpen = !ledOpen;
            OnOpenCloseLed?.Invoke(this, new OnOpenCloseLedEventArgs { Open = ledOpen });
            soundManager.PlaySuccessPlacement();
        }
        else if (Player.Instance.HasCarriableItem() && Player.Instance.GetCarriableItem() is CrumpledPaper && ledOpen)
        {
            Debug.Log((Player.Instance.GetCarriableItem() as CrumpledPaper).GetCorrectness());
            correctPapersDumped.Add((Player.Instance.GetCarriableItem() as CrumpledPaper).GetCorrectness());
            Player.Instance.GetCarriableItem().DeleteItem();
            OnPointsChanged?.Invoke(this, new OnPointsChangedEventArgs { currentRecyclePoints = GetCurrentPoints() });
            soundManager.PlayUploadSuccessful();
        }
        else if(Player.Instance.HasCarriableItem() && ledOpen)
        {
            Player.Instance.GetCarriableItem().DeleteItem();
            soundManager.PlayNotebookInteract();
            
        }
        else { soundManager.PlayFailInteract(); }

        if(correctPapersDumped.Count != 0)
        {
            foreach(bool x in correctPapersDumped)
            {
                //Debug.Log(x);
            }
        }
        OnInteractionVisualUpdate?.Invoke(this, EventArgs.Empty);
    }

    public void Carry()
    {
        if (!ledOpen && !Player.Instance.HasCarriableItem()) { 
        if (!Player.Instance.IsCarrying())
        {
            transform.parent = Player.Instance.GetCarryPoint();
            transform.localRotation = Quaternion.Euler(0, 90, 0);
            Player.Instance.SetCarrying(true);
        }
        else
        {
            //Debug.Log("false");
            transform.parent = floor;
            Player.Instance.SetCarrying(false);
        }

            OnInteractionVisualUpdate?.Invoke(this, EventArgs.Empty);
        }
    }

    public int GetCurrentPoints()
    {
        int counter = 0;

        foreach (bool answer in correctPapersDumped)
        {
            if (answer == true)
            { counter++;  }
        }
        return counter;
    }


    public void PointDeduction()
    {
        int counter = 0;
        int index = 0;
        while(counter < 3)
        {
            if (correctPapersDumped[index] == true)
            {
                correctPapersDumped.RemoveAt(index);
                counter++;
            }
            else { index++; }
        }

        OnPointsChanged?.Invoke(this, new OnPointsChangedEventArgs { currentRecyclePoints = GetCurrentPoints() }); ;
    }


    public bool IsLedOn()
    {
        return ledOpen;
    }
}

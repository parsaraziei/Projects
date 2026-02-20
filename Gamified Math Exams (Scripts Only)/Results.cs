using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class Results : MonoBehaviour
{
    [SerializeField] private ExitDoorOpen exitDoor;
    public event EventHandler OnGenrateData;


    private void Start()
    {
        exitDoor.OnGameExitted += ExitDoor_OnGameExitted;
        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(false);

        }
    }

    private void ExitDoor_OnGameExitted(object sender, System.EventArgs e)
    {
        foreach(Transform child in transform)
        {
            
            child.gameObject.SetActive(true);
  
        }
        OnGenrateData.Invoke(this, EventArgs.Empty);
    }
}

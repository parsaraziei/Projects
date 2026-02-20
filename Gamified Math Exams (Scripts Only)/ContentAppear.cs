using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class ContentAppear : MonoBehaviour
{

    public event EventHandler OnResultsActivated;
    [SerializeField] private ExitDoorOpen exitDoor;
    
    // Start is called before the first frame update
    void Start()
    {
        exitDoor.OnGameExitted += ExitDoor_OnGameExitted;
        foreach(Transform child in transform) {
            child.gameObject.SetActive(false);
        }
    }

    private void ExitDoor_OnGameExitted(object sender, EventArgs e)
    {
        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(true);
        }
        OnResultsActivated?.Invoke(this, EventArgs.Empty);

    }

    // Update is called once per frame
    
}

using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;

public class BoxUI : MonoBehaviour
{
    [SerializeField] private StationaryObjectVisualInteractEnabled visuals;



    private void Awake()
    {   
        visuals.OnlayerActive += Visuals_OnlayerActive;
       
    }


    private void Visuals_OnlayerActive(object sender, StationaryObjectVisualInteractEnabled.OnLayerActiveEventArgs e)
    {
        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(e.Active);
        }
    }

 
}

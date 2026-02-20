using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class StationaryObjectVisualInteractEnabledCrate : MonoBehaviour
{
    [SerializeField] private GameObject[] SelectedVisuals;
    [SerializeField] private Crate currentObject;
    [SerializeField] private Player player;
    public event EventHandler<OnLayerActiveEventArgs> OnlayerActive;
    
    public class OnLayerActiveEventArgs
    {
        public bool Active;
    }

    private void Update()
    {
        if (player.GetCurrentCrate() == currentObject) { Show(); }
        else { Hide(); }
    }
    private void Show()
    {
        OnlayerActive?.Invoke(this, new OnLayerActiveEventArgs { Active = true });
        foreach(GameObject visual in SelectedVisuals)
        {
            visual.SetActive(true);
        }
    }

    private void Hide()
    {
        OnlayerActive?.Invoke(this, new OnLayerActiveEventArgs { Active = false });
        foreach (GameObject visual in SelectedVisuals)
        {
            
            visual.SetActive(false);
        }
    }
}

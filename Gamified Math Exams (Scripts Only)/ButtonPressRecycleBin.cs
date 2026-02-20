using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ButtonPressRecycleBin : MonoBehaviour
{
    [SerializeField] private StationaryObjectVisualInteractEnabled RecycleBinVisuals;
    [SerializeField] private RecycleBin recycleBin;
    [SerializeField] private Image PressF;
    [SerializeField] private Image PressR;
    [SerializeField] private bool isActive;



    private void Start()
    {
        RecycleBinVisuals.OnlayerActive += RecycleBinVisuals_OnlayerActive;
        recycleBin.OnInteractionVisualUpdate += RecycleBin_OnInteractionVisualUpdate;
    }

    private void RecycleBin_OnInteractionVisualUpdate(object sender, System.EventArgs e)
    {
        if (isActive) {
            if (Player.Instance.IsCarrying())
            {
                PressF.gameObject.SetActive(false);
                PressR.gameObject.SetActive(true);
            }
            else if (!recycleBin.IsLedOn() && !Player.Instance.IsCarrying()) {
                PressF.gameObject.SetActive(true);
                PressR.gameObject.SetActive(true);
            }
            else if (recycleBin.IsLedOn() && !Player.Instance.IsCarrying())
            {
                PressF.gameObject.SetActive(true);
                PressR.gameObject.SetActive(false);
            }

        }
       
    }

    private void Update()
    {
        UpdateVisuals();
    }


    
    private void RecycleBinVisuals_OnlayerActive(object sender, StationaryObjectVisualInteractEnabled.OnLayerActiveEventArgs e)
    {
        isActive = e.Active;
        UpdateVisuals();    
    }

    private void UpdateVisuals()
    {
        if (isActive)
        {
            if (!recycleBin.IsLedOn() && Player.Instance.IsCarrying())
            {
                PressR.gameObject.SetActive(true);

            }
            else if (!recycleBin.IsLedOn() && !Player.Instance.IsCarrying()) { PressR.gameObject.SetActive(true); PressF.gameObject.SetActive(true); }
            else { PressF.gameObject.SetActive(true); }
        }
        else
        {
            PressF.gameObject.SetActive(false);
            PressR.gameObject.SetActive(Player.Instance.IsCarrying());
        }
    }
}


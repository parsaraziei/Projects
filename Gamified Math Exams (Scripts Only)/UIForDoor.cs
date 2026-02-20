
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using UnityEngine.UI;

public class UIForDoor : MonoBehaviour
{
    [SerializeField] private StationaryObjectVisualInteractEnabled visuals;
    [SerializeField] private MovementSystem movementSystem;
    [SerializeField] private Image loading;
    [SerializeField] private ExitDoorOpen exitDoor;


    private void Awake()
    {
        visuals.OnlayerActive += Visuals_OnlayerActive;
        movementSystem.OnExittingChanged += MovementSystem_OnExittingChanged;
        loading.fillAmount = 0;

    }

    private void MovementSystem_OnExittingChanged(object sender, MovementSystem.OnExittingChangedEventArgs e)
    {
        if (e.progress == 0)
        {
            loading.gameObject.SetActive(false);
        }
        else if (e.progress < 1 && e.progress > 0)
        {
            loading.gameObject.SetActive(true);
            loading.fillAmount = e.progress;
        }
        else if (e.progress >= 1)
        {
            movementSystem.DisableMovement();
            exitDoor.OpenDoor();
        }
    }

    private void Visuals_OnlayerActive(object sender, StationaryObjectVisualInteractEnabled.OnLayerActiveEventArgs e)
    {
        foreach (Transform child in transform)
        {
            child.gameObject.SetActive(e.Active);
        }
    }


}


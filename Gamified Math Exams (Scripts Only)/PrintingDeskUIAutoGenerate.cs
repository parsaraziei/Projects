using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PrintingDeskUIAutoGenerate : MonoBehaviour
{
    [SerializeField] private PrintingTable printingTable;
    [SerializeField] private StationaryObjectVisualInteractEnabled visuals;
    [SerializeField] private Transform pressF;
    [SerializeField] private Transform PressE;


    private void Awake()
    {
        visuals.OnlayerActive += Visuals_OnlayerActive;
    }

    private void Visuals_OnlayerActive(object sender, StationaryObjectVisualInteractEnabled.OnLayerActiveEventArgs e)
    {
        if (e.Active) { 
        switch (printingTable.GetDeskState())
        {
            case PrintingProgress.Idle:
                pressF.gameObject.SetActive(true);
                PressE.gameObject.SetActive(false);
                break;
            case PrintingProgress.Done:
                pressF.gameObject.SetActive(true);
                PressE.gameObject.SetActive(false);
                break;
            case PrintingProgress.Typing:
                pressF.gameObject.SetActive(false);
                PressE.gameObject.SetActive(true);
                break;
            case PrintingProgress.Placed:
                pressF.gameObject.SetActive(true);
                PressE.gameObject.SetActive(true);
                break;
        }
        }
        else
        {
            pressF.gameObject.SetActive(false);
            PressE.gameObject.SetActive(false);

        }
    }
}

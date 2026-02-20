using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RecycleBinVisuals : MonoBehaviour
{
    [SerializeField] private Transform capOn;
    [SerializeField] private Transform capOff;
    [SerializeField] private RecycleBin recycleBin;

    private void Start()
    {
        recycleBin.OnOpenCloseLed += RecycleBin_OnOpenCloseLed;
    }

    private void RecycleBin_OnOpenCloseLed(object sender, RecycleBin.OnOpenCloseLedEventArgs e)
    {
        capOn.gameObject.SetActive(!recycleBin.IsLedOn());
        capOff.gameObject.SetActive(recycleBin.IsLedOn());
    }
}

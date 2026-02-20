using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class PrintingDeskVIsualForAutoGenerate : MonoBehaviour
{
    [SerializeField] private PrintingTable printingTable;
    [SerializeField] private Transform laptopOn;
    [SerializeField] private Transform laptopOff;
    [SerializeField] private Transform PrintSuccessfulScreen;
    [SerializeField] private Transform printFailed;

    private void Start()
    {
        printingTable.OnPrintingProgressChanged += PrintingTable_OnPrintingProgressChanged;
    }

    private void PrintingTable_OnPrintingProgressChanged(object sender, PrintingTable.PrintingProgressChangedEventArgs e)
    {
        switch (e.currentPrintingState)
        {
            case PrintingProgress.Typing:
                PrintSuccessfulScreen.gameObject.SetActive(false);
                printFailed.gameObject.SetActive(false);
                laptopOff.gameObject.SetActive(false);
                laptopOn.gameObject.SetActive(true);
                break;

            case PrintingProgress.Idle:
                PrintSuccessfulScreen.gameObject.SetActive(false);
                printFailed.gameObject.SetActive(false);
                laptopOff.gameObject.SetActive(true);
                laptopOn.gameObject.SetActive(false);
                break;

            case PrintingProgress.Done:
                if (printingTable.VerifyAnswer())
                {
                    PrintSuccessfulScreen.gameObject.SetActive(true);
                    laptopOff.gameObject.SetActive(true);
                    laptopOn.gameObject.SetActive(false);
                }
                else
                {

                    printFailed.gameObject.SetActive(true);
                    laptopOff.gameObject.SetActive(true);
                    laptopOn.gameObject.SetActive(false);

                }
                break;

            case PrintingProgress.Placed:
                printFailed.gameObject.SetActive(false);
                PrintSuccessfulScreen.gameObject.SetActive(false);
                laptopOff.gameObject.SetActive(true);
                laptopOn.gameObject.SetActive(false);
                break;
        }
    }
}

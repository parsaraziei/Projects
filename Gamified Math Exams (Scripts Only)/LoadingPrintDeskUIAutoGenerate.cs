using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;



public class LoadingPrintDeskUIAutoGenerate : MonoBehaviour
{
    [SerializeField] private Image loading;
    [SerializeField] private PrintingTable printingTable;

    private void Start()
    {
        printingTable.OnPrintingProgressChanged += PrintingTable_OnPrintingProgressChanged;
    }

    private void PrintingTable_OnPrintingProgressChanged(object sender, PrintingTable.PrintingProgressChangedEventArgs e)
    {
        if (e.currentPrintingState == PrintingProgress.Typing)
        {
            loading.gameObject.SetActive(true);
            loading.fillAmount = e.progress;

        }
        else { loading.gameObject.SetActive(false); }
    }
  
}

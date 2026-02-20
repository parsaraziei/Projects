using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class LoadingPrintDeskUI : MonoBehaviour
{
    [SerializeField] private Image loading;
    [SerializeField] private PrintingTableImport printingTable;

    private void Start()
    {
        printingTable.OnPrintingProgressChanged += PrintingTable_OnPrintingProgressChanged1;
    }

    private void PrintingTable_OnPrintingProgressChanged1(object sender, PrintingTableImport.PrintingProgressChangedEventArgs e)
    {
        if (e.currentPrintingState == PrintingProgress.Typing)
        {
            loading.gameObject.SetActive(true);
            loading.fillAmount = e.progress;

        }
        else { loading.gameObject.SetActive(false); }
    }

  
}

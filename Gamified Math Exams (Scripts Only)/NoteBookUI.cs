using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class NoteBookUI : MonoBehaviour
{
    [SerializeField] private List<CarriableItemSO> alldigits;
    [SerializeField] private Transform container;
    [SerializeField] private NotePad notePad;


    private void Awake()
    {
        notePad.OnItemAdded += NotePad_OnItemAdded;
        container.gameObject.SetActive(false);
    }

    private void NotePad_OnItemAdded(object sender, System.EventArgs e)
    {
       
        foreach (Transform child in transform)
        {
            if (child != container) { 
                Destroy(child.gameObject);
            }
        }
        
        foreach (CarriableItem item in notePad.GetDigitList()) {
            
            foreach (CarriableItemSO digit in alldigits)
            {
                
                if (item.CompareTag(digit.carriableItem.tag))
                {   
                        Transform digitIndicator = Instantiate(container, transform);
                        digitIndicator.gameObject.SetActive(true);
                        digitIndicator.GetComponent<Container>().SetImage(digit);
                }
            }

        }

    }
}


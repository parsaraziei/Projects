using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class NoteBookdAnimations : MonoBehaviour
{
    [SerializeField] CarriableItem currentCarriableItem;
    private Animator animator;
    private const string NOTEBOOK_DROPPED = "NoteBook_Dropped";
    


    public void Awake()
    {
        Player.Instance.OnStateChanged += Instance_OnStateChanged;
        animator = GetComponent<Animator>();  
    }


    private void Instance_OnStateChanged(object sender, Player.OnStateChangedEventArgs e)
    {
        if (Player.Instance.HasCarriableItem() && animator != null && currentCarriableItem == Player.Instance.GetCarriableItem())
        {
            if (currentCarriableItem.GetItemOwner() is Player)
            {
 

                if (e.stateSent == State.Sitting)
                {
                    animator.SetBool(NOTEBOOK_DROPPED, true);
                }
                else { animator.SetBool(NOTEBOOK_DROPPED, false); }
            }
            else {  }
        }
        else 
        {
          ;
        }
    } 
    
}

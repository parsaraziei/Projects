using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class CardAnimations : MonoBehaviour
{
    [SerializeField] CarriableItem currentCarriableItem;
    private Animator animator;
    private const string CARD_DROPPED = "Drop_Card";
   

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
                    animator.SetBool(CARD_DROPPED, true);
                }
                else { animator.SetBool(CARD_DROPPED, false); }
            }
        }
    } 
    
}

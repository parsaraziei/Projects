using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class FoxAnimations : MonoBehaviour
{

    
    [SerializeField] private Player player;
    [SerializeField] private MovementSystem movementSystem;
    private Animator animator;
    public event EventHandler OnInteractionAnimationPerformed;
    private const string FOX_WALKING = "Fox_Walking";
    private const string FOX_SITTING = "Fox_Sitting";
    private const string FOX_INTERACTING = "Fox_Interacting";
    private const string CARRYING = "Carrying";
    private const string CAR_HIT = "Car_Hit";

    private void Awake()
    {
        
        animator = GetComponent<Animator>();
        player.OnStateChanged += Player_OnStateChanged;
        movementSystem.OnPlayerCrash += MovementSystem_OnPlayerCrash;
    }

    private void MovementSystem_OnPlayerCrash(object sender, MovementSystem.OnPlayerCrashEventArgs e)
    {
       if(e.playerDisabled == true) { animator.SetBool(CAR_HIT, true); } else { animator.SetBool(CAR_HIT, false); }
    }

    private void Player_OnStateChanged(object sender, Player.OnStateChangedEventArgs e)
    {
        if (e.stateSent == State.Interacting)
        {
            animator.SetBool(CARRYING, player.IsCarrying());
            animator.SetTrigger(FOX_INTERACTING);
            animator.SetBool(FOX_WALKING, false);
            animator.SetBool(FOX_SITTING, false);
            if (!player.IsCarrying()) { 
           OnInteractionAnimationPerformed?.Invoke(this, EventArgs.Empty);
            }

        }
        else if (e.stateSent == State.Idle) {

            animator.SetBool(CARRYING, player.IsCarrying());
            animator.SetBool(FOX_SITTING, false);
            animator.SetBool(FOX_WALKING, false);
        }
        else if(e.stateSent == State.Walking)
        {
            animator.SetBool(CARRYING, player.IsCarrying());
            animator.SetBool(FOX_SITTING, false);
            animator.SetBool(FOX_WALKING, true);
        }
        else if(e.stateSent == State.Sitting)
        {
            animator.SetBool(CARRYING, player.IsCarrying());
            animator.SetBool(FOX_WALKING, false);
            animator.SetBool(FOX_SITTING, true);
        }
        
    }
}

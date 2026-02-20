using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class FoxSpawnPointAnimation : MonoBehaviour
{


    [SerializeField] private FoxAnimations foxAnimations;
    private Animator animator;

    private const string FOX_INTERACTING = "Fox_Interacting";

    private void Awake()
    {
        
        animator = GetComponent<Animator>();
        foxAnimations.OnInteractionAnimationPerformed += FoxAnimations_OnInteractionAnimationPerformed;
    }

    private void FoxAnimations_OnInteractionAnimationPerformed(object sender, EventArgs e)
    {
        animator.SetTrigger(FOX_INTERACTING);
    }

   
}

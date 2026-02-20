using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CrateAnimation : MonoBehaviour
{
    private Animator animator;
    private const string FOX_INTERACTED = "Fox_Interacted";
    [SerializeField] private Crate thisCrate;
    public void Awake()
    {
        animator = GetComponent<Animator>();
        thisCrate.OnBoxInteracted += ThisCrate_OnBoxInteracted1;
    }

    private void ThisCrate_OnBoxInteracted1(object sender, System.EventArgs e)
    {
        animator.SetTrigger(FOX_INTERACTED);
    }

}

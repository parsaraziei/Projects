using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ExitDoor : StationaryObject
{
    
    public override void Interact()
    {
        Debug.Log("NoExit");
    }

    public void Exit()
    {
        
    }
}

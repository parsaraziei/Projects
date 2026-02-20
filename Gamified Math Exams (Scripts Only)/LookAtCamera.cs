using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LookAtCamera : MonoBehaviour
{
    [SerializeField] private Mode mode;
    public enum Mode
    {
        lookAt,
        LookAtInverted,
        Normal


    }

   
    private void LateUpdate()
    {
        
        switch (mode)
        {
            case Mode.lookAt:
                transform.LookAt(Camera.main.transform);
                break;

            case Mode.LookAtInverted:
                Vector3 dirFromCamera = transform.position - Camera.main.transform.position;
                transform.LookAt(transform.position + dirFromCamera);
                break;

            case Mode.Normal:
                transform.LookAt(Camera.main.transform);
                transform.Rotate(0, 180, 0);
                break;
                
        }
        
    }
}

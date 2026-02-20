using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SinglePowerUp : MonoBehaviour
{
    float maxTimer = 20f;
    float timer = 0f;
    [SerializeField] private string PowerUpType;
    [SerializeField] private SoundManager soundManager;
    private void Update()
    {
        ItemExpire();
        HandlePickUp();
    }

    private void ItemExpire()
    {
        if (timer < maxTimer) { timer += Time.deltaTime; }
        else { timer = 0;
            Destroy(this.gameObject);
        }
    }

    private void HandlePickUp()
    {
        
           if (Physics.SphereCastAll(transform.position,2f,Vector3.up,10f)!=null)
            foreach(RaycastHit ray in (Physics.SphereCastAll(transform.position, 2f, Vector3.up, 10f)))
            {
                if (ray.transform.CompareTag("Fox"))
                {
                    soundManager.PlayObjectPickUp();
                    Destroy(this.gameObject);
                    Player.Instance.ActivatePowerUp(PowerUpType);
                }
            }
/*
        if (Physics.SphereCastAll(transform.position, 2f, Vector3.down, 10f) != null)
            foreach (RaycastHit ray in (Physics.SphereCastAll(transform.position, 2f, Vector3.down, 10f)))
            {
                if (ray.transform.CompareTag("Fox"))
                {
                    Destroy(this.gameObject);
                    Debug.Log("destroyed2");
                }
            }
*/
    }



    private void ActivatePlayer()
    {


    }
}

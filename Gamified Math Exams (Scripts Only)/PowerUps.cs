using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PowerUps : MonoBehaviour
{
    [SerializeField] private Transform[] powerUps;
    [SerializeField] private Transform topRightcorner;
    [SerializeField] private Transform bottomLeftCorner;
   
    private float timer = 0f;
    private float MaxtTimer = 30f;
    private Transform currentPowerUp;
    private void Update()
    {
        if (timer < MaxtTimer) { 
            timer += Time.deltaTime;
            //Debug.Log(timer);
        }
        else {
            timer = 0f;
            float xPlacement = Random.Range(topRightcorner.position.x, bottomLeftCorner.position.x);
            float zPlacement = Random.Range(topRightcorner.position.z, bottomLeftCorner.position.z);
            if(currentPowerUp != null)
            {
               
                Destroy(currentPowerUp.gameObject);
                
            }
            Transform powerUpTransform = Instantiate(powerUps[Random.Range(0, powerUps.Length)],transform);
            powerUpTransform.gameObject.SetActive(true);
            Debug.Log(xPlacement + "," + zPlacement);
            
            powerUpTransform.position = new Vector3(xPlacement, 1.5f, zPlacement);
            currentPowerUp = powerUpTransform;
            
        }
    }
}

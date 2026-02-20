using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarGenerator : MonoBehaviour
{
    [SerializeField] private Transform approachingCarSpawnPoint;
    [SerializeField] private Transform leavingCarSpawnPoint;
    [SerializeField] private QuestionBuilder questionBuilder;

    [SerializeField] private GameObject carType1;
    [SerializeField] private GameObject carType2;

    private float timer = 0;
    private float maxTimer;
    private bool timerActive = false;

    private void Run()
    {
        int carSpawn = Random.Range(1, 3);
        if (carSpawn == 1)
        {
            Transform firstcarTransform = Instantiate(carType1.transform, transform);
            firstcarTransform.localPosition = leavingCarSpawnPoint.position;
            firstcarTransform.gameObject.SetActive(true);
            (firstcarTransform.GetComponent<CarDistancing>()).Activate();
        }
        else
        {
            Transform secondcarTransform = Instantiate(carType2.transform, transform);
            secondcarTransform.localPosition = approachingCarSpawnPoint.position;
            secondcarTransform.gameObject.SetActive(true);
            (secondcarTransform.GetComponent<CarApproaching>()).Activate();
        }

    }

    private void Update()
    {
        if (!questionBuilder.GetIsGameOver())
        {
            if (!timerActive)
            {
                maxTimer = Random.Range(1, 6);
                //Debug.Log(maxTimer);
                timerActive = true;
            }
            else
            {
                if (timer > maxTimer)
                {
                    timer = 0;
                    Run();
                    timerActive = false;
                }
                else
                {
                    timer += Time.deltaTime;
                }

            }

        }
        else {foreach(Transform child in transform) { Destroy(child.gameObject); } }
    }
}

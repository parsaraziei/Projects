using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using TMPro;
using UnityEngine.UI;



public class RecyclePointCounter : MonoBehaviour
{
    [SerializeField] private RecycleBin recycleBin;
    [SerializeField] private TextMeshProUGUI counter;
    private int currentRecyclePoints;
    private float timer = 0;
    private float maxTimer = 5f;
   
    // Start is called before the first frame update
    private void Start()
    {
        recycleBin.OnPointsChanged += RecycleBin_OnPointsChanged;
        TextMeshProUGUI counter = transform.GetComponent<TextMeshProUGUI>();
        transform.gameObject.SetActive(false);
    }

    private void Update()
    {   
        if (this.gameObject.activeSelf && currentRecyclePoints == 0)
        {
            //Debug.Log(this.gameObject.activeSelf);
            if (timer > maxTimer) {
                timer = 0;
                this.gameObject.SetActive(false);
            }
            else { timer += Time.deltaTime; }

        }
    }



    private void RecycleBin_OnPointsChanged(object sender, RecycleBin.OnPointsChangedEventArgs e)
    {
        this.gameObject.SetActive(true);
        counter.text = (e.currentRecyclePoints).ToString() + ":";
        this.currentRecyclePoints = e.currentRecyclePoints;
        
    }
}

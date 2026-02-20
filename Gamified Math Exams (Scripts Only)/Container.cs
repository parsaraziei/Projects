using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Container : MonoBehaviour
{
    [SerializeField] private Image image;
   public void SetImage(CarriableItemSO item) {
        //Debug.Log(item.Digit);
        image.sprite = item.Sprite;
    }
}

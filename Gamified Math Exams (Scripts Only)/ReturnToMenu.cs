using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
public class ReturnToMenu : MonoBehaviour
{
    [SerializeField] private Content content;
    [SerializeField] private Button button;

    private void Start()
    {

        button.onClick.AddListener(() => {  
            SceneManager.LoadScene(0);
            if (SceneManager.GetActiveScene().buildIndex != 2) { content.resetMark(); }
        });
    }
}

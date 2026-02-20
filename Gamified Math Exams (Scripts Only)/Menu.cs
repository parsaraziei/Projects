using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using TMPro;
public class Menu: MonoBehaviour
{
    [SerializeField] private Transform FirstMenu;
    [SerializeField] private Button Play;
    [SerializeField] private Button Tutorial;
    [SerializeField] private Button Exit;

    [SerializeField] private Transform SecondaryMenu;
    [SerializeField] private TMP_Dropdown SingleAttempt;
    [SerializeField] private Button Exam;
    [SerializeField] private Button Endless;
    [SerializeField] private Button Arcade;
    [SerializeField] private Button Back;
    [SerializeField] private AudioSource audioSource;
    [SerializeField] private Toggle Mute;



    private void Start()
    {
        if (ExamType.isGameMuted) {
            audioSource.Stop();
            Mute.isOn = true;
        }
        else { audioSource.Play();
            Mute.isOn = false;
        }
    }
    private void Awake()
    {
        Play.onClick.AddListener(() =>
        {
            FirstMenu.gameObject.SetActive(false);
            SecondaryMenu.gameObject.SetActive(true);
        });

        Back.onClick.AddListener(() =>
        {
            FirstMenu.gameObject.SetActive(true);
            SecondaryMenu.gameObject.SetActive(false);
        });

        Exit.onClick.AddListener(() =>
        {
            Application.Quit();

        });
        Exam.onClick.AddListener(() =>
        {
            if (ExamType.ExamFilePath != "")
            {
                SceneManager.LoadScene(3);
                ExamType.isSingleAttempt = (SingleAttempt.value == 1);
                ExamType.isGameMuted = Mute.isOn;
            }
        });
        Endless.onClick.AddListener(() =>
        {
            SceneManager.LoadScene(2);
            ExamType.isSingleAttempt = (SingleAttempt.value == 1);
            ExamType.isGameMuted = Mute.isOn;
        });
        Arcade.onClick.AddListener(() =>
        {
            if (ExamType.ExamFilePath != "")
            {
                SceneManager.LoadScene(1);
                ExamType.isSingleAttempt = (SingleAttempt.value == 1);
                ExamType.isGameMuted = Mute.isOn;
            }
        });

        Tutorial.onClick.AddListener(() => {
            SceneManager.LoadScene(4);
        });


    }

    private void Update()
    {
        if (!audioSource.isPlaying)
        {
            if (!Mute.isOn)
            {
                audioSource.Play();
            }
        }else if (audioSource.isPlaying)
        {
            if (Mute.isOn) {
                audioSource.Play();
            }
        }
    }

}

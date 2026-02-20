using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
public class ExitDoorOpen : MonoBehaviour
{
    
    [SerializeField] private Transform DoorClosed;
    [SerializeField] private Transform DoorOpened;
    [SerializeField] private QuestionBuilder questionBuilder;
    [SerializeField] private SoundManager soundManager;
    private float examTime;
    [SerializeField] private MovementSystem movementSystem;
    
    

    public event EventHandler OnGameExitted;


    private void Start()
    {
        if (ExamType.isGameMuted)
        {
            AudioSource[] audioSources = FindObjectsOfType<AudioSource>();


            foreach (AudioSource audioSource in audioSources)
            {
                audioSource.volume = 0;

            }
        }
    }

    public void OpenDoor()
    {
        soundManager.PlayOpenDoor();
        DoorClosed.gameObject.SetActive(false);
        DoorOpened.gameObject.SetActive(true);
        if (questionBuilder != null)
        {
            questionBuilder.GetTimerList();
            examTime = questionBuilder.TimeTookForExam();
            questionBuilder.EndGame();
            questionBuilder.gameObject.SetActive(false);
        }
        else
        {
            movementSystem.DisableMovement();
        }
        OnGameExitted?.Invoke(this, EventArgs.Empty);
    }

    public float GetExamTimer()
    {
        return examTime;
    }
}

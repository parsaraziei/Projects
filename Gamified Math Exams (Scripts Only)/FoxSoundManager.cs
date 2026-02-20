using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FoxSoundManager : MonoBehaviour
{
    [SerializeField] private Player player;
    [SerializeField] private AudioClip[] audioClips;
    private AudioSource walkingSound;

    private void Start()
    {
        walkingSound = GetComponent<AudioSource>();
    }

    private void Update()
    {
        if (player.currentState == State.Walking && player.IsCarrying()) { walkingSound.clip = audioClips[1]; playSound(); }
        else if (player.currentState == State.Walking) { walkingSound.clip = audioClips[0]; playSound(); }
        else PauseSound();
    }

    private void playSound()
    {
        if (!walkingSound.isPlaying)
        {
            if (!player.IsCarrying())
            {
                
                walkingSound.Play();
            }
            else
            {
               
                walkingSound.Play();
            }

        }
    }
    private void PauseSound()
    { 
        walkingSound.Pause();
    }
}

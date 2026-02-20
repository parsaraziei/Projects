using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoundManager : MonoBehaviour
{
    [SerializeField] private AudioClip[] audioClips;
    private AudioSource audioSource;
    private void Start()
    {
       audioSource = GetComponent<AudioSource>();
    }
    public void PlaySuccessInteract()
    {
        audioSource.clip = audioClips[0];
        audioSource.Play();
    }

    public void PlayFailInteract()
    {
        audioSource.clip = audioClips[1];
        audioSource.Play();
    }
    public void PlaySuccessDelete()
    {
        audioSource.clip = audioClips[2];
        audioSource.Play();
    }
    public void PlayFailDelete()
    {
        audioSource.clip = audioClips[1];
        audioSource.Play();
    }

   
    public void PlaySuccessPlacement()
    {
        audioSource.clip = audioClips[3];
        audioSource.Play();
    }
    public void PlayChangeQuestion()
    {
        audioSource.clip = audioClips[4];
        audioSource.Play();
    }
    public void PlayNotebookInteract()
    {
        audioSource.clip = audioClips[5];
        audioSource.Play();
    }
    

      public void PlayUploadSuccessful() {
        audioSource.clip = audioClips[6];
        audioSource.Play();
    }


        public void PlayUploadFail() {
        audioSource.clip = audioClips[7];
        audioSource.Play();
    }

    public void PlayOpenDoor()
    {
        audioSource.clip = audioClips[8];
        audioSource.Play();
    }

    public void PlayObjectPickUp()
    {
        audioSource.clip = audioClips[9];
        audioSource.Play();
    }


    public void PlayCrash()
    {
        audioSource.clip = audioClips[10];
        audioSource.Play();
    }


    public void PayPoints()
    {
        audioSource.clip = audioClips[11];
        audioSource.Play();
    }

}

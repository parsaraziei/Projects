using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarDistancing : StationaryObject
{
    private float timer = 0;
    private float Maxtimer = 20f;
    private bool carActivated = false;
    private float speedMultiplier;
    [SerializeField] private Transform crate;
    [SerializeField] private Transform laneMidPoint;
    [SerializeField] Player player;
    [SerializeField] private Transform smoke;
    [SerializeField] SoundManager soundManager;
    private AudioSource audioSource;
    private bool isDestroy;
    private float destroyTimer;
    
    private void Awake()
    {
        player.OntrafficPaused += Instance_OntrafficPaused;
    }

    private void Start()
    {
        audioSource = GetComponent<AudioSource>();
        if (ExamType.isGameMuted) audioSource.volume = 0;
    }

    private void Instance_OntrafficPaused(object sender, Player.OnTrafficPausedEventArgs e)
    {
        carActivated = !e.isPaused;
       
    }

    public override void Interact()
    {
       
    }

    

   
    public void Activate()
    {
        carActivated = true;
        speedMultiplier = Random.Range(20f, 50f);
        //audioSource.Play();
    }
    private void Update()
    {
        if (carActivated)
        {
            if (!audioSource.isPlaying) { audioSource.Play(); }
            Vector3 forwardvector3 = transform.position;
            forwardvector3.z += Time.deltaTime * speedMultiplier;
            transform.position = forwardvector3;

            if (Physics.BoxCast(transform.position, new Vector3(3, 2, 3), transform.forward, out RaycastHit raycast, Quaternion.identity, 5f))
            {
                if (raycast.transform.gameObject.CompareTag("Fox"))
                {
                    if ((raycast.transform.GetComponent<Player>()).shieldActive == true)
                    {
                        soundManager.PlayCrash();
                        smoke.GetComponent<ParticleSystem>().Play();
                        isDestroy = true;
                    }
                    else
                    {
                        if (crate.transform.position.z <= raycast.transform.position.z + 2)
                        {
                            if (raycast.transform.position.x < laneMidPoint.position.x)
                            {
                                raycast.transform.position += new Vector3(-3, 0, 0);

                            }
                            else { raycast.transform.position += new Vector3(3, 0, 0); }

                        }
                        else
                        {
                            if (raycast.transform.position.x < laneMidPoint.position.x)
                            {
                                raycast.transform.position += new Vector3(-3, 0, 1);
                            }
                            else { raycast.transform.position += new Vector3(3, 0, 1); }
                        }
                        raycast.transform.GetComponent<Player>().CarHit();
                        soundManager.PlayCrash();
                    }

                }

                if (raycast.transform.GetComponent<RecycleBin>() != null)
                {
                    raycast.transform.GetComponent<RecycleBin>().Respawn();
                    soundManager.PlayCrash();
                }

                if (raycast.transform.GetComponent<CarDistancing>() != null)
                {
                    speedMultiplier = raycast.transform.GetComponent<CarDistancing>().GetSpeed();
                }

                /*if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hit, 2f))
                    Debug.Log(hit.transform.gameObject);*/
            }

            if (timer < Maxtimer)
            {
                timer += Time.deltaTime;
            }
            else
            {
                Destroy(this.gameObject);
            }

        }
        else
        {
            if (audioSource.isPlaying)
            { audioSource.Pause(); }
        }
        if (isDestroy)
        {
            if (destroyTimer < 0.5f)
            {
                destroyTimer += Time.deltaTime;
            }
            else
            {
                Destroy(this.gameObject);
            }
        }
    }
    public float GetSpeed()
    {
        return speedMultiplier;
    }
}

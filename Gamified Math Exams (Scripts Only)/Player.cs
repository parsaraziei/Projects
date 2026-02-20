using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;


public enum State
{
    Sitting,
    Walking,
    Idle,
    Interacting
}
public class Player : MonoBehaviour, IItemOwner
{
    //public static event EventHandler<OnCrateEncounterEventArgs> OnCrateEncounter;
    public class OnCrateEncounterEventArgs
    {
        public Crate crate;
    }
    [SerializeField] private Transform paperSmoke;
    [SerializeField] private Transform foxCarryPoint;
    private CarriableItem currentCarriableItem = null;
    [SerializeField] private Transform foxSpawnPoint;
    private float foxRadius = 1.3f;
    [SerializeField] private float foxRadiusCarrying = 4f;
    private float foxRadiusWalking = 1.3f;
    private Vector3 aimDirection = Vector3.zero;
    [SerializeField] private Transform foxStartPoint;
    [SerializeField] private Transform foxEndPoint;
    private Vector3 moveDirectionXaxis;
    private Vector3 moveDirectionZaxis;
    public StationaryObject currentMainObject;
    private bool carrying = false;
    public static Player Instance { get; private set; }
    [SerializeField] private MovementSystem movementSystem;
    [SerializeField] public float playerSpeed;
    public State currentState = State.Idle;
    private float idleTimer;
    private float maxIdleTimer = 5;
    private RecycleBin currentRecycleBinBeingCarried;
    //private float animationTime = 1;
    float interacttimer = 0f;
    float maxInteracttimer = 0.5f;
    private Transform ItemCarryingInstanse;
    // private bool isHit = false ;
    /*   private float jumpTimer = 0;
       private float  maxJumpTimer = 3;
       private bool jumpActive = false;*/

    public event EventHandler<OnTrafficPausedEventArgs> OntrafficPaused;
    public class OnTrafficPausedEventArgs
    {
        public bool isPaused = false;
    }
   
    private float powerUpTimer = 0f;
    private float powerUpMaxTimer = 20f;
    private bool speedActive = false;
    private bool multiPickUp = false;
    public bool shieldActive = false;
    public bool isTrafficPaused= false;

    [SerializeField] private Transform shieldVisual;
    [SerializeField] private Transform SpeedVisual;
    [SerializeField] private Transform multipPickUpVisual;
    [SerializeField] private Transform timeFreeze;

    public event EventHandler<OnStateChangedEventArgs> OnStateChanged;
    public class OnStateChangedEventArgs
    {
        public State stateSent;
       
    }
    private void Awake()
    {
        Instance = this;
        movementSystem.onPlayerJump += MovementSystem_onPlayerJump;
        movementSystem.onPlayerInteract += MovementSystem_onPlayerInteract;
        movementSystem.onPlayerCarry += MovementSystem_onPlayerCarry;
        
    }

    private void MovementSystem_onPlayerCarry(object sender, EventArgs e)
    {
        HandleCarrying();
    }


    public void HandleCarrying()
    {
        if ((currentMainObject != null && currentMainObject is RecycleBin && !carrying))
        {
            (currentMainObject as RecycleBin).Carry();
            currentRecycleBinBeingCarried = currentMainObject as RecycleBin;
        }
        else if (carrying)
        {

            currentRecycleBinBeingCarried.Carry();
            currentRecycleBinBeingCarried = null;
        }
    }

    private void MovementSystem_onPlayerInteract(object sender, EventArgs e)
    {

        if(currentMainObject != null) {
            currentState = State.Interacting;
            OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = State.Interacting  });;
            currentMainObject.Interact();
            
           
        }
    }

    
    private void MovementSystem_onPlayerJump(object sender, System.EventArgs e)
    {
        
    }

    private void HandleSitting()
    {
        if (currentState == State.Idle)
        {
            idleTimer += Time.deltaTime;
            if (idleTimer > maxIdleTimer)
            {
                currentState = State.Sitting;
                OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
                idleTimer = 0;
            }
        }
        else { idleTimer = 0; }
    }

   public void ActivatePowerUp(string powerUpString)
    {
        if (powerUpString == "Shield") { powerUpTimer = 0;  /*GetComponent<CapsuleCollider>().enabled = false;*/ shieldActive = true;  }
        else if (powerUpString == "Time") { powerUpTimer = 0; PauseTraffic(); isTrafficPaused = true; }
        else if (powerUpString == "Speed") { speedActive = true; powerUpTimer = 0;  }
        else { multiPickUp = true; powerUpTimer = 0f; }

    }


    private void PauseTraffic()
    {
        if (isTrafficPaused)
        {
            if (powerUpTimer < powerUpMaxTimer)
            {
                timeFreeze.gameObject.SetActive(true);
                powerUpTimer += Time.deltaTime;
                OntrafficPaused?.Invoke(this, new OnTrafficPausedEventArgs { isPaused = isTrafficPaused });
            }
            else
            {
                timeFreeze.gameObject.SetActive(false);
                powerUpTimer = 0;
                isTrafficPaused = false;
                OntrafficPaused?.Invoke(this, new OnTrafficPausedEventArgs { isPaused = isTrafficPaused });
            }
        }
    }

    private void Update()
    {
        UpdatePosition();
        UpdateInteraction();
        HandleSitting();
        HandleInteractTimer();
        UpdateTyping();
        HandleExiting();
        HandleMovementSpeed();
        HandleSheild();
        HandleSpeed();
        HandleMultiPickUp();
        PauseTraffic(); 
    }



    private void HandleMultiPickUp()
    {
        if (multiPickUp)
        {
            if (powerUpTimer < powerUpMaxTimer)
            {
                powerUpTimer += Time.deltaTime;
                multipPickUpVisual.gameObject.SetActive(true);
            }
            else
            {
                multiPickUp = false;
                multipPickUpVisual.gameObject.SetActive(false);
                powerUpTimer = 0;
            }
        }
    }

    public bool isMultiPickUpActive() 
    { return multiPickUp; }
    private void HandleSpeed() {
        if (speedActive)
        {
            if (powerUpTimer < powerUpMaxTimer)
            {
                SpeedVisual.gameObject.SetActive(true);
                powerUpTimer += Time.deltaTime;
                if (!carrying)
                {                 
                    playerSpeed = 20f;  
                }
            }
            else {
                
                powerUpTimer = 0f;
                speedActive = false;
                if (!carrying)
                    playerSpeed = 12f;
                else
                    playerSpeed = 4f;
                SpeedVisual.gameObject.SetActive(false);
            }
        }
    }

    private void HandleSheild()
    {
        if (shieldActive) {
            if (powerUpTimer < powerUpMaxTimer)
            {
                shieldVisual.gameObject.SetActive(true);
                powerUpTimer += Time.deltaTime;
               
            }
            else { shieldActive = false;
                shieldVisual.gameObject.SetActive(false);
                /*GetComponent<CapsuleCollider>().enabled = true;*/
                powerUpTimer = 0f;
            }
        }
    }

    public void CarHit()
    {   
        if (HasCarriableItem()) {
            GetCarriableItem().DeleteItem();
            paperSmoke.transform.GetComponent<ParticleSystem>().Play();
        }

        if (IsCarrying()) {
            HandleCarrying();            
        }

        movementSystem.DisableTemporarely();    
    }
    private void HandleInteractTimer()
    {
        if (currentState == State.Interacting)
        {

            if (interacttimer < maxInteracttimer) { interacttimer += Time.deltaTime; }
            else
            {
                interacttimer = 0;
                currentState = State.Idle;
                OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
            }
        }
    }


    private void UpdateTyping()
    {
        if(currentMainObject is PrintingTable && !carrying)
        {
            if ((currentMainObject as PrintingTable).CanType())
            {
                movementSystem.KeyEHeld();
            }
        }
        if (currentMainObject is PrintingTableImport && !carrying)
        {
            if ((currentMainObject as PrintingTableImport).CanType())
            {
                movementSystem.KeyEHeld();
            }
        }

    }
    private void HandleExiting()
    {
        if (currentMainObject is ExitDoor)
        {
                movementSystem.Exiting(); 
        }
    }

    private void UpdateInteraction()
    {
        float maxDistance = 5f;
       /* if (Physics.Raycast(foxStartPoint.position, transform.forward, out RaycastHit raycasthit1, maxDistance))
        {
            // If the raycast hits something, draw a green line to visualize the raycast*/
            //Debug.DrawRay(transform.position, transform.forward, Color.green);


        if (Physics.Raycast(foxEndPoint.position, transform.forward ,out RaycastHit raycasthit, maxDistance)) {
           StationaryObject currentObject = raycasthit.transform.GetComponent<StationaryObject>();
            
           if(currentObject is StationaryObject && currentObject != null) {

                if (!carrying || (carrying && currentObject is RecycleBin)) { 
                currentMainObject = currentObject;
                    
                }
            }
           else { currentMainObject = null; }
        }
        else { currentMainObject = null; }


    }

    /* private void PerformJump()
     {
         jumpActive = true;
         jumpTimer = 0;
         while (jumpTimer < maxJumpTimer)
         {
             transform.position += Vector3.up;
         }
         jumpActive = false;
     }
 */

    /*private void UpdateJump()
    {

    }*/
    private void UpdatePosition() {
        
        float rotationSpeed = 8f;
        Vector3 movementDirection = movementSystem.GetMovementVector();
        if (movementDirection != Vector3.zero)
        {
            if (carrying) { foxRadius = foxRadiusCarrying; rotationSpeed = 1f; }
            else { foxRadius = foxRadiusWalking; rotationSpeed = 5f; }
            if (!Physics.CapsuleCast(transform.position, transform.position + Vector3.up, foxRadius, movementDirection, playerSpeed * Time.deltaTime))
            {
                aimDirection = movementDirection;
                idleTimer = 0;
                currentState = State.Walking;
                OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
                transform.position += movementDirection * Time.deltaTime * playerSpeed;
                transform.forward = Vector3.Slerp(transform.forward, movementDirection, Time.deltaTime * rotationSpeed);
            }
            else
            {
                moveDirectionXaxis = new Vector3(movementDirection.x, 0, 0);
                moveDirectionZaxis = new Vector3(0, 0, movementDirection.z);
                if (!Physics.CapsuleCast(transform.position, transform.position + Vector3.up, foxRadius, moveDirectionXaxis, playerSpeed * Time.deltaTime))
                {

                    aimDirection = moveDirectionXaxis;
                    idleTimer = 0;
                    currentState = State.Walking;
                    OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
                    transform.position += moveDirectionXaxis * Time.deltaTime * playerSpeed;
                    transform.forward = Vector3.Slerp(transform.forward, moveDirectionXaxis, Time.deltaTime * rotationSpeed);

                }
                else if (!Physics.CapsuleCast(transform.position, transform.position + Vector3.up, foxRadius, moveDirectionZaxis, playerSpeed * Time.deltaTime))
                {

                    aimDirection = moveDirectionZaxis;
                    idleTimer = 0;
                    currentState = State.Walking;
                    OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
                    transform.position += moveDirectionZaxis * Time.deltaTime * playerSpeed;
                    transform.forward = Vector3.Slerp(transform.forward, moveDirectionZaxis, Time.deltaTime * rotationSpeed);


                }
                else { aimDirection = movementDirection;
                }
            }
        }
        else
        {
            if (currentState == State.Walking)
            {
                currentState = State.Idle;
                OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
            }
            else if (currentState == State.Idle)
            {
                idleTimer += Time.deltaTime;
                if (idleTimer > maxIdleTimer)
                {
                    currentState = State.Sitting;
                    OnStateChanged?.Invoke(this, new OnStateChangedEventArgs { stateSent = currentState });
                    idleTimer = 0;
                }
            }
        }    
    }

  
    public StationaryObject GetCurrentCrate()
    {
        return currentMainObject;
    }

    public State getState()
    {
        return currentState;
    }
    //Interface Method Completion
    public void SetCarriableItem(CarriableItem carriableItem)
    {
        currentCarriableItem = carriableItem;
    }

    public void ClearCarriableItem()
    {
        currentCarriableItem = null;
    }
    public bool HasCarriableItem()
    {
        return (currentCarriableItem != null);
    }
    public CarriableItem GetCarriableItem()
    {
        return currentCarriableItem;
    }
    public Transform GetSpawnPoint()
    {
        return foxSpawnPoint;
    }
    public void ResetTimer()
    {
        interacttimer = 0;
    }

    public void SetCarrying(bool carrying) {
        this.carrying = carrying;
    }

    public bool IsCarrying()
    {
        return carrying;
    }

    public void HandleMovementSpeed() {
        if (carrying) { playerSpeed = 4f; }
        else playerSpeed = 12f;
    }
    public Transform GetCarryPoint()
    {
        return foxCarryPoint;
    }
    public void ClearMainObject()
    {
        currentMainObject = null;
    }

    public bool isSpeedActive() 
    {
        return speedActive;
    }

    public bool isMultipPickupActive()
    {
        return multiPickUp;
    }
    
}
    



    /*  public Transform GetCarryingItemTransform()
      {
          return ItemCarryingInstanse;
      }

      public void SetCarriableItemTranssform(Transform carriableItemTransform)
      {

          ItemCarryingInstanse = carriableItemTransform;
      }*/


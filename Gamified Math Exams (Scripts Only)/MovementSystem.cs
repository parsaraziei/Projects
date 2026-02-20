using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class MovementSystem : MonoBehaviour
{
    public event EventHandler onPlayerInteract;
    public event EventHandler onPlayerCarry;
    public event EventHandler onPlayerJump;
    private PlayerMovement inputSystem;
    private float ButtonHeldTimer;
    private float MaximumButtonHeldTimer = 2f;
    private float exittingTimer = 0f;
    private float maxExittingTimer = 2f;
    public event EventHandler<OnTypingChangedEventArgs> OnTypingChanged; 
    public event EventHandler<OnExittingChangedEventArgs> OnExittingChanged;
    public event EventHandler OnPlayVoice;
    private bool HoldCheck = true;
    private bool HoldVerify = true;
    public bool isPlayerDisabled = false;
    private float disableTimer = 0f;
    private float disableTimerMax = 2f;

    public event EventHandler<OnPlayerCrashEventArgs> OnPlayerCrash;
    public class OnPlayerCrashEventArgs
    {
        public bool playerDisabled;
    }
    public class OnTypingChangedEventArgs
    {
        public float progress;

    }

    public class OnExittingChangedEventArgs
    {
        public float progress;
    }
    // Start is called before the first frame update
    public static MovementSystem Instance { private set; get; }

    private void Awake()
    {
        inputSystem = new PlayerMovement();
        inputSystem.Player.Enable();

        inputSystem.Player.Jump.performed += Jump_performed;
        inputSystem.Player.Interact.performed += Interact_performed;
        inputSystem.Player.Carry.performed += Carry_performed;
        inputSystem.Player.Read.performed += Read_performed;
        }



    private void Update()
    {
        if (isPlayerDisabled) { 
            if(disableTimer <= disableTimerMax)
            {
                disableTimer += Time.deltaTime;
            }
            else {
                disableTimer = 0;
                inputSystem.Player.Enable();
                isPlayerDisabled = false;
                OnPlayerCrash?.Invoke(this, new OnPlayerCrashEventArgs { playerDisabled = isPlayerDisabled });
            }
        }

    }
    public void DisableMovement()
    {
        inputSystem.Player.Disable();
        inputSystem.Player.Read.Enable();
    }

    private void Read_performed(UnityEngine.InputSystem.InputAction.CallbackContext obj)
    {
        OnPlayVoice?.Invoke(this, EventArgs.Empty);
    }

  
    private void Carry_performed(UnityEngine.InputSystem.InputAction.CallbackContext obj)
    {
        onPlayerCarry?.Invoke(this, EventArgs.Empty);
            
    }

    private void Interact_performed(UnityEngine.InputSystem.InputAction.CallbackContext obj)
    {
        onPlayerInteract?.Invoke(this, EventArgs.Empty);
    }

    private void Jump_performed(UnityEngine.InputSystem.InputAction.CallbackContext obj)
    {
        onPlayerJump?.Invoke(this, EventArgs.Empty);
    }

    public Vector3 GetMovementVector()
    {
        Vector2 movementDirection2D = inputSystem.Player.Move.ReadValue<Vector2>();
        Vector3 movementDirection3D = new Vector3(movementDirection2D.x, 0 , movementDirection2D.y);
        return movementDirection3D;
    }
    

    public bool Viewing()
    {
        return (inputSystem.Player.View.ReadValue<float>() != 0);
    }

    public void Exiting()
    {
        if (inputSystem.Player.Type.ReadValue<float>() != 0)
        {
            if (exittingTimer < maxExittingTimer && HoldVerify)
            {

                exittingTimer += Time.deltaTime * 1.4f;
                OnExittingChanged?.Invoke(this, new OnExittingChangedEventArgs()
                {
                    progress = exittingTimer / maxExittingTimer
                });
                //Debug.Log(exittingTimer);
            }
            else if (HoldVerify)
            {
                //Debug.Log("success");
                OnExittingChanged?.Invoke(this, new OnExittingChangedEventArgs()
                {
                    progress = exittingTimer / maxExittingTimer
                });
                exittingTimer = 0;
                exittingTimer = 0;
                HoldVerify = false;

            }
        }
        else
        {
           exittingTimer = 0;
            OnExittingChanged?.Invoke(this, new OnExittingChangedEventArgs()
            {
                progress = exittingTimer / maxExittingTimer
            });

            HoldVerify = true;
        }
    }

    public void KeyEHeld()
    {
        
        if(inputSystem.Player.Type.ReadValue<float>() != 0) {
            if(ButtonHeldTimer < MaximumButtonHeldTimer && HoldCheck) {

                ButtonHeldTimer += Time.deltaTime;
                OnTypingChanged?.Invoke(this, new OnTypingChangedEventArgs()
                {
                    progress = ButtonHeldTimer / MaximumButtonHeldTimer
                });
               //Debug.Log(ButtonHeldTimer);
            }
            else if(HoldCheck){
                //Debug.Log("success");
                OnTypingChanged?.Invoke(this, new OnTypingChangedEventArgs()
                {
                    progress = ButtonHeldTimer / MaximumButtonHeldTimer
                });
                ButtonHeldTimer = 0;
                ButtonHeldTimer = 0;
                HoldCheck = false;
                
            }
        }
        else
        {
            ButtonHeldTimer = 0;
            OnTypingChanged?.Invoke(this, new OnTypingChangedEventArgs()
            {
                progress = ButtonHeldTimer / MaximumButtonHeldTimer
            });
           
            HoldCheck = true;
        }   

    }

    public void DisableTemporarely() 
    {
        disableTimer = 0f;
        isPlayerDisabled = true;
        OnPlayerCrash?.Invoke(this, new OnPlayerCrashEventArgs { playerDisabled = isPlayerDisabled });
        inputSystem.Player.Disable();
    }

}


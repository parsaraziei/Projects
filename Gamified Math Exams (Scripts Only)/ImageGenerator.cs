using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;
using System.Threading.Tasks;

public class ImageGenerator : MonoBehaviour
{
    [SerializeField] private Transform imageLayout;
    [SerializeField] private MovementSystem movementSystem;
    [SerializeField] private Player player;
    private bool isEnlarged = false;
    private float transparecnyTimer = 0;


    private void Start()
    {
        Player.Instance.OnStateChanged += Instance_OnStateChanged;
        imageLayout.gameObject.SetActive(false);
        }

    private void Update()
    {
        if(transform.childCount > 1 && movementSystem.Viewing()){ UpscaleImage(); isEnlarged = true; }
        else if(transform.childCount > 1 && !movementSystem.Viewing() && isEnlarged) { ResetImage(); isEnlarged = false; }
        if (transform.childCount > 1){
            if (player.getState() == State.Sitting | player.getState() == State.Idle) {
            if (transparecnyTimer > 4f)
            {
                    foreach (Transform child in transform)
                    {
                        Color color = child.GetComponentInChildren<Image>().color;
                        color.a = 0.8f;
                        child.GetComponentInChildren<Image>().color = color;
                    }
                }
                else
                {
                    transparecnyTimer += Time.deltaTime;
                    //Debug.Log(transparecnyTimer);
                }

            }
            else { transparecnyTimer = 0; }
        }
        else { transparecnyTimer = 0; }
        
    }

    private void UpscaleImage()
    {
        foreach(Transform child in transform)
        {
            HandleSizing(transform.childCount - 1, child);
            Color color = child.GetComponentInChildren<Image>().color;
            color.a = 0.8f;
            child.GetComponentInChildren<Image>().color = color;
        }
    }

    private void ResetImage()
    {
        foreach (Transform child in transform)
        {
            child.localScale = new Vector3(0.5f, 0.5f, 0.5f);
            child.parent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(150, 20);
            Color color = child.GetComponentInChildren<Image>().color;
            color.a = 0.5f;
            child.GetComponentInChildren<Image>().color = color;

        }
    }
    private void Instance_OnStateChanged(object sender, Player.OnStateChangedEventArgs e)
    {
        if (e.stateSent == State.Walking && !isEnlarged)
        {
            foreach (Transform child in transform)
            {

                Color color = child.GetComponentInChildren<Image>().color;
                color.a = 0.2f;
                child.GetComponentInChildren<Image>().color = color;
                //child.gameObject.SetActive(false);
            }
        }
       
    }


  /*  private void FixTransparency()
    {
        if (player.getState() == State.Walking && !isEnlarged)
        {
            foreach (Transform child in transform)
            {

                Color color = child.GetComponentInChildren<Image>().color;
                color.a = 0.2f;
                child.GetComponentInChildren<Image>().color = color;
                //child.gameObject.SetActive(false);
            }
        }
        else
        {
            foreach (Transform child in transform)
            {

                Color color = child.GetComponentInChildren<Image>().color;
                color.a = 0.8f;
                child.GetComponentInChildren<Image>().color = color;
            }
        }

    }*/

    public  async void GenratePictures(List<string> imageUrls)
    {
        foreach (Transform child in transform)
        {
            if (child != imageLayout) { Destroy(child.gameObject); }
        }
        int imageCount = imageUrls.Count;
        foreach (string url in imageUrls)
        {
          Sprite loadedSprite = await LoadImageFromUrl(url);
          if (loadedSprite != null)
           {
          Transform imageInfo = Instantiate(imageLayout, transform);
          imageInfo.gameObject.SetActive(true);
          imageInfo.GetComponentInChildren<Image>().sprite = loadedSprite;
          imageInfo.localScale = new Vector3(0.5f, 0.5f, 0.5f);
          imageInfo.parent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(150, 20);
          //HandleSizing(imageCount,imageInfo);
            }     
        }  
    }

    private void HandleSizing(int count, Transform image)
    {

        switch (count)
        {
            case 1:
                image.localScale = new Vector3(1.4f, 1.4f, 1.4f);
                image.parent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(400,20);
                /*image.parent.GetComponent<>*/
                break;
            case 2:
                image.localScale = new Vector3(1.3f, 1.3f, 1.3f);
                image.parent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(380, 20);
                break;

            case 3:
                image.localScale = new Vector3(1.1f, 1.1f, 1.1f);
                image.parent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(350, 20);
                break;

            default:
                image.localScale = new Vector3(1f, 1f, 1f);
                image.parent.GetComponent<GridLayoutGroup>().cellSize = new Vector2(300, 20);
                break;
        }

    }

   
    public async Task<Sprite> LoadImageFromUrl(string imageUrl)
    {
        UnityWebRequest request = UnityWebRequestTexture.GetTexture(imageUrl);
        var operation = request.SendWebRequest();

        // Wait until the operation is done
        while (!operation.isDone)
        {
            await Task.Yield();
        }

        if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
        {
            Debug.LogError("Failed to load image: " + request.error);
            return null;
        }
        else
        {
            // Get the downloaded texture
            Texture2D texture = DownloadHandlerTexture.GetContent(request);

            // Create a sprite from the texture
            Sprite sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.one * 0.5f);

            // Return the sprite
            return sprite;
        }
    }

}


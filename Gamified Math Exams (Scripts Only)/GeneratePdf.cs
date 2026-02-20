using UnityEngine;
using System.IO;
using UnityEngine.UI;

public class GeneratePdf : MonoBehaviour
{
    public Canvas canvasContainingScrollView;
    public ScrollRect scrollRectToCapture;
    public string screenshotFileName = "scrollViewScreenshot.png";

    public void GeneratePDF()
    {
        // Calculate the size of the content to be captured
        RectTransform contentRect = scrollRectToCapture.content.GetComponent<RectTransform>();
        Vector2 contentSize = contentRect.sizeDelta;

        // Create a render texture matching the size of the content
        RenderTexture renderTexture = new RenderTexture((int)contentSize.x, (int)contentSize.y, 24);
        renderTexture.Create();

        // Set the camera to render the content of the canvas to the render texture
        Camera canvasCamera = canvasContainingScrollView.worldCamera;
        canvasCamera.targetTexture = renderTexture;

        // Render the content
        canvasCamera.Render();

        // Read the pixels from the render texture
        Texture2D texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
        RenderTexture.active = renderTexture;
        texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
        texture.Apply();

        // Save the texture as an image file
        byte[] bytes = texture.EncodeToPNG();
        string desktopPath = System.Environment.GetFolderPath(System.Environment.SpecialFolder.Desktop);
        string filePath = Path.Combine(desktopPath, screenshotFileName);
        File.WriteAllBytes(filePath, bytes);

        // Clean up
        RenderTexture.active = null;
        canvasCamera.targetTexture = null;

        Debug.Log("Screenshot saved to: " + filePath);
    }
}
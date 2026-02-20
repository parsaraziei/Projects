using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using System.IO;
using System;
using System.Runtime.InteropServices;
public class FileManager : MonoBehaviour
{
    [SerializeField] private Button fileExplorer;
    [SerializeField] private TextMeshProUGUI label;
    private string path;

    private string initialDirectory; 

    private void OnDestroy()
    {
        Environment.CurrentDirectory = initialDirectory; 
    }
    private void Start()
    {

        initialDirectory = Environment.CurrentDirectory; 

        if (ExamType.ExamFilePath != "") label.text = ExamType.ExamFilePath;
        else label.text = "Exam File Path";
        fileExplorer.onClick.AddListener(() => {
           
            path = OpenFilePicker();
            if (path != null)
            {
                //path = EditorUtility.OpenFilePanel("Exam File", "", "json");
                ExamType.ExamFilePath = path;
                if (ExamType.ExamFilePath != "") label.text = ExamType.ExamFilePath;
                else label.text = "Exam File Path";
            }
        });
    }

    public string OpenFilePicker()
    {
        string path = ShowFileDialog();
        if (!string.IsNullOrEmpty(path))
        {
            Debug.Log("Selected file path: " + path);
            // Do something with the file path...
        }
        return path;
    }

    private string ShowFileDialog()
    {
        // Initialize open file dialog parameters
        OpenFileDialogParams ofn = new OpenFileDialogParams();
        ofn.structSize = Marshal.SizeOf(ofn);
        ofn.filter = "JSON Files (*.json)\0*.json\0All Files\0*.*\0"; // Filter for JSON files
        ofn.file = new string(new char[256]); // File name buffer
        ofn.maxFile = ofn.file.Length;
        ofn.fileTitle = new string(new char[64]); // File title buffer
        ofn.maxFileTitle = ofn.fileTitle.Length;
        ofn.initialDir = UnityEngine.Application.dataPath; // Initial directory
        ofn.title = "Open File"; // Dialog title
        ofn.flags = 0x00080000 | 0x00001000; // Flags: OFN_PATHMUSTEXIST, OFN_FILEMUSTEXIST

        // Show open file dialog
        if (GetOpenFileName(ofn))
        {
            return ofn.file;
        }
        else
        {
            return null;
        }
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    public class OpenFileDialogParams
    {
        public int structSize;
        public IntPtr dlgOwner;
        public IntPtr instance;
        public string filter;
        public string customFilter;
        public int maxCustFilter;
        public int filterIndex;
        public string file;
        public int maxFile;
        public string fileTitle;
        public int maxFileTitle;
        public string initialDir;
        public string title;
        public int flags;
        public short fileOffset;
        public short fileExtension;
        public string defExt;
        public IntPtr custData;
        public IntPtr hook;
        public string templateName;
        public IntPtr reservedPtr;
        public int reservedInt;
        public int flagsEx;
    }

    [DllImport("Comdlg32.dll", SetLastError = true, CharSet = CharSet.Auto)]
    public static extern bool GetOpenFileName([In, Out] OpenFileDialogParams ofn);
}


    


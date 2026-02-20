using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.IO;
using System;
using System.IO;

public class JSONReader : MonoBehaviour
{

    public event EventHandler OnDataProcessed; 
    public List<Question> questionList = new List<Question>();
    public List<Question> questionListIntact = new List<Question>();
    public List<gradeBoundaries> boundaries =  new List<gradeBoundaries>();
    [SerializeField] private ExitDoorOpen door;
    [SerializeField] private MovementSystem movementSystem;
    public int examTime = 0;
    public bool questionsLoad = true;
    public int Marks;
    public float timer = 0f;
    public float maxTimer = 3f;


    public static JSONReader Instance { get; private set; }

    private void Update()
    {
        if (!questionsLoad)
        {
            if (timer < maxTimer) timer += Time.deltaTime;
            else { door.OpenDoor(); movementSystem.DisableMovement(); }
            
        }
    }

    private void Start()
    {
        Debug.Log(ExamType.ExamFilePath);
        TextAsset jsonFile = LoadJSONAsTextAsset(ExamType.ExamFilePath);
        //TextAsset jsonFile = Resources.Load<TextAsset>("data");
        if (jsonFile != null)
        {
            
            string jsonText = jsonFile.text;
            try
            {
              Question[] questionArray = JsonUtility.FromJson<QuestionList>(jsonText).questions;

                //gradeBoundary[] boundaries = JsonUtility.FromJson<gradeBoundaries>(jsonText).gradeBoundary;
                ExamSheet examSheet = JsonUtility.FromJson<ExamSheet>(jsonText);
                // Debug.Log(examSheet.gradeBoundaries.gradeBoundary.Length);
                // QuestionList allQuestionList = examSheet.questions;
                try
                {
                    foreach (Question query in questionArray)
                    {

                        questionList.Add(query);
                        questionListIntact.Add(query);
                    }
                }
                catch (System.NullReferenceException)
                {
                    questionsLoad = false;
                }


                OnDataProcessed?.Invoke(this, EventArgs.Empty);
                try
                {

                    foreach (gradeBoundaries grades in JsonUtility.FromJson<gradesArray>(jsonText).gradeBoundaries)
                    {
                        boundaries.Add(grades);
                        /*Debug.Log(grades.grade);*/
                    }

                    examTime = examSheet.totalTime;
                    Marks = examSheet.totalMarks;
                }
                catch (System.NullReferenceException)
                {
                    examTime = 0;
                    questionsLoad = false;
                }


            } catch(ArgumentException e)
            { 
                Debug.LogError("JSON parse error: " + e.Message);
                questionsLoad = false;
            }
   /*Debug.Log(GetExamTime());*/
            
            
        }
        else
        {
            Debug.LogError("Failed to load JSON file!");
        }
        if (questionList.Count == 0) questionsLoad = false;
    }

    TextAsset LoadJSONAsTextAsset(string filePath)
    {
        if (File.Exists(filePath))
        {
            try
            {
                // Read the JSON file as text
                string jsonText = File.ReadAllText(filePath);

                // Create a new TextAsset instance with the JSON text
                TextAsset textAsset = new TextAsset(jsonText);

                // Return the TextAsset
                return textAsset;
            }
            catch (System.Exception e)
            {
                Debug.LogError("Error loading JSON file: " + e.Message);
                questionsLoad = false;
            }
        }
        else
        {
            Debug.LogError("JSON file not found at path: " + filePath);
            questionsLoad = false;
        }

        return null;
    }
    public List<gradeBoundaries> GetGradeBoundaries()
    { return boundaries; }
    public int GetExamTime()
    { return examTime; }

    public int GetFullMarks() {
        return Marks;
    }

    public List<Question> GetQuestions() {
        return questionList;
            }
    public List<Question> GetQuestionsIntact()
    {
        return questionListIntact;
    }
}



[System.Serializable]
public class ExamSheet
{
    public int totalMarks;
    public int totalTime;
    public gradeBoundaries gradeBoundaries;
    public QuestionList questions;
}


[System.Serializable]
public class Question
{
    public string questionType;
    public bool isOrderImportant;
    public QuestionBody body;
    public int[] solutions;
    public int marks;
}


[System.Serializable]
public class gradesArray
{
    public gradeBoundaries[] gradeBoundaries;
}


[System.Serializable]
public class gradeBoundaries
{
    public string grade;
    public int[] gradeBorders;
}

[System.Serializable]
public class QuestionList
{
    public Question[] questions;
}

[System.Serializable]
public class QuestionBody
{
    public string text;
    public string[] image;
}



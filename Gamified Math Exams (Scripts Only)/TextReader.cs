using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using SpeechLib;

public class TextReader : MonoBehaviour
{

    [SerializeField] private QuestionBuilder questionBuilder;
    [SerializeField] private MovementSystem movementSystem;
    [SerializeField] private Content content;

    SpVoice voice = new SpVoice();
    private void Start()
    {
        movementSystem.OnPlayVoice += Instance_OnPlayVoice;
    }

    private void Instance_OnPlayVoice(object sender, System.EventArgs e)
    {
        ReadQustion();
        Debug.Log(questionBuilder.GetIsGameOver());
    } 



    private void ReadQustion()
    {
        /*voice.Voice = voice.GetVoices().Item(1);*/
        voice.Rate = -2;
        if (!questionBuilder.GetIsGameOver())
        {
            Debug.Log("1");
            if (questionBuilder.GetCurrentQuestion() != null)
            {
                if (questionBuilder.GetCurrentQuestion().body.text != "")
                {
                    string line = (questionBuilder.GetCurrentQuestion().body.text).Replace("??", "blank");
                    line = line.Replace("-", "minus");
                    line = line.Replace("*", "multiplied");

                    voice.Speak(line, SpeechVoiceSpeakFlags.SVSFlagsAsync | SpeechVoiceSpeakFlags.SVSFPurgeBeforeSpeak);
                }
            }
        }
        else
        {
            Debug.Log("2");
            string currentSegment = "";
            string currentLine;
            if (content.GetDropdown().value == 0)
            {
                if (!content.GetStudentGradeInfo()[1].Contains("reset"))
                {
                    foreach (string voiceline in content.GetStudentGradeInfo())
                    {
                        currentLine = voiceline.Replace("/", "Out Of");
                        if (currentLine.Contains("Time Took"))
                        {
                            string[] parts = voiceline.Substring(9).Split(":");
                            currentLine = currentLine.Substring(0, 9) + (int.Parse(parts[0])).ToString() + "minutes and " + (int.Parse(parts[1])).ToString() + "seconds";
                        }
                        currentSegment += currentLine + "\n";

                    }
                    voice.Speak(currentSegment, SpeechVoiceSpeakFlags.SVSFlagsAsync | SpeechVoiceSpeakFlags.SVSFPurgeBeforeSpeak);
                }
            }
        }
    }
}

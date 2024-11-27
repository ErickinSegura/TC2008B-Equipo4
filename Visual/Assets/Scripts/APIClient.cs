using UnityEngine;
using UnityEngine.Networking;
using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;

public class APIClient : MonoBehaviour
{
    public string apiUrl = "https://example.com/api/getBots"; // URL del API
    public List<Bot> bots; // Lista de bots deserializados

    void Start()
    {
        StartCoroutine(FetchBotsFromAPI());
    }

    IEnumerator FetchBotsFromAPI()
    {
        using (UnityWebRequest request = UnityWebRequest.Get(apiUrl))
        {
            yield return request.SendWebRequest();

            if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError($"Error al conectar con el API: {request.error}");
            }
            else
            {
                string jsonResponse = request.downloadHandler.text;
                Debug.Log($"Datos del API: {jsonResponse}");
                BotList botList = JsonConvert.DeserializeObject<BotList>(jsonResponse);
                bots = botList.bots;
                InitializeBots();
            }
        }
    }

    void InitializeBots()
    {
        foreach (var bot in bots)
        {
            Debug.Log($"Bot {bot.id} con prioridad {bot.priority}");
            foreach (var point in bot.current_path)
            {
                Debug.Log($"Coordenada: ({point.x}, {point.y})");
            }
        }
    }
}

using UnityEngine;
using System.Collections.Generic;

public class BotController : MonoBehaviour
{
    public List<Vector3> pathPoints;
    public float speed = 2f;
    private int currentIndex = 0;

    void Update()
    {
        if (currentIndex < pathPoints.Count)
        {
            transform.position = Vector3.MoveTowards(transform.position, pathPoints[currentIndex], speed * Time.deltaTime);

            if (Vector3.Distance(transform.position, pathPoints[currentIndex]) < 0.1f)
            {
                currentIndex++;
            }
        }
    }

    public void SetPath(List<PathPoint> path)
    {
        pathPoints = new List<Vector3>();
        foreach (var point in path)
        {
            pathPoints.Add(new Vector3(point.x, 0, point.y));
        }
    }
}

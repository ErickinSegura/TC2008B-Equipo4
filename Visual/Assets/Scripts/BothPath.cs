using System.Collections.Generic;

[System.Serializable]
public class PathPoint
{
    public float x;
    public float y;
}

[System.Serializable]
public class Bot
{
    public int id;
    public int priority;
    public List<PathPoint> current_path;
}

[System.Serializable]
public class BotList
{
    public List<Bot> bots;
}

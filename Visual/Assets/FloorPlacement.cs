using UnityEngine;

public class FloorPlacement : MonoBehaviour
{
    // Tamaño del suelo en Unity Units
    public Vector3 tileSize = new Vector3(1.25f, 1f, 1.25f);

    // Método para posicionar los objetos correctamente en la cuadrícula
    public void PlaceTile(GameObject tilePrefab, Vector3 position)
    {
        // Alinear la posición a la cuadrícula
        float x = Mathf.Floor(position.x / tileSize.x) * tileSize.x;
        float y = position.y; // Mantener la altura
        float z = Mathf.Floor(position.z / tileSize.z) * tileSize.z;

        // Nueva posición alineada
        Vector3 alignedPosition = new Vector3(x, y, z);

        // Instanciar el objeto en la posición alineada
        Instantiate(tilePrefab, alignedPosition, Quaternion.identity);
    }
}

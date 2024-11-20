using UnityEngine;

public class Transformation : MonoBehaviour
{
    public Vector3 corner1;
    public Vector3 corner2;
    public Vector3 corner3;
    public Vector3 corner4;
    public GameObject robot;
    public float moveSpeed = 5f;      
    public float rotationSpeed = 5f;   
    
    private Vector3 targetPosition;
    private int currentCorner = 0;
    private bool isRotating = false;
    private Quaternion targetRotation;

    void Start()
    {
        if (robot == null)
        {
            Debug.LogError("Por favor asigna un objeto para 'robot' en el inspector.");
            return;
        }
        
        robot.transform.position = corner1;
        targetPosition = corner2;

        // Ajuste inicial de rotación
        UpdateTargetRotation();
        robot.transform.rotation = targetRotation;
    }

    void Update()
    {
        Vector3 direction = targetPosition - robot.transform.position;

        if (isRotating)
        {
            robot.transform.rotation = Quaternion.Slerp(
                robot.transform.rotation,
                targetRotation,
                rotationSpeed * Time.deltaTime
            );

            if (Quaternion.Angle(robot.transform.rotation, targetRotation) < 1f)
            {
                isRotating = false;
            }
            return; 
        }

        if (direction.magnitude < 0.1f)
        {
            currentCorner = (currentCorner + 1) % 4;
            
            switch(currentCorner)
            {
                case 0:
                    targetPosition = corner1;
                    break;
                case 1:
                    targetPosition = corner2;
                    break;
                case 2:
                    targetPosition = corner3;
                    break;
                case 3:
                    targetPosition = corner4;
                    break;
            }

            UpdateTargetRotation();
            isRotating = true;
            return;
        }

        if (!isRotating)
        {
            robot.transform.position = Vector3.MoveTowards(
                robot.transform.position,
                targetPosition,
                moveSpeed * Time.deltaTime
            );
        }
    }

    private void UpdateTargetRotation()
    {
        Vector3 direction = targetPosition - robot.transform.position;
        if (direction != Vector3.zero)
        {
            // Ajuste de -90 grados en el eje Y
            targetRotation = Quaternion.LookRotation(direction) * Quaternion.Euler(0, -90, 0);
        }
    }

    void OnDrawGizmos()
    {
        Gizmos.color = Color.yellow;
        Gizmos.DrawLine(corner1, corner2);
        Gizmos.DrawLine(corner2, corner3);
        Gizmos.DrawLine(corner3, corner4);
        Gizmos.DrawLine(corner4, corner1);
        
        Gizmos.color = Color.red;
        Gizmos.DrawSphere(corner1, 0.1f);
        Gizmos.DrawSphere(corner2, 0.1f);
        Gizmos.DrawSphere(corner3, 0.1f);
        Gizmos.DrawSphere(corner4, 0.1f);
    }
}

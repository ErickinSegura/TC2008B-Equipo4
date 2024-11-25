import pygame
import numpy as np
import heapq

x_min, x_max = -4, 4
y_min, y_max = -3, 3
scale = 100

obstacles = [
    {"x": [-0.3952, -1.0048, -1.0048, -0.3952], "y": [-0.1952, -0.1952, -0.8048, -0.8048], "color": (255, 165, 0)},
    {"x": [0.3048, -0.3048, -0.3048, 0.3048], "y": [-0.1952, -0.1952, -0.8048, -0.8048], "color": (255, 0, 0)},
    {"x": [0.3048, -0.3048, -0.3048, 0.3048], "y": [1.0048, 1.0048, 0.3952, 0.3952], "color": (0, 0, 255)},
    {"x": [1.5, 1.0689, 1.5, 1.9311], "y": [0.4311, 0, -0.4311, 0], "color": (0, 255, 0)}
]

targets = [(-1.25, 0), (0, 1.9), (-0.5, 0), (1.25, -0.4)]

def is_inside_obstacle(x, y):
    for obs in obstacles:
        vertices = list(zip(obs["x"], obs["y"]))
        inside = True
        for i in range(len(vertices)):
            x1, y1 = vertices[i]
            x2, y2 = vertices[(i + 1) % len(vertices)]
            if (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1) > 0:
                inside = False
                break
        if inside:
            return True
    return False

def check_robot_collision(position):
    robot_size = 0.2
    safety_margin = 0.05
    total_size = robot_size + safety_margin
    
    num_points = 32
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = position[0] + total_size * np.cos(angle)
        y = position[1] + total_size * np.sin(angle)
        if is_inside_obstacle(x, y):
            return True
    
    for dx in np.linspace(-total_size, total_size, 5):
        for dy in np.linspace(-total_size, total_size, 5):
            if dx*dx + dy*dy <= total_size*total_size:
                x = position[0] + dx
                y = position[1] + dy
                if is_inside_obstacle(x, y):
                    return True
    return False

def check_line_obstacle_intersection(start, end):
    num_checks = 20
    for i in range(num_checks + 1):
        t = i / num_checks
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        if check_robot_collision((x, y)):
            return True
    return False

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        if distance(current, goal) < step_size:
            return reconstruct_path(came_from, current)
        
        for angle in np.linspace(0, 2 * np.pi, num_angles):
            dx = step_size * np.cos(angle)
            dy = step_size * np.sin(angle)
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (x_min <= neighbor[0] <= x_max and 
                y_min <= neighbor[1] <= y_max and 
                not check_robot_collision(neighbor) and
                not check_line_obstacle_intersection(current, neighbor)):
                
                tentative_g_score = g_score[current] + step_size
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return []

num_angles = 16  # Reduce this number to speed up execution
step_size = 0.15

object_position = (-3, 0)
paths = []
for target in targets:
    path = astar(object_position, target)
    paths.append(path)
    object_position = target

pygame.init()
screen = pygame.display.set_mode(((x_max - x_min) * scale, (y_max - y_min) * scale))
clock = pygame.time.Clock()

def draw_grid():
    for x in range(x_min, x_max + 1):
        pygame.draw.line(screen, (200, 200, 200), ((x - x_min) * scale, 0), ((x - x_min) * scale, (y_max - y_min) * scale))
    for y in range(y_min, y_max + 1):
        pygame.draw.line(screen, (200, 200, 200), (0, (y_max - y) * scale), ((x_max - x_min) * scale, (y_max - y) * scale))

def draw_obstacles():
    for obs in obstacles:
        vertices = [(int((x - x_min) * scale), int((y_max - y) * scale)) for x, y in zip(obs["x"], obs["y"])]
        pygame.draw.polygon(screen, obs["color"], vertices)

def draw_targets():
    for target in targets:
        pygame.draw.circle(screen, (0, 255, 0), (int((target[0] - x_min) * scale), int((y_max - target[1]) * scale)), 10)

def draw_robot(position):
    x, y = position
    vertices = [
        (int((x - 0.15 - x_min) * scale), int((y_max - (y - 0.15)) * scale)),
        (int((x + 0.15 - x_min) * scale), int((y_max - (y - 0.15)) * scale)),
        (int((x + 0.15 - x_min) * scale), int((y_max - (y + 0.15)) * scale)),
        (int((x - 0.15 - x_min) * scale), int((y_max - (y + 0.15)) * scale))
    ]
    pygame.draw.polygon(screen, (128, 128, 128), vertices)
    pygame.draw.polygon(screen, (0, 0, 0), vertices, 1)
    print(f"Drawing robot at: {position}")

running = True
path_index = 0
pos_index = 0
current_pos = object_position

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((255, 255, 255))
    draw_grid()
    draw_obstacles()
    draw_targets()

    if path_index < len(paths) and paths[path_index]:
        if pos_index < len(paths[path_index]):
            current_pos = paths[path_index][pos_index]
            draw_robot(current_pos)
            pos_index += 1
        else:
            path_index += 1
            pos_index = 0
    else:
        draw_robot(current_pos)

    pygame.display.flip()
    clock.tick(5)

pygame.quit()
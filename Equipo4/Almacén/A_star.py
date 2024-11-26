from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
import random
import heapq
import networkx as nx
import json

desc = [
"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
"BFFFFFFFFFFSFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BBFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFBBFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFGFFFFFFFFFFFFFFFFFFFFGFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGB",
"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
]

def from_desc_to_maze(desc):
    start_positions = []
    goals = []
    walls = []
    width = len(desc[0])
    height = len(desc)
    start_row = None
    for y, row in enumerate(desc):
        for x, cell in enumerate(row):
            if cell == 'B':
                walls.append((x, y))
            elif cell == 'S':
                start_positions.append((x, y))
                start_row = y
            elif cell == 'F' and y == start_row:
                start_positions.append((x, y))
            elif cell == 'G':
                goals.append((x, y))
    return walls, start_positions, goals, width, height

class CollisionCounter(TextElement):
    def render(self, model):
        return f"Número de colisiones: {model.num_collisions}"

class CentralControlAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.robot_agents = []
        self.global_graph = self.create_navigation_graph()
        self.reservations = {}

    def plan_routes(self):
        for robot in self.robot_agents:
            if robot.needs_replan:
                for pos, t in list(self.reservations.keys()):
                    if self.reservations[(pos, t)] == robot.unique_id:
                        del self.reservations[(pos, t)]

        routes = {}
        for robot in sorted(self.robot_agents, key=lambda x: x.priority):
            if robot.state == 'finished':
                continue

            if robot.waiting or robot.needs_replan:
                current_pos = robot.pos
                target_pos = robot.goal_pos

                path = self.find_path_with_reservations(current_pos, target_pos, robot.priority)
                if path:
                    routes[robot.unique_id] = path
                    for t, pos in enumerate(path):
                        self.reservations[(pos, t)] = robot.unique_id
                    robot.path = path
                    robot.step_index = 0
                    robot.waiting = False
                    robot.needs_replan = False
                    print(f"Agente {robot.unique_id} planificó ruta: {path}")
                else:
                    print(f"No se encontró ruta para el robot {robot.unique_id}")
                    robot.waiting = True

    def create_navigation_graph(self):
        G = nx.grid_2d_graph(self.model.grid.width, self.model.grid.height)
        for wall in self.model.walls:
            if wall in G:
                G.remove_node(wall)
        return G

    def register_robot(self, robot):
        self.robot_agents.append(robot)

    def get_finished_agents_positions(self):
        finished_positions = set()
        for robot in self.robot_agents:
            if robot.state == 'finished':
                finished_positions.add(robot.pos)
        return finished_positions

    def find_path_with_reservations(self, start, goal, priority):
        finished_positions = self.get_finished_agents_positions()
        open_list = []
        heapq.heappush(open_list, (0, start, 0))
        came_from = {}
        g_score = {(start, 0): 0}

        while open_list:
            _, current, t = heapq.heappop(open_list)
            if current == goal:
                path = []
                while (current, t) in came_from:
                    path.append(current)
                    current, t = came_from[(current, t)]
                path.append(start)
                return path[::-1]

            neighbors = [
                (current[0] + dx, current[1] + dy)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if (0 <= current[0] + dx < self.model.grid.width and
                    0 <= current[1] + dy < self.model.grid.height and
                    (current[0] + dx, current[1] + dy) not in self.model.walls and
                    (current[0] + dx, current[1] + dy) not in finished_positions)
            ] + [current]

            for neighbor in neighbors:
                next_time = t + 1
                reserved_by = self.reservations.get((neighbor, next_time))
                if reserved_by is not None and reserved_by != self.unique_id:
                    other_robot = next((r for r in self.robot_agents if r.unique_id == reserved_by), None)
                    if other_robot and other_robot.priority <= priority:
                        continue

                if (neighbor, next_time) in [(current, t-1)]:
                    continue

                tentative_g = g_score[(current, t)] + 1
                if (neighbor, next_time) not in g_score or tentative_g < g_score[(neighbor, next_time)]:
                    came_from[(neighbor, next_time)] = (current, t)
                    g_score[(neighbor, next_time)] = tentative_g
                    f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    heapq.heappush(open_list, (f_score, neighbor, next_time))
        return None

    def step(self):
        self.plan_routes()

class MovingAgent(Agent):
    def __init__(self, unique_id, model, start_pos, goal_pos, priority):
        super().__init__(unique_id, model)
        self.start_pos = start_pos
        self.pos = start_pos
        self.goal_pos = goal_pos
        self.path = []
        self.step_index = 0
        self.state = 'to_goal'
        self.waiting = True
        self.collisions = 0
        self.needs_replan = True
        self.priority = priority
        self.wait_steps = 0
        self.safe_distance = 4
        self.stuck_counter = 0
        self.max_stuck_time = 5
        self.last_position = start_pos
        self.position_unchanged_counter = 0
        self.full_path = []
        self.full_path.append(start_pos)
        print(f"Agente {self.unique_id} iniciado en {self.start_pos} con objetivo {self.goal_pos}, prioridad {self.priority}")

    def get_distance_to(self, other_pos):
        return abs(self.pos[0] - other_pos[0]) + abs(self.pos[1] - other_pos[1])

    def get_orthogonal_neighbors(self, position, radius=1):
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for dx, dy in directions:
            for r in range(1, radius + 1):
                new_x = position[0] + (dx * r)
                new_y = position[1] + (dy * r)
                if (0 <= new_x < self.model.grid.width and 
                    0 <= new_y < self.model.grid.height):
                    neighbors.append((new_x, new_y))
        
        return neighbors

    def detect_nearby_agents(self):
        nearby_agents = []
        check_positions = self.get_orthogonal_neighbors(self.pos, self.safe_distance)
        
        for check_pos in check_positions:
            cell_agents = self.model.grid.get_cell_list_contents([check_pos])
            for agent in cell_agents:
                if isinstance(agent, MovingAgent) and agent.unique_id != self.unique_id:
                    if agent.state == 'finished':
                        return [(agent, agent.pos)]
                    else:
                        nearby_agents.append((agent, check_pos))
        return nearby_agents

    def predict_collision(self, next_pos):
        cell_contents = self.model.grid.get_cell_list_contents([next_pos])
        for agent in cell_contents:
            if isinstance(agent, MovingAgent) and agent.state == 'finished':
                return True, agent

        nearby_agents = self.detect_nearby_agents()
        for agent, agent_pos in nearby_agents:
            if agent.state == 'finished':
                if next_pos == agent_pos:
                    return True, agent
            elif agent.path and agent.step_index < len(agent.path):
                agent_next_pos = agent.path[agent.step_index]
                if (next_pos == agent_next_pos or  
                    (next_pos == agent.pos and agent_next_pos == self.pos)):
                    return True, agent
            elif self.get_distance_to(agent_pos) <= self.safe_distance:
                return True, agent
        return False, None


    def find_alternative_path(self):
        current_pos = self.pos
        target_pos = self.goal_pos
        
        temp_blocked = set()
        neighbors = self.get_orthogonal_neighbors(current_pos, 2)
        temp_blocked.update(neighbors)
        temp_blocked.add(current_pos)
        
        for agent in self.model.schedule.agents:
            if isinstance(agent, MovingAgent) and agent.state == 'finished':
                temp_blocked.add(agent.pos)
        
        G = nx.grid_2d_graph(self.model.grid.width, self.model.grid.height)
        for wall in self.model.walls:
            if wall in G:
                G.remove_node(wall)
        for blocked in temp_blocked:
            if blocked in G:
                G.remove_node(blocked)
        
        try:
            path = nx.shortest_path(G, current_pos, target_pos, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def find_evasive_position(self):
        neighbors = self.get_orthogonal_neighbors(self.pos, 1)
        random.shuffle(neighbors)
        
        neighbors.sort(key=lambda pos: abs(pos[0] - self.goal_pos[0]) + abs(pos[1] - self.goal_pos[1]))
        
        for pos in neighbors:
            if pos in self.model.walls:
                continue
                
            if self.model.grid.is_cell_empty(pos):
                will_collide, _ = self.predict_collision(pos)
                if not will_collide:
                    return pos
        return None

    def check_if_stuck(self):
        if self.pos == self.last_position:
            self.position_unchanged_counter += 1
        else:
            self.position_unchanged_counter = 0
            self.last_position = self.pos

        if self.position_unchanged_counter >= self.max_stuck_time:
            print(f"Agente {self.unique_id} estancado en {self.pos}. Buscando ruta alternativa...")
            return True
        return False

    def find_alternative_path(self):
        current_pos = self.pos
        target_pos = self.goal_pos
        
        temp_blocked = set()
        neighbors = self.get_orthogonal_neighbors(current_pos, 2)
        temp_blocked.update(neighbors)
        temp_blocked.add(current_pos)
        
        G = nx.grid_2d_graph(self.model.grid.width, self.model.grid.height)
        for wall in self.model.walls:
            if wall in G:
                G.remove_node(wall)
        for blocked in temp_blocked:
            if blocked in G:
                G.remove_node(blocked)
        
        try:
            path = nx.shortest_path(G, current_pos, target_pos, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def step(self):
        if self.state == 'finished':
            return

        if self.waiting:
            return

        if self.check_if_stuck():
            alternative_path = self.find_alternative_path()
            if alternative_path:
                print(f"Agente {self.unique_id} encontró ruta alternativa")
                self.path = alternative_path
                self.step_index = 0
                self.position_unchanged_counter = 0
                self.needs_replan = False
            else:
                print(f"Agente {self.unique_id} no pudo encontrar ruta alternativa, solicitando replanificación")
                self.waiting = True
                self.needs_replan = True
                self.position_unchanged_counter = 0
            return

        if self.step_index < len(self.path):
            next_pos = self.path[self.step_index]
            
            dx = abs(next_pos[0] - self.pos[0])
            dy = abs(next_pos[1] - self.pos[1])
            if dx + dy > 1:
                print(f"Advertencia: Movimiento no ortogonal detectado para agente {self.unique_id}")
                self.waiting = True
                self.needs_replan = True
                return

            will_collide, blocking_agent = self.predict_collision(next_pos)

            if will_collide:
                if blocking_agent and blocking_agent.priority < self.priority:
                    print(f"Agente {self.unique_id} detectó posible colisión con agente de mayor prioridad")
                    evasive_pos = self.find_evasive_position()
                    if evasive_pos:
                        print(f"Agente {self.unique_id} tomando acción evasiva hacia {evasive_pos}")
                        self.model.grid.move_agent(self, evasive_pos)
                        self.pos = evasive_pos
                        self.waiting = True
                        self.needs_replan = True
                    else:
                        self.wait_steps += 1
                        if self.wait_steps >= 2:
                            self.waiting = True
                            self.needs_replan = True
                            self.wait_steps = 0
                else:
                    self.wait_steps += 1
                    if self.wait_steps >= 1:
                        self.waiting = True
                        self.needs_replan = True
                        self.wait_steps = 0
            else:
                self.model.grid.move_agent(self, next_pos)
                self.pos = next_pos
                self.full_path.append(next_pos)
                self.step_index += 1
                self.wait_steps = 0

                if self.pos == self.goal_pos:
                    if self.state == 'to_goal':
                        print(f"Agente {self.unique_id} llegó al objetivo {self.goal_pos}")
                        self.state = 'to_start'
                        self.goal_pos = self.start_pos
                        self.waiting = True
                        self.needs_replan = True
                    elif self.state == 'to_start':
                        print(f"Agente {self.unique_id} regresó al inicio {self.start_pos}")
                        self.state = 'finished'
                        self.goal_pos = self.model.get_new_goal(self)
                        self.waiting = True
                        self.needs_replan = True
        else:
            self.waiting = True
            self.needs_replan = True
            self.wait_steps = 0

class WallAgent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

class GoalAgent(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos

class MultiAgentModel(Model):
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.walls, start_positions, goal_positions, width, height = from_desc_to_maze(desc)
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.num_collisions = 0
        self.datacollector = DataCollector(
            {"Collisions": lambda m: m.num_collisions}
        )
        self.start_positions = start_positions
        self.goal_positions = goal_positions.copy()
        self.available_goals = self.goal_positions.copy()

        central_agent = CentralControlAgent('central_control', self)
        self.central_agent = central_agent
        self.schedule.add(central_agent)

        for idx, wall in enumerate(self.walls):
            wall_agent = WallAgent(f'wall_{idx}', self)
            self.grid.place_agent(wall_agent, wall)

        for idx, goal_pos in enumerate(self.goal_positions):
            goal_agent = GoalAgent(f'goal_{idx}', self, goal_pos)
            self.grid.place_agent(goal_agent, goal_pos)

        available_starts = [pos for pos in self.start_positions if pos not in self.walls]
        agent_starts = random.sample(available_starts, min(self.num_agents, len(available_starts)))

        priorities = list(range(1, self.num_agents + 1))
        random.shuffle(priorities)

        for i, start_pos in enumerate(agent_starts):
            if self.available_goals:
                goal_pos = self.available_goals.pop(0)
            else:
                goal_pos = random.choice(self.goal_positions)
            priority = priorities.pop()
            agent = MovingAgent(i, self, start_pos, goal_pos, priority)
            self.grid.place_agent(agent, start_pos)
            self.schedule.add(agent)
            central_agent.register_robot(agent)

        self.running = True

    def get_new_goal(self, agent):
        if self.available_goals:
            new_goal = self.available_goals.pop(0)
            self.available_goals.append(agent.goal_pos)
        else:
            new_goal = agent.goal_pos
        print(f"Agente {agent.unique_id} asignado a nuevo objetivo {new_goal}")
        return new_goal

    def export_robot_paths(self):
        robot_data = {
            "robots": []
        }
        
        for agent in self.schedule.agents:
            if isinstance(agent, MovingAgent):
                # Convertir las tuplas directamente a pares de números
                path_coords = [[x, y] for x, y in agent.full_path]
                robot_info = {
                    "id": agent.unique_id,
                    "priority": agent.priority,
                    "path": {
                        "coordinates": path_coords,
                        "length": len(path_coords)
                    }
                }
                robot_data["robots"].append(robot_info)
        
        import os
        if not os.path.exists('results'):
            os.makedirs('results')
            
        with open('results/robot_paths.json', 'w') as f:
            json.dump(robot_data, f, indent=4)

    def step(self):
        self.datacollector.collect(self)
        self.central_agent.step()
        for agent in self.schedule.agents:
            if isinstance(agent, MovingAgent):
                agent.step()
        if all(agent.state == 'finished' for agent in self.schedule.agents if isinstance(agent, MovingAgent)):
            self.export_robot_paths()
            self.running = False

def agent_portrayal(agent):
    if isinstance(agent, MovingAgent):
        robot_colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "yellow", "pink"]
        color = robot_colors[agent.unique_id % len(robot_colors)] if agent.state != 'finished' else "grey"

        portrayal = {
            "Shape": "circle",
            "Color": color,
            "Filled": "true",
            "Layer": 2,
            "r": 1  ,
            "text": f"{agent.unique_id}",
            "text_color": "white"
        }
    elif isinstance(agent, WallAgent):
        portrayal = {
            "Shape": "rect",
            "Color": "black",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1
            
        }
    elif isinstance(agent, GoalAgent):
        portrayal = {
            "Shape": "rect",
            "Color": "red",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }
    else:
        portrayal = {}
    return portrayal

grid_width = len(desc[0])
grid_height = len(desc)

cell_size = 20
canvas_width = grid_width * cell_size
canvas_height = grid_height * cell_size

class RobotPositionsElement(TextElement):
    def render(self, model):
        positions_text = ""
        for agent in model.schedule.agents:
            if isinstance(agent, MovingAgent):
                positions_text += f"Robot {agent.unique_id}: {agent.pos}\n"
        return positions_text

grid = CanvasGrid(agent_portrayal, grid_width, grid_height, canvas_width, canvas_height)
collision_counter = CollisionCounter()
robot_positions = RobotPositionsElement()

server = ModularServer(
    MultiAgentModel,
    [grid, collision_counter, robot_positions],
    "Simulación Multiagente con Retorno al Inicio y Evitación Mejorada",
    {"num_agents": 3}
)

server.port = 8521
server.launch()

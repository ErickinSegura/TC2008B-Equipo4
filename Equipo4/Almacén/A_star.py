from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
import random
import heapq
import networkx as nx

desc = [
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
    "BFFFSFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFFFFFFFFFFFFB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFFFFFFFFFFFFB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFFFFFFFFFFFFB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFFFFFFFFFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBBB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBBB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFBBFFBBB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
    "BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFGFB",
    "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
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
        self.reservations = {}  # Tabla de reservas

    def create_navigation_graph(self):
        G = nx.grid_2d_graph(self.model.grid.width, self.model.grid.height)
        for wall in self.model.walls:
            if wall in G:
                G.remove_node(wall)
        return G

    def register_robot(self, robot):
        self.robot_agents.append(robot)

    def plan_routes(self):
        # Liberar reservas antiguas de agentes que necesitan replanificar
        for robot in self.robot_agents:
            if robot.needs_replan:
                for pos, t in list(self.reservations.keys()):
                    if self.reservations[(pos, t)] == robot.unique_id:
                        del self.reservations[(pos, t)]

        routes = {}
        # Asignar prioridades a los agentes
        for robot in sorted(self.robot_agents, key=lambda x: x.priority):
            if robot.state == 'finished':
                continue

            if robot.waiting or robot.needs_replan:
                current_pos = robot.pos
                target_pos = robot.goal_pos

                path = self.find_path_with_reservations(current_pos, target_pos, robot.priority)
                if path:
                    routes[robot.unique_id] = path
                    # Reservar las celdas en la tabla de reservas
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

    def find_path_with_reservations(self, start, goal, priority):
        open_list = []
        heapq.heappush(open_list, (0, start, 0))  # (f_score, position, time)
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
                for dx, dy in [ (0, 1), (1, 0), (0, -1), (-1, 0) ]
                if 0 <= current[0] + dx < self.model.grid.width and
                0 <= current[1] + dy < self.model.grid.height and
                (current[0] + dx, current[1] + dy) not in self.model.walls
            ] + [current]  # Agregar la opción de esperar en el mismo lugar

            for neighbor in neighbors:
                next_time = t + 1
                reserved_by = self.reservations.get((neighbor, next_time))
                if reserved_by is not None and reserved_by != self.unique_id:
                    # Verificar prioridad
                    other_robot = next((r for r in self.robot_agents if r.unique_id == reserved_by), None)
                    if other_robot and other_robot.priority <= priority:
                        continue  # La celda está reservada por un agente con mayor o igual prioridad

                # Evitar intercambios de posición (swap)
                if (neighbor, next_time) in [(current, t-1)]:
                    continue

                tentative_g = g_score[(current, t)] + 1
                if (neighbor, next_time) not in g_score or tentative_g < g_score[(neighbor, next_time)]:
                    came_from[(neighbor, next_time)] = (current, t)
                    g_score[(neighbor, next_time)] = tentative_g
                    f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                    heapq.heappush(open_list, (f_score, neighbor, next_time))
        return None  # No se encontró ruta

    def step(self):
        self.plan_routes()  # Planificar rutas en cada paso

class MovingAgent(Agent):
    def __init__(self, unique_id, model, start_pos, goal_pos, priority):
        super().__init__(unique_id, model)
        self.start_pos = start_pos
        self.pos = start_pos
        self.goal_pos = goal_pos
        self.path = []
        self.step_index = 0
        self.state = 'to_goal'  # Estados: 'to_goal', 'to_start', 'finished'
        self.waiting = True  # Esperar a que se planifique la primera ruta
        self.collisions = 0
        self.needs_replan = True  # Necesita planificar ruta inicialmente
        self.priority = priority  # Prioridad del agente
        self.wait_steps = 0  # Contador de pasos en espera
        print(f"Agente {self.unique_id} iniciado en {self.start_pos} con objetivo {self.goal_pos}, prioridad {self.priority}")

    def step(self):
        if self.state == 'finished':
            return

        if self.waiting:
            return  # El agente espera una nueva ruta

        # Seguir la ruta asignada
        if self.step_index < len(self.path):
            next_pos = self.path[self.step_index]
            # Verificar colisión
            cell_agents = self.model.grid.get_cell_list_contents([next_pos])
            collision = False
            blocking_agent = None
            for agent in cell_agents:
                if isinstance(agent, MovingAgent) and agent.unique_id != self.unique_id:
                    collision = True
                    blocking_agent = agent
                    break

            if not collision:
                self.model.grid.move_agent(self, next_pos)
                self.pos = next_pos  # Asegurar que la posición se actualiza
                self.step_index += 1
                self.wait_steps = 0  # Reiniciar contador de espera
                if self.pos == self.goal_pos:
                    if self.state == 'to_goal':
                        print(f"Agente {self.unique_id} llegó al objetivo {self.goal_pos}")
                        self.state = 'to_start'
                        self.goal_pos = self.start_pos
                        self.waiting = True
                        self.needs_replan = True
                    elif self.state == 'to_start':
                        print(f"Agente {self.unique_id} regresó al inicio {self.start_pos}")
                        self.state = 'to_goal'
                        self.goal_pos = self.model.get_new_goal(self)
                        self.waiting = True
                        self.needs_replan = True
            else:
                self.collisions += 1
                self.model.num_collisions += 1
                if blocking_agent.priority < self.priority:
                    # El agente bloqueante tiene mayor prioridad
                    print(f"Agente {self.unique_id} bloqueado por agente de mayor prioridad en {next_pos}")
                    moved = self.try_to_move_out_of_way()
                    if not moved:
                        self.wait_steps += 1
                        if self.wait_steps >= 3:
                            self.waiting = True  # Esperar replanificación
                            self.needs_replan = True
                            self.wait_steps = 0
                else:
                    # El agente bloqueante tiene menor prioridad
                    print(f"Agente {self.unique_id} bloqueado por agente de menor prioridad en {next_pos}")
                    self.wait_steps += 1
                    if self.wait_steps >= 1:
                        self.waiting = True  # Replanificar ruta
                        self.needs_replan = True
                        self.wait_steps = 0
        else:
            self.waiting = True  # Esperar replanificación
            self.needs_replan = True
            self.wait_steps = 0  # Asegurar que se reinicia el contador


    def try_to_move_out_of_way(self):
        # Intentar moverse a una posición adyacente libre y válida
        neighbors = list(self.model.grid.get_neighborhood(self.pos, moore=False, include_center=False))
        random.shuffle(neighbors)
        for neighbor in neighbors:
            x, y = neighbor
            # Verificar que la posición está dentro del grid
            if not (0 <= x < self.model.grid.width and 0 <= y < self.model.grid.height):
                continue  # Fuera del grid
            # Verificar que no sea una pared
            cell_contents = self.model.grid.get_cell_list_contents([neighbor])
            if any(isinstance(agent, WallAgent) for agent in cell_contents):
                continue  # Es una pared
            # Verificar que la celda esté vacía
            if self.model.grid.is_cell_empty(neighbor):
                # Moverse a la posición libre
                self.model.grid.move_agent(self, neighbor)
                self.pos = neighbor  # Actualizar posición
                print(f"Agente {self.unique_id} se movió a {neighbor} para no bloquear")
                # Restablecer contadores y estados
                self.wait_steps = 0
                self.waiting = True
                self.needs_replan = True
                return True
        return False  # No pudo moverse



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
        self.goal_positions = goal_positions.copy()  # Copia de la lista original
        self.available_goals = self.goal_positions.copy()  # Lista de objetivos disponibles

        central_agent = CentralControlAgent('central_control', self)
        self.central_agent = central_agent  # Referencia al agente central
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
        random.shuffle(priorities)  # Asignar prioridades aleatorias

        for i, start_pos in enumerate(agent_starts):
            # Asignar un objetivo diferente a cada agente
            if self.available_goals:
                goal_pos = self.available_goals.pop(0)
            else:
                # Si no hay suficientes objetivos, reutilizar los existentes
                goal_pos = random.choice(self.goal_positions)
            priority = priorities.pop()
            agent = MovingAgent(i, self, start_pos, goal_pos, priority)
            self.grid.place_agent(agent, start_pos)
            self.schedule.add(agent)
            central_agent.register_robot(agent)

        self.running = True

    def get_new_goal(self, agent):
        # Obtener un nuevo objetivo para el agente
        if self.available_goals:
            new_goal = self.available_goals.pop(0)
            # Reagregar el objetivo anterior a la lista de disponibles
            self.available_goals.append(agent.goal_pos)
        else:
            # Si no hay objetivos disponibles, asignar el mismo objetivo
            new_goal = agent.goal_pos
        print(f"Agente {agent.unique_id} asignado a nuevo objetivo {new_goal}")
        return new_goal

    def step(self):
        self.datacollector.collect(self)
        # Asegurar que el agente central actúe antes que los robots
        self.central_agent.step()
        for agent in self.schedule.agents:
            if isinstance(agent, MovingAgent):
                agent.step()
        # Verificar si todos los agentes han terminado (opcional)
        # if all(agent.state == 'finished' for agent in self.schedule.agents if isinstance(agent, MovingAgent)):
        #     self.running = False

def agent_portrayal(agent):
    if isinstance(agent, MovingAgent):
        # Lista de colores para los robots
        robot_colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "yellow", "pink"]
        color = robot_colors[agent.unique_id % len(robot_colors)] if agent.state != 'finished' else "grey"

        portrayal = {
            "Shape": "circle",
            "Color": color,
            "Filled": "true",
            "Layer": 2,
            "r": 0.5,
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

grid = CanvasGrid(agent_portrayal, grid_width, grid_height, 800, 800)
collision_counter = CollisionCounter()

server = ModularServer(
    MultiAgentModel,
    [grid, collision_counter],
    "Simulación Multiagente con Retorno al Inicio y Evitación Mejorada",
    {"num_agents": 3}  # Puedes cambiar el número de agentes aquí
)

server.port = 8521
server.launch()

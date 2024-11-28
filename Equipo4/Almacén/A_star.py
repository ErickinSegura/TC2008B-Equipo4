from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.visualization.modules import CanvasGrid, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
import random
import heapq
import networkx as nx

desc = [
"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB",
"BCFFFFFFFFFSFFFFFFGFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCB",
"BFFFFFFFFFFFFFFFFFGFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFGFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBNFPBBNPBBNPBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BBNFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNNBBNFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFB",
"BCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCB",
"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"
]

for i in range(len(desc)):
    desc[i] = desc[i][::-1]

def from_desc_to_maze(desc):
    start_positions = []
    goals = []
    walls = []
    chargers = []
    collection_points = []
    pallet_positions = []
    new_pallet_positions = []
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
            elif cell == 'C':
                chargers.append((x, y))
            elif cell == 'E':
                collection_points.append((x, y))
            elif cell == 'P':
                pallet_positions.append((x, y))  # Posiciones con pallets iniciales
            elif cell == 'N':
                new_pallet_positions.append((x, y))  # Posiciones para nuevos pallets
    return walls, start_positions, goals, chargers, collection_points, pallet_positions, new_pallet_positions, width, height


class PalletAgent(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.is_transported = False
        self.transported_by = None
        self.delivered = False  # Nuevo atributo para evitar que el pallet sea recogido nuevamente


class CollectionPointAgent(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos


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
        # Actualizar las prioridades efectivas de los agentes
        for robot in self.robot_agents:
            robot.update_priority()

        # Liberar reservas antiguas de agentes que necesitan replanificar
        for robot in self.robot_agents:
            if robot.needs_replan:
                for pos, t in list(self.reservations.keys()):
                    if self.reservations[(pos, t)] == robot.unique_id:
                        del self.reservations[(pos, t)]

        routes = {}
        # Ordenar los agentes según su prioridad efectiva (mayor prioridad primero)
        for robot in sorted(self.robot_agents, key=lambda x: x.priority, reverse=True):
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
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
                if (0 <= current[0] + dx < self.model.grid.width and
                    0 <= current[1] + dy < self.model.grid.height and
                    (current[0] + dx, current[1] + dy) not in self.model.walls and
                    (current[0] + dx, current[1] + dy) not in finished_positions)
            ] + [current]  # Agregar la opción de esperar en el mismo lugar

            for neighbor in neighbors:
                next_time = t + 1
                reserved_by = self.reservations.get((neighbor, next_time))
                if reserved_by is not None and reserved_by != self.unique_id:
                    other_robot = next((r for r in self.robot_agents if r.unique_id == reserved_by), None)
                    if other_robot and other_robot.priority >= priority:
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
        self.plan_routes()  # Planificar rutas en cada paso

class BatteryCharger(Agent):
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        self.pos = pos
        self.available = True
        self.charging_agent = None


class MovingAgent(Agent):
    def __init__(self, unique_id, model, start_pos, goal_pos, priority):
        super().__init__(unique_id, model)
        self.start_pos = start_pos
        self.pos = start_pos
        self.goal_pos = goal_pos
        self.path = []
        self.step_index = 0
        self.state = 'to_collection' if desc[start_pos[1]][start_pos[0]] == 'E' else 'to_goal'
        self.waiting = True
        self.collisions = 0
        self.needs_replan = True
        self.wait_steps = 0
        self.safe_distance = 4
        self.stuck_counter = 0  # Contador para tiempo estancado
        self.max_stuck_time = 5  # Máximo tiempo permitido estancado
        self.last_position = start_pos  # Última posición conocida
        self.position_unchanged_counter = 0  # Contador para posición sin cambios
        self.initial_priority = priority  # Prioridad inicial asignada
        self.priority = self.initial_priority  # Prioridad efectiva inicial
        self.battery_soc = 100.0  # State of Charge inicial
        self.battery_discharge_rate_moving = 20.0  # % por hora
        self.battery_discharge_rate_idle = 5.0  # % por hora
        self.battery_critical_threshold = 40.0  # % de batería crítico
        self.battery_low_threshold = 70.0  # % de batería para iniciar carga
        self.battery_charge_threshold = 90.0  # % de batería para detener carga
        self.battery_charge_rate = 20.0  # % cada 5 minutos (simulados)
        self.charging_status = False
        self.current_charger = None
        self.current_pallet = None
        self.collected_pallets = 0  # Inicializamos el contador de pallets recolectados
        self.previous_state = None
        self.previous_goal_pos = None
        print(f"Agente {self.unique_id} iniciado en {self.start_pos} con objetivo {self.goal_pos}, prioridad {self.priority}")

    def find_nearest_pallet(self):
        """Encuentra el pallet disponible más cercano que no ha sido entregado."""
        available_pallets = [p for p in self.model.pallet_agents if not p.is_transported and not p.delivered]
        if not available_pallets:
            return None

        # Encuentra el pallet más cercano a la posición actual del agente
        return min(available_pallets, key=lambda p: abs(p.pos[0] - self.pos[0]) + abs(p.pos[1] - self.pos[1]))


    def needs_charging(self):
        """Determina si el agente necesita ir a cargar"""
        return self.battery_soc < self.battery_low_threshold

    def is_battery_critical(self):
        """Verifica si la batería está en estado crítico"""
        return self.battery_soc < self.battery_critical_threshold

    def discharge_battery(self, is_moving=False):
        """Descarga la batería según su estado"""
        discharge_rate = self.battery_discharge_rate_moving if is_moving else self.battery_discharge_rate_idle
        self.battery_soc = max(0.0, self.battery_soc - discharge_rate / 240)  # Por minuto simulado

    def charge_battery(self, charger):
        """Carga la batería"""
        if self.battery_soc < self.battery_charge_threshold and not self.state == 'finished':
            self.battery_soc = min(self.battery_soc + self.battery_charge_rate, 100.0)
            self.charging_status = True
            self.current_charger = charger
            print(f"Agente {self.unique_id} cargando en {charger.pos}. SoC: {self.battery_soc:.2f}%")

    def stop_charging(self):
        """Detiene la carga"""
        if self.charging_status:
            print(f"Agente {self.unique_id} finaliza carga con SoC {self.battery_soc:.2f}%")
            self.charging_status = False
            if self.current_charger:
                self.current_charger.available = True
                self.current_charger.charging_agent = None
                self.current_charger = None


    def get_distance_to(self, other_pos):
        return abs(self.pos[0] - other_pos[0]) + abs(self.pos[1] - other_pos[1])

    def get_orthogonal_neighbors(self, position, radius=1):
        """Obtener vecinos solo en las cuatro direcciones principales hasta el radio especificado"""
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # arriba, derecha, abajo, izquierda
        
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
                    # Si el agente está en estado 'finished', tratarlo como obstáculo estático
                    if agent.state == 'finished':
                        return [(agent, agent.pos)]  # Retornar inmediatamente si encuentra un agente finished
                    else:
                        nearby_agents.append((agent, check_pos))
        return nearby_agents

    def predict_collision(self, next_pos):
        # Verificar primero si hay algún agente finished en la posición objetivo



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
        """Intenta encontrar una ruta alternativa evitando la zona actual y los agentes finished"""
        current_pos = self.pos
        target_pos = self.goal_pos
        
        # Marcar temporalmente la posición actual, adyacentes y posiciones de agentes finished como "bloqueadas"
        temp_blocked = set()
        neighbors = self.get_orthogonal_neighbors(current_pos, 2)
        temp_blocked.update(neighbors)
        temp_blocked.add(current_pos)
        
        # Añadir posiciones de agentes finished
        for agent in self.model.schedule.agents:
            if isinstance(agent, MovingAgent) and agent.state == 'finished':
                temp_blocked.add(agent.pos)
        
        # Crear un grafo temporal excluyendo las posiciones bloqueadas
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
        neighbors = self.get_orthogonal_neighbors(self.pos, 1)  # Aumentado el radio a 2 para más opciones
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
        """Verifica si el agente está estancado y necesita una ruta alternativa"""
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
        """Intenta encontrar una ruta alternativa evitando la zona actual"""
        current_pos = self.pos
        target_pos = self.goal_pos
        
        # Marcar temporalmente la posición actual y adyacentes como "bloqueadas"
        temp_blocked = set()
        neighbors = self.get_orthogonal_neighbors(current_pos, 2)
        temp_blocked.update(neighbors)
        temp_blocked.add(current_pos)
        
        # Crear un grafo temporal excluyendo las posiciones bloqueadas
        G = nx.grid_2d_graph(self.model.grid.width, self.model.grid.height)
        for wall in self.model.walls:
            if wall in G:
                G.remove_node(wall)
        for blocked in temp_blocked:
            if blocked in G:
                G.remove_node(blocked)
        
        try:
            # Intentar encontrar una ruta alternativa
            path = nx.shortest_path(G, current_pos, target_pos, weight='weight')
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def update_priority(self):
        # La prioridad efectiva es la prioridad inicial más un factor basado en el nivel de batería
        battery_factor = 100 - self.battery_soc
        self.priority = self.initial_priority + battery_factor

    def step(self):
        if self.state == 'finished':
            return

        if self.waiting:
            return

        # Verificar si está estancado
        if self.check_if_stuck():
            alternative_path = self.find_alternative_path()
            if (alternative_path):
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
                self.handle_collision(blocking_agent)
                return  # Agregar return para evitar que el agente se mueva en este paso
            else:
                # Mover al agente
                self.model.grid.move_agent(self, next_pos)
                self.pos = next_pos
                self.step_index += 1
                self.wait_steps = 0

                # Verificar si ha llegado al objetivo
                if self.pos == self.goal_pos:
                    if self.state == 'to_goal':
                        print(f"Agente {self.unique_id} llegó al pallet en {self.goal_pos}")
                        # Recoger el pallet
                        pallet = next((p for p in self.model.pallet_agents if p.pos == self.pos and not p.is_transported and not p.delivered), None)
                        if pallet:
                            pallet.is_transported = True
                            pallet.transported_by = self
                            self.current_pallet = pallet
                            self.model.grid.remove_agent(pallet)  # Remover el pallet del grid
                            self.model.generate_new_pallet()  # Generar un nuevo pallet
                            # Encontrar el punto de recolección más cercano
                            collection_points = self.model.collection_points
                            self.goal_pos = min(collection_points, key=lambda p: abs(p[0] - self.pos[0]) + abs(p[1] - self.pos[1]))
                            self.state = 'to_collection'
                            self.waiting = True
                            self.needs_replan = True

                        else:
                            # No se encontró el pallet (quizás otro robot lo recogió)
                            next_pallet = self.find_nearest_pallet()
                            if next_pallet:
                                self.goal_pos = next_pallet.pos
                                self.state = 'to_goal'
                                self.waiting = True
                                self.needs_replan = True
                            else:
                                # No hay más pallets, regresar al inicio
                                self.state = 'return_to_start'
                                self.goal_pos = self.start_pos
                                self.waiting = True
                                self.needs_replan = True
                    elif self.state == 'to_collection':
                        print(f"Agente {self.unique_id} entregó el pallet en {self.goal_pos}")
                        # Dejar el pallet
                        if self.current_pallet:
                            self.current_pallet.is_transported = False
                            self.current_pallet.transported_by = None
                            self.current_pallet.delivered = True  # Marcar como entregado
                            # No es necesario colocar el pallet en el grid si ya no es relevante
                            self.current_pallet = None
                            self.collected_pallets += 1
                        # Buscar el siguiente pallet
                        next_pallet = self.find_nearest_pallet()
                        if next_pallet:
                            self.goal_pos = next_pallet.pos
                            self.state = 'to_goal'
                            self.waiting = True
                            self.needs_replan = True
                        else:
                            # No hay más pallets, regresar al inicio
                            self.state = 'return_to_start'
                            self.goal_pos = self.start_pos
                            self.waiting = True
                            self.needs_replan = True
                    elif self.state == 'return_to_start':
                        print(f"Agente {self.unique_id} regresó a su punto de inicio en {self.start_pos}")
                        self.state = 'finished'
        else:
            # Si no hay más pasos en el camino, solicitar nueva planificación
            self.waiting = True
            self.needs_replan = True
            self.wait_steps = 0

    def handle_collision(self, blocking_agent):
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


def create_battery_chart_modules(num_agents):
    """
    Crea módulos de gráfico individuales para cada agente
    """
    # Lista de colores para los agentes
    colors = ["blue", "green", "red", "purple", "orange", 
              "cyan", "magenta", "yellow", "pink", "brown"]
    
    battery_charts = []
    for agent_id in range(num_agents):
        chart = ChartModule([
            {
                "Label": f"Agent {agent_id} Battery",
                "Color": colors[agent_id % len(colors)]
            }
        ], data_collector_name=f'battery_datacollector_agent_{agent_id}')
        battery_charts.append(chart)
    
    return battery_charts



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
        self.walls, start_positions, goal_positions, charger_positions, collection_points, pallet_positions, new_pallet_positions, width, height = from_desc_to_maze(desc[:-1])
        # Configuración del grid y programación
        self.grid = MultiGrid(width, height, torus=False)
        self.schedule = RandomActivation(self)
        self.num_collisions = 0
        
        # Recolector de datos para colisiones
        self.datacollector = DataCollector(
            {"Collisions": lambda m: m.num_collisions}
        )
        
        # Configuración de posiciones
        self.start_positions = start_positions
        self.goal_positions = goal_positions.copy()
        self.available_goals = self.goal_positions.copy()
        self.charger_positions = charger_positions  # Almacenar posiciones de cargadores

        # Crear agente de control central
        central_agent = CentralControlAgent('central_control', self)
        self.central_agent = central_agent
        self.schedule.add(central_agent)

        # Crear cargadores de batería
        self.battery_chargers = []
        self.create_battery_chargers()
        self.collection_points = collection_points

        # Preparar recolectores de datos de batería
        self.battery_datacollectors = {}

        def get_battery_soc_factory(agent_id):
            def get_battery_soc(m):
                for a in m.schedule.agents:
                    if isinstance(a, MovingAgent) and a.unique_id == agent_id:
                        return a.battery_soc
                return 0
            return get_battery_soc

        # Crear agentes de pared
        for idx, wall in enumerate(self.walls):
            wall_agent = WallAgent(f'wall_{idx}', self)
            self.grid.place_agent(wall_agent, wall)

        # Crear agentes de objetivo
        for idx, goal_pos in enumerate(self.goal_positions):
            goal_agent = GoalAgent(f'goal_{idx}', self, goal_pos)
            self.grid.place_agent(goal_agent, goal_pos)

        # Crear agentes de puntos de entrega
        for idx, pos in enumerate(self.collection_points):
            collection_agent = CollectionPointAgent(f'collection_{idx}', self, pos)
            self.grid.place_agent(collection_agent, pos)
            self.schedule.add(collection_agent)


        self.pallet_agents = []
        self.collection_points = collection_points

        # Create pallet agents (you can modify this to place pallets dynamically)
        for idx, goal_pos in enumerate(goal_positions):
            if desc[goal_pos[1]][goal_pos[0]] == 'P':  # Only create pallets on 'P' cells
                pallet = PalletAgent(f'pallet_{idx}', self, goal_pos)
                self.grid.place_agent(pallet, goal_pos)
                self.pallet_agents.append(pallet)

        # Seleccionar posiciones de inicio disponibles
        available_starts = [pos for pos in self.start_positions if pos not in self.walls]
        agent_starts = random.sample(available_starts, min(self.num_agents, len(available_starts)))


        self.pallet_positions = pallet_positions  # Posiciones con pallets iniciales
        self.available_pallet_positions = new_pallet_positions.copy()  # Posiciones donde se pueden generar nuevos pallets

        # Crear agentes de pallets iniciales
        self.pallet_agents = []
        for idx, pos in enumerate(self.pallet_positions):
            pallet = PalletAgent(f'pallet_{idx}', self, pos)
            self.grid.place_agent(pallet, pos)
            self.pallet_agents.append(pallet)
    

        # Crear recolectores de datos dinámicos para la batería de cada agente
        for i in range(num_agents):
            setattr(self, f'battery_datacollector_agent_{i}', DataCollector(
                {f"Agent {i} Battery": lambda m, agent_id=i: 
                 next((a.battery_soc for a in m.schedule.agents 
                       if isinstance(a, MovingAgent) and a.unique_id == agent_id), 0)}
            ))

        # Configurar recolectores de datos de batería
        for i in range(num_agents):
            get_battery_soc = get_battery_soc_factory(i)
            datacollector = DataCollector(
                {f"Agent {i} Battery": get_battery_soc}
            )
            setattr(self, f'battery_datacollector_agent_{i}', datacollector)
            self.battery_datacollectors[i] = datacollector  

        # Generar prioridades iniciales aleatorias
        initial_priorities = list(range(1, self.num_agents + 1))
        random.shuffle(initial_priorities)

        # Crear agentes móviles
        for i, start_pos in enumerate(agent_starts):
            # Asignar objetivo
            if self.available_goals:
                goal_pos = self.available_goals.pop(0)
            else:
                goal_pos = random.choice(self.goal_positions)
            
            initial_priority = initial_priorities.pop()
            agent = MovingAgent(i, self, start_pos, goal_pos, initial_priority)
            self.grid.place_agent(agent, start_pos)
            self.schedule.add(agent)
            central_agent.register_robot(agent)

        self.running = True

    def generate_new_pallet(self):
        if self.available_pallet_positions:
            new_pallet_pos = random.choice(self.available_pallet_positions)
            self.available_pallet_positions.remove(new_pallet_pos)
            pallet = PalletAgent(f'pallet_{len(self.pallet_agents)}', self, new_pallet_pos)
            self.grid.place_agent(pallet, new_pallet_pos)
            self.pallet_agents.append(pallet)
        else:
            print("No quedan más posiciones disponibles para pallets.")


    def create_battery_chargers(self):
        """Crear puntos de carga en las posiciones de los cargadores del mapa"""
        for idx, pos in enumerate(self.charger_positions):
            charger = BatteryCharger(f'charger_{idx}', self, pos)
            self.grid.place_agent(charger, pos)
            self.battery_chargers.append(charger)
            self.schedule.add(charger)

    def find_available_charger(self):
        """Encuentra un cargador disponible"""
        available_chargers = [c for c in self.battery_chargers if c.available]
        return random.choice(available_chargers) if available_chargers else None

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
        self.central_agent.step()
        
        for datacollector in self.battery_datacollectors.values():
            datacollector.collect(self)
        
        for agent in self.schedule.agents:
            if isinstance(agent, MovingAgent):
                # Si está en un cargador
                if agent.pos == agent.goal_pos and any(isinstance(charger, BatteryCharger) and charger.pos == agent.pos for charger in self.battery_chargers):
                    charger = next((c for c in self.battery_chargers if c.pos == agent.pos), None)
                    if charger:
                        # Solo cargar si el agente está en estado de carga
                        if agent.state == 'charging':
                            agent.charge_battery(charger)
                            
                            if agent.battery_soc >= agent.battery_charge_threshold:
                                agent.stop_charging()
                                # Verificar si hay pallets disponibles antes de restaurar la misión anterior
                                next_pallet = agent.find_nearest_pallet()
                                if next_pallet:
                                    # Restaurar la misión anterior si hay pallets disponibles
                                    if agent.previous_goal_pos and agent.previous_state:
                                        agent.goal_pos = agent.previous_goal_pos
                                        agent.state = agent.previous_state
                                        agent.waiting = True
                                        agent.needs_replan = True
                                        print(f"Agente {agent.unique_id} terminó de cargar. Reanudando misión.")
                                else:
                                    # Si no hay pallets disponibles, regresar al inicio
                                    agent.state = 'return_to_start'
                                    agent.goal_pos = agent.start_pos
                                    agent.charging_status = False
                                    agent.waiting = True
                                    agent.needs_replan = True
                                    print(f"Agente {agent.unique_id} terminó de cargar. No hay más pallets. Regresando al inicio.")
                else:
                    # Comportamiento normal de descarga
                    if not agent.charging_status:
                        is_moving = agent.step_index < len(agent.path) if hasattr(agent, 'path') else False
                        agent.discharge_battery(is_moving)
        
                        if agent.is_battery_critical():
                            print(f"Agente {agent.unique_id} - Batería en nivel crítico")
                            agent.waiting = True
                            agent.needs_replan = True
        
                        if agent.needs_charging() and not agent.charging_status and not agent.state == 'finished':
                            available_charger = self.find_available_charger()
                            if available_charger:
                                print(f"Agente {agent.unique_id} dirigiéndose a cargar con SoC {agent.battery_soc:.2f}%")
                                # Almacenar estado y objetivo previos
                                agent.previous_state = agent.state
                                agent.previous_goal_pos = agent.goal_pos
        
                                agent.goal_pos = available_charger.pos
                                agent.state = 'charging'  # Nuevo estado para indicar que va a cargar
                                agent.waiting = True
                                agent.needs_replan = True
                                available_charger.available = False
                                available_charger.charging_agent = agent
        
                # Realizar el step del agente normalmente
                agent.step()

        # Verificar si todos los agentes han terminado
        if all(agent.state == 'finished' for agent in self.schedule.agents if isinstance(agent, MovingAgent)):
            self.running = False

class RobotState:
    def __init__(self, pos, has_pallet=False):
        self.pos = pos
        self.has_pallet = has_pallet

class PalletMaze:
    def __init__(self, desc):
        self.start_positions, self.goals, self.walls, self.chargers, self.collection_points = from_desc_to_maze(desc.reverse())
        self.width = len(desc[0])
        self.height = len(desc)
        self.pallets = self.goals.copy()  # Track

def agent_portrayal(agent):
    if isinstance(agent, MovingAgent):
        # Lista de colores para los robots
        robot_colors = ["blue", "green", "purple", "orange", "cyan", "magenta", "yellow", "pink"]
        color = robot_colors[agent.unique_id % len(robot_colors)] if agent.state != 'finished' else "grey"


        if agent.current_pallet:
            text = f"{agent.unique_id}📦"
        else:
            text = f"{agent.unique_id}"
        portrayal = {
            "Shape": "circle",
            "Color": color,
            "Filled": "true",
            "Layer": 2,
            "r": 1,
            "text": text,
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
    elif isinstance(agent, BatteryCharger):
        portrayal = {
            "Shape": "rect",
            "Color": "green",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }
    elif isinstance(agent, PalletAgent):
        portrayal = {
            "Shape": "rect",
            "Color": "brown" if not agent.is_transported else "red",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1
        }
    elif isinstance(agent, CollectionPointAgent):
        portrayal = {
            "Shape": "rect",
            "Color": "yellow",
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

cell_size = 20  # Tamaño deseado para cada celda en píxeles
canvas_width = grid_width * cell_size
canvas_height = grid_height * cell_size

grid = CanvasGrid(agent_portrayal, grid_width, grid_height, canvas_width, canvas_height)
collision_counter = CollisionCounter()
visualization_modules = [grid, collision_counter]
battery_charts = create_battery_chart_modules(3)
visualization_modules.extend(battery_charts)


server = ModularServer(
        MultiAgentModel,
        visualization_modules,
        "Simulación Multiagente",
        {"num_agents": 3}
    )

server.port = 8521
server.launch()



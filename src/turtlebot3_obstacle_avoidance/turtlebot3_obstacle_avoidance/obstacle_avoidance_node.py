import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan


class ObstacleAvoidanceNode(Node):
    """
    Nodo de ROS 2 para evasión de obstáculos simple.

    Implementa una máquina de estados (avanzar, parar, rotar) basada en
    las lecturas del LIDAR (LaserScan) y la odometría (Odometry) para
    controlar un robot.

    El robot avanza hasta detectar un obstáculo, se detiene brevemente,
    rota un ángulo fijo y reanuda el avance.
    """

    _FORWARD = "forward"
    _STOPPING = "stopping"
    _ROTATING = "rotating"

    def __init__(self) -> None:
        """Inicializa el nodo, declara parámetros y configura publicadores/suscriptores."""
        super().__init__("obstacle_avoidance")

        # --- Declaración de Parámetros ---
        self.declare_parameter("linear_speed", 0.5)
        self.declare_parameter("angular_speed", 1.0)
        self.declare_parameter("rotation_angle_deg", 89.5)
        self.declare_parameter("obstacle_distance", 0.5)
        self.declare_parameter("front_sector_width_deg", 80.0)
        self.declare_parameter("front_sector_center_deg", -90.0)
        self.declare_parameter("pre_rotation_stop_time", 0.1)
        self.declare_parameter("post_rotation_ignore_time", 0.5)
        self.declare_parameter("filter_zero_intensity", True)
        self.declare_parameter("lidar_forward_offset", 0.04)
        self.declare_parameter("rotation_tolerance_deg", 5.0)

        # --- Obtención y Almacenamiento de Parámetros ---
        self.linear_speed: float = self.get_parameter("linear_speed").get_parameter_value().double_value
        self.angular_speed: float = self.get_parameter("angular_speed").get_parameter_value().double_value
        rotation_angle_deg: float = (
            self.get_parameter("rotation_angle_deg").get_parameter_value().double_value
        )
        self.rotation_angle: float = math.radians(rotation_angle_deg)
        self.obstacle_distance: float = self.get_parameter("obstacle_distance").get_parameter_value().double_value
        front_sector_deg: float = (
            self.get_parameter("front_sector_width_deg").get_parameter_value().double_value
        )
        self.front_sector_width: float = math.radians(front_sector_deg)
        front_sector_center_deg: float = (
            self.get_parameter("front_sector_center_deg").get_parameter_value().double_value
        )
        self.front_sector_center: float = math.radians(front_sector_center_deg)
        self.pre_rotation_stop_time: float = (
            self.get_parameter("pre_rotation_stop_time").get_parameter_value().double_value
        )
        self.post_rotation_ignore_time: float = (
            self.get_parameter("post_rotation_ignore_time").get_parameter_value().double_value
        )
        self.filter_zero_intensity: bool = (
            self.get_parameter("filter_zero_intensity").get_parameter_value().bool_value
        )
        self.lidar_forward_offset: float = (
            self.get_parameter("lidar_forward_offset").get_parameter_value().double_value
        )
        rotation_tolerance_deg: float = (
            self.get_parameter("rotation_tolerance_deg").get_parameter_value().double_value
        )
        self.rotation_tolerance: float = math.radians(rotation_tolerance_deg)

        # Cálculo de la duración teórica de la rotación
        self.rotation_duration: float = self.rotation_angle / max(self.angular_speed, 1e-6)

        # --- Configuración de ROS (Pub/Sub/Timers) ---
        self.cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self.scan_sub = self.create_subscription(LaserScan, "scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "odom", self.odom_callback, 10)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        # --- Variables de Estado ---
        self.front_distance: float = float("inf")
        self.last_scan: Optional[LaserScan] = None
        self.state: str = self._FORWARD
        self.stop_until: Optional[Time] = None
        self.rotation_start: Optional[Time] = None
        self.rotation_target: Optional[float] = None
        self.cooldown_until = self.get_clock().now()
        self.current_yaw: Optional[float] = None
        self.rotation_timeout_warned = False

        self.get_logger().info(
            f"Nodo de evasión de obstáculos inicializado "
            f"(avance {self.linear_speed:.2f} m/s, giro {self.angular_speed:.2f} rad/s, "
            f"ángulo {math.degrees(self.rotation_angle):.1f}°)."
        )

    def scan_callback(self, msg: LaserScan) -> None:
        """
        Callback para el suscriptor de LaserScan.

        Se ejecuta cada vez que se recibe un mensaje del LIDAR.
        Almacena el mensaje y calcula la distancia mínima al frente
        llamando a _compute_front_distance.

        Args:
            msg: El mensaje de LaserScan recibido.
        """
        self.last_scan = msg
        self.front_distance = self._compute_front_distance(msg)

    def odom_callback(self, msg: Odometry) -> None:
        """
        Callback para el suscriptor de Odometry.

        Se ejecuta cada vez que se recibe un mensaje de odometría.
        Extrae la orientación (cuaternión) y la convierte a un ángulo
        de yaw (en radianes), que almacena en self.current_yaw.

        Args:
            msg: El mensaje de Odometry recibido.
        """
        self.current_yaw = self._quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )

    def _compute_front_distance(self, msg: LaserScan) -> float:
        """
        Calcula la distancia mínima en el sector frontal del robot.

        Analiza las lecturas del LaserScan 'msg' dentro de un sector angular
        definido por 'front_sector_center' y 'front_sector_width'.
        Filtra lecturas inválidas (inf, nan) y, opcionalmente, lecturas
        con intensidad cero. Aplica un 'lidar_forward_offset'.

        Args:
            msg: El mensaje de LaserScan a procesar.

        Returns:
            La distancia mínima válida (en metros) en el sector frontal,
            o float('inf') si no hay lecturas válidas.
        """
        half_sector = self.front_sector_width / 2.0
        center_angle = self.front_sector_center
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        total_samples = len(msg.ranges)

        if total_samples == 0 or angle_increment == 0.0:
            return float("inf")

        # Calcular índices del sector frontal
        start_angle = center_angle - half_sector
        end_angle = center_angle + half_sector

        index_min = max(0, int(math.ceil((start_angle - angle_min) / angle_increment)))
        index_max = min(total_samples - 1, int(math.floor((end_angle - angle_min) / angle_increment)))

        if index_min > index_max:
            # Esto puede pasar si el sector está mal configurado o cruza el punto 0
            index_min = 0
            index_max = total_samples - 1

        # Filtrar rangos dentro del sector
        valid_ranges = []
        has_intensity = len(msg.intensities) == total_samples

        for i in range(index_min, index_max + 1):
            distance = msg.ranges[i]
            if not math.isfinite(distance):
                continue
            if has_intensity and self.filter_zero_intensity and msg.intensities[i] <= 0.0:
                continue
            valid_ranges.append(distance)

        if not valid_ranges:
            return float("inf")

        # Devolver la distancia mínima, ajustada por el offset del lidar
        distance = min(valid_ranges)
        adjusted_distance = max(distance - self.lidar_forward_offset, 0.0)
        return adjusted_distance

    def control_loop(self) -> None:
        """
        Bucle de control principal, ejecutado por un temporizador.

        Implementa la máquina de estados del robot:
        - _FORWARD: Avanza si no hay obstáculos. Si detecta uno,
          transiciona a _STOPPING.
        - _STOPPING: Permanece detenido por 'pre_rotation_stop_time'
          segundos y luego transiciona a _ROTATING.
        - _ROTATING: Gira 'rotation_angle' rad/s usando la odometría como
          referencia (con un fallback de tiempo). Al terminar,
          transiciona a _FORWARD y activa un cooldown.

        Publica el mensaje Twist apropiado en 'cmd_vel' según el estado.
        """
        now = self.get_clock().now()
        cmd = Twist()

        # --- Lógica de la Máquina de Estados ---

        if self.state == self._FORWARD:
            # Estado de avance
            if self.front_distance <= self.obstacle_distance and now >= self.cooldown_until:
                # Obstáculo detectado y fuera del cooldown: transicionar a PARAR
                self.state = self._STOPPING
                self.stop_until = now + Duration(seconds=self.pre_rotation_stop_time)
                self.get_logger().info(
                    f"Obstáculo detectado a {self.front_distance:.2f} m: iniciando frenado."
                )
            else:
                # Sin obstáculo o en cooldown: seguir avanzando
                cmd.linear.x = self.linear_speed

        elif self.state == self._STOPPING:
            # Estado de frenado
            if self.stop_until is not None and now >= self.stop_until:
                # Tiempo de frenado completado: transicionar a ROTAR
                self.state = self._ROTATING
                self.rotation_start = now
                self.rotation_target = None  # Se calculará cuando llegue la odometría
                self.rotation_timeout_warned = False
                self.get_logger().info(
                    f"Comenzando rotación de {math.degrees(self.rotation_angle):.1f}° "
                    f"a {self.angular_speed:.2f} rad/s."
                )
            # Si no, cmd se queda en (0, 0), manteniendo el robot frenado

        elif self.state == self._ROTATING:
            # Estado de rotación
            if self.rotation_start is None:
                self.rotation_start = now  # Salvaguarda por si se pierde el primer frame

            # Calcular el yaw objetivo si aún no se ha hecho
            if self.rotation_target is None and self.current_yaw is not None:
                direction = 1.0 if self.angular_speed >= 0 else -1.0
                self.rotation_target = self._normalize_angle(
                    self.current_yaw + direction * self.rotation_angle
                )

            # Comprobar si se ha alcanzado el objetivo de yaw
            yaw_reached = False
            if self.rotation_target is not None and self.current_yaw is not None:
                yaw_error = self._angle_difference(self.rotation_target, self.current_yaw)
                yaw_reached = abs(yaw_error) <= self.rotation_tolerance

            # Comprobar si se ha alcanzado el tiempo de rotación (fallback)
            elapsed = now - self.rotation_start
            rotation_time_ns = self.rotation_duration * 1e9
            time_reached = elapsed.nanoseconds >= rotation_time_ns

            # Condición de fin de rotación
            if yaw_reached or (self.rotation_target is None and time_reached):
                # Objetivo alcanzado (por odometría o por tiempo si la odometría falla)
                self.state = self._FORWARD
                self.cooldown_until = now + Duration(seconds=self.post_rotation_ignore_time)
                self.front_distance = float("inf")  # Limpiar la distancia antigua
                self.rotation_start = None
                self.stop_until = None
                self.rotation_target = None
                self.get_logger().info("Rotación completa; retomando avance.")
            else:
                # Seguir rotando
                cmd.angular.z = self.angular_speed
                if (
                    self.rotation_target is not None
                    and time_reached
                    and not self.rotation_timeout_warned
                ):
                    # Advertir si el tiempo se cumplió pero el yaw no (odom lenta o atascado)
                    self.get_logger().warn(
                        "Tiempo estimado de rotación alcanzado pero el yaw objetivo aún "
                        "no se logra; continuando giro bajo control de odometría."
                    )
                    self.rotation_timeout_warned = True

        # Publicar el comando de velocidad (avanzar, girar o parar)
        self.cmd_pub.publish(cmd)

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        """
        Convierte un cuaternión (x, y, z, w) en un ángulo de yaw.

        El ángulo de yaw representa la rotación alrededor del eje Z.

        Args:
            x: Componente x del cuaternión.
            y: Componente y del cuaternión.
            z: Componente z del cuaternión.
            w: Componente w del cuaternión.

        Returns:
            El ángulo de yaw (en radianes) normalizado en el rango [-pi, pi].
        """
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """
        Normaliza un ángulo al rango [-pi, pi].

        Args:
            angle: El ángulo (en radianes) a normalizar.

        Returns:
            El ángulo normalizado (en radianes).
        """
        return math.atan2(math.sin(angle), math.cos(angle))

    def _angle_difference(self, target: float, current: float) -> float:
        """
        Calcula la diferencia más corta entre dos ángulos normalizados.

        Args:
            target: El ángulo objetivo (en radianes).
            current: El ángulo actual (en radianes).

        Returns:
            La diferencia angular (en radianes) en el rango [-pi, pi].
        """
        return self._normalize_angle(target - current)


def main(args=None) -> None:
    """Función principal para inicializar y ejecutar el nodo de ROS 2."""
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Limpieza
        node.get_logger().info("Cerrando el nodo de evasión de obstáculos.")
        # Publicar un comando de parada antes de cerrar
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
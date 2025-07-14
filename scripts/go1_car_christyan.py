from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
import omni.appwindow
import omni.graph.core as og
from omni.isaac.core import World
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.quadruped.robots import Unitree
from omni.isaac.core.utils.stage import open_stage
from std_msgs.msg import Float32MultiArray
import omni.replicator.core as rep
from omni.kit.commands import execute
import math



# Enable ROS bridge extension
enable_extension("omni.isaac.ros_bridge")
simulation_app.update()

# Check if rosmaster node is running
import rosgraph
if not rosgraph.is_master_online():
    carb.log_error("Please run roscore before executing this script")
    simulation_app.close()
    exit()

import rospy
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from pxr import UsdGeom, Gf, Usd
import usdrt.Sdf

import omni.usd
import omni.syntheticdata as sd
#from omni.syntheticdata import SyntheticData
#from pxr import UsdGeom
#import Sdf
from pxr import Usd, UsdGeom, Sdf


class Go1_runner(object):
    def __init__(self, physics_dt, render_dt) -> None:
        """Initialize simulation world and add Unitree robot"""
        open_stage("/home/robcib/Desktop/Alicia/car_arena_christyan.usd")
        simulation_app.update()

        self.robotpose=Pose()

        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)
        # yaw_rotation = R.from_euler('z', 90, degrees=True).as_quat()  # [x, y, z, w]
        self._go1 = self._world.scene.add(
            Unitree(
                prim_path="/World/Go1",
                name="Go1",
                position=np.array([9, -11, 0.5]),  # Para dentro de la nave: [0, 0, 0]
                orientation=np.array([0, 0, 0, 1]), # yaw_rotation
                physics_dt=physics_dt,
                model="Go1"
            )
        )

        self._world.reset()
        self._enter_toggled = 0
        self._base_command = [0.0, 0.0, 0.0, 0]
        self._event_flag = False
        self._input_keyboard_mapping = {
            "NUMPAD_8": [1.8, 0.0, 0.0],
            "UP": [1.8, 0.0, 0.0],
            "NUMPAD_2": [-1.8, 0.0, 0.0],
            "DOWN": [-1.8, 0.0, 0.0],
            "NUMPAD_6": [0.0, -1.8, 0.0],
            "RIGHT": [0.0, -1.8, 0.0],
            "NUMPAD_4": [0.0, 1.8, 0.0],
            "LEFT": [0.0, 1.8, 0.0],
            "NUMPAD_7": [0.0, 0.0, 1.0],
            "N": [0.0, 0.0, 1.0],
            "NUMPAD_9": [0.0, 0.0, -1.0],
            "M": [0.0, 0.0, -1.0],
        }


        # Creating an ondemand push graph with ROS Clock, everything in the ROS environment must synchronize with this clock
        try:
            keys = og.Controller.Keys
            (self._clock_graph, _, _, _) = og.Controller.edit(
                {
                    "graph_path": "/ROS_Clock",
                    "evaluator_name": "push",
                    "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
                },
                {
                    keys.CREATE_NODES: [
                        ("OnTick", "omni.graph.action.OnTick"),
                        ("readSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                        ("publishClock", "omni.isaac.ros_bridge.ROS1PublishClock"),
                    ],
                    keys.CONNECT: [
                        ("OnTick.outputs:tick", "publishClock.inputs:execIn"),
                        ("readSimTime.outputs:simulationTime", "publishClock.inputs:timeStamp"),
                    ],
                },
            )
        except Exception as e:
            print(e)
            simulation_app.close()
            exit()

        self._pub = rospy.Publisher("/isaac_a1/output", Float32MultiArray, queue_size=10)
        self._joint_pos_pub = rospy.Publisher("/isaac_go1/joint_position", Float32MultiArray, queue_size=10)
        self._joint_vel_pub = rospy.Publisher("/isaac_go1/joint_velocity", Float32MultiArray, queue_size=10)
        self.robotpose_pub = rospy.Publisher("/isaac_go1/pose", Pose, queue_size=10)

        self._setup_camera()
        self._setup_lidar()

        self._waypoints = [
            np.array([9, -11]),
            np.array([9.5, -11]),
            np.array([10, -11.0]),
            np.array([10.5, -11.0]),
            np.array([11, -10.9]),
            np.array([11.5, -10.9]),
            np.array([12, -10.9]),
            np.array([12.5, -11.0]),
            np.array([13, -11.0]),
            np.array([13.5, -11.0]),
            np.array([14, -11.0]),
            np.array([14.5, -11.0]),
            np.array([15, -10.75]),
            np.array([15.25, -10.5]),
            np.array([15.5, -10.0]),
            np.array([15.5, -9.5]),
            np.array([15.5, -9.0]),
            np.array([15.5, -8.5]),
            np.array([15.3, -8.0]),
            np.array([14.8, -7.7]),
            np.array([14.3, -7.5]),
            np.array([14.3, -7.0]),
            np.array([14.0, -6.5]),
            np.array([13.5, -6.1]),
            np.array([13.0, -6.1]),
            np.array([12.5, -6.1]),
            np.array([12.0, -6.2]),
            np.array([11.5, -6.3]),
            np.array([11.0, -6.4]),
            np.array([10.5, -6.5]),
            np.array([10.0, -6.5]),
            np.array([9.5, -6.5]),
            np.array([9.0, -6.5])
        ]

        self._waypoint_index = 0
        self._tolerance = 0.3  # distancia para considerar el waypoint alcanzado

    
    def _get_yaw_from_quaternion(self, quat):
        # IsaacSim da [w, x, y, z] pero scipy espera [x, y, z, w]
        if len(quat) == 4:
            # Detectamos si el formato parece ser [w, x, y, z]
            w, x, y, z = quat
            quat_scipy = [x, y, z, w]
        else:
            raise ValueError("Cuaternión inválido")

        r = R.from_quat(quat_scipy)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)
        yaw_deg = math.degrees(yaw)
        # print(f"yaw_deg: {yaw_deg:.2f}, yaw_rad: {yaw:.4f}")
        return yaw
    
    def _normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
    
    # Waypoints
    def _update_base_command_from_waypoints(self):
        if self._waypoint_index >= len(self._waypoints):
            self._base_command[0:3] = [0.0, 0.0, 0.0]  # Detenerse
            return
        
        robot_pose = self._go1.get_world_pose()
        robot_pos = robot_pose[0][:2]
        robot_yaw = self._get_yaw_from_quaternion(robot_pose[1])  # extraer yaw

        self.robotpose.position.x=float(robot_pose[0][0])
        self.robotpose.position.y=float(robot_pose[0][1])
        self.robotpose.position.z=float(robot_pose[0][2])
        self.robotpose.orientation.x=float(robot_pose[1][1])
        self.robotpose.orientation.y=float(robot_pose[1][2])
        self.robotpose.orientation.z=float(robot_pose[1][3])
        self.robotpose.orientation.w=float(robot_pose[1][0])

        target = self._waypoints[self._waypoint_index]
        direction = target - robot_pos
        distance = np.linalg.norm(direction)

        if distance < self._tolerance:
            print(f"[INFO] Waypoint alcanzado: {self._waypoints[self._waypoint_index]}")
            self._waypoint_index += 1
            return

        target_yaw = math.atan2(direction[1], direction[0])
        yaw_error = self._normalize_angle(target_yaw - robot_yaw)

        angle_tolerance = 0.1   # tolerancia angular (radianes), ajusta a tu gusto
        max_angular_speed = 1.5
        kp_angular = 2      # ganancia proporcional para giro (ajustable)

        # print(f"robot_yaw: {robot_yaw:.4f} rad, yaw_error: {yaw_error:.4f} rad")
        
        if abs(yaw_error) > angle_tolerance:
            angular_speed = kp_angular * yaw_error
            # Limitar la velocidad angular para que no gire demasiado rápido
            angular_speed = max(min(angular_speed, max_angular_speed), -max_angular_speed)
            
            # Mientras gira, no avanza hacia adelante
            self._base_command[0] = 0.0
            self._base_command[1] = 0.0
            self._base_command[2] = angular_speed
        else:
            # Cuando está orientado, avanza recto hacia el waypoint
            direction /= distance  # normalizar
            angular_speed = kp_angular * yaw_error
            speed = 1.5
            self._base_command[0] = speed # * direction[0]
            self._base_command[1] = 0.0
            self._base_command[2] = angular_speed



    def _setup_camera(self):
        """Create and attach RGB-D camera to the robot"""
        stage = omni.usd.get_context().get_stage()
        camera_prim_path = "/World/Go1/base/camara_robot"
        if not stage.GetPrimAtPath(camera_prim_path):
            camera_prim = UsdGeom.Camera(stage.DefinePrim(camera_prim_path, "Camera")) 
            xform_api = UsdGeom.XformCommonAPI(camera_prim)
            xform_api.SetTranslate(Gf.Vec3d(0.4, 0.0, 0.5))  # Relative to Go1 base
            xform_api.SetRotate((70, 0, -90))  # No initial rotation con ((90, 0, -90))
            camera_prim.GetHorizontalApertureAttr().Set(21)
            camera_prim.GetVerticalApertureAttr().Set(16)
            camera_prim.GetFocalLengthAttr().Set(12) # Con 24 tiene demasiado zoom

        # Create ROS graph for camera
        self._setup_ros_camera_graph(camera_prim_path)

    def _setup_ros_camera_graph(self, camera_prim_path):
        """Set up ROS graph to publish camera RGB and depth topics"""
        keys = og.Controller.Keys
        (ros_camera_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": "/ROS_Camera",
                "evaluator_name": "push",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
            },
            {
                keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("createViewport", "omni.isaac.core_nodes.IsaacCreateViewport"),
                    ("getRenderProduct", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                    ("setCamera", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                    ("cameraHelperRgb", "omni.isaac.ros_bridge.ROS1CameraHelper"),
                    ("cameraHelperDepth", "omni.isaac.ros_bridge.ROS1CameraHelper"),
                ],
                keys.CONNECT: [
                    ("OnTick.outputs:tick", "createViewport.inputs:execIn"),
                    ("createViewport.outputs:execOut", "getRenderProduct.inputs:execIn"),
                    ("createViewport.outputs:viewport", "getRenderProduct.inputs:viewport"),
                    ("getRenderProduct.outputs:execOut", "setCamera.inputs:execIn"),
                    ("getRenderProduct.outputs:renderProductPath", "setCamera.inputs:renderProductPath"),
                    ("setCamera.outputs:execOut", "cameraHelperRgb.inputs:execIn"),
                    ("setCamera.outputs:execOut", "cameraHelperDepth.inputs:execIn"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperRgb.inputs:renderProductPath"),
                    ("getRenderProduct.outputs:renderProductPath", "cameraHelperDepth.inputs:renderProductPath"),
                ],
                keys.SET_VALUES: [
                    ("createViewport.inputs:viewportId", 0),
                    ("cameraHelperRgb.inputs:frameId", "camara_robot"),
                    ("cameraHelperRgb.inputs:topicName", "rgb"),
                    ("cameraHelperRgb.inputs:type", "rgb"),
                    ("cameraHelperDepth.inputs:frameId", "camara_robot"),
                    ("cameraHelperDepth.inputs:topicName", "depth"),
                    ("cameraHelperDepth.inputs:type", "depth"),
                    ("setCamera.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
                ],
            },
        )
        og.Controller.evaluate_sync(ros_camera_graph)



    def _setup_lidar(self):
        """Create and attach a LiDAR sensor to the robot's base and set up ROS publishers for it"""
        # Se define la ruta para el LiDAR como hijo del prim "base" del robot
        lidar_sensor_path = "/World/Go1/base/lidar_sensor"
        parent_path = "/World/Go1/base"
        # Crear el sensor LiDAR usando el comando de Isaac
        _, sensor = execute(
            "IsaacSensorCreateRtxLidar",
            path=lidar_sensor_path,
            parent=parent_path,
            config="Example_Rotary",
            translation=(0, 0, 1.0),  # Ajusta la traslación relativa al 'base'
            orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Ajusta la orientación según necesites
        )
        simulation_app.update()
        # Crear el hydra texture a partir del sensor (necesario para los pipelines)
        hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
        
        # Configurar los pipelines ROS para publicar la nube de puntos, el LaserScan y la visualización de debug
        writer = rep.writers.get("RtxLidarROS1PublishPointCloud")
        writer.initialize(topicName="point_cloud", frameId="sim_lidar")
        writer.attach([hydra_texture])
        
        # writer = rep.writers.get("RtxLidarDebugDrawPointCloud")
        # writer.attach([hydra_texture])
        
        writer = rep.writers.get("RtxLidarROS1PublishLaserScan")
        writer.initialize(topicName="laser_scan", frameId="sim_lidar")
        writer.attach([hydra_texture])

    def setup(self) -> None:
        """Set robot's default state and physics callback"""
        self._go1.set_state(self._go1._default_a1_state)
        self._world.add_physics_callback("a1_advance", callback_fn=self.on_physics_step)

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._sub_keyboard_event)

        self._base_command = [0.0, 0.0, 0.0, 1]
        self._setup_replicator()


    def on_physics_step(self, step_size) -> None:
        """Physics callback to control robot and publish sensor data"""

        # Waypoints
        if self._base_command[3] == 1:  # Solo si está activado el movimiento
            self._update_base_command_from_waypoints()  # AÑADIDO

        if self._event_flag:
            self._go1._qp_controller.switch_mode()
            self._event_flag = False

        self._go1.advance(step_size, self._base_command)

        # Tick the ROS Clock
        og.Controller.evaluate_sync(self._clock_graph)

        self._pub.publish(Float32MultiArray(data=self.get_footforce_data()))
        self._joint_pos_pub.publish(self.get_joint_position_msg())
        self._joint_vel_pub.publish(self.get_joint_velocity_msg())

        self.robotpose_pub.publish(self.robotpose)


    def get_footforce_data(self) -> np.array:
        """
        [Summary]

        get foot force and position data
        """
        data = np.concatenate((self._go1.foot_force, self._go1._qp_controller._ctrl_states._foot_pos_abs[:, 2]))
        return data

        """AÑADIDO POR CHRISTYAN PARA EXTRAER DATOS DEL ROBOT"""
    def get_joint_position_msg(self) -> Float32MultiArray:
        msg = Float32MultiArray()
        msg.data = self._go1._state.joint_pos.tolist()
        return msg

    def get_joint_velocity_msg(self) -> Float32MultiArray:
        msg = Float32MultiArray()
        msg.data = self._go1._state.joint_vel.tolist()
        return msg


    def run(self) -> None:
        """Main simulation loop"""
        while simulation_app.is_running():
            self._world.step(render=True)
        return

    def _sub_keyboard_event(self, event, *args, **kwargs) -> None:
        """
        [Summary]

        Subscriber callback to when kit is updated.

        """
        # reset event
        self._event_flag = False
        # when a key is pressedor released  the command is adjusted w.r.t the key-mapping
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # print(f"Tecla presionada: {event.input.name}") # Comprobar que tecla pulso
            # on pressing, the command is incremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] += np.array(self._input_keyboard_mapping[event.input.name])
                self._event_flag = True

            # enter, toggle the last command
            if event.input.name == "ENTER" and self._enter_toggled is False:
                self._enter_toggled = True
                if self._base_command[3] == 0:
                    self._base_command[3] = 1
                else:
                    self._base_command[3] = 0
                self._event_flag = True
            # print(f"Base Command (after key press): {self._base_command}")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # on release, the command is decremented
            if event.input.name in self._input_keyboard_mapping:
                self._base_command[0:3] -= np.array(self._input_keyboard_mapping[event.input.name])
                self._event_flag = True
            # enter, toggle the last command
            if event.input.name == "ENTER":
                self._enter_toggled = False
        # since no error, we are fine :)
        return True
    
    def _setup_replicator(self):
        """Setup replicator to save RGB and semantic segmentation images"""
        import omni.replicator.core as rep
        import time
        from PIL import Image
        import os
        from omni.syntheticdata import helpers
        from omni.replicator.core import AnnotatorRegistry
        
        print("[INFO] Configurando Replicator...")

        camera_path = "/World/Go1/base/camara_robot"
        #render_product = rep.create.render_product(camera_path, resolution=(640, 480))
        render_product = rep.create.render_product(camera_path, resolution=(1280, 720))

        # Guardará en esta carpeta dentro de tu home, puedes cambiar la ruta
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(
            output_dir="/home/robcib/Desktop/Christyan/dataset_",
            rgb=True,
            semantic_segmentation=True,
            instance_segmentation=True # si hay multiples objetos, ademas del elemento, da la clase a la que pertenece.
        )
        writer.attach([render_product])

    
    
def main() -> None:
    rospy.init_node("go1_runner", anonymous=False, disable_signals=True, log_level=rospy.ERROR)
    rospy.set_param("use_sim_time", True)
    runner = Go1_runner(physics_dt=1 / 250.0, render_dt=1 / 30) # Cuanto mayor physics_dt más fuerza en las patas y cuanto más render, más despacio las mueve pero se mantiene mejor en su sitio
    simulation_app.update()
    runner.setup()
    runner.run()
    rospy.signal_shutdown("go1 complete")
    simulation_app.close()

if __name__ == "__main__":
    main()

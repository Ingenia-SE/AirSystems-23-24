import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectFollower(Node):

    def __init__(self):
        super().__init__('object_follower')

        # Configurar el suscriptor para leer las imágenes de la cámara
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  # Cambia esto si es necesario
            self.image_callback,
            10)
        # Crear un publicador para las imágenes procesadas
        self.publisher_compressed = self.create_publisher(CompressedImage, '/camera/image_processed/compressed', 10)
        # Crear un publicador para los comandos de velocidad a MAVROS
        self.publisher_cmd_vel = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)

        # Inicializar el puente entre ROS y OpenCV
        self.bridge = CvBridge()

        # Definir el rango de color rojo en HSV
        self.lower_color = (0, 100, 100)
        self.upper_color = (10, 255, 255)

    def image_callback(self, msg):
        # Convertir el mensaje de imagen de ROS a una imagen OpenCV
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convertir la imagen a HSV para la detección de colores
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Crear una máscara para el color rojo
        mask = cv2.inRange(hsv_image, self.lower_color, self.upper_color)

        # Encontrar contornos en la máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Si se encuentran contornos, dibujar un rectángulo alrededor del objeto más grande
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Dibujar el rectángulo en la imagen original
            cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Si se encuentran contornos, calcular el centro y ajustar la velocidad
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Calcular el centro del rectángulo
            center_x = x + w // 2
            center_y = y + h // 2

            # Calcular errores con respecto al centro de la imagen
            image_center_x = cv_image.shape[1] // 2
            image_center_y = cv_image.shape[0] // 2
            error_x = center_x - image_center_x
            error_y = center_y - image_center_y

            # Crear un mensaje de tipo Twist para enviar comandos de velocidad
            cmd_vel = Twist()

            # Ajustar el movimiento en el eje x (izquierda/derecha) y el eje y (adelante/atrás)
            cmd_vel.linear.x = -0.005 * error_y  # Avanzar/retroceder
            cmd_vel.linear.y = -0.005 * error_x  # Desplazar lateralmente (izquierda/derecha)

            # Publicar el comando de velocidad a MAVROS
            self.publisher_cmd_vel.publish(cmd_vel)
            
             # Convertir la imagen procesada a formato comprimido (JPEG)
        _, compressed_img = cv2.imencode('.jpg', cv_image)
        processed_msg = CompressedImage()
        processed_msg.header = msg.header
        processed_msg.format = "jpeg"
        processed_msg.data = np.array(compressed_img).tobytes()
        
         # Publicar la imagen procesada en formato comprimido
        self.publisher_compressed.publish(processed_msg)

def main(args=None):
    rclpy.init(args=args)

    # Iniciar el nodo seguidor de objetos
    object_follower = ObjectFollower()

    # Ejecutar el nodo hasta que se cierre
    rclpy.spin(object_follower)

    # Finalizar el nodo
    object_follower.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

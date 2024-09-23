import cv2
from mtcnn import MTCNN
import os
from ultralytics import YOLO



# 初始化MTCNN
detector = MTCNN()

# 使用正确的路径加载YOLOv8模型
model_path = 'E:\\College Student Innovation\\YOLOv8s\\weight\\basis\\yolov8s.pt'
try:
    model = YOLO(model_path)
except Exception as e:
    print(f"加载模型失败: {e}")
    exit(1)

def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 如果图像为空则返回
    if image is None:
        print("无法读取图像，请检查图像路径。")
        return None

    return image


def detect_faces(image):
    # 使用MTCNN检测面部
    faces = detector.detect_faces(image)
    detected_faces = []
    for face in faces:
        x, y, w, h = face['box']
        detected_faces.append((x, y, w, h))
    return detected_faces


def draw_faces(image, detected_faces):
    # 在图像上绘制检测到的面部
    for (x, y, w, h) in detected_faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)


def main():
    # 输入图像路径
    input_image_path = "E:\\College Student Innovation\\data set\\a2.PNG"

    # 输出文件夹路径
    output_folder = "E:\\College Student Innovation\\processed_images"

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 对图像进行预处理
    original_image = preprocess_image(input_image_path)

    # 如果预处理后的图像不为空，则进行面部检测
    if original_image is not None:
        # 检测图像中的面部
        detected_faces = detect_faces(original_image)

        # 打印检测到的面部数量
        print(f"检测到的面部数量: {len(detected_faces)}")

        # 在原始图像上绘制检测结果
        draw_faces(original_image, detected_faces)

        # 输出图像文件路径
        output_image_path = os.path.join(output_folder, "processed_image.jpg")

        # 保存带有面部检测结果的图像
        try:
            cv2.imwrite(output_image_path, original_image)
        except Exception as e:
            print(f"保存图像失败: {e}")
            return

        # 显示带有面部检测结果的图像
        cv2.imshow("Face Detection", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
확장 1단계: YOLOv8 자기장 원 검출 학습 (GPU 데스크탑에서 실행).

사전:
  pip install ultralytics          # (requirements-detect.txt)
  python scripts/build_detection_dataset.py   # data/detect/ 생성

실행:
  python ml/train_yolo.py

산출물: runs/detect/train*/weights/best.pt (학습된 검출 모델)
이후 색상 기반(circle_detector)과 성능 비교 → 좋으면 /api/detect에 연결.
"""
import os

from ultralytics import YOLO

BASE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(BASE, "data", "detect", "data.yaml")


def main():
    # yolov8s: 정확도/속도 균형. RTX 5060(8GB)에서 imgsz 768, batch 16 정도 무난.
    # VRAM 부족하면 batch를 8로, 더 가볍게는 yolov8n.pt 사용.
    model = YOLO("yolov8s.pt")
    model.train(
        data=DATA,
        epochs=100,
        imgsz=768,
        batch=16,
        device=0,          # GPU. CPU면 'cpu'
        patience=20,       # 조기 종료
        project=os.path.join(BASE, "runs"),
        name="zone_detect",
    )
    # 검증 지표(mAP 등) 출력
    metrics = model.val()
    print("mAP50-95:", getattr(metrics.box, "map", "n/a"))
    print("학습 완료. weights: runs/zone_detect*/weights/best.pt")


if __name__ == "__main__":
    main()

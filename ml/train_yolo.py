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
    # yolov8n(nano): 배포 메모리 제약(Railway 512MB)에서 yolov8s가 OOM → nano로 교체.
    # 파라미터 ~1/4, ONNX ~12MB, 추론 메모리 대폭 감소. 우리 합성 검출 과제는 쉬워서
    # nano로도 mAP 충분(compare_detectors로 확인). 정확도 더 필요하면 yolov8s.pt로 되돌리고
    # 대신 Railway 메모리를 올린다.
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA,
        epochs=100,
        imgsz=768,
        batch=16,
        device=0,          # GPU. CPU면 'cpu'
        patience=20,       # 조기 종료
        project=os.path.join(BASE, "runs"),
        name="zone_detect_n",
    )
    # 검증 지표(mAP 등) 출력
    metrics = model.val()
    print("mAP50-95:", getattr(metrics.box, "map", "n/a"))
    print("학습 완료. weights: runs/zone_detect*/weights/best.pt")


if __name__ == "__main__":
    main()

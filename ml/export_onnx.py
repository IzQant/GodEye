"""
학습된 YOLOv8(best.pt) → ONNX 내보내기 (데스크탑에서 실행).

서빙은 torch/ultralytics 없이 OpenCV DNN으로 ONNX를 추론하므로(배포 경량),
학습 후 이 스크립트로 ONNX를 만들어 ml/models/zone_detect.onnx로 저장한다.

사용:
  python ml/export_onnx.py runs/zone_detect/weights/best.pt
  (인자 생략 시 가장 최근 best.pt 자동 탐색)

주의: imgsz는 서빙(app/services/yolo_detector.INPUT_SIZE=768)과 반드시 일치.
"""
import glob
import os
import shutil
import sys

from ultralytics import YOLO

BASE = os.path.dirname(os.path.dirname(__file__))
OUT = os.path.join(BASE, "ml", "models", "zone_detect.onnx")
IMGSZ = 768


def find_best():
    cands = glob.glob(os.path.join(BASE, "runs", "**", "weights", "best.pt"), recursive=True)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)


def main():
    weights = sys.argv[1] if len(sys.argv) > 1 else find_best()
    if not weights or not os.path.exists(weights):
        print("best.pt를 찾을 수 없습니다. 경로를 인자로 주세요.")
        return
    print(f"내보내기: {weights} (imgsz={IMGSZ})")
    model = YOLO(weights)
    # opset 12: OpenCV DNN 호환성 안정. NMS는 서빙에서 직접 수행하므로 미포함.
    path = model.export(format="onnx", imgsz=IMGSZ, opset=12, simplify=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copy(path, OUT)
    print(f"저장: {OUT}")
    print("→ app이 자동으로 이 파일을 감지해 YOLO 검출을 사용합니다.")
    print("→ 배포하려면 git add ml/models/zone_detect.onnx 후 commit/push.")


if __name__ == "__main__":
    main()

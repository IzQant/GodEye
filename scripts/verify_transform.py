"""
Day 26 완료 기준 확인: 좌표 변환 정확도 검증.

'알려진 기준점'으로 변환을 보정한 뒤, 보정에 쓰지 않은 검증점에서
예측 맵 좌표와 실제 맵 좌표의 오차를 측정한다(residual).

1) 어파인: 축 정렬 미니맵. 일부 점으로 적합 → 나머지로 검증.
2) 호모그래피: 회전/원근 왜곡이 있는 경우. 4점으로 적합 → 나머지로 검증.

실행: python scripts/verify_transform.py
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services.coordinate_transform import fit_affine, fit_homography

rng = np.random.default_rng(0)


def test_affine():
    # 가상의 '진짜' 변환: 미니맵 800px가 맵 816000cm에 대응(스케일 1020), 오프셋 약간
    SX, OX, SY, OY = 1020.0, 500.0, 1020.0, -300.0
    pix = rng.uniform(0, 800, size=(12, 2))
    mp = np.column_stack([OX + SX * pix[:, 0], OY + SY * pix[:, 1]])

    # 앞 4점으로 보정, 나머지 8점으로 검증
    tf = fit_affine(pix[:4], mp[:4])
    errs = []
    for (px, py), (mx, my) in zip(pix[4:], mp[4:]):
        ex, ey = tf.apply(px, py)
        errs.append(np.hypot(ex - mx, ey - my))
    errs = np.array(errs)
    print(f"[어파인] 검증 8점 오차(cm): 평균 {errs.mean():.3f}, 최대 {errs.max():.3f}")
    return errs.max() < 1.0  # cm 단위, 사실상 0


def test_homography():
    # 회전+원근을 흉내낸 '진짜' 호모그래피로 대응점 생성
    true_H = np.array([
        [1000.0,   30.0,  200.0],
        [ -25.0, 1010.0, -150.0],
        [ 1e-5,   2e-5,    1.0],
    ])
    pix = rng.uniform(0, 800, size=(10, 2))
    mp = []
    for px, py in pix:
        v = true_H @ np.array([px, py, 1.0])
        mp.append([v[0] / v[2], v[1] / v[2]])
    mp = np.array(mp)

    # 4점으로 보정, 나머지 6점 검증
    tf = fit_homography(pix[:4], mp[:4])
    errs = []
    for (px, py), (mx, my) in zip(pix[4:], mp[4:]):
        ex, ey = tf.apply(px, py)
        errs.append(np.hypot(ex - mx, ey - my))
    errs = np.array(errs)
    print(f"[호모그래피] 검증 6점 오차(cm): 평균 {errs.mean():.3f}, 최대 {errs.max():.3f}")
    return errs.max() < 1.0


def main():
    ok_a = test_affine()
    ok_h = test_homography()
    if ok_a and ok_h:
        print("\n✅ Day 26 완료 기준 통과: 기준점 보정 후 변환 오차 ~0 (정확도 검증)")
    else:
        print("\n❌ 변환 오차가 큼 — 로직 점검 필요")


if __name__ == "__main__":
    main()

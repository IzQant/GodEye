"""
레이트리밋 설정 (Day 33).

slowapi로 클라이언트(IP)별 요청 빈도를 제한한다.
- PUBG API 자체 한도(분당 약 10회) 보호 겸용.
- 과다 요청 시 429 응답.
기본값: IP당 분당 30회 (필요 시 조정).
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])

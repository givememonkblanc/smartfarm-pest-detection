# 정답 프로젝트 — AI 해충 탐지 서비스 (전체 명세 구현본)

학생이 따라오다 막혔을 때 참고할 **완성 레퍼런스**입니다.
화면 명세서(M-01·M-02·W-01·W-02), 기능 정의서(F-01~F-12), ERD(4테이블)를 그대로 구현했습니다.

## 실행
```bash
pip install -r requirements.txt
# (선택) 학습한 가중치 사용: ../weights/pests.pt 자동 인식
python app.py              # http://localhost:8100   (GPU 문제 시 DEVICE=cpu)
```

## 화면
| ID | 경로 | 설명 |
|----|------|------|
| M-01 | `GET /` | 모바일 촬영/업로드 (카메라·갤러리·가이드) |
| M-02 | `GET /result/<id>` | 분석 결과 (BBox·정상/이상·해충·신뢰도·저장/재촬영) |
| W-01 | `GET /dashboard` | 대시보드 (요약위젯·추이차트·필터·리스트·엑셀) |
| W-02 | `GET /dashboard/<id>` | 상세 (원본/결과 비교·탐지좌표·메타·관리자메모) |

## API (명세 그대로)
| 메서드 | 경로 | 기능 |
|--------|------|------|
| POST | `/api/v1/analysis/upload` | F-03/04 이미지 업로드·분석 |
| GET | `/api/v1/analysis/result/{id}` | 분석 결과 조회 |
| POST | `/api/v1/analysis/save` | F-05 최종 저장 확정 |
| GET | `/api/v1/dashboard/stats` | F-06 통계 요약 |
| GET | `/api/v1/dashboard/history` | F-08/09 이력 리스트(필터) |
| GET | `/api/v1/dashboard/history/{id}` | F-10 상세 조회 |
| PUT | `/api/v1/dashboard/history/{id}/memo` | F-11 관리자 메모 |
| GET | `/api/v1/dashboard/export` | F-12 엑셀(.xlsx) 다운로드 |

## DB (ERD)
`users` · `farms` · `analysis_histories` · `detection_results` — `db.py` 참조.
처음 실행 시 `service.db` 와 데모 사용자(김농부/행복농원)가 자동 생성됩니다.

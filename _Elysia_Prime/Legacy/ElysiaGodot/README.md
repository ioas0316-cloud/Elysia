ElysiaGodot (Godot 4 Client)

Purpose
- Godot 4 프론트엔드에서 Python 셀월드(백엔드)를 WebSocket으로 구독/조작합니다.
- 렌더·UI·입력은 Godot, 세계 법칙/시뮬/학습/로그는 Python이 담당합니다.

빠른 시작
1) Python 브리지 서버 실행

   .venv\Scripts\python.exe tools\godot_bridge_server.py --host 127.0.0.1 --port 8765 --rate 4

2) Godot 4 다운로드(일반판)
   - https://godotengine.org/download 에서 Godot 4.x Standard(비-모노) zip 다운로드
   - 압축 해제 후 Godot 실행 파일을 더블클릭 (설치 필요 없음)

3) Godot에서 이 폴더(ElysiaGodot)를 새 프로젝트로 열고, 최소 씬/스크립트 생성 후 접속 코드를 붙여 넣으세요.
   (스캐폴딩은 다음 커밋에서 자동 생성 예정: Client.gd / WorldView.tscn)

프로토콜 요약
- 서버: tools/godot_bridge_server.py (WebSocket)
- 메시지 타입: init, frame, input
- 오버레이: threat/value/will → PNG base64 (uint8)

입력 매핑(예시)
- F1/F2/F3: 레이어 레벨 변경
- V/I/R/P/G/M/H: 가치장/의지장/위협장/등고선/그리드/라벨/도움말 토글
- +/-: 배속
- 마우스: 휠(줌), 중클릭 드래그(이동), 좌클릭(선택)

비고
- .NET(모노) 없이 GDScript만으로 충분합니다. 향후 C# 필요 시 Godot .NET 버전으로 전환 가능합니다.


## Auto Launch
- Windows: scripts\\run_worldviewer.bat 를 실행하면 브리지 서버와 Godot 프로젝트가 자동으로 올라갑니다.
- 로그: logs\\godot_bridge.out, logs\\godot_bridge.err 를 확인하세요.

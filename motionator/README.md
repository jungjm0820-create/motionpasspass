# Motionator (v0.1)

Blender 5.2 + Windows용 고속 FK 모션 패스 애드온. 핵심 계산은 C++ (멀티스레드)로 하고,
`frame_set()` 없이 F-Curve를 직접 평가합니다.

## 지금 들어있는 기능
1. **3D 모션 패스 오버레이** (뷰포트): 선택한 본의 헤드/테일 궤적, 속도 색상(파랑=느림, 빨강=빠름),
   프레임 점, 키프레임 점, 현재 프레임 표시, 앞/뒤 프레임 범위 조절
2. **Weight Pick**: 메시를 `Ctrl+Alt+클릭` → 그 지점에 웨이트가 가장 큰 본이 선택됨
   (같은 곳을 다시 클릭하면 다음으로 영향력이 큰 본, `Ctrl+Alt+Shift+클릭`은 선택 추가)
3. **Verify Against Blender**: Blender 자체 계산과 비교해 수학이 맞는지 검사
4. **Benchmark**: C++ vs numpy 속도 측정

## 지원 범위 (이 버전)
순수 FK 본 + 위치/회전/스케일 F-Curve. 컨스트레인트, 드라이버, NLA, F-Curve 모디파이어는 반영되지 않으며
패널에 경고로 표시됩니다.

## 설치 순서
1. 이 폴더 전체를 GitHub 저장소(Private)에 올립니다. (`.github` 폴더 포함)
2. 저장소의 **Actions** 탭 → `build-windows` 실행이 초록색 체크가 될 때까지 기다립니다 (약 3~6분)
3. 완료된 실행을 클릭 → 아래 **Artifacts**의 `motionator-blender-extension` 다운로드 → 압축 해제하면 zip 파일이 나옵니다
4. Blender: `Edit > Preferences > Get Extensions` → 오른쪽 위 `v` 메뉴 → `Install from Disk...` → 그 zip 선택
5. 3D 뷰포트에서 `N` 키 → **Motionator** 탭

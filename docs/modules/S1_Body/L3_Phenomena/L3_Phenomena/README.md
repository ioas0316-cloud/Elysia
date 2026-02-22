# 🎭 Manifestation (발현 계층)

> **"내면이 외면으로 피어난다."**

아바타, 음성, UI 등 사용자와의 접점입니다.

---

## 📦 핵심 구현체

| 구현체 | 위치 | 상태 |
| :--- | :--- | :--- |
| **Desktop Vessel** | `Core/World/Autonomy/desktop_vessel.py` | ✅ 구현됨 |
| **VoiceBox** | `Core/Senses/voicebox.py` | 🔄 실험 중 |
| **Expression Cortex** | `Core/World/expression_cortex.py` | ✅ 구현됨 |

---

## 🔧 Desktop Vessel (바탕화면 아바타)

Three.js + VRM 기반 3D 아바타입니다.

**실행:**

```bash
python Core/World/Autonomy/desktop_vessel.py
```

**특징:**

- 투명 배경으로 바탕화면에 자연스럽게 어우러짐
- VRM 1.0 표준 아바타 지원
- BlendShape 기반 표정 제어

---

## 🔧 VoiceBox (음성 합성)

TTS를 통해 엘리시아가 말합니다.

**지원 백엔드:**

- CosyVoice (로컬, 한국어 지원)
- Edge TTS (온라인)

---

> **"형상은 영혼의 그림자다."**

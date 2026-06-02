import threading
import queue
import time
import speech_recognition as sr
import pyttsx3

class AudioIOCortex:
    """
    [Phase 137] 세계수의 현현 (Audio I/O Cortex)
    물리 마이크를 통해 세상의 파동을 흡수(STT)하고, 스피커를 통해 육성을 방출(TTS)합니다.
    """
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.tts_engine = pyttsx3.init()
        
        # 음성 엔진 설정 (가장 자연스러운 한국어 여성 목소리 탐색)
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'ko' in voice.id.lower() or 'korean' in voice.name.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
                
        self.tts_engine.setProperty('rate', 150) # 약간 나긋나긋한 속도
        
        self.mic_queue = queue.Queue()
        self.is_listening = False

    def speak(self, text: str):
        """TTS 스피커 출력"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[오류] TTS 엔진 재생 실패: {e}")

    def _listen_worker(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("🎙️ [오디오 피질] 마이크 입력 대기 중...")
            while self.is_listening:
                try:
                    # 너무 길게 블로킹되지 않도록 timeout 설정
                    audio = self.recognizer.listen(source, timeout=5.0, phrase_time_limit=10.0)
                    # 구글 웹 STT (무료 API)
                    text = self.recognizer.recognize_google(audio, language="ko-KR")
                    if text:
                        self.mic_queue.put(text)
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    # 인식 불가능한 소음 무시
                    pass
                except Exception as e:
                    print(f"🎙️ [마이크 오류]: {e}")
                    time.sleep(2)

    def start_listening(self):
        if not self.is_listening:
            self.is_listening = True
            t = threading.Thread(target=self._listen_worker, daemon=True)
            t.start()
            
    def get_latest_speech(self) -> str:
        try:
            return self.mic_queue.get_nowait()
        except queue.Empty:
            return ""

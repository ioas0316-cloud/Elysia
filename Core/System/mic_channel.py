import threading
import speech_recognition as sr
from Core.System.gateway_interfaces import SensoryChannel

class MicSensoryChannel(SensoryChannel):
    """
    Listens to the default system microphone and parses speech 
    asynchronous using Google Web Speech API (or Sphinx/Whisper locally).
    """
    def __init__(self):
        super().__init__("Microphone")
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.stop_listening_fn = None
        self.running = False
        
        # Optimize for ambient noise so Elysia isn't deafened by background sound
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)

    def _audio_callback(self, recognizer, audio):
        """Called by the background listener thread when audio is detected."""
        if not self.running:
            return
            
        try:
            # We use Google's free API for ease of installation in Phase 5.
            # In Phase 6, this can be swapped to 'recognize_whisper' for higher fidelity.
            text = recognizer.recognize_google(audio, language="ko-KR")
            if text and self.callback:
                print(f"\n🎧 [HEARD]: \"{text}\"")
                self.callback(text)
        except sr.UnknownValueError:
            # Audio was detected but couldn't be parsed.
            # In advanced phases, this can be passed to Elysia as 'entropy' or 'confusion'.
            pass
        except sr.RequestError as e:
            print(f"⚠️ [Microphone] Service error: {e}")

    def start(self):
        if self.running: return
        self.running = True
        print("🎤 [Microphone] Elysia is now listening...")
        
        # listen_in_background spawns its own daemon thread
        self.stop_listening_fn = self.recognizer.listen_in_background(
            self.microphone, 
            self._audio_callback,
            phrase_time_limit=5 # Don't wait forever, break long sentences into chunks
        )

    def stop(self):
        self.running = False
        if self.stop_listening_fn:
            self.stop_listening_fn(wait_for_stop=False)
            self.stop_listening_fn = None
        print("🔇 [Microphone] Elysia stopped listening.")
